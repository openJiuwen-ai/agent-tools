"""
OpenJiuwen Code Generator

Generates openJiuwen code from the intermediate representation (IR).
"""

import ast
import re
from typing import List, Dict, Optional
from .ir_models import (
    AgentIR,
    WorkflowIR,
    WorkflowNodeIR,
    WorkflowNodeType,
    WorkflowEdgeIR,
    ParseResult,
    NodeFunctionInfo,
    ConditionalEdgeInfo,
    StateFieldInfo,
    ToolIR,
    ParamInfo,
    LLMConfigIR,
)


class RouterTransformer(ast.NodeTransformer):
    """AST transformer to convert LangGraph conditional function to openJiuwen router.

    Transforms:
    - state['field'] -> runtime.get_global_state("node_id.field")
    - state.get("field", default) -> runtime.get_global_state("node_id.field") or default
    - state['field'] = value -> runtime.update_global_state({"node_id.field": value})
    - state['field'] -= 1 -> (handled specially)
    - return 'value' -> return "target_node" (according to mapping)
    - return END -> return "end"

    Note: Uses node_id.field format to access global state, enabling proper
    component-to-component data flow in openJiuwen.
    """

    def __init__(self, state_param: str, edge_mapping: Dict[str, str], node_id: str = ""):
        """
        Args:
            state_param: The state parameter name in original function (e.g., "state")
            edge_mapping: Mapping from original return values to target node IDs
            node_id: The node ID to use as prefix for global state keys (e.g., "validate_city")
        """
        self.state_param = state_param
        self.edge_mapping = edge_mapping
        self.node_id = node_id
        self.mutations = []  # Track state mutations for special handling

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Transform state['field'] access"""
        # Check if it's state[key] pattern
        if (isinstance(node.value, ast.Name) and
            node.value.id == self.state_param and
            isinstance(node.slice, ast.Constant)):

            field_name = node.slice.value
            # Build global state key with node_id prefix: "node_id.field"
            global_state_key = f"{self.node_id}.{field_name}" if self.node_id else field_name
            # Create: runtime.get_global_state("node_id.field") or default
            get_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='runtime', ctx=ast.Load()),
                    attr='get_global_state',
                    ctx=ast.Load()
                ),
                args=[ast.Constant(value=global_state_key)],
                keywords=[]
            )

            # Add default value based on field name heuristics
            # For common string fields, use "", for numbers use 0
            default_val = self._get_default_for_field(field_name)
            if default_val is not None:
                return ast.BoolOp(
                    op=ast.Or(),
                    values=[get_call, ast.Constant(value=default_val)]
                )

            return get_call

        return self.generic_visit(node)

    def _get_default_for_field(self, field_name: str):
        """Infer default value based on field name"""
        # Fields that are likely strings
        string_fields = {'answer', 'response', 'content', 'text', 'message', 'question', 'query', 'input', 'output', 'result', 'name', 'title', 'description'}
        # Fields that are likely integers
        int_fields = {'count', 'retry', 'retry_left', 'retries', 'attempts', 'step', 'index', 'num', 'number', 'iteration', 'limit', 'max', 'min'}

        field_lower = field_name.lower()

        if field_lower in string_fields or any(s in field_lower for s in ['_name', '_text', '_content', '_message']):
            return ""
        if field_lower in int_fields or any(s in field_lower for s in ['_count', '_num', '_retry', '_left', '_index']):
            return 0

        return None

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Transform state.get("field", default) calls"""
        # Check for state.get() pattern
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == self.state_param and
            node.func.attr == 'get'):

            if node.args and isinstance(node.args[0], ast.Constant):
                field_name = node.args[0].value
                # Build global state key with node_id prefix: "node_id.field"
                global_state_key = f"{self.node_id}.{field_name}" if self.node_id else field_name

                # Create runtime.get_global_state call
                get_call = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='runtime', ctx=ast.Load()),
                        attr='get_global_state',
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=global_state_key)],
                    keywords=[]
                )

                # If there's a default value, add "or default"
                if len(node.args) > 1:
                    default_val = node.args[1]
                    return ast.BoolOp(
                        op=ast.Or(),
                        values=[get_call, default_val]
                    )

                return get_call

        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Transform state['field'] = value assignments

        Translates to: runtime.update_global_state({"node_id.field": value})
        """
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and
                isinstance(target.value, ast.Name) and
                target.value.id == self.state_param and
                isinstance(target.slice, ast.Constant)):
                field_name = target.slice.value
                # Build global state key with node_id prefix: "node_id.field"
                global_state_key = f"{self.node_id}.{field_name}" if self.node_id else field_name
                # Transform to: runtime.update_global_state({"node_id.field": value})
                return ast.Expr(value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='runtime', ctx=ast.Load()),
                        attr='update_global_state',
                        ctx=ast.Load()
                    ),
                    args=[ast.Dict(
                        keys=[ast.Constant(value=global_state_key)],
                        values=[self.generic_visit(node.value)]
                    )],
                    keywords=[]
                ))

        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        """Transform state['field'] -= 1 (augmented assignments)

        Translates to: runtime.update_global_state({"node_id.field": runtime.get_global_state("node_id.field") - 1})
        """
        if (isinstance(node.target, ast.Subscript) and
            isinstance(node.target.value, ast.Name) and
            node.target.value.id == self.state_param and
            isinstance(node.target.slice, ast.Constant)):
            field_name = node.target.slice.value
            # Build global state key with node_id prefix: "node_id.field"
            global_state_key = f"{self.node_id}.{field_name}" if self.node_id else field_name

            # Build: runtime.get_global_state("node_id.field") op value
            get_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='runtime', ctx=ast.Load()),
                    attr='get_global_state',
                    ctx=ast.Load()
                ),
                args=[ast.Constant(value=global_state_key)],
                keywords=[]
            )

            # Add default value for safety
            get_with_default = ast.BoolOp(
                op=ast.Or(),
                values=[get_call, ast.Constant(value=0)]
            )

            # Build: get_with_default op value (e.g., get_with_default - 1)
            new_value = ast.BinOp(
                left=get_with_default,
                op=node.op,
                right=self.generic_visit(node.value)
            )

            # Transform to: runtime.update_global_state({"node_id.field": new_value})
            return ast.Expr(value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='runtime', ctx=ast.Load()),
                    attr='update_global_state',
                    ctx=ast.Load()
                ),
                args=[ast.Dict(
                    keys=[ast.Constant(value=global_state_key)],
                    values=[new_value]
                )],
                keywords=[]
            ))

        return self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        """Transform ternary expressions (a if cond else b) to handle END constant

        Example: END if state.get("error") else "call_weather"
        -> "end" if runtime.get_global_state("node.error") else "call_weather"
        """
        # Transform the condition
        new_test = self.visit(node.test)

        # Transform body (true branch) - handle END constant
        new_body = node.body
        if isinstance(node.body, ast.Name) and node.body.id == "END":
            new_body = ast.Constant(value="end")
        elif isinstance(node.body, ast.Constant) and isinstance(node.body.value, str):
            target = self.edge_mapping.get(node.body.value, node.body.value)
            if target == "END":
                target = "end"
            new_body = ast.Constant(value=target)
        else:
            new_body = self.visit(node.body)

        # Transform orelse (false branch) - handle END constant
        new_orelse = node.orelse
        if isinstance(node.orelse, ast.Name) and node.orelse.id == "END":
            new_orelse = ast.Constant(value="end")
        elif isinstance(node.orelse, ast.Constant) and isinstance(node.orelse.value, str):
            target = self.edge_mapping.get(node.orelse.value, node.orelse.value)
            if target == "END":
                target = "end"
            new_orelse = ast.Constant(value=target)
        else:
            new_orelse = self.visit(node.orelse)

        return ast.IfExp(test=new_test, body=new_body, orelse=new_orelse)

    def visit_Return(self, node: ast.Return) -> ast.AST:
        """Transform return values according to edge mapping"""
        if node.value:
            # Handle string constant returns
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                original_val = node.value.value
                target = self.edge_mapping.get(original_val, original_val)
                if target == "END":
                    target = "end"
                return ast.Return(value=ast.Constant(value=target))

            # Handle END constant
            if isinstance(node.value, ast.Name) and node.value.id == "END":
                return ast.Return(value=ast.Constant(value="end"))

        return self.generic_visit(node)


def transform_conditional_function(func_body: str, edge_mapping: Dict[str, str], node_id: str = "") -> str:
    """Transform a LangGraph conditional function to openJiuwen router code.

    Args:
        func_body: The original function source code
        edge_mapping: Mapping from return values to target nodes
        node_id: The node ID to use as prefix for global state keys (e.g., "validate_city")

    Returns:
        Transformed function body as string (list of statements)
    """
    try:
        # Parse the function
        tree = ast.parse(func_body)

        if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None

        func_node = tree.body[0]

        # Get state parameter name
        state_param = "state"
        if func_node.args.args:
            state_param = func_node.args.args[0].arg

        # Create transformer with node_id for proper global state key format (node_id.field)
        transformer = RouterTransformer(state_param, edge_mapping, node_id)

        # Transform the function body (not the function itself)
        new_body = []
        for stmt in func_node.body:
            # Skip docstring
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    continue

            transformed = transformer.visit(stmt)
            new_body.append(transformed)

        # Fix missing line info
        ast.fix_missing_locations(ast.Module(body=new_body, type_ignores=[]))

        # Generate code from transformed AST
        lines = []
        for stmt in new_body:
            lines.append(ast.unparse(stmt))

        return lines

    except SyntaxError as e:
        return None
    except Exception as e:
        return None


class OpenJiuwenGenerator:
    """Generates openJiuwen Python code from IR"""

    def __init__(self):
        self.warnings: List[str] = []
        self.manual_tasks: List[str] = []
        self._indent = "    "
        self._end_source_nodes: List[str] = []  # Nodes that connect to End
        self._end_common_output: str = "response"  # Common output field for End

    def generate(self, agent_ir: AgentIR, output_name: str = None) -> str:
        """Generate complete openJiuwen code"""
        self.warnings = []
        self.manual_tasks = []
        self._agent_ir = agent_ir  # Store for use in other methods
        self._end_source_nodes = []
        self._end_common_output = "response"

        name = output_name or agent_ir.name

        # Pre-calculate which nodes connect to End (needed for component generation)
        self._calculate_end_sources(agent_ir)

        sections = []

        # 1. Generate file header
        sections.append(self._generate_header(agent_ir))

        # 2. Generate imports
        sections.append(self._generate_imports(agent_ir))

        # 3. Generate global variables (if any)
        global_vars = self._generate_global_variables(agent_ir)
        if global_vars:
            sections.append(global_vars)

        # 4. Generate tool definitions (if any)
        if agent_ir.tools:
            sections.append(self._generate_tools(agent_ir.tools))

        # 5. Generate component classes
        sections.append(self._generate_components(agent_ir))

        # 6. Generate router functions (v2: for conditional connections)
        routers = self._generate_routers(agent_ir)
        if routers:
            sections.append(routers)

        # 7. Generate workflow builder
        sections.append(self._generate_workflow_builder(agent_ir))

        # 8. Generate main function
        sections.append(self._generate_main(agent_ir))

        return "\n\n".join(filter(None, sections))

    def _calculate_end_sources(self, agent_ir: AgentIR) -> None:
        """Pre-calculate which nodes connect to End and their common output field"""
        if not agent_ir.workflow:
            return

        end_edges = [e for e in agent_ir.workflow.edges
                     if e.target_node == "END" or e.target_node == "end"]

        if not end_edges:
            return

        source_nodes = list(set(e.source_node for e in end_edges))

        if len(source_nodes) > 1:
            # Multiple sources - need to track them
            self._end_source_nodes = source_nodes

            # Find common output field
            output_map = self._build_output_map(agent_ir.workflow, agent_ir)
            common_output = "response"

            for src in source_nodes:
                if common_output not in output_map.get(src, []):
                    for field in output_map.get(src, []):
                        if field in ["response", "result", "output"]:
                            common_output = field
                            break

            self._end_common_output = common_output

    def _generate_header(self, agent_ir: AgentIR) -> str:
        """Generate file header with docstring"""
        return f'''"""
{agent_ir.name} - Migrated from LangGraph to openJiuwen

Auto-generated by lg2jiuwentool
Source framework: {agent_ir.source_framework}
"""
'''

    def _has_llm_nodes(self, agent_ir: AgentIR) -> bool:
        """Check if the agent has any LLM nodes"""
        if not agent_ir.workflow:
            return False
        return any(node.node_type == WorkflowNodeType.LLM for node in agent_ir.workflow.nodes)

    def _generate_imports(self, agent_ir: AgentIR) -> str:
        """Generate import statements (v2: removed BranchRouter, using router functions)"""
        imports = [
            "import os",
            "from typing import Dict, Any, Optional, List",
        ]

        # Add SSL bypass for LLM calls if needed
        if self._has_llm_nodes(agent_ir) or agent_ir.llm_config:
            imports.append("os.environ['LLM_SSL_VERIFY'] = 'false'")

        imports.extend([
            "# openJiuwen imports",
            "from openjiuwen.core.workflow.base import Workflow",
            "from openjiuwen.core.component.start_comp import Start",
            "from openjiuwen.core.component.end_comp import End",
            "from openjiuwen.core.component.base import WorkflowComponent",
            "from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output",
            "from openjiuwen.core.runtime.runtime import Runtime",
            "from openjiuwen.core.runtime.workflow import WorkflowRuntime",
            "from openjiuwen.core.context_engine.base import Context",
        ])

        # Add LLM imports if there are LLM nodes
        if self._has_llm_nodes(agent_ir) or agent_ir.llm_config:
            imports.extend([
                "",
                "# LLM imports",
                "from openjiuwen.core.utils.llm.model_library.openai import OpenAIChatModel",
            ])

        if agent_ir.tools:
            imports.extend([
                "",
                "# Tool imports",
                "from openjiuwen.core.utils.tool.param import Param",
                "from openjiuwen.core.utils.tool.tool import tool",
            ])

        # Add any additional imports from original code that might be needed
        extra_imports = self._extract_needed_imports(agent_ir)
        if extra_imports:
            imports.extend(["", "# Additional imports"])
            imports.extend(extra_imports)

        return "\n".join(imports)

    def _extract_needed_imports(self, agent_ir: AgentIR) -> List[str]:
        """Extract additional imports that might be needed from original code"""
        imports = []

        # Get import statements from metadata (extracted from source)
        source_imports = agent_ir.metadata.get("import_statements", [])

        # Filter out LangGraph-specific imports and keep only needed ones
        langgraph_prefixes = [
            "langgraph",
            "langchain",
            "langchain_core",
            "langchain_openai",
            "langchain_anthropic",
            "langchain_google",
        ]

        # Keep track of what we import from openJiuwen to avoid duplicates
        openjiuwen_imports = {
            "os", "typing", "Dict", "Any", "Optional", "List",
            "Workflow", "Start", "End", "WorkflowComponent",
            "ComponentExecutable", "Input", "Output", "Runtime",
            "WorkflowRuntime", "Context", "OpenAIChatModel",
            "tool", "Param",
        }

        for stmt in source_imports:
            # Skip LangGraph/LangChain imports
            skip = False
            for prefix in langgraph_prefixes:
                if prefix in stmt:
                    skip = True
                    break

            if skip:
                continue

            # Skip imports we already have
            if any(imp in stmt for imp in openjiuwen_imports):
                continue

            # Skip typing imports (we already have our own)
            if stmt.startswith("from typing import") or stmt == "import typing":
                continue

            # Keep useful imports like httpx, requests, json, etc.
            imports.append(stmt)

        # Also check function bodies for common imports
        if agent_ir.workflow:
            for node in agent_ir.workflow.nodes:
                body = node.config.get("function_body", "")
                if "random." in body and "import random" not in imports:
                    imports.append("import random")
                if "json." in body and "import json" not in imports:
                    imports.append("import json")
                if "re." in body and "import re" not in imports:
                    imports.append("import re")

        # Check tool function bodies
        for tool in agent_ir.tools:
            body = tool.function_body or ""
            if "httpx" in body and "import httpx" not in imports:
                imports.append("import httpx")
            if "requests" in body and "import requests" not in imports:
                imports.append("import requests")
            if "json." in body and "import json" not in imports:
                imports.append("import json")
            if "datetime" in body and "import datetime" not in imports:
                imports.append("import datetime")
            if "re." in body and "import re" not in imports:
                imports.append("import re")

        return list(set(imports))

    def _generate_global_variables(self, agent_ir: AgentIR) -> str:
        """Generate global variable definitions from metadata"""
        global_vars = agent_ir.metadata.get("global_variables", [])

        if not global_vars:
            return ""

        lines = ["# ============ Global Variables ============", ""]

        for var in global_vars:
            name = var.get("name", "")
            value = var.get("value", "")
            type_hint = var.get("type_hint")

            if name and value:
                if type_hint:
                    lines.append(f"{name}: {type_hint} = {value}")
                else:
                    lines.append(f"{name} = {value}")

        return "\n".join(lines)

    def _generate_tools(self, tools: List[ToolIR]) -> str:
        """Generate tool definitions"""
        tool_defs = []

        for tool in tools:
            params_code = self._generate_tool_params(tool.params)

            # Use original function body if available, otherwise placeholder
            if tool.function_body and tool.function_body.strip():
                # Indent function body properly
                body_lines = tool.function_body.strip().split('\n')
                indented_body = '\n'.join('    ' + line if line.strip() else '' for line in body_lines)
                function_body = indented_body
            else:
                function_body = '    # TODO: Implement tool logic\n    pass'

            tool_def = f'''@tool(
    name="{tool.name}",
    description="{tool.description}",
    params=[
{params_code}
    ]
)
def {tool.name}({self._generate_function_params(tool.params)}) -> {tool.return_type or "str"}:
    """
    {tool.description}
    """
{function_body}
'''
            tool_defs.append(tool_def)

        return "# ============ Tool Definitions ============\n\n" + "\n\n".join(tool_defs)

    def _generate_tool_params(self, params: List[ParamInfo]) -> str:
        """Generate Param objects for tool decorator"""
        param_lines = []
        for p in params:
            type_map = {
                "str": "string",
                "int": "int",
                "float": "float",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
            }
            param_type = type_map.get(p.type_hint, "string") if p.type_hint else "string"

            parts = [
                f'name="{p.name}"',
                f'description="{p.description or p.name}"',
                f'type="{param_type}"',
                f'required={p.required}',
            ]
            if p.default_value:
                parts.append(f'default={p.default_value}')

            param_lines.append(f'{self._indent * 2}Param({", ".join(parts)})')

        return ",\n".join(param_lines)

    def _generate_function_params(self, params: List[ParamInfo]) -> str:
        """Generate function parameter string"""
        parts = []
        for p in params:
            type_hint = p.type_hint or "str"
            if p.default_value:
                parts.append(f'{p.name}: {type_hint} = {p.default_value}')
            else:
                parts.append(f'{p.name}: {type_hint}')
        return ", ".join(parts)

    def _generate_components(self, agent_ir: AgentIR) -> str:
        """Generate all component classes"""
        components = []

        if not agent_ir.workflow:
            return ""

        # First, generate LLM configuration if there are LLM nodes
        llm_config_code = self._generate_llm_config(agent_ir)
        if llm_config_code:
            components.append(llm_config_code)

        for node in agent_ir.workflow.nodes:
            if node.node_type in (WorkflowNodeType.START, WorkflowNodeType.END):
                continue

            # LLM nodes also use regular component generation to preserve original logic
            # The _generate_component method handles state->inputs conversion
            component = self._generate_component(node, agent_ir)
            components.append(component)

        return "# ============ Component Definitions ============\n\n" + "\n\n".join(components)

    def _generate_llm_config(self, agent_ir: AgentIR) -> str:
        """Generate LLM configuration code based on parsed LLM config"""
        if not self._has_llm_nodes(agent_ir) and not agent_ir.llm_config:
            return ""

        llm_config = agent_ir.llm_config
        if not llm_config:
            # Use default config
            llm_config = LLMConfigIR(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.7,
            )

        provider = llm_config.provider or "openai"
        model = llm_config.model or "gpt-3.5-turbo"
        api_base = llm_config.api_base or "https://api.openai.com/v1"
        api_key = llm_config.api_key or ""

        return f'''# LLM Configuration
# TODO: Configure your LLM provider settings
MODEL_PROVIDER = "{provider}"
MODEL_NAME = "{model}"
API_BASE = os.getenv("OPENAI_API_BASE", "{api_base}")
API_KEY = os.getenv("OPENAI_API_KEY", "{api_key}")'''

    def _generate_llm_component(self, node: WorkflowNodeIR, agent_ir: AgentIR) -> str:
        """Generate LLM component code for nodes that use LLM

        Uses WorkflowComponent + ComponentExecutable with OpenAIChatModel directly.
        Reads from and writes to global state (AgentState mapping).
        """
        class_name = self._to_class_name(node.id)
        docstring = node.config.get("docstring", f"LLM Component: {node.name}")
        is_conditional = node.config.get("is_conditional", False)

        # Extract input/output fields from state accesses
        state_accesses = node.config.get("state_accesses", [])
        read_keys = [access.get("key", "") for access in state_accesses
                     if access.get("key") and access.get("access_type") == "read"]
        write_keys = [access.get("key", "") for access in state_accesses
                      if access.get("key") and access.get("access_type") == "write"]

        # Extract output field from return keys or write keys
        return_keys = node.config.get("return_keys", [])
        output_key = return_keys[0] if return_keys else (write_keys[0] if write_keys else "answer")

        # Store output key for workflow builder reference
        node.config["llm_output_key"] = output_key

        # Determine the primary input key for LLM message
        primary_input = read_keys[0] if read_keys else "question"

        # Infer default value for primary input
        default_val = '""'
        if primary_input in ['question', 'query', 'input', 'message', 'content', 'text']:
            default_val = '""'

        # Generate routing code for conditional LLM components
        routing_code = ""
        if is_conditional:
            routing_code = self._generate_llm_routing_code(node, agent_ir, output_key)

        return f'''class {class_name}(WorkflowComponent, ComponentExecutable):
    """
    {docstring}

    Input fields: {read_keys}
    Output fields: {output_key}
    """

    def __init__(self):
        super().__init__()
        self._llm = OpenAIChatModel(
            api_key=API_KEY,
            api_base=API_BASE,
            timeout=30,
        )

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        # Read from inputs (provided by inputs_schema)
        {primary_input} = inputs.get("{primary_input}") or {default_val}

        messages = [{{"role": "user", "content": {primary_input}}}]
        llm_response = await self._llm.ainvoke(
            model_name=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            top_p=0.9,
        )

        {output_key} = llm_response.content or ""

        return {{"{output_key}": {output_key}}}
'''

    def _generate_llm_routing_code(self, node: WorkflowNodeIR, agent_ir: AgentIR, output_key: str) -> str:
        """Generate routing logic code for conditional LLM components.

        Embeds the conditional function's logic directly in the component,
        using local variables where possible (e.g., the just-computed answer).
        """
        node_id = node.id

        # Find conditional edge info
        edge_mapping = {}
        condition_func_name = None
        for edge in agent_ir.metadata.get("conditional_edges", []):
            if edge.get("source_node") == node_id:
                edge_mapping = edge.get("mapping", {})
                condition_func_name = edge.get("condition_function")
                break

        if not condition_func_name:
            return ""

        # Get the conditional function info
        cond_func_info = agent_ir.metadata.get("conditional_functions", {}).get(condition_func_name)
        if not cond_func_info:
            return ""

        original_body = cond_func_info.get("body", "")
        if not original_body:
            return ""

        # Transform the conditional function using AST, pass node_id for proper global state keys
        transformed_lines = transform_conditional_function(original_body, edge_mapping, node_id)
        if not transformed_lines:
            return ""

        # Build the routing code block
        lines = [
            "",
            "        # Compute route decision (from conditional function)",
        ]

        for stmt in transformed_lines:
            # Replace global state read for output_key with local variable
            # e.g., runtime.get_global_state('node_id.answer') -> answer
            stmt = stmt.replace(f"runtime.get_global_state('{node_id}.{output_key}')", output_key)
            stmt = stmt.replace(f'runtime.get_global_state("{node_id}.{output_key}")', output_key)

            # Handle multi-line statements
            for line in stmt.split('\n'):
                if line.strip().startswith('return '):
                    # Convert return to route assignment and save to global state
                    route_value = line.strip()[7:].strip()  # Remove 'return '
                    lines.append(f"        _route = {route_value}")
                else:
                    lines.append(f"        {line}")

        # Use dot format for global state key: node_id.route
        lines.append(f'        runtime.update_global_state({{"{node_id}.route": _route}})')

        return "\n".join(lines)

    def _extract_template_content(self, node: WorkflowNodeIR, input_keys: List[str]) -> str:
        """Extract and convert template content from function body

        openJiuwen uses double curly braces for template variables: {{variable}}
        """
        body_statements = node.config.get("body_statements", [])

        # Default template - use the first input key as the user message
        if input_keys:
            primary_input = input_keys[0]
            # Note: In Python string, we use {{{{ to produce {{ in output
            default_template = f'''[
            {{"role": "user", "content": "{{{{{primary_input}}}}}"}}
        ]'''
        else:
            default_template = '''[
            {"role": "user", "content": "{{query}}"}
        ]'''

        # Try to extract template from body if it contains message construction
        for stmt in body_statements:
            # Look for patterns like: messages = [{"role": "user", "content": ...}]
            if "messages" in stmt and "role" in stmt and "user" in stmt:
                # Try to extract and convert the message content
                # Match content field value
                match = re.search(r'"content":\s*([^}]+)', stmt)
                if match:
                    content = match.group(1).strip().rstrip(',').strip()
                    # Convert state["key"] or inputs.get("key") to {{key}}
                    for key in input_keys:
                        content = re.sub(rf'inputs\.get\("{key}"\)', f'{{{{{key}}}}}', content)
                        content = re.sub(rf'state\["{key}"\]', f'{{{{{key}}}}}', content)

                    # If the content is still a variable reference, use template syntax
                    if "inputs.get" in content or "state[" in content:
                        continue

                    return f'''[
            {{"role": "user", "content": {content}}}
        ]'''

        return default_template

    def _generate_routers(self, agent_ir: AgentIR) -> str:
        """Generate router functions for conditional connections (v2 spec)

        In v2, conditional edges use add_conditional_connection with a router function
        that takes Runtime and returns the target node ID.
        """
        if not agent_ir.workflow:
            return ""

        routers = []
        for node in agent_ir.workflow.nodes:
            if node.config.get("is_conditional", False):
                router_code = self._generate_router_function(node, agent_ir)
                routers.append(router_code)

        if not routers:
            return ""

        return "# ============ Router Functions ============\n\n" + "\n\n".join(routers)

    def _generate_router_function(self, node: WorkflowNodeIR, agent_ir: AgentIR) -> str:
        """Generate a router function for conditional connection.

        Router function signature: def router(runtime: Runtime) -> str
        Returns target node ID based on component outputs.

        The router reads component outputs from global state (node_id.field format)
        and applies the original LangGraph conditional logic to determine the route.
        """
        node_id = node.id

        # Find conditional edge info
        condition_func_name = None
        edge_mapping = {}
        for edge in agent_ir.metadata.get("conditional_edges", []):
            if edge.get("source_node") == node_id:
                condition_func_name = edge.get("condition_function")
                edge_mapping = edge.get("mapping", {})
                break

        # Get the conditional function info
        cond_func_info = agent_ir.metadata.get("conditional_functions", {}).get(condition_func_name)

        # Build router function header
        lines = [
            f"def {node_id}_router(runtime: Runtime) -> str:",
            f'    """',
            f'    Router function for {node_id} conditional connection.',
        ]

        if condition_func_name:
            lines.append(f'    ')
            lines.append(f'    Original LangGraph function: {condition_func_name}')

        lines.append(f'    """')

        # Generate routing logic from conditional function
        if cond_func_info and cond_func_info.get("body"):
            original_body = cond_func_info.get("body", "")
            # Transform the conditional function using AST
            transformed_lines = transform_conditional_function(original_body, edge_mapping, node_id)

            if transformed_lines:
                for stmt in transformed_lines:
                    # Handle multi-line statements
                    for line in stmt.split('\n'):
                        lines.append(f'    {line}')
            else:
                # Fallback: simple routing based on first return value
                default_target = list(edge_mapping.values())[0] if edge_mapping else "end"
                lines.append(f'    # TODO: Implement routing logic from {condition_func_name}')
                lines.append(f'    return "{default_target}"')
        else:
            # No conditional function found, generate fallback
            default_target = list(edge_mapping.values())[0] if edge_mapping else "end"
            lines.append(f'    # Fallback: no conditional function found')
            lines.append(f'    return "{default_target}"')

        return "\n".join(lines)

    def _generate_component(self, node: WorkflowNodeIR, agent_ir: AgentIR) -> str:
        """Generate a single component class"""
        class_name = self._to_class_name(node.id)
        is_conditional = node.config.get("is_conditional", False)
        docstring = node.config.get("docstring", f"Component: {node.name}")

        if is_conditional:
            return self._generate_conditional_component(node, class_name, docstring, agent_ir)
        else:
            return self._generate_simple_component(node, class_name, docstring, agent_ir)

    def _generate_simple_component(self, node: WorkflowNodeIR, class_name: str,
                                     docstring: str, agent_ir: AgentIR = None) -> str:
        """Generate a simple (non-conditional) component class"""
        invoke_body = self._convert_function_body(node, is_conditional=False, agent_ir=agent_ir)

        # Check if we need aggregation support
        aggregation_code = self._generate_aggregation_code(node, agent_ir)

        # Check if this node uses LLM
        uses_llm = node.config.get("uses_llm", False)

        if uses_llm:
            # LLM component with self._llm initialization
            return f'''class {class_name}(WorkflowComponent, ComponentExecutable):
    """{docstring}"""

    def __init__(self):
        super().__init__()
        self._llm = OpenAIChatModel(
            api_key=API_KEY,
            api_base=API_BASE,
            timeout=30,
        )

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
{aggregation_code}{invoke_body}
'''
        else:
            return f'''class {class_name}(WorkflowComponent, ComponentExecutable):
    """{docstring}"""

    def __init__(self):
        super().__init__()

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
{aggregation_code}{invoke_body}
'''

    def _generate_aggregation_code(self, node: WorkflowNodeIR, agent_ir: AgentIR = None) -> str:
        """Generate code for handling aggregated fields like messages"""
        if not agent_ir:
            return ""

        # Find fields with aggregators
        aggregated_fields = []
        for field in agent_ir.state_fields:
            if field.has_aggregator and field.aggregator:
                aggregated_fields.append(field)

        if not aggregated_fields:
            return ""

        return_keys = node.config.get("return_keys", [])
        lines = []

        for field in aggregated_fields:
            if field.name in return_keys:
                lines.append(f'{self._indent * 2}# Aggregate {field.name} (original aggregator: {field.aggregator})')
                # Note: Aggregated fields use shared global state key for accumulation
                lines.append(f'{self._indent * 2}_{field.name}_history = runtime.get_global_state("{node.id}.{field.name}") or []')
                lines.append("")

        if lines:
            return "\n".join(lines) + "\n"
        return ""

    def _generate_aggregation_logic(self, node: WorkflowNodeIR, agent_ir: AgentIR = None) -> str:
        """Generate aggregation logic for fields with aggregators (like messages).

        This combines reading history and preparing aggregated values.
        No update_global_state call needed - framework handles it via return value.
        """
        if not agent_ir:
            return ""

        # Find fields with aggregators that are in return keys
        aggregated_fields = []
        return_keys = node.config.get("return_keys", [])
        return_statement = node.config.get("return_statement", "")

        for field in agent_ir.state_fields:
            if field.has_aggregator and field.aggregator and field.name in return_keys:
                aggregated_fields.append(field)

        if not aggregated_fields:
            return ""

        lines = []
        for field in aggregated_fields:
            # Extract the value for this field from return statement
            field_value = self._extract_field_value_from_return(return_statement, field.name)

            if field_value:
                # Fix any nested quote issues in the field value
                fixed_field_value = self._fix_fstring_quotes(field_value)
                lines.append(f'{self._indent * 2}# Aggregate {field.name} (original aggregator: {field.aggregator})')
                lines.append(f'{self._indent * 2}# Read history from inputs (upstream component output)')
                lines.append(f'{self._indent * 2}_{field.name}_history = inputs.get("{field.name}") or []')
                lines.append(f'{self._indent * 2}_{field.name}_current = {fixed_field_value}')
                lines.append(f'{self._indent * 2}_{field.name}_aggregated = _{field.name}_history + _{field.name}_current')
                lines.append("")

        if lines:
            return "\n".join(lines)
        return ""

    def _extract_field_value_from_return(self, return_statement: str, field_name: str) -> Optional[str]:
        """Extract the value for a field from a return statement dict"""
        if not return_statement:
            return None

        # Find the starting position of the field
        import re
        pattern = rf'"{field_name}":\s*'
        match = re.search(pattern, return_statement)
        if not match:
            return None

        start_pos = match.end()

        # Now extract the value, handling nested brackets
        value = ""
        depth = 0
        in_string = False
        string_char = None
        i = start_pos

        while i < len(return_statement):
            char = return_statement[i]

            # Handle string boundaries
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char and return_statement[i-1] != '\\':
                    in_string = False

            if not in_string:
                if char in '[{(':
                    depth += 1
                elif char in ']})':
                    if depth == 0:
                        break
                    depth -= 1
                elif char == ',' and depth == 0:
                    break

            value += char
            i += 1

        return value.strip() if value else None

    def _generate_conditional_component(self, node: WorkflowNodeIR, class_name: str,
                                        docstring: str, agent_ir: AgentIR) -> str:
        """Generate a conditional component class (v2: simplified, no BranchRouter integration)

        In v2, conditional logic is handled by external router functions,
        not embedded BranchRouter. The component just returns a 'route' value.
        """
        invoke_body = self._convert_function_body(node, is_conditional=True, agent_ir=agent_ir)
        aggregation_code = self._generate_aggregation_code(node, agent_ir)

        # Check if this node uses LLM
        uses_llm = node.config.get("uses_llm", False)

        if uses_llm:
            # LLM component with self._llm initialization
            return f'''class {class_name}(WorkflowComponent, ComponentExecutable):
    """{docstring}"""

    def __init__(self):
        super().__init__()
        self._llm = OpenAIChatModel(
            api_key=API_KEY,
            api_base=API_BASE,
            timeout=30,
        )

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
{aggregation_code}{invoke_body}
'''
        else:
            # v2: Simple component class without BranchRouter
            return f'''class {class_name}(WorkflowComponent, ComponentExecutable):
    """{docstring}"""

    def __init__(self):
        super().__init__()

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
{aggregation_code}{invoke_body}
'''

    def _generate_branch_definitions(self, node: WorkflowNodeIR, agent_ir: AgentIR) -> str:
        """Generate branch definitions for BranchRouter"""
        condition_returns = node.config.get("condition_returns", [])

        if not condition_returns:
            return ""

        # Find the mapping from conditional edges
        mapping = {}
        for edge in agent_ir.metadata.get("conditional_edges", []):
            if edge.get("source_node") == node.id:
                mapping = edge.get("mapping", {})
                break

        lines = []
        for i, ret_val in enumerate(condition_returns, 1):
            target = mapping.get(ret_val, ret_val)
            if target == "END":
                target = "end"
            lines.append(
                f'{self._indent * 2}self._router.add_branch(\n'
                f'{self._indent * 3}condition="${{{node.id}.route}} == \'{ret_val}\'",\n'
                f'{self._indent * 3}target=["{target}"],\n'
                f'{self._indent * 3}branch_id="{i}"\n'
                f'{self._indent * 2})'
            )

        return "\n".join(lines)

    def _fix_fstring_quotes(self, code: str) -> str:
        """Fix nested quotes in f-strings for Python < 3.12 compatibility.

        Changes outer quotes from single to double when there are single quotes
        inside the f-string expression part (inside {}).
        """
        result = []
        i = 0
        n = len(code)

        while i < n:
            # Check for f-string start: f' or f" or F' or F"
            if i < n - 1 and code[i] in 'fF' and code[i + 1] in '\'"':
                prefix = code[i]
                outer_quote = code[i + 1]
                fstring_start = i
                i += 2  # Skip f and opening quote

                # Parse the f-string content
                content = []
                has_quotes_in_expr = False
                depth = 0  # Brace depth for nested expressions

                while i < n:
                    char = code[i]

                    if char == '\\' and i + 1 < n:
                        # Escaped character - keep as is
                        content.append(code[i:i+2])
                        i += 2
                        continue

                    if char == outer_quote and depth == 0:
                        # End of f-string
                        break

                    if char == '{':
                        if i + 1 < n and code[i + 1] == '{':
                            # Escaped brace {{
                            content.append('{{')
                            i += 2
                            continue
                        depth += 1
                        content.append(char)
                    elif char == '}':
                        if i + 1 < n and code[i + 1] == '}':
                            # Escaped brace }}
                            content.append('}}')
                            i += 2
                            continue
                        depth -= 1
                        content.append(char)
                    else:
                        content.append(char)
                        # Check if there's a quote inside an expression
                        if depth > 0 and char == outer_quote:
                            has_quotes_in_expr = True

                    i += 1

                content_str = ''.join(content)

                # If single quotes outside and single quotes inside expressions, switch to double
                if outer_quote == "'" and has_quotes_in_expr:
                    # Switch to double quotes - escape any existing double quotes in content
                    fixed_content = content_str.replace('"', '\\"')
                    result.append(f'{prefix}"{fixed_content}"')
                else:
                    result.append(f'{prefix}{outer_quote}{content_str}{outer_quote}')

                i += 1  # Skip closing quote
            else:
                result.append(code[i])
                i += 1

        return ''.join(result)

    def _convert_function_body(self, node: WorkflowNodeIR, is_conditional: bool = False,
                                agent_ir: AgentIR = None) -> str:
        """Convert LangGraph function body to openJiuwen format.

        Component only handles business logic, same as LangGraph node.
        Routing logic is handled by router function separately.
        Framework automatically stores return values to global state.
        """
        body_statements = node.config.get("body_statements", [])
        return_statement = node.config.get("return_statement", None)
        return_keys = node.config.get("return_keys", [])
        uses_llm = node.config.get("uses_llm", False)

        lines = []

        # Initialize return_keys variables to None to ensure they're defined in all code paths
        # This handles cases like try/except where different variables are set in different branches
        if return_keys:
            # Deduplicate return_keys while preserving order
            unique_keys = list(dict.fromkeys(return_keys))
            lines.append(f'{self._indent * 2}# Initialize output variables')
            for key in unique_keys:
                lines.append(f'{self._indent * 2}{key} = None')
            lines.append("")

        # Add the actual business logic statements (same as LangGraph node)
        if body_statements:
            for stmt in body_statements:
                # Fix any nested quote issues in f-strings
                fixed_stmt = self._fix_fstring_quotes(stmt)

                # Convert LLM calls if this node uses LLM
                if uses_llm:
                    fixed_stmt = self._convert_llm_calls(fixed_stmt)

                # Add proper indentation
                for line in fixed_stmt.split('\n'):
                    lines.append(f'{self._indent * 2}{line}')
            lines.append("")

        # Add aggregation logic for fields with aggregators (like messages)
        aggregation_code = self._generate_aggregation_logic(node, agent_ir)
        if aggregation_code:
            lines.append(aggregation_code)

        # Add return statement - framework will auto-store to global state
        if return_statement:
            # Replace aggregated field values with their aggregated versions
            modified_return = return_statement
            if agent_ir:
                for field in agent_ir.state_fields:
                    if field.has_aggregator and field.name in return_keys:
                        original_value = self._extract_field_value_from_return(modified_return, field.name)
                        if original_value:
                            modified_return = modified_return.replace(
                                f'"{field.name}": {original_value}',
                                f'"{field.name}": _{field.name}_aggregated'
                            )
            # Convert the return statement
            converted_return = self._convert_return_statement(modified_return, is_conditional=False)
            lines.append(f'{self._indent * 2}return {converted_return}')
        else:
            # Generate default return
            if return_keys:
                return_dict = ', '.join([f'"{k}": {k}' for k in return_keys])
                lines.append(f'{self._indent * 2}return {{{return_dict}}}')
            else:
                lines.append(f'{self._indent * 2}return ' + '{}')

        return "\n".join(lines)

    def _convert_return_statement(self, return_stmt: str, is_conditional: bool) -> str:
        """Convert return statement, adding route if conditional"""
        import re
        # Convert any remaining state references in return statement
        return_stmt = re.sub(r'state\["(\w+)"\]', r'\1', return_stmt)
        return_stmt = re.sub(r'state\.get\("(\w+)"[^)]*\)', r'\1', return_stmt)

        if is_conditional and '"route"' not in return_stmt:
            # Insert route into the return dict
            if return_stmt.startswith("{") and return_stmt.endswith("}"):
                inner = return_stmt[1:-1].strip()
                if inner:
                    return '{"route": route, ' + inner + '}'
                else:
                    return '{"route": route}'
        return return_stmt

    def _convert_llm_calls(self, stmt: str) -> str:
        """Convert LangGraph LLM calls to openJiuwen format.

        Transforms:
        - llm.invoke(messages) -> await self._llm.ainvoke(model_name=MODEL_NAME, messages=messages, temperature=0.7, top_p=0.9)
        - llm.invoke(messages).content -> (await self._llm.ainvoke(...)).content
        - ans = llm.invoke(messages).content.strip() -> llm_response = await self._llm.ainvoke(...); ans = llm_response.content.strip()
        """
        import re

        # Pattern: var = llm.invoke(messages).content.strip() or similar
        # Convert to: llm_response = await self._llm.ainvoke(...)\nvar = llm_response.content...
        # Note: No extra indentation - _convert_function_body handles indentation
        pattern1 = r'(\w+)\s*=\s*(\w+)\.invoke\(([^)]+)\)\.content(\.strip\(\))?'
        match1 = re.search(pattern1, stmt)
        if match1:
            var_name = match1.group(1)
            messages_arg = match1.group(3)
            strip_call = match1.group(4) or ""
            return f"llm_response = await self._llm.ainvoke(model_name=MODEL_NAME, messages={messages_arg}, temperature=0.7, top_p=0.9)\n{var_name} = llm_response.content{strip_call}"

        # Pattern: llm.invoke(messages).content
        pattern2 = r'(\w+)\.invoke\(([^)]+)\)\.content'
        if re.search(pattern2, stmt):
            stmt = re.sub(pattern2, r'(await self._llm.ainvoke(model_name=MODEL_NAME, messages=\2, temperature=0.7, top_p=0.9)).content', stmt)
            return stmt

        # Pattern: llm.invoke(messages)
        pattern3 = r'(\w+)\.invoke\(([^)]+)\)'
        if re.search(pattern3, stmt):
            stmt = re.sub(pattern3, r'await self._llm.ainvoke(model_name=MODEL_NAME, messages=\2, temperature=0.7, top_p=0.9)', stmt)
            return stmt

        return stmt

    def _find_routing_variable(self, body_statements: List[str], return_keys: List[str]) -> Optional[str]:
        """Find the variable used for routing decision from body statements"""
        import re

        # Look for variables that indicate validity/result
        for stmt in body_statements:
            # Match patterns like: is_valid = ..., result = ..., status = ...
            match = re.match(r'^\s*(is_\w+|result|status|valid\w*)\s*=', stmt)
            if match:
                var_name = match.group(1)
                return var_name

        # Fallback to return keys
        for key in return_keys:
            if 'valid' in key.lower() or 'result' in key.lower() or 'status' in key.lower():
                return key

        return None

    def _generate_workflow_builder(self, agent_ir: AgentIR) -> str:
        """Generate workflow builder function"""
        if not agent_ir.workflow:
            return ""

        lines = [
            "# ============ Workflow Builder ============",
            "",
            "def create_workflow() -> Workflow:",
            '    """Create the openJiuwen workflow"""',
            "    flow = Workflow()",
            "",
        ]

        # Generate start component
        start_inputs = self._build_start_inputs(agent_ir)
        lines.extend([
            "    # Start component",
            "    flow.set_start_comp(",
            '        "start",',
            "        Start(),",
            f'        inputs_schema={start_inputs}',
            "    )",
            "",
        ])

        # Generate node components
        conditional_nodes = []
        for node in agent_ir.workflow.nodes:
            if node.node_type in (WorkflowNodeType.START, WorkflowNodeType.END):
                continue

            class_name = self._to_class_name(node.id)
            inputs_schema = self._build_inputs_schema(node, agent_ir.workflow)
            is_conditional = node.config.get("is_conditional", False)

            if is_conditional:
                conditional_nodes.append(node.id)
                # v2: Simple component registration, routing handled by add_conditional_connection
                lines.extend([
                    f'    # {node.name} component (conditional)',
                    "    flow.add_workflow_comp(",
                    f'        "{node.id}",',
                    f"        {class_name}(),",
                    f'        inputs_schema={inputs_schema}',
                    "    )",
                    "",
                ])
            else:
                lines.extend([
                    f'    # {node.name} component',
                    "    flow.add_workflow_comp(",
                    f'        "{node.id}",',
                    f"        {class_name}(),",
                    f'        inputs_schema={inputs_schema}',
                    "    )",
                    "",
                ])

        # Generate end component
        end_inputs = self._build_end_inputs(agent_ir)
        lines.extend([
            "    # End component",
            "    flow.set_end_comp(",
            '        "end",',
            "        End(),",
            f'        inputs_schema={end_inputs}',
            "    )",
            "",
        ])

        # Generate connections (deduplicated)
        lines.extend([
            "    # Connections",
        ])

        seen_connections = set()

        # Add regular edge connections (non-conditional)
        for edge in agent_ir.workflow.edges:
            if not edge.condition:
                source = edge.source_node
                target = edge.target_node if edge.target_node != "END" else "end"
                conn_key = (source, target)
                if conn_key not in seen_connections:
                    seen_connections.add(conn_key)
                    lines.append(f'    flow.add_connection("{source}", "{target}")')

        # v2: Add conditional connections using router functions
        if conditional_nodes:
            lines.append("")
            lines.append("    # Conditional connections (v2: using router functions)")
            for node_id in conditional_nodes:
                lines.append(f'    flow.add_conditional_connection("{node_id}", router={node_id}_router)')

        lines.extend([
            "",
            "    return flow",
        ])

        return "\n".join(lines)

    def _build_start_inputs(self, agent_ir: AgentIR) -> str:
        """Build inputs_schema for Start component - only actual user inputs"""
        fields = agent_ir.state_fields
        if not fields:
            return '{}'

        # Determine which fields are actual user inputs vs computed
        # User inputs are typically: fields that are read by the first node
        # and not produced by any node as output
        output_map = self._build_output_map(agent_ir.workflow, agent_ir)

        # Collect all fields that are output by some node
        all_outputs = set()
        for node_id, outputs in output_map.items():
            if node_id != "start":
                all_outputs.update(outputs)

        # Find the entry node and see what it reads
        entry_node = None
        if agent_ir.workflow:
            for edge in agent_ir.workflow.edges:
                if edge.source_node == "start":
                    entry_node = edge.target_node
                    break

        entry_inputs = set()
        if entry_node:
            for node in agent_ir.workflow.nodes:
                if node.id == entry_node:
                    for access in node.config.get("state_accesses", []):
                        entry_inputs.add(access.get("key", ""))

        parts = []
        for f in fields:
            # Include field if:
            # 1. It's read by the entry node, OR
            # 2. It has an aggregator (needs initialization), OR
            # 3. It's not produced by any node (true user input)
            is_user_input = (
                f.name in entry_inputs or
                f.has_aggregator or
                f.name not in all_outputs
            )

            if is_user_input:
                parts.append(f'"{f.name}": "${{user_inputs.{f.name}}}"')

        return "{" + ", ".join(parts) + "}"

    def _build_end_inputs(self, agent_ir: AgentIR) -> str:
        """Build inputs_schema for End component handling multiple sources

        End node should receive ALL outputs from ALL nodes that connect to it.
        This ensures the workflow result contains all relevant data.
        """
        # Find all nodes that connect to END (explicit edges)
        end_edges = [e for e in agent_ir.workflow.edges if e.target_node == "END" or e.target_node == "end"]
        source_nodes = list(set(e.source_node for e in end_edges))

        # Also check conditional edges that may route to END
        conditional_edges = agent_ir.metadata.get("conditional_edges", [])
        for ce in conditional_edges:
            if "END" in ce.get("mapping", {}).values() or "end" in ce.get("mapping", {}).values():
                source = ce.get("source_node")
                if source and source not in source_nodes:
                    source_nodes.append(source)

        # Also check nodes with conditional routing that can go to end
        if not source_nodes:
            for node in agent_ir.workflow.nodes:
                if node.config.get("is_conditional"):
                    condition_returns = node.config.get("condition_returns", [])
                    if "END" in condition_returns:
                        source_nodes.append(node.id)

        if not source_nodes:
            return '{}'

        # Find output fields from source nodes
        output_map = self._build_output_map(agent_ir.workflow, agent_ir)

        # Collect ALL outputs from ALL source nodes
        parts = []
        seen_fields = set()  # Track field names to handle duplicates

        for src in source_nodes:
            src_outputs = output_map.get(src, [])
            for output_field in src_outputs:
                # Use node.field as the key to avoid conflicts between nodes
                key = f"{src}_{output_field}"
                if key not in seen_fields:
                    seen_fields.add(key)
                    parts.append(f'"{key}": "${{{src}.{output_field}}}"')

        if not parts:
            return '{}'

        return "{" + ", ".join(parts) + "}"

    def _build_inputs_schema(self, node: WorkflowNodeIR, workflow: WorkflowIR) -> str:
        """Build inputs_schema for a node with proper data flow tracking"""
        state_accesses = node.config.get("state_accesses", [])

        # For LLM nodes, only use read accesses as inputs
        if node.node_type == WorkflowNodeType.LLM:
            state_accesses = [a for a in state_accesses if a.get("access_type") == "read"]

        if not state_accesses:
            return '{}'

        # Build a map of which node outputs which fields
        output_map = self._build_output_map(workflow, self._agent_ir)

        # Find upstream nodes in topological order
        upstream_nodes = self._get_upstream_chain(node.id, workflow)

        parts = []
        seen_keys = set()
        for access in state_accesses:
            key = access.get("key", "")
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            # Find the closest upstream node that outputs this key
            source = self._find_source_for_key(key, upstream_nodes, output_map)
            parts.append(f'"{key}": "${{{source}.{key}}}"')

        return "{" + ", ".join(parts) + "}"

    def _build_output_map(self, workflow: WorkflowIR, agent_ir: AgentIR = None) -> Dict[str, List[str]]:
        """Build a map of node_id -> list of output keys"""
        output_map = {}
        for node in workflow.nodes:
            if node.node_type == WorkflowNodeType.LLM:
                # LLM nodes output the key defined in output_config
                llm_output_key = node.config.get("llm_output_key")
                if llm_output_key:
                    output_map[node.id] = [llm_output_key]
                else:
                    # Fallback to return_keys or write accesses
                    return_keys = node.config.get("return_keys", [])
                    if return_keys:
                        output_map[node.id] = list(return_keys)
                    else:
                        # Check write accesses
                        state_accesses = node.config.get("state_accesses", [])
                        write_keys = [a.get("key") for a in state_accesses if a.get("access_type") == "write"]
                        output_map[node.id] = write_keys if write_keys else ["answer"]
            else:
                return_keys = node.config.get("return_keys", [])
                # Only include actual return keys, not state accesses
                output_map[node.id] = list(return_keys)

        # Start node outputs all state fields (user inputs)
        if agent_ir and agent_ir.state_fields:
            output_map["start"] = [f.name for f in agent_ir.state_fields]
        else:
            output_map["start"] = []

        return output_map

    def _get_upstream_chain(self, node_id: str, workflow: WorkflowIR) -> List[str]:
        """Get all upstream nodes in reverse topological order (closest first)"""
        visited = set()
        chain = []

        def dfs(nid):
            if nid in visited or nid == node_id:
                return
            visited.add(nid)
            chain.append(nid)
            for upstream in workflow.get_upstream_nodes(nid):
                dfs(upstream)

        for upstream in workflow.get_upstream_nodes(node_id):
            dfs(upstream)
            if upstream not in visited:
                chain.append(upstream)
                visited.add(upstream)

        return chain

    def _find_source_for_key(self, key: str, upstream_nodes: List[str], output_map: Dict[str, List[str]]) -> str:
        """Find the closest upstream node that outputs the given key"""
        for node_id in upstream_nodes:
            if key in output_map.get(node_id, []):
                return node_id

        # Fallback to start if not found
        return "start"

    def _generate_main(self, agent_ir: AgentIR) -> str:
        """Generate main function and entry point"""
        initial_state = self._generate_initial_state(agent_ir)

        return f'''# ============ Main Entry Point ============

async def run_workflow(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run the workflow with given inputs"""
    flow = create_workflow()
    runtime = WorkflowRuntime(
        workflow_id="{agent_ir.name}",
        session_id="default"
    )

    result = await flow.invoke(
        {{"user_inputs": inputs}},
        runtime
    )
    return result


# Example usage
if __name__ == "__main__":
    import asyncio

    # Example initial state
    initial_inputs = {initial_state}

    # Run the workflow
    result = asyncio.run(run_workflow(initial_inputs))
    print("Result:", result)
'''

    def _generate_initial_state(self, agent_ir: AgentIR) -> str:
        """Generate example initial state based on type hints and example_inputs from source"""
        if not agent_ir.state_fields:
            return '{}'

        # Get example inputs from metadata (parsed from if __name__ == "__main__":)
        example_inputs = agent_ir.metadata.get("example_inputs", {})

        parts = []
        for f in agent_ir.state_fields:
            # Use example value if available
            if f.name in example_inputs:
                value = example_inputs[f.name]
                if isinstance(value, str):
                    parts.append(f'"{f.name}": "{value}"')
                elif isinstance(value, (list, dict)):
                    import json
                    parts.append(f'"{f.name}": {json.dumps(value, ensure_ascii=False)}')
                else:
                    parts.append(f'"{f.name}": {value}')
            # Fall back to type-based defaults
            elif "list" in f.type_hint.lower() or "List" in f.type_hint:
                parts.append(f'"{f.name}": []')
            elif "dict" in f.type_hint.lower() or "Dict" in f.type_hint:
                parts.append(f'"{f.name}": {{}}')
            elif "bool" in f.type_hint.lower():
                parts.append(f'"{f.name}": False')
            elif "int" in f.type_hint.lower():
                parts.append(f'"{f.name}": 0')
            elif "str" in f.type_hint.lower():
                parts.append(f'"{f.name}": ""')
            else:
                parts.append(f'"{f.name}": None')

        return "{\n        " + ",\n        ".join(parts) + "\n    }"

    def _to_class_name(self, node_id: str) -> str:
        """Convert node_id to PascalCase class name"""
        parts = node_id.replace("-", "_").split("_")
        return "".join(p.capitalize() for p in parts) + "Component"
