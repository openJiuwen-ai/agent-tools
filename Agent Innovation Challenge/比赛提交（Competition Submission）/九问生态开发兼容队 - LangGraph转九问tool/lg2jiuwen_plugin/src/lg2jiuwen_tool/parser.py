"""
LangGraph AST Parser

Parses LangGraph source code using Python AST and extracts:
- State class definitions
- Node functions
- Edges and conditional edges
- Tool definitions
- LLM configurations
"""

import ast
import re
from typing import Dict, List, Optional, Set, Tuple
from .ir_models import (
    ParseResult,
    StateFieldInfo,
    NodeFunctionInfo,
    ParamInfo,
    StateAccess,
    EdgeInfo,
    ConditionalEdgeInfo,
    ToolInfo,
    LLMConfigIR,
    GlobalVarInfo,
)


class LangGraphParser:
    """Parser for LangGraph source code"""

    # Known LLM class names from various providers
    LLM_CLASS_NAMES = {
        "ChatOpenAI", "OpenAI", "AzureChatOpenAI", "AzureOpenAI",
        "ChatAnthropic", "Anthropic", "ChatGoogleGenerativeAI",
        "ChatOllama", "Ollama", "ChatHuggingFace", "HuggingFaceHub",
        "ChatCohere", "Cohere", "ChatMistralAI", "ChatVertexAI",
        "ChatBedrock", "BedrockChat", "ChatTongyi", "Tongyi",
        "ChatZhipuAI", "ChatBaichuan", "ChatSparkLLM", "ChatWenxin",
    }

    def __init__(self):
        self.state_class_name: Optional[str] = None
        self.state_fields: List[StateFieldInfo] = []
        self.node_functions: Dict[str, NodeFunctionInfo] = {}
        self.conditional_functions: Dict[str, NodeFunctionInfo] = {}
        self.edges: List[EdgeInfo] = []
        self.conditional_edges: List[ConditionalEdgeInfo] = []
        self.tools: List[ToolInfo] = []
        self.entry_point: Optional[str] = None
        self.graph_variable: Optional[str] = None
        self.imports: List[str] = []
        self.import_statements: List[str] = []  # Full import statement strings
        self.global_variables: List[GlobalVarInfo] = []
        self._registered_nodes: Set[str] = set()
        self._node_to_func_map: Dict[str, str] = {}  # node_name -> function_name mapping from add_node()
        self._all_functions: Dict[str, NodeFunctionInfo] = {}  # All parsed functions (temporary storage)
        self._source_lines: List[str] = []
        # LLM related fields
        self.llm_config: Optional[LLMConfigIR] = None
        self.llm_variable_name: Optional[str] = None
        self._llm_using_functions: Set[str] = set()  # Functions that call LLM
        # Example inputs from if __name__ == "__main__"
        self.example_inputs: Dict[str, any] = {}

    def parse(self, source_code: str) -> ParseResult:
        """Parse LangGraph source code and extract components"""
        self._reset()
        self._source_lines = source_code.split('\n')

        tree = ast.parse(source_code)

        # First pass: collect module-level definitions only
        for node in tree.body:
            if isinstance(node, ast.Import):
                self._parse_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._parse_import_from(node)
            elif isinstance(node, ast.ClassDef):
                self._parse_class(node)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                self._parse_function(node)
            elif isinstance(node, ast.Assign):
                self._parse_assignment(node, source_code, is_module_level=True)
            elif isinstance(node, ast.If):
                # Check for if __name__ == "__main__":
                self._parse_main_block(node)

        # Second pass: parse graph construction (need ast.walk for method calls)
        self._parse_graph_construction(tree, source_code)

        return self._build_result(source_code)

    def _reset(self):
        """Reset parser state"""
        self.state_class_name = None
        self.state_fields = []
        self.node_functions = {}
        self.conditional_functions = {}
        self.edges = []
        self.conditional_edges = []
        self.tools = []
        self.entry_point = None
        self.graph_variable = None
        self.imports = []
        self.import_statements = []
        self.global_variables = []
        self._registered_nodes = set()
        self._node_to_func_map = {}  # Reset node-to-function mapping
        self._all_functions = {}  # Reset all functions storage
        self._source_lines = []
        # Reset LLM related fields
        self.llm_config = None
        self.llm_variable_name = None
        self._llm_using_functions = set()
        # Reset example inputs
        self.example_inputs = {}

    def _parse_import(self, node: ast.Import):
        """Parse import statement"""
        for alias in node.names:
            self.imports.append(alias.name)
        # Store full import statement
        self.import_statements.append(ast.unparse(node))

    def _parse_import_from(self, node: ast.ImportFrom):
        """Parse from ... import statement"""
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        # Store full import statement
        self.import_statements.append(ast.unparse(node))

    def _parse_main_block(self, node: ast.If):
        """Parse if __name__ == "__main__": block to extract example inputs

        Looks for patterns like:
        - sentence = "明天北京天气"
        - result = graph.invoke({"sentence": sentence})
        """
        # Check if this is if __name__ == "__main__":
        if not (isinstance(node.test, ast.Compare) and
                isinstance(node.test.left, ast.Name) and
                node.test.left.id == "__name__" and
                len(node.test.comparators) == 1 and
                isinstance(node.test.comparators[0], ast.Constant) and
                node.test.comparators[0].value == "__main__"):
            return

        # Track local variables defined in main block
        local_vars = {}

        # Parse statements in main block
        for stmt in node.body:
            # Track variable assignments: sentence = "明天北京天气"
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        # Try to evaluate the value
                        if isinstance(stmt.value, ast.Constant):
                            local_vars[var_name] = stmt.value.value
                        elif isinstance(stmt.value, ast.List):
                            # Handle list literals
                            local_vars[var_name] = self._eval_list_literal(stmt.value)
                        elif isinstance(stmt.value, ast.Dict):
                            # Handle dict literals
                            local_vars[var_name] = self._eval_dict_literal(stmt.value, local_vars)

            # Look for graph.invoke({...}) or graph.invoke(var) calls
            if isinstance(stmt, ast.Assign) or isinstance(stmt, ast.Expr):
                call_node = None
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    call_node = stmt.value
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    call_node = stmt.value

                if call_node and isinstance(call_node.func, ast.Attribute):
                    if call_node.func.attr == "invoke" and call_node.args:
                        # Extract the invoke argument
                        arg = call_node.args[0]
                        if isinstance(arg, ast.Dict):
                            # Direct dict literal: graph.invoke({"key": value})
                            self.example_inputs = self._eval_dict_literal(arg, local_vars)
                        elif isinstance(arg, ast.Name):
                            # Variable reference: graph.invoke(initial_state)
                            var_name = arg.id
                            if var_name in local_vars and isinstance(local_vars[var_name], dict):
                                self.example_inputs = local_vars[var_name]

    def _eval_list_literal(self, node: ast.List) -> list:
        """Evaluate a list literal AST node"""
        result = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                result.append(elt.value)
            elif isinstance(elt, ast.Dict):
                result.append(self._eval_dict_literal(elt, {}))
        return result

    def _eval_dict_literal(self, node: ast.Dict, local_vars: dict) -> dict:
        """Evaluate a dict literal AST node, resolving local variable references"""
        result = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant):
                key_str = key.value
                # Resolve the value
                if isinstance(value, ast.Constant):
                    result[key_str] = value.value
                elif isinstance(value, ast.Name):
                    # Variable reference - look up in local_vars
                    var_name = value.id
                    if var_name in local_vars:
                        result[key_str] = local_vars[var_name]
                    else:
                        result[key_str] = f"${{{var_name}}}"  # Placeholder
                elif isinstance(value, ast.List):
                    result[key_str] = self._eval_list_literal(value)
                elif isinstance(value, ast.Dict):
                    result[key_str] = self._eval_dict_literal(value, local_vars)
        return result

    def _parse_class(self, node: ast.ClassDef):
        """Parse class definition"""
        # Check if it's a TypedDict (State class)
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name in ("TypedDict", "typing.TypedDict"):
                self._parse_state_class(node)
                return

    def _parse_state_class(self, node: ast.ClassDef):
        """Parse State TypedDict class"""
        self.state_class_name = node.name

        for item in node.body:
            if isinstance(item, ast.AnnAssign) and item.target:
                field_name = self._get_name(item.target)
                type_hint = ast.unparse(item.annotation) if item.annotation else "Any"

                # Check for Annotated with aggregator
                has_aggregator = False
                aggregator = None
                if "Annotated" in type_hint:
                    has_aggregator = True
                    # Extract aggregator function
                    match = re.search(r'Annotated\[.*?,\s*([\w.]+)\]', type_hint)
                    if match:
                        aggregator = match.group(1)

                self.state_fields.append(StateFieldInfo(
                    name=field_name,
                    type_hint=type_hint,
                    has_aggregator=has_aggregator,
                    aggregator=aggregator
                ))

    def _parse_function(self, node: ast.FunctionDef):
        """Parse function definition

        Store all functions to _all_functions for later filtering.
        The actual node functions are determined by add_node() calls.
        """
        # Check for @tool decorator
        if self._has_decorator(node, "tool"):
            self._parse_tool_function(node)
            return

        # Extract function info and store to _all_functions
        # We'll determine if it's a node function later based on add_node() calls
        func_info = self._extract_function_info(node)
        self._all_functions[node.name] = func_info

    def _is_node_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is a node function"""
        if not node.args.args:
            return False

        first_param = node.args.args[0]
        if first_param.annotation:
            annotation = ast.unparse(first_param.annotation)
            # Check if annotation matches state class name or common patterns
            if self.state_class_name and self.state_class_name in annotation:
                return True
            if "State" in annotation or "state" in first_param.arg:
                return True
        elif first_param.arg == "state":
            return True

        return False

    def _extract_function_info(self, node: ast.FunctionDef) -> NodeFunctionInfo:
        """Extract information from a function"""
        # Get docstring
        docstring = ast.get_docstring(node)

        # Get function body as string (full function)
        full_body = ast.unparse(node)

        # Extract just the function body (excluding def line and docstring)
        body_statements = self._extract_function_body_statements(node)

        # Extract parameters
        params = []
        for arg in node.args.args:
            type_hint = ast.unparse(arg.annotation) if arg.annotation else None
            params.append(ParamInfo(name=arg.arg, type_hint=type_hint))

        # Find state accesses
        state_accesses = self._find_state_accesses(node)

        # Find return keys and return statement
        return_keys = self._find_return_keys(node)

        # Extract return statement
        return_statement = self._extract_return_statement(node)

        # Check if conditional function
        is_conditional = self._is_conditional_function(node)
        condition_returns = []
        if is_conditional:
            condition_returns = self._extract_condition_returns(node)

        # Check if this function uses LLM
        uses_llm = self._function_uses_llm(node)
        if uses_llm:
            self._llm_using_functions.add(node.name)

        func_info = NodeFunctionInfo(
            name=node.name,
            params=params,
            body=full_body,
            docstring=docstring,
            state_accesses=state_accesses,
            return_keys=return_keys,
            is_conditional=is_conditional,
            condition_returns=condition_returns
        )

        # Store additional extracted info
        func_info.body_statements = body_statements
        func_info.return_statement = return_statement
        func_info.uses_llm = uses_llm  # Mark if function uses LLM

        return func_info

    def _function_uses_llm(self, node: ast.FunctionDef) -> bool:
        """Check if a function uses LLM (calls llm.invoke, llm.call, etc.)"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Check for llm.invoke(), llm.call(), llm.generate(), etc.
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr in ("invoke", "call", "generate", "predict", "agenerate", "ainvoke"):
                        # Check if the object is likely an LLM
                        if isinstance(child.func.value, ast.Name):
                            var_name = child.func.value.id
                            # Check if it matches known LLM variable or common names
                            if (self.llm_variable_name and var_name == self.llm_variable_name) or \
                               var_name.lower() in ("llm", "model", "chat", "chatmodel", "chat_model"):
                                return True
        return False

    def _extract_function_body_statements(self, node: ast.FunctionDef) -> List[str]:
        """Extract individual statements from function body"""
        statements = []
        state_param = node.args.args[0].arg if node.args.args else "state"

        for stmt in node.body:
            # Skip docstring
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    continue

            # Skip return statement (handled separately)
            if isinstance(stmt, ast.Return):
                continue

            stmt_str = ast.unparse(stmt)

            # Convert state access patterns
            stmt_str = self._convert_state_access(stmt_str, state_param)

            statements.append(stmt_str)

        return statements

    def _convert_state_access(self, code: str, state_param: str) -> str:
        """Convert state access patterns to openJiuwen format.

        In LangGraph, state is a shared dict passed between nodes.
        In openJiuwen:
        - Read data from upstream: inputs.get("key")
        - Write data: use local variables, then return them
        - Runtime automatically records the return values to the node's state

        Transformations:
        - Read: state["key"] -> inputs.get("key")
        - Read: state.get("key") -> inputs.get("key")
        - Write: state["key"] = value -> key = value (local variable)
        - AugAssign: state["key"] -= 1 -> key = inputs.get("key") - 1
        - return state -> return {"key1": key1, "key2": key2, ...}
        """
        import re

        state_write_keys = []

        # Step 1: Find all state write keys from regular assignments
        write_pattern = rf'{state_param}\[(["\'])(\w+)\1\]\s*=\s*(.+)'
        for match in re.finditer(write_pattern, code):
            key = match.group(2)
            if key not in state_write_keys:
                state_write_keys.append(key)

        # Step 2: Find all state write keys from augmented assignments (+=, -=, etc.)
        # Pattern: state["key"] += value or state["key"] -= value
        aug_pattern = rf'{state_param}\[(["\'])(\w+)\1\]\s*([+\-*/%]|//|<<|>>|&|\||\^)=\s*(.+)'
        for match in re.finditer(aug_pattern, code):
            key = match.group(2)
            if key not in state_write_keys:
                state_write_keys.append(key)

        # Step 3: Apply augmented assignment replacement FIRST
        # state["key"] -= value -> key = inputs.get("key") - value
        def aug_replacement(match):
            key = match.group(2)
            op = match.group(3)
            value = match.group(4)
            return f'{key} = inputs.get("{key}") {op} {value}'

        code = re.sub(aug_pattern, aug_replacement, code)

        # Step 4: Apply regular write pattern replacement: state["key"] = value -> key = value
        def write_replacement(match):
            key = match.group(2)
            value = match.group(3)
            return f'{key} = {value}'

        code = re.sub(write_pattern, write_replacement, code)

        # Step 5: Handle return state -> generate return dict with collected state write keys
        return_state_pattern = rf'\breturn\s+{state_param}\b'
        if re.search(return_state_pattern, code):
            if state_write_keys:
                return_dict = ', '.join([f'"{k}": {k}' for k in state_write_keys])
                return_statement = f'return {{{return_dict}}}'
            else:
                return_statement = 'return {}'
            code = re.sub(return_state_pattern, return_statement, code)

        # Step 6: Handle read patterns: state["key"] -> inputs.get("key")
        code = re.sub(
            rf'{state_param}\[(["\'])(\w+)\1\]',
            r'inputs.get("\2")',
            code
        )

        # Convert state.get("key", default) -> inputs.get("key", default)
        code = re.sub(
            rf'{state_param}\.get\(',
            'inputs.get(',
            code
        )

        return code

    def _extract_return_statement(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract the return statement from function.

        Handles:
        - return {"key": value, ...} -> {"key": value, ...}
        - return state -> generate return dict from write accesses
        """
        state_param = node.args.args[0].arg if node.args.args else "state"

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value:
                # Case 1: return {"key": value, ...}
                if isinstance(stmt.value, ast.Dict):
                    parts = []
                    for key, value in zip(stmt.value.keys, stmt.value.values):
                        if isinstance(key, ast.Constant):
                            key_str = key.value
                            val_str = ast.unparse(value)
                            parts.append(f'"{key_str}": {val_str}')
                    return "{" + ", ".join(parts) + "}"

                # Case 2: return state -> generate return dict from write accesses
                if isinstance(stmt.value, ast.Name) and stmt.value.id == state_param:
                    # Find all write accesses in this function
                    write_keys = self._find_write_keys(node, state_param)
                    if write_keys:
                        # Generate return dict with local variable references
                        parts = [f'"{k}": {k}' for k in write_keys]
                        return "{" + ", ".join(parts) + "}"
                    return "{}"

        return None

    def _find_write_keys(self, node: ast.FunctionDef, state_param: str) -> List[str]:
        """Find all keys that are written to state in a function."""
        write_keys = []
        for child in ast.walk(node):
            # Pattern: state["key"] = value
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Name) and target.value.id == state_param:
                            if isinstance(target.slice, ast.Constant):
                                key = target.slice.value
                                if key not in write_keys:
                                    write_keys.append(key)
            # Pattern: state["key"] -= value (augmented assignment)
            elif isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Subscript):
                    if isinstance(child.target.value, ast.Name) and child.target.value.id == state_param:
                        if isinstance(child.target.slice, ast.Constant):
                            key = child.target.slice.value
                            if key not in write_keys:
                                write_keys.append(key)
        return write_keys

    def _find_state_accesses(self, node: ast.FunctionDef) -> List[StateAccess]:
        """Find all state access patterns in function (both read and write)"""
        accesses = []
        state_param = node.args.args[0].arg if node.args.args else "state"

        for child in ast.walk(node):
            # Pattern: state["key"] = value (write access)
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Name) and target.value.id == state_param:
                            if isinstance(target.slice, ast.Constant):
                                accesses.append(StateAccess(
                                    key=target.slice.value,
                                    access_type="write"
                                ))

            # Pattern: state["key"] -= 1 (augmented assignment - also write access)
            if isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Subscript):
                    if isinstance(child.target.value, ast.Name) and child.target.value.id == state_param:
                        if isinstance(child.target.slice, ast.Constant):
                            accesses.append(StateAccess(
                                key=child.target.slice.value,
                                access_type="write"
                            ))

            # Pattern: state["key"] (read access - when not in assignment target)
            if isinstance(child, ast.Subscript):
                if isinstance(child.value, ast.Name) and child.value.id == state_param:
                    if isinstance(child.slice, ast.Constant):
                        # Check if this is not an assignment target (regular or augmented)
                        is_write_target = False
                        for assign in ast.walk(node):
                            if isinstance(assign, ast.Assign):
                                for t in assign.targets:
                                    if t is child:
                                        is_write_target = True
                                        break
                            elif isinstance(assign, ast.AugAssign):
                                if assign.target is child:
                                    is_write_target = True
                                    break
                        if not is_write_target:
                            accesses.append(StateAccess(
                                key=child.slice.value,
                                access_type="read"
                            ))

            # Pattern: state.get("key") or state.get("key", default)
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if (isinstance(child.func.value, ast.Name) and
                        child.func.value.id == state_param and
                        child.func.attr == "get"):
                        if child.args and isinstance(child.args[0], ast.Constant):
                            default_val = None
                            if len(child.args) > 1:
                                default_val = ast.unparse(child.args[1])
                            accesses.append(StateAccess(
                                key=child.args[0].value,
                                access_type="read",
                                default_value=default_val
                            ))

        return accesses

    def _find_return_keys(self, node: ast.FunctionDef) -> List[str]:
        """Find keys in return statement dict, or infer from state writes"""
        keys = []
        state_param = node.args.args[0].arg if node.args.args else "state"

        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                if isinstance(child.value, ast.Dict):
                    # Return statement is a dict literal - extract keys directly
                    for key in child.value.keys:
                        if isinstance(key, ast.Constant):
                            keys.append(key.value)
                elif isinstance(child.value, ast.Name) and child.value.id == state_param:
                    # Return statement is "return state" - infer keys from state writes
                    for stmt in ast.walk(node):
                        # Handle regular assignments: state["key"] = value
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Subscript):
                                    if isinstance(target.value, ast.Name) and target.value.id == state_param:
                                        if isinstance(target.slice, ast.Constant):
                                            keys.append(target.slice.value)
                        # Handle augmented assignments: state["key"] -= 1
                        elif isinstance(stmt, ast.AugAssign):
                            if isinstance(stmt.target, ast.Subscript):
                                if isinstance(stmt.target.value, ast.Name) and stmt.target.value.id == state_param:
                                    if isinstance(stmt.target.slice, ast.Constant):
                                        keys.append(stmt.target.slice.value)
        return keys

    def _is_conditional_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is a conditional routing function"""
        # Check return type annotation for Literal
        if node.returns:
            return_str = ast.unparse(node.returns)
            if "Literal" in return_str:
                return True

        # Check for multiple string/constant returns or END constant
        return_statements = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
        if len(return_statements) > 1:
            def is_routing_return(r):
                if not r.value:
                    return False
                # String constant
                if isinstance(r.value, ast.Constant) and isinstance(r.value.value, str):
                    return True
                # Named constant like END
                if isinstance(r.value, ast.Name) and r.value.id in ("END", "START"):
                    return True
                return False

            if all(is_routing_return(r) for r in return_statements if r.value):
                return True

        return False

    def _extract_condition_returns(self, node: ast.FunctionDef) -> List[str]:
        """Extract all possible return values from conditional function"""
        returns = []
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                if isinstance(child.value, ast.Constant) and isinstance(child.value.value, str):
                    returns.append(child.value.value)
                elif isinstance(child.value, ast.Name):
                    # Handle named constants like END
                    if child.value.id == "END":
                        returns.append("END")
                    else:
                        returns.append(child.value.id)
        return returns

    def _parse_tool_function(self, node: ast.FunctionDef):
        """Parse a @tool decorated function"""
        docstring = ast.get_docstring(node) or ""

        # Extract description from docstring
        description = docstring.split('\n')[0] if docstring else node.name

        # Extract parameters
        params = []
        for arg in node.args.args:
            type_hint = ast.unparse(arg.annotation) if arg.annotation else "string"

            # Try to find description from docstring
            param_desc = ""
            if docstring:
                match = re.search(rf'{arg.arg}:\s*(.+?)(?:\n|$)', docstring)
                if match:
                    param_desc = match.group(1).strip()

            # Check if has default value
            required = True
            default_val = None
            # defaults are aligned to the end of args
            default_idx = len(node.args.args) - len(node.args.defaults) - 1
            arg_idx = node.args.args.index(arg)
            if arg_idx > default_idx and node.args.defaults:
                default = node.args.defaults[arg_idx - default_idx - 1]
                default_val = ast.unparse(default)
                required = False

            params.append(ParamInfo(
                name=arg.arg,
                type_hint=type_hint,
                default_value=default_val,
                description=param_desc,
                required=required
            ))

        # Get return type
        return_type = ast.unparse(node.returns) if node.returns else None

        # Extract function body (skip docstring if present)
        body_stmts = node.body
        if body_stmts and isinstance(body_stmts[0], ast.Expr) and isinstance(body_stmts[0].value, ast.Constant):
            # Skip docstring
            body_stmts = body_stmts[1:]
        function_body = '\n'.join(ast.unparse(stmt) for stmt in body_stmts)

        self.tools.append(ToolInfo(
            name=node.name,
            description=description,
            params=params,
            return_type=return_type,
            function_body=function_body
        ))

    def _parse_assignment(self, node: ast.Assign, source_code: str, is_module_level: bool = False):
        """Parse assignment to detect StateGraph creation, LLM initialization, and global variables"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                if isinstance(node.value, ast.Call):
                    func_name = self._get_call_name(node.value)
                    if func_name == "StateGraph":
                        self.graph_variable = var_name
                        continue
                    # Detect LLM initialization
                    elif func_name in self.LLM_CLASS_NAMES:
                        self.llm_variable_name = var_name
                        self.llm_config = self._extract_llm_config(node.value, func_name)
                        continue

                # Extract module-level constants (UPPER_CASE names)
                if is_module_level and var_name.isupper():
                    value_str = ast.unparse(node.value)
                    self.global_variables.append(GlobalVarInfo(
                        name=var_name,
                        value=value_str,
                        type_hint=None
                    ))

    def _extract_llm_config(self, call_node: ast.Call, class_name: str) -> LLMConfigIR:
        """Extract LLM configuration from initialization call"""
        config = LLMConfigIR()

        # Determine provider from class name
        provider_map = {
            "ChatOpenAI": "openai", "OpenAI": "openai",
            "AzureChatOpenAI": "azure", "AzureOpenAI": "azure",
            "ChatAnthropic": "anthropic", "Anthropic": "anthropic",
            "ChatGoogleGenerativeAI": "google",
            "ChatOllama": "ollama", "Ollama": "ollama",
            "ChatTongyi": "tongyi", "Tongyi": "tongyi",
            "ChatZhipuAI": "zhipu",
            "ChatBaichuan": "baichuan",
            "ChatSparkLLM": "spark",
            "ChatWenxin": "wenxin",
        }
        config.provider = provider_map.get(class_name, "openai")

        # Extract keyword arguments
        for keyword in call_node.keywords:
            key = keyword.arg
            value = keyword.value

            if key == "model" or key == "model_name":
                if isinstance(value, ast.Constant):
                    config.model = value.value
            elif key == "temperature":
                if isinstance(value, ast.Constant):
                    config.temperature = float(value.value)
            elif key == "max_tokens":
                if isinstance(value, ast.Constant):
                    config.max_tokens = int(value.value)
            elif key in ("api_base", "openai_api_base", "base_url"):
                if isinstance(value, ast.Constant):
                    config.api_base = value.value
            elif key in ("api_key", "openai_api_key"):
                if isinstance(value, ast.Constant):
                    config.api_key = value.value

        return config

    def _parse_graph_construction(self, tree: ast.Module, source_code: str):
        """Parse graph construction calls"""
        # Track processed calls to avoid duplicates
        processed_calls = set()

        for node in ast.walk(tree):
            call_node = None
            if isinstance(node, ast.Call):
                call_node = node
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call_node = node.value

            if call_node is not None:
                # Use id() to track which call nodes we've processed
                call_id = id(call_node)
                if call_id not in processed_calls:
                    processed_calls.add(call_id)
                    self._parse_graph_call(call_node)

    def _parse_graph_call(self, node: ast.Call):
        """Parse a graph method call"""
        if not isinstance(node.func, ast.Attribute):
            return

        method_name = node.func.attr

        if method_name == "add_node":
            self._parse_add_node(node)
        elif method_name == "add_edge":
            self._parse_add_edge(node)
        elif method_name == "add_conditional_edges":
            self._parse_conditional_edges(node)
        elif method_name == "set_entry_point":
            self._parse_entry_point(node)

    def _parse_add_node(self, node: ast.Call):
        """Parse add_node call to extract node_name -> function_name mapping

        LangGraph: workflow.add_node("node_name", function_name)
        - First arg: node name (string)
        - Second arg: function reference (Name node)
        """
        if len(node.args) >= 2:
            node_name = self._get_string_value(node.args[0])
            if node_name:
                self._registered_nodes.add(node_name)

                # Extract function name from second argument
                func_arg = node.args[1]
                if isinstance(func_arg, ast.Name):
                    func_name = func_arg.id
                    self._node_to_func_map[node_name] = func_name
                elif isinstance(func_arg, ast.Attribute):
                    # Handle cases like module.function
                    func_name = func_arg.attr
                    self._node_to_func_map[node_name] = func_name

    def _parse_add_edge(self, node: ast.Call):
        """Parse add_edge call"""
        if len(node.args) >= 2:
            source = self._get_string_value(node.args[0])
            target = self._get_string_value(node.args[1])

            # Handle END constant
            if isinstance(node.args[1], ast.Name) and node.args[1].id == "END":
                target = "END"

            if source and target:
                self.edges.append(EdgeInfo(source=source, target=target))

    def _parse_conditional_edges(self, node: ast.Call):
        """Parse add_conditional_edges call"""
        if len(node.args) >= 3:
            source_node = self._get_string_value(node.args[0])

            # Get condition function name
            condition_func = None
            if isinstance(node.args[1], ast.Name):
                condition_func = node.args[1].id

            # Get mapping
            mapping = {}
            if isinstance(node.args[2], ast.Dict):
                for key, value in zip(node.args[2].keys, node.args[2].values):
                    # Handle key - can be string or END constant
                    if isinstance(key, ast.Name) and key.id == "END":
                        key_str = "END"
                    else:
                        key_str = self._get_string_value(key)

                    # Handle value - can be string or END constant
                    if isinstance(value, ast.Name) and value.id == "END":
                        value_str = "END"
                    else:
                        value_str = self._get_string_value(value)

                    if key_str and value_str:
                        mapping[key_str] = value_str

            if source_node and condition_func:
                self.conditional_edges.append(ConditionalEdgeInfo(
                    source_node=source_node,
                    condition_function=condition_func,
                    mapping=mapping
                ))

    def _parse_entry_point(self, node: ast.Call):
        """Parse set_entry_point call"""
        if node.args:
            self.entry_point = self._get_string_value(node.args[0])

    def _has_decorator(self, node: ast.FunctionDef, name: str) -> bool:
        """Check if function has a specific decorator"""
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == name:
                return True
            if isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name) and dec.func.id == name:
                    return True
        return False

    def _get_name(self, node) -> str:
        """Get name from various AST node types"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ""

    def _get_call_name(self, node: ast.Call) -> str:
        """Get function name from Call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _get_string_value(self, node) -> Optional[str]:
        """Get string value from AST node"""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
            return node.s
        return None

    def _build_result(self, source_code: str) -> ParseResult:
        """Build the final parse result

        Use _node_to_func_map (from add_node() calls) to determine which functions are node functions.
        This is more accurate than heuristic-based detection.
        """
        # Build node_functions using the node_name -> func_name mapping from add_node()
        # Key is node_name (not function_name), value is function info
        filtered_node_functions = {}

        for node_name, func_name in self._node_to_func_map.items():
            if func_name in self._all_functions:
                func_info = self._all_functions[func_name]
                # Use node_name as key (important when node_name != func_name)
                filtered_node_functions[node_name] = func_info

        # Get functions used in conditional edges
        conditional_edge_funcs = {ce.condition_function for ce in self.conditional_edges}

        # Build conditional_functions from _all_functions
        filtered_conditional_functions = {}
        for func_name in conditional_edge_funcs:
            if func_name in self._all_functions:
                filtered_conditional_functions[func_name] = self._all_functions[func_name]

        return ParseResult(
            graph_name=self.graph_variable or "workflow",
            state_class_name=self.state_class_name,
            state_fields=self.state_fields,
            node_functions=filtered_node_functions,
            conditional_functions=filtered_conditional_functions,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            tools=self.tools,
            entry_point=self.entry_point,
            llm_config=self.llm_config,
            imports=self.imports,
            import_statements=self.import_statements,
            global_variables=self.global_variables,
            example_inputs=self.example_inputs,
            raw_source=source_code
        )
