"""
IR 构建器组件

将 ExtractionResult 转换为 IR
"""

import re
from typing import Any, Dict, List, Optional

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.context_engine.base import Context

from ..workflow.state import ExtractionResult, ConvertedNode, EdgeInfo
from ..ir.models import (
    WorkflowNodeIR,
    WorkflowEdgeIR,
    ToolIR,
    AgentIR,
    WorkflowIR,
    LLMConfigIR,
    MigrationIR,
)


class IRBuilderComp(WorkflowComponent, ComponentExecutable):
    """
    IR 构建器组件

    功能：
    - 将 ExtractionResult 转换为标准 IR
    - 此时所有代码已转换完成，只需组装结构
    - 不需要 AI
    """

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
        # 从 inputs 获取（通过 transformer 传入）
        extraction_result: ExtractionResult = inputs.get("extraction_result")

        if extraction_result is None:
            raise ValueError("无法获取 extraction_result")

        # 构建节点 IR
        nodes_ir = self._build_nodes_ir(extraction_result)

        # 构建边 IR
        edges_ir = self._build_edges_ir(extraction_result)

        # 构建工具 IR
        tools_ir = self._build_tools_ir(extraction_result)

        # 构建 LLM 配置 IR
        llm_config_ir = self._build_llm_config_ir(extraction_result)

        # 构建 Agent IR
        agent_ir = AgentIR(
            name=self._infer_agent_name(extraction_result),
            llm_config=llm_config_ir,
            tools=tools_ir,
            state_fields=[
                {"name": s.name, "type": s.type_hint, "default": s.default}
                for s in extraction_result.states
            ],
            global_vars=extraction_result.global_vars,
            tool_related_vars=extraction_result.tool_related_vars,
            tool_map_var_name=extraction_result.tool_map_var_name,
            imports=extraction_result.imports,
            initial_inputs=extraction_result.initial_inputs,
            example_inputs=extraction_result.example_inputs
        )

        # 构建 Workflow IR
        workflow_ir = WorkflowIR(
            nodes=nodes_ir,
            edges=edges_ir,
            entry_node=extraction_result.entry_point,
            state_class_name=extraction_result.state_class_name
        )

        # 构建完整的 Migration IR
        migration_ir = MigrationIR(
            agent_ir=agent_ir,
            workflow_ir=workflow_ir,
            conversion_stats={
                "rule_count": extraction_result.rule_count,
                "ai_count": extraction_result.ai_count,
                "total_nodes": len(extraction_result.nodes),
                "total_edges": len(extraction_result.edges),
                "total_tools": len(extraction_result.tools)
            }
        )

        return {
            "agent_ir": agent_ir,
            "workflow_ir": workflow_ir,
            "migration_ir": migration_ir,
            "extraction_result": extraction_result
        }

    def _build_nodes_ir(self, result: ExtractionResult) -> List[WorkflowNodeIR]:
        """构建节点 IR"""
        nodes_ir = []
        for node in result.nodes:
            nodes_ir.append(WorkflowNodeIR(
                name=node.name,
                class_name=self._to_class_name(node.name),
                converted_body=node.converted_body,
                inputs=node.inputs,
                outputs=node.outputs,
                conversion_source=node.conversion_source,
                docstring=node.docstring,
                has_llm=self._has_llm_call(node.converted_body),
                has_tools=self._has_tool_call(node.converted_body, result)
            ))
        return nodes_ir

    def _build_edges_ir(self, result: ExtractionResult) -> List[WorkflowEdgeIR]:
        """构建边 IR"""
        edges_ir = []
        for edge in result.edges:
            router_name = None
            condition_func_code = None

            if edge.is_conditional and edge.condition_func:
                # 使用统一的命名：{source}_router
                router_name = f"{edge.source}_router"
                # 获取上游组件的输出字段列表
                source_outputs = self._get_node_outputs(edge.source, result)
                # 转换条件函数，使用相同的命名
                condition_func_code = self._convert_condition_func(
                    router_name,  # 直接使用 router_name 作为函数名
                    edge.source,
                    result,
                    edge.condition_func_code,  # 传入原始代码
                    source_outputs  # 传入上游组件的输出字段
                )

            edges_ir.append(WorkflowEdgeIR(
                source=edge.source,
                target=edge.target,
                is_conditional=edge.is_conditional,
                condition_func=condition_func_code,
                condition_map=edge.condition_map,
                router_name=router_name
            ))
        return edges_ir

    def _build_tools_ir(self, result: ExtractionResult) -> List[ToolIR]:
        """构建工具 IR"""
        tools_ir = []
        for tool in result.tools:
            # 转换工具代码
            converted_body = tool.converted_code or self._convert_tool_body(tool.original_code)

            tools_ir.append(ToolIR(
                name=tool.name,
                func_name=tool.name,
                description=tool.description or f"{tool.name} 工具",
                parameters=tool.parameters,
                converted_body=converted_body
            ))
        return tools_ir

    def _build_llm_config_ir(self, result: ExtractionResult) -> Optional[LLMConfigIR]:
        """构建 LLM 配置 IR"""
        if not result.llm_configs:
            return None

        config = result.llm_configs[0]
        return LLMConfigIR(
            model_name=config.model_name or "gpt-4",
            temperature=config.temperature or 0.7,
            other_params=config.other_params
        )

    def _to_class_name(self, name: str) -> str:
        """转换为类名 (PascalCase + Comp)"""
        # snake_case -> PascalCase
        parts = name.split("_")
        pascal = "".join(p.capitalize() for p in parts)
        return f"{pascal}Comp"

    def _infer_agent_name(self, result: ExtractionResult) -> str:
        """推断 Agent 名称"""
        if result.graph_name:
            return result.graph_name
        if result.state_class_name:
            # AgentState -> Agent
            name = result.state_class_name.replace("State", "")
            return name or "Agent"
        return "MigratedAgent"

    def _has_llm_call(self, code: str) -> bool:
        """检查代码中是否有 LLM 调用"""
        return "self._llm" in code or "ainvoke" in code

    def _has_tool_call(self, code: str, result: ExtractionResult) -> bool:
        """检查代码中是否有工具调用"""
        tool_names = [t.name for t in result.tools]
        for name in tool_names:
            if name in code:
                return True
        return False

    def _get_node_outputs(self, node_name: str, result: ExtractionResult) -> List[str]:
        """获取节点的输出字段列表"""
        for node in result.nodes:
            if node.name == node_name:
                return node.outputs
        return []

    def _convert_condition_func(
        self,
        func_name: str,
        source_node: str,
        result: ExtractionResult,
        original_code: Optional[str] = None,
        source_outputs: Optional[List[str]] = None
    ) -> str:
        """转换条件函数为 openJiuwen 格式"""
        source_outputs = source_outputs or []

        if original_code:
            # 尝试解析并转换原始代码
            try:
                converted = self._try_convert_router_func(
                    func_name, source_node, original_code, source_outputs
                )
                if converted:
                    return converted
            except Exception:
                pass

        # 回退：生成 TODO 模板
        return f'''def {func_name}(runtime: WorkflowRuntime) -> str:
    """路由函数：根据 {source_node} 的输出决定下一个节点"""
    # TODO: 实现路由逻辑
    # 使用 runtime.get_global_state("{source_node}.field_name") 获取上游组件输出
    # 使用 runtime.get_global_state("field_name") 获取全局状态
    return "next_node"'''

    def _try_convert_router_func(
        self,
        func_name: str,
        source_node: str,
        original_code: str,
        source_outputs: Optional[List[str]] = None
    ) -> Optional[str]:
        """尝试转换路由函数"""
        import ast as ast_module

        source_outputs = source_outputs or []
        tree = ast_module.parse(original_code)
        func_def = tree.body[0]

        if not isinstance(func_def, ast_module.FunctionDef):
            return None

        # 获取函数体，跳过 docstring
        body_stmts = []
        for stmt in func_def.body:
            # 跳过 docstring
            if isinstance(stmt, ast_module.Expr) and isinstance(stmt.value, ast_module.Constant):
                if isinstance(stmt.value.value, str):
                    continue
            body_stmts.append(stmt)

        if not body_stmts:
            return None

        # 单语句函数（简单 return）
        if len(body_stmts) == 1:
            stmt = body_stmts[0]
            if isinstance(stmt, ast_module.Return) and stmt.value:
                converted_return = self._convert_router_return(
                    stmt.value, source_node, source_outputs
                )
                if converted_return:
                    return f'''def {func_name}(runtime: WorkflowRuntime) -> str:
    """路由函数：根据 {source_node} 的输出决定下一个节点"""
    {converted_return}'''

        # 多语句函数（如 if-elif-return 结构）
        converted_lines = []
        for stmt in body_stmts:
            converted_stmt = self._convert_router_statement(stmt, source_node, source_outputs)
            if converted_stmt is None:
                return None  # 无法转换，回退到 TODO
            converted_lines.append(converted_stmt)

        body_code = "\n    ".join(converted_lines)
        return f'''def {func_name}(runtime: WorkflowRuntime) -> str:
    """路由函数：根据 {source_node} 的输出决定下一个节点"""
    {body_code}'''

    def _convert_router_statement(
        self, stmt: Any, source_node: str, source_outputs: Optional[List[str]] = None
    ) -> Optional[str]:
        """转换路由函数中的单个语句"""
        import ast as ast_module

        source_outputs = source_outputs or []

        # return 语句
        if isinstance(stmt, ast_module.Return):
            if stmt.value:
                value = self._convert_router_value(stmt.value)
                return f"return {value}"
            return "return"

        # if 语句
        if isinstance(stmt, ast_module.If):
            test = self._convert_router_condition(stmt.test, source_node, source_outputs)
            body_lines = []
            for s in stmt.body:
                converted = self._convert_router_statement(s, source_node, source_outputs)
                if converted is None:
                    return None
                body_lines.append(converted)

            result = f"if {test}:\n        " + "\n        ".join(body_lines)

            # 处理 elif/else
            if stmt.orelse:
                if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast_module.If):
                    # elif
                    elif_stmt = self._convert_router_statement(stmt.orelse[0], source_node, source_outputs)
                    if elif_stmt:
                        result += "\n    el" + elif_stmt
                else:
                    # else
                    else_lines = []
                    for s in stmt.orelse:
                        converted = self._convert_router_statement(s, source_node, source_outputs)
                        if converted is None:
                            return None
                        else_lines.append(converted)
                    result += "\n    else:\n        " + "\n        ".join(else_lines)

            return result

        # 表达式语句（如 print）
        if isinstance(stmt, ast_module.Expr):
            return ast_module.unparse(stmt)

        # 其他语句，尝试直接转换
        try:
            return ast_module.unparse(stmt)
        except Exception:
            return None

    def _convert_router_return(
        self,
        node: Any,  # ast.AST
        source_node: str,
        source_outputs: Optional[List[str]] = None
    ) -> Optional[str]:
        """转换路由函数的返回值"""
        import ast as ast_module

        source_outputs = source_outputs or []

        # 条件表达式: END if state.get("error") else "call_weather"
        if isinstance(node, ast_module.IfExp):
            # 转换条件
            test_code = self._convert_router_condition(node.test, source_node, source_outputs)
            # 转换 body (if 分支)
            body_code = self._convert_router_value(node.body)
            # 转换 orelse (else 分支)
            else_code = self._convert_router_value(node.orelse)

            return f'return {body_code} if {test_code} else {else_code}'

        # 简单值
        value_code = self._convert_router_value(node)
        return f'return {value_code}'

    def _convert_router_condition(
        self, node: Any, source_node: str, source_outputs: Optional[List[str]] = None
    ) -> str:
        """转换路由条件

        规则：
        - 如果 key 在上游组件的输出中 → runtime.get_global_state("{source_node}.{key}")
        - 如果 key 不在上游组件的输出中 → runtime.get_global_state("{key}") 访问全局状态
        """
        import ast as ast_module

        source_outputs = source_outputs or []

        # state.get("key") 或 state.get("key", default)
        if isinstance(node, ast_module.Call):
            if isinstance(node.func, ast_module.Attribute) and node.func.attr == "get":
                if isinstance(node.func.value, ast_module.Name):
                    if node.func.value.id in ("state", "State"):
                        if node.args:
                            key = self._extract_string_value(node.args[0])
                            if key:
                                # 判断 key 是否是上游组件的输出
                                if key in source_outputs:
                                    # 上游组件的输出 → 带节点前缀
                                    get_state = f'runtime.get_global_state("{source_node}.{key}")'
                                else:
                                    # 全局状态 → 不带前缀
                                    get_state = f'runtime.get_global_state("{key}")'
                                # 处理默认值
                                if len(node.args) >= 2:
                                    default_val = ast_module.unparse(node.args[1])
                                    return f'({get_state} or {default_val})'
                                return get_state

        # state["key"]
        if isinstance(node, ast_module.Subscript):
            if isinstance(node.value, ast_module.Name):
                if node.value.id in ("state", "State"):
                    key = self._extract_string_value(node.slice)
                    if key:
                        # 判断 key 是否是上游组件的输出
                        if key in source_outputs:
                            return f'runtime.get_global_state("{source_node}.{key}")'
                        else:
                            return f'runtime.get_global_state("{key}")'

        # 比较表达式: 递归转换左右两边
        if isinstance(node, ast_module.Compare):
            left = self._convert_router_condition(node.left, source_node, source_outputs)
            comparators = [self._convert_router_condition(c, source_node, source_outputs) for c in node.comparators]
            ops = [self._get_compare_op(op) for op in node.ops]
            result = left
            for op, comp in zip(ops, comparators):
                result += f' {op} {comp}'
            return result

        return ast_module.unparse(node)

    def _get_compare_op(self, op: Any) -> str:
        """获取比较运算符字符串"""
        import ast as ast_module
        op_map = {
            ast_module.Eq: '==',
            ast_module.NotEq: '!=',
            ast_module.Lt: '<',
            ast_module.LtE: '<=',
            ast_module.Gt: '>',
            ast_module.GtE: '>=',
            ast_module.Is: 'is',
            ast_module.IsNot: 'is not',
            ast_module.In: 'in',
            ast_module.NotIn: 'not in',
        }
        return op_map.get(type(op), '==')

    def _convert_router_value(self, node: Any) -> str:
        """转换路由返回值"""
        import ast as ast_module

        # END -> "end"
        if isinstance(node, ast_module.Name) and node.id == "END":
            return '"end"'

        # 字符串保持不变
        if isinstance(node, ast_module.Constant) and isinstance(node.value, str):
            return f'"{node.value}"'

        return ast_module.unparse(node)

    def _extract_string_value(self, node: Any) -> Optional[str]:
        """从 AST 节点提取字符串值"""
        import ast as ast_module

        if isinstance(node, ast_module.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast_module.Str):
            return node.s
        return None

    def _convert_tool_body(self, original_code: str) -> str:
        """转换工具函数体（只提取函数体内容，不包含def行）"""
        import ast as ast_module

        try:
            # 解析原始代码
            tree = ast_module.parse(original_code)

            # 找到函数定义
            for node in ast_module.walk(tree):
                if isinstance(node, ast_module.FunctionDef):
                    # 提取函数体
                    body_lines = []
                    for stmt in node.body:
                        # 跳过 docstring
                        if isinstance(stmt, ast_module.Expr) and isinstance(stmt.value, ast_module.Constant):
                            if isinstance(stmt.value.value, str):
                                continue
                        body_lines.append(ast_module.unparse(stmt))

                    return "\n".join(body_lines)
        except Exception:
            pass

        # 回退：简单处理
        lines = original_code.split("\n")
        result_lines = []
        in_body = False
        base_indent = None

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("@"):
                continue
            if stripped.startswith("def "):
                in_body = True
                continue  # 跳过 def 行
            if in_body:
                # 跳过 docstring
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                if line.strip():  # 非空行
                    if base_indent is None:
                        base_indent = len(line) - len(line.lstrip())
                    # 移除基础缩进
                    if len(line) >= base_indent:
                        result_lines.append(line[base_indent:])
                    else:
                        result_lines.append(line.strip())
                else:
                    result_lines.append("")

        return "\n".join(result_lines)
