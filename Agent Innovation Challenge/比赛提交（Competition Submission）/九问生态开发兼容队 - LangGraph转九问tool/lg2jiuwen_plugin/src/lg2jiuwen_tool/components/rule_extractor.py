"""
规则提取器组件

核心组件：提取 LangGraph 结构并转换代码
"""

import ast
from typing import Any, Dict, List, Optional, Set, Tuple

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.context_engine.base import Context

from ..workflow.state import (
    PendingType,
    PendingItem,
    ConvertedNode,
    ExtractionResult,
    StateField,
    EdgeInfo,
    ToolInfo,
    LLMConfig,
)
from ..rules.base import RuleChain, ConversionResult, PassthroughRule


class RuleExtractorComp(WorkflowComponent, ComponentExecutable):
    """
    规则提取器组件

    功能：
    1. 提取 LangGraph 结构（状态、节点、边、工具）
    2. 尝试用规则转换函数体
    3. 无法处理的生成 pending_item
    """

    def __init__(self):
        # 延迟导入规则，避免循环依赖
        self._rule_chain: Optional[RuleChain] = None
        self._tool_call_rule: Optional["ToolCallRule"] = None

    def _get_rule_chain(self) -> RuleChain:
        """获取规则链"""
        if self._rule_chain is None:
            from ..rules.state_rules import StateAccessRule, StateAssignRule
            from ..rules.llm_rules import LLMInvokeRule
            from ..rules.tool_rules import ToolCallRule, ToolMapCallRule
            from ..rules.edge_rules import ReturnRule

            self._tool_call_rule = ToolCallRule()
            self._rule_chain = RuleChain([
                StateAccessRule(),
                StateAssignRule(),
                LLMInvokeRule(),
                self._tool_call_rule,
                ToolMapCallRule(),  # 处理 tool_map[key].run() 模式
                ReturnRule(),
                PassthroughRule(),  # fallback: 保持原样
            ])
        return self._rule_chain

    def _update_tool_names(self, tool_names: List[str]):
        """更新工具名列表到 ToolCallRule"""
        # 确保规则链已初始化
        self._get_rule_chain()
        if self._tool_call_rule is not None:
            self._tool_call_rule.set_tool_names(tool_names)

    def _unwrap_value(self, value):
        """解包可能被 openJiuwen 包装的值"""
        if isinstance(value, dict) and "" in value and len(value) == 1:
            return value[""]
        return value

    def _decode_path(self, path: str) -> str:
        """解码文件路径"""
        return path.replace("__DOT__", ".")

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
        # 从 inputs 获取并解包
        ast_map_raw = self._unwrap_value(inputs.get("ast_map", {}))
        dependency_order_raw = self._unwrap_value(inputs.get("dependency_order", []))

        # 解码文件路径
        ast_map: Dict[str, ast.AST] = {}
        if isinstance(ast_map_raw, dict):
            for encoded_path, tree in ast_map_raw.items():
                decoded_path = self._decode_path(encoded_path)
                ast_map[decoded_path] = tree

        dependency_order: List[str] = []
        if isinstance(dependency_order_raw, list):
            dependency_order = [self._decode_path(p) for p in dependency_order_raw]

        # 验证数据类型
        if not isinstance(ast_map, dict):
            raise TypeError(f"ast_map 应为 dict，实际为 {type(ast_map)}")

        result = ExtractionResult()

        # 第一遍：收集所有文件中的信息
        global_func_to_node: Dict[str, str] = {}
        global_func_defs: Dict[str, ast.FunctionDef] = {}

        for file_path in dependency_order:
            if file_path not in ast_map:
                continue
            tree = ast_map[file_path]

            # 收集节点引用（add_node 调用）
            func_to_node = self._find_node_references(tree)
            global_func_to_node.update(func_to_node)

            # 收集所有函数定义（用于跨文件查找路由函数）
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    global_func_defs[node.name] = node

        # 第二遍：按依赖顺序处理文件
        for file_path in dependency_order:
            if file_path not in ast_map:
                continue
            tree = ast_map[file_path]

            # 1. 提取导入语句和全局变量
            self._extract_imports_and_globals(tree, result)

            # 2. 提取状态类
            self._extract_states(tree, result)

            # 3. 提取 LLM 配置
            self._extract_llm_configs(tree, result)

            # 4. 提取工具（使用全局函数定义映射支持跨文件查找）
            self._extract_tools(tree, result, file_path, global_func_defs)

            # 4.5 更新工具名列表到规则链
            tool_names = [t.name for t in result.tools]
            self._update_tool_names(tool_names)

            # 5. 提取并转换节点（使用全局的 func_to_node 映射）
            self._extract_and_convert_nodes_with_mapping(tree, result, file_path, global_func_to_node)

            # 6. 提取边（使用全局的函数定义映射）
            self._extract_edges_with_global_funcs(tree, result, global_func_defs)

            # 7. 提取初始输入（从 invoke() 调用）
            self._extract_initial_inputs(tree, result)

        # 8. 分类全局变量：识别工具相关的变量
        self._classify_global_vars(result)

        return {"extraction_result": result}

    def _classify_global_vars(self, result: ExtractionResult):
        """分类全局变量，将工具相关的变量移到 tool_related_vars"""
        # 收集工具函数名
        tool_func_names = set()
        for tool in result.tools:
            tool_func_names.add(tool.name)
            tool_func_names.add(tool.name.lower())
            if tool.func_name:
                tool_func_names.add(tool.func_name)

        # 需要排除的模式（LangGraph 特有，不需要迁移）
        exclude_patterns = [
            '.compile()',
            'graph.compile',
            'StateGraph(',
        ]

        new_global_vars = []
        for var in result.global_vars:
            # 检查是否需要完全排除
            should_exclude = any(pattern in var for pattern in exclude_patterns)
            if should_exclude:
                continue

            # 检查是否引用了工具函数
            is_tool_related = False
            for func_name in tool_func_names:
                # 检查变量值中是否引用了工具函数名
                if f': {func_name}' in var or f':{func_name}' in var or f'= {func_name}' in var:
                    is_tool_related = True
                    break

            if is_tool_related:
                result.tool_related_vars.append(var)
                # 提取工具映射变量名（从 "var_name = {...}" 中提取 var_name）
                if result.tool_map_var_name is None:
                    var_name = self._extract_var_name(var)
                    if var_name:
                        result.tool_map_var_name = var_name
            else:
                new_global_vars.append(var)

        result.global_vars = new_global_vars

    def _extract_var_name(self, var_def: str) -> Optional[str]:
        """从变量定义中提取变量名（如从 'tool_map = {...}' 中提取 'tool_map'）"""
        if '=' in var_def:
            # 取等号左边部分并去除空格
            left_part = var_def.split('=')[0].strip()
            # 验证是有效的变量名（不包含特殊字符）
            if left_part.isidentifier():
                return left_part
        return None

    def _extract_initial_inputs(self, tree: ast.AST, result: ExtractionResult):
        """
        提取初始输入（从 app.invoke() 或 graph.invoke() 调用）

        分析源代码中的 invoke 调用，提取初始输入字段
        例如: app.invoke({"input": input_text, "is_end": False, "loop_count": 0})

        同时提取 main 函数中的示例输入值
        例如: input_text = "100加200等于多少？"
        """
        # 收集 main 块中的变量赋值（示例值）
        main_vars = {}
        self._extract_main_example_vars(tree, main_vars)

        for node in ast.walk(tree):
            # 查找 invoke 调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "invoke":
                    # 检查调用对象是否为已知的图变量
                    if isinstance(node.func.value, ast.Name):
                        var_name = node.func.value.id
                        if var_name in ("app", "graph", "workflow", "agent"):
                            # 提取第一个参数（应该是输入字典）
                            if node.args:
                                self._extract_dict_inputs(node.args[0], result, main_vars)

    def _extract_main_example_vars(self, tree: ast.AST, main_vars: Dict[str, Any]):
        """提取 main 块中的变量赋值作为示例值"""
        for node in ast.walk(tree):
            # 查找 if __name__ == "__main__": 块
            if isinstance(node, ast.If):
                # 检查是否为 if __name__ == "__main__":
                if self._is_main_check(node.test):
                    # 遍历 main 块中的语句
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            # 简单赋值: input_text = "..."
                            for target in stmt.targets:
                                if isinstance(target, ast.Name):
                                    value = self._extract_value(stmt.value)
                                    main_vars[target.id] = value

    def _is_main_check(self, test: ast.AST) -> bool:
        """检查是否为 if __name__ == "__main__" 条件"""
        if isinstance(test, ast.Compare):
            if isinstance(test.left, ast.Name) and test.left.id == "__name__":
                if test.ops and isinstance(test.ops[0], ast.Eq):
                    if test.comparators and isinstance(test.comparators[0], ast.Constant):
                        return test.comparators[0].value == "__main__"
        return False

    def _extract_dict_inputs(
        self, node: ast.AST, result: ExtractionResult, main_vars: Optional[Dict[str, Any]] = None
    ):
        """从字典节点提取输入字段"""
        main_vars = main_vars or {}

        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key is not None and isinstance(key, ast.Constant):
                    field_name = key.value
                    # 提取值（尝试解析为 Python 值）
                    field_value = self._extract_value(value)
                    result.initial_inputs[field_name] = field_value

                    # 如果是变量引用，尝试从 main_vars 获取示例值
                    if isinstance(field_value, str) and field_value.startswith("${"):
                        var_name = field_value[2:-1]  # 去掉 ${ 和 }
                        if var_name in main_vars:
                            result.example_inputs[field_name] = main_vars[var_name]
                    elif not isinstance(field_value, str) or not field_value.startswith("${"):
                        # 直接值也作为示例
                        result.example_inputs[field_name] = field_value

    def _extract_value(self, node: ast.AST) -> Any:
        """从 AST 节点提取值"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            # 变量引用，返回占位符
            if node.id in ("True", "False"):
                return node.id == "True"
            elif node.id == "None":
                return None
            else:
                return f"${{{node.id}}}"  # 变量占位符
        elif isinstance(node, ast.List):
            return [self._extract_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._extract_value(k): self._extract_value(v)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }
        else:
            # 回退：返回代码字符串
            return ast.unparse(node)

    def _extract_imports_and_globals(self, tree: ast.AST, result: ExtractionResult):
        """提取导入语句和全局变量"""
        for node in ast.iter_child_nodes(tree):
            # 提取 import 语句
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                result.imports.append(ast.unparse(node))

            # 提取全局变量赋值（排除函数、类定义等）
            elif isinstance(node, ast.Assign):
                # 检查是否是简单的全局变量赋值（如 API_KEY = "xxx"）
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # 排除已知的特殊变量
                        if target.id not in ("workflow", "graph", "llm", "model", "__all__"):
                            result.global_vars.append(ast.unparse(node))

    def _extract_states(self, tree: ast.AST, result: ExtractionResult):
        """提取状态类定义"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否为 TypedDict 子类
                if self._is_typed_dict(node):
                    result.state_class_name = node.name
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            field = StateField(
                                name=item.target.id,
                                type_hint=ast.unparse(item.annotation) if item.annotation else "Any",
                                default=ast.unparse(item.value) if item.value else None
                            )
                            result.states.append(field)

    def _is_typed_dict(self, node: ast.ClassDef) -> bool:
        """检查是否为 TypedDict"""
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "TypedDict":
                return True
            if isinstance(base, ast.Attribute) and base.attr == "TypedDict":
                return True
        return False

    def _extract_llm_configs(self, tree: ast.AST, result: ExtractionResult):
        """提取 LLM 配置"""
        # 先收集全局变量赋值（用于解析变量引用）
        global_vars_map = self._collect_global_vars_map(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if self._is_llm_creation(node.value):
                    config = self._parse_llm_config(node, global_vars_map)
                    if config:
                        result.llm_configs.append(config)

    def _collect_global_vars_map(self, tree: ast.AST) -> Dict[str, Any]:
        """收集全局变量映射 {变量名: 值}"""
        global_vars_map = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # 只处理简单的常量赋值
                        if isinstance(node.value, ast.Constant):
                            global_vars_map[target.id] = node.value.value
        return global_vars_map

    def _is_llm_creation(self, node: ast.AST) -> bool:
        """检查是否为 LLM 创建"""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                return func.id in ("ChatOpenAI", "ChatAnthropic", "ChatModel")
            if isinstance(func, ast.Attribute):
                return func.attr in ("ChatOpenAI", "ChatAnthropic", "ChatModel")
        return False

    def _parse_llm_config(
        self, node: ast.Assign, global_vars_map: Optional[Dict[str, Any]] = None
    ) -> Optional[LLMConfig]:
        """解析 LLM 配置

        Args:
            node: LLM 赋值语句节点
            global_vars_map: 全局变量映射 {变量名: 值}，用于解析变量引用
        """
        if not node.targets:
            return None

        global_vars_map = global_vars_map or {}

        var_name = ""
        if isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id

        call = node.value
        if not isinstance(call, ast.Call):
            return None

        model_class = ""
        if isinstance(call.func, ast.Name):
            model_class = call.func.id
        elif isinstance(call.func, ast.Attribute):
            model_class = call.func.attr

        config = LLMConfig(var_name=var_name, model_class=model_class)

        # 解析参数（支持字符串字面量和变量引用）
        for keyword in call.keywords:
            value = self._resolve_value(keyword.value, global_vars_map)

            if keyword.arg == "model" or keyword.arg == "model_name":
                if value is not None:
                    config.model_name = value
            elif keyword.arg == "temperature":
                if value is not None:
                    config.temperature = value
            elif keyword.arg in ("openai_api_key", "api_key"):
                if value is not None:
                    config.other_params["api_key"] = value
            elif keyword.arg in ("openai_api_base", "api_base", "base_url"):
                if value is not None:
                    config.other_params["api_base"] = value

        return config

    def _resolve_value(
        self, node: ast.AST, global_vars_map: Dict[str, Any]
    ) -> Optional[Any]:
        """解析 AST 节点的值（支持字面量和变量引用）"""
        # 字符串/数字字面量
        if isinstance(node, ast.Constant):
            return node.value
        # 变量引用 -> 从全局变量映射中查找
        if isinstance(node, ast.Name):
            return global_vars_map.get(node.id)
        return None

    def _extract_tools(
        self,
        tree: ast.AST,
        result: ExtractionResult,
        file_path: str,
        global_func_defs: Optional[Dict[str, ast.FunctionDef]] = None
    ):
        """提取工具函数"""
        # 使用全局函数定义（支持跨文件查找）或仅当前文件
        func_defs = global_func_defs or {}

        # 当前文件的函数定义也加入（确保本地优先）
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_defs[node.name] = node

        for node in ast.walk(tree):
            # 方式1: @tool 装饰器
            if isinstance(node, ast.FunctionDef):
                if self._has_tool_decorator(node):
                    tool_info = ToolInfo(
                        name=node.name,
                        original_code=ast.unparse(node),
                        description=ast.get_docstring(node) or "",
                        parameters=self._extract_parameters(node)
                    )
                    result.tools.append(tool_info)

            # 方式2: Tool(name=..., func=..., description=...) 类实例化
            # 匹配: tools = [Tool(...), Tool(...)] 或 tool = Tool(...)
            elif isinstance(node, ast.Assign):
                self._extract_tools_from_assignment(node, func_defs, result)

    def _extract_tools_from_assignment(
        self,
        node: ast.Assign,
        func_defs: Dict[str, ast.FunctionDef],
        result: ExtractionResult
    ):
        """从赋值语句中提取 Tool() 类创建的工具"""
        # 处理列表: tools = [Tool(...), Tool(...)]
        if isinstance(node.value, ast.List):
            for elt in node.value.elts:
                self._extract_tool_from_call(elt, func_defs, result)
        # 处理单个: tool = Tool(...)
        elif isinstance(node.value, ast.Call):
            self._extract_tool_from_call(node.value, func_defs, result)

    def _extract_tool_from_call(
        self,
        node: ast.AST,
        func_defs: Dict[str, ast.FunctionDef],
        result: ExtractionResult
    ):
        """从 Tool() 调用中提取工具信息"""
        if not isinstance(node, ast.Call):
            return

        # 检查是否是 Tool(...) 调用
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name != "Tool":
            return

        # 提取参数
        tool_name = None
        tool_func = None
        tool_desc = ""

        for keyword in node.keywords:
            if keyword.arg == "name":
                if isinstance(keyword.value, ast.Constant):
                    tool_name = keyword.value.value
            elif keyword.arg == "func":
                if isinstance(keyword.value, ast.Name):
                    tool_func = keyword.value.id
            elif keyword.arg == "description":
                if isinstance(keyword.value, ast.Constant):
                    tool_desc = keyword.value.value

        if tool_name and tool_func:
            # 查找对应的函数定义
            func_def = func_defs.get(tool_func)
            original_code = ast.unparse(func_def) if func_def else ""
            parameters = self._extract_parameters(func_def) if func_def else []

            # 避免重复添加
            existing_names = {t.name for t in result.tools}
            if tool_name not in existing_names:
                tool_info = ToolInfo(
                    name=tool_name,
                    func_name=tool_func,
                    original_code=original_code,
                    description=tool_desc,
                    parameters=parameters
                )
                result.tools.append(tool_info)

    def _has_tool_decorator(self, node: ast.FunctionDef) -> bool:
        """检查是否有 @tool 装饰器"""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "tool":
                return True
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == "tool":
                    return True
        return False

    def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """提取函数参数"""
        params = []
        for arg in node.args.args:
            param = {
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation else "Any"
            }
            params.append(param)
        return params

    def _extract_and_convert_nodes(
        self,
        tree: ast.AST,
        result: ExtractionResult,
        file_path: str
    ):
        """提取并转换节点函数（单文件模式）"""
        func_to_node = self._find_node_references(tree)
        self._extract_and_convert_nodes_with_mapping(tree, result, file_path, func_to_node)

    def _extract_and_convert_nodes_with_mapping(
        self,
        tree: ast.AST,
        result: ExtractionResult,
        file_path: str,
        func_to_node: Dict[str, str]
    ):
        """提取并转换节点函数（使用外部提供的映射）"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in func_to_node:
                    # 使用 add_node 中定义的节点名，而不是函数名
                    actual_node_name = func_to_node[node.name]

                    # 尝试规则转换
                    conversion = self._try_convert_body(node, result)

                    if conversion.success:
                        # 规则转换成功
                        result.nodes.append(ConvertedNode(
                            name=actual_node_name,  # 使用节点名
                            original_code=ast.unparse(node),
                            converted_body=conversion.code,
                            inputs=conversion.inputs,
                            outputs=conversion.outputs,
                            conversion_source="rule",
                            docstring=ast.get_docstring(node)
                        ))
                        result.rule_count += 1
                    else:
                        # 生成 pending_item
                        result.pending_items.append(PendingItem(
                            id=f"{file_path}:{actual_node_name}",
                            pending_type=PendingType.NODE_BODY,
                            source_code=ast.unparse(node),
                            context={
                                "state_fields": [s.name for s in result.states],
                                "available_tools": [t.name for t in result.tools],
                                "failed_lines": conversion.failed_lines
                            },
                            question=self._build_question(node, conversion.failed_lines),
                            location=f"{file_path}:{node.lineno}"
                        ))

    def _find_node_references(self, tree: ast.AST) -> Dict[str, str]:
        """找出所有被 add_node 引用的函数名，返回 {函数名: 节点名} 映射"""
        func_to_node: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # 匹配 workflow.add_node("name", func) 或 graph.add_node("name", func)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "add_node":
                    if len(node.args) >= 2:
                        # 第一个参数是节点名
                        node_name_arg = node.args[0]
                        node_name = None
                        if isinstance(node_name_arg, ast.Constant) and isinstance(node_name_arg.value, str):
                            node_name = node_name_arg.value
                        elif isinstance(node_name_arg, ast.Str):
                            node_name = node_name_arg.s

                        # 第二个参数是函数
                        func_arg = node.args[1]
                        if isinstance(func_arg, ast.Name) and node_name:
                            func_to_node[func_arg.id] = node_name
        return func_to_node

    def _try_convert_body(
        self,
        func: ast.FunctionDef,
        result: ExtractionResult
    ) -> ConversionResult:
        """尝试用规则转换函数体"""
        rule_chain = self._get_rule_chain()
        return rule_chain.convert_statements(func.body)

    def _build_question(
        self,
        func: ast.FunctionDef,
        failed_lines: List[Dict[str, Any]]
    ) -> str:
        """为 AI 构建具体问题"""
        lines_desc = "\n".join([
            f"- 第{l['line']}行: `{l['code']}`"
            for l in failed_lines
        ])

        return f"""
函数 `{func.name}` 中以下代码无法用规则转换，请转换为 openJiuwen 格式：

{lines_desc}

转换要求：
1. 状态读取: state["x"] → inputs["x"]
2. 状态写入: 收集到返回字典
3. LLM调用: llm.invoke(msgs) → await self._llm.ainvoke(model_name=self.model_name, messages=msgs)
4. 保持原有逻辑不变

请只输出转换后的代码行。
"""

    def _extract_edges(self, tree: ast.AST, result: ExtractionResult):
        """提取边定义（单文件模式）"""
        # 先收集所有函数定义，用于查找路由函数
        func_defs: Dict[str, ast.FunctionDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_defs[node.name] = node

        self._extract_edges_with_global_funcs(tree, result, func_defs)

    def _extract_edges_with_global_funcs(
        self,
        tree: ast.AST,
        result: ExtractionResult,
        global_func_defs: Dict[str, ast.FunctionDef]
    ):
        """提取边定义（使用全局函数定义映射）"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "add_edge":
                    edge = self._parse_edge(node)
                    if edge:
                        result.edges.append(edge)
                elif node.func.attr == "add_conditional_edges":
                    edge = self._parse_conditional_edge(node, global_func_defs)
                    if edge:
                        result.edges.append(edge)
                elif node.func.attr == "set_entry_point":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        result.entry_point = node.args[0].value

    def _parse_edge(self, node: ast.Call) -> Optional[EdgeInfo]:
        """解析普通边"""
        if len(node.args) < 2:
            return None

        source = self._get_string_value(node.args[0]) or self._get_name_value(node.args[0])
        target = self._get_string_value(node.args[1]) or self._get_name_value(node.args[1])

        # 处理 END -> "end"
        if source == "END":
            source = "end"
        if target == "END":
            target = "end"

        if source and target:
            return EdgeInfo(source=source, target=target)
        return None

    def _parse_conditional_edge(
        self,
        node: ast.Call,
        func_defs: Dict[str, ast.FunctionDef]
    ) -> Optional[EdgeInfo]:
        """解析条件边"""
        if len(node.args) < 2:
            return None

        source = self._get_string_value(node.args[0])
        if not source:
            return None

        # 第二个参数是路由函数
        condition_func = None
        condition_func_code = None
        if isinstance(node.args[1], ast.Name):
            condition_func = node.args[1].id
            # 查找函数定义并提取代码
            if condition_func in func_defs:
                condition_func_code = ast.unparse(func_defs[condition_func])

        # 第三个参数是条件映射
        condition_map = {}
        if len(node.args) >= 3 and isinstance(node.args[2], ast.Dict):
            for key, value in zip(node.args[2].keys, node.args[2].values):
                k = self._get_string_value(key) or self._get_name_value(key)
                v = self._get_string_value(value) or self._get_name_value(value)
                if k and v:
                    # 处理 END
                    if k == "END":
                        k = "end"
                    if v == "END":
                        v = "end"
                    condition_map[k] = v

        return EdgeInfo(
            source=source,
            target="",  # 条件边没有单一目标
            is_conditional=True,
            condition_func=condition_func,
            condition_func_code=condition_func_code,
            condition_map=condition_map
        )

    def _get_string_value(self, node: ast.AST) -> Optional[str]:
        """获取字符串值"""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Str):
            return node.s
        return None

    def _get_name_value(self, node: ast.AST) -> Optional[str]:
        """获取名称值"""
        if isinstance(node, ast.Name):
            return node.id
        return None
