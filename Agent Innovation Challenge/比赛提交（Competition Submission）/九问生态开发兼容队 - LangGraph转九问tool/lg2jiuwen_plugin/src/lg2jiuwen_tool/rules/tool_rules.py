"""
工具调用规则

处理 tool.invoke() 和 tool() 的转换
"""

import ast
from typing import List, Optional, Set

from .base import BaseRule, StatementRule, ConversionResult


class ToolCallRule(StatementRule):
    """
    工具调用规则

    转换为 openJiuwen 格式:
    - result = tool.invoke({"arg": value}) → result = tool.invoke(inputs={"arg": value})
    - result = tool(arg=value) → result = tool.invoke(inputs={"arg": value})

    openJiuwen 中工具需要通过 .invoke(inputs={...}) 方法调用
    """

    def __init__(self, tool_names: Optional[Set[str]] = None):
        """
        Args:
            tool_names: 已知的工具函数名集合，用于识别直接工具调用
        """
        self.tool_names: Set[str] = tool_names or set()

    def set_tool_names(self, tool_names: List[str]):
        """设置工具名列表"""
        self.tool_names = set(tool_names)

    def matches(self, node: ast.AST) -> bool:
        """判断是否为工具调用"""
        if isinstance(node, ast.Assign):
            return self._is_tool_invoke(node.value) or self._is_direct_tool_call(node.value)
        if isinstance(node, ast.Expr):
            return self._is_tool_invoke(node.value) or self._is_direct_tool_call(node.value)
        return False

    def _is_tool_invoke(self, node: ast.AST) -> bool:
        """检查是否为 tool.invoke() 调用（只匹配简单变量，不匹配下标访问）"""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "invoke":
                    # 只匹配简单变量调用 (如 tool.invoke())
                    # 不匹配 tool_map[key].invoke() 这种下标访问
                    if isinstance(node.func.value, ast.Name):
                        return True
        return False

    def _is_direct_tool_call(self, node: ast.AST) -> bool:
        """检查是否为直接工具调用 tool(...)"""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id in self.tool_names
        return False

    def convert(self, node: ast.AST) -> ConversionResult:
        """转换工具调用"""
        if isinstance(node, ast.Assign):
            target = ast.unparse(node.targets[0])
            call_code = self._convert_tool_call(node.value)
            return ConversionResult.success_result(
                code=f'{target} = {call_code}'
            )

        if isinstance(node, ast.Expr):
            call_code = self._convert_tool_call(node.value)
            return ConversionResult.success_result(code=call_code)

        return ConversionResult.failure()

    def _convert_tool_call(self, node: ast.Call) -> str:
        """转换工具调用表达式为 tool.invoke(inputs={...}) 格式"""
        tool_name = ""
        inputs_dict_parts = []

        if isinstance(node.func, ast.Attribute):
            # tool.invoke(...) 格式
            if isinstance(node.func.value, ast.Name):
                tool_name = node.func.value.id

            # 获取参数
            if node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Dict):
                    # 已经是字典格式，提取键值对
                    for key, value in zip(arg.keys, arg.values):
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            inputs_dict_parts.append(f'"{key.value}": {ast.unparse(value)}')
                        elif isinstance(key, ast.Str):
                            inputs_dict_parts.append(f'"{key.s}": {ast.unparse(value)}')
                else:
                    # 其他情况直接使用
                    return f'{tool_name}.invoke(inputs={ast.unparse(arg)})'

            # 处理 keyword 参数 (如 inputs=...)
            for kw in node.keywords:
                if kw.arg == "inputs":
                    return f'{tool_name}.invoke(inputs={ast.unparse(kw.value)})'

        elif isinstance(node.func, ast.Name):
            # 直接调用 tool(...) 格式
            tool_name = node.func.id

            # 收集所有参数为字典格式
            # 位置参数（假设有参数名定义）
            for i, arg in enumerate(node.args):
                inputs_dict_parts.append(f'"arg{i}": {ast.unparse(arg)}')

            # 关键字参数
            for kw in node.keywords:
                if kw.arg:
                    inputs_dict_parts.append(f'"{kw.arg}": {ast.unparse(kw.value)}')

        # 构建 inputs 字典
        inputs_dict = "{" + ", ".join(inputs_dict_parts) + "}"
        return f'{tool_name}.invoke(inputs={inputs_dict})'


class ToolMapCallRule(StatementRule):
    """
    工具映射调用规则

    处理 tool_map[key].run(arg) 或 tool_map[key].invoke(arg) 模式
    转换为 openJiuwen 格式: invoke_tool(key, arg)

    在 openJiuwen 中，@tool 装饰的函数返回 LocalFunction，
    需要通过 .invoke(inputs={param_name: arg}) 调用
    由于不同工具有不同参数名，使用 invoke_tool 辅助函数处理
    """

    def matches(self, node: ast.AST) -> bool:
        """判断是否为 tool_map[key].run/invoke() 模式"""
        call_node = None
        if isinstance(node, ast.Assign):
            call_node = node.value
        elif isinstance(node, ast.Expr):
            call_node = node.value

        if isinstance(call_node, ast.Call):
            if isinstance(call_node.func, ast.Attribute):
                # 检查是 .run() 或 .invoke() 方法
                if call_node.func.attr in ("run", "invoke"):
                    # 检查调用对象是 subscript (如 tool_map[key])
                    if isinstance(call_node.func.value, ast.Subscript):
                        return True
        return False

    def convert(self, node: ast.AST) -> ConversionResult:
        """转换 tool_map[key].run(arg) → invoke_tool(key, arg)"""
        call_node = None
        target = None

        if isinstance(node, ast.Assign):
            target = ast.unparse(node.targets[0])
            call_node = node.value
        elif isinstance(node, ast.Expr):
            call_node = node.value

        if not isinstance(call_node, ast.Call):
            return ConversionResult.failure()

        # 获取 tool_map[key] 部分
        subscript = call_node.func.value  # ast.Subscript
        key = ast.unparse(subscript.slice)  # e.g., "selected_tool"

        # 获取参数（第一个参数）
        arg = ""
        if call_node.args:
            arg = ast.unparse(call_node.args[0])

        # 生成 invoke_tool 调用: invoke_tool(key, arg)
        call_code = f'invoke_tool({key}, {arg})'

        if target:
            return ConversionResult.success_result(code=f'{target} = {call_code}')
        else:
            return ConversionResult.success_result(code=call_code)


class ToolResultRule(StatementRule):
    """
    工具结果处理规则

    处理 ToolMessage 等工具相关的结果
    """

    def matches(self, node: ast.AST) -> bool:
        """判断是否为 ToolMessage 创建"""
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    return node.value.func.id == "ToolMessage"
        return False

    def convert(self, node: ast.AST) -> ConversionResult:
        """转换 ToolMessage"""
        if isinstance(node, ast.Assign):
            # 在 openJiuwen 中可能不需要 ToolMessage
            # 直接返回结果即可
            original = ast.unparse(node)
            return ConversionResult.success_result(
                code=f'# TODO: ToolMessage 在 openJiuwen 中可能不需要\n        # 原始代码: {original}'
            )

        return ConversionResult.failure()
