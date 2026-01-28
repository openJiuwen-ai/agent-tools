"""
状态相关规则

处理 state["x"] 和 state.get("x") 的转换
"""

import ast
from typing import Optional, Set

from .base import BaseRule, StatementRule, ConversionResult


class StateAccessRule(BaseRule):
    """
    状态访问规则

    转换策略：
    - 全局状态变量（initial_inputs）: runtime.get_global_state("key")
    - 其他变量: inputs["key"]（通过 inputs_schema 从上游组件获取）

    由于规则转换时不知道哪些是全局状态，使用通用模式：
    - state["key"] → inputs["key"]（工作流通过 inputs_schema 传递）
    - state.get("key") → inputs.get("key")
    - state.get("key", default) → inputs.get("key", default)

    全局状态的处理在 code_generator 中根据 initial_inputs 决定
    """

    def matches(self, node: ast.AST) -> bool:
        """判断是否为状态访问"""
        # 匹配 state["key"]
        if isinstance(node, ast.Subscript):
            return self._is_state_var(node.value)

        # 匹配 state.get("key")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "get":
                    return self._is_state_var(node.func.value)

        return False

    def convert(self, node: ast.AST) -> ConversionResult:
        """转换状态访问"""
        if isinstance(node, ast.Subscript):
            key = self._extract_string_key(node.slice)
            if key:
                # 使用 inputs（工作流通过 inputs_schema 传递值）
                return ConversionResult.success_result(
                    code=f'inputs["{key}"]',
                    inputs=[key]
                )

        if isinstance(node, ast.Call):
            # state.get("key") 或 state.get("key", default)
            if node.args:
                key = self._extract_string_key(node.args[0])
                if key:
                    if len(node.args) > 1:
                        default = ast.unparse(node.args[1])
                        return ConversionResult.success_result(
                            code=f'inputs.get("{key}", {default})',
                            inputs=[key]
                        )
                    else:
                        return ConversionResult.success_result(
                            code=f'inputs.get("{key}")',
                            inputs=[key]
                        )

        return ConversionResult.failure(error_message="无法解析状态访问")


class StateAssignRule(StatementRule):
    """
    状态赋值规则

    转换:
    - state["key"] = value → 记录输出，生成赋值到局部变量

    注意：最终的 return 语句会在 ReturnRule 中处理
    """

    def __init__(self):
        self._collected_outputs: Set[str] = set()

    def matches(self, node: ast.AST) -> bool:
        """判断是否为状态赋值"""
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    if self._is_state_var(target.value):
                        return True
        return False

    def convert(self, node: ast.AST) -> ConversionResult:
        """转换状态赋值"""
        if not isinstance(node, ast.Assign):
            return ConversionResult.failure()

        outputs = []
        code_lines = []

        for target in node.targets:
            if isinstance(target, ast.Subscript) and self._is_state_var(target.value):
                key = self._extract_string_key(target.slice)
                if key:
                    outputs.append(key)
                    # 转换为局部变量赋值
                    value_code = ast.unparse(node.value)
                    code_lines.append(f'{key} = {value_code}')

        if outputs:
            return ConversionResult.success_result(
                code="\n".join(code_lines),
                outputs=outputs
            )

        return ConversionResult.failure(error_message="无法解析状态赋值")


class StateUpdateRule(StatementRule):
    """
    状态更新规则

    处理 state.update({...}) 的情况
    """

    def matches(self, node: ast.AST) -> bool:
        """判断是否为 state.update()"""
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute):
                if call.func.attr == "update":
                    return self._is_state_var(call.func.value)
        return False

    def convert(self, node: ast.AST) -> ConversionResult:
        """转换 state.update()"""
        if not isinstance(node, ast.Expr):
            return ConversionResult.failure()

        call = node.value
        if not isinstance(call, ast.Call):
            return ConversionResult.failure()

        # 解析 update 的参数
        if call.args and isinstance(call.args[0], ast.Dict):
            dict_node = call.args[0]
            outputs = []
            code_lines = []

            for key_node, value_node in zip(dict_node.keys, dict_node.values):
                key = self._extract_string_key(key_node)
                if key:
                    outputs.append(key)
                    value_code = ast.unparse(value_node)
                    code_lines.append(f'{key} = {value_code}')

            if outputs:
                return ConversionResult.success_result(
                    code="\n".join(code_lines),
                    outputs=outputs
                )

        return ConversionResult.failure(error_message="无法解析 state.update()")
