"""
边和返回规则

处理 return 语句和边定义的转换
"""

import ast
from typing import Any, Dict, List, Optional

from .base import BaseRule, StatementRule, ConversionResult


class ReturnRule(StatementRule):
    """
    返回语句规则

    转换:
    - return state → return {"key1": key1, "key2": key2, ...}
    - return END → return "end"
    - return {"key": value} → return {"key": value}
    - return "node_name" → return "node_name"
    """

    def matches(self, node: ast.AST) -> bool:
        """判断是否为返回语句"""
        return isinstance(node, ast.Return)

    def convert(self, node: ast.AST) -> ConversionResult:
        """转换返回语句"""
        if not isinstance(node, ast.Return):
            return ConversionResult.failure()

        if node.value is None:
            return ConversionResult.success_result(code="return {}")

        # return END
        if isinstance(node.value, ast.Name) and node.value.id == "END":
            return ConversionResult.success_result(code='return "end"')

        # return state (变量名)
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            if var_name.lower() in ("state", "state_"):
                # 使用占位符，code_generator 会替换为实际的输出字典
                return ConversionResult.success_result(
                    code='return __COLLECTED_OUTPUTS__',
                    outputs=[]
                )
            else:
                # 可能是路由返回
                return ConversionResult.success_result(
                    code=f'return {ast.unparse(node.value)}'
                )

        # return {"key": value}
        if isinstance(node.value, ast.Dict):
            outputs = []
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    outputs.append(key.value)
                elif isinstance(key, ast.Str):
                    outputs.append(key.s)

            return ConversionResult.success_result(
                code=f'return {ast.unparse(node.value)}',
                outputs=outputs
            )

        # return "string" (路由返回)
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            value = node.value.value
            # END 转换为 "end"
            if value == "END":
                return ConversionResult.success_result(code='return "end"')
            return ConversionResult.success_result(code=f'return "{value}"')

        # 条件表达式: return END if condition else "node"
        if isinstance(node.value, ast.IfExp):
            return self._convert_conditional_return(node.value)

        # 其他情况
        return ConversionResult.success_result(
            code=f'return {ast.unparse(node.value)}'
        )

    def _convert_conditional_return(self, node: ast.IfExp) -> ConversionResult:
        """转换条件返回"""
        # 转换条件
        test_code = self._convert_condition(node.test)

        # 转换 body (if 分支)
        body_code = self._convert_return_value(node.body)

        # 转换 orelse (else 分支)
        orelse_code = self._convert_return_value(node.orelse)

        return ConversionResult.success_result(
            code=f'return {body_code} if {test_code} else {orelse_code}'
        )

    def _convert_condition(self, node: ast.AST) -> str:
        """转换条件表达式"""
        # state.get("key") → inputs.get("key")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id.lower() in ("state",):
                        if node.args:
                            key = ast.unparse(node.args[0])
                            return f'inputs.get({key})'

        # state["key"] → inputs["key"]
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                if node.value.id.lower() in ("state",):
                    return f'inputs[{ast.unparse(node.slice)}]'

        return ast.unparse(node)

    def _convert_return_value(self, node: ast.AST) -> str:
        """转换返回值"""
        if isinstance(node, ast.Name) and node.id == "END":
            return '"end"'
        if isinstance(node, ast.Constant) and node.value == "END":
            return '"end"'
        return ast.unparse(node)


class EdgeExtractor:
    """
    边提取器

    从 AST 中提取边定义（不是规则，是提取器）
    """

    def extract_edges(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """提取所有边定义"""
        edges = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "add_edge":
                    edge = self._parse_simple_edge(node)
                    if edge:
                        edges.append(edge)
                elif node.func.attr == "add_conditional_edges":
                    edge = self._parse_conditional_edge(node)
                    if edge:
                        edges.append(edge)
        return edges

    def _parse_simple_edge(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        """解析简单边"""
        if len(node.args) < 2:
            return None

        source = self._get_string_value(node.args[0])
        target = self._get_string_value(node.args[1])

        if source and target:
            return {
                "source": source,
                "target": target,
                "is_conditional": False
            }
        return None

    def _parse_conditional_edge(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        """解析条件边"""
        if len(node.args) < 2:
            return None

        source = self._get_string_value(node.args[0])
        if not source:
            return None

        # 第二个参数是路由函数
        router_func = None
        if isinstance(node.args[1], ast.Name):
            router_func = node.args[1].id

        # 第三个参数是映射
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

        return {
            "source": source,
            "target": None,
            "is_conditional": True,
            "router_func": router_func,
            "condition_map": condition_map
        }

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
