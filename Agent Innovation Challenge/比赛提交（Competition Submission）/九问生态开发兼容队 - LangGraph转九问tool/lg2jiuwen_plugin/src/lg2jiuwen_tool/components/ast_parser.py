"""
AST 解析组件

解析 Python 文件的抽象语法树
"""

import ast
from typing import Any, Dict, List

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.context_engine.base import Context


class ASTParserComp(WorkflowComponent, ComponentExecutable):
    """
    AST 解析组件

    功能：
    - 解析每个文件的 AST
    - 返回文件路径到 AST 的映射
    """

    def _unwrap_value(self, value):
        """解包可能被 openJiuwen 包装的值"""
        if isinstance(value, dict) and "" in value and len(value) == 1:
            return value[""]
        return value

    def _decode_path(self, path: str) -> str:
        """解码文件路径（将 __DOT__ 还原为 .）"""
        return path.replace("__DOT__", ".")

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
        # 从 inputs 获取并解包
        file_contents_raw = self._unwrap_value(inputs.get("file_contents", {}))
        dependency_order_raw = self._unwrap_value(inputs.get("dependency_order", []))

        # 解码文件路径
        file_contents: Dict[str, str] = {}
        if isinstance(file_contents_raw, dict):
            for encoded_path, content in file_contents_raw.items():
                decoded_path = self._decode_path(encoded_path)
                file_contents[decoded_path] = content

        dependency_order: List[str] = []
        if isinstance(dependency_order_raw, list):
            dependency_order = [self._decode_path(p) for p in dependency_order_raw]

        # 验证数据类型
        if not isinstance(file_contents, dict):
            raise TypeError(f"file_contents 应为 dict，实际为 {type(file_contents)}: {file_contents}")

        ast_map: Dict[str, ast.AST] = {}
        parse_errors: List[str] = []

        for file_path, content in file_contents.items():
            if not isinstance(content, str):
                parse_errors.append(f"文件内容不是字符串 {file_path}: {type(content)}")
                continue
            try:
                tree = ast.parse(content, filename=file_path)
                ast_map[file_path] = tree
            except SyntaxError as e:
                parse_errors.append(
                    f"语法错误 {file_path}:{e.lineno}: {e.msg}"
                )

        if parse_errors and not ast_map:
            raise ValueError(f"无法解析任何文件: {'; '.join(parse_errors)}")

        # 对输出的 key 进行编码，避免 openJiuwen 将 '.' 解析为路径分隔符
        def encode_path(path: str) -> str:
            return path.replace(".", "__DOT__")

        encoded_ast_map = {encode_path(k): v for k, v in ast_map.items()}
        encoded_dependency_order = [encode_path(p) for p in dependency_order]

        return {
            "ast_map": encoded_ast_map,
            "dependency_order": encoded_dependency_order,
            "parse_errors": parse_errors
        }
