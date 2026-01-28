"""
文件加载组件

读取文件内容
"""

import os
from typing import Dict, List

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.context_engine.base import Context


class FileLoaderComp(WorkflowComponent, ComponentExecutable):
    """
    文件加载组件

    功能：
    - 读取文件列表中的所有文件内容
    - 返回文件路径到内容的映射
    """

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
        # 从 inputs 获取（通过 transformer 传入）
        file_list: List[str] = inputs.get("file_list", [])
        dependency_order: List[str] = inputs.get("dependency_order", [])

        file_contents: Dict[str, str] = {}
        errors: List[str] = []

        for file_path in file_list:
            try:
                content = self._read_file(file_path)
                file_contents[file_path] = content
            except Exception as e:
                errors.append(f"读取文件失败 {file_path}: {str(e)}")

        if errors and not file_contents:
            raise ValueError(f"无法读取任何文件: {'; '.join(errors)}")

        # 对文件路径进行编码，避免 openJiuwen 将 '.' 解析为路径分隔符
        # 使用 __DOT__ 替换 '.'
        encoded_file_contents = {}
        for path, content in file_contents.items():
            encoded_path = path.replace(".", "__DOT__")
            encoded_file_contents[encoded_path] = content

        encoded_dependency_order = [p.replace(".", "__DOT__") for p in dependency_order]

        return {
            "file_contents": encoded_file_contents,
            "dependency_order": encoded_dependency_order,
            "load_errors": errors
        }

    def _read_file(self, file_path: str) -> str:
        """读取单个文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 尝试多种编码
        encodings = ["utf-8", "gbk", "latin-1"]
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise ValueError(f"无法解码文件: {file_path}")
