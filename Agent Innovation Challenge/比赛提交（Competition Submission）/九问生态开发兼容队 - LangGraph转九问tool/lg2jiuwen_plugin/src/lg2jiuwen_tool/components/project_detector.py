"""
项目检测组件

检测项目类型（单文件/多文件），分析文件依赖关系
"""

import ast
import os
from typing import Any, Dict, List, Set, Tuple

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.runtime.workflow import WorkflowRuntime
from openjiuwen.core.context_engine.base import Context


class ProjectDetectorComp(WorkflowComponent, ComponentExecutable):
    """
    项目检测组件

    功能：
    - 检测输入是单文件还是目录
    - 扫描目录下所有 Python 文件
    - 分析文件间的 import 依赖关系
    - 按拓扑排序返回处理顺序
    """

    async def invoke(
        self,
        inputs: Input,
        runtime: Runtime,
        context: Context
    ) -> Output:
        # 从 inputs 获取
        source_path = inputs.get("source_path", "")

        # 判断是文件还是目录
        if os.path.isfile(source_path):
            if not source_path.endswith(".py"):
                raise ValueError(f"不支持的文件类型: {source_path}")
            # 单文件模式
            return {
                "is_multi_file": False,
                "file_list": [source_path],
                "dependency_order": [source_path],
                "project_root": os.path.dirname(source_path) or "."
            }
        elif os.path.isdir(source_path):
            # 多文件模式
            files = self._scan_python_files(source_path)
            if not files:
                raise ValueError(f"目录中没有 Python 文件: {source_path}")

            deps = self._analyze_dependencies(files, source_path)
            order = self._topological_sort(deps, files)

            return {
                "is_multi_file": True,
                "file_list": files,
                "dependency_order": order,
                "project_root": source_path
            }
        else:
            raise ValueError(f"路径不存在: {source_path}")

    def _scan_python_files(self, directory: str) -> List[str]:
        """扫描目录下所有 Python 文件（包括 __init__.py）"""
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                # 包含 __init__.py，但排除 __pycache__ 等
                if file.endswith(".py"):
                    # 跳过 __pycache__ 目录
                    if "__pycache__" in root:
                        continue
                    full_path = os.path.join(root, file)
                    python_files.append(full_path)
        return sorted(python_files)

    def _file_to_module(self, file_path: str, project_root: str) -> str:
        """将文件路径转换为模块名"""
        rel_path = os.path.relpath(file_path, project_root)
        module = rel_path.replace(os.sep, ".").replace("/", ".")
        if module.endswith(".py"):
            module = module[:-3]
        return module

    def _analyze_dependencies(
        self,
        files: List[str],
        project_root: str
    ) -> Dict[str, List[str]]:
        """
        分析文件间依赖关系

        返回: {file_path: [dependent_file_paths]}
        """
        # 建立模块名到文件路径的映射
        module_to_file: Dict[str, str] = {}
        for file_path in files:
            module = self._file_to_module(file_path, project_root)
            module_to_file[module] = file_path
            # 也记录包名（不带最后一级）
            parts = module.split(".")
            if len(parts) > 1:
                parent = ".".join(parts[:-1])
                if parent not in module_to_file:
                    module_to_file[parent] = file_path

        deps: Dict[str, List[str]] = {f: [] for f in files}

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content)
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                imported_modules = self._get_imported_modules(node)
                for module in imported_modules:
                    # 检查是否为项目内部模块
                    if module in module_to_file:
                        dep_file = module_to_file[module]
                        if dep_file != file_path and dep_file not in deps[file_path]:
                            deps[file_path].append(dep_file)

        return deps

    def _get_imported_modules(self, node: ast.AST) -> List[str]:
        """从 import 语句中提取模块名"""
        modules = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
                # 也添加被导入的名称作为可能的子模块
                for alias in node.names:
                    modules.append(f"{node.module}.{alias.name}")
        return modules

    def _topological_sort(
        self,
        deps: Dict[str, List[str]],
        files: List[str]
    ) -> List[str]:
        """
        拓扑排序

        确保被依赖的文件先处理
        """
        in_degree: Dict[str, int] = {f: 0 for f in files}
        graph: Dict[str, List[str]] = {f: [] for f in files}

        # 构建图：如果 A 依赖 B，则 B -> A
        for file, dependencies in deps.items():
            for dep in dependencies:
                if dep in graph:
                    graph[dep].append(file)
                    in_degree[file] += 1

        # Kahn's algorithm
        queue = [f for f in files if in_degree[f] == 0]
        result = []

        while queue:
            # 按字母顺序处理，保证确定性
            queue.sort()
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 如果有循环依赖，添加剩余文件
        if len(result) < len(files):
            remaining = [f for f in files if f not in result]
            result.extend(sorted(remaining))

        return result
