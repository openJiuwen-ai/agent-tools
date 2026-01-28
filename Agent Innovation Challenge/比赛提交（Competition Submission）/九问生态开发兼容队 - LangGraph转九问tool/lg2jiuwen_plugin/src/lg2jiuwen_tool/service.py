"""
LG2Jiuwen 服务接口

提供迁移服务的编程接口
"""

import asyncio
import os
import subprocess
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjiuwen.core.runtime.workflow import WorkflowRuntime

from .workflow.migration_workflow import (
    build_migration_workflow,
    build_simple_migration_workflow,
)


@dataclass
class MigrationOptions:
    """迁移选项"""
    use_ai: bool = True                  # 是否使用 AI 处理
    preserve_comments: bool = True       # 是否保留注释
    include_report: bool = True          # 是否生成报告
    verbose: bool = False                # 是否输出详细信息


@dataclass
class MigrationResult:
    """迁移结果"""
    success: bool                        # 是否成功
    generated_files: List[str]           # 生成的文件列表
    report: str                          # 迁移报告
    rule_count: int                      # 规则处理数量
    ai_count: int                        # AI 处理数量
    errors: List[str]                    # 错误信息


async def migrate_async(
    source_path: str,
    output_dir: str = "./output",
    options: Optional[MigrationOptions] = None,
    llm=None
) -> MigrationResult:
    """
    异步迁移 LangGraph 代码到 openJiuwen

    Args:
        source_path: 源文件或目录路径
        output_dir: 输出目录
        options: 迁移选项
        llm: 可选的 LLM 实例

    Returns:
        MigrationResult: 迁移结果
    """
    options = options or MigrationOptions()

    # 验证输入
    if not os.path.exists(source_path):
        return MigrationResult(
            success=False,
            generated_files=[],
            report="",
            rule_count=0,
            ai_count=0,
            errors=[f"源路径不存在: {source_path}"]
        )

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 构建工作流
        if options.use_ai and llm:
            workflow = build_migration_workflow(llm=llm)
        else:
            workflow = build_simple_migration_workflow()

        # 创建运行时
        runtime = WorkflowRuntime()

        # 执行工作流
        inputs = {
            "source_path": source_path,
            "output_dir": output_dir
        }

        result = await workflow.invoke(inputs, runtime)
        result = result.result
        # 提取结果 - WorkflowOutput 对象需要通过 .output 属性访问
        output_data = result["output"] if "output" in result else result
        if isinstance(output_data, dict):
            generated_files = output_data.get("generated_files", [])
            report = output_data.get("report", "")
        else:
            generated_files = getattr(output_data, 'generated_files', [])
            report = getattr(output_data, 'report', "")

        # 从报告中提取统计
        rule_count = 0
        ai_count = 0
        if "规则处理" in report:
            # 简单解析
            import re
            rule_match = re.search(r"规则处理 \| (\d+)", report)
            if rule_match:
                rule_count = int(rule_match.group(1))
            ai_match = re.search(r"AI 处理 \| (\d+)", report)
            if ai_match:
                ai_count = int(ai_match.group(1))

        return MigrationResult(
            success=True,
            generated_files=generated_files,
            report=report,
            rule_count=rule_count,
            ai_count=ai_count,
            errors=[]
        )

    except Exception as e:
        return MigrationResult(
            success=False,
            generated_files=[],
            report="",
            rule_count=0,
            ai_count=0,
            errors=[str(e)]
        )


async def migrate_new(
    source_path: str,
    output_dir: str = "./output",
    options: Optional[MigrationOptions] = None,
    llm=None
) -> MigrationResult:
    """
    同步迁移 LangGraph 代码到 openJiuwen (新版本)

    Args:
        source_path: 源文件或目录路径
        output_dir: 输出目录
        options: 迁移选项
        llm: 可选的 LLM 实例

    Returns:
        MigrationResult: 迁移结果
    """
    return await migrate_async(source_path, output_dir, options, llm)


# ==================== 兼容旧版本接口 ====================

async def migrate(source_path: str, output_dir: str) -> str:
    """
    迁移接口（兼容旧版本）

    Args:
        source_path: 源文件路径
        output_dir: 输出目录

    Returns:
        str: 生成的文件路径
    """
    base_dir = os.getenv("BASE_DIR", "/")
    source_path = os.path.join(base_dir, source_path.strip(os.path.sep))
    if not source_path.endswith(".py"):
        raise ValueError(f"{source_path} must be a python file")
    if not os.path.exists(source_path):
        raise ValueError(f"{source_path} does not exist")
    output_dir = os.path.join(base_dir, output_dir.strip(os.path.sep))
    os.makedirs(output_dir, exist_ok=True)

    options = MigrationOptions(preserve_comments=True, use_ai=False)
    result = await migrate_new(source_path, output_dir, options)

    result_file = ""
    for f in result.generated_files:
        if f.endswith(".py"):
            result_file = os.path.relpath(f, base_dir)
            break
    return result_file


def get_file_content(file_path: str) -> str:
    """获取文件内容"""
    base_dir = os.getenv("BASE_DIR", "/")
    file_path = os.path.join(base_dir, file_path.strip(os.path.sep))
    if not os.path.exists(file_path):
        raise ValueError(f"{file_path} does not exist")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def run(source_path: str) -> str:
    """运行指定的Python文件或目录下的main.py文件"""
    base_dir = os.getenv("BASE_DIR","/")
    source_path = os.path.join(base_dir,source_path.strip(os.path.sep))
    if not os.path.exists(source_path):
        raise ValueError(f"{source_path} does not exist")

    cmd = f"source {os.path.join(os.getcwd(),".venv/bin/activate")}"
    if platform.system().lower() == "windows":
        cmd = f"{os.path.join(os.getcwd(),".venv/Scripts/activate.bat")}"
    shell = os.getenv("SHELL",None)
    if Path(source_path).is_dir():
        if not os.path.exists(os.path.join(source_path,"main.py")):
            raise ValueError(f"{source_path} is a directory but does not contain main.py")
        main_file = os.path.join(source_path,"main.py")
        result = subprocess.run(
            f"{cmd} && python {main_file}", shell=True, executable=shell, cwd=source_path, capture_output=True
        )
    elif source_path.endswith(".py"):
        result = subprocess.run(
            f"{cmd} && python {source_path}",
            shell=True,
            executable=shell,
            cwd=os.path.dirname(source_path),
            capture_output=True,
        )
    else:
        raise ValueError(f"{source_path} is not a directory or a Python file")

    return result.stdout.decode("utf-8")


# ==================== 服务类 ====================

class MigrationService:
    """
    迁移服务类

    提供更灵活的迁移服务接口
    """

    def __init__(self, llm=None, options: Optional[MigrationOptions] = None):
        """
        初始化迁移服务

        Args:
            llm: LLM 实例
            options: 默认迁移选项
        """
        self._llm = llm
        self._options = options or MigrationOptions()

    async def migrate_file(
        self,
        source_file: str,
        output_dir: str = "./output"
    ) -> MigrationResult:
        """迁移单个文件"""
        return await migrate_async(
            source_file,
            output_dir,
            self._options,
            self._llm
        )

    async def migrate_project(
        self,
        source_dir: str,
        output_dir: str = "./output"
    ) -> MigrationResult:
        """迁移项目目录"""
        return await migrate_async(
            source_dir,
            output_dir,
            self._options,
            self._llm
        )

    def migrate_file_sync(
        self,
        source_file: str,
        output_dir: str = "./output"
    ) -> MigrationResult:
        """同步迁移单个文件"""
        return asyncio.run(self.migrate_file(source_file, output_dir))

    def migrate_project_sync(
        self,
        source_dir: str,
        output_dir: str = "./output"
    ) -> MigrationResult:
        """同步迁移项目目录"""
        return asyncio.run(self.migrate_project(source_dir, output_dir))
