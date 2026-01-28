"""
lg2jiuwen_tool - LangGraph to openJiuwen Migration Tool

A tool for migrating LangGraph agents to openJiuwen framework.

使用方式:
    from lg2jiuwen_tool import migrate_new, MigrationOptions

    result = migrate_new(
        source_path="path/to/agent.py",
        output_dir="./output"
    )
"""

from .service import (
    migrate_new,
    migrate_async,
    MigrationOptions,
    MigrationResult,
)
from .ir.models import (
    AgentIR,
    WorkflowIR,
    WorkflowNodeIR,
    WorkflowEdgeIR,
    ToolIR,
    MigrationIR,
)

__version__ = "2.0.0"
__all__ = [
    # 迁移函数
    "migrate_new",
    "migrate_async",
    # 选项和结果
    "MigrationOptions",
    "MigrationResult",
    # IR 模型
    "AgentIR",
    "WorkflowIR",
    "WorkflowNodeIR",
    "WorkflowEdgeIR",
    "ToolIR",
    "MigrationIR",
]
