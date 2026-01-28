"""
待处理检查组件

检查是否有需要 AI 处理的待处理项
"""

from typing import Dict

from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.runtime.workflow import WorkflowRuntime
from openjiuwen.core.context_engine.base import Context

from ..workflow.state import ExtractionResult


class PendingCheckComp(WorkflowComponent, ComponentExecutable):
    """
    待处理检查组件

    功能：
    - 检查 extraction_result.pending_items 是否为空
    - 返回路由决策信息
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

        has_pending = extraction_result.has_pending()
        pending_count = len(extraction_result.pending_items)
        pending_summary = extraction_result.get_pending_summary()

        return {
            "has_pending": has_pending,
            "pending_count": pending_count,
            "pending_summary": pending_summary,
            "extraction_result": extraction_result
        }


def pending_router(runtime: Runtime) -> str:
    """
    路由函数：是否需要 AI 处理

    Returns:
        "ai" 如果有待处理项
        "ir_builder_direct" 如果没有待处理项
    """
    has_pending = runtime.get_global_state("checker.has_pending")
    return "ai" if has_pending else "ir_builder_direct"
