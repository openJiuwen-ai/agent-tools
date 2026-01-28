"""
路由函数
"""

from openjiuwen.core.runtime.workflow import WorkflowRuntime
from .config import MAX_LOOPS

def judge_router(runtime: WorkflowRuntime) -> str:
    """路由函数：根据 judge 的输出决定下一个节点"""
    if runtime.get_global_state("judge.is_end"):
        return "end"
    if (runtime.get_global_state("loop_count") or 0) >= MAX_LOOPS:
        print(f'达到最大循环次数 {MAX_LOOPS}，强制结束')
        return "end"
    return "think"
