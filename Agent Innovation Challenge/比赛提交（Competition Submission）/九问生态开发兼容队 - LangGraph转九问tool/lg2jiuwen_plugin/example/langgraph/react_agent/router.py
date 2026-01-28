"""
路由函数
"""

from .state import AgentState, MAX_LOOPS


def judge_router(state: AgentState) -> str:
    """判断是否结束（达到最大循环次数或任务完成）"""
    if state.get("is_end"):
        return "end"
    if state.get("loop_count", 0) >= MAX_LOOPS:
        print(f"达到最大循环次数 {MAX_LOOPS}，强制结束")
        return "end"
    return "think"
