"""
状态定义
"""

from typing import TypedDict, Optional


# 最大循环次数
MAX_LOOPS = 3


class AgentState(TypedDict):
    input: str                    # 用户输入问题
    thought: str                  # 思考内容
    selected_tool: Optional[str]  # 选择的工具名称
    tool_input: str               # 工具输入参数
    result: str                   # 工具执行结果
    is_end: bool                  # 是否终止循环
    loop_count: int               # 当前循环次数
