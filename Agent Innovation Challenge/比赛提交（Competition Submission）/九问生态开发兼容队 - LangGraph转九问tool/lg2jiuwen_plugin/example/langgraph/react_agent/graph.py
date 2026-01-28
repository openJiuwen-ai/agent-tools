"""
图构建
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import think_node, select_tool_node, judge_node
from .router import judge_router


# 构建图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("think", think_node)
graph.add_node("select_tool", select_tool_node)
graph.add_node("judge", judge_node)

# 设置流转规则
graph.set_entry_point("think")
graph.add_edge("think", "select_tool")
graph.add_edge("select_tool", "judge")

# 条件边
graph.add_conditional_edges(
    "judge",
    judge_router,
    {"end": END, "think": "think"}
)

# 编译
app = graph.compile()
