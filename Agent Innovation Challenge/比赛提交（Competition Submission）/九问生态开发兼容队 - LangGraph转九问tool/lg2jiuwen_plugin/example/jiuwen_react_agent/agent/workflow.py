"""
工作流构建
"""

from openjiuwen.core.workflow.base import Workflow
from openjiuwen.core.component.start_comp import Start
from openjiuwen.core.component.end_comp import End

from .components.think_comp import ThinkComp
from .components.select_tool_comp import SelectToolComp
from .components.judge_comp import JudgeComp
from .routers import judge_router


def build_agent_workflow() -> Workflow:
    """构建 Agent 工作流"""

    workflow = Workflow()

    # 设置起点
    workflow.set_start_comp("start", Start(), inputs_schema={
        "input": "${input}",
        "is_end": "${is_end}",
        "loop_count": "${loop_count}",
    })

    # 添加组件
    workflow.add_workflow_comp(
        "think",
        ThinkComp(),
        inputs_schema={"loop_count": "${start.loop_count}", "input": "${start.input}"}
    )

    workflow.add_workflow_comp(
        "select_tool",
        SelectToolComp(),
        inputs_schema={"tool_input": "${think.tool_input}", "selected_tool": "${think.selected_tool}"}
    )

    workflow.add_workflow_comp(
        "judge",
        JudgeComp(),
        inputs_schema={"selected_tool": "${think.selected_tool}", "result": "${select_tool.result}", "input": "${start.input}"}
    )

    # 设置终点
    workflow.set_end_comp("end", End(), inputs_schema={"tool_input": "${think.tool_input}", "selected_tool": "${think.selected_tool}", "thought": "${think.thought}", "loop_count": "${think.loop_count}", "result": "${select_tool.result}", "is_end": "${judge.is_end}", "reason": "${judge.reason}"})

    # 添加连接
    workflow.add_connection("start", "think")
    workflow.add_connection("think", "select_tool")
    workflow.add_connection("select_tool", "judge")
    workflow.add_conditional_connection("judge", judge_router)

    return workflow