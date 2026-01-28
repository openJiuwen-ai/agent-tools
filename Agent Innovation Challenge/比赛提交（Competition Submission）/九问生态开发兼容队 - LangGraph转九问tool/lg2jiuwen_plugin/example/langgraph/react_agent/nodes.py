"""
节点函数
"""

from .state import AgentState
from .config import llm
from .tools import tool_map


def think_node(state: AgentState) -> dict:
    """思考节点：分析问题并选择合适的工具"""
    content = f"""用户问题：{state['input']}

可用工具：
1. Calculator - 用于数学加减乘除计算
2. Weather - 用于查询城市天气

请分析用户意图，选择合适的工具。
返回格式（严格遵守）：
工具：<工具名>
参数：<工具所需参数>
思考：<一句话说明理由>

示例1：
工具：Calculator
参数：100+200
思考：用户想计算数学表达式

示例2：
工具：Weather
参数：北京 今天
思考：用户想查询天气"""

    messages = [{"role": "user", "content": content}]
    response = llm.invoke(messages).content

    # 解析响应
    selected_tool = None
    tool_input = ""
    thought = response

    print("response:", response)
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("工具：") or line.startswith("工具:"):
            selected_tool = line.split("：")[-1].split(":")[-1].strip()
        elif line.startswith("参数：") or line.startswith("参数:"):
            tool_input = line.split("：")[-1].split(":")[-1].strip()
        elif line.startswith("思考：") or line.startswith("思考:"):
            thought = line.split("：")[-1].split(":")[-1].strip()

    # 增加循环计数
    loop_count = state.get("loop_count", 0) + 1

    return {
        "thought": thought,
        "selected_tool": selected_tool,
        "tool_input": tool_input,
        "loop_count": loop_count
    }


def select_tool_node(state: AgentState) -> dict:
    """工具执行节点：根据选择调用对应工具"""
    selected_tool = state.get("selected_tool")
    tool_input = state.get("tool_input", "")

    if not selected_tool or selected_tool not in tool_map:
        return {"result": f"未知工具：{selected_tool}，可用工具：Calculator, Weather"}

    result = tool_map[selected_tool].invoke(tool_input)
    return {"result": result}


def judge_node(state: AgentState) -> dict:
    """终止判断节点"""
    content = f"""用户问题：{state['input']}
使用工具：{state.get('selected_tool', '')}
工具结果：{state.get('result', '')}

问题：根据工具结果，是否已经能够回答用户的问题？不能回答需要给出原因
返回格式（严格遵守）：
结果：True 或 False
原因：<一句话说明> （可选，只有结果为False时才给出原因）

示例1：
结果：True

示例2：
结果：False
原因：天气查询失败，缺失查询日期 

示例3：
结果：False
原因：计算器无法处理该输入，请重新输入

"""


    messages = [{"role": "user", "content": content}]
    reason = None
    response = llm.invoke(messages).content.strip().lower()
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("结果：") or line.startswith("结果:"):
            res = line.split("：")[-1].split(":")[-1].strip()
            is_end = res in ("true","True", "yes", "是", "1")
        if is_end:
            break
        
        if line.startswith("原因：") or line.startswith("原因:"):
            reason = line.split("：")[-1].split(":")[-1].strip()
            print(reason)    
    return {"is_end": is_end, "reason": reason}