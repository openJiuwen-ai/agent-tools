import os
import httpx
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ------------------ 0. 状态 ------------------
class AgentState(TypedDict):
    sentence: str
    city: Optional[str]
    date: Optional[str]          # 自然语言日期
    weather: Optional[str]
    error: Optional[str]

# ------------------ 1. 工具 ------------------
SENIVERSE_API_KEY = "SBM-xxxxxx"

@tool
def get_weather(city: str, date: str) -> str:
    """按城市+自然语言日期查天气（使用心知天气API）"""
    # 日期转索引：今天=0, 明天=1, 后天=2
    day_index = {"今天": 0, "明天": 1, "后天": 2}.get(date, 0)

    # 心知天气API: start=开始天数, days=返回天数
    url = f"https://api.seniverse.com/v3/weather/daily.json?key={SENIVERSE_API_KEY}&location={city}&language=zh-Hans&unit=c&start={day_index}&days=1"

    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # 解析响应
        result = data["results"][0]
        location = result["location"]["name"]
        daily = result["daily"][0]

        return f"{location} {daily['date']}: {daily['low']}~{daily['high']}°C, 白天{daily['text_day']}, 夜间{daily['text_night']}"
    except httpx.HTTPStatusError as e:
        return f"天气服务异常：HTTP {e.response.status_code}"
    except (KeyError, IndexError) as e:
        return f"天气数据解析异常：{e}"
    except Exception as e:
        return f"天气服务异常：{e}"


# ------------------ 2. 大模型 ----------------
llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0
)

# ------------------ 3. 节点函数 --------------
def parse_input_llm(state: AgentState) -> AgentState:
    """LLM 提取 city & date，缺一个就报错"""
    sys_prompt = (
        "用户会输入一句话，请提取“城市”和“日期”。\n"
        "日期可以是：今天、明天、后天 等。\n"
        "如果城市或日期缺失，请只返回 ERROR: 缺失城市 或 ERROR: 缺失日期，不要多余文字。\n"
        "否则返回 JSON: {\"city\": \"城市\", \"date\": \"日期\"}"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": state["sentence"]}
    ]
    ans = llm.invoke(messages).content.strip()
    print("llm ans:",ans)
    try:
        import json
        obj = json.loads(ans)
        state["city"] = obj["city"]
        state["date"] = obj["date"]
    except Exception:
        state["error"] = ans
    return state

def route_after_extract(state: AgentState) -> str:
    return END if state.get("error") else "call_weather"

def call_weather(state: AgentState) -> AgentState:
    """直接调用天气工具获取结果"""
    city = state.get("city")
    date = state.get("date")

    if not city or not date:
        state["error"] = "缺少城市或日期参数"
        return state

    # 直接调用工具函数（不通过 ToolNode）
    result = get_weather.invoke({"city": city, "date": date})
    print("weather result:", result)

    state["weather"] = result

    return state

# ------------------ 4. 构图 --------------------
workflow = StateGraph(AgentState)
workflow.add_node("extract", parse_input_llm)
workflow.add_node("call_weather", call_weather)

workflow.set_entry_point("extract")
workflow.add_conditional_edges("extract", route_after_extract, {"call_weather": "call_weather", END: END})
workflow.add_edge("call_weather", END)

graph = workflow.compile()

# ------------------ 5. 运行 --------------------
if __name__ == "__main__":
    sentence = "明天北京天气"
    result = graph.invoke({"sentence": sentence})
    if result.get("error"):
        print("❌", result["error"])
    else:
        print("✅ 天气：", result["weather"])