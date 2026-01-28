"""
示例工具：天气查询
"""

import httpx
from langchain_core.tools import tool

from ..config import SENIVERSE_API_KEY


@tool
def weather(params: str) -> str:
    """按城市+自然语言日期查天气（使用心知天气API）"""
    # 日期转索引：今天=0, 明天=1, 后天=2
    try:
        city = params.split(" ")[0]
        date = params.split(" ")[1]
        day_index = {"今天": 0, "明天": 1, "后天": 2}.get(date, 0)
    except IndexError:
        return f"参数错误：params={params}, 缺失日期或城市"

    # 心知天气API
    url = f"https://api.seniverse.com/v3/weather/daily.json?key={SENIVERSE_API_KEY}&location={city}&language=zh-Hans&unit=c&start={day_index}&days=1"

    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

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
