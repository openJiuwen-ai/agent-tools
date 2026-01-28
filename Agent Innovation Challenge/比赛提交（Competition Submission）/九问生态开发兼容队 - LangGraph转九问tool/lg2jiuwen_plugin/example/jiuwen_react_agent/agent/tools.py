"""
工具函数
"""

import httpx
from openjiuwen.core.utils.tool.param import Param
from openjiuwen.core.utils.tool.tool import tool

from .config import *

@tool(
    name="calculator",
    description="用于数学加减乘除计算",
    params=[
        Param(name="expression", description="expression", type="string", required=True)
    ]
)
def calculator(expression: str) -> str:
    """用于数学加减乘除计算"""
    try:
        result = eval(expression, {'__builtins__': {}}, {})
        return str(result)
    except Exception as e:
        return f'计算失败：{e}'

@tool(
    name="weather",
    description="按城市+自然语言日期查天气（使用心知天气API）",
    params=[
        Param(name="params", description="params", type="string", required=True)
    ]
)
def weather(params: str) -> str:
    """按城市+自然语言日期查天气（使用心知天气API）"""
    try:
        city = params.split(' ')[0]
        date = params.split(' ')[1]
        day_index = {'今天': 0, '明天': 1, '后天': 2}.get(date, 0)
    except IndexError:
        return f'参数错误：params={params}, 缺失日期或城市'
    url = f'https://api.seniverse.com/v3/weather/daily.json?key={SENIVERSE_API_KEY}&location={city}&language=zh-Hans&unit=c&start={day_index}&days=1'
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        result = data['results'][0]
        location = result['location']['name']
        daily = result['daily'][0]
        return f'{location} {daily['date']}: {daily['low']}~{daily['high']}°C, 白天{daily['text_day']}, 夜间{daily['text_night']}'
    except httpx.HTTPStatusError as e:
        return f'天气服务异常：HTTP {e.response.status_code}'
    except (KeyError, IndexError) as e:
        return f'天气数据解析异常：{e}'
    except Exception as e:
        return f'天气服务异常：{e}'

# 工具映射
tool_map = {'Calculator': calculator, 'Weather': weather}


def invoke_tool(tool_name: str, arg: str) -> str:
    """
    调用工具的辅助函数

    openJiuwen 的 @tool 装饰器返回 LocalFunction，
    需要通过 .invoke(inputs={param_name: arg}) 调用。
    此函数自动处理参数名映射。
    """
    tool_func = tool_map.get(tool_name)
    if tool_func is None:
        return f"未知工具: {tool_name}"
    # 获取工具的第一个参数名
    if hasattr(tool_func, "params") and tool_func.params:
        param_name = tool_func.params[0].name
    else:
        param_name = "input"
    return tool_func.invoke(inputs={param_name: arg})
