"""当前天气工具：OpenWeather Current weather data API (2.5)。"""

from typing import Any

import requests
from openjiuwen.core.foundation.tool import tool

from .common import (
    CURRENT_WEATHER_API_URL,
    OpenWeatherQuery,
    http_error_message,
    open_weather_query_from_resolved,
    resolve_location_and_common,
)


def _fetch_current_weather(query: OpenWeatherQuery) -> dict:
    """调用 Current weather data API (2.5)，支持经纬度或城市名 q。"""
    params = {"appid": query.api_key, "units": query.units, "lang": query.lang}
    if query.coords is not None:
        params["lat"] = query.coords.lat
        params["lon"] = query.coords.lon
    elif query.city:
        params["q"] = query.city
    else:
        raise ValueError("必须提供 lat/lon 或 city")
    resp = requests.get(CURRENT_WEATHER_API_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _format_current_weather_report(data: dict, units: str) -> str:
    """将 Current weather API (2.5) 返回整理为可读摘要。"""
    name = data.get("name", "")
    sys_country = (data.get("sys") or {}).get("country", "")
    location = f"{name}, {sys_country}" if sys_country else (name or "—")

    main = data.get("main") or {}
    temp = main.get("temp")
    feels_like = main.get("feels_like")
    temp_unit = "℃" if units == "metric" else "℉" if units == "imperial" else "K"
    temp_str = f"{temp}{temp_unit}" if temp is not None else "N/A"
    feels_str = f"{feels_like}{temp_unit}" if feels_like is not None else "N/A"

    weather_list = data.get("weather") or []
    desc = weather_list[0].get("description", "N/A") if weather_list else "N/A"
    humidity = main.get("humidity")
    humidity_str = f"{humidity}%" if humidity is not None else "N/A"
    wind = data.get("wind") or {}
    speed = wind.get("speed")
    wind_str = f"{speed} m/s" if speed is not None else "N/A"

    lines = [
        f"位置: {location}",
        f"天气: {desc}",
        f"温度: {temp_str} (体感 {feels_str})",
        f"湿度: {humidity_str}",
        f"风速: {wind_str}",
    ]
    return "\n".join(lines)


@tool(
    name="openweathermap_weather",
    description=(
        "基于 OpenWeather Current weather data API (2.5) 查询当前天气，"
        "支持经纬度或城市名（英文）。免费 Key 即可用。"
        "Query current weather by lat/lon or city name using OpenWeather API 2.5."
    ),
    input_params={
        "type": "object",
        "properties": {
            "lat": {
                "type": "number",
                "description": "可选，纬度，十进制度数，范围 -90~90。未提供时可用 city。",
            },
            "lon": {
                "type": "number",
                "description": "可选，经度，十进制度数，范围 -180~180。未提供时可用 city。",
            },
            "city": {
                "type": "string",
                "description": "可选，城市名称（英文，如 Beijing、London），使用 API 内置 q 参数。",
            },
            "units": {
                "type": "string",
                "description": "温度单位：standard、metric（摄氏度）或 imperial（华氏度）。",
                "enum": ["standard", "metric", "imperial"],
                "default": "metric",
            },
            "lang": {
                "type": "string",
                "description": "返回描述的语言代码，如 zh_cn、en。",
                "default": "zh_cn",
            },
        },
        "required": [],
    },
)
def openweathermap_weather(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """调用 Current weather data 2.5，返回当前天气摘要与原始数据。支持 lat/lon 或 city 二选一。"""
    params = params or kwargs
    resolved = resolve_location_and_common(params)
    if "error" in resolved:
        return resolved
    query = open_weather_query_from_resolved(resolved)
    try:
        data = _fetch_current_weather(query)
    except requests.exceptions.HTTPError as e:
        return {"error": f"Current weather API 请求失败: {http_error_message(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"网络请求异常: {e}"}
    report = _format_current_weather_report(data, resolved["units"])
    return {"report": report, "raw": data}
