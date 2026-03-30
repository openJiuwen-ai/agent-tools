"""16 天每日预报工具：OpenWeather Daily Forecast 16 Days API。"""

from typing import Any

import requests
from openjiuwen.core.foundation.tool import tool

from .common import (
    FORECAST_16D_API_URL,
    ForecastDailyRequest,
    fetch_forecast_daily,
    format_forecast_daily_report,
    http_error_message,
    open_weather_query_from_resolved,
    resolve_location_and_common,
)


@tool(
    name="openweathermap_forecast_16d",
    description=(
        "基于 OpenWeather Daily Forecast 16 Days API 查询未来 1~16 天每日预报，"
        "支持经纬度或城市名。返回每日日温、天气描述、降水概率等。"
        "Query 1-16 day daily forecast by lat/lon or city. See forecast16 docs."
    ),
    input_params={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "可选，纬度 -90~90。未提供时用 city。"},
            "lon": {"type": "number", "description": "可选，经度 -180~180。未提供时用 city。"},
            "city": {"type": "string", "description": "可选，城市英文名，如 Beijing、London。"},
            "cnt": {
                "type": "integer",
                "description": "返回天数，1~16，默认 7。",
                "minimum": 1,
                "maximum": 16,
                "default": 7,
            },
            "units": {
                "type": "string",
                "enum": ["standard", "metric", "imperial"],
                "default": "metric",
            },
            "lang": {"type": "string", "default": "zh_cn"},
        },
        "required": [],
    },
)
def openweathermap_forecast_16d(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """调用 16 天每日预报接口，返回摘要与原始数据。"""
    params = params or kwargs
    resolved = resolve_location_and_common(params)
    if "error" in resolved:
        return resolved
    cnt = max(1, min(16, int(params.get("cnt", 7))))
    req = ForecastDailyRequest(
        url=FORECAST_16D_API_URL,
        cnt=cnt,
        query=open_weather_query_from_resolved(resolved),
    )
    try:
        data = fetch_forecast_daily(req)
    except requests.exceptions.HTTPError as e:
        return {"error": f"16 天预报 API 请求失败: {http_error_message(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"网络请求异常: {e}"}
    report = format_forecast_daily_report(data, resolved["units"], "未来 16 天每日预报")
    return {"report": report, "raw": data}
