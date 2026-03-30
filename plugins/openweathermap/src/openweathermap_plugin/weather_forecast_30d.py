"""30 天气候预报工具：OpenWeather Climatic forecast 30 days API（Pro）。"""

from typing import Any

import requests
from openjiuwen.core.foundation.tool import tool

from .common import (
    FORECAST_30D_API_URL,
    ForecastDailyRequest,
    fetch_forecast_daily,
    format_forecast_daily_report,
    http_error_message,
    open_weather_query_from_resolved,
    resolve_location_and_common,
)


@tool(
    name="openweathermap_forecast_30d",
    description=(
        "基于 OpenWeather Climatic forecast 30 days API 查询未来 1~30 天每日气候预报，"
        "支持经纬度或城市名。需 Pro 订阅。返回每日日温、天气描述等。"
        "Query 1-30 day climatic forecast by lat/lon or city. Requires Pro. See forecast30 docs."
    ),
    input_params={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "可选，纬度 -90~90。未提供时用 city。"},
            "lon": {"type": "number", "description": "可选，经度 -180~180。未提供时用 city。"},
            "city": {"type": "string", "description": "可选，城市英文名，如 Beijing、London。"},
            "cnt": {
                "type": "integer",
                "description": "返回天数，1~30，默认 7。",
                "minimum": 1,
                "maximum": 30,
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
def openweathermap_forecast_30d(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """调用 30 天气候预报接口（Pro），返回摘要与原始数据。"""
    params = params or kwargs
    resolved = resolve_location_and_common(params)
    if "error" in resolved:
        return resolved
    cnt = max(1, min(30, int(params.get("cnt", 7))))
    req = ForecastDailyRequest(
        url=FORECAST_30D_API_URL,
        cnt=cnt,
        query=open_weather_query_from_resolved(resolved),
    )
    try:
        data = fetch_forecast_daily(req)
    except requests.exceptions.HTTPError as e:
        return {"error": f"30 天气候预报 API 请求失败: {http_error_message(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"网络请求异常: {e}"}
    report = format_forecast_daily_report(data, resolved["units"], "未来 30 天气候预报")
    return {"report": report, "raw": data}
