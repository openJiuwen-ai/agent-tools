"""5 天 / 3 小时步进预报工具：OpenWeather 5 Day / 3 Hour Forecast API。"""

from typing import Any

import requests
from openjiuwen.core.foundation.tool import tool

from .common import (
    Forecast5dRequest,
    fetch_forecast_5d,
    format_forecast_5d_report,
    http_error_message,
    open_weather_query_from_resolved,
    resolve_location_and_common,
)


@tool(
    name="openweathermap_forecast_5d",
    description=(
        "基于 OpenWeather 5 Day / 3 Hour Forecast API 查询未来 1~5 天（每 3 小时一档）的预报，"
        "支持经纬度或城市名。返回每档温度、天气描述、降水概率等。"
        "Query 1-5 day forecast with 3-hour step by lat/lon or city. See forecast5 docs."
    ),
    input_params={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "可选，纬度 -90~90。未提供时用 city。"},
            "lon": {"type": "number", "description": "可选，经度 -180~180。未提供时用 city。"},
            "city": {"type": "string", "description": "可选，城市英文名，如 Beijing、London。"},
            "cnt": {
                "type": "integer",
                "description": "可选，返回的时间点数量，不传则返回完整 5 天约 40 档。",
                "minimum": 1,
                "maximum": 40,
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
def openweathermap_forecast_5d(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """调用 5 天 / 3 小时步进预报接口，返回摘要与原始数据。"""
    params = params or kwargs
    resolved = resolve_location_and_common(params)
    if "error" in resolved:
        return resolved
    cnt_val = params.get("cnt")
    cnt = None
    if cnt_val is not None:
        try:
            cnt = max(1, min(40, int(cnt_val)))
        except (TypeError, ValueError):
            pass
    req = Forecast5dRequest(query=open_weather_query_from_resolved(resolved), cnt=cnt)
    try:
        data = fetch_forecast_5d(req)
    except requests.exceptions.HTTPError as e:
        return {"error": f"5 天预报 API 请求失败: {http_error_message(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"网络请求异常: {e}"}
    report = format_forecast_5d_report(data, resolved["units"], "未来 5 天（每 3 小时一档）")
    return {"report": report, "raw": data}
