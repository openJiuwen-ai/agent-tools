"""获取当前时间工具，支持时区与格式化。"""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from openjiuwen.core.foundation.tool import tool

# 常用格式预设
FORMAT_PRESETS = {
    "iso": "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 风格
    "datetime": "%Y-%m-%d %H:%M:%S",  # 日期时间
    "date": "%Y-%m-%d",  # 仅日期
    "time": "%H:%M:%S",  # 仅时间
    "full": "%Y年%m月%d日 %H时%M分%S秒",  # 中文完整
}


def _get_now_in_tz(timezone: str) -> datetime:
    """获取指定时区的当前时间。"""
    tz = ZoneInfo(timezone) if timezone else timezone.utc  # 或本地时区
    return datetime.now(tz=tz)


def _format_dt(dt: datetime, fmt: str) -> str:
    """按预设名或 strftime 格式字符串格式化。"""
    pattern = FORMAT_PRESETS.get(fmt, fmt)
    return dt.strftime(pattern)


@tool(
    name="get_current_time",
    description=(
        "获取当前时间。可指定时区（如 Asia/Shanghai、UTC、America/New_York）和输出格式，"
        "返回格式化后的时间字符串。Get current time with optional timezone and format."
    ),
    input_params={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "时区名称（IANA），如 Asia/Shanghai、UTC、America/New_York、Europe/London。留空则使用本机本地时间。",
                "default": "UTC",
            },
            "format": {
                "type": "string",
                "description": "输出格式：预设 iso|datetime|date|time|full，或自定义 strftime 格式（如 %Y-%m-%d %H:%M:%S）。",
                "default": "datetime",
            },
        },
        "required": [],
    },
)
def get_current_time(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """返回指定时区、格式化后的当前时间。"""
    params = params or kwargs
    timezone = (params.get("timezone") or "UTC").strip() or "UTC"
    fmt = (params.get("format") or "datetime").strip() or "datetime"

    try:
        dt = _get_now_in_tz(timezone)
    except Exception as e:
        return {"error": f"无效的时区「{timezone}」: {e}"}

    try:
        formatted = _format_dt(dt, fmt)
    except Exception as e:
        return {"error": f"格式化失败: {e}"}

    return {
        "current_time": formatted,
        "timezone": timezone,
        "iso": dt.isoformat(),
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "minute": dt.minute,
        "second": dt.second,
        "weekday": dt.strftime("%A"),
    }
