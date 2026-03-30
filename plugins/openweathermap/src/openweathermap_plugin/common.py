"""OpenWeatherMap 插件公共常量与工具函数。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

import requests

CURRENT_WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_5D_API_URL = "https://api.openweathermap.org/data/2.5/forecast"
FORECAST_16D_API_URL = "https://api.openweathermap.org/data/2.5/forecast/daily"
FORECAST_30D_API_URL = "https://pro.openweathermap.org/data/2.5/forecast/climate"


@dataclass(frozen=True)
class OpenWeatherCoords:
    """成对经纬度，与按城市查询互斥。"""

    lat: float
    lon: float


@dataclass(frozen=True)
class OpenWeatherQuery:
    """认证、单位/语言与地点（坐标或城市名二选一）。各类 weather / forecast 接口共用。"""

    api_key: str
    units: str = "metric"
    lang: str = "zh_cn"
    coords: OpenWeatherCoords | None = None
    city: str | None = None


@dataclass(frozen=True)
class ForecastDailyRequest:
    """多日每日预报（16 天 / 30 天气候）一次请求。"""

    url: str
    cnt: int
    query: OpenWeatherQuery


@dataclass(frozen=True)
class Forecast5dRequest:
    """5 天 / 3 小时步进预报一次请求。"""

    query: OpenWeatherQuery
    cnt: int | None = None


def open_weather_query_from_resolved(resolved: dict[str, Any]) -> OpenWeatherQuery:
    """由 resolve_location_and_common 成功结果构造 OpenWeatherQuery。"""
    fetch_kw = resolved["fetch_kw"]
    if "lat" in fetch_kw:
        return OpenWeatherQuery(
            api_key=resolved["api_key"],
            units=resolved["units"],
            lang=resolved["lang"],
            coords=OpenWeatherCoords(lat=fetch_kw["lat"], lon=fetch_kw["lon"]),
            city=None,
        )
    return OpenWeatherQuery(
        api_key=resolved["api_key"],
        units=resolved["units"],
        lang=resolved["lang"],
        coords=None,
        city=fetch_kw["city"],
    )


def get_api_key() -> str | None:
    """从环境变量读取 API Key。"""
    return os.getenv("openweathermap_api_key")


def resolve_location_and_common(params: dict) -> dict:
    """
    解析 lat/lon 或 city 及 units、lang。
    成功返回 {"api_key", "units", "lang", "fetch_kw"}，失败返回 {"error": "..."}。
    """
    api_key = get_api_key()
    if not api_key:
        return {
            "error": "未配置 OpenWeatherMap API Key。请在环境中设置 openweathermap_api_key 或 OPENWEATHERMAP_API_KEY。"
        }
    lat_val = params.get("lat")
    lon_val = params.get("lon")
    city = (params.get("city") or "").strip()
    if lat_val is not None and lon_val is not None:
        try:
            lat = float(lat_val)
            lon = float(lon_val)
        except (TypeError, ValueError):
            return {"error": "lat 与 lon 必须为数值类型。"}
        fetch_kw = {"lat": lat, "lon": lon}
    elif city:
        fetch_kw = {"city": city}
    else:
        return {"error": "请提供经纬度（lat、lon）或城市名称（city）至少一种方式。"}
    units = (params.get("units") or "metric").strip() or "metric"
    if units not in ("standard", "metric", "imperial"):
        units = "metric"
    lang = (params.get("lang") or "zh_cn").strip() or "zh_cn"
    return {"api_key": api_key, "units": units, "lang": lang, "fetch_kw": fetch_kw}


def fetch_forecast_daily(req: ForecastDailyRequest) -> dict:
    """调用多日预报接口（16 天或 30 天），支持经纬度或城市名。"""
    q = req.query
    params = {"appid": q.api_key, "cnt": req.cnt, "units": q.units, "lang": q.lang}
    if q.coords is not None:
        params["lat"] = q.coords.lat
        params["lon"] = q.coords.lon
    elif q.city:
        params["q"] = q.city
    else:
        raise ValueError("必须提供 lat/lon 或 city")
    resp = requests.get(req.url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def format_forecast_daily_report(data: dict, units: str, title: str) -> str:
    """将多日预报 list 整理为可读摘要。"""
    city_info = data.get("city") or {}
    name = city_info.get("name", "")
    country = city_info.get("country", "")
    location = f"{name}, {country}" if country else (name or "—")
    tz_sec = data.get("timezone", 0) or 0
    tz = timezone(timedelta(seconds=tz_sec))
    temp_unit = "℃" if units == "metric" else "℉" if units == "imperial" else "K"

    lines = [f"位置: {location}", f"预报: {title}", ""]
    for item in (data.get("list") or [])[:16]:
        dt = item.get("dt")
        date_str = datetime.fromtimestamp(dt, tz=tz).strftime("%Y-%m-%d %A") if dt else "—"
        temp = item.get("temp") or {}
        t_min = temp.get("min")
        t_max = temp.get("max")
        t_day = temp.get("day")
        temp_str = (
            f"{t_min}~{t_max}{temp_unit}"
            if t_min is not None and t_max is not None
            else (f"{t_day}{temp_unit}" if t_day is not None else "N/A")
        )
        weather_list = item.get("weather") or []
        desc = weather_list[0].get("description", "N/A") if weather_list else "N/A"
        pop = item.get("pop")
        pop_str = f" 降水概率 {int(pop * 100)}%" if pop is not None else ""
        lines.append(f"  {date_str}: {desc}, {temp_str}{pop_str}")
    return "\n".join(lines)


def http_error_message(e: requests.exceptions.HTTPError) -> str:
    """从 HTTPError 中提取可读错误信息。"""
    try:
        if e.response is not None:
            return e.response.json().get("message", e.response.text)
    except Exception:
        if e.response is not None:
            return getattr(e.response, "text", str(e))
    return str(e)


def fetch_forecast_5d(req: Forecast5dRequest) -> dict:
    """调用 5 天 / 3 小时步进预报接口，支持经纬度或城市名。"""
    q = req.query
    params = {"appid": q.api_key, "units": q.units, "lang": q.lang}
    if req.cnt is not None:
        params["cnt"] = req.cnt
    if q.coords is not None:
        params["lat"] = q.coords.lat
        params["lon"] = q.coords.lon
    elif q.city:
        params["q"] = q.city
    else:
        raise ValueError("必须提供 lat/lon 或 city")
    resp = requests.get(FORECAST_5D_API_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def format_forecast_5d_report(data: dict, units: str, title: str) -> str:
    """将 5 天 3 小时步进预报 list 整理为可读摘要。"""
    city_info = data.get("city") or {}
    name = city_info.get("name", "")
    country = city_info.get("country", "")
    location = f"{name}, {country}" if country else (name or "—")
    tz_sec = city_info.get("timezone", 0) or 0
    tz = timezone(timedelta(seconds=tz_sec))
    temp_unit = "℃" if units == "metric" else "℉" if units == "imperial" else "K"

    lines = [f"位置: {location}", f"预报: {title}", ""]
    for item in (data.get("list") or [])[:40]:
        dt = item.get("dt")
        if dt is not None:
            time_str = datetime.fromtimestamp(dt, tz=tz).strftime("%Y-%m-%d %H:%M")
        else:
            time_str = (item.get("dt_txt") or "").strip() or "—"
        main = item.get("main") or {}
        temp = main.get("temp")
        temp_str = f"{temp}{temp_unit}" if temp is not None else "N/A"
        weather_list = item.get("weather") or []
        desc = weather_list[0].get("description", "N/A") if weather_list else "N/A"
        pop = item.get("pop")
        pop_str = f" 降水 {int(pop * 100)}%" if pop is not None else ""
        lines.append(f"  {time_str}: {desc}, {temp_str}{pop_str}")
    return "\n".join(lines)
