"""SerpAPI 公共逻辑：API 地址、国家/语言校验、请求封装。"""

import json
import os
from typing import Any

import requests

SERP_API_URL = "https://serpapi.com/search"

_API_KEY_ENV = "serpapi_api_key"
_ERROR_NO_API_KEY = (
    "未配置 SerpAPI API Key。请在环境变量中设置 serpapi_api_key，或从 https://serpapi.com/manage-api-key 获取。"
)
REQUEST_TIMEOUT = 30


def _get_data_path(filename: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", filename)


def _load_valid_countries() -> set[str]:
    with open(_get_data_path("google-countries.json"), encoding="utf-8") as f:
        countries = json.load(f)
        return {c["country_code"] for c in countries}


def _load_valid_languages() -> set[str]:
    with open(_get_data_path("google-languages.json"), encoding="utf-8") as f:
        languages = json.load(f)
        return {lang["language_code"] for lang in languages}


VALID_COUNTRIES = _load_valid_countries()
VALID_LANGUAGES = _load_valid_languages()


def get_api_key() -> str | None:
    """从环境变量读取 SerpAPI API Key。"""
    return os.getenv(_API_KEY_ENV)


def error_no_api_key() -> dict[str, str]:
    """返回「未配置 API Key」的错误 dict，供各工具统一使用。"""
    return {"error": _ERROR_NO_API_KEY}


def validate_hl_gl(hl: str, gl: str) -> dict[str, str] | None:
    """
    校验 hl/gl 是否在有效列表中。无效时返回错误 dict，否则返回 None。
    """
    if hl not in VALID_LANGUAGES:
        return {"error": f"无效的语言代码 'hl': {hl}。请参考 https://serpapi.com/google-languages"}
    if gl not in VALID_COUNTRIES:
        return {"error": f"无效的国家代码 'gl': {gl}。请参考 https://serpapi.com/google-countries"}
    return None


def call_serpapi(
    request_params: dict[str, Any],
    timeout: int = REQUEST_TIMEOUT,
    request_error_hint: str = "调用接口时出错",
) -> tuple[dict | None, dict[str, str] | None]:
    """
    发起 SerpAPI 请求。成功返回 (response_json, None)，失败返回 (None, error_dict)。
    """
    try:
        response = requests.get(SERP_API_URL, params=request_params, timeout=timeout)
        response.raise_for_status()
        return (response.json(), None)
    except requests.exceptions.RequestException as e:
        return (
            None,
            {"error": f"{request_error_hint}: {str(e)}。可参考 https://serpapi.com/locations-api 查看有效位置列表。"},
        )
