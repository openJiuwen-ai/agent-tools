import json
import os
from typing import Any

import requests

from openjiuwen.core.foundation.tool import tool


def _build_reader_headers(params: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json"}

    api_key = os.getenv("jina_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    target_selector = (params.get("target_selector") or "").strip()
    if target_selector:
        headers["X-Target-Selector"] = target_selector
    wait_for_selector = (params.get("wait_for_selector") or "").strip()
    if wait_for_selector:
        headers["X-Wait-For-Selector"] = wait_for_selector

    if bool(params.get("remove_images", False)):
        headers["X-Retain-Images"] = "none"
    if bool(params.get("image_caption", False)):
        headers["X-With-Generated-Alt"] = "true"
    if bool(params.get("gather_all_links_at_the_end", False)):
        headers["X-With-Links-Summary"] = "true"
    if bool(params.get("gather_all_images_at_the_end", False)):
        headers["X-With-Images-Summary"] = "true"

    proxy_server = (params.get("proxy_server") or "").strip()
    if proxy_server:
        headers["X-Proxy-Url"] = proxy_server

    if bool(params.get("no_cache", False)):
        headers["X-No-Cache"] = "true"
    if bool(params.get("no_cache_track", False)):
        headers["DNT"] = "1"

    content_format = (params.get("content_format") or "default").strip()
    if content_format and content_format != "default":
        headers["X-Return-Format"] = content_format

    return headers


def _parse_request_params(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return None, "request_params 必须是 JSON 字符串或对象。"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"request_params 不是合法 JSON: {str(e)}"
    if not isinstance(parsed, dict):
        return None, 'request_params 必须是 JSON 对象，例如 {"key": "value"}。'
    return parsed, None


@tool(
    name="jina_reader",
    description=(
        "获取目标网址（可为 PDF/网页）并输出适合大模型处理的文本/Markdown。"
        "Fetch a URL (can be PDF/webpage) via Jina Reader and return LLM-friendly content."
    ),
    input_params={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "目标网页/PDF URL"},
            "request_params": {
                "type": "string",
                "description": '请求 querystring 参数（JSON 字符串），如 {"key":"value"}',
            },
            "target_selector": {"type": "string", "description": "CSS 选择器，仅抓取匹配元素"},
            "wait_for_selector": {"type": "string", "description": "等待某元素出现后再抓取"},
            "remove_images": {"type": "boolean", "description": "移除图片", "default": False},
            "image_caption": {"type": "boolean", "description": "为图片生成说明", "default": False},
            "gather_all_links_at_the_end": {
                "type": "boolean",
                "description": "在末尾汇总链接",
                "default": False,
            },
            "gather_all_images_at_the_end": {
                "type": "boolean",
                "description": "在末尾汇总图片",
                "default": False,
            },
            "proxy_server": {"type": "string", "description": "代理 URL"},
            "no_cache": {"type": "boolean", "description": "绕过缓存", "default": False},
            "no_cache_track": {"type": "boolean", "description": "不缓存/不追踪", "default": False},
            "content_format": {
                "type": "string",
                "description": "default/markdown/html/text/screenshot/pageshot",
                "enum": ["default", "markdown", "html", "text", "screenshot", "pageshot"],
                "default": "default",
            },
        },
        "required": ["url"],
    },
)
def jina_reader(params: dict[str, Any] | None = None, **kwargs) -> dict:
    params = params or kwargs
    url = (params.get("url") or "").strip()
    if not url:
        return {"error": "请提供 url。"}

    request_params, err = _parse_request_params(params.get("request_params"))
    if err:
        return {"error": err}

    headers = _build_reader_headers(params)
    endpoint = "https://r.jina.ai/"

    try:
        resp = requests.get(endpoint + url, headers=headers, params=request_params, timeout=(10, 60))
        return {"report": resp.text, "status_code": resp.status_code}
    except Exception as e:
        return {"error": f"Jina Reader 请求失败: {str(e)}"}
