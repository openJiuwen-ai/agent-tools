import os
from typing import Any

import requests

from openjiuwen.core.foundation.tool import tool


def _build_search_headers(params: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json", "X-Respond-With": "no-content"}

    api_key = os.getenv("jina_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

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

    return headers


@tool(
    name="jina_search",
    description=(
        "针对给定查询在互联网上进行搜索，并返回适合大模型处理的结果文本。"
        "Search the public web via Jina Search and return LLM-friendly content."
    ),
    input_params={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索查询（问题/关键词）"},
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
        },
        "required": ["query"],
    },
)
def jina_search(params: dict[str, Any] | None = None, **kwargs) -> dict:
    params = params or kwargs
    query = (params.get("query") or "").strip()
    if not query:
        return {"error": "请提供 query。"}

    headers = _build_search_headers(params)
    endpoint = "https://s.jina.ai/"

    try:
        resp = requests.get(endpoint + query, headers=headers, timeout=(10, 60))
        return {"report": resp.text, "status_code": resp.status_code}
    except Exception as e:
        return {"error": f"Jina Search 请求失败: {str(e)}"}
