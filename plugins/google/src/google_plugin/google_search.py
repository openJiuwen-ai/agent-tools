"""Google 网页搜索工具，基于 SerpAPI。"""

from typing import Any

from openjiuwen.core.foundation.tool import tool

from google_plugin._serpapi import (
    call_serpapi,
    error_no_api_key,
    get_api_key,
    validate_hl_gl,
)


def _parse_search_response(response: dict) -> dict:
    result = {}
    if "knowledge_graph" in response:
        result["title"] = response["knowledge_graph"].get("title", "")
        result["description"] = response["knowledge_graph"].get("description", "")
    if "organic_results" in response:
        result["organic_results"] = [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            for item in response["organic_results"]
        ]
    return result


def _format_search_results_as_text(data: dict) -> str:
    lines = []
    if data.get("title") or data.get("description"):
        lines.append(f"**{data.get('title', '')}**")
        if data.get("description"):
            lines.append(data["description"])
        lines.append("")
    for idx, item in enumerate(data.get("organic_results", []), 1):
        lines.append(f"# {idx}. [{item.get('title', '')}]({item.get('link', '')})")
        lines.append(f"**URL:** {item.get('link', '')}")
        if item.get("snippet"):
            lines.append(f"**摘要:** {item['snippet']}")
        lines.append("---")
    return "\n".join(lines)


@tool(
    name="google_search",
    description=(
        "执行 Google 网页搜索并返回片段与链接。适用于实时信息、事实核查。输入应为搜索关键词。"
        "A tool for Google SERP search, returning snippets and links."
    ),
    input_params={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词 / Search query."},
            "hl": {
                "type": "string",
                "description": "搜索界面语言代码，如 en、zh-cn。详见 https://serpapi.com/google-languages",
                "default": "en",
            },
            "gl": {
                "type": "string",
                "description": "搜索国家/地区代码，如 us、cn、uk。详见 https://serpapi.com/google-countries",
                "default": "us",
            },
            "location": {
                "type": "string",
                "description": "可选。搜索发起位置（城市等），用于模拟当地用户。",
            },
            "result_type": {
                "type": "string",
                "description": "Result type: 'link' for links only, or omit for full results.",
                "enum": ["link", "full"],
                "default": "full",
            },
        },
        "required": ["query"],
    },
)
def google_search(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """调用 SerpAPI 执行 Google 网页搜索。"""
    params = params or kwargs
    query = (params.get("query") or "").strip()
    if not query:
        return {"error": "请输入搜索关键词。"}

    api_key = get_api_key()
    if not api_key:
        return error_no_api_key()

    hl = params.get("hl", "en")
    gl = params.get("gl", "us")
    location = (params.get("location") or "").strip()

    err = validate_hl_gl(hl, gl)
    if err:
        return err

    request_params = {
        "api_key": api_key,
        "q": query,
        "engine": "google",
        "google_domain": "google.com",
        "gl": gl,
        "hl": hl,
    }
    if location:
        request_params["location"] = location

    data, error = call_serpapi(request_params, request_error_hint="调用搜索接口时出错")
    if error:
        return error

    parsed = _parse_search_response(data)
    report = _format_search_results_as_text(parsed)
    return {"report": report, "raw": parsed}
