"""Google 图片搜索工具，基于 SerpAPI。"""

from itertools import islice
from typing import Any

from openjiuwen.core.foundation.tool import tool

from google_plugin._serpapi import (
    call_serpapi,
    error_no_api_key,
    get_api_key,
    validate_hl_gl,
)


def _parse_image_response(response: dict, max_results: int) -> dict:
    result = {}
    if "images_results" in response:
        result["images"] = [
            {
                "title": item.get("title", ""),
                "image": item.get("original", ""),
                "thumbnail": item.get("thumbnail", ""),
                "url": item.get("link", ""),
                "height": item.get("original_height", ""),
                "width": item.get("original_width", ""),
                "source": item.get("source", ""),
            }
            for item in islice(response["images_results"], max_results)
        ]
    return result


def _format_image_results_as_text(images: list[dict]) -> str:
    parts = []
    for item in images:
        img_url = item.get("image") or item.get("thumbnail") or ""
        title = item.get("title") or "Image"
        if img_url:
            parts.append(f"![{title}]({img_url})")
    return "\n\n".join(parts) if parts else ""


@tool(
    name="google_image_search",
    description=(
        "执行 Google 图片搜索并返回图片链接与缩略图。输入应为图片搜索关键词。"
        "A tool for Google image search, returning image URLs and thumbnails."
    ),
    input_params={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "图片搜索关键词 / Image search query.",
            },
            "hl": {
                "type": "string",
                "description": "搜索界面语言代码。详见 https://serpapi.com/google-languages",
                "default": "en",
            },
            "gl": {
                "type": "string",
                "description": "国家/地区代码。详见 https://serpapi.com/google-countries",
                "default": "us",
            },
            "location": {"type": "string", "description": "可选。搜索发起位置。"},
            "max_results": {
                "type": "integer",
                "description": "返回图片数量上限（1-20）",
                "minimum": 1,
                "maximum": 20,
                "default": 3,
            },
        },
        "required": ["query"],
    },
)
def google_image_search(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """调用 SerpAPI 执行 Google 图片搜索。"""
    params = params or kwargs
    query = (params.get("query") or "").strip()
    if not query:
        return {"error": "请输入图片搜索关键词。"}

    api_key = get_api_key()
    if not api_key:
        return error_no_api_key()

    hl = params.get("hl", "en")
    gl = params.get("gl", "us")
    location = (params.get("location") or "").strip()
    max_results = min(20, max(1, int(params.get("max_results", 3))))

    err = validate_hl_gl(hl, gl)
    if err:
        return err

    request_params = {
        "api_key": api_key,
        "q": query,
        "engine": "google_images",
        "gl": gl,
        "hl": hl,
    }
    if location:
        request_params["location"] = location

    data, error = call_serpapi(request_params, request_error_hint="调用图片搜索接口时出错")
    if error:
        return error

    parsed = _parse_image_response(data, max_results)
    images = parsed.get("images", [])
    report = _format_image_results_as_text(images)
    return {"report": report, "raw": parsed, "images": images}
