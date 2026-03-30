"""维基百科搜索工具，基于 wikipedia 库。"""

from typing import Any

import wikipedia

from openjiuwen.core.foundation.tool import tool

WIKIPEDIA_MAX_QUERY_LENGTH = 300


def _fetch_page(title: str):
    """获取维基百科页面，解析失败时返回 None。"""
    try:
        return wikipedia.page(title=title, auto_suggest=False)
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
        return None


def _run_search(query: str, lang: str = "en", top_k: int = 3, max_chars: int = 4000) -> str:
    """执行维基百科搜索并返回摘要文本。"""
    if lang and lang in wikipedia.languages():
        wikipedia.set_lang(lang)
    query = (query or "")[:WIKIPEDIA_MAX_QUERY_LENGTH].strip()
    if not query:
        return ""

    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[:top_k]:
        page = _fetch_page(page_title)
        if page and getattr(page, "summary", None):
            summaries.append(f"Page: {page_title}\nSummary: {page.summary}")

    if not summaries:
        return "No good Wikipedia Search Result was found"
    return ("\n\n".join(summaries))[:max_chars]


@tool(
    name="wikipedia_search",
    description=(
        "在维基百科中搜索并返回页面摘要。适用于人物、地点、公司、事实、历史事件等通用知识。输入应为搜索关键词。"
        "A tool for Wikipedia search and page summaries. Input should be a search query."
    ),
    input_params={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词 / Search query."},
            "language": {
                "type": "string",
                "description": (
                    "维基百科语言代码，如 en（英语）、zh（中文）、ja（日语）。"
                    "Language code: en, zh, ja, de, fr, ko, etc."
                ),
                "default": "en",
            },
            "top_k_results": {
                "type": "integer",
                "description": "返回的条目数量上限（1-10）",
                "minimum": 1,
                "maximum": 10,
                "default": 3,
            },
            "doc_content_chars_max": {
                "type": "integer",
                "description": "返回内容的最大字符数",
                "minimum": 500,
                "maximum": 8000,
                "default": 4000,
            },
        },
        "required": ["query"],
    },
)
def wikipedia_search(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """调用维基百科 API 进行搜索并返回摘要。"""
    params = params or kwargs
    query = (params.get("query") or "").strip()
    if not query:
        return {"error": "请输入搜索关键词。"}

    language = (params.get("language") or "en").strip() or "en"
    top_k = max(1, min(10, int(params.get("top_k_results", 3))))
    max_chars = max(500, min(8000, int(params.get("doc_content_chars_max", 4000))))

    try:
        report = _run_search(query, lang=language, top_k=top_k, max_chars=max_chars)
        if not report or report == "No good Wikipedia Search Result was found":
            return {"error": f"未找到与「{query}」相关的维基百科结果。", "report": report or ""}
        return {"report": report}
    except Exception as e:
        return {"error": f"维基百科搜索出错: {str(e)}"}
