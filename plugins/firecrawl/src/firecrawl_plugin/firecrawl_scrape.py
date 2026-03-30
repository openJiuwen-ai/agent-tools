"""单页抓取：firecrawl-py V2 `Firecrawl.scrape`。"""

from typing import Any

from openjiuwen.core.foundation.tool import tool

from firecrawl_plugin._config import get_firecrawl_client
from firecrawl_plugin._serialize import firecrawl_to_plain
from firecrawl_plugin.param_utils import get_array_params, get_json_params
from firecrawl_plugin.v2_formats import build_formats_with_extract


@tool(
    name="firecrawl_scrape",
    description=(
        "将指定 URL 转为干净、可供大模型使用的数据（默认含 Markdown）。使用 Firecrawl API V2：`Firecrawl.scrape`。"
    ),
    input_params={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的页面 URL"},
            "formats": {
                "type": "string",
                "description": (
                    "输出格式，逗号分隔：markdown, html, rawHtml, links, screenshot, "
                    "extract, json, screenshot@fullPage 等"
                ),
            },
            "onlyMainContent": {
                "type": "boolean",
                "description": "仅返回正文，排除页眉页脚导航等（默认 true）",
                "default": True,
            },
            "includeTags": {
                "type": "string",
                "description": "仅保留的标签/类/id，逗号分隔，如 script, .ad, #footer",
            },
            "excludeTags": {
                "type": "string",
                "description": "要排除的标签/类/id，逗号分隔",
            },
            "headers": {
                "type": "string",
                "description": '请求头 JSON 字符串，如 {"Cookie": "..."}',
            },
            "waitFor": {
                "type": "integer",
                "description": "页面加载等待毫秒数",
                "default": 0,
                "minimum": 0,
            },
            "timeout": {
                "type": "integer",
                "description": "请求超时（毫秒）",
                "default": 30000,
                "minimum": 0,
            },
            "schema": {
                "type": "string",
                "description": "结构化提取 JSON Schema（JSON 字符串）；V2 对应 json format",
            },
            "systemPrompt": {"type": "string", "description": "提取用的 system prompt（V2 json format）"},
            "prompt": {"type": "string", "description": "无 schema 时用于提取的 prompt（V2 json format）"},
        },
        "required": ["url"],
    },
)
def firecrawl_scrape(params: dict[str, Any] | None = None, **kwargs) -> dict:
    params = params or kwargs
    url = (params.get("url") or "").strip()
    if not url:
        return {"error": "请提供 url。"}

    client, err = get_firecrawl_client()
    if err:
        return {"error": err}
    if client is None:
        return {"error": "无法创建 Firecrawl 客户端。"}

    try:
        sch = get_json_params(params, "schema")
        formats_list = build_formats_with_extract(
            get_array_params(params, "formats"),
            schema=sch,
            prompt=params.get("prompt"),
            system_prompt=params.get("systemPrompt"),
        )
        if not formats_list:
            formats_list = ["markdown"]

        scrape_kw: dict[str, Any] = {
            "formats": formats_list,
            "only_main_content": params.get("onlyMainContent", True),
        }
        inc = get_array_params(params, "includeTags")
        if inc:
            scrape_kw["include_tags"] = inc
        exc = get_array_params(params, "excludeTags")
        if exc:
            scrape_kw["exclude_tags"] = exc
        hdrs = get_json_params(params, "headers")
        if hdrs is not None:
            scrape_kw["headers"] = hdrs
        wf = params.get("waitFor", 0)
        if wf not in (None, 0, ""):
            scrape_kw["wait_for"] = int(wf)
        to = params.get("timeout", 30000)
        if to is not None:
            scrape_kw["timeout"] = int(to)

        resp = client.scrape(url, **scrape_kw)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Firecrawl scrape 失败: {e!s}"}

    plain = firecrawl_to_plain(resp)
    markdown_result = (plain or {}).get("markdown") or "" if isinstance(plain, dict) else ""
    if not markdown_result and hasattr(resp, "markdown") and resp.markdown:
        markdown_result = resp.markdown
    return {"report": markdown_result, "raw": plain}
