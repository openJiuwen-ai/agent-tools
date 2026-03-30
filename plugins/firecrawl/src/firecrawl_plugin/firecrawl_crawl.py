"""深度爬取：firecrawl-py V2 `crawl`（同步等待直至完成）。"""

import json
from typing import Any

from firecrawl.types import ScrapeOptions
from openjiuwen.core.foundation.tool import tool

from firecrawl_plugin._config import get_firecrawl_client
from firecrawl_plugin._serialize import firecrawl_to_plain
from firecrawl_plugin.param_utils import get_array_params, get_json_params
from firecrawl_plugin.v2_formats import build_formats_with_extract


def _build_scrape_options(params: dict[str, Any]) -> ScrapeOptions | None:
    fmts = build_formats_with_extract(get_array_params(params, "formats"))
    hdrs = get_json_params(params, "headers")
    it = get_array_params(params, "includeTags")
    et = get_array_params(params, "excludeTags")
    kw: dict[str, Any] = {}
    if fmts:
        kw["formats"] = fmts
    if hdrs is not None:
        kw["headers"] = hdrs
    if it:
        kw["include_tags"] = it
    if et:
        kw["exclude_tags"] = et
    omc = params.get("onlyMainContent")
    if omc is not None:
        kw["only_main_content"] = bool(omc)
    wf = params.get("waitFor", 0)
    if wf not in (None, 0, ""):
        kw["wait_for"] = int(wf)
    if not kw:
        return None
    return ScrapeOptions(**kw)


@tool(
    name="firecrawl_crawl",
    description=(
        "从起始 URL 递归爬取站点并收集页面内容（Firecrawl Crawl）。"
        "V2：仅使用 `crawl`，在服务端轮询直至任务完成后再返回结果。"
    ),
    input_params={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "爬取起始 URL"},
            "poll_interval": {
                "type": "integer",
                "description": "轮询任务状态的间隔（秒）",
                "default": 5,
                "minimum": 1,
            },
            "excludePaths": {
                "type": "string",
                "description": "排除的 URL 模式，逗号分隔，如 blog/*,about/*",
            },
            "includePaths": {
                "type": "string",
                "description": "仅包含的 URL 模式，逗号分隔",
            },
            "maxDepth": {
                "type": "integer",
                "description": "发现链接的最大深度（对应 V2 `max_discovery_depth`）",
                "default": 2,
                "minimum": 0,
            },
            "ignoreSitemap": {
                "type": "boolean",
                "description": "忽略 sitemap（V2：`sitemap=skip`）",
                "default": True,
            },
            "limit": {
                "type": "integer",
                "description": "最大页面数",
                "default": 5,
                "minimum": 1,
            },
            "allowBackwardLinks": {
                "type": "boolean",
                "description": "允许向上/整域爬取（V2：`crawl_entire_domain`）",
                "default": False,
            },
            "allowExternalLinks": {
                "type": "boolean",
                "description": "允许跟随外链",
                "default": False,
            },
            "webhook": {"type": "string", "description": "Webhook URL"},
            "formats": {
                "type": "string",
                "description": "子页面 scrape 输出格式，逗号分隔：markdown, html, rawHtml, links, screenshot 等",
            },
            "headers": {
                "type": "string",
                "description": "scrape 请求头 JSON 字符串",
            },
            "includeTags": {"type": "string", "description": "仅包含标签，逗号分隔"},
            "excludeTags": {"type": "string", "description": "排除标签，逗号分隔"},
            "onlyMainContent": {
                "type": "boolean",
                "description": "每页仅主要内容",
                "default": False,
            },
            "waitFor": {
                "type": "integer",
                "description": "每页加载等待毫秒数",
                "default": 0,
                "minimum": 0,
            },
        },
        "required": ["url"],
    },
)
def firecrawl_crawl(params: dict[str, Any] | None = None, **kwargs) -> dict:
    params = params or kwargs
    url = (params.get("url") or "").strip()
    if not url:
        return {"error": "请提供 url。"}

    client, err = get_firecrawl_client()
    if err:
        return {"error": err}
    if client is None:
        return {"error": "无法创建 Firecrawl 客户端。"}

    poll_interval = max(1, int(params.get("poll_interval", 5) or 5))

    excl = get_array_params(params, "excludePaths")
    incl = get_array_params(params, "includePaths")
    scrape_options = _build_scrape_options(params)

    sitemap: str = "skip" if params.get("ignoreSitemap", True) else "include"

    crawl_kw: dict[str, Any] = {
        "max_discovery_depth": int(params.get("maxDepth", 2)),
        "sitemap": sitemap,
        "limit": int(params.get("limit", 5)),
        "crawl_entire_domain": bool(params.get("allowBackwardLinks", False)),
        "allow_external_links": bool(params.get("allowExternalLinks", False)),
        "scrape_options": scrape_options,
    }
    if excl:
        crawl_kw["exclude_paths"] = excl
    if incl:
        crawl_kw["include_paths"] = incl
    wh = (params.get("webhook") or "").strip()
    if wh:
        crawl_kw["webhook"] = wh

    try:
        crawl_result = client.crawl(url, poll_interval=poll_interval, **crawl_kw)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Firecrawl crawl 失败: {e!s}"}

    plain = firecrawl_to_plain(crawl_result)
    report = json.dumps(plain, ensure_ascii=False, indent=2)
    return {"report": report, "raw": plain}
