"""站点 Map：firecrawl-py V2 `Firecrawl.map`。"""

import json
from typing import Any

from openjiuwen.core.foundation.tool import tool

from firecrawl_plugin._config import get_firecrawl_client
from firecrawl_plugin._serialize import firecrawl_to_plain


@tool(
    name="firecrawl_map",
    description=(
        "输入站点根 URL，快速列出站内 URL（Firecrawl Map）。"
        "使用 Firecrawl API V2：`Firecrawl.map`。"
    ),
    input_params={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "起始 URL（站点根或入口页）"},
            "search": {
                "type": "string",
                "description": "可选，映射时的搜索关键词",
            },
            "ignoreSitemap": {
                "type": "boolean",
                "description": "忽略 sitemap（V2：`sitemap=skip`）",
                "default": True,
            },
            "includeSubdomains": {
                "type": "boolean",
                "description": "是否包含子域名",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "最大返回条数",
                "default": 5000,
                "minimum": 0,
            },
        },
        "required": ["url"],
    },
)
def firecrawl_map(params: dict[str, Any] | None = None, **kwargs) -> dict:
    params = params or kwargs
    url = (params.get("url") or "").strip()
    if not url:
        return {"error": "请提供 url。"}

    client, err = get_firecrawl_client()
    if err:
        return {"error": err}
    if client is None:
        return {"error": "无法创建 Firecrawl 客户端。"}

    sitemap: str = "skip" if params.get("ignoreSitemap", True) else "include"
    kw: dict[str, Any] = {
        "include_subdomains": params.get("includeSubdomains", False),
        "limit": params.get("limit", 5000),
        "sitemap": sitemap,
    }
    if params.get("search") not in (None, ""):
        kw["search"] = params.get("search")

    try:
        map_result = client.map(url, **kw)
    except Exception as e:
        return {"error": f"Firecrawl map 失败: {e!s}"}

    plain = firecrawl_to_plain(map_result)
    report = json.dumps(plain, ensure_ascii=False, indent=2)
    return {"report": report, "raw": plain}
