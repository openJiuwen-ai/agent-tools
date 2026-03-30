"""从环境变量创建官方 firecrawl-py 客户端（统一入口，使用 V2 API）。"""

import os

from firecrawl import Firecrawl

DEFAULT_API_URL = "https://api.firecrawl.dev"


def get_firecrawl_client() -> tuple[Firecrawl | None, str | None]:
    """返回 (Firecrawl 实例, 错误信息)。与 SDK 一致：仅访问官方云时必须提供 API Key。"""
    api_key = os.getenv("firecrawl_api_key")
    api_url_raw = os.getenv("firecrawl_base_url") or DEFAULT_API_URL
    api_url = str(api_url_raw).strip().rstrip("/") or DEFAULT_API_URL

    if "api.firecrawl.dev" in api_url and (not api_key or not str(api_key).strip()):
        return None, (
            "缺少 Firecrawl API Key。请设置 firecrawl_api_key。"
            "自托管时请设置 firecrawl_base_url 指向你的实例，并可使用任意非空 key（若你的实例不校验）。"
        )

    key = str(api_key).strip() if api_key else ""
    return Firecrawl(api_key=key or None, api_url=api_url), None
