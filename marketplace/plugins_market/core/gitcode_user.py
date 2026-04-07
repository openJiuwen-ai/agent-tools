"""GitCode Open API：使用 query 参数 access_token 拉取当前用户信息。"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from plugins_market.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_AUTH_USER_API_URL = "https://gitcode.com/api/v5/user"


async def fetch_gitcode_profile(access_token: str) -> dict[str, Any] | None:
    """
    GET settings.auth_user_api_url，params=access_token。
    成功返回 dict（id 为 str）；失败返回 None。
    """
    token = (access_token or "").strip()
    if not token:
        return None
    url = (settings.auth_user_api_url or DEFAULT_AUTH_USER_API_URL).strip()
    if not url:
        logger.warning("auth_user_api_url is not configured")
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(
                url,
                params={"access_token": token},
                headers={"Accept": "application/json"},
            )
        if res.status_code != 200:
            return None
        try:
            data = res.json()
        except ValueError:
            logger.warning("GitCode user API non-JSON: %s", res.text[:200])
            return None
        if not isinstance(data, dict):
            return None
        if data.get("error_code") is not None:
            return None
        gid = data.get("id")
        if gid is None:
            return None
        out = dict(data)
        out["id"] = str(gid).strip()
        if not out["id"]:
            return None
        return out
    except httpx.RequestError as e:
        logger.warning("fetch_gitcode_profile request error: %s", e)
        return None
    except Exception as e:
        logger.warning("fetch_gitcode_profile failed: %s", e)
        return None
