"""鉴权：Authorization Bearer 与 X-System-Token 二选一，不能同时传也不能都不传。"""

import logging
from typing import Optional, Tuple, Any

import httpx
from fastapi import Header, HTTPException, status

from common.security.security_utils import SecurityUtils
from plugins_market.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_AUTH_USER_API_URL = "https://gitcode.com/api/v5/user"


def _has_bearer(authorization: Optional[str]) -> bool:
    return bool(
        authorization
        and authorization.strip().lower().startswith("bearer ")
        and authorization[7:].strip()
    )


def _has_system_token(x_system_token: Optional[str]) -> bool:
    return bool(x_system_token is not None and x_system_token.strip())


def _resolved_system_admin_token() -> str:
    return SecurityUtils.get_decrypt_secret("SYSTEM_ADMIN_TOKEN", default="") or ""


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not _has_bearer(authorization):
        return None
    return authorization[7:].strip()


async def _fetch_gitcode_profile(token: str) -> dict[str, Any] | None:
    """
    使用 GitCode 用户接口校验 token 并返回 profile。

    GitCode 当前项目约定：GET /api/v5/user?access_token=<token>
    """
    t = (token or "").strip()
    if not t:
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                (settings.auth_user_api_url or DEFAULT_AUTH_USER_API_URL).strip(),
                params={"access_token": t},
                headers={"Accept": "application/json"},
            )
    except Exception:
        return None

    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except Exception:
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


async def get_gitcode_user_id(token: str) -> str:
    """返回 GitCode 用户 id（字符串）。token 无效则抛 401。"""
    profile = await _fetch_gitcode_profile(token)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
        )
    return str(profile["id"]).strip()


async def require_auth(
    authorization: Optional[str] = Header(None),
    x_system_token: Optional[str] = Header(None, alias="X-System-Token"),
) -> Tuple[bool, Optional[str]]:
    """
    接口鉴权：Authorization 与 X-System-Token 二选一。
    - Bearer：使用 token 调用 GitCode /api/v5/user 校验；返回 (False, gitcode_user_id)。
    - X-System-Token：与 SYSTEM_ADMIN_TOKEN 比对；返回 (True, None)。
    """
    has_bearer = _has_bearer(authorization)
    has_system = _has_system_token(x_system_token)

    if has_bearer and has_system:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Cannot use both Authorization and X-System-Token; provide exactly one",
        )
    if not has_bearer and not has_system:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization: provide Authorization: Bearer <token> or X-System-Token (exactly one)",
        )

    if has_system:
        system_admin_token = _resolved_system_admin_token()
        if system_admin_token and x_system_token.strip() == system_admin_token:
            return (True, settings.system_admin_user)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-System-Token",
        )

    token = _extract_bearer_token(authorization) or ""
    gitcode_user_id = await get_gitcode_user_id(token)
    return (False, gitcode_user_id)
