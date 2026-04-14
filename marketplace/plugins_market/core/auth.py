"""鉴权：Authorization Bearer 与 X-System-Token 二选一，不能同时传也不能都不传。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, status

from common.security.security_utils import SecurityUtils
from plugins_market.core.config import settings
from plugins_market.core.gitcode_user import fetch_gitcode_profile

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AuthContext:
    """`require_auth` 成功后的调用方身份；两种路径下 `acting_user_id` 均为非空字符串。

    - `is_admin=True`：合法 `X-System-Token`，`acting_user_id` 为 `settings.system_admin_user`。
    - `is_admin=False`：合法 Bearer，`acting_user_id` 为 GitCode 用户 id。
    """

    is_admin: bool
    acting_user_id: str


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


async def get_gitcode_user_id_and_login(token: str) -> tuple[str, str]:
    """返回 (GitCode 用户 id, 展示用发布者名)。

    发布者名与前端 Skill 打包逻辑一致：优先 ``login``，其次 ``username``，否则回退为 id。
    token 无效则抛 401。
    """
    profile = await fetch_gitcode_profile(token)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
        )
    gid = str(profile["id"]).strip()
    if not gid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
        )
    login = (profile.get("login") or profile.get("username") or "").strip() or gid
    return gid, login


async def get_gitcode_user_id(token: str) -> str:
    """返回 GitCode 用户 id（字符串）。token 无效则抛 401。"""
    uid, _ = await get_gitcode_user_id_and_login(token)
    return uid


async def require_auth(
    authorization: Optional[str] = Header(None),
    x_system_token: Optional[str] = Header(None, alias="X-System-Token"),
) -> AuthContext:
    """
    接口鉴权：Authorization 与 X-System-Token 二选一。
    - Bearer：GitCode `/api/v5/user` 校验成功后，`AuthContext(False, gitcode_user_id)`。
    - X-System-Token：与 `SYSTEM_ADMIN_TOKEN` 比对成功后，`AuthContext(True, settings.system_admin_user)`。
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
            return AuthContext(is_admin=True, acting_user_id=settings.system_admin_user)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-System-Token",
        )

    token = _extract_bearer_token(authorization) or ""
    gitcode_user_id = await get_gitcode_user_id(token)
    return AuthContext(is_admin=False, acting_user_id=gitcode_user_id)
