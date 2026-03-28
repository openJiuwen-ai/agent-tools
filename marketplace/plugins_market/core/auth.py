"""鉴权：Authorization Bearer 与 X-System-Token 二选一，不能同时传也不能都不传。"""

import logging
from typing import Optional, Tuple

import httpx
from fastapi import Header, HTTPException, Query, Request, status

from common.security.security_utils import SecurityUtils
from plugins_market.core.config import settings

logger = logging.getLogger(__name__)


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


async def verify_bearer_with_studio(token: str) -> bool:
    """调用 Studio 接口校验 access token。"""
    url = f"{settings.auth_service_base_url.rstrip('/')}/api/v1/auth/verify_access_token"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
    except Exception:
        return False
    if resp.status_code != 200:
        return False
    data = resp.json()
    if data.get("code") != 200:
        return False
    inner = data.get("data") or {}
    return inner.get("valid") is True


async def require_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_system_token: Optional[str] = Header(None, alias="X-System-Token"),
) -> None:
    """
    鉴权依赖：请求头中必须提供以下二者之一，且只能选其一：
    - Authorization: Bearer <token>，会调用 Studio 鉴权接口校验；
    - X-System-Token: <token>，与环境变量 SYSTEM_ADMIN_TOKEN 比较。
    同时传或同时不传均返回 401。
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
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-System-Token",
        )

    token = authorization[7:].strip()
    if await verify_bearer_with_studio(token):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired access token",
    )


async def verify_bearer_via_user(user_id: str, token: str) -> None:
    """
    调用 Studio GET /api/v1/users/{user_id} 校验 token 与 user_id 一致性。
    成功返回 None；否则根据响应抛出 401 或 403。
    """
    url = f"{settings.auth_service_base_url.rstrip('/')}/api/v1/users/{user_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        ) from e
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, dict) and data.get("code") == 200 and data.get("data"):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    if resp.status_code == 403:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    detail = "Invalid authentication credentials"
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("detail"):
            detail = body["detail"]
    except Exception as e:
        logger.debug("Failed to parse error response body: %s", e)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


async def require_auth_with_user_id(
    authorization: Optional[str] = Header(None),
    x_system_token: Optional[str] = Header(None, alias="X-System-Token"),
    user_id: Optional[str] = Query(None, description="普通用户必传，与 Authorization 一起使用"),
) -> Tuple[bool, Optional[str]]:
    """
    删除接口鉴权：Authorization 与 X-System-Token 二选一。
    - Bearer：必须传 query user_id，并请求 Studio GET /api/v1/users/{user_id} 校验 token 与 user_id 一致；返回 (False, user_id)。
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
            return (True, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-System-Token",
        )

    token = authorization[7:].strip()
    if not (user_id and user_id.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required when using Authorization",
        )
    await verify_bearer_via_user(user_id.strip(), token)
    return (False, user_id.strip())
