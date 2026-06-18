from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from openjiuwentools.infer_router.config.config import settings

security = HTTPBearer()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = security):
    """验证API密钥"""
    if not settings.enable_auth:
        return True

    if not settings.api_key:
        logger.error("API authentication enabled but no API key configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )

    if credentials.credentials != settings.api_key:
        logger.warning(f"Invalid API key: {credentials.credentials}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


async def api_key_auth_middleware(request: Request, call_next):
    """API密钥认证中间件"""
    if not settings.enable_auth:
        return await call_next(request)

    # 跳过健康检查接口的认证
    if request.url.path == "/health":
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")

    if not api_key:
        # 尝试从Authorization头获取
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]

    if not api_key or api_key != settings.api_key:
        logger.warning(f"Unauthorized request to {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": {"message": "Invalid API key", "type": "unauthorized"}},
        )

    return await call_next(request)
