import os
import logging
from pathlib import Path

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from plugins_market.core.config import settings
from plugins_market.core.database import engine, DATABASE_URL
from plugins_market.core.errors import PublishError
from plugins_market.models.base import Base
from plugins_market.routers.register import router_register
from plugins_market.core.s3_storage_client import close_storage_client_if_initialized

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Store service entrypoint."""

    fastapi_app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @fastapi_app.exception_handler(PublishError)
    async def publish_error_handler(request: Request, exc: PublishError):
        """失败：HTTP 状态码 + {"detail": ...}（detail 为业务错误体）。"""
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @fastapi_app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        """失败：HTTP 状态码 + {"detail": ...}（detail 与 HTTPException.detail 一致，可为 str/dict/list）。"""
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @fastapi_app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """失败：422 + {"detail": 校验错误列表}。"""
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @fastapi_app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """失败：500 + {"detail": 错误说明}。"""
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc) or "服务器内部错误，请稍后重试"},
        )

    @fastapi_app.get("/api/health")
    async def health():
        return {"status": "ok"}

    router_register(fastapi_app)

    Base.metadata.create_all(bind=engine)

    @fastapi_app.on_event("shutdown")
    async def _shutdown_cleanup() -> None:
        # best-effort cleanup for long-lived clients / pools
        try:
            close_storage_client_if_initialized()
        except Exception as e:
            logger.warning("shutdown cleanup: failed to close storage client: %s", e)
        try:
            engine.dispose()
        except Exception as e:
            logger.warning("shutdown cleanup: failed to dispose db engine: %s", e)

    return fastapi_app


app = create_app()


def main() -> None:
    """Run service via `python main.py`."""
    host = os.getenv("STORE_HOST", settings.host)
    port = int(os.getenv("STORE_PORT", settings.port))
    workers = int(os.getenv("STORE_WORKERS", "1").strip() or "1")

    # uvicorn reload 与多进程 worker 不建议同时启用
    reload = bool(settings.debug)
    if reload:
        workers = 1

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()

