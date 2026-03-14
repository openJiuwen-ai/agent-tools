import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from plugins_market.core.config import settings
from plugins_market.core.database import engine, DATABASE_URL
from plugins_market.models.base import Base
from plugins_market.routers.register import router_register


def create_app() -> FastAPI:
    """Store 服务入口。"""

    fastapi_app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/")
    async def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/api/docs",
            "health": "/api/health",
        }

    @fastapi_app.get("/api/health")
    async def health():
        return {"status": "ok"}

    router_register(fastapi_app)

    Base.metadata.create_all(bind=engine)

    return fastapi_app


app = create_app()


def main() -> None:
    """通过 `python main.py` 启动服务。"""
    host = os.getenv("STORE_HOST", settings.host)
    port = int(os.getenv("STORE_PORT", settings.port))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()

