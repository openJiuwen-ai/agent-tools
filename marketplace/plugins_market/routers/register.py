from fastapi import FastAPI

from plugins_market.routers import plugin as plugin_routers


def router_register(app: FastAPI) -> None:
    """注册所有路由。"""

    app.include_router(plugin_routers.router, prefix="/api/v1")

