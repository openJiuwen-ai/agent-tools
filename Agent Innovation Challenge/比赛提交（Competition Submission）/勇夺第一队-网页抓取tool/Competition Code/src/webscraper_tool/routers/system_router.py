import datetime

from fastapi import APIRouter

system_router = APIRouter()


@system_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "WebScraperTool",
        "version": "0.1.0",
        "timestamp": datetime.datetime.now().isoformat(),
    }
