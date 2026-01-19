from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.scrape_router import scrape_router
from .routers.system_router import system_router

app = FastAPI(
    title="WebScraperTool",
    description="General-purpose web scraping tool server (for OpenJiuwen Agent Studio URL plugin).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router, prefix="/system", tags=["system"])
app.include_router(scrape_router, prefix="/scrape", tags=["scrape"])
