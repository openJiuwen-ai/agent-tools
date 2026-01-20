from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analysts import router as analysts_router
from app.api.data import router as data_router
from app.api.decision import router as decision_router

app = FastAPI(
    title="AI Investment Analyst API",
    description="Investment analysis service powered by AI agents",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data_router)
app.include_router(analysts_router)
app.include_router(decision_router)


@app.get("/")
def read_root():
    return {"message": "AI Investment Analyst API", "version": "0.1.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8123)
