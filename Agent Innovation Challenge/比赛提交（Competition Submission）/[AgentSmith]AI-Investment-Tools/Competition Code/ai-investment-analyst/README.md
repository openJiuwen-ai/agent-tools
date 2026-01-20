# AI Investment Analyst

RESTful API service for AI-powered investment analysis.

## Setup

```bash
# Install uv
pip install uv

# Create virtual environment and install dependencies
uv sync

# Copy environment file
cp .env.example .env

# Edit .env with your API keys
```

## Run

```bash
# Development
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Test

```bash
uv run pytest
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
ai-investment-analyst/
├── app/
│   ├── main.py          # FastAPI application
│   ├── api/             # API routes
│   ├── core/            # Core configuration
│   ├── models/          # Pydantic models
│   └── services/        # Business logic
├── tests/               # Tests
├── pyproject.toml       # Dependencies
└── README.md
```
