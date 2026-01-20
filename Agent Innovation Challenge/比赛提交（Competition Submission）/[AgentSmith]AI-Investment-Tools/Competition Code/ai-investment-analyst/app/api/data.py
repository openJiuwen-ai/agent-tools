from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from app.models.analyst import LineItem
from app.models.data import CompanyNews, FinancialMetrics, InsiderTrade, Price
from app.services.data_fetching import (
    DataFetchingError,
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_prices,
)
from app.services.line_items import get_market_cap, search_line_items as fetch_line_items

router = APIRouter(prefix="/data", tags=["Data Fetching"])


@router.get("/prices", response_model=list[Price])
def fetch_prices(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., AAPL)"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
):
    """Fetch historical price data for a stock."""
    try:
        return get_prices(ticker, start_date, end_date)
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/financial-metrics", response_model=list[FinancialMetrics])
def fetch_financial_metrics(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., AAPL)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    period: str = Query("ttm", description="Reporting period: ttm, quarterly, or annual"),
    limit: int = Query(10, description="Number of records to fetch"),
):
    """Fetch financial metrics for a stock."""
    try:
        return get_financial_metrics(ticker, end_date, period, limit)
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insider-trades", response_model=list[InsiderTrade])
def fetch_insider_trades(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., AAPL)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD), optional"),
    limit: int = Query(1000, description="Records per page"),
):
    """Fetch insider trading data for a stock."""
    try:
        return get_insider_trades(ticker, end_date, start_date, limit)
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news", response_model=list[CompanyNews])
def fetch_company_news(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., AAPL)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD), optional"),
    limit: int = Query(1000, description="Records per page"),
):
    """Fetch company news for a stock."""
    try:
        return get_company_news(ticker, end_date, start_date, limit)
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/line-items", response_model=list[LineItem])
def fetch_line_items(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., AAPL)"),
    line_items: list[str] = Query(..., description="List of line item names to fetch"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    period: str = Query("ttm", description="Reporting period: ttm, quarterly, or annual"),
    limit: int = Query(10, description="Number of records to fetch"),
):
    """Fetch financial line items for a stock."""
    try:
        return fetch_line_items(ticker, line_items, end_date, period, limit)
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-cap")
def fetch_market_cap(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., AAPL)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
):
    """Fetch market cap for a stock."""
    try:
        market_cap = get_market_cap(ticker, end_date)
        return {"ticker": ticker, "market_cap": market_cap}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
