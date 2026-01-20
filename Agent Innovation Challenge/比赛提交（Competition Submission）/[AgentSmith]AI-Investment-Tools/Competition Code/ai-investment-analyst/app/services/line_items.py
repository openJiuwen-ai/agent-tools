import os
import time

import requests

from app.models.analyst import LineItem


class DataFetchingError(Exception):
    """Data fetching error."""
    pass


def _make_api_request(
    url: str,
    headers: dict,
    method: str = "GET",
    json_data: dict | None = None,
) -> requests.Response:
    """Make an API request."""
    if method.upper() == "POST":
        return requests.post(url, headers=headers, json=json_data)
    return requests.get(url, headers=headers)


def _get_api_key() -> str | None:
    """Get API key from environment."""
    return os.environ.get("FINANCIAL_DATASETS_API_KEY")


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch financial line items from API.

    Args:
        ticker: Stock ticker symbol
        line_items: List of line item names to fetch
        end_date: End date (YYYY-MM-DD)
        period: Reporting period (ttm, quarterly, annual)
        limit: Number of records to fetch

    Returns:
        List of LineItem objects
    """
    headers = {}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }

    response = _make_api_request(url, headers, method="POST", json_data=body)
    if response.status_code != 200:
        raise DataFetchingError(f"API error: {response.status_code} - {response.text}")

    try:
        data = response.json()
        return [LineItem(**item) for item in data.get("search_results", [])]
    except Exception as e:
        raise DataFetchingError(f"Parse error: {e}") from e


def get_market_cap(ticker: str, end_date: str) -> float | None:
    """Fetch market cap from API.

    Args:
        ticker: Stock ticker symbol
        end_date: End date (YYYY-MM-DD)

    Returns:
        Market cap value or None
    """
    headers = {}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"

    response = _make_api_request(url, headers)
    if response.status_code != 200:
        return None

    try:
        data = response.json()
        return data.get("company_facts", {}).get("market_cap")
    except Exception:
        return None
