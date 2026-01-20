import os
import time

import requests

from app.models.data import CompanyNews, FinancialMetrics, InsiderTrade, Price


class DataFetchingError(Exception):
    """Data fetching error."""
    pass


def _make_api_request(
    url: str,
    headers: dict,
    method: str = "GET",
    json_data: dict | None = None,
    max_retries: int = 3,
) -> requests.Response:
    """Make an API request with rate limiting handling."""
    for attempt in range(max_retries + 1):
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)

        if response.status_code == 429 and attempt < max_retries:
            delay = 60 + (30 * attempt)
            time.sleep(delay)
            continue

        return response
    return requests.get(url, headers=headers)


def _get_api_key() -> str | None:
    """Get API key from environment."""
    return os.environ.get("FINANCIAL_DATASETS_API_KEY")


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from API.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of Price objects
    """
    headers = {}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-KEY"] = api_key

    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}&interval=day&interval_multiplier=1"
        f"&start_date={start_date}&end_date={end_date}"
    )

    response = _make_api_request(url, headers)
    if response.status_code != 200:
        raise DataFetchingError(f"API error: {response.status_code} - {response.text}")

    try:
        data = response.json()
        return [Price(**price) for price in data.get("prices", [])]
    except Exception as e:
        raise DataFetchingError(f"Parse error: {e}") from e


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from API.

    Args:
        ticker: Stock ticker symbol
        end_date: End date (YYYY-MM-DD)
        period: Reporting period (ttm, quarterly, annual)
        limit: Number of records to fetch

    Returns:
        List of FinancialMetrics objects
    """
    headers = {}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-KEY"] = api_key

    url = (
        f"https://api.financialdatasets.ai/financial-metrics/"
        f"?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    )

    response = _make_api_request(url, headers)
    if response.status_code != 200:
        raise DataFetchingError(f"API error: {response.status_code} - {response.text}")

    try:
        data = response.json()
        return [FinancialMetrics(**m) for m in data.get("financial_metrics", [])]
    except Exception as e:
        raise DataFetchingError(f"Parse error: {e}") from e


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from API.

    Args:
        ticker: Stock ticker symbol
        end_date: End date (YYYY-MM-DD)
        start_date: Start date (YYYY-MM-DD), optional
        limit: Records per page

    Returns:
        List of InsiderTrade objects
    """
    headers = {}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            insider_trades = [InsiderTrade(**t) for t in data.get("insider_trades", [])]
        except Exception:
            break

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        if not start_date or len(insider_trades) < limit:
            break

        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]
        if current_end_date <= start_date:
            break

    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from API.

    Args:
        ticker: Stock ticker symbol
        end_date: End date (YYYY-MM-DD)
        start_date: Start date (YYYY-MM-DD), optional
        limit: Records per page

    Returns:
        List of CompanyNews objects
    """
    headers = {}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            company_news = [CompanyNews(**n) for n in data.get("news", [])]
        except Exception:
            break

        if not company_news:
            break

        all_news.extend(company_news)

        if not start_date or len(company_news) < limit:
            break

        current_end_date = min(news.date for news in company_news).split("T")[0]
        if current_end_date <= start_date:
            break

    return all_news
