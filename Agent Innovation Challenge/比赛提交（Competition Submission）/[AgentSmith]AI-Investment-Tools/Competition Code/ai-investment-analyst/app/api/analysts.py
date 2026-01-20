from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.models.analyst import LineItem
from app.models.data import FinancialMetrics, Price, CompanyNews, InsiderTrade
from app.services.data_fetching import (
    DataFetchingError,
    get_financial_metrics,
    get_prices,
    get_company_news,
    get_insider_trades,
)
from app.services.line_items import get_market_cap, search_line_items

# 投资风格分析师服务
from app.services.value_investor_analysis import generate_value_investor_prompt
from app.services.deep_value_analysis import generate_deep_value_prompt
from app.services.growth_innovation_analysis import generate_growth_innovation_prompt
from app.services.growth_value_analysis import generate_growth_value_prompt
from app.services.quality_growth_analysis import generate_quality_growth_prompt
from app.services.rational_value_analysis import generate_rational_value_prompt
from app.services.contrarian_analysis import generate_contrarian_prompt
from app.services.macro_trend_analysis import generate_macro_trend_prompt
from app.services.valuation_expert_analysis import generate_valuation_expert_prompt
from app.services.activist_analysis import generate_activist_prompt
from app.services.clone_investor_analysis import generate_clone_investor_prompt
from app.services.emerging_growth_analysis import generate_emerging_growth_prompt

# 技术分析服务
from app.services.technical_analysis import generate_technical_prompt
from app.services.fundamentals_analysis import generate_fundamentals_prompt
from app.services.growth_analysis import generate_growth_prompt
from app.services.valuation_analysis import generate_valuation_prompt
from app.services.sentiment_analysis import (
    generate_news_sentiment_prompt,
    generate_market_sentiment_prompt,
)

router = APIRouter(prefix="/analysts", tags=["Analysts"])


def _get_common_data(ticker: str, end_date: str, period: str = "ttm", limit: int = 10):
    """获取通用财务数据。"""
    try:
        metrics = get_financial_metrics(ticker, end_date, period=period, limit=limit)
        line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization",
                "net_income",
                "outstanding_shares",
                "total_assets",
                "total_liabilities",
                "shareholders_equity",
                "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares",
                "gross_profit",
                "revenue",
                "free_cash_flow",
                "research_and_development",
                "operating_expense",
                "current_assets",
                "current_liabilities",
                "cash_and_equivalents",
                "book_value_per_share",
                "earnings_per_share",
                "ebitda",
                "inventory_turnover",
                "return_on_invested_capital",
                "debt_to_equity",
                "interest_coverage",
            ],
            end_date,
            period=period,
            limit=limit,
        )
        market_cap = get_market_cap(ticker, end_date)
        return metrics, line_items, market_cap
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 投资风格分析师 ====================

@router.get("/value-investor/prompt")
def value_investor_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """价值投资类分析师 - 巴菲特风格"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_value_investor_prompt(ticker, metrics, line_items, market_cap)


@router.get("/deep-value/prompt")
def deep_value_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """深度价值投资类分析师 - 格雷厄姆风格"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_deep_value_prompt(ticker, metrics, line_items, market_cap)


@router.get("/growth-innovation/prompt")
def growth_innovation_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """增长创新投资类分析师 - 专注颠覆性技术"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_growth_innovation_prompt(ticker, metrics, line_items, market_cap)


@router.get("/growth-value/prompt")
def growth_value_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """成长价值投资类分析师 - Peter Lynch风格"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date, period="annual")
    return generate_growth_value_prompt(ticker, metrics, line_items, market_cap)


@router.get("/quality-growth/prompt")
def quality_growth_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """质量成长投资类分析师 - Phil Fisher风格"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_quality_growth_prompt(ticker, metrics, line_items)


@router.get("/rational-value/prompt")
def rational_value_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """理性价值投资类分析师 - Charlie Munger风格"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_rational_value_prompt(ticker, metrics, line_items, market_cap)


@router.get("/contrarian/prompt")
def contrarian_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """逆向投资类分析师 - Michael Burry风格"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_contrarian_prompt(ticker, metrics, line_items, market_cap)


@router.get("/macro-trend/prompt")
def macro_trend_endpoint(
    ticker: str = Query(..., description="股票代码"),
    start_date: str = Query(..., description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """宏观趋势投资类分析师 - Druckenmiller风格"""
    try:
        prices = get_prices(ticker, start_date, end_date)
        return generate_macro_trend_prompt(ticker, prices)
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/valuation-expert/prompt")
def valuation_expert_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """估值专家类分析师 - Damodaran风格"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_valuation_expert_prompt(ticker, metrics, line_items, market_cap)


@router.get("/activist/prompt")
def activist_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """激进投资类分析师 - Ackman风格"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_activist_prompt(ticker, metrics, line_items, market_cap)


@router.get("/clone-investor/prompt")
def clone_investor_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """克隆投资类分析师 - Pabrai风格"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_clone_investor_prompt(ticker, metrics, line_items, market_cap)


@router.get("/emerging-growth/prompt")
def emerging_growth_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """新兴市场增长投资类分析师 - Jhunjhunwala风格"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_emerging_growth_prompt(ticker, metrics, line_items)


# ==================== 技术分析师 ====================

@router.get("/technical/prompt")
def technical_endpoint(
    ticker: str = Query(..., description="股票代码"),
    start_date: str = Query(..., description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """技术分析师"""
    try:
        prices = get_prices(ticker, start_date, end_date)
        return generate_technical_prompt(ticker, prices)
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fundamentals/prompt")
def fundamentals_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """基本面分析师"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    return generate_fundamentals_prompt(ticker, metrics, line_items)


@router.get("/growth-analyst/prompt")
def growth_analyst_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """增长分析师"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date, period="annual")
    return generate_growth_prompt(ticker, metrics, line_items)


@router.get("/valuation-analyst/prompt")
def valuation_analyst_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """估值分析师"""
    metrics, line_items, market_cap = _get_common_data(ticker, end_date)
    if not market_cap:
        raise HTTPException(status_code=400, detail="需要市值数据")
    return generate_valuation_prompt(ticker, metrics, line_items, market_cap)


@router.get("/news-sentiment/prompt")
def news_sentiment_endpoint(
    ticker: str = Query(..., description="股票代码"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """新闻情绪分析师"""
    try:
        news = get_company_news(ticker, end_date, None, 100)
        return generate_news_sentiment_prompt(ticker, news)
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-sentiment/prompt")
def market_sentiment_endpoint(
    ticker: str = Query(..., description="股票代码"),
    start_date: str = Query(..., description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
):
    """市场情绪分析师"""
    try:
        prices = get_prices(ticker, start_date, end_date)
        insider_trades = get_insider_trades(ticker, end_date, None, 100)
        return generate_market_sentiment_prompt(ticker, insider_trades, prices)
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 获取所有分析师列表 ====================

@router.get("/list")
def list_analysts():
    """获取所有可用的分析师列表。"""
    return {
        "investor_style": [
            {"key": "value-investor", "name": "价值投资类分析师", "description": "巴菲特风格：护城河、安全边际"},
            {"key": "deep-value", "name": "深度价值分析师", "description": "格雷厄姆风格：净净法、格雷厄姆数"},
            {"key": "growth-innovation", "name": "增长创新分析师", "description": "专注颠覆性技术和创新"},
            {"key": "growth-value", "name": "成长价值分析师", "description": "Peter Lynch风格：PEG比率、可理解性"},
            {"key": "quality-growth", "name": "质量成长分析师", "description": "Phil Fisher风格：管理层、创新能力"},
            {"key": "rational-value", "name": "理性价值分析师", "description": "Charlie Munger风格：优质生意、合理价格"},
            {"key": "contrarian", "name": "逆向投资分析师", "description": "Michael Burry风格：寻找被忽视的低估资产"},
            {"key": "macro-trend", "name": "宏观趋势分析师", "description": "Druckenmiller风格：宏观趋势跟踪"},
            {"key": "valuation-expert", "name": "估值专家分析师", "description": "Damodaran风格：DCF估值专家"},
            {"key": "activist", "name": "激进投资分析师", "description": "Ackman风格：价值释放、催化剂"},
            {"key": "clone-investor", "name": "克隆投资分析师", "description": "Pabrai风格：价值克隆、低风险高不确定性"},
            {"key": "emerging-growth", "name": "新兴市场增长分析师", "description": "Jhunjhunwala风格：消费升级、新兴市场机会"},
        ],
        "technical_analyst": [
            {"key": "technical", "name": "技术分析师", "description": "图表分析、技术指标"},
            {"key": "fundamentals", "name": "基本面分析师", "description": "财务报表分析"},
            {"key": "growth-analyst", "name": "增长分析师", "description": "收入和盈利增长分析"},
            {"key": "valuation-analyst", "name": "估值分析师", "description": "P/E、P/B、EV/EBITDA分析"},
            {"key": "news-sentiment", "name": "新闻情绪分析师", "description": "新闻情绪分析"},
            {"key": "market-sentiment", "name": "市场情绪分析师", "description": "内部交易、价格动量分析"},
        ]
    }
