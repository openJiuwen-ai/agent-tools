import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_pe_ratio(metrics: list[FinancialMetrics], market_cap: float, line_items: list[LineItem]) -> dict:
    """分析市盈率。"""
    if not metrics or not market_cap:
        return {"score": 0, "details": "市盈率分析数据不足"}

    latest = metrics[0]
    latest_item = line_items[0] if line_items else None

    if not latest_item:
        return {"score": 0, "details": "无法计算市盈率"}

    net_income = latest.net_income or latest_item.model_dump().get("net_income")
    shares = latest_item.model_dump().get("outstanding_shares")

    if not net_income or not shares or net_income <= 0:
        return {"score": 0, "details": "无正向盈利用于市盈率计算"}

    eps = net_income / shares
    price_per_share = market_cap / shares
    pe_ratio = price_per_share / eps if eps > 0 else None

    score = 0
    details = []

    if pe_ratio:
        if pe_ratio < 15:
            score += 4
            details.append(f"低市盈率 {pe_ratio:.1f}，价值被低估")
        elif pe_ratio < 25:
            score += 2
            details.append(f"中等市盈率 {pe_ratio:.1f}")
        else:
            details.append(f"高市盈率 {pe_ratio:.1f}，可能高估")

    return {"score": score, "max_score": 4, "details": "; ".join(details) if details else "无市盈率数据"}


def analyze_pb_ratio(line_items: list[LineItem], market_cap: float) -> dict:
    """分析市净率。"""
    if not line_items or not market_cap:
        return {"score": 0, "details": "市净率分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    book_value = latest_data.get("shareholders_equity")
    shares = latest_data.get("outstanding_shares")

    if not book_value or not shares or book_value <= 0:
        return {"score": 0, "details": "无正向账面价值用于市净率计算"}

    book_value_per_share = book_value / shares
    price_per_share = market_cap / shares
    pb_ratio = price_per_share / book_value_per_share if book_value_per_share > 0 else None

    score = 0
    details = []

    if pb_ratio:
        if pb_ratio < 1.5:
            score += 3
            details.append(f"低市净率 {pb_ratio:.2f}，价值被低估")
        elif pb_ratio < 3:
            score += 2
            details.append(f"中等市净率 {pb_ratio:.2f}")
        else:
            details.append(f"高市净率 {pb_ratio:.2f}，可能高估")

    return {"score": score, "max_score": 3, "details": "; ".join(details) if details else "无市净率数据"}


def analyze_ev_ebitda(metrics: list[FinancialMetrics], line_items: list[LineItem], market_cap: float) -> dict:
    """分析EV/EBITDA。"""
    if not metrics or not line_items or not market_cap:
        return {"score": 0, "details": "EV/EBITDA分析数据不足"}

    latest = metrics[0]
    latest_item = line_items[0]
    latest_data = latest_item.model_dump()

    ebitda = latest.ebitda
    debt = latest_data.get("total_liabilities") or 0
    cash = latest_data.get("cash_and_equivalents") or 0

    if not ebitda or ebitda <= 0:
        return {"score": 0, "details": "无正向EBITDA用于估值"}

    ev = market_cap + debt - cash
    ev_ebitda = ev / ebitda

    score = 0
    details = []

    if ev_ebitda < 10:
        score += 3
        details.append(f"低EV/EBITDA {ev_ebitda:.1f}，价值被低估")
    elif ev_ebitda < 15:
        score += 2
        details.append(f"中等EV/EBITDA {ev_ebitda:.1f}")
    else:
        details.append(f"高EV/EBITDA {ev_ebitda:.1f}，可能高估")

    return {"score": score, "max_score": 3, "details": "; ".join(details) if details else "无EV/EBITDA数据"}


def generate_valuation_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
    market_cap: float,
) -> dict:
    """生成估值分析提示词。"""

    pe_analysis = analyze_pe_ratio(financial_metrics, market_cap, financial_line_items)
    pb_analysis = analyze_pb_ratio(financial_line_items, market_cap)
    ev_ebitda_analysis = analyze_ev_ebitda(financial_metrics, financial_line_items, market_cap)

    total_score = pe_analysis["score"] + pb_analysis["score"] + ev_ebitda_analysis["score"]
    max_possible_score = 10

    if total_score >= 0.7 * max_possible_score:
        signal = "bearish"  # 低估值 = 看涨信号，但这里score高表示估值便宜
    elif total_score <= 0.3 * max_possible_score:
        signal = "bullish"  # 高估值 = 看跌信号
    else:
        signal = "neutral"

    facts = {
        "signal": signal,
        "score": total_score,
        "max_score": max_possible_score,
        "pe_analysis": pe_analysis,
        "pb_analysis": pb_analysis,
        "ev_ebitda_analysis": ev_ebitda_analysis,
    }

    system_prompt = """你是估值分析类分析师。基于以下原则做出投资决策：

1. 市盈率(P/E)：低市盈率表示价值被低估
2. 市净率(P/B)：低市净率表示价值被低估
3. EV/EBITDA：低EV/EBITDA表示价值被低估

决策规则：
- 看涨：低估值指标（市盈率<15、市净率<1.5、EV/EBITDA<10）
- 看跌：高估值指标（市盈率>30、市净率>4、EV/EBITDA>20）
- 中性：估值指标在中等范围

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"valuation_analyst"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "valuation_analyst"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
