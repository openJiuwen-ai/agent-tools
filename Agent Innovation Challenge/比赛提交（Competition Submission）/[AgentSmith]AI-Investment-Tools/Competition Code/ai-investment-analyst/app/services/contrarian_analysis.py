import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_market_overvaluation(metrics: list[FinancialMetrics], line_items: list[LineItem], market_cap: float | None) -> dict:
    """分析市场是否过度高估。"""
    score = 0
    details = []

    if not metrics or not line_items:
        return {"score": 0, "details": "估值分析数据不足"}

    latest = metrics[0]
    latest_item = line_items[0]
    latest_data = latest_item.model_dump()

    # 检查高市盈率
    shares = latest_data.get("outstanding_shares")
    net_income = latest.net_income or latest_data.get("net_income")

    if net_income and shares and net_income > 0 and market_cap:
        eps = net_income / shares
        price = market_cap / shares
        pe = price / eps if eps > 0 else 0

        if pe > 40:
            score += 3
            details.append(f"极高市盈率 {pe:.1f}，可能过度高估")
        elif pe > 30:
            score += 2
            details.append(f"高市盈率 {pe:.1f}")
        elif pe > 25:
            score += 1
            details.append(f"中等高市盈率 {pe:.1f}")

    # 检查低股息率（高增长公司通常不分红）
    dividends = latest_data.get("dividends_and_other_cash_distributions")
    if dividends is not None and dividends >= 0:
        score += 1
        details.append("无股息支付，可能过度关注增长")

    return {"score": score, "max_score": 4, "details": "; ".join(details) if details else "无估值信号"}


def analyze_financial_risk(line_items: list[LineItem]) -> dict:
    """分析财务风险。"""
    score = 0
    details = []

    if not line_items:
        return {"score": 0, "details": "财务风险分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    total_debt = latest_data.get("total_liabilities") or 0
    total_assets = latest_data.get("total_assets") or 0
    current_assets = latest_data.get("current_assets") or 0
    current_liabilities = latest_data.get("current_liabilities") or 0

    # 债务比率
    if total_assets > 0:
        debt_ratio = total_debt / total_assets
        if debt_ratio > 0.8:
            score += 3
            details.append(f"高债务比率 {debt_ratio:.2f}，风险较高")
        elif debt_ratio > 0.6:
            score += 2
            details.append(f"中等债务比率 {debt_ratio:.2f}")
        elif debt_ratio > 0.4:
            score += 1
            details.append(f"可接受债务比率 {debt_ratio:.2f}")

    # 流动性风险
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio < 1.0:
            score += 2
            details.append(f"低流动比率 {current_ratio:.2f}，流动性风险")
        elif current_ratio < 1.5:
            score += 1
            details.append(f"中等流动比率 {current_ratio:.2f}")

    return {"score": score, "max_score": 5, "details": "; ".join(details) if details else "无风险信号"}


def analyze_hidden_value(line_items: list[LineItem]) -> dict:
    """分析被忽视的资产价值。"""
    score = 0
    details = []

    if not line_items:
        return {"score": 0, "details": "隐藏价值分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    # 净净资产价值
    current_assets = latest_data.get("current_assets") or 0
    total_liabilities = latest_data.get("total_liabilities") or 0

    net_current_assets = current_assets - total_liabilities

    if net_current_assets > 0:
        score += 3
        details.append(f"正向净流动资产价值 {net_current_assets:,.0f}")
    elif net_current_assets > -current_assets * 0.5:
        score += 1
        details.append("净流动资产价值接近零")

    return {"score": score, "max_score": 3, "details": "; ".join(details) if details else "无隐藏价值信号"}


def generate_contrarian_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
    market_cap: float | None,
) -> dict:
    """生成逆向投资分析提示词。"""

    overvaluation = analyze_market_overvaluation(financial_metrics, financial_line_items, market_cap)
    financial_risk = analyze_financial_risk(financial_line_items)
    hidden_value = analyze_hidden_value(financial_line_items)

    # 逆向投资：寻找被市场忽视的低估资产
    contrarian_score = hidden_value["score"]
    risk_penalty = financial_risk["score"]
    overvaluation_score = overvaluation["score"]

    total_score = contrarian_score + (4 - overvaluation_score) - risk_penalty
    max_possible_score = 7

    if total_score >= 0.6 * max_possible_score:
        signal = "bullish"  # 被忽视但有价值
    elif total_score <= 0.2 * max_possible_score:
        signal = "bearish"  # 高估且高风险
    else:
        signal = "neutral"

    facts = {
        "signal": signal,
        "score": total_score,
        "max_score": max_possible_score,
        "overvaluation_analysis": overvaluation,
        "financial_risk": financial_risk,
        "hidden_value": hidden_value,
    }

    system_prompt = """你是逆向投资类分析师。基于以下原则做出投资决策：

1. 寻找被市场过度低估的资产
2. 关注高估泡沫和财务风险
3. 识别被忽视的隐藏价值
4. 愿意与市场共识相反

决策规则：
- 看涨：被市场忽视但具有隐藏价值
- 看跌：市场过度高估且财务风险高
- 中性：信号混合

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"contrarian"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "contrarian"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
