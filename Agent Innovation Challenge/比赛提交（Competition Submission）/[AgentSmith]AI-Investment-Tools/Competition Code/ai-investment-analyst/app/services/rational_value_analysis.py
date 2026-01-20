import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_business_quality(metrics: list[FinancialMetrics]) -> dict:
    """分析业务质量 - Charlie Munger强调的"好生意"。"""
    if not metrics:
        return {"score": 0, "details": "业务质量分析数据不足"}

    latest = metrics[0]
    score = 0
    details = []

    # 高ROE = 优质业务
    if latest.return_on_equity and latest.return_on_equity > 0.20:
        score += 3
        details.append(f"卓越ROE {latest.return_on_equity:.1%}，优质业务")
    elif latest.return_on_equity and latest.return_on_equity > 0.15:
        score += 2
        details.append(f"良好ROE {latest.return_on_equity:.1%}")

    # 高毛利率 = 定价权
    if latest.gross_margin and latest.gross_margin > 0.60:
        score += 3
        details.append(f"卓越毛利率 {latest.gross_margin:.1%}，强定价权")
    elif latest.gross_margin and latest.gross_margin > 0.40:
        score += 2
        details.append(f"良好毛利率 {latest.gross_margin:.1%}")

    # 高净利率 = 运营效率
    if latest.net_margin and latest.net_margin > 0.20:
        score += 2
        details.append(f"卓越净利率 {latest.net_margin:.1%}")
    elif latest.net_margin and latest.net_margin > 0.15:
        score += 1
        details.append(f"良好净利率 {latest.net_margin:.1%}")

    return {"score": score, "max_score": 8, "details": "; ".join(details) if details else "无业务质量数据"}


def analyze_competitive_advantage(metrics: list[FinancialMetrics]) -> dict:
    """分析竞争优势（护城河）。"""
    if not metrics or len(metrics) < 5:
        return {"score": 0, "details": "竞争优势分析数据不足"}

    # ROE一致性
    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]

    if len(roes) >= 5:
        high_roe_count = sum(1 for roe in roes if roe > 0.15)
        consistency = high_roe_count / len(roes)

        if consistency >= 0.8:
            score = 5
            details = [f"ROE高度一致 {consistency:.1%}，强护城河"]
        elif consistency >= 0.6:
            score = 3
            details = [f"ROE较一致 {consistency:.1%}，中等护城河"]
        else:
            score = 1
            details = [f"ROE不一致 {consistency:.1%}，护城河较弱"]
    else:
        score = 0
        details = ["ROE数据不足以评估护城河"]

    return {"score": score, "max_score": 5, "details": "; ".join(details)}


def analyze_rational_price(line_items: list[LineItem], market_cap: float | None) -> dict:
    """分析合理价格 - Munger强调"以合理价格买入优质业务"。"""
    if not line_items or not market_cap:
        return {"score": 0, "details": "价格合理性分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    owners_earnings = latest_data.get("free_cash_flow") or latest_data.get("net_income") or 0

    if owners_earnings <= 0:
        return {"score": 0, "details": "无正向现金流"}

    # 简化的合理估值：10-15倍所有者收益
    fair_value_low = owners_earnings * 10
    fair_value_high = owners_earnings * 15

    score = 0
    details = []

    if market_cap < fair_value_low:
        score = 5
        details.append(f"市值{market_cap:,.0f}低于合理估值下限{fair_value_low:,.0f}")
    elif market_cap < fair_value_high:
        score = 3
        details.append(f"市值{market_cap:,.0f}在合理估值范围内")
    else:
        score = 0
        details.append(f"市值{market_cap:,.0f}高于合理估值上限{fair_value_high:,.0f}")

    return {"score": score, "max_score": 5, "details": "; ".join(details)}


def generate_rational_value_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
    market_cap: float | None,
) -> dict:
    """生成理性价值投资分析提示词。"""

    quality = analyze_business_quality(financial_metrics)
    advantage = analyze_competitive_advantage(financial_metrics)
    price = analyze_rational_price(financial_line_items, market_cap)

    total_score = quality["score"] + advantage["score"] + price["score"]
    max_possible_score = 18

    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"

    facts = {
        "signal": signal,
        "score": total_score,
        "max_score": max_possible_score,
        "business_quality": quality,
        "competitive_advantage": advantage,
        "rational_price": price,
    }

    system_prompt = """你是理性价值投资类分析师。基于以下原则做出投资决策：

1. 业务质量：寻找高ROE、高毛利率的优质业务
2. 竞争优势：评估护城河的持续性和强度
3. 合理价格：以合理价格买入优质业务

决策规则：
- 看涨：优质业务 + 强护城河 + 合理价格
- 看跌：低质量业务或价格过高
- 中性：信号混合

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"rational_value"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "rational_value"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
