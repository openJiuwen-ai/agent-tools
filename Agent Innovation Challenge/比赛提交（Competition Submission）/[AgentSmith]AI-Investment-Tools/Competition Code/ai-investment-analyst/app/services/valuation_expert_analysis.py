import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_dcf_valuation(line_items: list[LineItem], market_cap: float) -> dict:
    """分析DCF估值 - Damodaran的专业领域。"""
    if not line_items or not market_cap:
        return {"score": 0, "details": "DCF估值数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    fcf = latest_data.get("free_cash_flow") or latest_data.get("net_income") or 0

    if fcf <= 0:
        return {"score": 0, "details": "无正向现金流"}

    # 三阶段DCF（保守假设）
    growth_stage1 = 0.10
    growth_stage2 = 0.05
    growth_terminal = 0.025
    discount_rate = 0.10

    stage1_years = 5
    stage2_years = 5

    pv = 0
    for year in range(1, stage1_years + 1):
        pv += fcf * (1 + growth_stage1) ** year / (1 + discount_rate) ** year

    stage1_final = fcf * (1 + growth_stage1) ** stage1_years
    for year in range(1, stage2_years + 1):
        pv += stage1_final * (1 + growth_stage2) ** year / (1 + discount_rate) ** (stage1_years + year)

    stage2_final = stage1_final * (1 + growth_stage2) ** stage2_years
    terminal_value = stage2_final * (1 + growth_terminal) / (discount_rate - growth_terminal)
    pv += terminal_value / (1 + discount_rate) ** (stage1_years + stage2_years)

    margin = (pv - market_cap) / market_cap

    score = 0
    details = []

    if margin > 0.5:
        score = 5
        details.append(f"巨大安全边际 {margin:.1%}，显著低估")
    elif margin > 0.2:
        score = 3
        details.append(f"良好安全边际 {margin:.1%}，低估")
    elif margin > 0:
        score = 1
        details.append(f"小幅安全边际 {margin:.1%}")
    else:
        details.append(f"负安全边际 {margin:.1%}，可能高估")

    return {"score": score, "max_score": 5, "details": "; ".join(details), "intrinsic_value": pv}


def analyze_relative_valuation(metrics: list[FinancialMetrics], line_items: list[LineItem], market_cap: float) -> dict:
    """分析相对估值。"""
    if not metrics or not line_items or not market_cap:
        return {"score": 0, "details": "相对估值数据不足"}

    latest = metrics[0]
    latest_item = line_items[0]
    latest_data = latest_item.model_dump()

    score = 0
    details = []

    # EV/EBITDA
    ebitda = latest.ebitda
    if ebitda and ebitda > 0:
        shares = latest_data.get("outstanding_shares") or 1
        debt = latest_data.get("total_liabilities") or 0
        cash = latest_data.get("cash_and_equivalents") or 0
        ev = market_cap + debt - cash
        ev_ebitda = ev / ebitda

        if ev_ebitda < 10:
            score += 2
            details.append(f"低EV/EBITDA {ev_ebitda:.1f}")
        elif ev_ebitda < 15:
            score += 1
            details.append(f"中等EV/EBITDA {ev_ebitda:.1f}")

    # P/B
    book_value = latest_data.get("shareholders_equity")
    if book_value and book_value > 0:
        pb = market_cap / book_value
        if pb < 1.5:
            score += 2
            details.append(f"低市净率 {pb:.2f}")
        elif pb < 3:
            score += 1
            details.append(f"中等市净率 {pb:.2f}")

    return {"score": score, "max_score": 4, "details": "; ".join(details) if details else "无相对估值数据"}


def analyze_growth_quality(metrics: list[FinancialMetrics]) -> dict:
    """分析增长质量。"""
    if not metrics or len(metrics) < 3:
        return {"score": 0, "details": "增长质量分析数据不足"}

    # ROE趋势
    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]

    score = 0
    details = []

    if len(roes) >= 3:
        avg_roe = sum(roes) / len(roes)
        if avg_roe > 0.20:
            score += 3
            details.append(f"卓越平均ROE {avg_roe:.1%}")
        elif avg_roe > 0.15:
            score += 2
            details.append(f"良好平均ROE {avg_roe:.1%}")

    # ROIC（如果有）
    if metrics[0].return_on_invested_capital:
        roic = metrics[0].return_on_invested_capital
        if roic > 0.15:
            score += 2
            details.append(f"卓越ROIC {roic:.1%}")
        elif roic > 0.10:
            score += 1
            details.append(f"良好ROIC {roic:.1%}")

    return {"score": score, "max_score": 5, "details": "; ".join(details) if details else "无增长质量数据"}


def generate_valuation_expert_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
    market_cap: float,
) -> dict:
    """生成估值专家分析提示词。"""

    dcf = analyze_dcf_valuation(financial_line_items, market_cap)
    relative = analyze_relative_valuation(financial_metrics, financial_line_items, market_cap)
    quality = analyze_growth_quality(financial_metrics)

    total_score = dcf["score"] + relative["score"] + quality["score"]
    max_possible_score = 14

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
        "dcf_valuation": dcf,
        "relative_valuation": relative,
        "growth_quality": quality,
    }

    system_prompt = """你是估值专家类分析师。基于以下原则做出投资决策：

1. DCF估值：使用现金流折现模型计算内在价值
2. 相对估值：比较EV/EBITDA、P/B等指标
3. 增长质量：评估ROE和ROIC等资本回报指标

决策规则：
- 看涨：DCF显示显著低估 + 相对估值低 + 高质量增长
- 看跌：DCF显示高估或增长质量低
- 中性：估值合理或信号混合

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"valuation_expert"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "valuation_expert"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
