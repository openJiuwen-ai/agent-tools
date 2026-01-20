import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_peg_ratio(metrics: list[FinancialMetrics], market_cap: float, line_items: list[LineItem]) -> dict:
    """分析PEG比率（市盈增长比率）- Peter Lynch的标志性指标。"""
    if not metrics or not line_items or not market_cap:
        return {"score": 0, "details": "PEG分析数据不足"}

    latest = metrics[0]
    latest_item = line_items[0]
    latest_data = latest_item.model_dump()

    net_income = latest.net_income or latest_data.get("net_income")
    shares = latest_data.get("outstanding_shares")

    if not net_income or not shares or net_income <= 0:
        return {"score": 0, "details": "无正向盈利用于PEG计算"}

    eps = net_income / shares
    price = market_cap / shares
    pe = price / eps if eps > 0 else None

    if not pe:
        return {"score": 0, "details": "无法计算市盈率"}

    # 估算增长率
    revenues = [item.model_dump().get("revenue") for item in line_items[:5] if item.model_dump().get("revenue")]

    if len(revenues) < 2:
        return {"score": 0, "details": "收入数据不足以估算增长率"}

    growth_rate = (revenues[0] - revenues[-1]) / abs(revenues[-1]) if revenues[-1] != 0 else 0
    growth_rate = max(0, min(growth_rate, 1))  # 限制在0-100%

    if growth_rate == 0:
        return {"score": 0, "details": "无增长，无法计算PEG"}

    peg = pe / (growth_rate * 100)  # 转换为百分比

    score = 0
    details = []

    if peg < 1.0:
        score += 4
        details.append(f"低PEG {peg:.2f}，被低估")
    elif peg < 1.5:
        score += 3
        details.append(f"合理PEG {peg:.2f}")
    elif peg < 2.0:
        score += 1
        details.append(f"略高PEG {peg:.2f}")
    else:
        details.append(f"高PEG {peg:.2f}，可能高估")

    return {"score": score, "max_score": 4, "details": "; ".join(details) if details else "无PEG数据"}


def analyze_understandability(line_items: list[LineItem]) -> dict:
    """分析业务可理解性 - Peter Lynch的"买你所知"原则。"""
    score = 0
    details = []

    if not line_items:
        return {"score": 0, "details": "业务分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    # 简单的业务模式（高毛利率、清晰的收入来源）
    gross_margin = latest_data.get("gross_margin")

    if gross_margin and gross_margin > 0.40:
        score += 2
        details.append(f"高毛利率 {gross_margin:.1%}，业务模式清晰")
    elif gross_margin and gross_margin > 0.25:
        score += 1
        details.append(f"中等毛利率 {gross_margin:.1%}")

    # 持续的盈利历史
    net_income = latest_data.get("net_income")
    if net_income and net_income > 0:
        score += 2
        details.append("持续正向盈利")

    # 低复杂性（稳定的运营支出）
    operating_expense = latest_data.get("operating_expense")
    revenue = latest_data.get("revenue")

    if operating_expense and revenue and revenue > 0:
        opex_ratio = operating_expense / revenue
        if opex_ratio < 0.3:
            score += 2
            details.append(f"低运营支出比率 {opex_ratio:.1%}，运营高效")
        elif opex_ratio < 0.5:
            score += 1
            details.append(f"中等运营支出比率 {opex_ratio:.1%}")

    return {"score": score, "max_score": 6, "details": "; ".join(details) if details else "无可理解性数据"}


def analyze_growth_sustainability(metrics: list[FinancialMetrics], line_items: list[LineItem]) -> dict:
    """分析增长可持续性。"""
    score = 0
    details = []

    if not metrics or not line_items:
        return {"score": 0, "details": "增长可持续性分析数据不足"}

    # ROE分析（可持续增长率的关键指标）
    latest = metrics[0]
    if latest.return_on_equity and latest.return_on_equity > 0.15:
        score += 3
        details.append(f"高ROE {latest.return_on_equity:.1%}，增长可持续")
    elif latest.return_on_equity and latest.return_on_equity > 0.10:
        score += 2
        details.append(f"良好ROE {latest.return_on_equity:.1%}")

    # 收入增长一致性
    revenues = [item.model_dump().get("revenue") for item in line_items[:5] if item.model_dump().get("revenue")]

    if len(revenues) >= 3:
        growth_periods = sum(1 for i in range(len(revenues) - 1) if revenues[i] > revenues[i + 1])
        consistency = growth_periods / (len(revenues) - 1)

        if consistency >= 0.8:
            score += 3
            details.append(f"高度一致的增长 {consistency:.1%}")
        elif consistency >= 0.6:
            score += 2
            details.append(f"良好一致的增长 {consistency:.1%}")

    return {"score": score, "max_score": 6, "details": "; ".join(details) if details else "无可持续性数据"}


def generate_growth_value_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
    market_cap: float,
) -> dict:
    """生成成长价值投资分析提示词。"""

    peg = analyze_peg_ratio(financial_metrics, market_cap, financial_line_items)
    understandability = analyze_understandability(financial_line_items)
    sustainability = analyze_growth_sustainability(financial_metrics, financial_line_items)

    total_score = peg["score"] + understandability["score"] + sustainability["score"]
    max_possible_score = 16

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
        "peg_analysis": peg,
        "understandability": understandability,
        "sustainability": sustainability,
    }

    system_prompt = """你是成长价值投资类分析师。基于以下原则做出投资决策：

1. PEG比率：寻找PEG < 1的增长股
2. 业务可理解性："买你所知"原则
3. 增长可持续性：ROE和收入增长一致性

决策规则：
- 看涨：PEG低 + 业务可理解 + 增长可持续
- 看跌：PEG高或增长不可持续
- 中性：信号混合

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"growth_value"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "growth_value"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
