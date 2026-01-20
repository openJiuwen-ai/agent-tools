import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_revenue_growth(line_items: list[LineItem]) -> dict:
    """分析收入增长。"""
    if len(line_items) < 3:
        return {"score": 0, "details": "收入增长分析数据不足"}

    revenues = [item.model_dump().get("revenue") for item in line_items if item.model_dump().get("revenue")]

    if len(revenues) < 3:
        return {"score": 0, "details": "收入数据不足"}

    score = 0
    details = []

    # 计算增长率
    growth_rates = []
    for i in range(len(revenues) - 1):
        if revenues[i + 1] != 0:
            growth_rates.append((revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]))

    if not growth_rates:
        return {"score": 0, "details": "无法计算增长率"}

    latest_growth = growth_rates[0]
    avg_growth = sum(growth_rates) / len(growth_rates)

    if latest_growth > 0.30:
        score += 4
        details.append(f"卓越的最新收入增长 {latest_growth:.1%}")
    elif latest_growth > 0.20:
        score += 3
        details.append(f"强劲的最新收入增长 {latest_growth:.1%}")
    elif latest_growth > 0.10:
        score += 2
        details.append(f"良好的最新收入增长 {latest_growth:.1%}")
    elif latest_growth > 0:
        score += 1
        details.append(f"正向的收入增长 {latest_growth:.1%}")

    # 增长加速度
    if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[-1]:
        score += 2
        details.append("收入增长加速")

    return {"score": score, "max_score": 6, "details": "; ".join(details) if details else "无收入增长数据"}


def analyze_earnings_growth(metrics: list[FinancialMetrics]) -> dict:
    """分析盈利增长。"""
    if len(metrics) < 3:
        return {"score": 0, "details": "盈利增长分析数据不足"}

    score = 0
    details = []

    net_income_vals = [m.net_income for m in metrics if m.net_income is not None]

    if len(net_income_vals) < 2:
        return {"score": 0, "details": "盈利数据不足"}

    growth_rates = []
    for i in range(len(net_income_vals) - 1):
        if net_income_vals[i + 1] != 0:
            growth_rates.append((net_income_vals[i] - net_income_vals[i + 1]) / abs(net_income_vals[i + 1]))

    if growth_rates:
        avg_growth = sum(growth_rates) / len(growth_rates)
        if avg_growth > 0.20:
            score += 4
            details.append(f"卓越的盈利增长 {avg_growth:.1%}")
        elif avg_growth > 0.10:
            score += 3
            details.append(f"强劲的盈利增长 {avg_growth:.1%}")
        elif avg_growth > 0:
            score += 2
            details.append(f"正向的盈利增长 {avg_growth:.1%}")
        else:
            details.append(f"负向的盈利增长 {avg_growth:.1%}")

    return {"score": score, "max_score": 4, "details": "; ".join(details) if details else "无盈利增长数据"}


def analyze_growth_consistency(line_items: list[LineItem]) -> dict:
    """分析增长一致性。"""
    if len(line_items) < 4:
        return {"score": 0, "details": "增长一致性分析数据不足"}

    revenues = [item.model_dump().get("revenue") for item in line_items if item.model_dump().get("revenue")]

    if len(revenues) < 4:
        return {"score": 0, "details": "收入数据不足以分析一致性"}

    score = 0
    details = []

    # 检查持续增长
    growth_periods = sum(1 for i in range(len(revenues) - 1) if revenues[i] > revenues[i + 1])
    consistency = growth_periods / (len(revenues) - 1)

    if consistency >= 0.8:
        score += 5
        details.append(f"高度一致的增长 {consistency:.1%}")
    elif consistency >= 0.6:
        score += 3
        details.append(f"良好的一致性 {consistency:.1%}")
    elif consistency >= 0.4:
        score += 1
        details.append(f"中等的一致性 {consistency:.1%}")
    else:
        details.append(f"增长不一致 {consistency:.1%}")

    return {"score": score, "max_score": 5, "details": "; ".join(details) if details else "无一致性数据"}


def generate_growth_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
) -> dict:
    """生成增长分析提示词。"""

    revenue_growth = analyze_revenue_growth(financial_line_items)
    earnings_growth = analyze_earnings_growth(financial_metrics)
    growth_consistency = analyze_growth_consistency(financial_line_items)

    total_score = revenue_growth["score"] + earnings_growth["score"] + growth_consistency["score"]
    max_possible_score = 15

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
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "growth_consistency": growth_consistency,
    }

    system_prompt = """你是增长分析类分析师。基于以下原则做出投资决策：

1. 收入增长：关注收入增长率和增长加速度
2. 盈利增长：分析净利润增长趋势
3. 增长一致性：评估增长的可持续性

决策规则：
- 看涨：评分 >= 10分（总分15分），增长强劲
- 看跌：评分 <= 5分，增长疲弱
- 中性：评分在6-9分之间，增长中等

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"growth_analyst"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "growth_analyst"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
