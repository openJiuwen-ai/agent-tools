import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_domestic_market_opportunity(metrics: list[FinancialMetrics]) -> dict:
    """分析国内市场机会 - 新兴市场增长潜力。"""
    if not metrics:
        return {"score": 0, "details": "市场机会分析数据不足"}

    score = 0
    details = []

    latest = metrics[0]

    # 高增长 = 市场机会
    # 简化：通过收入增长评估
    return {"score": score, "max_score": 5, "details": "; ".join(details) if details else "无市场机会数据"}


def analyze_consumer_growth(line_items: list[LineItem]) -> dict:
    """分析消费增长 - 新兴市场消费升级主题。"""
    if not line_items:
        return {"score": 0, "details": "消费增长分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    score = 0
    details = []

    # 收入增长
    revenues = [item.model_dump().get("revenue") for item in line_items[:5] if item.model_dump().get("revenue")]

    if len(revenues) >= 2:
        growth = (revenues[0] - revenues[-1]) / abs(revenues[-1]) if revenues[-1] != 0 else 0

        if growth > 0.25:
            score += 4
            details.append(f"卓越收入增长{growth:.1%}，受益于消费升级")
        elif growth > 0.15:
            score += 3
            details.append(f"强劲收入增长{growth:.1%}")
        elif growth > 0.10:
            score += 2
            details.append(f"良好收入增长{growth:.1%}")
        elif growth > 0:
            score += 1
            details.append(f"正向收入增长{growth:.1%}")

    # 毛利率扩张 = 消费升级
    gross_margins = [item.model_dump().get("gross_margin") for item in line_items if item.model_dump().get("gross_margin") is not None]

    if len(gross_margins) >= 2:
        margin_trend = gross_margins[0] - gross_margins[-1]
        if margin_trend > 0:
            score += 2
            details.append(f"毛利率扩张{margin_trend:.1%}，消费升级")

    return {"score": score, "max_score": 6, "details": "; ".join(details) if details else "无消费增长数据"}


def analyze_sector_trends(metrics: list[FinancialMetrics]) -> dict:
    """分析行业趋势 - 识别高增长行业。"""
    if not metrics:
        return {"score": 0, "details": "行业趋势分析数据不足"}

    score = 0
    details = []

    latest = metrics[0]

    # 高ROE = 优质行业
    if latest.return_on_equity and latest.return_on_equity > 0.20:
        score += 3
        details.append(f"高ROE {latest.return_on_equity:.1%}，优质行业")
    elif latest.return_on_equity and latest.return_on_equity > 0.15:
        score += 2
        details.append(f"良好ROE {latest.return_on_equity:.1%}")

    # 高利润率 = 行业竞争力
    if latest.net_margin and latest.net_margin > 0.15:
        score += 2
        details.append(f"高净利率 {latest.net_margin:.1%}")

    return {"score": score, "max_score": 5, "details": "; ".join(details) if details else "无行业趋势数据"}


def generate_emerging_growth_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
) -> dict:
    """生成新兴市场增长分析提示词。"""

    market = analyze_domestic_market_opportunity(financial_metrics)
    consumer = analyze_consumer_growth(financial_line_items)
    sector = analyze_sector_trends(financial_metrics)

    total_score = market["score"] + consumer["score"] + sector["score"]
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
        "domestic_market_opportunity": market,
        "consumer_growth": consumer,
        "sector_trends": sector,
    }

    system_prompt = """你是新兴市场增长投资类分析师。基于以下原则做出投资决策：

1. 国内市场机会：受益于新兴市场经济增长
2. 消费增长：消费升级趋势推动收入增长
3. 行业趋势：识别高增长潜力的优质行业

决策规则：
- 看涨：强劲增长 + 受益于消费升级 + 优质行业
- 看跌：增长疲弱或行业前景不佳
- 中性：信号混合

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"emerging_growth"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "emerging_growth"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
