import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_profitability(metrics: list[FinancialMetrics]) -> dict:
    """分析盈利能力。"""
    if not metrics:
        return {"score": 0, "details": "盈利能力分析数据不足"}

    latest = metrics[0]
    score = 0
    details = []

    # ROE分析
    if latest.return_on_equity and latest.return_on_equity > 0.15:
        score += 3
        details.append(f"强劲的ROE {latest.return_on_equity:.1%}")
    elif latest.return_on_equity and latest.return_on_equity > 0.10:
        score += 2
        details.append(f"良好的ROE {latest.return_on_equity:.1%}")
    elif latest.return_on_equity:
        score += 1
        details.append(f"疲弱的ROE {latest.return_on_equity:.1%}")

    # ROA分析
    if latest.return_on_assets and latest.return_on_assets > 0.10:
        score += 2
        details.append(f"强劲的ROA {latest.return_on_assets:.1%}")
    elif latest.return_on_assets and latest.return_on_assets > 0.05:
        score += 1
        details.append(f"中等的ROA {latest.return_on_assets:.1%}")

    # 净利率分析
    if latest.net_margin and latest.net_margin > 0.15:
        score += 2
        details.append(f"强劲的净利率 {latest.net_margin:.1%}")
    elif latest.net_margin and latest.net_margin > 0.08:
        score += 1
        details.append(f"中等的净利率 {latest.net_margin:.1%}")

    return {"score": score, "max_score": 7, "details": "; ".join(details) if details else "无盈利能力数据"}


def analyze_efficiency(metrics: list[FinancialMetrics]) -> dict:
    """分析运营效率。"""
    if not metrics:
        return {"score": 0, "details": "运营效率分析数据不足"}

    latest = metrics[0]
    score = 0
    details = []

    # 资产周转率
    if latest.asset_turnover and latest.asset_turnover > 1.5:
        score += 2
        details.append(f"高资产周转率 {latest.asset_turnover:.2f}")
    elif latest.asset_turnover and latest.asset_turnover > 1.0:
        score += 1
        details.append(f"中等资产周转率 {latest.asset_turnover:.2f}")

    # 存货周转率
    if latest.inventory_turnover and latest.inventory_turnover > 5:
        score += 2
        details.append(f"高存货周转率 {latest.inventory_turnover:.2f}")
    elif latest.inventory_turnover and latest.inventory_turnover > 3:
        score += 1
        details.append(f"中等存货周转率 {latest.inventory_turnover:.2f}")

    return {"score": score, "max_score": 4, "details": "; ".join(details) if details else "无运营效率数据"}


def analyze_liquidity_safety(metrics: list[FinancialMetrics], line_items: list[LineItem]) -> dict:
    """分析流动性和安全性。"""
    score = 0
    details = []

    # 流动比率
    if metrics and metrics[0].current_ratio:
        if metrics[0].current_ratio >= 2.0:
            score += 2
            details.append(f"强劲的流动比率 {metrics[0].current_ratio:.2f}")
        elif metrics[0].current_ratio >= 1.5:
            score += 1
            details.append(f"中等的流动比率 {metrics[0].current_ratio:.2f}")

    # 利息保障倍数
    if metrics and metrics[0].interest_coverage:
        if metrics[0].interest_coverage >= 5:
            score += 2
            details.append(f"强劲的利息保障倍数 {metrics[0].interest_coverage:.2f}")
        elif metrics[0].interest_coverage >= 3:
            score += 1
            details.append(f"中等的利息保障倍数 {metrics[0].interest_coverage:.2f}")

    # 债务股本比
    if metrics and metrics[0].debt_to_equity:
        if metrics[0].debt_to_equity < 0.5:
            score += 2
            details.append(f"保守的债务股本比 {metrics[0].debt_to_equity:.2f}")
        elif metrics[0].debt_to_equity < 1.0:
            score += 1
            details.append(f"中等的债务股本比 {metrics[0].debt_to_equity:.2f}")

    return {"score": score, "max_score": 6, "details": "; ".join(details) if details else "无流动性数据"}


def generate_fundamentals_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
) -> dict:
    """生成基本面分析提示词。"""

    profitability = analyze_profitability(financial_metrics)
    efficiency = analyze_efficiency(financial_metrics)
    liquidity = analyze_liquidity_safety(financial_metrics, financial_line_items)

    total_score = profitability["score"] + efficiency["score"] + liquidity["score"]
    max_possible_score = 17

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
        "profitability": profitability,
        "efficiency": efficiency,
        "liquidity": liquidity,
    }

    system_prompt = """你是基本面分析类分析师。基于以下原则做出投资决策：

1. 盈利能力：关注ROE、ROA、净利率等指标
2. 运营效率：分析资产周转率和存货周转率
3. 流动性安全性：评估流动比率、利息保障、债务水平

决策规则：
- 看涨：评分 >= 12分（总分17分），基本面强劲
- 看跌：评分 <= 5分，基本面疲弱
- 中性：评分在6-11分之间，基本面中等

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"fundamentals"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "fundamentals"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
