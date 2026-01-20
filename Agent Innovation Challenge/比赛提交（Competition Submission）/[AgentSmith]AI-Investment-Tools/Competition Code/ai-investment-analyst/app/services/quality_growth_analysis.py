import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_management_quality(line_items: list[LineItem]) -> dict:
    """分析管理层质量 - Phil Fisher的核心关注点。"""
    score = 0
    details = []

    if not line_items:
        return {"score": 0, "details": "管理层质量分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    # ROE - 管理层资本配置能力
    # 需要从financial_metrics获取
    return {"score": score, "max_score": 4, "details": "; ".join(details) if details else "无管理层数据"}


def analyze_innovation_capability(line_items: list[LineItem]) -> dict:
    """分析创新能力。"""
    score = 0
    details = []

    if not line_items:
        return {"score": 0, "details": "创新能力分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    # 研发投入
    rd = latest_data.get("research_and_development")
    revenue = latest_data.get("revenue")

    if rd and revenue and revenue > 0:
        rd_ratio = rd / revenue
        if rd_ratio > 0.10:
            score += 3
            details.append(f"高研发投入占比 {rd_ratio:.1%}")
        elif rd_ratio > 0.05:
            score += 2
            details.append(f"中等研发投入占比 {rd_ratio:.1%}")
        elif rd_ratio > 0.02:
            score += 1
            details.append(f"一定研发投入占比 {rd_ratio:.1%}")

    return {"score": score, "max_score": 3, "details": "; ".join(details) if details else "无创新数据"}


def analyze_competitive_position(metrics: list[FinancialMetrics]) -> dict:
    """分析竞争地位。"""
    score = 0
    details = []

    if not metrics:
        return {"score": 0, "details": "竞争地位分析数据不足"}

    latest = metrics[0]

    # 高毛利率 = 竞争优势
    if latest.gross_margin and latest.gross_margin > 0.50:
        score += 3
        details.append(f"卓越毛利率 {latest.gross_margin:.1%}，强竞争优势")
    elif latest.gross_margin and latest.gross_margin > 0.35:
        score += 2
        details.append(f"良好毛利率 {latest.gross_margin:.1%}")
    elif latest.gross_margin and latest.gross_margin > 0.20:
        score += 1
        details.append(f"中等毛利率 {latest.gross_margin:.1%}")

    # 高ROE = 优质业务
    if latest.return_on_equity and latest.return_on_equity > 0.20:
        score += 2
        details.append(f"卓越ROE {latest.return_on_equity:.1%}")
    elif latest.return_on_equity and latest.return_on_equity > 0.15:
        score += 1
        details.append(f"良好ROE {latest.return_on_equity:.1%}")

    return {"score": score, "max_score": 5, "details": "; ".join(details) if details else "无竞争地位数据"}


def analyze_long_term_potential(metrics: list[FinancialMetrics], line_items: list[LineItem]) -> dict:
    """分析长期潜力。"""
    score = 0
    details = []

    if not metrics or not line_items:
        return {"score": 0, "details": "长期潜力分析数据不足"}

    # 收入增长趋势
    revenues = [item.model_dump().get("revenue") for item in line_items[:5] if item.model_dump().get("revenue")]

    if len(revenues) >= 3:
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i + 1] != 0:
                growth_rates.append((revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]))

        if growth_rates:
            avg_growth = sum(growth_rates) / len(growth_rates)
            if avg_growth > 0.20:
                score += 4
                details.append(f"强劲长期增长 {avg_growth:.1%}")
            elif avg_growth > 0.10:
                score += 3
                details.append(f"良好长期增长 {avg_growth:.1%}")
            elif avg_growth > 0.05:
                score += 1
                details.append(f"中等长期增长 {avg_growth:.1%}")

    return {"score": score, "max_score": 4, "details": "; ".join(details) if details else "无长期潜力数据"}


def generate_quality_growth_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
) -> dict:
    """生成质量成长投资分析提示词。"""

    management = analyze_management_quality(financial_line_items)
    innovation = analyze_innovation_capability(financial_line_items)
    competitive = analyze_competitive_position(financial_metrics)
    long_term = analyze_long_term_potential(financial_metrics, financial_line_items)

    total_score = management["score"] + innovation["score"] + competitive["score"] + long_term["score"]
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
        "management_quality": management,
        "innovation_capability": innovation,
        "competitive_position": competitive,
        "long_term_potential": long_term,
    }

    system_prompt = """你是质量成长投资类分析师。基于以下原则做出投资决策：

1. 管理层质量：诚信和能力
2. 创新能力：研发投入和技术领先
3. 竞争地位：护城河和市场份额
4. 长期潜力：可持续的增长前景

决策规则：
- 看涨：高质量成长股（高ROE、高毛利率、强创新）
- 看跌：低质量或增长乏力
- 中性：信号混合

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"quality_growth"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "quality_growth"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
