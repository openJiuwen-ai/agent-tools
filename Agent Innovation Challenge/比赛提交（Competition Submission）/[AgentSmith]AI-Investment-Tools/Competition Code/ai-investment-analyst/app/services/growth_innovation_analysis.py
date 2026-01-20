import json

from app.models.analyst import LineItem
from app.models.data import FinancialMetrics


def analyze_disruptive_potential(metrics: list[FinancialMetrics], financial_line_items: list[LineItem]) -> dict:
    """分析颠覆性潜力 - 革命性产品、技术或商业模式。"""
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "max_score": 12, "details": "颠覆性潜力分析数据不足"}

    # 1. 收入增长加速分析
    revenues = [item.model_dump().get("revenue") for item in financial_line_items if item.model_dump().get("revenue")]
    if len(revenues) >= 3:
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i] and revenues[i + 1]:
                growth_rate = (revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]) if revenues[i + 1] != 0 else 0
                growth_rates.append(growth_rate)

        if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[-1]:
            score += 2
            details.append(f"收入增长加速：{(growth_rates[0]*100):.1f}% vs {(growth_rates[-1]*100):.1f}%")

        latest_growth = growth_rates[0] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"卓越的收入增长：{(latest_growth*100):.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"强劲的收入增长：{(latest_growth*100):.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"中等的收入增长：{(latest_growth*100):.1f}%")
    else:
        details.append("收入增长分析数据不足")

    # 2. 毛利率扩张分析
    gross_margins = [item.model_dump().get("gross_margin") for item in financial_line_items if item.model_dump().get("gross_margin") is not None]
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[0] - gross_margins[-1]
        if margin_trend > 0.05:
            score += 2
            details.append(f"毛利率扩张：+{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"毛利率略有改善：+{(margin_trend*100):.1f}%")

        if gross_margins[0] > 0.50:
            score += 2
            details.append(f"高毛利率：{(gross_margins[0]*100):.1f}%")
    else:
        details.append("毛利率数据不足")

    # 3. 运营杠杆分析
    operating_expenses = [item.model_dump().get("operating_expense") for item in financial_line_items if item.model_dump().get("operating_expense") is not None]

    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])

        if rev_growth > opex_growth:
            score += 2
            details.append("正向运营杠杆：收入增长快于费用增长")
    else:
        details.append("运营杠杆分析数据不足")

    # 4. 研发投资分析
    rd_expenses = [item.model_dump().get("research_and_development") for item in financial_line_items if item.model_dump().get("research_and_development") is not None]
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[0] / revenues[0]
        if rd_intensity > 0.15:
            score += 3
            details.append(f"高研发投资：占收入{(rd_intensity*100):.1f}%")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"中等研发投资：占收入{(rd_intensity*100):.1f}%")
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"一定研发投资：占收入{(rd_intensity*100):.1f}%")
    else:
        details.append("无研发数据可用")

    max_possible_score = 12
    normalized_score = (score / max_possible_score) * 5 if max_possible_score > 0 else 0

    return {"score": normalized_score, "max_score": 5, "details": "; ".join(details), "raw_score": score}


def analyze_innovation_growth(metrics: list[FinancialMetrics], financial_line_items: list[LineItem]) -> dict:
    """评估创新驱动的增长潜力。"""
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "max_score": 15, "details": "创新增长分析数据不足"}

    # 1. 研发投资趋势
    rd_expenses = [item.model_dump().get("research_and_development") for item in financial_line_items if item.model_dump().get("research_and_development") is not None]
    revenues = [item.model_dump().get("revenue") for item in financial_line_items if item.model_dump().get("revenue")]

    if rd_expenses and revenues and len(rd_expenses) >= 2:
        rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1]) if rd_expenses[-1] != 0 else 0
        if rd_growth > 0.5:
            score += 3
            details.append(f"强劲的研发投资增长：+{(rd_growth*100):.1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"中等的研发投资增长：+{(rd_growth*100):.1f}%")

        rd_intensity_start = rd_expenses[-1] / revenues[-1]
        rd_intensity_end = rd_expenses[0] / revenues[0]
        if rd_intensity_end > rd_intensity_start:
            score += 2
            details.append(f"研发强度提升：{(rd_intensity_end*100):.1f}% vs {(rd_intensity_start*100):.1f}%")
    else:
        details.append("研发趋势分析数据不足")

    # 2. 自由现金流分析
    fcf_vals = [item.model_dump().get("free_cash_flow") for item in financial_line_items if item.model_dump().get("free_cash_flow")]
    if fcf_vals and len(fcf_vals) >= 2:
        fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)

        if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
            score += 3
            details.append("强劲且一致的FCF增长，优秀的创新资金能力")
        elif positive_fcf_count >= len(fcf_vals) * 0.75:
            score += 2
            details.append("一致的正FCF，良好的创新资金能力")
        elif positive_fcf_count > len(fcf_vals) * 0.5:
            score += 1
            details.append("中等一致的FCF，适当的创新资金能力")
    else:
        details.append("FCF分析数据不足")

    # 3. 运营效率分析
    op_margin_vals = [item.model_dump().get("operating_margin") for item in financial_line_items if item.model_dump().get("operating_margin") is not None]
    if op_margin_vals and len(op_margin_vals) >= 2:
        margin_trend = op_margin_vals[0] - op_margin_vals[-1]

        if op_margin_vals[0] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"强劲且改善的营业利润率：{(op_margin_vals[0]*100):.1f}%")
        elif op_margin_vals[0] > 0.10:
            score += 2
            details.append(f"健康的营业利润率：{(op_margin_vals[0]*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("改善的运营效率")
    else:
        details.append("营业利润率数据不足")

    # 4. 资本配置分析
    capex = [item.model_dump().get("capital_expenditure") for item in financial_line_items if item.model_dump().get("capital_expenditure") is not None]
    if capex and revenues and len(capex) >= 2:
        capex_intensity = abs(capex[0]) / revenues[0]
        capex_growth = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0

        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("强劲的增长基础设施投资")
        elif capex_intensity > 0.05:
            score += 1
            details.append("中等的增长基础设施投资")
    else:
        details.append("资本支出数据不足")

    # 5. 增长再投资分析
    dividends = [item.model_dump().get("dividends_and_other_cash_distributions") for item in financial_line_items if item.model_dump().get("dividends_and_other_cash_distributions") is not None]
    if dividends and fcf_vals:
        latest_payout_ratio = dividends[0] / fcf_vals[0] if fcf_vals[0] != 0 else 1
        if latest_payout_ratio < 0.2:
            score += 2
            details.append("专注于再投资而非股息")
        elif latest_payout_ratio < 0.4:
            score += 1
            details.append("适度专注于再投资而非股息")
    else:
        details.append("股息数据不足")

    max_possible_score = 15
    normalized_score = (score / max_possible_score) * 5 if max_possible_score > 0 else 0

    return {"score": normalized_score, "max_score": 5, "details": "; ".join(details), "raw_score": score}


def analyze_high_growth_valuation(financial_line_items: list[LineItem], market_cap: float | None) -> dict:
    """高增长估值分析 - 关注长期指数增长潜力。"""
    if not financial_line_items or market_cap is None:
        return {"score": 0, "max_score": 5, "details": "估值数据不足"}

    latest = financial_line_items[0]
    latest_data = latest.model_dump()
    fcf = latest_data.get("free_cash_flow") if latest_data.get("free_cash_flow") else 0

    if fcf <= 0:
        return {"score": 0, "max_score": 5, "details": f"无正向FCF用于估值；FCF = {fcf}", "intrinsic_value": None}

    # 高增长公司的高增长假设
    growth_rate = 0.20  # 20%年增长
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # 终值
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.5:
        score = 5
    elif margin_of_safety > 0.2:
        score = 3
    elif margin_of_safety > 0:
        score = 1

    details = [
        f"计算内在价值：~{intrinsic_value:,.2f}",
        f"市值：~{market_cap:,.2f}",
        f"安全边际：{margin_of_safety:.2%}"
    ]

    return {"score": score, "max_score": 5, "details": "; ".join(details), "intrinsic_value": intrinsic_value, "margin_of_safety": margin_of_safety}


def generate_growth_innovation_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
    market_cap: float | None = None,
) -> dict:
    """生成增长创新投资分析提示词。"""

    # 执行所有分析
    disruptive_analysis = analyze_disruptive_potential(financial_metrics, financial_line_items)
    innovation_analysis = analyze_innovation_growth(financial_metrics, financial_line_items)
    valuation_analysis = analyze_high_growth_valuation(financial_line_items, market_cap)

    # 汇总评分
    total_score = disruptive_analysis["score"] + innovation_analysis["score"] + valuation_analysis["score"]
    max_possible_score = 15

    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.3 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"

    # 构建事实
    facts = {
        "signal": signal,
        "score": total_score,
        "max_score": max_possible_score,
        "disruptive_analysis": disruptive_analysis,
        "innovation_analysis": innovation_analysis,
        "valuation_analysis": valuation_analysis,
    }

    # 构建提示词（中文）
    system_prompt = """你是增长创新类分析师。基于以下原则做出投资决策：

1. 寻求利用颠覆性创新的公司
2. 强调指数增长潜力、巨大的潜在市场
3. 专注科技、医疗保健或其他面向未来的行业
4. 考虑多年时间框架内的潜在突破
5. 接受较高波动性以追求高回报
6. 评估管理层的愿景和研发投资能力

决策规则：
- 看涨：评分 >= 10分（总分15分），显示强劲的增长创新特征
- 看跌：评分 <= 5分，缺乏增长创新特征
- 中性：评分在6-9分之间，混合信号

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"growth_innovation"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "growth_innovation"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
