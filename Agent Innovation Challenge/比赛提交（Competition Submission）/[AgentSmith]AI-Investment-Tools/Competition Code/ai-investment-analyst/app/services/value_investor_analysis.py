import json

from app.models.analyst import LineItem
from app.models.data import FinancialMetrics
from app.services.data_fetching import DataFetchingError, get_financial_metrics


def analyze_fundamentals(metrics: list[FinancialMetrics]) -> dict:
    """基于价值投资标准分析公司基本面。"""
    if not metrics:
        return {"score": 0, "details": "基本面数据不足"}

    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    # 检查 ROE (股本回报率)
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"强劲的ROE {latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"疲弱的ROE {latest_metrics.return_on_equity:.1%}")
    else:
        reasoning.append("ROE数据不可用")

    # 检查债务股本比
    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("保守的债务水平")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"高债务股本比 {latest_metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("债务股本比数据不可用")

    # 检查营业利润率
    if latest_metrics.operating_margin and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("强劲的营业利润率")
    elif latest_metrics.operating_margin:
        reasoning.append(f"疲弱的营业利润率 {latest_metrics.operating_margin:.1%}")
    else:
        reasoning.append("营业利润率数据不可用")

    # 检查流动比率
    if latest_metrics.current_ratio and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("良好的流动性状况")
    elif latest_metrics.current_ratio:
        reasoning.append(f"流动性较弱，流动比率 {latest_metrics.current_ratio:.1f}")
    else:
        reasoning.append("流动比率数据不可用")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_consistency(financial_line_items: list[LineItem]) -> dict:
    """分析盈利一致性和增长。"""
    if len(financial_line_items) < 4:
        return {"score": 0, "details": "历史数据不足"}

    score = 0
    reasoning = []

    # 检查盈利增长趋势
    earnings_values = [
        item.model_dump().get("net_income")
        for item in financial_line_items
        if item.model_dump().get("net_income") is not None
    ]

    if len(earnings_values) >= 4:
        earnings_growth = all(
            earnings_values[i] > earnings_values[i + 1]
            for i in range(len(earnings_values) - 1)
        )

        if earnings_growth:
            score += 3
            reasoning.append("过去期间盈利持续增长")
        else:
            reasoning.append("盈利增长模式不稳定")

        # 计算从最早到最新期间的总增长率
        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(
                f"过去{len(earnings_values)}期盈利总增长率为{growth_rate:.1%}"
            )
    else:
        reasoning.append("盈利数据不足以进行趋势分析")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_moat(metrics: list[FinancialMetrics]) -> dict:
    """评估公司是否拥有持久的竞争优势（护城河）。"""
    if not metrics or len(metrics) < 5:
        return {"score": 0, "max_score": 5, "details": "数据不足以进行全面护城河分析"}

    reasoning = []
    moat_score = 0
    max_score = 5

    # 1. 资本回报率一致性
    historical_roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]

    if len(historical_roes) >= 5:
        high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
        roe_consistency = high_roe_periods / len(historical_roes)

        if roe_consistency >= 0.8:
            moat_score += 2
            avg_roe = sum(historical_roes) / len(historical_roes)
            reasoning.append(
                f"卓越的ROE一致性：{high_roe_periods}/{len(historical_roes)}期>15%（平均：{avg_roe:.1%}）"
            )
        elif roe_consistency >= 0.6:
            moat_score += 1
            reasoning.append(f"良好的ROE表现：{high_roe_periods}/{len(historical_roes)}期>15%")
        else:
            reasoning.append(f"ROE不稳定：仅{high_roe_periods}/{len(historical_roes)}期>15%")
    else:
        reasoning.append("ROE历史数据不足以进行护城河分析")

    # 2. 营业利润率稳定性
    historical_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(historical_margins) >= 5:
        avg_margin = sum(historical_margins) / len(historical_margins)
        recent_margins = historical_margins[:3]
        older_margins = historical_margins[-3:]

        recent_avg = sum(recent_margins) / len(recent_margins)
        older_avg = sum(older_margins) / len(older_margins)

        if avg_margin > 0.2 and recent_avg >= older_avg:
            moat_score += 1
            reasoning.append(f"强劲且稳定的营业利润率（平均：{avg_margin:.1%}）表明拥有定价权护城河")
        elif avg_margin > 0.15:
            reasoning.append(f"尚可的营业利润率（平均：{avg_margin:.1%}）暗示一定竞争优势")
        else:
            reasoning.append(f"低营业利润率（平均：{avg_margin:.1%}）表明定价权有限")

    # 3. 资产效率
    if len(metrics) >= 5:
        asset_turnovers = [m.asset_turnover for m in metrics if m.asset_turnover is not None]

        if len(asset_turnovers) >= 3 and any(t > 1.0 for t in asset_turnovers):
            moat_score += 1
            reasoning.append("高效的资产利用表明运营护城河")

    # 4. 竞争地位强度
    if len(historical_roes) >= 5 and len(historical_margins) >= 5:
        roe_avg = sum(historical_roes) / len(historical_roes)
        roe_variance = sum((roe - roe_avg) ** 2 for roe in historical_roes) / len(historical_roes)
        roe_stability = 1 - (roe_variance**0.5) / roe_avg if roe_avg > 0 else 0

        margin_avg = sum(historical_margins) / len(historical_margins)
        margin_variance = sum((margin - margin_avg) ** 2 for margin in historical_margins) / len(
            historical_margins
        )
        margin_stability = 1 - (margin_variance**0.5) / margin_avg if margin_avg > 0 else 0

        overall_stability = (roe_stability + margin_stability) / 2

        if overall_stability > 0.7:
            moat_score += 1
            reasoning.append(f"高业绩稳定性（{overall_stability:.1%}）表明强劲的竞争护城河")

    moat_score = min(moat_score, max_score)

    return {
        "score": moat_score,
        "max_score": max_score,
        "details": "; ".join(reasoning) if reasoning else "护城河分析信息有限",
    }


def analyze_management_quality(financial_line_items: list[LineItem]) -> dict:
    """检查股份稀释或持续回购，以及股息支付记录。"""
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "管理分析数据不足"}

    reasoning = []
    mgmt_score = 0

    latest = financial_line_items[0]
    latest_data = latest.model_dump()

    # 检查股份回购
    if (
        latest_data.get("issuance_or_purchase_of_equity_shares")
        and latest_data["issuance_or_purchase_of_equity_shares"] < 0
    ):
        mgmt_score += 1
        reasoning.append("公司一直在回购股票（对股东友好）")
    elif latest_data.get("issuance_or_purchase_of_equity_shares"):
        reasoning.append("近期发行了普通股（潜在稀释）")
    else:
        reasoning.append("未检测到重大新股发行")

    # 检查股息
    if (
        latest_data.get("dividends_and_other_cash_distributions")
        and latest_data["dividends_and_other_cash_distributions"] < 0
    ):
        mgmt_score += 1
        reasoning.append("公司有支付股息的记录")
    else:
        reasoning.append("未支付或支付极少的股息")

    return {"score": mgmt_score, "max_score": 2, "details": "; ".join(reasoning)}


def calculate_owner_earnings(financial_line_items: list[LineItem]) -> dict:
    """计算所有者收益（价值投资首选的真实盈利能力指标）。"""
    if not financial_line_items or len(financial_line_items) < 2:
        return {"owner_earnings": None, "details": ["数据不足以计算所有者收益"]}

    latest = financial_line_items[0]
    latest_data = latest.model_dump()
    details = []

    # 核心组成部分
    net_income = latest_data.get("net_income")
    depreciation = latest_data.get("depreciation_and_amortization")
    capex = latest_data.get("capital_expenditure")

    if not all([net_income is not None, depreciation is not None, capex is not None]):
        missing = []
        if net_income is None:
            missing.append("net income")
        if depreciation is None:
            missing.append("depreciation")
        if capex is None:
            missing.append("capital expenditure")
        return {"owner_earnings": None, "details": [f"Missing components: {', '.join(missing)}"]}

    # 估算维护性资本支出
    maintenance_capex = estimate_maintenance_capex(financial_line_items)

    # 计算所有者收益
    owner_earnings = net_income + depreciation - maintenance_capex

    details.extend([
        f"Net income: ${net_income:,.0f}",
        f"Depreciation: ${depreciation:,.0f}",
        f"Estimated maintenance capex: ${maintenance_capex:,.0f}",
        f"Owner earnings: ${owner_earnings:,.0f}",
    ])

    return {
        "owner_earnings": owner_earnings,
        "components": {
            "net_income": net_income,
            "depreciation": depreciation,
            "maintenance_capex": maintenance_capex,
            "total_capex": abs(capex) if capex else 0,
        },
        "details": details,
    }


def estimate_maintenance_capex(financial_line_items: list[LineItem]) -> float:
    """估算维护性资本支出。"""
    if not financial_line_items:
        return 0

    latest_data = financial_line_items[0].model_dump()
    latest_depreciation = latest_data.get("depreciation_and_amortization", 0)
    latest_capex = abs(latest_data.get("capital_expenditure", 0))

    # 使用总资本支出的85%作为维护性支出
    method_1 = latest_capex * 0.85
    method_2 = latest_depreciation

    return max(method_1, method_2)


def calculate_intrinsic_value(financial_line_items: list[LineItem]) -> dict:
    """使用所有者收益的DCF模型计算内在价值。"""
    if not financial_line_items or len(financial_line_items) < 3:
        return {"intrinsic_value": None, "details": ["数据不足以进行可靠估值"]}

    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]
    latest = financial_line_items[0]
    latest_data = latest.model_dump()
    shares_outstanding = latest_data.get("outstanding_shares")

    if not shares_outstanding or shares_outstanding <= 0:
        return {"intrinsic_value": None, "details": ["流通股数据缺失或无效"]}

    # 估算增长率
    historical_earnings = []
    for item in financial_line_items[:5]:
        if item.model_dump().get("net_income"):
            historical_earnings.append(item.model_dump()["net_income"])

    if len(historical_earnings) >= 3:
        oldest_earnings = historical_earnings[-1]
        latest_earnings = historical_earnings[0]
        years = len(historical_earnings) - 1

        if oldest_earnings > 0:
            historical_growth = ((latest_earnings / oldest_earnings) ** (1 / years)) - 1
            historical_growth = max(-0.05, min(historical_growth, 0.15))
            conservative_growth = historical_growth * 0.7
        else:
            conservative_growth = 0.03
    else:
        conservative_growth = 0.03

    # 保守假设
    stage1_growth = min(conservative_growth, 0.08)
    stage2_growth = min(conservative_growth * 0.5, 0.04)
    terminal_growth = 0.025
    discount_rate = 0.10

    # 三阶段DCF
    stage1_years = 5
    stage2_years = 5

    # 第一阶段
    stage1_pv = 0
    for year in range(1, stage1_years + 1):
        future_earnings = owner_earnings * (1 + stage1_growth) ** year
        pv = future_earnings / (1 + discount_rate) ** year
        stage1_pv += pv

    # 第二阶段
    stage2_pv = 0
    stage1_final_earnings = owner_earnings * (1 + stage1_growth) ** stage1_years
    for year in range(1, stage2_years + 1):
        future_earnings = stage1_final_earnings * (1 + stage2_growth) ** year
        pv = future_earnings / (1 + discount_rate) ** (stage1_years + year)
        stage2_pv += pv

    # 终值
    final_earnings = stage1_final_earnings * (1 + stage2_growth) ** stage2_years
    terminal_earnings = final_earnings * (1 + terminal_growth)
    terminal_value = terminal_earnings / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / (1 + discount_rate) ** (stage1_years + stage2_years)

    intrinsic_value = stage1_pv + stage2_pv + terminal_pv
    conservative_intrinsic_value = intrinsic_value * 0.85

    return {
        "intrinsic_value": conservative_intrinsic_value,
        "raw_intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "stage1_growth": stage1_growth,
            "stage2_growth": stage2_growth,
            "terminal_growth": terminal_growth,
            "discount_rate": discount_rate,
        },
    }


def analyze_pricing_power(financial_line_items: list[LineItem], metrics: list[FinancialMetrics]) -> dict:
    """分析定价权——企业护城河的关键指标。"""
    if not financial_line_items or not metrics:
        return {"score": 0, "details": "定价权分析数据不足"}

    score = 0
    reasoning = []

    # 检查毛利率趋势
    gross_margins = [
        item.model_dump().get("gross_margin")
        for item in financial_line_items
        if item.model_dump().get("gross_margin") is not None
    ]

    if len(gross_margins) >= 3:
        recent_avg = sum(gross_margins[:2]) / 2
        older_avg = sum(gross_margins[-2:]) / 2

        if recent_avg > older_avg + 0.02:
            score += 3
            reasoning.append("毛利率扩张表明定价权强劲")
        elif recent_avg > older_avg:
            score += 2
            reasoning.append("毛利率改善表明定价权良好")
        elif abs(recent_avg - older_avg) < 0.01:
            score += 1
            reasoning.append("经济不确定性期间毛利率稳定")
        else:
            reasoning.append("毛利率下降可能表明定价压力")

    if gross_margins:
        avg_margin = sum(gross_margins) / len(gross_margins)
        if avg_margin > 0.5:
            score += 2
            reasoning.append(f"持续高毛利率（{avg_margin:.1%}）表明定价权强劲")
        elif avg_margin > 0.3:
            score += 1
            reasoning.append(f"良好的毛利率（{avg_margin:.1%}）表明定价权尚可")

    return {"score": score, "details": "; ".join(reasoning) if reasoning else "定价权分析信息有限"}


def analyze_book_value_growth(financial_line_items: list[LineItem]) -> dict:
    """分析每股账面价值增长——价值投资的关键指标。"""
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "账面价值分析数据不足"}

    # 提取每股账面价值
    book_values = []
    for item in financial_line_items:
        data = item.model_dump()
        equity = data.get("shareholders_equity")
        shares = data.get("outstanding_shares")
        if equity and shares and shares > 0:
            book_values.append(equity / shares)

    if len(book_values) < 3:
        return {"score": 0, "details": "账面价值数据不足以进行增长分析"}

    score = 0
    reasoning = []

    # 分析增长一致性
    growth_periods = sum(1 for i in range(len(book_values) - 1) if book_values[i] > book_values[i + 1])
    growth_rate = growth_periods / (len(book_values) - 1)

    if growth_rate >= 0.8:
        score += 3
        reasoning.append("每股账面价值持续增长")
    elif growth_rate >= 0.6:
        score += 2
        reasoning.append("每股账面价值增长模式良好")
    elif growth_rate >= 0.4:
        score += 1
        reasoning.append("每股账面价值中等增长")
    else:
        reasoning.append("每股账面价值增长不稳定")

    # 计算CAGR
    if len(book_values) >= 2:
        oldest_bv, latest_bv = book_values[-1], book_values[0]
        years = len(book_values) - 1

        if oldest_bv > 0 and latest_bv > 0:
            cagr = ((latest_bv / oldest_bv) ** (1 / years)) - 1
            if cagr > 0.15:
                score += 2
                reasoning.append(f"卓越的账面价值CAGR：{cagr:.1%}")
            elif cagr > 0.1:
                score += 1
                reasoning.append(f"良好的账面价值CAGR：{cagr:.1%}")
            else:
                reasoning.append(f"账面价值CAGR：{cagr:.1%}")

    return {"score": score, "details": "; ".join(reasoning)}


def generate_value_investor_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
    market_cap: float | None = None,
) -> dict:
    """生成价值投资分析提示词。"""

    # Run all analyses
    fundamental_analysis = analyze_fundamentals(financial_metrics)
    consistency_analysis = analyze_consistency(financial_line_items)
    moat_analysis = analyze_moat(financial_metrics)
    management_analysis = analyze_management_quality(financial_line_items)
    pricing_power_analysis = analyze_pricing_power(financial_line_items, financial_metrics)
    book_value_analysis = analyze_book_value_growth(financial_line_items)
    intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)

    # Calculate total score
    total_score = (
        fundamental_analysis["score"]
        + consistency_analysis["score"]
        + moat_analysis["score"]
        + management_analysis["score"]
        + pricing_power_analysis["score"]
        + book_value_analysis["score"]
    )

    max_possible_score = (
        10  # fundamental_analysis
        + moat_analysis["max_score"]
        + management_analysis["max_score"]
        + 5  # pricing_power
        + 5  # book_value_growth
    )

    # Calculate margin of safety
    margin_of_safety = None
    intrinsic_value = intrinsic_value_analysis.get("intrinsic_value")
    if intrinsic_value and market_cap:
        margin_of_safety = (intrinsic_value - market_cap) / market_cap

    # Build facts
    facts = {
        "score": total_score,
        "max_score": max_possible_score,
        "fundamentals": fundamental_analysis.get("details"),
        "consistency": consistency_analysis.get("details"),
        "moat": moat_analysis.get("details"),
        "pricing_power": pricing_power_analysis.get("details"),
        "book_value": book_value_analysis.get("details"),
        "management": management_analysis.get("details"),
        "intrinsic_value": intrinsic_value,
        "market_cap": market_cap,
        "margin_of_safety": margin_of_safety,
    }

    # Build prompt (中文)
    system_prompt = """你是价值投资类分析师。仅根据提供的事实决定看涨、看跌或中性。

决策检查清单：
- 能力圈范围
- 竞争护城河
- 管理层质量
- 财务实力
- 估值与内在价值对比
- 长期前景

信号规则：
- 看涨：优质业务 且 margin_of_safety > 0（安全边际为正）
- 看跌：劣质业务 或 明显高估
- 中性：优质业务但 margin_of_safety <= 0，或信号混合

信心度标尺：
- 90-100%：能力圈内的卓越企业，价格诱人
- 70-89%：具有良好护城河的优质企业，估值合理
- 50-69%：信号混合，需要更多信息或更好的价格
- 30-49%：超出我的能力圈或基本面令人担忧
- 10-29%：劣质业务或严重高估

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例字符串（避免在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"value_investor"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{"signal": "bullish" | "bearish" | "neutral", "confidence": int, "reasoning": "简短理由", "analyst_name": "value_investor"}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
