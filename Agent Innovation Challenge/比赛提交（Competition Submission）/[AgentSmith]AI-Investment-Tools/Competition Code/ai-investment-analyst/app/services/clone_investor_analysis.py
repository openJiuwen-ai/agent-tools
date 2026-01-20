import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_value_clone_opportunity(metrics: list[FinancialMetrics], line_items: list[LineItem], market_cap: float) -> dict:
    """分析价值克隆机会 - 复制成功投资者的策略。"""
    if not metrics or not line_items:
        return {"score": 0, "details": "克隆机会分析数据不足"}

    score = 0
    details = []

    latest = metrics[0]
    latest_item = line_items[0]
    latest_data = latest_item.model_dump()

    # 低估值 + 质量 = 克隆机会
    if latest.return_on_equity and latest.return_on_equity > 0.15:
        shares = latest_data.get("outstanding_shares") or 1
        net_income = latest.net_income or latest_data.get("net_income") or 0

        if net_income > 0:
            eps = net_income / shares
            price = market_cap / shares
            pe = price / eps if eps > 0 else 999

            if pe < 15 and latest.return_on_equity > 0.20:
                score += 4
                details.append(f"低PE{pe:.1f} + 高ROE{latest.return_on_equity:.1%}，经典价值机会")
            elif pe < 20:
                score += 3
                details.append(f"合理PE{pe:.1f} + 良好ROE{latest.return_on_equity:.1%}")
            elif latest.return_on_equity > 0.20:
                score += 2
                details.append(f"高ROE{latest.return_on_equity:.1%}")

    # 简单业务 = 易于理解
    if latest.gross_margin and latest.gross_margin > 0.40:
        score += 2
        details.append(f"高毛利率{latest.gross_margin:.1%}，业务模式清晰")

    return {"score": score, "max_score": 6, "details": "; ".join(details) if details else "无克隆机会数据"}


def analyze_low_risk_high_uncertainty(line_items: list[LineItem]) -> dict:
    """分析低风险高不确定性 - Pabrai的"Heads I Win, Tails I Don't Lose Much"原则。"""
    if not line_items:
        return {"score": 0, "details": "风险不确定性分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    score = 0
    details = []

    # 净净安全边际
    current_assets = latest_data.get("current_assets") or 0
    total_liabilities = latest_data.get("total_liabilities") or 0

    net_current_assets = current_assets - total_liabilities

    if net_current_assets > 0:
        score += 3
        details.append(f"正向净流动资产{net_current_assets:,.0f}，下行保护强")

    # 稳定现金流
    fcf = latest_data.get("free_cash_flow") or 0
    if fcf > 0:
        score += 2
        details.append("正向自由现金流，财务安全")

    # 低债务
    if total_liabilities > 0:
        total_assets = latest_data.get("total_assets") or 1
        debt_ratio = total_liabilities / total_assets

        if debt_ratio < 0.3:
            score += 2
            details.append(f"低债务比率{debt_ratio:.2f}，风险低")

    return {"score": score, "max_score": 7, "details": "; ".join(details) if details else "无风险不确定性数据"}


def analyze_management_alignment(line_items: list[LineItem]) -> dict:
    """分析管理层一致性。"""
    if not line_items:
        return {"score": 0, "details": "管理层一致性分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    score = 0
    details = []

    # 股份回购
    equity_shares = latest_data.get("issuance_or_purchase_of_equity_shares")
    if equity_shares and equity_shares < 0:
        score += 2
        details.append("公司回购股票，管理层与股东利益一致")
    elif equity_shares and equity_shares > 0:
        details.append("公司发行新股，潜在稀释")

    # 股息
    dividends = latest_data.get("dividends_and_other_cash_distributions")
    if dividends and dividends < 0:
        score += 1
        details.append("支付股息，股东友好")

    return {"score": score, "max_score": 3, "details": "; ".join(details) if details else "无管理层一致性数据"}


def generate_clone_investor_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
    market_cap: float,
) -> dict:
    """生成克隆投资分析提示词。"""

    clone_opportunity = analyze_value_clone_opportunity(financial_metrics, financial_line_items, market_cap)
    low_risk = analyze_low_risk_high_uncertainty(financial_line_items)
    management = analyze_management_alignment(financial_line_items)

    total_score = clone_opportunity["score"] + low_risk["score"] + management["score"]
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
        "clone_opportunity": clone_opportunity,
        "low_risk_high_uncertainty": low_risk,
        "management_alignment": management,
    }

    system_prompt = """你是克隆投资类分析师。基于以下原则做出投资决策：

1. 价值克隆机会：低估值+高质量的组合
2. 低风险高不确定性："Heads I Win, Tails I Don't Lose Much"
3. 管理层一致性：回购和股息显示股东友好

决策规则：
- 看涨：低估值 + 强下行保护 + 管理层一致
- 看跌：估值高或下行风险大
- 中性：信号混合

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"clone_investor"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "clone_investor"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
