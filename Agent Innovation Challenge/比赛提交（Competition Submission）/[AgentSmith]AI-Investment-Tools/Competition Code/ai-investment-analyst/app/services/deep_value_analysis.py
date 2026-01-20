import json
import math

from app.models.analyst import LineItem
from app.models.data import FinancialMetrics
from app.services.data_fetching import DataFetchingError


def analyze_earnings_stability(metrics: list[FinancialMetrics], financial_line_items: list[LineItem]) -> dict:
    """分析盈利稳定性 - 深度价值投资的核心指标。"""
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": score, "details": "盈利稳定性分析数据不足"}

    eps_vals = []
    for item in financial_line_items:
        eps = item.model_dump().get("earnings_per_share")
        if eps is not None:
            eps_vals.append(eps)

    if len(eps_vals) < 2:
        details.append("多年EPS数据不足")
        return {"score": score, "details": "; ".join(details)}

    # 持续正EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append("EPS在所有可用期间均为正")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append("EPS在大多数期间为正")
    else:
        details.append("EPS在多个期间为负")

    # EPS增长
    if eps_vals[0] > eps_vals[-1]:
        score += 1
        details.append("EPS从最早到最新期间有所增长")
    else:
        details.append("EPS从最早到最新期间未增长")

    return {"score": score, "details": "; ".join(details)}


def analyze_financial_strength(financial_line_items: list[LineItem]) -> dict:
    """分析财务实力 - 流动性、债务和股息记录。"""
    score = 0
    details = []

    if not financial_line_items:
        return {"score": score, "details": "财务实力分析数据不足"}

    latest_item = financial_line_items[0]
    latest_data = latest_item.model_dump()
    total_assets = latest_data.get("total_assets") or 0
    total_liabilities = latest_data.get("total_liabilities") or 0
    current_assets = latest_data.get("current_assets") or 0
    current_liabilities = latest_data.get("current_liabilities") or 0

    # 流动比率
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"流动比率 = {current_ratio:.2f} (>=2.0：强劲)")
        elif current_ratio >= 1.5:
            score += 1
            details.append(f"流动比率 = {current_ratio:.2f} (中等强劲)")
        else:
            details.append(f"流动比率 = {current_ratio:.2f} (<1.5：流动性较弱)")
    else:
        details.append("无法计算流动比率（流动负债缺失或为零）")

    # 债务比率
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"债务比率 = {debt_ratio:.2f}，低于0.50（保守）")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"债务比率 = {debt_ratio:.2f}，略高但可接受")
        else:
            details.append(f"债务比率 = {debt_ratio:.2f}，按深度价值标准较高")
    else:
        details.append("无法计算债务比率（总资产缺失）")

    # 股息记录
    div_periods = [
        item.model_dump().get("dividends_and_other_cash_distributions")
        for item in financial_line_items
        if item.model_dump().get("dividends_and_other_cash_distributions") is not None
    ]
    if div_periods:
        div_paid_years = sum(1 for d in div_periods if d < 0)
        if div_paid_years > 0:
            if div_paid_years >= (len(div_periods) // 2 + 1):
                score += 1
                details.append("公司在大多数报告年度支付股息")
            else:
                details.append("公司有一些股息支付，但非大多数年份")
        else:
            details.append("公司在此期间未支付股息")
    else:
        details.append("无股息数据可用于评估支付一致性")

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation_deep_value(financial_line_items: list[LineItem], market_cap: float | None) -> dict:
    """深度价值估值分析：净净法和格雷厄姆数。"""
    if not financial_line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "details": "估值数据不足"}

    latest = financial_line_items[0]
    latest_data = latest.model_dump()
    current_assets = latest_data.get("current_assets") or 0
    total_liabilities = latest_data.get("total_liabilities") or 0
    book_value_ps = latest_data.get("book_value_per_share") or 0
    eps = latest_data.get("earnings_per_share") or 0
    shares_outstanding = latest_data.get("outstanding_shares") or 0

    details = []
    score = 0

    # 净净法检查
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding

        details.append(f"净流动资产价值 = {net_current_asset_value:,.2f}")
        details.append(f"每股NCAV = {net_current_asset_value_per_share:,.2f}")
        details.append(f"每股价格 = {price_per_share:,.2f}")

        if net_current_asset_value > market_cap:
            score += 4
            details.append("净净法：NCAV > 市值（经典深度价值）")
        else:
            if net_current_asset_value_per_share >= (price_per_share * 0.67):
                score += 2
                details.append("每股NCAV >= 每股价格的2/3（中等净净折扣）")
    else:
        details.append("NCAV未超过市值或净净法数据不足")

    # 格雷厄姆数
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"格雷厄姆数 = {graham_number:.2f}")
    else:
        details.append("无法计算格雷厄姆数（EPS或账面价值缺失/<=0）")

    # 相对于格雷厄姆数的安全边际
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"安全边际（格雷厄姆数）= {margin_of_safety:.2%}")
            if margin_of_safety > 0.5:
                score += 3
                details.append("价格远低于格雷厄姆数（>=50%安全边际）")
            elif margin_of_safety > 0.2:
                score += 1
                details.append("相对于格雷厄姆数有一定安全边际")
            else:
                details.append("价格接近或高于格雷厄姆数，安全边际较低")

    return {"score": score, "details": "; ".join(details)}


def generate_deep_value_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
    market_cap: float | None = None,
) -> dict:
    """生成深度价值投资分析提示词。"""

    # 执行所有分析
    earnings_analysis = analyze_earnings_stability(financial_metrics, financial_line_items)
    strength_analysis = analyze_financial_strength(financial_line_items)
    valuation_analysis = analyze_valuation_deep_value(financial_line_items, market_cap)

    # 汇总评分
    total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
    max_possible_score = 15

    # 映射总评分到信号
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
        "earnings_analysis": earnings_analysis,
        "strength_analysis": strength_analysis,
        "valuation_analysis": valuation_analysis,
    }

    # 构建提示词（中文）
    system_prompt = """你是深度价值投资类分析师。基于以下原则做出投资决策：

1. 坚持安全边际，以低于内在价值的价格买入（如格雷厄姆数、净净法）
2. 强调公司财务实力（低杠杆、充足的流动资产）
3. 要求多年稳定盈利
4. 考虑股息记录以增加安全性
5. 避免投机性或高增长假设；专注于经过验证的指标

决策规则：
- 看涨：评分 >= 10分（总分15分），显示强劲的深度价值特征
- 看跌：评分 <= 5分，缺乏深度价值特征
- 中性：评分在6-9分之间，混合信号

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"deep_value"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "deep_value"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
