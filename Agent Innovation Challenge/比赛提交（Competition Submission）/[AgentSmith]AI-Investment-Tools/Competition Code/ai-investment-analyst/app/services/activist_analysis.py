import json

from app.models.data import FinancialMetrics
from app.models.analyst import LineItem


def analyze_undervalued_assets(line_items: list[LineItem], market_cap: float) -> dict:
    """分析被低估资产 - 激进投资者的机会识别。"""
    if not line_items or not market_cap:
        return {"score": 0, "details": "资产价值分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    score = 0
    details = []

    # 净资产价值
    shareholders_equity = latest_data.get("shareholders_equity") or 0

    if shareholders_equity > 0:
        pb = market_cap / shareholders_equity
        if pb < 1.0:
            score += 3
            details.append(f"市净率{pb:.2f} < 1，资产被低估")
        elif pb < 1.5:
            score += 2
            details.append(f"低市净率{pb:.2f}")

    # 营运价值
    ebitda = latest_data.get("ebitda") or 0
    if ebitda > 0:
        debt = latest_data.get("total_liabilities") or 0
        cash = latest_data.get("cash_and_equivalents") or 0
        ev = market_cap + debt - cash
        ev_ebitda = ev / ebitda

        if ev_ebitda < 8:
            score += 3
            details.append(f"低EV/EBITDA {ev_ebitda:.1f}")
        elif ev_ebitda < 12:
            score += 2
            details.append(f"中等EV/EBITDA {ev_ebitda:.1f}")

    return {"score": score, "max_score": 6, "details": "; ".join(details) if details else "无资产价值数据"}


def analyze_catalyst_potential(metrics: list[FinancialMetrics]) -> dict:
    """分析催化剂潜力 - 可以释放价值的事件。"""
    if not metrics:
        return {"score": 0, "details": "催化剂分析数据不足"}

    score = 0
    details = []

    latest = metrics[0]

    # 低效率 = 价值释放机会
    if latest.asset_turnover and latest.asset_turnover < 0.5:
        score += 3
        details.append(f"低资产周转率{latest.asset_turnover:.2f}，运营改进潜力大")
    elif latest.asset_turnover and latest.asset_turnover < 0.8:
        score += 2
        details.append(f"中等资产周转率{latest.asset_turnover:.2f}")

    # 低利润率 = 优化机会
    if latest.operating_margin and latest.operating_margin < 0.10:
        score += 2
        details.append(f"低营业利润率{latest.operating_margin:.1%}，利润改善空间大")
    elif latest.operating_margin and latest.operating_margin < 0.15:
        score += 1
        details.append(f"中等营业利润率{latest.operating_margin:.1%}")

    return {"score": score, "max_score": 5, "details": "; ".join(details) if details else "无催化剂数据"}


def analyze_balance_sheet_strength(line_items: list[LineItem]) -> dict:
    """分析资产负债表强度 - 为激进行动提供基础。"""
    if not line_items:
        return {"score": 0, "details": "资产负债表分析数据不足"}

    latest = line_items[0]
    latest_data = latest.model_dump()

    score = 0
    details = []

    # 现金储备
    cash = latest_data.get("cash_and_equivalents") or 0
    current_liabilities = latest_data.get("current_liabilities") or 0

    if current_liabilities > 0:
        cash_ratio = cash / current_liabilities
        if cash_ratio > 1.0:
            score += 3
            details.append(f"强劲现金比率{cash_ratio:.2f}")
        elif cash_ratio > 0.5:
            score += 2
            details.append(f"良好现金比率{cash_ratio:.2f}")

    # 债务水平（不要太高以允许激进行动）
    total_assets = latest_data.get("total_assets") or 0
    total_liabilities = latest_data.get("total_liabilities") or 0

    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.3:
            score += 2
            details.append(f"低债务比率{debt_ratio:.2f}，激进行动空间大")
        elif debt_ratio < 0.5:
            score += 1
            details.append(f"中等债务比率{debt_ratio:.2f}")

    return {"score": score, "max_score": 5, "details": "; ".join(details) if details else "无资产负债表数据"}


def generate_activist_prompt(
    ticker: str,
    financial_metrics: list[FinancialMetrics],
    financial_line_items: list[LineItem],
    market_cap: float,
) -> dict:
    """生成激进投资分析提示词。"""

    undervalued = analyze_undervalued_assets(financial_line_items, market_cap)
    catalyst = analyze_catalyst_potential(financial_metrics)
    balance_sheet = analyze_balance_sheet_strength(financial_line_items)

    total_score = undervalued["score"] + catalyst["score"] + balance_sheet["score"]
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
        "undervalued_assets": undervalued,
        "catalyst_potential": catalyst,
        "balance_sheet_strength": balance_sheet,
    }

    system_prompt = """你是激进投资类分析师。基于以下原则做出投资决策：

1. 被低估资产：寻找市净率<1或低EV/EBITDA的公司
2. 催化剂潜力：识别运营改进和利润提升机会
3. 资产负债表强度：确保有财务能力推动变革

决策规则：
- 看涨：显著被低估 + 明确催化剂 + 强资产负债表
- 看跌：估值合理或无明显催化剂
- 中性：信号混合

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"activist"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "activist"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
