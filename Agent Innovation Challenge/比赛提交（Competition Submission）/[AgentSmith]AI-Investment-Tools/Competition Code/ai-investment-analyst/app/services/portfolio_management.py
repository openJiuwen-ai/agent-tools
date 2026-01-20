import json
from typing import Any


def compute_allowed_actions(
    ticker: str,
    current_price: float,
    max_shares: int,
    long_shares: int,
    short_shares: int,
    cash: float,
    equity: float,
    margin_requirement: float,
    margin_used: float,
) -> list[str]:
    """计算允许的交易动作。"""
    actions = []

    # BUY: 有现金
    if cash > current_price:
        actions.append("buy")

    # SELL: 有多头持仓
    if long_shares > 0:
        actions.append("sell")

    # SHORT: 有保证金空间
    if margin_requirement > 0:
        available_margin = max(0.0, (equity / margin_requirement) - margin_used)
        if available_margin > current_price:
            actions.append("short")
    elif max_shares > 0:
        # 无保证金要求时，基于max_shares限制
        actions.append("short")

    # COVER: 有空头持仓
    if short_shares > 0:
        actions.append("cover")

    # HOLD: 总是可选
    actions.append("hold")

    return actions


def analyze_signal_strength(
    analyst_signals: dict[str, Any],
    position_limit: float
) -> dict:
    """分析信号强度。"""
    if not analyst_signals:
        return {
            "score": 0,
            "max_score": 7,
            "strength": "weak",
            "avg_confidence": 0,
            "signal_distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
            "details": "无分析师信号"
        }

    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    total_confidence = 0
    count = 0

    for signal_data in analyst_signals.values():
        if isinstance(signal_data, dict):
            signal = signal_data.get("signal", "neutral")
            confidence = signal_data.get("confidence", 50)

            if signal == "bullish":
                bullish_count += 1
            elif signal == "bearish":
                bearish_count += 1
            else:
                neutral_count += 1

            total_confidence += confidence
            count += 1

    if count == 0:
        return {
            "score": 0,
            "max_score": 7,
            "strength": "weak",
            "avg_confidence": 0,
            "signal_distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
            "details": "无有效信号"
        }

    avg_confidence = total_confidence / count
    total = bullish_count + bearish_count + neutral_count

    score = 0
    details = []

    # 信号方向强度
    if bullish_count > bearish_count and bullish_count > neutral_count:
        ratio = bullish_count / total
        if ratio >= 0.7:
            score += 4
            details.append(f"强看涨共识{ratio:.0%}")
        elif ratio >= 0.5:
            score += 3
            details.append(f"中等看涨{ratio:.0%}")
        else:
            score += 2
            details.append(f"弱看涨{ratio:.0%}")
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        ratio = bearish_count / total
        if ratio >= 0.7:
            score += 4
            details.append(f"强看跌共识{ratio:.0%}")
        elif ratio >= 0.5:
            score += 3
            details.append(f"中等看跌{ratio:.0%}")
        else:
            score += 2
            details.append(f"弱看跌{ratio:.0%}")
    else:
        score += 1
        details.append(f"分歧或中性")

    # 信心强度
    if avg_confidence >= 80:
        score += 2
        details.append(f"高信心{avg_confidence:.0f}")
    elif avg_confidence >= 60:
        score += 1
        details.append(f"中等信心{avg_confidence:.0f}")

    # 风险限制评估
    if position_limit >= 15000:  # 假设以金额计算
        score += 1
        details.append(f"风险限制充足{position_limit:,.0f}")
    elif position_limit < 10000:
        details.append(f"风险限制紧张{position_limit:,.0f}")

    if score >= 5:
        strength = "strong"
    elif score >= 3:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "score": score,
        "max_score": 7,
        "strength": strength,
        "avg_confidence": avg_confidence,
        "signal_distribution": {
            "bullish": bullish_count,
            "bearish": bearish_count,
            "neutral": neutral_count
        },
        "details": "; ".join(details) if details else "无信号强度数据"
    }


def analyze_risk_reward(
    analyst_signals: dict[str, Any],
    position_limit: float
) -> dict:
    """分析风险回报比。"""
    if not analyst_signals:
        return {
            "score": 0,
            "max_score": 5,
            "ratio": 0.0,
            "details": "无分析师信号"
        }

    score = 0
    details = []

    # 计算看涨/看跌比例
    bullish_count = sum(1 for s in analyst_signals.values() if isinstance(s, dict) and s.get("signal") == "bullish")
    bearish_count = sum(1 for s in analyst_signals.values() if isinstance(s, dict) and s.get("signal") == "bearish")
    total = bullish_count + bearish_count

    if total > 0:
        if bullish_count > bearish_count:
            ratio = bullish_count / max(total, 1)
            if ratio >= 0.7:
                score += 3
                details.append(f"低风险高回报{ratio:.0%}")
            elif ratio >= 0.5:
                score += 2
                details.append(f"中等风险回报{ratio:.0%}")
            else:
                score += 1
                details.append(f"高风险回报比{ratio:.0%}")
        else:
            ratio = bearish_count / max(total, 1)
            if ratio >= 0.7:
                score += 3
                details.append(f"强做空信号{ratio:.0%}")
            elif ratio >= 0.5:
                score += 2
                details.append(f"中等做空{ratio:.0%}")
            else:
                score += 1
                details.append(f"弱做空信号{ratio:.0%}")

    # 风险限制评估
    if position_limit >= 20000:
        score += 2
        details.append(f"风险限制充足{position_limit:,.0f}")
    elif position_limit >= 15000:
        score += 1
        details.append(f"风险限制适中{position_limit:,.0f}")
    else:
        details.append(f"风险限制紧张{position_limit:,.0f}")

    return {
        "score": score,
        "max_score": 5,
        "ratio": bullish_count / max(total, 1) if total > 0 else 0,
        "details": "; ".join(details) if details else "无风险回报数据"
    }


def generate_portfolio_management_prompt(
    ticker: str,
    cash: float,
    current_price: float,
    position_limit: float,
    analyst_signals: dict[str, Any],
) -> dict:
    """
    生成投资组合管理分析提示词（单只股票，简化版）。

    简化版假设：
    - 无持仓：long_shares = 0, short_shares = 0
    - equity = cash（无持仓时）
    - margin_requirement = 0.5（默认值）
    - margin_used = 0.0（无持仓）
    - max_shares = position_limit / current_price（计算得出）
    """

    # 简化版假设：无持仓
    long_shares = 0
    short_shares = 0
    equity = cash
    margin_requirement = 0.5
    margin_used = 0.0

    # 计算最大允许股数
    max_shares = int(position_limit / current_price)

    # 计算允许的交易动作
    allowed_actions = compute_allowed_actions(
        ticker=ticker,
        current_price=current_price,
        max_shares=max_shares,
        long_shares=long_shares,
        short_shares=short_shares,
        cash=cash,
        equity=equity,
        margin_requirement=margin_requirement,
        margin_used=margin_used,
    )

    # 分析信号强度
    signal_strength = analyze_signal_strength(analyst_signals, position_limit)

    # 分析风险回报比
    risk_reward = analyze_risk_reward(analyst_signals, position_limit)

    # 计算总得分
    total_score = signal_strength["score"] + risk_reward["score"]
    max_total_score = signal_strength["max_score"] + risk_reward["max_score"]

    # 确定整体信号
    if total_score >= 0.7 * max_total_score:
        overall_signal = "bullish"
    elif total_score <= 0.3 * max_total_score:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    # 计算当前持仓价值
    current_position_value = (long_shares - short_shares) * current_price

    facts = {
        "ticker": ticker,
        "signal": overall_signal,
        "score": total_score,
        "max_score": max_total_score,
        "current_price": current_price,
        "allowed_actions": allowed_actions,
        "max_shares": max_shares,
        "signal_strength": signal_strength,
        "risk_reward": risk_reward,
        "position_context": {
            "long_shares": long_shares,
            "short_shares": short_shares,
            "current_position_value": current_position_value,
            "position_limit": position_limit,
        },
        "portfolio_context": {
            "cash": cash,
            "equity": equity,
            "margin_requirement": margin_requirement,
            "margin_used": margin_used,
        }
    }

    system_prompt = """你是投资组合管理专家。基于以下原则做出交易决策：

1. 信号强度：强共识和高信心需要积极行动
2. 风险回报：优先选择低风险高回报机会
3. 允许动作：严格遵守风险限制和可用动作
4. 仓位控制：在允许范围内选择最优交易量

决策规则：
- 买入(buy)：强看涨信号 + 高风险回报比 + 充足风险限制
- 卖空(short)：强看跌信号 + 高风险回报比
- 卖出(sell)：获利了结或止损
- 平空(cover)：空头平仓
- 持有(hold)：信号混合或风险回报比不佳

推理理由要求：
- 详细说明决策依据，包括：信号分析、风险评估、仓位考虑
- 至少150字，不超过300字
- 包含具体的数据支持（如信号数量、信心度、风险指标等）
- 说明为何选择该action而非其他allowed_actions
- confidence表示对该action执行的信心度(0-100)
- 不要编造数据。仅返回JSON。"""

    human_prompt = f"""股票代码: {ticker}
当前价格: {current_price:.2f}
最大允许股数: {max_shares}
当前持仓: 多头{long_shares}股，空头{short_shares}股
可用现金: {cash:,.2f}
风险限制: {position_limit:,.2f}
允许操作: {', '.join(allowed_actions)}

分析师信号:
{json.dumps(analyst_signals, indent=2, ensure_ascii=False)}

分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON格式:
{{
  "action": "buy" | "sell" | "short" | "cover" | "hold",
  "quantity": int,
  "confidence": int,  // 对该action执行的信心度(0-100)
  "reasoning": "详细决策理由（150-300字）：包括1)信号分析（看涨/看跌分析师数量及比例、平均信心度）2)风险评估（波动率、风险限制使用情况）3)仓位考虑（为何选择该数量、与其他allowed_actions比较）4)综合判断（为何此时执行该操作）"
}}"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
