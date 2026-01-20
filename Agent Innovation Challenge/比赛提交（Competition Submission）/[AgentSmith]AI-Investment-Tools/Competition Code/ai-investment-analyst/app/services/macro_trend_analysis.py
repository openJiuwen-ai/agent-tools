import json

from app.models.data import Price


def analyze_macro_trend(prices: list[Price]) -> dict:
    """分析宏观趋势。"""
    if len(prices) < 50:
        return {"score": 0, "details": "宏观趋势分析数据不足"}

    close_prices = [p.close for p in prices]

    # 长期移动平均
    ma_50 = sum(close_prices[-50:]) / 50
    ma_200 = sum(close_prices[-200:]) / 200 if len(close_prices) >= 200 else sum(close_prices) / len(close_prices)

    # 当前价格
    current_price = close_prices[-1]

    score = 0
    details = []

    # 价格相对MA50的位置
    if current_price > ma_50 * 1.1:
        score += 3
        details.append(f"价格强劲高于50日均线 {(current_price/ma_50 - 1):.1%}")
    elif current_price > ma_50:
        score += 2
        details.append("价格高于50日均线")
    elif current_price > ma_50 * 0.9:
        score += 1
        details.append("价格接近50日均线")
    else:
        details.append("价格低于50日均线")

    # MA50相对MA200的位置（金叉/死叉）
    if ma_50 > ma_200 * 1.05:
        score += 3
        details.append("强劲上升趋势（MA50 > MA105）")
    elif ma_50 > ma_200:
        score += 2
        details.append("上升趋势（MA50 > MA200）")
    else:
        details.append("下降趋势（MA50 < MA200）")

    return {"score": score, "max_score": 6, "details": "; ".join(details)}


def analyze_momentum_strength(prices: list[Price]) -> dict:
    """分析动量强度。"""
    if len(prices) < 20:
        return {"score": 0, "details": "动量强度分析数据不足"}

    close_prices = [p.close for p in prices]

    # 计算不同期间的收益率
    returns_1m = (close_prices[-1] - close_prices[-21]) / close_prices[-21] if len(close_prices) >= 21 else 0
    returns_3m = (close_prices[-1] - close_prices[-63]) / close_prices[-63] if len(close_prices) >= 63 else 0
    returns_6m = (close_prices[-1] - close_prices[-126]) / close_prices[-126] if len(close_prices) >= 126 else 0

    score = 0
    details = []

    if returns_1m > 0.10:
        score += 2
        details.append(f"强劲1个月动量 {returns_1m:.1%}")
    elif returns_1m > 0.05:
        score += 1
        details.append(f"良好1个月动量 {returns_1m:.1%}")

    if returns_3m > 0.15:
        score += 2
        details.append(f"强劲3个月动量 {returns_3m:.1%}")
    elif returns_3m > 0.10:
        score += 1
        details.append(f"良好3个月动量 {returns_3m:.1%}")

    if returns_6m > 0.20:
        score += 2
        details.append(f"强劲6个月动量 {returns_6m:.1%}")
    elif returns_6m > 0.10:
        score += 1
        details.append(f"良好6个月动量 {returns_6m:.1%}")

    return {"score": score, "max_score": 6, "details": "; ".join(details) if details else "无动量数据"}


def analyze_trend_strength(prices: list[Price]) -> dict:
    """分析趋势强度。"""
    if len(prices) < 20:
        return {"score": 0, "details": "趋势强度分析数据不足"}

    close_prices = [p.close for p in prices]

    # 简化的ADX计算：检查价格趋势的一致性
    up_days = 0
    down_days = 0

    for i in range(1, min(21, len(close_prices))):
        if close_prices[i] > close_prices[i - 1]:
            up_days += 1
        else:
            down_days += 1

    trend_strength = abs(up_days - down_days) / 20

    score = 0
    details = []

    if trend_strength > 0.6:
        score += 4
        details.append(f"强劲趋势强度 {trend_strength:.1%}")
    elif trend_strength > 0.4:
        score += 3
        details.append(f"良好趋势强度 {trend_strength:.1%}")
    elif trend_strength > 0.2:
        score += 1
        details.append(f"中等趋势强度 {trend_strength:.1%}")
    else:
        details.append("趋势强度弱")

    return {"score": score, "max_score": 4, "details": "; ".join(details) if details else "无趋势强度数据"}


def generate_macro_trend_prompt(
    ticker: str,
    prices: list[Price],
) -> dict:
    """生成宏观趋势投资分析提示词。"""

    macro = analyze_macro_trend(prices)
    momentum = analyze_momentum_strength(prices)
    strength = analyze_trend_strength(prices)

    total_score = macro["score"] + momentum["score"] + strength["score"]
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
        "macro_trend": macro,
        "momentum_strength": momentum,
        "trend_strength": strength,
    }

    system_prompt = """你是宏观趋势投资类分析师。基于以下原则做出投资决策：

1. 宏观趋势：识别长期价格趋势和移动平均交叉
2. 动量强度：评估多时间框架动量
3. 趋势强度：确认趋势的可持续性

决策规则：
- 看涨：强劲上升趋势 + 强劲动量
- 看跌：下降趋势 + 疲弱动量
- 中性：趋势不明确或信号混合

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"macro_trend"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "macro_trend"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
