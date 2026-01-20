import json
from datetime import datetime, timedelta
from typing import Literal

from app.models.data import CompanyNews


def analyze_news_sentiment_trend(news: list[CompanyNews]) -> dict:
    """分析新闻情绪趋势。"""
    if not news or len(news) < 3:
        return {"score": 0, "details": "新闻情绪分析数据不足"}

    score = 0
    details = []

    # 按时间排序（最新的在前）
    sorted_news = sorted(news, key=lambda x: x.date, reverse=True)

    # 分析最近新闻
    recent_news = sorted_news[:10]
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for item in recent_news:
        sentiment = item.sentiment.lower() if item.sentiment else "neutral"
        if "positive" in sentiment or "bullish" in sentiment:
            sentiment_counts["positive"] += 1
        elif "negative" in sentiment or "bearish" in sentiment:
            sentiment_counts["negative"] += 1
        else:
            sentiment_counts["neutral"] += 1

    total = len(recent_news)
    positive_ratio = sentiment_counts["positive"] / total if total > 0 else 0
    negative_ratio = sentiment_counts["negative"] / total if total > 0 else 0

    if positive_ratio >= 0.6:
        score += 4
        details.append(f"积极新闻比例高 {positive_ratio:.1%}")
    elif positive_ratio >= 0.4:
        score += 2
        details.append(f"积极新闻比例中等 {positive_ratio:.1%}")

    if negative_ratio >= 0.5:
        score -= 2
        details.append(f"消极新闻比例高 {negative_ratio:.1%}")

    return {"score": max(score, 0), "max_score": 4, "details": "; ".join(details) if details else "无新闻情绪数据"}


def analyze_news_volume(news: list[CompanyNews]) -> dict:
    """分析新闻量。"""
    if not news:
        return {"score": 0, "details": "新闻量分析数据不足"}

    now = datetime.now()
    recent_7d = [n for n in news if (now - n.date).days <= 7]
    recent_30d = [n for n in news if (now - n.date).days <= 30]

    score = 0
    details = []

    if len(recent_7d) >= 10:
        score += 2
        details.append(f"高新闻量：7天内{len(recent_7d)}条")
    elif len(recent_7d) >= 5:
        score += 1
        details.append(f"中等新闻量：7天内{len(recent_7d)}条")

    if len(recent_30d) >= 30:
        score += 1
        details.append(f"持续关注度：30天内{len(recent_30d)}条")

    return {"score": score, "max_score": 3, "details": "; ".join(details) if details else "无新闻量数据"}


def analyze_sentiment_consistency(news: list[CompanyNews]) -> dict:
    """分析情绪一致性。"""
    if len(news) < 5:
        return {"score": 0, "details": "情绪一致性分析数据不足"}

    sorted_news = sorted(news, key=lambda x: x.date, reverse=True)
    recent = sorted_news[:15]

    sentiments = []
    for item in recent:
        if item.sentiment:
            s = item.sentiment.lower()
            if "positive" in s or "bullish" in s:
                sentiments.append(1)
            elif "negative" in s or "bearish" in s:
                sentiments.append(-1)
            else:
                sentiments.append(0)

    if not sentiments:
        return {"score": 0, "details": "无情绪数据"}

    # 计算标准差来评估一致性
    avg = sum(sentiments) / len(sentiments)
    variance = sum((s - avg) ** 2 for s in sentiments) / len(sentiments)

    score = 0
    details = []

    if variance < 0.5:
        score += 3
        details.append("情绪高度一致")
    elif variance < 1.0:
        score += 2
        details.append("情绪较为一致")
    else:
        details.append("情绪波动较大")

    return {"score": score, "max_score": 3, "details": "; ".join(details) if details else "无一致性数据"}


def generate_news_sentiment_prompt(
    ticker: str,
    news: list[CompanyNews],
) -> dict:
    """生成新闻情绪分析提示词。"""

    sentiment_trend = analyze_news_sentiment_trend(news)
    volume = analyze_news_volume(news)
    consistency = analyze_sentiment_consistency(news)

    total_score = sentiment_trend["score"] + volume["score"] + consistency["score"]
    max_possible_score = 10

    if total_score >= 0.7 * max_possible_score:
        signal = "bullish"
    elif total_score <= 0.2 * max_possible_score:
        signal = "bearish"
    else:
        signal = "neutral"

    facts = {
        "signal": signal,
        "score": total_score,
        "max_score": max_possible_score,
        "sentiment_trend": sentiment_trend,
        "volume": volume,
        "consistency": consistency,
    }

    system_prompt = """你是新闻情绪分析类分析师。基于以下原则做出投资决策：

1. 新闻情绪趋势：分析正面/负面新闻比例
2. 新闻量：评估市场关注度和活跃度
3. 情绪一致性：检查情绪方向的稳定性

决策规则：
- 看涨：积极情绪占主导（60%+）
- 看跌：消极情绪占主导（50%+）
- 中性：情绪混合或数据不足

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"news_sentiment"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "news_sentiment"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }


def generate_market_sentiment_prompt(
    ticker: str,
    insider_trades: list,
    prices: list,
) -> dict:
    """生成市场情绪分析提示词。"""

    # 简化的市场情绪分析
    score = 0
    details = []

    # 内部交易分析
    if insider_trades:
        recent_trades = insider_trades[:10]
        buy_ratio = sum(1 for t in recent_trades if getattr(t, 'transaction_type', '').lower() in ['buy', 'purchase']) / len(recent_trades)

        if buy_ratio >= 0.7:
            score += 3
            details.append(f"内部人积极买入 {buy_ratio:.1%}")
        elif buy_ratio >= 0.5:
            score += 1
            details.append(f"内部人中等买入 {buy_ratio:.1%}")
        else:
            details.append(f"内部人卖出较多 {1-buy_ratio:.1%}")

    # 价格动量（简化版）
    if prices and len(prices) >= 20:
        recent_prices = [p.close for p in prices[-20:]]
        first_price = recent_prices[0]
        last_price = recent_prices[-1]
        momentum = (last_price - first_price) / first_price if first_price > 0 else 0

        if momentum > 0.1:
            score += 3
            details.append(f"强劲价格动量 {momentum:.1%}")
        elif momentum > 0.05:
            score += 2
            details.append(f"良好价格动量 {momentum:.1%}")
        elif momentum > 0:
            score += 1
            details.append(f"正向价格动量 {momentum:.1%}")
        else:
            details.append(f"负向价格动量 {momentum:.1%}")

    max_score = 6
    if score >= 0.7 * max_score:
        signal = "bullish"
    elif score <= 0.3 * max_score:
        signal = "bearish"
    else:
        signal = "neutral"

    facts = {
        "signal": signal,
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details) if details else "数据不足",
    }

    system_prompt = """你是市场情绪分析类分析师。基于以下原则做出投资决策：

1. 内部交易：关注内部人买卖行为
2. 价格动量：评估近期价格趋势

决策规则：
- 看涨：内部人买入积极 + 价格动量强劲
- 看跌：内部人卖出较多 + 价格动量疲弱
- 中性：信号混合

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"market_sentiment"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "market_sentiment"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
