import json
import math
from typing import Any

from app.models.data import Price


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换为浮点数，处理NaN情况。"""
    try:
        import pandas as pd
        import numpy as np
        if pd.isna(value) or np.isnan(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError, ImportError):
        return float(value) if value is not None else default


def calculate_ema(prices: list[float], window: int) -> list[float]:
    """计算指数移动平均线。"""
    if len(prices) < window:
        return [None] * len(prices)

    ema = [None] * len(prices)
    multiplier = 2 / (window + 1)

    # 计算第一个EMA值（使用SMA）
    sma = sum(prices[:window]) / window
    ema[window - 1] = sma

    # 计算后续EMA值
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

    return ema


def calculate_rsi(prices: list[float], period: int = 14) -> list[float]:
    """计算相对强弱指标。"""
    if len(prices) < period + 1:
        return [None] * len(prices)

    rsi_values = [None] * len(prices)
    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    # 计算初始平均增益和损失
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        rsi_values[period] = 100
    else:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100 - (100 / (1 + rs))

    # 计算后续RSI值
    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi_values[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))

    return rsi_values


def calculate_bollinger_bands(prices: list[float], window: int = 20) -> tuple[list[float], list[float]]:
    """计算布林带。"""
    if len(prices) < window:
        return [None] * len(prices), [None] * len(prices)

    upper_band = [None] * len(prices)
    lower_band = [None] * len(prices)

    for i in range(window - 1, len(prices)):
        slice_prices = prices[i - window + 1:i + 1]
        sma = sum(slice_prices) / window
        variance = sum((p - sma) ** 2 for p in slice_prices) / window
        std = math.sqrt(variance)

        upper_band[i] = sma + (std * 2)
        lower_band[i] = sma - (std * 2)

    return upper_band, lower_band


def calculate_adx(high_prices: list[float], low_prices: list[float], close_prices: list[float], period: int = 14) -> list[float]:
    """计算平均趋向指标。"""
    if len(close_prices) < period + 1:
        return [None] * len(close_prices)

    adx_values = [None] * len(close_prices)

    # 计算True Range
    tr_list = []
    for i in range(1, len(close_prices)):
        high_low = high_prices[i] - low_prices[i]
        high_close = abs(high_prices[i] - close_prices[i - 1])
        low_close = abs(low_prices[i] - close_prices[i - 1])
        tr_list.append(max(high_low, high_close, low_close))

    # 计算方向移动
    plus_dm_list = []
    minus_dm_list = []
    for i in range(1, len(close_prices)):
        up_move = high_prices[i] - high_prices[i - 1]
        down_move = low_prices[i - 1] - low_prices[i]

        if up_move > down_move and up_move > 0:
            plus_dm_list.append(up_move)
        else:
            plus_dm_list.append(0)

        if down_move > up_move and down_move > 0:
            minus_dm_list.append(down_move)
        else:
            minus_dm_list.append(0)

    # 计算ADX
    atr = sum(tr_list[:period]) / period
    plus_di = 100 * (sum(plus_dm_list[:period]) / period) / atr if atr > 0 else 0
    minus_di = 100 * (sum(minus_dm_list[:period]) / period) / atr if atr > 0 else 0

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    adx_values[period] = dx

    for i in range(period + 1, len(close_prices)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_di_new = (plus_di * (period - 1) + plus_dm_list[i]) / period
        minus_di_new = (minus_di * (period - 1) + minus_dm_list[i]) / period

        dx = 100 * abs(plus_di_new - minus_di_new) / (plus_di_new + minus_di_new) if (plus_di_new + minus_di_new) > 0 else 0
        adx_values[i] = (adx_values[i - 1] * (period - 1) + dx) / period if adx_values[i - 1] else dx
        plus_di = plus_di_new
        minus_di = minus_di_new

    return adx_values


def calculate_trend_signals(prices: list[Price]) -> dict:
    """趋势跟踪策略。"""
    close_prices = [p.close for p in prices]
    high_prices = [p.high for p in prices]
    low_prices = [p.low for p in prices]

    if len(close_prices) < 55:
        return {"signal": "neutral", "confidence": 0.5, "metrics": {"adx": 0, "trend_strength": 0}}

    # 计算EMA
    ema_8 = calculate_ema(close_prices, 8)
    ema_21 = calculate_ema(close_prices, 21)
    ema_55 = calculate_ema(close_prices, 55)

    # 计算ADX
    adx = calculate_adx(high_prices, low_prices, close_prices, 14)

    # 确定趋势方向
    short_trend = ema_8[-1] > ema_21[-1] if ema_8[-1] and ema_21[-1] else False
    medium_trend = ema_21[-1] > ema_55[-1] if ema_21[-1] and ema_55[-1] else False

    trend_strength = safe_float(adx[-1]) / 100.0 if adx[-1] else 0.5

    if short_trend and medium_trend:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend and not medium_trend:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": safe_float(adx[-1]),
            "trend_strength": safe_float(trend_strength),
        },
    }


def calculate_mean_reversion_signals(prices: list[Price]) -> dict:
    """均值回归策略。"""
    close_prices = [p.close for p in prices]

    if len(close_prices) < 50:
        return {"signal": "neutral", "confidence": 0.5, "metrics": {"z_score": 0, "price_vs_bb": 0, "rsi_14": 50}}

    # 计算Z-score
    window = 50
    ma = sum(close_prices[-window:]) / window
    variance = sum((p - ma) ** 2 for p in close_prices[-window:]) / window
    std = math.sqrt(variance)
    z_score = (close_prices[-1] - ma) / std if std > 0 else 0

    # 计算布林带
    bb_upper, bb_lower = calculate_bollinger_bands(close_prices, 20)
    price_vs_bb = (close_prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] and bb_lower[-1] and (bb_upper[-1] - bb_lower[-1]) > 0 else 0.5

    # 计算RSI
    rsi_14 = calculate_rsi(close_prices, 14)
    rsi_28 = calculate_rsi(close_prices, 28)

    # 生成信号
    if z_score < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score) / 4, 1.0)
    elif z_score > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": safe_float(z_score),
            "price_vs_bb": safe_float(price_vs_bb),
            "rsi_14": safe_float(rsi_14[-1]) if rsi_14[-1] else 50,
            "rsi_28": safe_float(rsi_28[-1]) if rsi_28[-1] else 50,
        },
    }


def calculate_momentum_signals(prices: list[Price]) -> dict:
    """动量策略。"""
    close_prices = [p.close for p in prices]
    volumes = [p.volume for p in prices]

    if len(close_prices) < 126:
        return {"signal": "neutral", "confidence": 0.5, "metrics": {"momentum_1m": 0, "momentum_3m": 0, "momentum_6m": 0, "volume_momentum": 1.0}}

    # 价格动量
    returns = []
    for i in range(1, len(close_prices)):
        if close_prices[i - 1] != 0:
            returns.append((close_prices[i] - close_prices[i - 1]) / close_prices[i - 1])
        else:
            returns.append(0)

    # 1个月、3个月、6个月动量
    mom_1m = sum(returns[-21:]) if len(returns) >= 21 else 0
    mom_3m = sum(returns[-63:]) if len(returns) >= 63 else 0
    mom_6m = sum(returns[-126:]) if len(returns) >= 126 else 0

    # 成交量动量
    volume_ma = sum(volumes[-21:]) / 21 if len(volumes) >= 21 else volumes[-1]
    volume_momentum = volumes[-1] / volume_ma if volume_ma > 0 else 1.0

    # 计算动量得分
    momentum_score = 0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m

    volume_confirmation = volume_momentum > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": safe_float(mom_1m),
            "momentum_3m": safe_float(mom_3m),
            "momentum_6m": safe_float(mom_6m),
            "volume_momentum": safe_float(volume_momentum),
        },
    }


def calculate_volatility_signals(prices: list[Price]) -> dict:
    """波动率策略。"""
    close_prices = [p.close for p in prices]

    if len(close_prices) < 84:
        return {"signal": "neutral", "confidence": 0.5, "metrics": {"historical_volatility": 0.2, "volatility_regime": 1.0, "volatility_z_score": 0}}

    # 计算收益率
    returns = []
    for i in range(1, len(close_prices)):
        if close_prices[i - 1] != 0:
            returns.append((close_prices[i] - close_prices[i - 1]) / close_prices[i - 1])
        else:
            returns.append(0)

    # 历史波动率
    hist_vol = returns[-21:] if len(returns) >= 21 else returns
    hist_vol_value = sum(hist_vol) / len(hist_vol) if hist_vol else 0

    # 波动率制度检测
    vol_ma = sum(returns[-63:]) / 63 if len(returns) >= 63 else hist_vol_value
    vol_regime = hist_vol_value / vol_ma if vol_ma > 0 else 1.0

    # 生成信号
    if vol_regime < 0.8:
        signal = "bullish"
        confidence = min(abs(vol_regime - 1) / 3, 1.0)
    elif vol_regime > 1.2:
        signal = "bearish"
        confidence = min(abs(vol_regime - 1) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": safe_float(hist_vol_value),
            "volatility_regime": safe_float(vol_regime),
            "volatility_z_score": safe_float(vol_regime - 1),
        },
    }


def weighted_signal_combination(signals: dict, weights: dict) -> dict:
    """加权信号组合。"""
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {"signal": signal, "confidence": abs(final_score)}


def generate_technical_prompt(
    ticker: str,
    prices: list[Price],
) -> dict:
    """生成技术分析提示词。"""

    # 执行所有技术分析
    trend_signals = calculate_trend_signals(prices)
    mean_reversion_signals = calculate_mean_reversion_signals(prices)
    momentum_signals = calculate_momentum_signals(prices)
    volatility_signals = calculate_volatility_signals(prices)

    # 加权组合
    strategy_weights = {
        "trend": 0.35,
        "mean_reversion": 0.25,
        "momentum": 0.25,
        "volatility": 0.15,
    }

    combined_signal = weighted_signal_combination(
        {
            "trend": trend_signals,
            "mean_reversion": mean_reversion_signals,
            "momentum": momentum_signals,
            "volatility": volatility_signals,
        },
        strategy_weights,
    )

    # 构建事实
    facts = {
        "signal": combined_signal["signal"],
        "confidence": int(combined_signal["confidence"] * 100),
        "analysis": {
            "trend_following": trend_signals,
            "mean_reversion": mean_reversion_signals,
            "momentum": momentum_signals,
            "volatility": volatility_signals,
        },
    }

    # 构建提示词（中文）
    system_prompt = """你是技术分析类分析师。基于以下原则做出投资决策：

1. 趋势跟踪：使用EMA和ADX识别趋势方向和强度
2. 均值回归：使用布林带和RSI识别超买超卖
3. 动量分析：识别价格和成交量的动量变化
4. 波动率分析：评估市场波动率制度

决策规则：
- 看涨：综合信号得分 > 0.2
- 看跌：综合信号得分 < -0.2
- 中性：综合信号得分在 -0.2 到 0.2 之间

推理理由控制在120字以内。不要编造数据。仅返回JSON字符串。"""

    # JSON示例（不能在f-string中使用反斜杠）
    json_example = '{"signal":"bullish","confidence":75,"reasoning":"理由","analyst_name":"technical"}'

    human_prompt = f"""股票代码: {ticker}
分析事实:
{json.dumps(facts, separators=(",", ":"), ensure_ascii=False)}

请严格返回以下JSON字符串格式（字符串形式，不是JSON对象）:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "简短理由",
  "analyst_name": "technical"
}}
注意：必须返回JSON字符串，例如"{json_example}"，而不是JSON对象。"""

    prompt = f"""<system>{system_prompt}</system>

<human>{human_prompt}</human>"""

    return {
        "prompt": prompt,
        "facts": facts,
    }
