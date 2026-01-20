from typing import Any
import numpy as np
from numpy.typing import NDArray


def calculate_returns(prices: list[float]) -> list[float]:
    """计算价格收益率。"""
    if len(prices) < 2:
        return []
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
    return returns


def calculate_volatility_metrics(prices: list[float], lookback_days: int = 60) -> dict:
    """从价格数据计算综合波动率指标。"""
    if len(prices) < 2:
        return {
            "daily_volatility": 0.05,
            "annualized_volatility": 0.05 * np.sqrt(252),
            "volatility_percentile": 100,
            "data_points": len(prices)
        }

    # 计算日收益率
    returns = calculate_returns(prices)

    if len(returns) < 2:
        return {
            "daily_volatility": 0.05,
            "annualized_volatility": 0.05 * np.sqrt(252),
            "volatility_percentile": 100,
            "data_points": len(returns)
        }

    # 使用最近的lookback_days计算波动率
    recent_returns = returns[-min(lookback_days, len(returns)):]

    # 计算波动率指标
    daily_vol = float(np.std(recent_returns))
    annualized_vol = daily_vol * np.sqrt(252)  # 年化，假设252个交易日

    # 计算当前波动率相对于历史波动率的百分位
    if len(returns) >= 30:  # 需要足够的历史数据来计算百分位
        # 计算完整历史的30天滚动波动率
        rolling_vols = []
        for i in range(29, len(returns)):
            window = returns[i-29:i+1]
            rolling_vols.append(np.std(window))
        if rolling_vols:
            current_vol_percentile = sum(1 for v in rolling_vols if v <= daily_vol) / len(rolling_vols) * 100
        else:
            current_vol_percentile = 50  # 默认中位数
    else:
        current_vol_percentile = 50  # 数据不足时默认中位数

    return {
        "daily_volatility": daily_vol if not np.isnan(daily_vol) else 0.025,
        "annualized_volatility": annualized_vol if not np.isnan(annualized_vol) else 0.25,
        "volatility_percentile": current_vol_percentile if not np.isnan(current_vol_percentile) else 50.0,
        "data_points": len(recent_returns)
    }


def calculate_volatility_adjusted_limit(annualized_volatility: float, base_limit: float = 0.20) -> float:
    """
    根据波动率计算持仓限制百分比。

    逻辑：
    - 低波动率 (<15%): 最多25%配置
    - 中等波动率 (15-30%): 15-20%配置
    - 高波动率 (>30%): 10-15%配置
    - 极高波动率 (>50%): 最多10%配置
    """
    if annualized_volatility < 0.15:  # 低波动率
        vol_multiplier = 1.25  # 最多25%
    elif annualized_volatility < 0.30:  # 中等波动率
        vol_multiplier = 1.0 - (annualized_volatility - 0.15) * 0.5  # 20% -> 12.5%
    elif annualized_volatility < 0.50:  # 高波动率
        vol_multiplier = 0.75 - (annualized_volatility - 0.30) * 0.5  # 15% -> 5%
    else:  # 极高波动率 (>50%)
        vol_multiplier = 0.50  # 最多10%

    # 确保合理的限制范围
    vol_multiplier = max(0.25, min(1.25, vol_multiplier))  # 5%到25%范围

    return base_limit * vol_multiplier


def analyze_risk_management(
    prices: list[float],
    cash: float,
    base_position_limit: float = 0.20,
) -> dict:
    """
    执行风险管理分析（纯计算，无LLM调用）。

    简化版假设：
    - 无持仓：current_position_weight = 0
    - portfolio_value = cash

    参数：
    - prices: 价格历史列表
    - cash: 可用现金（假设无持仓，portfolio_value = cash）
    - base_position_limit: 基础持仓限制（默认20%）

    返回：
    - 风险分析结果，包含波动率调整和持仓限制
    """
    if not prices:
        return {
            "error": "无价格数据"
        }

    current_price = prices[-1]

    # 简化版假设：无持仓
    current_position_weight = 0.0
    portfolio_value = cash  # 假设无持仓时，portfolio_value = cash

    # 计算波动率指标
    volatility_metrics = calculate_volatility_metrics(prices)

    # 波动率调整后的持仓限制百分比
    vol_adjusted_limit_pct = calculate_volatility_adjusted_limit(
        volatility_metrics["annualized_volatility"],
        base_position_limit
    )

    # 计算当前持仓价值（简化版为0）
    current_position_value = 0.0

    # 转换为美元持仓限制
    position_limit = portfolio_value * vol_adjusted_limit_pct

    # 计算剩余限制（简化版等于position_limit，因为无持仓）
    remaining_position_limit = position_limit

    # 确保不超过可用现金
    max_position_size = min(remaining_position_limit, cash)

    return {
        "current_price": float(current_price),
        "remaining_position_limit": float(max_position_size),
        "volatility_metrics": {
            "daily_volatility": float(volatility_metrics["daily_volatility"]),
            "annualized_volatility": float(volatility_metrics["annualized_volatility"]),
            "volatility_percentile": float(volatility_metrics["volatility_percentile"]),
            "data_points": int(volatility_metrics["data_points"])
        },
        "adjustments": {
            "portfolio_value": float(portfolio_value),
            "current_position_value": float(current_position_value),
            "base_position_limit_pct": float(vol_adjusted_limit_pct),
            "position_limit": float(position_limit),
            "remaining_limit": float(remaining_position_limit),
            "available_cash": float(cash),
        }
    }
