from typing import Any, Dict
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.data_fetching import DataFetchingError, get_prices
from app.services.risk_management import analyze_risk_management
from app.services.portfolio_management import generate_portfolio_management_prompt

router = APIRouter(prefix="/decision", tags=["Decision"])


class RiskManagementRequest(BaseModel):
    """风险管理请求模型（简化版，仅cash）。"""
    ticker: str = Field(..., description="股票代码")
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)")
    cash: float = Field(..., description="可用现金（假设无持仓，portfolio_value = cash）")


class PortfolioManagementRequest(BaseModel):
    """投资组合管理请求模型（简化版，无持仓）。"""
    ticker: str = Field(..., description="股票代码")
    cash: float = Field(..., description="可用现金")
    current_price: float = Field(..., description="当前价格（来自风险分析）")
    position_limit: float = Field(..., description="持仓限制（来自风险分析）")
    signals: str = Field(..., description="分析师信号，多个JSON用|分隔，每个JSON包含signal/confidence/reasoning/analyst_name")


# ==================== 风险管理 ====================

@router.post("/risk-management/analyze")
def risk_management_endpoint(request: RiskManagementRequest):
    """
    风险管理分析 - 纯计算服务，无LLM调用（简化版）

    基于波动率计算持仓限制调整。
    - 假设无持仓：portfolio_value = cash
    - 低波动率资产允许更高配置
    - 高波动率资产降低持仓限制
    """
    try:
        # 获取价格数据用于波动率计算
        prices = get_prices(request.ticker, request.start_date, request.end_date)
        price_list = [p.close for p in prices]

        return analyze_risk_management(
            prices=price_list,
            cash=request.cash,
        )
    except DataFetchingError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 投资组合管理 ====================

@router.post("/portfolio-management/prompt")
def portfolio_management_endpoint(request: PortfolioManagementRequest):
    """
    投资组合管理 - 使用LLM生成交易决策（简化版，无持仓）

    基于分析师信号和风险限制做出交易决策。
    - 信号强度：强共识和高信心需要积极行动
    - 风险回报：优先选择低风险高回报机会
    - 允许动作：严格遵守风险限制和可用动作
    """
    # 解析signals参数：用|分隔的多个JSON字符串
    analyst_signals = {}
    try:
        # 按|分割，去除空格
        signal_items = [s.strip() for s in request.signals.split("|") if s.strip()]

        for signal_json in signal_items:
            # 解析每个JSON字符串
            signal_data = json.loads(signal_json)

            # 提取analyst_name作为key
            analyst_name = signal_data.get("analyst_name")
            if analyst_name:
                analyst_signals[analyst_name] = {
                    "signal": signal_data.get("signal"),
                    "confidence": signal_data.get("confidence"),
                    "reasoning": signal_data.get("reasoning", "")
                }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON解析错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"参数解析错误: {str(e)}")

    return generate_portfolio_management_prompt(
        ticker=request.ticker,
        cash=request.cash,
        current_price=request.current_price,
        position_limit=request.position_limit,
        analyst_signals=analyst_signals,
    )


# ==================== 获取所有决策服务列表 ====================

@router.get("/list")
def list_decision_services():
    """获取所有可用的决策服务列表。"""
    return {
        "decision_services": [
            {
                "key": "risk-management",
                "name": "风险管理服务",
                "description": "波动率调整（纯计算，无LLM，简化版假设无持仓）",
                "endpoint": "POST /decision/risk-management/analyze",
                "uses_llm": False
            },
            {
                "key": "portfolio-management",
                "name": "投资组合管理服务",
                "description": "交易决策、仓位管理、风险控制（单只股票，简化版无持仓）",
                "endpoint": "POST /decision/portfolio-management/prompt",
                "uses_llm": True
            }
        ]
    }
