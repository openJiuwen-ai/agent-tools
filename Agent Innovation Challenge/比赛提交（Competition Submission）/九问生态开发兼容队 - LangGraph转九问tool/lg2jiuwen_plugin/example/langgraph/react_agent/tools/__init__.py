"""
工具模块
"""

from .calculator import calculator
from .weather import weather

# 工具映射
tool_map = {
    "Calculator": calculator,
    "Weather": weather,
}

__all__ = ["calculator", "weather", "tool_map"]
