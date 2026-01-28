"""
示例工具：计算器
"""

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """用于数学加减乘除计算"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算失败：{e}"
