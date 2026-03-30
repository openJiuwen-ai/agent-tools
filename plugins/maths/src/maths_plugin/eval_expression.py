"""数学表达式计算工具，基于 NumExpr（本地计算）。"""

from __future__ import annotations

from typing import Any

import numexpr as ne

from openjiuwen.core.foundation.tool import tool


@tool(
    name="eval_expression",
    description=(
        "计算数学表达式的结果（本地使用 NumExpr）。"
        "A tool for evaluating a math expression, calculated locally with NumExpr."
    ),
    input_params={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式，例如：cos(60)、1+(2+3)*4",
            }
        },
        "required": ["expression"],
    },
)
def eval_expression(params: dict[str, Any] | None = None, **kwargs) -> dict:
    """计算数学表达式，并返回结果。"""
    params = params or kwargs
    expression = (params.get("expression") or "").strip()
    if not expression:
        return {"error": "Invalid expression"}

    try:
        result = ne.evaluate(expression)
        result_str = str(result)
        return {
            "report": f'The result of the expression "{expression}" is {result_str}',
            "result": result_str,
            "expression": expression,
        }
    except Exception as e:
        return {
            "error": f"Invalid expression: {expression}, error: {str(e)}",
            "expression": expression,
        }
