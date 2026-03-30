"""工具参数解析（逗号列表、JSON 字符串）。"""

import json
from typing import Any


def get_array_params(tool_parameters: dict[str, Any], key: str) -> list[str] | None:
    param = tool_parameters.get(key)
    if not param:
        return None
    if isinstance(param, list):
        return [str(x).strip() for x in param if str(x).strip()]
    s = str(param).strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def get_json_params(tool_parameters: dict[str, Any], key: str) -> dict[str, Any] | None:
    param = tool_parameters.get(key)
    if param is None or param == "":
        return None
    if isinstance(param, dict):
        return param
    if not isinstance(param, str):
        return None
    try:
        text = param.replace("'", '"')
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Invalid {key} format: {e}") from e
