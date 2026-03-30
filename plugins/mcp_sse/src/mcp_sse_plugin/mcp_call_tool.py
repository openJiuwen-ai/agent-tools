"""调用 MCP 服务端工具（基于 openjiuwen 官方 MCP 客户端）。"""

import json
import logging
from typing import Any

from openjiuwen.core.foundation.tool import tool

from ._openjiuwen_mcp import call_tool as _call_tool
from .mcp_list_tools import _get_servers_config

logger = logging.getLogger(__name__)


@tool(
    name="mcp_call_tool",
    description=(
        "用于执行「mcp_list_tools 返回的远程 MCP 工具」。配置从环境变量 MCP_SERVERS_CONFIG 读取。\n"
        "请勿直接调用 tools_list 里的工具名（它们不是本地工具）。\n"
        "用法：tool_name 取自 mcp_list_tools 返回的 name（多 server 时为 server_name__tool_name），"
        "arguments 为 JSON 字符串，需符合该工具的 parameters。"
    ),
    input_params={
        "type": "object",
        "properties": {
            "servers_config": {
                "type": "string",
                "description": "可选，MCP 服务配置 JSON；不传则使用环境变量 MCP_SERVERS_CONFIG。",
            },
            "tool_name": {
                "type": "string",
                "description": "要执行的 MCP 工具名称（来自 mcp_list_tools 返回的 name）。",
            },
            "arguments": {
                "type": "string",
                "description": "工具参数 JSON 字符串，例如: {\"query\": \"hello\"}",
            },
        },
        "required": ["tool_name", "arguments"],
    },
)
def mcp_call_tool(
    params: dict[str, Any] | None = None, **kwargs
) -> dict[str, Any]:
    """调用指定 MCP 工具并返回结果。配置来自环境变量 MCP_SERVERS_CONFIG。"""
    params = params or kwargs
    servers_config, err = _get_servers_config(params)
    if err:
        return {"error": err}

    tool_name = (params.get("tool_name") or "").strip()
    if not tool_name:
        return {"error": "请填写 tool_name"}

    arguments_json = params.get("arguments", "")
    if not arguments_json:
        return {"error": "请填写 arguments"}
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as e:
        return {"error": f"arguments 必须是合法 JSON: {e}"}

    return _call_tool(servers_config, tool_name, arguments)
