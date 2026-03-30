"""获取 MCP 服务端工具列表（基于 openjiuwen 官方 MCP 客户端）。"""

import json
import logging
import os
from typing import Any

from openjiuwen.core.foundation.tool import tool

from ._openjiuwen_mcp import list_tools as _list_tools

logger = logging.getLogger(__name__)

# 环境变量名：MCP 服务配置 JSON 字符串，不传参时从此读取
MCP_SERVERS_CONFIG_ENV = "MCP_SERVERS_CONFIG"


def _get_servers_config(params: dict[str, Any]) -> tuple[dict | None, str | None]:
    """从 params 或环境变量 MCP_SERVERS_CONFIG 获取配置，成功返回 (config, None)，失败返回 (None, error_msg)。"""
    raw = (params.get("servers_config") or "").strip()
    if not raw:
        raw = (os.getenv(MCP_SERVERS_CONFIG_ENV) or "").strip()
    if not raw:
        return None, f"未配置 MCP 服务：请在环境变量中设置 {MCP_SERVERS_CONFIG_ENV}（JSON 字符串）"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, f"servers_config 必须是合法 JSON: {e}"


@tool(
    name="mcp_list_tools",
    description=(
        "获取 MCP 服务端的工具列表（远程工具清单）。配置从环境变量 MCP_SERVERS_CONFIG 读取。\n"
        "重要：tools_list 里的 name 是「远程 MCP 工具名」，不是本地可直接调用的工具。\n"
        "要执行其中某个工具，必须调用本插件的 mcp_call_tool，并将该 name 作为 tool_name，"
        "同时按 parameters 构造 arguments（JSON 字符串）。"
    ),
    input_params={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def mcp_list_tools(params: dict[str, Any] | None = None, **kwargs) -> dict[str, Any]:
    """连接 MCP 服务并返回工具列表。配置来自环境变量 MCP_SERVERS_CONFIG。"""
    params = params or kwargs
    servers_config, err = _get_servers_config(params)
    if err:
        return {"error": err}
    resp = _list_tools(servers_config)
    if "error" in resp:
        return resp
    note = (
        "注意：tools_list 中的 name 为远程 MCP 工具名，不能直接当作本地工具调用；"
        "请通过 mcp_call_tool(tool_name=..., arguments=...) 间接调用。"
    )
    resp["note"] = note
    if isinstance(resp.get("summary"), str):
        resp["summary"] = f"{resp['summary']}。{note}"
    return resp
