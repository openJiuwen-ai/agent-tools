"""使用 openjiuwen 官方的 MCP 客户端实现 list_tools / call_tool。"""

import asyncio
import concurrent.futures
import json
import logging
from typing import Any

from openjiuwen.core.foundation.tool.mcp.client.mcp_client import McpClient
from openjiuwen.core.foundation.tool.mcp.client.sse_client import SseClient
from openjiuwen.core.foundation.tool.mcp.client.streamable_http_client import (
    StreamableHttpClient,
)

logger = logging.getLogger(__name__)

# 多 server 时工具名前缀分隔符，与参考实现一致
SERVER_TOOL_SEP = "__"


def _run_async(coro):
    """
    在同步上下文中执行协程：无运行中 loop 时用 asyncio.run；
    已有运行中 loop 时在子线程中 asyncio.run，避免 "cannot be called from a running event loop"。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def _parse_servers_config(servers_config: dict) -> list[dict[str, Any]]:
    """解析为 server 配置列表，含 name/url/transport/headers/timeout 等。"""
    if "mcpServers" in servers_config:
        servers_config = servers_config["mcpServers"]
    out = []
    for name, cfg in servers_config.items():
        url = cfg.get("url") or cfg.get("server_path")
        if not url:
            raise ValueError(f"Server {name!r} 缺少 url")
        transport = (cfg.get("transport") or "sse").lower()
        headers = cfg.get("headers") or {}
        auth_query = cfg.get("auth_query_params") or {}
        timeout = cfg.get("timeout")
        if timeout is not None:
            timeout = float(timeout)
        out.append(
            {
                "name": name,
                "url": url,
                "transport": transport,
                "auth_headers": headers,
                "auth_query_params": auth_query,
                "timeout": timeout,
            }
        )
    return out


def _make_client(server_info: dict[str, Any]) -> McpClient:
    """根据 server 配置创建 SseClient 或 StreamableHttpClient。"""
    name = server_info["name"]
    url = server_info["url"]
    transport = server_info.get("transport", "sse")
    auth_headers = server_info.get("auth_headers") or {}
    auth_query = server_info.get("auth_query_params") or {}
    if transport == "streamable_http":
        return StreamableHttpClient(
            server_path=url,
            name=name,
            auth_headers=auth_headers or None,
            auth_query_params=auth_query or None,
        )
    return SseClient(
        server_path=url,
        name=name,
        auth_headers=auth_headers or None,
        auth_query_params=auth_query or None,
    )


def _tool_card_to_prompt_tool(card: Any, display_name: str) -> dict[str, Any]:
    """将 McpToolCard 转为 {name, description, parameters}。"""
    return {
        "name": display_name,
        "description": getattr(card, "description", "") or "",
        "parameters": getattr(card, "input_params", None) or {},
    }


async def _list_tools_async(servers_list: list[dict]) -> list[dict]:
    """聚合多 server 的 list_tools。多 server 时工具名为 server_name__tool_name。"""
    all_tools = []
    need_prefix = len(servers_list) > 1

    for srv in servers_list:
        client = _make_client(srv)
        try:
            connect_kw: dict[str, Any] = {}
            if srv.get("timeout") is not None:
                connect_kw["timeout"] = float(srv["timeout"])
            ok = await client.connect(**connect_kw)
            if not ok:
                logger.warning("Connect failed for server %s", srv["name"])
                continue
            list_kw: dict[str, Any] = {}
            if srv.get("timeout") is not None:
                list_kw["timeout"] = float(srv["timeout"])
            cards = await client.list_tools(**list_kw)
            for c in cards:
                display_name = f"{srv['name']}{SERVER_TOOL_SEP}{c.name}" if need_prefix else c.name
                all_tools.append(_tool_card_to_prompt_tool(c, display_name))
        finally:
            await client.disconnect()

    return all_tools


async def _call_tool_async(
    servers_list: list[dict],
    tool_name: str,
    arguments: dict,
) -> Any:
    """多 server 时 tool_name 须为 server_name__tool_name；单 server 时直接用工具名。"""
    server_name: str | None = None
    actual_tool_name = tool_name
    if SERVER_TOOL_SEP in tool_name:
        parts = tool_name.split(SERVER_TOOL_SEP, 1)
        if len(parts) == 2:
            server_name, actual_tool_name = parts[0], parts[1]
    if not server_name:
        if len(servers_list) == 1:
            server_name = servers_list[0]["name"]
            actual_tool_name = tool_name
        else:
            raise ValueError(f"多 server 时 tool_name 须为 server_name{SERVER_TOOL_SEP}tool_name")

    srv = next((s for s in servers_list if s["name"] == server_name), None)
    if not srv:
        raise ValueError(f"未找到 server: {server_name}")

    client = _make_client(srv)
    try:
        connect_kw: dict[str, Any] = {}
        if srv.get("timeout") is not None:
            connect_kw["timeout"] = float(srv["timeout"])
        ok = await client.connect(**connect_kw)
        if not ok:
            raise RuntimeError(f"连接 MCP 服务失败: {server_name}")
        call_kw: dict[str, Any] = {}
        if srv.get("timeout") is not None:
            call_kw["timeout"] = float(srv["timeout"])
        return await client.call_tool(actual_tool_name, arguments, **call_kw)
    finally:
        await client.disconnect()


def list_tools(servers_config: dict) -> dict[str, Any]:
    """同步封装：获取工具列表。"""
    servers_list = _parse_servers_config(servers_config)
    if not servers_list:
        return {"error": "servers_config 中无有效 MCP 服务", "tools_list": []}

    try:
        tools_list = _run_async(_list_tools_async(servers_list))
        return {
            "tools_list": tools_list,
            "summary": f"MCP 共 {len(tools_list)} 个工具",
        }
    except Exception as e:
        logger.exception("list_tools failed: %s", e)
        return {"error": str(e), "tools_list": []}


def call_tool(
    servers_config: dict,
    tool_name: str,
    arguments: dict,
) -> dict[str, Any]:
    """同步封装：调用工具。openjiuwen 客户端目前只返回文本，故 result 为 text 类型。"""
    servers_list = _parse_servers_config(servers_config)
    if not servers_list:
        return {"error": "servers_config 中无有效 MCP 服务"}

    try:
        result = _run_async(_call_tool_async(servers_list, tool_name, arguments))
        if result is None:
            result = ""
        return {"result": result, "type": "text"}
    except Exception as e:
        logger.exception("call_tool failed: %s", e)
        return {"error": str(e)}
