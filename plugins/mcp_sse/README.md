# MCP SSE Plugin（openJiuwen）

如果你想尝试 MCP，但是不知道如何部署 SSE 传输的 MCP Server，可以尝试连接托管的 MCP 服务器。本插件通过 **HTTP with SSE** 或 **Streamable HTTP** 使用 [MCP（Model Context Protocol）](https://modelcontextprotocol.io/) 发现和调用远程工具。本插件直接复用 openjiuwen 官方的 MCP 客户端（`SseClient` / `StreamableHttpClient`），无需自维护协议实现。

## 功能

| 工具 | 说明 |
|------|------|
| **mcp_list_tools** | 获取已配置 MCP 服务端的工具列表；多 server 时工具名为 `server_name__tool_name` |
| **mcp_call_tool** | 根据工具名和参数调用 MCP 工具，返回文本结果 |

## 配置

### 1. 环境变量

MCP 服务配置从环境变量 **MCP_SERVERS_CONFIG** 读取，无需在调用工具时传入。在运行 openjiuwen 的环境中设置：

```bash
export MCP_SERVERS_CONFIG='{"mcpServers":{"my_server":{"url":"http://127.0.0.1:8000/sse","headers":{},"transport":"sse","timeout":50}}}'
```

或写入 `.env` 文件（值为单行 JSON 字符串）。

### 2. 配置项说明

配置为 JSON，支持两种写法：

**写法一：带 `mcpServers` 键**

```json
{
  "mcpServers": {
    "server_name": {
      "url": "http://127.0.0.1:8000/sse",
      "headers": {},
      "transport": "sse",
      "timeout": 50
    }
  }
}
```

**写法二：直接为 server 名到配置的映射**（插件会自动识别 `mcpServers` 键）

```json
{
  "server_name": {
    "url": "http://127.0.0.1:8000/sse",
    "transport": "sse",
    "headers": {},
    "timeout": 50
  }
}
```

| 字段 | 说明 |
|------|------|
| **url** | MCP 服务地址（必填）。SSE 为 GET 连接 endpoint，Streamable HTTP 为 POST 地址 |
| **transport** | 传输方式：`sse`（默认）或 `streamable_http` |
| **headers** | 可选，HTTP 头（作为 auth_headers 传给 openjiuwen 客户端） |
| **timeout** | 可选，连接及 list/call 超时（秒），不配置时使用客户端默认（如 60 秒） |

## 安装

在插件目录下安装：

```bash
cd plugins/mcp_sse
pip install -e .
```

或在项目的 `pyproject.toml` 中作为依赖引用。

## 使用说明

1. **配置**：在运行环境中设置 `MCP_SERVERS_CONFIG`（见上文）。
2. **列工具**：先调用 **mcp_list_tools**（无需传参），获取工具列表与参数结构。
3. **调工具**：再调用 **mcp_call_tool**，传入：
   - **tool_name**：来自 list 返回的 `name`；多 server 时为 `server_name__tool_name`
   - **arguments**：JSON 字符串，如 `{"query":"hello"}`

多 server 时，工具名会带前缀 `server_name__`，call 时须使用该完整名称。

## 返回示例

### mcp_list_tools

```json
{
  "tools_list": [
    {
      "name": "my_server__get_weather",
      "description": "Get current weather for a location.",
      "parameters": { "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] }
    }
  ],
  "summary": "MCP 共 1 个工具"
}
```

### mcp_call_tool

```json
{
  "result": "北京 晴 25℃",
  "type": "text"
}
```

错误时返回 `{"error": "错误信息"}`。

## 依赖

- Python >= 3.11
- openjiuwen >= 0.1.5, < 0.2.0（内含 MCP 客户端及 mcp/fastmcp 等依赖）

## 样例

### 高德地图

#### 1. 获取API Key
在 [高德开放平台控制台](https://console.amap.com/dev/index) 注册并获取 API Key。[操作指导](https://lbs.amap.com/api/mcp-server/create-project-and-key)

#### 2. 配置环境变量
```json
{
  "mcpServers": {
    "amap-maps-streamableHTTP": {
      "url": "https://mcp.amap.com/mcp?key=your_api_key",
      "transport": "streamable_http",
      "headers": {},
      "timeout": 10
    }
  }
}

## 参考

- [MCP 协议说明](https://modelcontextprotocol.io/)
- [openjiuwen 文档](https://openjiuwen.com/docs-page)
