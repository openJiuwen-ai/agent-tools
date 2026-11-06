# Jiuwen Agent Router API 文档

## 概述

Jiuwen Agent Router 提供了一组 RESTful API 接口，用于处理客户端的聊天请求并进行智能路由。

## 认证

API 支持 API 密钥认证，认证方式有两种：

1. 通过 `X-API-Key` 请求头
2. 通过 `Authorization: Bearer <api_key>` 请求头

认证可以通过配置文件中的 `enable_auth` 参数启用或禁用。

## 接口列表

### 1. 健康检查接口

```text
GET /health
```

**描述**：检查服务是否正常运行

**参数**：无

**响应**：

```json
{
  "status": "healthy",
  "timestamp": "2026-04-07T12:00:00Z"
}
```

### 2. 模型列表接口

```text
GET /v1/models
```

**描述**：返回 Router 当前已发现的 worker 所服务的模型，模型名称会去重并稳定排序。

**响应**：

```json
{
  "object": "list",
  "data": [
    {
      "id": "model-name",
      "object": "model",
      "owned_by": "jiuwen"
    }
  ]
}
```

### 3. 聊天完成接口

```text
POST /v1/chat/completions
```

**描述**：处理聊天请求并返回响应

**请求体**：

```json
{
  "model": "model-name",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "n": 1,
  "stop": null,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "jiuwenext": {
    "agent_hints": {
      "priority": 5,
      "estimated_output_tokens": 256,
      "next_turn_prefill": true
    }
  }
}
```

**参数说明**：

- `model`：要使用的模型名称
- `messages`：聊天消息列表
  - `role`：消息角色（"system", "user", "assistant"）
  - `content`：消息内容
- `temperature`：采样温度，控制输出的随机性（0-2）
- `max_tokens`：旧版生成 token 上限；默认省略时 Router 原样透传，由后端按剩余上下文决定
- `max_completion_tokens`：新版 Chat Completions 生成上限，优先于 `max_tokens`
- `n`：生成的响应数量
- `stop`：停止生成的序列
- `top_p`：核采样参数
- `frequency_penalty`：频率惩罚
- `presence_penalty`：存在惩罚
- `tools`：OpenAI function tools 定义，Router 会原样转发，并在路由分词时纳入工具 schema
- `tool_choice`：支持 `none`、`auto`、`required` 或指定函数对象
- `parallel_tool_calls`：是否允许并行工具调用
- `jiuwenext`：九问扩展参数
  - `agent_hints`：Agent 提示信息
    - `priority`：请求优先级（0-10）
    - `estimated_output_tokens`：预期输出 token 数量
    - `next_turn_prefill`：是否启用下一轮预填充

**响应**：

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  },
  "jiuwenext": {
    "route_info": {
      "worker_id": "worker-1",
      "queue_time": 10,
      "processing_time": 50
    }
  }
}
```

## 错误处理

API 使用以下 HTTP 状态码表示错误：

- `400 Bad Request`：请求参数错误
- `401 Unauthorized`：API 密钥无效
- `500 Internal Server Error`：服务器内部错误
- `503 Service Unavailable`：服务不可用

错误响应格式：

```json
{
  "error": {
    "message": "Error message",
    "type": "error_type"
  }
}
```

## 示例请求

### 使用 curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"model":"model-name","messages":[{"role":"user","content":"Hello"}],"jiuwenext":{"agent_hints":{"priority":5,"estimated_output_tokens":256,"next_turn_prefill":true}}}'
```

### 使用 Python

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key"
}
data = {
    "model": "model-name",
    "messages": [{"role": "user", "content": "Hello"}],
    "jiuwenext": {
        "agent_hints": {
            "priority": 5,
            "estimated_output_tokens": 256,
            "next_turn_prefill": true
        }
    }
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

## JiuwenSwarm / jiuwenswarm-tui 接入

JiuwenSwarm 使用 OpenAI Chat Completions 协议。将模型地址指向 Router 的
`/v1` 根路径即可，例如 `~/.jiuwenswarm/config/config.yaml`：

```yaml
model_name: your-served-model-name
api_base: http://127.0.0.1:7000/v1
api_key: dummy
model_provider: OpenAI
```

Router 支持 JiuwenSwarm Agent 所需的流式 `stream_options.include_usage`、
`tools`、`tool_choice`、`parallel_tool_calls` 和多轮 assistant/tool 消息透传。
模型探测接口是复数形式 `GET /v1/models`，不是 `GET /v1/model`。

默认情况下 Router 不注入 `max_tokens`，因此 JiuwenSwarm 未设置输出上限时，
行为与直接访问 vLLM serve 一致。若部署需要统一硬限制，可在 Router 配置中显式设置：

```yaml
default_max_tokens: 2048
```

工具调用的文本解析由 worker 完成。`jiuwen-infer-worker` 按以下优先级自动识别
parser：显式配置 > 模型工件（`config.json` 的 `model_type` + chat template 内容
签名）> 模型名匹配（仅当本地工件不可读时兜底）。普通 Qwen3（包括微调版）会自动
选择 `hermes` 工具 parser、`qwen3` reasoning parser，并开启自动工具调用，因此
最小 `worker.yaml` 只需要正常配置模型：

```yaml
model: /path/to/Qwen3-4B
served_model_name: qwen3-4b
```

自动识别以 `config.json` 的 `model_type`（如 `qwen3`、`qwen3_coder`）判定模型
家族——这是模型作者声明的结构化字段，微调模型继承基座配置，不受目录名影响；
同时读取 chat template（`tokenizer_config.json` 内嵌或独立的 `chat_template.jinja`）
中的格式标记（`<tool_call>` / `<tool_call_start>` / `<think>`）判定实际输出格式。
模板签名与家族推断冲突时以模板为准；模板被定制得不含任何已知标记时不猜测。
当前映射包括：普通 Qwen3 → `hermes + qwen3`、Qwen3-Coder → `qwen3_xml`、
Qwen2.5/QwQ → `hermes`。Qwen-VL、Qwen3-Next 等变体即使名字像 Qwen3 也不会
猜测 parser。需要覆盖时仍可显式设置 `tool_call_parser`、`reasoning_parser` 和
`enable_auto_tools`；设置 `auto_detect_parsers: false` 可完全关闭自动识别。
YAML 中也接受 vLLM 风格的 `enable_auto_tool_choice` 字段。
