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

### 2. 聊天完成接口

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
- `max_tokens`：生成的最大 token 数量
- `n`：生成的响应数量
- `stop`：停止生成的序列
- `top_p`：核采样参数
- `frequency_penalty`：频率惩罚
- `presence_penalty`：存在惩罚
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
