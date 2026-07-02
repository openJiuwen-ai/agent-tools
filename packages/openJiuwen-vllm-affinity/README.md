# 简介

openJiuwen-vllm-affinity 提供了和 vLLM 亲和的增强能力，这些能力通过插件形式集成到 vLLM 中。openJiuwen 可以利用这些能力增强 vLLM 的功能和性能。

# 特性1：Prefix 双区缓存主动管理

## 背景

多轮对话中，上下文引擎会对部分历史对话进行总结压缩，或者对冗余的工具调用返回结果进行卸载，减少历史对话数据量，缩短上下文长度。
上下文的变化会导致部分 KV cache 失效。当前 vLLM 使用 LRU 缓存更新策略，被释放的 cache 处于队列尾部，无法快速清理空间，影响整体缓存命中率和 TTFT。

## 方案

实现双区缓存和主动缓存管理。

空闲块队列在逻辑上分为 **release 区**（主动释放、优先复用/淘汰）和 **normal 区**（普通归还、较低优先级）。从队首分配时，release 区的块会先于 normal 区的块被复用，从而使 `release_kv_cache` 标记的块更快腾出空间。

> 实现上队列头部还存在仅在初始化时出现的 **free 段**（从未分配过的块），分配优先级为 free > release > normal。运行期不再产生新的 free 块，日常理解与排障主要关注 **release 与 normal 的相对优先级** 即可。

主动释放（`release_kv_cache`）时，若明确后续不会再使用某缓存块，则将其移入 release 区尾部以加速老化；若仍可能被复用，则随请求正常归还至 normal 区尾部，避免过早失效。

对于 prefix hash 未覆盖的 **partial tail 块**（例如满块 206–212 之后的尾部块 213），插件会通过 session 绑定与孤儿块收集一并纳入释放范围。

## 接口

### chat 接口

chat 修改如下：

增加 `cache_sharing`：为 `true` 时启用缓存共享，`cache_salt` 作为客户端标识，不同 `cache_salt` 的客户端可以共享缓存块；为 `false` 时，`cache_salt` 用于隔离不同客户端的缓存块。

```RESTFul
POST /v1/chat/completions
{
    "cache_salt": "1234",
    "cache_sharing": true,
    "model": "Qwen3-32B",
    "messages": [],
    ...
}
```

### 释放缓存接口

```RESTFul
POST /release_kv_cache
{
    "messages_released_index": 20,
    "tools_released_index": 100,
    "cache_salt": "1234",
    "cache_sharing": true,
    "model": "Qwen3-32B",
    "messages": [],
    "tools": [],
    ...
}
```

请求体各字段与 `/v1/chat/completions` 基本一致，额外增加：

| 字段 | 说明 |
|------|------|
| `messages_released_index` | 压缩后 messages 开始变化的下标；若 messages 未变，传 `messages` 的长度 |
| `tools_released_index` | tools 开始变化的下标；若 tools 未变，可传 tools 最大长度或不传 |

**响应示例：**

```json
{
    "cache_salt": "1234",
    "block_released": 8
}
```

| 字段 | 说明 |
|------|------|
| `cache_salt` | 回显请求中的 `cache_salt` |
| `block_released` | 本次从 session 解绑的 KV block 数量 |

`block_released` 统计的是 session 解绑数，而非调用瞬间立即移入 release 区的数量。若对应 block 仍被进行中的请求持有（`ref_cnt > 0`），会先登记为 `pending_aging`，待请求结束、block 归还空闲队列后再移入 release 区——这在典型调用时序（chat 完成后立即 release）下是正常现象。

## 可观测性

设置环境变量 `JIUWEN_KV_DEBUG=1`（或 `true` / `yes` / `on`）可开启 `[jiuwen-kv]` 追踪日志，例如：

| 日志阶段 | 含义 |
|----------|------|
| `1-allocate` | 新分配 block |
| `2-prefix_hit` | prefix cache 命中复用 |
| `3-release` | session 解绑；常态为 `pending_aging=N`，表示延后老化 |
| `4-evict` | 从 release/normal 段复用分配 |
