# RelayClaw / JiuWenClaw 架构边界

这份参考用于在日志分析时快速建立系统边界。优先根据日志证据定界，不要先假设某一层有问题。

关键日志逐条语义、字段和误判规则见 `references/key-log-map.md`；本文件只保留架构链路和边界判断。

适用性：

- 本文件记录稳定架构和关键日志语义，不保证等同远端默认分支最新代码。
- 如果用户问题来自特定发布版本，优先以该版本日志、采集包、tag/commit 或研发确认为准。
- 当调用链、状态机、日志字段、错误码、fallback、前端文案或 trace 事件在新版本中变化时，必须更新本文件或在相关条目旁标注适用版本。
- 如果当前日志与本文件描述冲突，不要强套本文件结论；应读取对应版本源码或向用户补要版本信息。

## 1. 总体链路

典型一次 OfficeClaw 调用链路：

1. 前端/桌面/外部渠道触发用户消息或定时任务。
2. 上层 sidecar / runtime 处于可用状态，必要时启动或复用进程。
3. 上层 OfficeClaw/RelayClaw API 创建 `invocationId`。
4. API 绑定或恢复 `cliSessionId` / `threadId` / `catId`。
5. API 通过 relayclaw-agent 把请求发送到 jiuwenclaw，生成 `request_id`。
6. RelayClaw WebSocket 连接等待 jiuwenclaw `connection.ack`，收到后认为 Agent WebSocket ready。
7. jiuwenclaw AgentWebSocketServer 收到原始 payload，记录 `Inbound raw payload`。
8. jiuwenclaw 解析 E2AEnvelope，记录 `[E2A][in]`，再转换为内部 `AgentRequest`。
9. jiuwenclaw 创建/复用 AgentManager 和 JiuWenClawDeepAdapter。
10. jiuwenclaw 构造 system prompt、上下文、工具列表。
11. jiuwenclaw 调 LLM，进入 ReAct/tool loop。
12. jiuwenclaw 通过 E2A chunk 返回内容或错误。
13. API 按 `request_id` 将 frame 投递到 request queue，转换成 OfficeClaw `AgentMessage`。
14. API 更新 invocation 状态，并尝试 outbound delivery。
15. 前端展示完成、失败或兜底错误文案。

判断问题边界时，先问：

- invocation 是否创建？
- sidecar/runtime 在调用期间是否被 stop、kill、重启或 runtime signature 变更？
- 是否生成 request_id 并发送到 jiuwenclaw？
- jiuwenclaw 是否收到 E2A 请求？
- 是否进入 LLM 调用？
- 是否有 complete/error chunk？
- API 是否成功包装错误？
- outbound delivery 是否成功？

## 2. 上层 OfficeClaw / RelayClaw

职责：

- 前端会话、线程、invocation 管理。
- sidecar / runtime 进程生命周期管理。
- 定时任务 scheduler。
- 用户登录态、Huawei MaaS / ModelArts 会话。
- 与 jiuwenclaw agent service 通信。
- 错误包装、fallback、前端文案映射。
- outbound delivery / channel binding。

常见日志：

- `install_logs/api.YYYY-MM-DD.N.log`
- `install_logs/desktop-launcher.log`

核心字段：

- `time`: 多为 UTC ISO 时间。
- `module`: `invoke`、`relayclaw-agent`、`relayclaw-connection`。
- `invocationId`: 上层一次调用 ID。
- `threadId`: 前端/渠道线程。
- `catId`: 通常如 `office`。
- `cliSessionId`: jiuwenclaw 会话 ID。
- `requestId`: 发送给 jiuwenclaw 的请求 ID。
- `sidecarPid`: sidecar/runtime 进程 ID。

关键日志模式：

```text
[scheduler] dyn-...: tick completed
Created invocation
Session init: binding session
sidecar start
sidecar stop
willKill
runtime_signature_changed
process exit
process spawn
SIGTERM
SIGKILL
jiuwen request sent
Agent service emitted error message
invokeSingleCat crashed before fallback error emission
OutboundDeliveryHook deliver() called
No bindings found for thread - skipping outbound delivery
Invocation ... completed
```

### 2.1 readiness / queue 边界

RelayClaw 侧要区分四种“可用”：

- `sidecar spawned`：进程已启动，不代表端口、应用或 Agent WebSocket 可用。
- `tcp_ready` / `app_ready` / `fully ready`：runtime 启动逐步就绪，但仍要看具体请求是否发送成功。
- `connection.ack`：jiuwenclaw Agent WebSocket 已应用层 ready，不代表某个业务请求已处理。
- active request queue：API 按 `requestId` 等待 jiuwenclaw frame；超时、abort 或连接关闭后，后续 frame 可能变成 late / unknown request。

因此，`jiuwenclaw 已完成但前端失败` 这类问题不能只看 jiuwenclaw final。还要查 RelayClaw 是否已清理 request queue、是否出现 hard timeout / `AbortError`、是否收到 unknown/expired request frame，以及 outbound / frontend 映射是否失败。

## 3. jiuwenclaw

职责：

- 接收 E2A / WebSocket 请求。
- 管理 agent workspace、配置、工具。
- 构造 prompt / context / skill / memory。
- 调用 LLM。
- 执行工具。
- 写 session memory / notes。
- 输出 chunk / complete / error。
- 记录 `LLM_IO_TRACE`。

常见日志：

- `runtime_logs/full.log`
- `runtime_logs/openjiuwen/run/jiuwen.log`
- `runtime_logs/openjiuwen/llm.log`

核心字段：

- `request_id`: E2A/LLM 调用主关联 ID。
- `session_id`: 会话 ID，通常如 `officeclaw_<hex>`。
- `iteration`: LLM/ReAct 迭代。
- `model_name`: 如 `glm-5.1`。
- `response_kind`: E2A 输出类型，如 `e2a.complete`。

关键日志模式：

```text
[AgentWebSocketServer] Inbound raw payload
[E2A][in] request_id=...
[AgentWebSocketServer] 收到请求
[TenantAgentPool] 创建新 AgentManager 实例
[AgentManager] Creating officeclaw agent
[JiuWenClawDeepAdapter]
ReAct iteration ...
Before create openai client
LLM invoke 失败
retry notification sent successfully
[LLM_IO_TRACE] event=invoke_request
[LLM_IO_TRACE] event=invoke_output
Executing tool:
[E2A][wire][out] chunk request_id=...
流式响应已发送
```

### 3.1 E2A 后的请求分流

`Inbound raw payload` / `[E2A][in]` 只证明请求进入 jiuwenclaw WebSocket / E2A 层，不保证进入 LLM。当前 AgentWebSocketServer 会先把请求转换为 `AgentRequest`，再按 `req_method` / `channel_id` / `event_type` 分流：

- `chat.send` / `chat.resume` / `chat.answer` 等主对话请求：进入 stream / unary chat 路径，后续才可能出现 AgentManager、ReAct、LLM、tool 日志。
- `session.*`、`history.*`、`command.*`、`browser.*`、`config.*`、`extensions.*`：属于管理或控制请求，通常不应当按模型调用失败分析。
- `file_transfer.*`：属于文件传输链路，重点看 payload、路径、编码和 wire response。
- 自定义 WebSocket handler：可能绕过标准 ReqMethod 路由，需看 handler 名称、返回 payload 和异常。

所以看到 `[E2A][in]` 后，下一步必须确认 `收到请求`、`处理 chat 流式请求` / `处理 chat 非流式请求`，或对应的非 chat 分支日志，再决定是否继续追 LLM。

### 3.2 Agent 执行层

主 chat 路径大致是：

```text
AgentWebSocketServer
  -> TenantAgentPool
  -> AgentManager
  -> JiuWenClawDeepAdapter / DeepAgent
  -> openjiuwen ReActAgent
  -> LLM / tool / SessionMemory / SpawnAgent
```

边界判断：

- `TenantAgentPool` / `AgentManager` 负责 agent 实例、workspace、配置和会话复用；卡在这里通常还没进入 openjiuwen ReAct。
- `JiuWenClawDeepAdapter` / `DeepAgent` 负责把 jiuwenclaw 请求接入 openjiuwen runtime，并组织 prompt、skills、tools、stream 输出。
- `ReAct iteration`、`[LLM] >>> request`、`[LLM] <<< response`、`Executing tool:` 才是 openjiuwen ReAct / tool loop 的推进证据。
- `[SessionMemory]` 多数是后台记忆更新；`[SpawnAgent]` / subagent 日志表示主 Agent 进入子代理链路，需区分主任务与子任务。

## 4. ID 关系

| ID | 所属层 | 作用 | 常见位置 |
|---|---|---|---|
| `invocationId` | RelayClaw API | 一次上层调用 | `api*.log`, `desktop-launcher.log` |
| `request_id` / `requestId` | RelayClaw 与 jiuwenclaw 之间 | E2A 请求、LLM trace 串联 | API、`full.log`、`LLM_IO_TRACE` |
| `session_id` / `cliSessionId` | 会话 | history/trace/session memory 串联 | API、`full.log`、`sessions/*/history.json` |
| `threadId` | 前端/渠道线程 | outbound 和 UI 线程 | API、launcher |
| `task_id` | scheduler | 定时任务 | `[scheduler] dyn-...` |
| `sidecarPid` | sidecar/runtime | 进程生命周期、kill、重启串联 | API、launcher |

不要只用一个 ID 下结论。可靠定界通常至少需要：时间窗口 + `invocationId` + `request_id` 或 `session_id`。

## 5. 常见边界判断

### API 创建 invocation 后没有 `jiuwen request sent`

优先怀疑上层 API / 登录态 / 权限 / MaaS session。

如果同时出现 `AbortError`，还必须查前后 30 秒是否有 sidecar stop、`willKill`、`runtime_signature_changed`、process exit 或 SIGTERM/SIGKILL。此时 invocation 崩溃可能只是进程生命周期事件的结果。

### 有 `jiuwen request sent`，但 jiuwenclaw 无 `[E2A][in]`

优先怀疑 API 到 jiuwenclaw 的连接、agent service、WebSocket、进程状态。

同样要检查 sidecar/runtime 是否在请求发送后被重启或杀掉。

### jiuwenclaw 有 `[E2A][in]` 和 `Before create openai client`，随后 `LLM invoke 失败`

优先怀疑 jiuwenclaw 到模型网关的连接、认证、模型服务。

### jiuwenclaw 完成，但 API/launcher 出现 `No bindings found for thread`

优先怀疑 outbound/channel binding，不是模型执行失败。

### `session_memory_update_...`

这是后台会话记忆更新链路，不能直接等同用户主任务。

