# 关键日志地图

这份文档是排障时的“先查什么日志”地图，按链路层级组织。它不替代 `fingerprints.md` 的案例化指纹，也不替代 `architecture.md` 的结构说明；它用于快速找到关键日志、字段和边界含义。

适用当前本地代码基线：

- RelayClaw：`E:\code\relay-claw`
- jiuwenclaw：`E:\code\relay-claw\vendor\jiuwenclaw`
- openjiuwen：`E:\code\relay-claw\vendor\jiuwenclaw\.venv\Lib\site-packages\openjiuwen`

判读原则：

- 先按时间窗口和 ID 串链路，再解释单条日志。
- 每条日志都只证明它所在层级已经走到某一步，不自动证明下游成功。
- `invocationId`、`requestId/request_id`、`sessionId/session_id`、`threadId`、`sidecarPid` 是最小关联字段。
- 如果日志来自 DEBUG 或敏感模式受控路径，要先确认采集包是否开启了对应级别。

## 1. RelayClaw API / Invocation

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `Created invocation` | `packages/api/src/domains/agents/services/agents/invocation/invoke-single-agent.ts` | `invocationId`, `agentId`, `threadId`, `userId`, `traceId` | 上层调用已创建，是 API 层业务调用的起点。 | 不代表已经发给 jiuwenclaw；后面必须找 `jiuwen request sent`。 |
| `Dispatching to agent service` | `invoke-single-agent.ts` | `invocationId`, `agentId`, `sessionId`, `attempt`, `promptLength` | 即将调用具体 agent service。 | debug 级别；没有这条不一定没 dispatch。 |
| `Agent service emitted error message` | `invoke-single-agent.ts` | `invocationId`, `agentId`, `threadId`, `sessionId`, `error` | 下游 service 返回 `type=error`，上层记录错误。 | 它是汇聚点，不是根因；要继续用 `requestId/sessionId` 查 jiuwenclaw 或 transport。 |
| `Invocation hard timeout fired` | `invoke-single-agent.ts` | `invocationId`, `agentId`, `threadId`, `timeoutMs` | 上层 invocation 硬超时触发，会 abort 等待中的调用。 | 后续 `AbortError` 可能只是超时结果；查 timeout 前是否有模型/工具/sidecar 停滞。 |
| `invokeSingleCat crashed before fallback error emission` | `invoke-single-agent.ts` | `invocationId`, `agentId`, `threadId`, `error` | 单个 agent 调用在 fallback 错误发出前崩溃。 | 这是错误处理失效信号；根因通常在上一条错误或 sidecar/会话/模型。 |

优先串联：

```text
Created invocation
-> Dispatching to agent service
-> jiuwen request sent
-> Agent service emitted error message / completed
```

## 2. RelayClaw Agent Service / WebSocket

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `jiuwen request sent` | `packages/api/src/domains/agents/services/agents/providers/RelayClawAgentService.ts` | `requestId`, `agentId`, `sessionId`, `promptLen`, `traceId` | RelayClaw 已构造请求并调用 WebSocket `send()`。 | 只能说明上层发送动作完成；仍需在 jiuwenclaw 侧找 `Inbound raw payload` 或 `[E2A][in]`。 |
| `connection.ack` | `relayclaw-connection.ts` / `relayclaw-event-transform.ts` | `event`, `payload.status` | jiuwenclaw Agent WebSocket ready，RelayClaw 才认为连接可用。 | TCP ready 不等于 app ready；没有 ack 时不要判定 jiuwen 已可用。 |
| `jiuwen frame without request_id — skipped` | `relayclaw-connection.ts` | `eventType` | 收到无法路由的 frame。 | 可能是协议漂移、非请求事件或异常帧；不能归因到模型。 |
| `jiuwen frame for unknown/expired request — possible late delivery` | `relayclaw-connection.ts` | `requestId`, `eventType` | 收到的 frame 找不到活动 queue，常见于请求已超时、已完成或重连后的迟到帧。 | 如果 jiuwenclaw 已完成但上层显示失败，要重点查 queue 是否已被 timeout/abort 清理。 |
| `[USAGE_DEBUG] WS frame received: event_type=%s metadata=%s is_complete=%s payload_keys=%s` | `relayclaw-connection.ts` | `event_type`, `metadata`, `is_complete`, `payload_keys` | 上层收到带 metadata 或 final/error 的 frame。 | 只对部分 frame 打印，不是完整流式日志。 |
| `jiuwen WebSocket connection closed unexpectedly` | `relayclaw-connection.ts` | 无内置 requestId，需结合活动 queue | WebSocket close 时注入到所有活动请求的 `chat.error`。 | 不是模型错误；要查 sidecar stop/exit/runtime signature/进程 kill。 |

优先串联：

```text
jiuwen request sent requestId=...
-> connection.ack 已完成或 ensureConnected 成功
-> jiuwenclaw Inbound raw payload request_id=...
-> WS frame received / chat.error / chat.final
```

## 3. RelayClaw Sidecar / Runtime 生命周期

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `relayclaw sidecar reused (cache hit)` | `relayclaw-sidecar.ts` | `agentId`, `sidecarPid`, `port` | 复用已有 sidecar。 | 如果后续连接失败，查 tcp probe 和进程是否刚退出。 |
| `relayclaw sidecar alive but tcp probe failed — will restart` | `relayclaw-sidecar.ts` | `agentId`, `sidecarPid`, `port` | 进程看似活着但端口不可用，准备重启。 | 边界在 sidecar/runtime，不是 LLM。 |
| `relayclaw sidecar runtime signature changed — restarting` | `relayclaw-sidecar.ts` | `agentId`, `sidecarPid`, `signatureDiff` | 运行时签名变化触发重启。 | `signatureDiff` 是关键证据，常解释调用中断或 WebSocket close。 |
| `relayclaw sidecar stop invoked` | `relayclaw-sidecar.ts` | `agentId`, `sidecarPid`, `willKill`, `reason` | 上层主动 stop sidecar。 | `willKill=true` 说明会杀进程；后续 Abort/WebSocket close 多半是结果。 |
| `relayclaw sidecar spawned` | `relayclaw-sidecar.ts` | `sidecarPid`, `agentPort`, `webPort`, `startCount`, `command`, `args`, `cwd` | sidecar 进程已拉起。 | spawned 不等于 ready。 |
| `jiuwen sidecar tcp_ready` | `relayclaw-sidecar.ts` | `agentId`, `agentPort`, `elapsedMs` | agent port TCP 可连。 | 只说明端口开放，不代表 AgentWebSocketServer 已发 ack。 |
| `jiuwen sidecar app_ready` | `relayclaw-sidecar.ts` | `agentId`, `elapsedMs` | sidecar 日志出现 app ready 标记。 | 仍建议看 `fully ready` 和 `connection.ack`。 |
| `jiuwen sidecar fully ready` | `relayclaw-sidecar.ts` | `agentId`, `agentPort`, `webPort`, `elapsedMs` | RelayClaw 认为 runtime ready。 | ready 后失败需看具体 request。 |
| `relayclaw sidecar exited` | `relayclaw-sidecar.ts` | `sidecarPid`, `code`, `exitSignal`, `uptimeMs`, `logTail`, `stderrPreview` | sidecar 退出。 | 若贴近 invocation 失败，优先查进程生命周期。 |
| `relayclaw sidecar startup failed: process exited before ready` | `relayclaw-sidecar.ts` | `sidecarPid`, `exitCode`, `logTail`, `stderrPreview` | 启动期进程提前退出。 | 请求通常没进入 jiuwen 主链路。 |
| `relayclaw sidecar startup failed: readiness timeout` | `relayclaw-sidecar.ts` | `sidecarPid`, `exitCode`, `logTail`, `stderrPreview` | 启动到超时仍未 ready。 | 查依赖、端口、app 日志、Python 环境。 |

## 4. Scheduler / Connector / Outbound

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `[scheduler] <task>: tick completed` | `packages/api/src/infrastructure/scheduler/TaskRunner.ts` / `execute-pipeline.ts` | `task.id/name`, work item count, duration | 调度 tick 完成或生成 work items。 | 不代表 agent 处理成功；必须继续找 invocation 和投递。 |
| `[ConnectorInvokeTrigger] Duplicate invocation ... skipping` | `ConnectorInvokeTrigger.ts` | `messageId`, `threadId`, `agentId` | 外部消息重复触发被跳过。 | 用户可能表现为“没执行”，但不是 jiuwen 问题。 |
| `[ConnectorInvokeTrigger] Invocation failed` | `ConnectorInvokeTrigger.ts` | `threadId`, `agentId`, error | connector 触发的 invocation 失败汇总。 | 要回到 invocation/agent service 日志定根因。 |
| `[OutboundDeliveryHook] deliver() called` | `OutboundDeliveryHook.ts` | `threadId`, `agentId`, blocks/message | 开始投递结果到外部渠道。 | 说明 agent 可能已有结果，问题可能在投递层。 |
| `[OutboundDeliveryHook] No bindings found for thread — skipping outbound delivery` | `OutboundDeliveryHook.ts` | `threadId` | 没有渠道绑定，跳过投递。 | 如果用户说“没收到外部消息”，这是根因候选；如果前端处理失败，通常不是模型根因。 |
| `[OutboundDeliveryHook] Found bindings, delivering` | `OutboundDeliveryHook.ts` | `threadId`, connector binding | 找到绑定并开始发。 | 还要看 adapter 的发送结果。 |
| `[OutboundDeliveryHook] Phase J: file block skipped — resolver failed and url is not https` | `OutboundDeliveryHook.ts` | file block / url | 文件块无法投递。 | 常见于产物路径/外部 URL 问题，不是 LLM 失败。 |

## 5. jiuwenclaw AgentWebSocketServer 入口

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `[AgentWebSocketServer] 握手检查 path=%s origin=%s allowed=%s` | `vendor/jiuwenclaw/jiuwenclaw/agentserver/agent_ws_server.py` | `path`, `origin`, `allowed` | WebSocket 握手 Origin 校验。 | `allowed=false` 会在进入请求前失败。 |
| `[AgentWebSocketServer] 新连接: %s` | `agent_ws_server.py` | remote address | 上层或内部 gateway 建立连接。 | 连接建立不代表请求已到达。 |
| `[AgentWebSocketServer] 已发送 connection.ack: %s` | `agent_ws_server.py` | remote address | 服务端已发 ready ack。 | ack 只是连接 ready，不是业务请求成功。 |
| `[AgentWebSocketServer] Inbound raw payload: %s` | `agent_ws_server.py` | `request_id`, `channel/channel_id`, `session_id`, `method/req_method`, `is_stream`, `params`, `metadata`, `channel_context` | jiuwenclaw 收到 WebSocket 原始 payload。 | `query/system_prompt/supplementary_info` 会脱敏；这条不代表已进 LLM。 |
| `Inbound raw payload json 解析成功: request_id=...` | `agent_ws_server.py` | `request_id` | JSON 可解析。 | 还没说明 E2A 协议可解析。 |
| `Inbound raw payload json 解析失败: error=...` | `agent_ws_server.py` | parse error | JSON 解析失败，会返回 parse error wire。 | 根因在协议/数据，不是模型。 |
| `Inbound raw payload E2A协议解析成功: request_id=...` | `agent_ws_server.py` | `request_id` | E2AEnvelope 解析成功。 | 仍需看 `[E2A][in]` 和 `收到请求`。 |
| `Inbound raw payload E2A协议解析失败，按旧 payload 解析: ...` | `agent_ws_server.py` | parse error | E2A 解析失败，走 legacy payload。 | 不一定失败；如果后面有 `收到请求`，兼容路径继续。 |
| `[E2A][fallback] using legacy_agent_request request_id=%s` | `agent_ws_server.py` | `request_id` | E2A fallback 使用 legacy request。 | 查 `channel_context` 内部 fallback 标记。 |
| `[E2A][in] request_id=%s channel=%s method=%s is_stream=%s` | `agent_ws_server.py` | `request_id`, `channel`, `method`, `is_stream` | jiuwenclaw E2A 入口确认。 | 只说明协议入口，不代表 AgentManager/LLM 已执行。 |
| `[AgentWebSocketServer] 收到请求: request_id=%s channel_id=%s is_stream=%s` | `agent_ws_server.py` | `request_id`, `channel_id`, `is_stream` | 已转换为内部 `AgentRequest`。 | 之后可能进入 history/session/file/custom handler，不一定是 chat。 |
| `[AgentWebSocketServer] 处理 chat 流式请求: request_id=...` | `agent_ws_server.py` | `request_id` | 进入流式 chat 主路径。 | 后续要看 `流式响应开始` 和 LLM/ReAct。 |
| `[AgentWebSocketServer] 处理请求失败: request_id=%s: %s` | `agent_ws_server.py` | `request_id`, exception | request 分发或处理异常。 | 这是 jiuwenclaw 内部请求处理错误。 |

## 6. jiuwenclaw E2A / Wire Codec

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `[E2A][norm] request_id=%s channel=%s method=%s is_stream=%s params_keys=%s` | `jiuwenclaw/e2a/gateway_normalize.py` | `request_id`, `channel`, `method`, `params_keys` | gateway payload 归一化结果。 | params_keys 可判断关键字段是否存在，不代表值正确。 |
| `[E2A][enable_memory] ...` | `gateway_normalize.py` | memory flags | enable memory 相关归一化。 | 用于区分记忆链路配置。 |
| `[E2A][compat] unknown E2A method=%r request_id=%s` | `e2a/agent_compat.py` | `method`, `request_id` | 未知 E2A method。 | 常见于协议不匹配或新增 method 未兼容。 |
| `[E2A][wire][out] chunk request_id=%s response_id=%s seq=%s response_kind=%s is_final=%s ...` | `e2a/wire_codec.py` | `request_id`, `response_id`, `seq`, `response_kind`, `is_final` | jiuwenclaw 编码发往上层的流式 chunk。 | `is_final=true` 表示 jiuwen 侧已发最终帧；前端失败需查上层接收/queue/投递。 |
| `[E2A][wire][out][FAIL] stage=...` | `e2a/wire_codec.py` | `stage`, `request_id`, `response_id`, `err` | E2A 输出编码失败。 | 边界在 jiuwenclaw wire 编码/兼容层。 |
| `[E2A][wire][in] chunk/unary ...` | `e2a/wire_codec.py` | `request_id`, `response_kind`, `is_final` | 解析 wire frame。 | 注意 in/out 方向，不要把 gateway client 与 server 方向混淆。 |
| `[E2A][out][stream] request_id=%s channel=%s method=%s is_stream=%s` | `jiuwenclaw/gateway/agent_client.py` | `request_id`, `channel`, `method`, `is_stream` | jiuwen 内部 gateway client 发出流式请求。 | OfficeClaw 主链路通常直接连 AgentWebSocketServer；内部 WebChannel 也会经过 agent_client。 |
| `[WebSocketAgentServerClient] 等待 connection.ack 超时` | `gateway/agent_client.py` | timeout | jiuwen 内部 gateway client 等 ack 超时。 | 这是 jiuwen 内部 WebChannel/Gateway 链路，不一定是 OfficeClaw 主调用链路。 |

## 7. jiuwenclaw Stream / Keepalive

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `[AgentWebSocketServer] DIAGNOSTIC: _handle_stream 开始执行 | request_id=%s | is_stream=%s | channel=%s` | `agent_ws_server.py` | `request_id`, `is_stream`, `channel` | stream handler 已进入。 | 还没说明 AgentManager 产出 chunk。 |
| `[AgentWebSocketServer] 流式响应开始: request_id=...` | `agent_ws_server.py` | `request_id` | 第一个真实 chunk 已从 AgentManager 产出并准备发送。 | 可证明 jiuwen 执行已有输出。 |
| `[AgentWebSocketServer] 流式响应进度: request_id=... chunk_count=...` | `agent_ws_server.py` | `request_id`, `chunk_count` | 每 10 个 chunk 的进度。 | 只能说明 chunk 发送数量，不说明内容质量。 |
| `[AgentWebSocketServer] keepalive chunk 发送: request_id=%s` | `agent_ws_server.py` | `request_id` | 空闲超过心跳间隔发送 keepalive。 | keepalive 不是 LLM 进展；不能单独判定“模型仍在工作”。 |
| `[AgentWebSocketServer] 流式响应已发送: request_id=%s 共 %s 个 chunk` | `agent_ws_server.py` | `request_id`, `chunk_count` | stream handler 自然结束。 | 如果上层失败，查 frame 接收、queue、final/error 映射。 |

## 8. LLM Trace / Model Client

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `[LLM_IO_TRACE] event=stream_request ... body=...` | `jiuwenclaw/agentserver/llm_io_trace.py` | `session_id`, `request_id`, `iteration`, `model_name`, `body` | DEBUG trace：流式 LLM 输入，body 含 messages/tools/model/max_tokens 等。 | DEBUG 才有；body 可能分片。 |
| `[LLM_IO_TRACE] event=invoke_request ... body=...` | `llm_io_trace.py` | 同上 | DEBUG trace：非流式 LLM 输入。 | 只有 request 无 output 不等于模型挂死，要看 ReAct `[LLM] <<< response`。 |
| `[LLM_IO_TRACE] event=reasoning_delta ...` | `llm_io_trace.py` | `reasoning_seq`, body | reasoning 流片段。 | 可用于判断模型是否在持续输出 reasoning。 |
| `[LLM_IO_TRACE] event=stream_output ... body=...` | `llm_io_trace.py` | content, reasoning_content, tool_calls, finish_reason, usage_metadata | DEBUG trace：流式聚合后的 LLM 输出。 | 如果输出是 tool_calls，继续查工具执行。 |
| `[LLM_IO_TRACE] event=invoke_output ... body=...` | `llm_io_trace.py` | 同上 | DEBUG trace：非流式 LLM 输出。 | 需要和 `[LLM] <<< response` 交叉验证。 |
| `[LLM_IO_TRACE] event=chat.final` | `llm_io_trace.py` | `session_id`, `request_id`, `iteration`, `model_name` | 用户可见 final 边界，不记录 final 内容。 | final 边界不是内容证据。 |
| `LLM_CALL_START` / `llm_call_start` | `openjiuwen/core/foundation/llm/model_clients/*` | `model_name`, `model_provider`, `messages`, `tools`, `is_stream`, sampling params | 模型客户端请求参数准备好，开始调用模型服务。 | BaseModelClient 记录 start；具体客户端才有 end/error。 |
| `LLM_CALL_END` / `llm_call_end` | `openjiuwen/core/foundation/llm/model_clients/*` | response/error/usage/cost | 模型调用结束或失败。 | 不同 provider 记录字段不同；结合错误栈和 retry。 |
| `Before create openai client` | jiuwenclaw patch / adapter 层 | model/provider/base url 相关上下文 | 准备创建 OpenAI 兼容客户端。 | 不是 LLM 调用失败证据，只是准备阶段。 |
| `LLM invoke 失败` / `model call failed` / `openAI API async ... error` | jiuwenclaw / model client | error type, retry count | 模型调用失败。 | 有重试时要看是否最终成功；不要只截第一条。 |
| `retry notification sent successfully` | jiuwenclaw retry path | attempt / max attempts | 已向上层/用户发送重试提示。 | 它说明有 retry 机制，不是最终失败结论。 |

`body_part` 规则：

```text
[LLM_IO_TRACE] ... body_part=1/3 body=...
[LLM_IO_TRACE] ... body_part=2/3 body=...
[LLM_IO_TRACE] ... body_part=3/3 body=...
```

分析时必须按同一 header 拼接所有 `body_part`，否则容易误判 prompt、tools 或输出缺失。

## 9. openjiuwen ReAct / Tool / Context

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `ReAct iteration X/Y` | `openjiuwen/core/single_agent/agents/react_agent.py` | `iteration`, `max_iterations` | ReAct 主循环推进到第几轮。 | 没有下一轮可能是 final、工具中断、异常或卡住，需要看邻近日志。 |
| `[LLM] context_for_model session_id=%s context_id=%s message_count=%s tool_count=%s total_content_chars=%s active_skill_pin_messages=%s` | `react_agent.py` | session/context/message/tool counts | 已形成送模型上下文。 | 用于判断上下文是否异常膨胀、工具是否进入模型。 |
| `[LLM] >>> request: msg_count=..., tool_count=...` | `react_agent.py` | `msg_count`, `tool_count` | ReAct 侧准备调用模型。 | 不代表模型服务已收到；结合 `LLM_CALL_START`。 |
| `[LLM]   msg[i] role=...` | `react_agent.py` | role/content/tool_calls/tool_call_id | 非敏感模式下的消息摘要。 | 敏感模式可能没有详细内容。 |
| `[LLM] <<< response: ...` | `react_agent.py` | content length/content/tool_call_count/tokens | ReAct 侧拿到模型返回。 | 如果 full.log 无 invoke_output，这条可证明模型已返回。 |
| `[LLM]   tool_call: name(args)` | `react_agent.py` | tool name, args | LLM 选择了工具。 | 后续要看 `Executing tool` 和 tool result。 |
| `Executing tool: <name> with args: <args>` | `react_agent.py` | tool name, args | 开始执行工具。 | 说明 LLM 已返回 tool call，问题可能转到工具执行。 |
| `[ReActAgent] get_context_window start/done` | `react_agent.py` | `session_id`, `elapsed_ms`, `messages_out`, `tools_out` | context window 构建耗时和输出规模。 | 性能问题可用来区分慢在上下文构建还是 LLM。 |
| `[ReActAgent] llm.invoke start` / `llm.stream start` | `react_agent.py` | `session_id`, message/tool counts | ReAct 即将调用 invoke/stream。 | 如果后续无 LLM_CALL_START，查 model wrapper/参数构造。 |

## 10. SessionMemory

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `[SessionMemory] initialized ...` | `openjiuwen/core/context_engine/context/session_memory_manager.py` | mode, incremental, thresholds | SessionMemory 初始化。 | 不代表本次会话已更新。 |
| `[SessionMemory] should_update triggered/skipped ...` | `session_memory_manager.py` | token/tool deltas, thresholds | 判断是否需要更新记忆。 | skipped 不是异常。 |
| `[SessionMemory] schedule update ...` | `session_memory_manager.py` | `session_id`, `notes_path`, message count, incremental | 安排后台记忆更新。 | 后台链路，不等同用户主任务。 |
| `[SessionMemory] agent_invoke ...` | `session_memory_manager.py` | `agent`, `conversation_id`, `session_id`, tokens, inputs | 记忆更新 agent 开始执行。 | `conversation_id` 常是 `session_memory_update_*`，不要当主任务。 |
| `[SessionMemory] update complete ...` | `session_memory_manager.py` | `notes_upto`, count, tokens, tool_calls, incremental | 记忆更新完成。 | 只证明 memory 更新，不证明用户任务完成。 |
| `[SessionMemory] update failed ...` | `session_memory_manager.py` | `session_id`, `notes_path`, `pending_notes_path` | 记忆更新失败。 | 需要判断是否影响主任务展示；通常不应覆盖主任务成功。 |
| `[SessionMemory] update_background finalize ...` | `session_memory_manager.py` | `session_obj`, `session_id`, runtime | 后台记忆更新收尾。 | 它之后无事件只是候选停滞信号，不能单独归因。 |
| `[SessionMemoryManager] Session memory background task failed` | `session_memory_manager.py` | exception | 后台任务异常未处理。 | 需区分主任务与后台任务。 |

## 11. SpawnAgent / Subagent

| 日志原文 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `[SpawnAgent] Starting execution, task_id=..., role_id=...` | `jiuwenclaw/agentserver/tools/subagent_executor/executor.py` | `task_id`, `role_id` | 子 Agent 开始执行。 | 主任务可能在等待子 Agent。 |
| `[SpawnAgent] Created spawn agent instance, task_id=%s, max_iterations=%s` | `executor.py` | `task_id`, `max_iterations` | 子 Agent 实例创建。 | 创建成功不代表执行成功。 |
| `[SpawnAgent] workspace_dir=%s source=%s` | `executor.py` | workspace, source | 子 Agent 工作目录。 | 路径问题、文件访问问题重点看这里。 |
| `[SpawnAgent] Inherited ... tools from parent agent` | `executor.py` | inherited count | 子 Agent 继承工具。 | 工具缺失会影响子 Agent 能力。 |
| `[SpawnAgent] Execution completed, task_id=...` | `executor.py` | `task_id` | 子 Agent 完成。 | 还要看主 Agent 是否消费结果。 |
| `[SpawnAgent] task_id=... usage: ...` | `executor.py` | usage | 子 Agent 用量信息。 | 性能/成本分析用。 |
| `[SpawnAgent] Timeout after ... seconds, task_id=...` | `executor.py` | timeout, task_id | 子 Agent 超时。 | 用户主任务卡住可能由子 Agent 超时导致。 |
| `[SpawnAgent] Execution failed: ...` | `executor.py` | exception | 子 Agent 执行失败。 | 需判断是否被主 Agent 捕获/重试。 |

## 12. 前端/友好错误映射

| 日志/文案 | 代码位置 | 关键字段 | 说明 | 误判提醒 |
|---|---|---|---|---|
| `这次处理没有顺利完成` | 前端/错误转换链路 | `agentId`, raw error, reference id | 用户看到的兜底失败文案。 | 不代表 jiuwenclaw；优先查 API/fallback/session/sidecar。 |
| `模型调用异常` | 前端/错误转换链路 | raw model error | 模型类错误的用户文案。 | 仍需确认是否有 jiuwen LLM 错误和重试耗尽。 |
| `timeout_diagnostics` | provider service / web hook | first event, silence duration, process alive | provider 超时诊断 system_info。 | `system_info` 不算实质模型输出；可能触发 retry-without-session。 |

## 13. 最小链路检查清单

每次定界按顺序找这些日志：

```text
1. Scheduler/Frontend/Connector
   [scheduler] tick completed / connector trigger / user message

2. API invocation
   Created invocation
   Dispatching to agent service
   Invocation hard timeout fired

3. RelayClaw -> jiuwen request
   jiuwen request sent requestId=...
   connection.ack

4. sidecar/runtime
   sidecar spawned/tcp_ready/app_ready/fully ready
   runtime signature changed
   sidecar stop invoked
   sidecar exited

5. jiuwen WebSocket/E2A
   Inbound raw payload
   JSON parse ok
   E2A parse ok / fallback legacy
   [E2A][in]
   收到请求

6. jiuwen stream
   处理 chat 流式请求
   流式响应开始
   keepalive chunk
   流式响应已发送
   [E2A][wire][out] chunk ... is_final=True

7. LLM/ReAct/tool
   ReAct iteration
   context_for_model
   [LLM] >>> request
   LLM_CALL_START
   [LLM_IO_TRACE] request/output
   [LLM] <<< response
   Executing tool

8. background/subagent
   SessionMemory schedule/update/finalize
   SpawnAgent start/completed/timeout/failed

9. result delivery
   WS frame received
   Agent service emitted error message
   OutboundDeliveryHook deliver/found/no bindings
```

如果某一层缺失，优先定位在缺失点之前或该层内部；不要跳到下游猜测。
