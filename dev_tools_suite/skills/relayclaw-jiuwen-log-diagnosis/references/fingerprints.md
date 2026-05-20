# 日志指纹库

这份参考用于从日志片段快速判断故障边界。结论必须结合时间窗口和关联 ID。

如果一个指纹已经有完整事故复盘和修复建议，优先在 `cases/log_case_repo.md` 中维护具体案例；本文件只保留跨案例可复用的日志模式、边界含义和反误判规则。

## 1. 上层 API / RelayClaw 指纹

### `Huawei MaaS session not found`

边界：上层 OfficeClaw/RelayClaw API。

常见同伴日志：

```text
Created invocation
Huawei MaaS session not found
invokeSingleCat crashed before fallback error emission
```

含义：

- API 层找不到 Huawei MaaS / ModelArts 会话。
- 常见于隔夜、休眠、唤醒后首次定时任务、登录态过期。
- 请求可能尚未稳定进入 jiuwenclaw 主业务链路。

用户现象：

- 前端显示 `这次处理没有顺利完成`
- 前端没有具体 MaaS/session 文案

推荐结论：

> 问题边界在上层 RelayClaw/OfficeClaw API，直接原因是 Huawei MaaS session 缺失，不是 jiuwenclaw 模型调用失败。

修复方向：

- 定时任务触发前校验/刷新 MaaS session。
- session 缺失时返回明确错误码和用户文案。
- 修复 fallback 前崩溃，避免只显示笼统失败。

### `invokeSingleCat crashed before fallback error emission`

边界：上层 API 错误处理。

含义：

- 单次调用在 fallback 错误消息发出前崩溃。
- 前端可能只能显示兜底失败文案。

分析要点：

- 找同一 `invocationId` 上一条错误。
- 如果上一条是 MaaS/session/auth，则根因不是 fallback 本身，fallback 是二次问题。
- 如果异常类型是 `AbortError`，必须继续追触发者：在失败点前后 30 秒搜索 sidecar/进程生命周期事件。
- 如果没有上一条错误，需要拉 API 源码或更详细日志看异常路径。

### sidecar / 进程生命周期事件

边界：上层 OfficeClaw/RelayClaw sidecar 或进程生命周期管理。

常见同伴日志：

```text
sidecar stop
sidecar start
willKill
runtime_signature_changed
process exit
process spawn
SIGTERM
SIGKILL
sidecarPid
AbortError
invokeSingleCat crashed before fallback error emission
```

含义：

- 进程停止、重启、runtime 签名变化或 kill 可能导致正在进行的 invocation 变成 `AbortError`。
- `AbortError` 不是最终根因，只是请求被中止的表现；必须查谁触发了 abort。
- 如果 sidecar/process 事件贴近 invocation 崩溃，而 jiuwenclaw 没有自然 complete/error chunk，应优先考虑上层进程生命周期边界。

分析要点：

- 在 `api*.log` 和 `desktop-launcher.log` 中，以失败时间点前后 30 秒为最小窗口搜索。
- 同时保留 `invocationId`、`requestId`、`sidecarPid`、进程退出码或 kill 原因。
- 区分用户取消、超时、runtime 变更、进程崩溃、守护进程主动重启。
- 不要只写“invokeSingleCat crashed”，要说明触发 abort 的进程生命周期事件是否存在。

### `Agent service emitted error message`

边界：下游 agent service 已返回错误，上层记录。

含义：

- 通常说明请求已到 jiuwenclaw 或 agent service。
- 需要转到 `full.log` 用 `requestId/sessionId` 查根因。

### `[scheduler] dyn-...: tick completed`

边界：调度器。

含义：

- scheduler tick 完成只代表调度器扫描/触发动作完成。
- 不代表 agent 业务完成。
- 后面必须看 `Created invocation`、`jiuwen request sent`、agent complete。

### `No bindings found for thread - skipping outbound delivery`

边界：投递/渠道绑定。

含义：

- 处理结果可能已经生成，但没有 outbound binding。
- 如果用户说“没收到消息”，这是重要线索。
- 如果用户说“处理失败”，它通常不是模型根因。

## 2. jiuwenclaw 指纹

### `LLM invoke 失败 [连接错误]`

边界：jiuwenclaw 调模型链路。

常见同伴日志：

```text
Before create openai client
model call failed
openAI API async invoke error: Connection error
openAI API async stream error: APIConnectionError
将在 10.0s 后重试
第 1 次重试 / 共 3 次
retry notification sent successfully
```

含义：

- 请求已进入 jiuwenclaw。
- jiuwenclaw 在调用 OpenAI 兼容接口/模型网关时连接失败。
- 如果重试 3 次仍失败，前端可能显示模型调用异常。

修复方向：

- 查网络、代理、DNS、证书、模型服务可用性。
- 查 API key / endpoint / timeout。
- 休眠恢复场景增加连接健康检查。

### `[LLM_IO_TRACE] event=invoke_request`

边界：jiuwenclaw DEBUG trace。

含义：

- 已记录 LLM 输入。
- 可以还原 messages、tools、system prompt、上下文。
- 如果只有 request 没有 output，说明调用中断或日志缺失。

### `[LLM_IO_TRACE] event=invoke_output`

边界：jiuwenclaw DEBUG trace。

含义：

- 已记录 LLM 输出。
- 可以看 content、tool_calls、finish_reason、token。
- 如果 output 是 tool_calls，需要继续找工具结果。

### `Executing tool: <name>`

边界：jiuwenclaw 工具执行。

含义：

- LLM 已产生工具调用。
- 需要找工具结果、异常、耗时。

### `[E2A][wire][out] chunk ... response_kind=e2a.complete is_final=True`

边界：jiuwenclaw 已完成输出。

含义：

- jiuwenclaw 已向上层发送完成 chunk。
- 如果前端仍失败，转查 API 接收、状态更新、投递、前端展示。

### `session_memory_update_...`

边界：后台记忆更新。

含义：

- 不是用户主任务的直接 session。
- 该链路失败不一定导致主任务失败。
- 需要看同时间主 invocation/request。

## 3. 前端现象到日志指纹映射

| 前端现象 | 优先查找 | 候选边界 |
|---|---|---|
| `这次处理没有顺利完成` | `api*.log` 的 `error`, `invokeSingleCat`, `MaaS`, `session`, `sidecar`, `willKill`, `runtime_signature_changed`, `AbortError` | API/fallback/登录态/sidecar |
| `模型调用异常` | `full.log` 的 `LLM invoke 失败`, `model call failed`, `retry` | jiuwenclaw LLM |
| 定时任务没触发 | `[scheduler]`, task id, next trigger | scheduler |
| 定时任务触发但没消息 | `OutboundDeliveryHook`, `No bindings found` | outbound/channel |
| 任务做到一半停了 | `history.json`, `[LLM_IO_TRACE]`, `Executing tool` | agent/tool/LLM |
| 打开电脑后首次失败 | `MaaS session not found`, auth/session | 上层登录态恢复 |

## 4. 反误判规则

- API UTC 时间 `02:05Z` 可能是北京时间 `10:05`，不要误判为凌晨问题。
- `tick completed` 不是业务完成。
- `Before create openai client` 只能说明准备创建模型客户端，不足以说明模型调用失败。
- `session_memory_update` 不是主任务。
- 没有 jiuwenclaw 错误不代表 jiuwenclaw 无日志漏记，可能是请求没进 jiuwenclaw。
- 有同一 `session_id` 不代表同一次调用，还要看 `request_id` 和 `invocationId`。
- `AbortError` 不是根因结论；必须追查超时、取消、sidecar stop、runtime_signature_changed、process kill 等触发者。

