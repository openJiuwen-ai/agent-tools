---
name: relayclaw-jiuwen-log-diagnosis
description: Boundary-first diagnosis for OfficeClaw/RelayClaw/JiuWenClaw failures from collected logs. Use this skill whenever the user provides jiuwenclaw full.log, relayclaw/api logs, desktop-launcher logs, session history.json, LLM_IO_TRACE logs, scheduled task failures, sidecar/runtime/process lifecycle events, AbortError, frontend messages like "这次处理没有顺利完成" or "模型调用异常", session interruption/history loss, multi-agent context issues, PPT/HTML/output-path/file-delivery anomalies, skill installation failures, repeated/no responses, or asks whether a problem belongs to jiuwenclaw or the upper RelayClaw/OfficeClaw/frontend/tooling layer. The skill starts by recalling likely historical cases from the single-file case repo, then validates them against current logs to decide whether the boundary is jiuwenclaw or upper RelayClaw/OfficeClaw/frontend/tooling, and only uses history/trace/source tools when they add evidence.
---

# RelayClaw / JiuWenClaw 日志诊断技能

首要职责：**定界**。先回答“这是不是 jiuwenclaw 问题”，再分析直接原因、证据强度和修复方向。不要先解释所有日志，也不要因为前端显示 `模型调用异常` 或 `这次处理没有顺利完成` 就预设根因。

适用场景：

- 前端失败文案、无回复、重复展示、长时间无响应。
- 定时任务、休眠/唤醒、隔夜、登录态恢复后的首次触发失败。
- sidecar/runtime/process 生命周期：`AbortError`、`runtime_signature_changed`、进程退出、WebSocket close。
- session/history/多 Agent/mention/dispatch/产物路径/投递/工具链异常。
- 需要判断边界属于 RelayClaw API、scheduler、MaaS/session、frontend/outbound、jiuwenclaw E2A/LLM/tool/session memory，还是外部工具链。

---

## 1. 参考文件分工

本技能按 harness 结构维护，`SKILL.md` 只保留入口规程和高频定界原则。

```text
relayclaw-jiuwen-log-diagnosis/
├── SKILL.md
├── references/
│   ├── architecture.md          # 架构边界和链路层级
│   ├── key-log-map.md           # 当前代码关键日志地图：原文、字段、含义、误判
│   ├── fingerprints.md          # 高频日志指纹与反误判规则
│   ├── log-bundle-structure.md  # 4.2 采集包结构
│   ├── issue-taxonomy.md        # 测试现象分流
│   ├── external-tools.md        # history / trace 工具说明
│   ├── source-checkout.md       # 源码版本查证规则
│   ├── harness-principles.md    # 技能维护准则
│   └── report-template.md       # 正式报告模板
├── cases/
│   └── log_case_repo.md         # 单文件历史案例库
├── scripts/
│   ├── office_claw_log_collector_4.2.bat
│   └── analyze_history_tool_call_speed.py
└── evals/
    └── evals.json
```

使用策略：

- 开始分析时先轻量检索 `cases/log_case_repo.md`，把历史案例当候选方向，不能直接套结论。
- 架构边界不清时读 `references/architecture.md`。
- 需要查日志原文、字段、源码位置和误判规则时读 `references/key-log-map.md`。
- 看到明确错误词或前端文案时读 `references/fingerprints.md`。
- 用户给采集包目录时读 `references/log-bundle-structure.md`。
- 用户描述的是效果、性能、路径、中断、多 Agent 等测试标题时读 `references/issue-taxonomy.md`。
- 需要 history/trace 工具时读 `references/external-tools.md`。
- 需要源码修复点、文案映射、fallback/状态机语义时读 `references/source-checkout.md`，优先使用用户给出的本地或对应版本源码。
- 要输出正式复盘材料时读 `references/report-template.md`。

---

## 2. 最小定界原则

第一句话必须体现边界：

> 问题边界在 `<layer>/<module>`，这是/不是 jiuwenclaw 问题。直接原因是 `<evidence-backed cause>`。

最低链路检查：

```text
Frontend / Scheduler / Connector
  -> API Created invocation?
  -> API jiuwen request sent?
  -> sidecar/runtime stable?
  -> jiuwenclaw Inbound raw payload / [E2A][in]?
  -> jiuwenclaw LLM / ReAct / tool loop?
  -> jiuwenclaw complete/error chunk?
  -> API state update / error fallback?
  -> outbound delivery / frontend display?
```

证据强度：

- `confirmed`：同一时间窗口、同一 ID 链路上有入口、失败点和排除其他层的证据。
- `probable`：核心证据足够指向某层，但缺少一个关键出口、关联 ID 或版本语义确认。
- `insufficient`：只能说明现象，无法把失败绑定到具体层。

不要把以下内容单独当根因：

- `tick completed`：只说明 scheduler tick 完成，不说明业务处理成功。
- `Created invocation`：只说明 API 创建调用，不说明 jiuwenclaw 已收到请求。
- `jiuwen request sent`：只说明上层已发送，不说明 jiuwenclaw 已处理。
- `Inbound raw payload`：只说明请求到达 jiuwenclaw WebSocket handler，不说明已进入 LLM。
- `Before create openai client`：只是准备创建模型客户端，不等于模型调用失败。
- `keepalive`：只是连接保活，不等于 LLM 或工具仍在推进。
- `AbortError`：只是中止表现，必须追查 timeout、cancel、sidecar stop、runtime signature、process exit。
- `session_memory_update_*`：通常是后台记忆更新，不等同用户主任务。
- `No bindings found for thread`：通常是投递/渠道绑定问题，不是模型失败。

### 2.1 高频强制分流提醒

遇到下列现象时，先按这里分流，再展开完整链路：

- `Huawei MaaS session not found`、登录态/session/auth/token 相关错误：优先判上层 API / MaaS session，不要先查 jiuwenclaw LLM。
- `LLM invoke 失败`、`openAI API async ... error`、`APIConnectionError`、重试耗尽：优先判 jiuwenclaw LLM 调用链路，但要确认是否属于主 `request_id/session_id`。
- `AbortError`、`jiuwen WebSocket connection closed unexpectedly`、`runtime_signature_changed`、`relayclaw sidecar stop invoked`、`relayclaw sidecar exited`：优先查 sidecar/runtime 生命周期，不要只写 invocation 崩溃。
- `OutboundDeliveryHook`、`No bindings found for thread`、外部渠道没收到：优先查 outbound/channel binding，不要归因模型失败。
- `session_memory_update_*`、`[SessionMemory] update_*`：优先按后台记忆更新处理，必须用主 `invocationId/request_id` 证明它影响了用户任务。
- `SpawnAgent`、`subagent_`、多 Agent 看不到上下文：优先查子代理/session/history，而不是只看主 LLM trace。

强制查阅规则：

- 遇到未知日志、字段含义不清、或需要解释日志原文时，先读 `references/key-log-map.md`。
- 遇到明确错误词、前端文案或高频现象时，先读 `references/fingerprints.md`。
- 遇到与历史问题相似的描述时，先读 `cases/log_case_repo.md`，但必须用本次日志验证后才能复用结论。

---

## 3. 输入与文件优先级

如果用户给的是 `office_claw_log_*` 采集包：

1. 读 `MANIFEST.txt`：确认版本、采集范围、安装目录、数据目录和时区线索。
2. 读 `install_logs/api*.log`、`desktop-launcher.log`：先定上层 invocation、MaaS/session、sidecar、AbortError。
3. 如果请求进入 jiuwenclaw，再读：
   - `runtime_logs/full.log`：E2A、LLM trace、`[LLM_IO_TRACE]`、`LLM_CALL_START/END`、keepalive。
   - `runtime_logs/openjiuwen/run/jiuwen.log`：ReAct、工具、SessionMemory、SpawnAgent、LLM request/response。
   - `runtime_logs/openjiuwen/llm.log`：模型调用结构化日志、耗时、错误。
4. 根据问题类型再读 `sessions/<session_id>/history.json`、gateway 日志或 trace 报告。

如果用户没有采集包，先要最少信息：

- 前端完整文案和问题发生时间（本地时间/时区）。
- 问题入口：手动对话、定时任务、外部渠道、工具调用、继续会话、后台任务。
- 上层 API/launcher 日志、jiuwenclaw `full.log`、`jiuwen.log`、`llm.log`、目标 `history.json`。

---

## 4. 时间与关联 ID

先统一时间，再串 ID。

- `api*.log` 的 `time` 多为 UTC ISO，例如 `2026-05-08T02:05:32.044Z`。
- `full.log` 常见为本地时间，例如 `2026-05-08 10:05:42.929`。
- 北京时间换算：`2026-05-08T02:05:32Z` = `2026-05-08 10:05:32`。

必须优先抓这些 ID：

| ID | 所属层 | 用途 |
|---|---|---|
| `invocationId` | RelayClaw API | 判断上层是否创建调用、是否 fallback、是否完成。 |
| `requestId` / `request_id` | RelayClaw ↔ jiuwenclaw / E2A / LLM trace | 串联 `jiuwen request sent`、`Inbound raw payload`、`[E2A][in]`、chunk、LLM trace。 |
| `sessionId` / `session_id` / `cliSessionId` | 会话和 history | 找 history、区分主会话/后台 memory/subagent。 |
| `threadId` | 前端/渠道线程 | 判断 outbound、UI 线程和 connector。 |
| `task_id` | scheduler | 判断定时任务触发。 |
| `sidecarPid` | sidecar/runtime | 串进程 spawn、stop、exit、kill。 |

同一个 `session_id` 不代表同一次业务调用；必须结合时间窗口、`request_id`、`invocationId` 和调用类型。

---

## 5. 标准分析流程

### Step 0：候选召回

先读 `cases/log_case_repo.md` 的分类、案例标题和关键日志特征。把命中的历史案例标为 `candidate`，用本次日志验证时间窗口、ID 和入口/出口后才能复用结论。

### Step 1：校验日志范围

确认日志覆盖问题发生前后：

- 普通问题先取前后 5 分钟。
- 休眠/隔夜/唤醒/进程生命周期问题扩大到 10-30 分钟。
- 发现 `AbortError`、WebSocket close、缺少自然 complete/error chunk 时，至少查失败点前后 30 秒的 sidecar/process 事件。

### Step 2：先查上层

在 `api*.log` / `desktop-launcher.log` 找：

```text
Created invocation
Dispatching to agent service
jiuwen request sent
Agent service emitted error message
Invocation hard timeout fired
invokeSingleCat crashed before fallback error emission
Huawei MaaS session not found
AbortError
runtime_signature_changed
relayclaw sidecar stop invoked
relayclaw sidecar exited
jiuwen WebSocket connection closed unexpectedly
OutboundDeliveryHook
No bindings found for thread
```

判断：

- 是否创建 invocation？
- 是否发送 jiuwen request？
- 是否在发送前因 MaaS/session/auth/config 失败？
- 是否 sidecar/runtime 在同一窗口被 stop、重启、kill 或退出？
- 是否只是投递/渠道绑定失败？

### Step 3：再查 jiuwenclaw 入口和执行流

在 `full.log` / `jiuwen.log` 找同一 `request_id/session_id`：

```text
Inbound raw payload
Inbound raw payload json 解析成功
Inbound raw payload E2A协议解析成功
[E2A][in]
收到请求
处理 chat 流式请求
流式响应开始 / 流式响应已发送
[E2A][wire][out] chunk
ReAct iteration
[LLM] >>> request
[LLM] <<< response
LLM_CALL_START / LLM_CALL_END / LLM_CALL_ERROR
[LLM_IO_TRACE] event=*_request / *_output
Executing tool:
[SessionMemory] ...
[SpawnAgent] ...
```

判断：

- 请求是否真正到达 jiuwenclaw？
- 是否已转换为内部 `AgentRequest`？
- 是否进入 ReAct / LLM？
- LLM 是否返回？是否产生 tool call？
- 工具或子 agent 是否卡住/失败？
- jiuwenclaw 是否发出 complete/error chunk？

详细字段和日志语义查 `references/key-log-map.md`；不要在入口文件里扩展单条日志。

### Step 4：交叉验证

- `full.log` 无 `invoke_output` 不等于 LLM 挂死；先查 `jiuwen.log` 是否有 `[LLM] <<< response`、后续 tool call 或新 ReAct iteration。
- 持续 keepalive 不等于 SSE/LLM 正常推进；查最后一个 LLM response、tool、ReAct iteration 和上层 timeout/abort。
- `SessionMemory update_background finalize` 后无新事件只能作为候选停滞信号，不能直接写成确认根因。
- 上层有 `Agent service emitted error message` 时，要回到 jiuwenclaw 或 transport 找原始错误。
- jiuwenclaw 已有 final/complete，但前端失败时，转查 API queue、event transform、friendly error、outbound、frontend display。

### Step 5：必要时使用 history / trace

只有在已确定目标 `session_id` 或候选范围足够小时使用外部工具。

- 过程/路径/工具/中断/多 Agent 问题：分析 `sessions/<session_id>/history.json`。
- 工具调用前后很慢：运行 `scripts/analyze_history_tool_call_speed.py` 区分 LLM tool-call delta 和工具真实执行耗时。
- 上下文、prompt、tool_calls、模型为什么这么做：分析 `[LLM_IO_TRACE]` 或 trace 报告。

不要无目标批量跑所有 session；如果问题明确发生在 Agent 启动前，先完成上层定界。

### Step 6：形成结论

结论必须回答：

1. 是不是 jiuwenclaw 问题？
2. 如果不是，是 API、scheduler、MaaS/session、sidecar/runtime、fallback、outbound、frontend 还是安装环境？
3. 如果是，是 WebSocket/E2A、agent runtime、prompt/context、LLM、工具、SessionMemory、trace、文件系统哪一块？
4. 关键证据是哪几条日志？
5. 为什么不是相邻层？
6. 证据强度是 `confirmed`、`probable` 还是 `insufficient`？

---

## 6. 高频分流矩阵

| 现象/日志 | 优先边界 | 下一步 |
|---|---|---|
| `Created invocation` 后无 `jiuwen request sent` | API / MaaS / auth / config | 查 API error、MaaS session、fallback。 |
| `jiuwen request sent` 后无 `Inbound raw payload` / `[E2A][in]` | WebSocket / sidecar / transport | 查 connection ack、sidecar ready/exit、WS close。 |
| 有 `[E2A][in]`，无 ReAct/LLM | jiuwenclaw request routing / AgentManager / context | 查 `收到请求`、`处理 chat 流式请求`、AgentManager。 |
| 有 LLM request，无 LLM response | LLM/网络/trace 缺失/上层中断 | 用 `jiuwen.log` 与 `LLM_IO_TRACE` 交叉验证。 |
| 有 `[LLM] <<< response` 和 `Executing tool:` 后停 | 工具 / 子 agent / 文件系统 / 外部服务 | 查 tool result、SpawnAgent、history。 |
| `LLM invoke 失败` / `APIConnectionError` / 重试耗尽 | jiuwenclaw LLM 调用链路 | 查模型网关、网络、证书、代理、API key、服务可用性。 |
| `AbortError` + sidecar stop/exit/runtime signature | 上层 sidecar/runtime 生命周期 | 查触发 stop/restart 的 reason/signatureDiff。 |
| `No bindings found for thread` | outbound / channel binding | 查 thread binding、connector、投递策略。 |
| `session_memory_update_*` 出错 | 后台记忆更新 | 用主 `invocationId/request_id` 判断是否影响用户任务。 |
| 前端只有 `这次处理没有顺利完成` | 兜底文案 | 用 error-audit/raw error/API 日志反查，不要直接归 jiuwenclaw。 |

更多指纹查 `references/fingerprints.md`。

---

## 7. 源码查证规则

默认不要一上来拉远端最新代码。代码只用于解释日志语义、状态机、fallback 或给修复点。

优先级：

1. 用户提供的本地路径、仓库、分支、tag、commit。
2. 日志/`MANIFEST.txt` 可识别发布版本时，请求对应版本源码或 checkout 到对应版本。
3. 无版本信息时，仅在日志和参考资料不足以定界、或用户明确要代码修复点时，才查默认分支，并在结论中标注源码版本未确认一致。

必须源码/架构确认后才能给 `confirmed` 的情况：

- 未知模块、未知错误码、未知链路名。
- 同一现象同时符合上层和 jiuwenclaw 两种边界。
- 结论依赖 fallback、错误文案映射、状态覆盖、complete/error chunk 语义。
- 用户明确要求修复点、补丁或 PR。

---

## 8. 输出格式

默认回答结构：

```markdown
## 结论
问题边界在 <layer/module>，这是/不是 jiuwenclaw 问题。直接原因是 <cause>。证据强度：confirmed/probable/insufficient。

## 证据链
- <本地时间>，<日志文件>，<关键日志原文/字段>，说明 <含义>。
- <本地时间>，<日志文件>，<关键日志原文/字段>，说明 <含义>。

## 为什么不是另一层
说明为什么不是 jiuwenclaw / 上层 / 前端 / 投递。

## 修正策略
- 立即规避：...
- 研发修复：...
- 测试补充：...

## 还需要补充的日志
证据不足时列出最少文件和时间窗口。
```

如果用户没看懂，改成极简解释：

```markdown
一句话结论：问题出在 <层级>，不是 <容易误解的层级>。

为什么：<日志原文> 表示 <含义>。

前端为什么显示那句话：因为 <错误被兜底包装/投递失败/下游错误透传>。
```

---

## 9. 最小证据标准

不要在证据不足时给过度确定结论。最低需要：

- 用户现象和发生时间。
- 上层 API 或 launcher 同一时间窗口。
- jiuwenclaw `full.log` 或 `jiuwen.log` 同一时间窗口。
- 至少一个关联 ID：`invocationId`、`request_id/requestId`、`session_id/sessionId`、`task_id`。

缺证据时明确写：

> 目前能确定 `<已有结论>`，但还不能确认 `<缺失部分>`。需要补充 `<文件>` 中 `<时间窗口>` 的日志。

---

## 10. 维护准则

- `SKILL.md` 只保留入口流程、定界矩阵、证据标准和输出要求。
- 稳定知识放 `references/`，真实已定位案例放 `cases/`，确定性操作放 `scripts/`。
- 新案例优先追加到 `cases/log_case_repo.md`；只有变成通用规律后，才提升到 `references/fingerprints.md` 或 `references/issue-taxonomy.md`。
- 关键日志新增或语义变化时，优先更新 `references/key-log-map.md`，不要把单条日志解释堆回入口文件。
- 外部工具和源码是深入分析手段，不是入口；先完成最小定界，再决定是否使用。
- 最终回答先给结论，少贴日志，只引用最关键原文；对不确定点明确标注。

<!-- evolution-index-start -->
## Evolution Experiences

This skill has accumulated **2** evolution experiences (2 body).

### Top Experiences

- [ev_6909ec7f] (score=0.50) ## 非UTF-8编码日志文件的读取处理
- [ev_6d61d387] (score=0.50) ## 大体积API日志的grep搜索策略

| Type | Count | Details |
|------|-------|---------|
| Instructions | 1 | [→ evolution/instructions.md](evolution/instructions.md) |
| Troubleshooting | 1 | [→ evolution/troubleshooting.md](evolution/troubleshooting.md) |

*Last updated: 2026-05-12T09:11:13+00:00*
<!-- evolution-index-end -->
