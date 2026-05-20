---
name: relayclaw-jiuwen-log-diagnosis
description: Boundary-first diagnosis for OfficeClaw/RelayClaw/JiuWenClaw failures from collected logs. Use this skill whenever the user provides jiuwenclaw full.log, relayclaw/api logs, desktop-launcher logs, session history.json, LLM_IO_TRACE logs, scheduled task failures, sidecar/runtime/process lifecycle events, AbortError, frontend messages like "这次处理没有顺利完成" or "模型调用异常", session interruption/history loss, multi-agent context issues, PPT/HTML/output-path/file-delivery anomalies, skill installation failures, repeated/no responses, or asks whether a problem belongs to jiuwenclaw or the upper RelayClaw/OfficeClaw/frontend/tooling layer. The skill starts by recalling likely historical cases from the single-file case repo, then validates them against current logs to decide whether the boundary is jiuwenclaw or upper RelayClaw/OfficeClaw/frontend/tooling, and only uses history/trace/source tools when they add evidence.
---

# RelayClaw / JiuWenClaw 日志诊断技能

这个技能用于分析 OfficeClaw/RelayClaw/JiuWenClaw 的真实测试问题，尤其是：

- 前端失败文案或用户现象需要定界：`这次处理没有顺利完成`、`模型调用异常`、未回复、重复展示、长时间无响应
- 定时任务/周期任务在休眠、唤醒、隔夜或登录态恢复后首次触发失败，需要区分 scheduler、RelayClaw API、MaaS session 与 jiuwenclaw
- sidecar/runtime/process 生命周期问题：`AbortError`、`sidecar stop`、`willKill`、`runtime_signature_changed`、进程退出或 kill 导致 invocation 中断
- 会话生命周期问题：任务中断后继续、历史丢失、多 session 相互中断、新对话中止长任务、checkpoint 未落盘
- 多 Agent / mention / dispatch 协作问题：后续 Agent 看不到前序回复、上下文传递不完整、调用顺序与产品预期不一致
- 产物与工具链问题：PPT/HTML 转换异常、图表或渲染不一致、输出路径混用、发送不存在文件、技能安装文件校验失败
- jiuwenclaw 的 `full.log` 与上层 `api.*.log` / `desktop-launcher.log` 结论对不齐，需要判断边界属于 jiuwenclaw、RelayClaw API、前端、调度器、MaaS 登录态、模型网关、工具调用、投递或外部工具链

首要职责：定界。先回答“这是不是 jiuwenclaw 问题”，再进一步分析是 API、调度器、MaaS/session、agent service、jiuwenclaw LLM、工具、投递、前端展示还是安装环境问题。所有深入分析都必须服务于这个边界结论。

---

## 技能包结构

这个技能不是单文件提示词，而是一个可运行的诊断工具包：

```text
relayclaw-jiuwen-log-diagnosis/
├── SKILL.md
├── references/
│   ├── architecture.md       # RelayClaw/JiuWenClaw 架构与日志地图
│   ├── fingerprints.md       # 关键日志指纹库与误判规则
│   ├── log-bundle-structure.md # 4.2 日志采集包结构与使用方式
│   ├── issue-taxonomy.md     # 历史测试问题类型经验库
│   ├── external-tools.md     # history/trace 两套外部工具使用说明
│   ├── harness-principles.md # skill harness 分层、数据集和扩展准则
│   ├── source-checkout.md    # 必要时拉取 relay-claw/jiuwenclaw 源码的方式
│   └── report-template.md    # 标准报告、极简解释、研发复盘模板
├── cases/
│   └── log_case_repo.md      # 单文件历史问题案例库，团队按模板追加
├── scripts/
│   └── office_claw_log_collector_4.2.bat # OfficeClaw 日志采集工具
└── evals/
    └── evals.json            # 典型测试 prompts
```

使用策略：

- 快速定界：先读本 `SKILL.md`，并在开始分析时轻量检索 `cases/log_case_repo.md`，把历史案例当成候选方向；必要时再读 `references/fingerprints.md`。
- 架构不清：读 `references/architecture.md`。
- 需要理解 skill 如何扩展、如何沉淀问题库、如何保持入口简洁：读 `references/harness-principles.md`。
- 用户要正式报告：读 `references/report-template.md`。
- 需要使用 history/trace 外部工具：读 `references/external-tools.md`。
- 日志指纹不足、需要代码级修复或确认前端文案映射：读 `references/source-checkout.md`，优先使用用户提供的源码路径或已发布版本对应源码；不要默认依赖远端最新代码。
- 用户给的是采集包目录：先读 `references/log-bundle-structure.md`，按采集包结构分析。
- 用户描述的是效果、性能、路径、中断、多 agent 等测试标题：读 `references/issue-taxonomy.md` 分流。
- 历史案例库：开始分析时先读 `cases/log_case_repo.md` 的分类、案例标题和关键日志特征，做“候选召回”；完成当前日志核验后再决定能否复用案例结论。新增案例时直接复制文件末尾模板追加。

---

## 日志采集包优先

目前 OfficeClaw 已有成熟日志采集工具：

```powershell
"<skill-dir>\scripts\office_claw_log_collector_4.2.bat"
```

大部分问题定位场景中，用户已经通过该工具采集好了日志。此时不要再要求用户找原始日志，直接分析采集包目录。典型结构：

```text
office_claw_log_YYYYMMDD_HHMMSS/
├── MANIFEST.txt
├── runtime_logs/
├── install_logs/
└── sessions/
```

分析入口按问题边界逐层读取，避免只看单一日志就下结论：

1. 先读 `MANIFEST.txt` 确认采集版本、时间范围、安装目录、数据目录。
2. 再看 `install_logs/api*.log` 和 `desktop-launcher.log` 做上层定界，确认 invocation、MaaS session、sidecar/进程生命周期和 `AbortError`。
3. 若请求已经进入 jiuwenclaw，同时读取：
   - `runtime_logs/openjiuwen/run/jiuwen.log`：执行流日志，重点看 ReAct 迭代、工具调用、SessionMemory 更新、LLM request/response。
   - `runtime_logs/full.log`：LLM trace / E2A 视角，重点看 `[LLM_IO_TRACE]`、`LLM_CALL_START/END`、invoke input/output、keepalive。
   - `runtime_logs/openjiuwen/llm.log`：LLM 调用结构化信息，重点看模型名、流式/非流式、耗时和错误。
4. 最后根据问题类型决定是否分析 `sessions/<session_id>/history.json`、`gateway.log` 和更细的 `[LLM_IO_TRACE]`。

关键原则：`jiuwen.log` 与 `full.log` 视角不同，不能互相替代。`full.log` 更适合确认 LLM trace / E2A 层事件，`jiuwen.log` 更适合补足 ReAct、工具、SessionMemory 等执行流。两者不一致时，先把差异写成待解释现象，再找同一时间窗口、同一 request/session 的其他日志交叉验证。

如果用户还没有采集包，再让用户运行 bat，并建议采集覆盖问题发生前后的完整窗口。采集包结构和时间窗口建议见 `references/log-bundle-structure.md`。

---

## 0. 首要职责：先定界，再深入

这个 skill 的首要职责不是解释所有日志，而是用最少证据先定位责任边界：

1. 这次问题是不是 jiuwenclaw 的问题？
2. 如果不是 jiuwenclaw，是上层哪一块：前端、scheduler、API、MaaS/session、agent service、fallback、outbound delivery、安装环境？
3. 如果是 jiuwenclaw，是哪一块：E2A/WebSocket、agent runtime、prompt/context、LLM invoke、工具调用、session memory、LLM trace、文件系统？
4. 证据是否足够支撑 `confirmed`，还是只能给 `probable` 或 `insufficient`？

边界判定的最低链路检查：

```text
Frontend/Scheduler
  -> API Created invocation?
  -> API jiuwen request sent?
  -> jiuwenclaw [E2A][in]?
  -> jiuwenclaw LLM/tool loop?
  -> jiuwenclaw complete/error chunk?
  -> API state update / fallback?
  -> outbound delivery / frontend display?
```

输出结论时，第一句话必须体现边界：

> 问题边界在 `<layer>/<module>`，这是/不是 jiuwenclaw 问题。直接原因是 `<evidence-backed cause>`。

历史案例匹配要前置，但只能用于缩短搜索路径，不能替代本次日志证据。先用案例库提出候选边界和关键词，再用本次时间窗口、关联 ID 和入口/出口日志确认；确认后再进入 history/trace、源码定位或修复建议。

---

## 0.1 决策：代码语义必须支撑结论，但不默认拉远端最新代码

默认采用“固化架构、关键日志指纹、历史案例 + 必要时轻量源码查证”的方式，而不是每次都先拉远端最新代码。

原因：

- 日志排障的高频问题大多能通过固定日志指纹判断边界，例如 `Huawei MaaS session not found`、`invokeSingleCat crashed before fallback error emission`、`LLM invoke 失败`、`[E2A][in]`、`[LLM_IO_TRACE]`。
- 但如果不理解日志对应的代码模块、状态机或错误处理路径，不能给 `confirmed` 结论，只能给 `probable`，并说明需要源码或架构确认。
- 每次先拉远端最新源码会增加 token 和耗时，也容易让分析偏离日志事实。
- 诊断对象通常是已经发布的版本，远端仓库默认分支可能已经领先。未对齐版本的源码只能辅助理解模块位置，不能作为根因证据。
- 用户通常需要的是“这次失败到底是哪一层的问题”，不一定需要代码级根因。

代码参与分三层：

1. **固化代码语义**：优先使用本技能的架构图、日志地图、指纹库和历史案例。这些内容来自已知代码与真实问题沉淀，适合快速判断常见边界。
2. **轻量源码查证**：当日志模块含义不清、存在两个可竞争边界、或需要解释错误如何从下游映射到前端时，先读取用户提供源码或对应版本源码中的相关函数/模块，确认语义后再下结论。
3. **深度源码定位**：当用户需要修复点、补丁、PR，或日志显示 fallback/状态机/投递链路本身异常时，再系统阅读相关源码并给代码路径。

源码使用分三档：

1. 用户在 query 中提供了本地源码路径、仓库 URL、分支、tag 或 commit：先记录并验证路径/版本，后续需要代码时优先使用它。
2. 用户没有提供源码，但日志或 `MANIFEST.txt` 能识别发布版本、tag、commit 或安装包版本：需要代码级定位时，先请求用户提供对应版本源码；如果用户无法提供，再自行拉取远端并尽量 checkout 到对应 tag/commit。
3. 用户没有提供源码，日志也无法识别发布版本：不要在入口阶段拉源码。只有当日志和历史案例都不足以解释、或用户明确要代码修复点时，再拉取官方仓库默认分支，并在结论中标注“源码版本未与发布版本确认一致”。

以下场景必须阅读架构参考或源码后，才能给 `confirmed` 结论：

- 日志中出现未知模块、未知错误码或未知链路名。
- 当前现象同时符合两个以上候选案例，且边界分别指向上层和 jiuwenclaw。
- 结论依赖“某个错误会不会 fallback、是否会映射到前端文案、状态机会不会覆盖主任务结果”。
- 需要判断 `Created invocation`、`jiuwen request sent`、`Agent service emitted error message`、`complete/error chunk` 等日志在当前版本中的精确含义。
- 同一错误指纹在不同版本中的含义可能变化。
- 用户明确要求给出代码修复点、补丁或 PR。
- 日志显示“错误处理/降级逻辑本身异常”，需要定位源码中的异常捕获和 fallback。
- 需要确认前端文案如何由后端错误映射而来。

源码拉取规则见 `references/source-checkout.md`。默认仓库：

- `relay-claw`：`https://gitcode.com/openJiuwen/relay-claw.git`，默认 `main` 分支。
- `jiuwenclaw`：`https://gitcode.com/openJiuwen/jiuwenclaw.git`，默认 `enterprise_dev` 分支。
- `jiuwenclaw` 可放在 `relay-claw/vendor/jiuwenclaw`，也可使用用户提供的独立路径；最终报告中说明实际读取的路径和版本。

---

## 1. 启动时先问用户要什么

如果用户没有一次性给全路径，先用简短问题收集这些信息：

1. 问题现象：
   - 前端显示的完整文案是什么？
   - 大概发生时间，最好给本地时间和时区。
   - 是手动对话、定时任务、系统心跳、外部渠道触发，还是工具调用？

2. 日志路径：
   - jiuwenclaw 主日志：通常是 `runtime_logs/full.log`，或 `~/.office-claw/.jiuwenclaw/service_default/.logs/full.log`
   - jiuwenclaw 分模块日志：如 `runtime_logs/openjiuwen/run/jiuwen.log`、`runtime_logs/openjiuwen/llm.log`
   - 上层 API 日志：通常是 `install_logs/api.YYYY-MM-DD.N.log`
   - 桌面启动器/前端聚合日志：通常是 `install_logs/desktop-launcher.log`
   - 会话历史：`sessions/<session_id>/history.json`
   - 如有 LLM debug trace：包含 `[LLM_IO_TRACE]` 的日志文件

3. 用户期望：
   - 只要定界结论？
   - 要完整时间线？
   - 要修复策略？
   - 要给研发/测试复盘材料？

如果用户已经给了日志目录，优先自动在目录下找这些文件：

- `MANIFEST.txt`
- `runtime_logs/full.log`
- `runtime_logs/openjiuwen/run/jiuwen.log`
- `runtime_logs/openjiuwen/llm.log`
- `install_logs/api*.log`
- `install_logs/desktop-launcher.log`
- `sessions/**/history.json`

如果存在 `MANIFEST.txt` 且目录形如 `office_claw_log_*`，按 4.2 采集包处理：先确认采集时间范围，再进入分析。不要要求用户重复提供原始 runtime/install/session 路径。

---

## 1.1 单文件历史案例库的使用与回写

问题库位于 `cases/log_case_repo.md`，用于沉淀团队已经定位过的问题。它的作用是提高复用效率，不是替代本次日志分析。为了方便团队成员维护，案例库采用单文件追加模式，不维护多文件 schema、index 或分类子目录。

### 什么时候读问题库

- 开始分析时就读，先用用户现象、前端文案、时间场景和明显错误词做候选召回。
- 用户描述与历史问题相似时优先读，例如“唤醒后首次定时任务失败”“模型调用异常重试后终止”“没收到投递消息”。
- 已完成最小证据核验后再读一次候选案例，用本次时间窗口、关联 ID 和入口/出口日志确认是否能复用结论。
- 需要判断一个日志片段是不是已知问题，或需要给研发复盘材料。

### 怎么读

1. 先读 `cases/log_case_repo.md` 的目录分类和案例标题，按用户现象、关键字、前端文案和时间场景找候选案例。
2. 抽取候选案例的「关键日志特征」作为优先搜索词，并把候选边界标为 `candidate`。
3. 用本次日志的时间窗口、关联 ID、入口/出口日志确认是否能类比。
4. 再对照候选案例的「定界」「补充说明」，确认复用条件和排除条件。
5. 如果不满足复用条件，不能套用案例结论，只能把它作为排查方向。

### 什么时候回写

当一次问题已经形成可复用结论时，直接在 `cases/log_case_repo.md` 中追加案例，使用文件末尾「新增案例模板」：

- 按最贴近的问题类型放到对应分类下，没有合适分类时先新增分类。
- 案例编号按全文件递增，避免多人维护时重复编号。
- 「定界」必须先写是不是 jiuwenclaw 问题，再写归属模块。
- 「关键日志特征」必须包含可搜索关键字和必要关联 ID。
- 证据不足时写清楚缺少什么日志，不要把猜测沉淀成确认案例。

如果新增的是通用日志指纹，再同步更新 `references/fingerprints.md`；如果新增的是测试分流经验，再同步更新 `references/issue-taxonomy.md`。

---

## 2. 先统一时间：UTC 与本地时间

OfficeClaw/RelayClaw/JiuWenClaw 常见时间格式不同，必须先对齐：

- `full.log` 常见为本地时间，例如：`2026-05-08 10:05:42.929`
- `api.*.log` JSON 字段 `time` 常见为 UTC，例如：`2026-05-08T02:05:32.044Z`
- `desktop-launcher.log` 外层可能有本地/UTC包装，例如：`2026-05-08 10:05:32Z [start] {"time":"2026-05-08T02:05:32.044Z"...}`

分析时要明确说明：

- `2026-05-08T02:05:32Z` = 北京时间 `2026-05-08 10:05:32`
- 不要把 API 日志中的 UTC `02:05` 误解为本地凌晨 `02:05`
- 对齐时间后再判断“同一次触发”

建议建立时间线表：

| 本地时间 | 日志文件 | request_id / invocationId / session_id | 事件 | 初步含义 |
|---|---|---|---|---|

---

## 3. 先抓三类关联 ID

定位问题时，先找 ID，再串链路。

### 3.1 invocationId

上层 OfficeClaw/RelayClaw API 的一次调用 ID。常见日志：

- `Created invocation`
- `invokeSingleCat crashed before fallback error emission`
- `executeInBackground`
- `Invocation ... completed`

用途：

- 判断一次前端/调度触发是否在 API 层创建了调用。
- 判断是否在进入 jiuwenclaw 前就失败。
- 判断错误是否被 fallback 包装后返回。

### 3.2 request_id

RelayClaw 与 jiuwenclaw/E2A 通信的请求 ID。常见日志：

- API：`jiuwen request sent`
- jiuwenclaw：`[AgentWebSocketServer] inbound raw payload`
- jiuwenclaw：`[E2A][in] request_id=...`
- jiuwenclaw：`[E2A][wire][out] chunk request_id=...`
- trace：`[LLM_IO_TRACE] ... request_id='...'`

用途：

- 判断请求是否进入 jiuwenclaw。
- 串联 LLM 请求、工具调用和最终响应。

### 3.3 session_id

用户会话 ID。常见格式：

- `officeclaw_<hex>`
- `session_id='officeclaw_...'`
- `cliSessionId`
- `sessions/<session_id>/history.json`

用途：

- 使用 history/trace 工具前必须先确定 session_id。
- 判断当前错误是否发生在主会话、session_memory_update、subagent 或 fork agent。

注意：同一个 `request_id` 或 `session_id` 可能出现在补跑、记忆更新、重连后继续处理等场景。不要只看一条日志就下结论，要结合 `invocationId`、时间、调用类型一起判断。

---

## 4. 架构速记

### 4.1 上层 RelayClaw / OfficeClaw

负责：

- 前端/桌面启动器
- Fastify/Node API
- 用户登录态、Huawei MaaS / ModelArts 会话
- 调度器 scheduled task
- invocation 创建与状态管理
- 调用 jiuwenclaw agent service
- outbound delivery / channel binding
- 错误包装和前端失败文案

常见日志文件：

- `install_logs/api.YYYY-MM-DD.N.log`
- `install_logs/desktop-launcher.log`

关键模块/日志：

- `module":"invoke"`
- `module":"relayclaw-agent"`
- `module":"relayclaw-connection"`
- `[scheduler] <task_id>: tick completed`
- `Created invocation`
- `Session init: binding session`
- `jiuwen request sent`
- `Agent service emitted error message`
- `invokeSingleCat crashed before fallback error emission`
- `OutboundDeliveryHook`
- `No bindings found for thread`

### 4.2 jiuwenclaw

负责：

- agent runtime
- E2A WebSocket 接入
- prompt/system prompt 构造
- ReAct 循环
- LLM 调用
- 工具注册与调用
- session memory / notes update
- LLM_IO_TRACE debug 记录
- 向上层流式输出 chunk / complete / error

常见日志文件：

- `runtime_logs/openjiuwen/run/jiuwen.log`：执行流日志，常用于确认 ReAct 迭代、SessionMemory、工具调用、LLM 请求/响应是否推进。
- `runtime_logs/full.log`：LLM IO trace / E2A 视角，常用于确认 invoke_request/invoke_output、LLM_CALL_START/END、keepalive 等事件。
- `runtime_logs/openjiuwen/llm.log`：LLM 调用结构化日志。

关键模块/日志：

- `jiuwenclaw.agentserver.agent_ws_server`
- `[AgentWebSocketServer] inbound raw payload`
- `[E2A][in]`
- `[E2A][wire][out] chunk`
- `Creating officeclaw agent`
- `JiuWenClawDeepAdapter`
- `Before create openai client`
- `LLM invoke 失败`
- `retry notification sent successfully`
- `[LLM_IO_TRACE] event=invoke_request`
- `[LLM_IO_TRACE] event=invoke_output`
- `Executing tool:`
- `ReAct iteration`
- `[LLM] >>> request`（jiuwen.log）
- `[LLM] <<< response`（jiuwen.log，用于交叉确认 LLM 是否返回）
- `[SessionMemory] update complete`（jiuwen.log）
- `[SessionMemory] update_background finalize`（jiuwen.log，用于判断记忆更新阶段是否结束）
- `SpawnAgent` / `subagent_`（jiuwen.log，子代理生命周期）

---

## 5. 定界矩阵：先判断是不是 jiuwenclaw 问题

使用这个矩阵时先做二分：请求是否稳定进入 jiuwenclaw 主执行链路。如果没有，默认先查上层；如果已经进入，再查 jiuwenclaw 内部模块。不要因为用户前端看到“模型调用异常”就直接判定为 jiuwenclaw，也不要因为 `full.log` 没有错误就断言 jiuwenclaw 漏日志。

### 5.1 上层 RelayClaw / OfficeClaw API 问题

典型证据：

- API 日志有 `Created invocation`，但没有对应 `jiuwen request sent`
- API 日志出现：
  - `Huawei MaaS session not found`
  - `invokeSingleCat crashed before fallback error emission`
  - `Session not found`
  - `Unauthorized`
  - 登录态、token、MaaS、ModelArts、permission 相关错误
- jiuwenclaw `full.log` 中没有同一时间、同一 request_id 的完整进入日志
- 前端只收到笼统失败：`这次处理没有顺利完成`

结论模板：

> 这次失败边界在上层 RelayClaw/OfficeClaw API。调度/前端触发后，API 创建了 invocation，但在进入或完整调用 jiuwenclaw 前因 `<错误>` 失败。jiuwenclaw 日志没有对应完整错误是合理的，因为请求没有稳定进入 jiuwenclaw 主执行链路。

修复方向：

- 检查 Huawei MaaS / ModelArts 登录态是否在休眠或隔夜后失效。
- API 层在定时任务触发前刷新或校验 MaaS session。
- 对 `session not found` 做明确错误映射，不要只给前端笼统失败。
- 修复 `crashed before fallback error emission`，保证异常能走统一 fallback 和用户可见错误。

### 5.2 jiuwenclaw LLM 调用问题

典型证据：

- `full.log` 中已经有同一 request_id/session_id 的进入日志。
- 出现：
  - `Before create openai client`
  - `LLM invoke 失败`
  - `model call failed`
  - `openAI API async invoke error`
  - `openAI API async stream error`
  - `APIConnectionError`
  - `Connection error`
  - `retry notification sent successfully, (1/3)...`
- API 日志可能出现：
  - `Agent service emitted error message`

结论模板：

> 请求已经进入 jiuwenclaw，失败发生在 jiuwenclaw 调模型阶段。直接原因是 `<模型错误>`，系统按配置进行了 `<N>` 次重试，最终仍失败。上层前端看到的失败是 jiuwenclaw 错误透传或包装后的结果。

修复方向：

- 检查模型网关、网络、代理、DNS、证书、API key、模型服务可用性。
- 区分 invoke error 和 stream error。
- 检查 `max_retries`、timeout、重试间隔是否合理。
- 对休眠/网络恢复场景增加连接重建或首包健康检查。

### 5.3 调度器问题

典型证据：

- 没有 `[scheduler] <task_id>: tick completed`
- next trigger 计算异常
- 同一任务重复触发或缺失触发
- 系统休眠后补偿触发行为异常

结论模板：

> 问题边界在调度器。没有看到任务 tick 或 tick 时间与预期不一致，因此下游 agent 失败不是首要问题。

修复方向：

- 明确休眠期间是否补触发。
- 检查 interval 任务在进程暂停/恢复后的 next trigger 计算。
- 检查任务持久化与恢复加载。

### 5.4 Outbound / 渠道投递问题

典型证据：

- agent 生成成功，但出现：
  - `OutboundDeliveryHook deliver() called`
  - `No bindings found for thread — skipping outbound delivery`
- 前端/外部渠道没有收到消息，但日志中有最终 content。

结论模板：

> agent 处理本身已经完成，失败边界在结果投递/渠道绑定。不是模型或 jiuwenclaw 执行失败。

修复方向：

- 检查 thread binding、channel binding、outbound hook 配置。
- 明确当前触发来源是否允许异步投递。

### 5.5 session memory / notes 更新问题

典型证据：

- session_id 类似 `session_memory_update_...`
- 请求内容是更新 `session_context.pending.md`
- 工具调用为 `edit_file`
- 与用户主任务时间相近，但不是主任务本身。

结论模板：

> 这条错误发生在会话记忆更新链路，不等同于用户主任务失败。需要用 invocationId/request_id 区分主任务和后台记忆更新。

修复方向：

- 如果只影响记忆更新，降低用户可见失败等级。
- 后台 notes 更新失败不要覆盖主任务成功状态。

### 5.6 sidecar / 进程生命周期问题

典型证据：

- API 或 launcher 日志在失败时间点附近出现：
  - `sidecar stop`
  - `sidecar start`
  - `willKill`
  - `runtime_signature_changed`
  - `process exit`
  - `process spawn`
  - `SIGTERM`
  - `SIGKILL`
  - `sidecarPid`
  - `AbortError`
- `invokeSingleCat crashed before fallback error emission` 的异常类型为 `AbortError`。
- jiuwenclaw 侧缺少同一请求的自然完成/error chunk，或进程生命周期事件先于/贴近 invocation 崩溃。

结论模板：

> 问题边界在上层 sidecar/进程生命周期管理。`invokeSingleCat` 的 `AbortError` 不是根因本身，而是进程停止、重启、runtime 签名变化或 kill 触发后的表现。需要用失败点前后 30 秒内的 sidecar/process 日志确认触发者。

修复方向：

- 查为什么触发 `runtime_signature_changed`、`willKill` 或 sidecar stop。
- 区分用户取消、超时、版本/runtime 变更、进程守护策略、崩溃退出。
- API 层对 sidecar 被杀导致的 AbortError 做明确错误分类，避免只显示笼统 invocation 崩溃。

---

## 6. 常见错误指纹与含义

### `Huawei MaaS session not found`

边界：上层 RelayClaw/OfficeClaw API。

含义：

- 华为 MaaS / ModelArts 会话不存在、过期、丢失或未恢复。
- 常见于隔夜、休眠、重新打开电脑、登录态丢失后首次触发。

配套指纹：

- `invokeSingleCat crashed before fallback error emission`
- 前端显示：`这次处理没有顺利完成`
- jiuwenclaw `full.log` 可能没有对应完整错误。

结论：

> 不是 jiuwenclaw 模型执行失败，而是上层 API 在调用 agent 前或调用过程中因 MaaS 会话缺失失败。

### `invokeSingleCat crashed before fallback error emission`

边界：上层 API 错误处理。

含义：

- 单次 agent 调用在 fallback 错误消息生成前崩溃。
- 前端可能只能看到笼统失败，而不是具体原因。
- 如果异常类型是 `AbortError`，不能把 `invokeSingleCat crashed` 当作最终根因；必须追问是谁触发了 abort。

建议：

- 在崩溃时间点前后 30 秒内追加搜索 sidecar/进程生命周期事件：`sidecar stop`、`sidecar start`、`willKill`、`runtime_signature_changed`、`process exit`、`process spawn`、`SIGTERM`、`SIGKILL`、`sidecarPid`。
- API 层捕获该类异常并稳定发出结构化错误。
- 把 MaaS/session 类错误映射为“登录态已失效，请重新登录/刷新”。
- 把 sidecar/process kill 类 AbortError 映射为进程生命周期/运行时重启问题，不要只归为 invocation 崩溃。

### `LLM invoke 失败 [连接错误]`

边界：jiuwenclaw LLM 调用链路。

含义：

- jiuwenclaw 已经开始调用模型。
- 模型网关/网络出现连接错误。
- 需要看是否有 1/3、2/3、3/3 重试。

常见原文：

- `model call failed, reason: openAI API async invoke error: Connection error`
- `openAI API async stream error: APIConnectionError`
- `retry notification sent successfully`

### `Agent service emitted error message`

边界：上层收到下游 agent service 错误。

含义：

- 请求通常已经进入 jiuwenclaw。
- 上层把 jiuwenclaw 报错记录为 agent service error。

需要结合 full.log 判断根因。

### `No bindings found for thread — skipping outbound delivery`

边界：渠道投递。

含义：

- 已经准备投递，但该 thread 没有 outbound binding。
- 如果用户抱怨“没收到消息”，这是重点；如果用户抱怨“处理失败”，这通常不是根因，除非业务依赖投递确认。

### `[LLM_IO_TRACE] event=invoke_request / invoke_output`

边界：jiuwenclaw LLM trace。

含义：

- DEBUG 级别下记录了 LLM 输入/输出。
- 可以还原 messages、tools、tool_calls、reasoning、耗时。

用途：

- 找上下文污染、系统提示异常、工具列表异常、模型返回异常、工具调用耗时。

---

## 7. 标准分析流程

### Step 0: 历史案例候选召回

在正式串日志前，先轻量读取 `cases/log_case_repo.md`：

1. 用用户现象、前端文案、时间场景和明显错误词匹配分类与案例标题。
2. 提取 1-3 个候选案例的「关键日志特征」作为优先搜索词。
3. 把候选案例标为 `candidate`，不要直接写成结论。
4. 若没有命中案例，继续按标准链路分析，不要为了匹配案例而扩大解释。

案例命中后的验证条件：

- 本次日志中能在同一时间窗口找到案例要求的核心关键字。
- 能用 `invocationId`、`request_id`、`session_id` 或 `task_id` 串起同一次触发。
- 能解释为什么不是相似但边界不同的案例。

### Step 1: 收集与校验文件

确认路径存在，并列出找到的日志文件。不要一开始全文读取超大日志。

如果用户给的是 4.2 采集包目录，优先读 `MANIFEST.txt` 和目录结构；如果还没有采集包，再建议运行 `scripts/office_claw_log_collector_4.2.bat`。

同时扫描用户 query 中是否给了源码信息：

- 本地路径：如 `relay-claw`、`jiuwenclaw`、`vendor/jiuwenclaw` 所在目录。
- 仓库 URL、分支、tag、commit、发布版本号。
- 如果没有源码信息，不要阻塞日志分析；只在后续确实需要代码级定位时再询问或拉取。

优先使用搜索：

- 用户给出的错误文案
- 用户给出的时间点
- `session_id`
- `request_id`
- `invocationId`
- `Created invocation`
- `jiuwen request sent`
- `Agent service emitted error message`
- `LLM invoke 失败`
- `Huawei MaaS session not found`
- `sidecar stop`
- `sidecar start`
- `willKill`
- `runtime_signature_changed`
- `process exit`
- `process spawn`
- `SIGTERM`
- `SIGKILL`
- `AbortError`
- `sidecarPid`
- `这次处理没有顺利完成`
- `模型调用异常`
- `[LLM_IO_TRACE]`
- `ReAct iteration`（jiuwen.log）
- `[LLM] <<< response`（jiuwen.log）
- `[SessionMemory] update_background finalize`（jiuwen.log）
- `Executing tool:`（jiuwen.log）
- `SpawnAgent`（jiuwen.log）

### Step 2: 对齐时间窗口

围绕用户给出的时间点取窗口：

- 先看前后 5 分钟。
- 如果涉及休眠/隔夜，扩大到前后 10-30 分钟。
- 如果 API 使用 UTC，换算成本地时间。

在同一时间窗口内，API 日志搜索不能只覆盖 invocation 生命周期；还必须覆盖 sidecar/进程生命周期事件。尤其是发现 `AbortError`、请求中断、jiuwenclaw 侧没有自然完成时，窗口至少取失败点前后 30 秒。

### Step 3: 找上层 invocation

在 `api*.log` 和 `desktop-launcher.log` 中找：

- `Created invocation`
- `invocationId`
- `catId`
- `threadId`
- `cliSessionId`
- `jiuwen request sent`
- `error`
- `msg`
- `AbortError`

同时在同一批 `api*.log` 和 `desktop-launcher.log` 中找 sidecar/进程生命周期：

- `sidecar stop`
- `sidecar start`
- `willKill`
- `runtime_signature_changed`
- `process exit`
- `process spawn`
- `SIGTERM`
- `SIGKILL`
- `sidecarPid`

判断：

- 是否创建 invocation？
- 是否绑定 session？
- 是否发出 jiuwen request？
- 是否在发出前失败？
- 如果出现 `AbortError`，前后 30 秒内是否有 sidecar stop、process kill、runtime_signature_changed 或进程退出？
- invocation 崩溃是根因，还是 sidecar/进程生命周期事件造成的结果？

### Step 4: 找 jiuwenclaw 入口并补足执行流

此步骤同时看 `full.log` 和 `jiuwen.log`：前者提供 trace / E2A 视角，后者补足 ReAct、工具调用、SessionMemory 等执行流。

在 `full.log` 找同一 request_id/session_id：

- `[AgentWebSocketServer] inbound raw payload`
- `[E2A][in]`
- `收到请求`
- `Creating officeclaw agent`
- `Before create openai client`
- `[LLM_IO_TRACE]`

在 `jiuwen.log`（`runtime_logs/openjiuwen/run/jiuwen.log`）找同一时间窗口的执行流事件：

- `ReAct iteration N/M` — 迭代进度，判断执行是否停滞
- `[LLM] >>> request` — LLM 请求发出
- `[LLM] <<< response` — LLM 响应返回；当 `full.log` 缺少 invoke_output 时，可用于交叉验证 LLM 是否已经返回。
- `Executing tool: <tool_name>` — 工具调用
- `[SessionMemory] update complete` — SessionMemory 更新完成
- `[SessionMemory] update_background finalize` — SessionMemory 后台任务结束
- `[SessionMemory] update_runtime` — SessionMemory 运行时状态更新
- `SpawnAgent` / `subagent_` — 子代理创建和执行
- `LLM invoke 失败` / `不可重试` — LLM 调用错误

判断：

- 请求是否进入 jiuwenclaw？
- 是否进入 LLM 调用？
- LLM 调用是否返回？不要只凭 `full.log` 是否有 invoke_output 下结论。
- 是否产生 tool call？
- ReAct 迭代是否继续推进？
- 是否发出 complete/error chunk？

### 交叉验证规则（防止 full.log 误导）

**规则 1：full.log 无 invoke_output ≠ LLM 调用挂死**

如果 `full.log` 中有 `LLM_CALL_START` 但无 `invoke_output` / `LLM_CALL_END`，不要直接判定“LLM 调用挂死”。先到 `jiuwen.log` 检查同一时间窗口是否有 `[LLM] <<< response`、后续 tool call 或新的 ReAct 迭代。如果 `jiuwen.log` 显示 LLM 已返回，则需要解释为什么 trace 视角缺失输出，而不是把它写成模型调用未返回。

**规则 2：keepalive 持续 ≠ LLM 流式挂死**

如果 `full.log` 只有 keepalive chunk，不能直接判定“SSE 流挂死”。先到 `jiuwen.log` 检查：
- 最后一个 `[LLM] <<< response` 的时间
- 最后一个 `ReAct iteration` 的时间
- 最后一个 `Executing tool` 的时间
- 如果 LLM 已返回但 ReAct 不再推进，把“执行循环停滞”列为候选边界，并继续找 complete/error chunk、进程生命周期和上层 abort 证据。

**规则 3：SessionMemory update_background finalize 后无新事件只是候选信号**

如果 `jiuwen.log` 中最后的事件是 `[SessionMemory] update_background finalize` 或 `[SessionMemory] update_runtime`，且之后长时间没有新的 `ReAct iteration`、`[LLM] >>> request`、`Executing tool`、complete/error chunk，只能先写为“疑似执行流在 SessionMemory 相关阶段之后停滞”。除非还有代码级证据或多个日志层面的闭环证据，不要直接写成“update_background 导致 ReAct 主循环卡死”的已确认根因。

### Step 5: 确定 session_id

确定 session_id 后，再决定是否使用外部工具：

- 如果用户问题是“这次任务做了什么、做到哪一步、为什么停”，使用 history.json 分析工具。
- 如果用户问题是“工具调用前一直没反应、模型生成工具调用很慢、工具调用卡住”，优先用 `scripts/analyze_history_tool_call_speed.py` 分离 `chat.tool_calls.delta` 流式返回耗时和工具执行耗时。
- 如果用户问题是“模型上下文、工具调用、耗时、输入输出是什么”，使用 LLM trace 分析工具。

### Step 6: 形成定界结论

结论必须先回答：

1. 是不是 jiuwenclaw 问题？
2. 如果是上层，具体是 API、调度器、登录态/MaaS、投递、前端文案中的哪一块？
3. 如果是 jiuwenclaw，具体是 WebSocket/E2A、prompt 构造、LLM、工具调用、session memory、trace、文件系统中的哪一块？
4. 证据是哪几条日志？
5. 能否匹配 `cases/` 中已定位问题；如果能，匹配哪个案例和哪些复用条件？
6. 如果读取了源码，源码版本是否与问题发生的发布版本一致；如果不一致，只能作为辅助说明。

### Step 7: 给修正策略

按层给建议：

- 立即规避：重新登录、刷新 token、重启非核心服务、重试任务、补发任务。
- 产品改进：错误文案映射、异常分类、状态恢复。
- 研发修复：代码路径、异常捕获、fallback、重试、健康检查。
- 测试补充：休眠恢复、token 过期、断网恢复、定时任务补触发、无 channel binding。

---

## 8. 何时使用两套外部分析工具

这两套工具不只是异常定位工具，更是还原 Agent 执行流的核心工具。对于任何能对应到一次具体 Agent 执行的问题，默认尽早尝试使用 history/trace 还原执行流程；只有问题明确发生在 Agent 启动之前，才先不跑外部工具。

原则：

- 先做“最小定界”：确认采集包时间范围、问题时间、候选 `session_id` / `request_id` / `invocationId`，避免跑错 session。
- 只要 Agent 曾经开始处理任务，就优先用 session history 还原“这次任务怎么走的”。
- 如果性能问题集中在工具调用前后，先用 history 工具调用速度脚本看 `chat.tool_calls.delta` 的首包、尾包、片段间隔和工具执行边界。
- 如果存在 `[LLM_IO_TRACE]`，再用 LLM trace 还原“模型当时看到了什么、为什么这么调用工具、耗时在哪里”。
- 外部工具适用于功能失败、效果问题、性能问题、路径错误、工具异常、任务中断、多 agent 协作、UI 与实际输出不一致等问题。
- 如果暂时没有 `session_id`，先从 `sessions/` 候选目录、API 的 `cliSessionId`、full.log 的 `session_id` 找目标；仍然不要无目标地批量跑所有 session。
- 如果日志已明确显示问题发生在 Agent 之前（例如安装失败、启动失败、API 登录态/MaaS session 缺失且没有 jiuwen request），先完成上层定界；只有需要确认用户任务上下文时再用 history/trace。

### 8.1 session history 分析工具

用途：

- 解析 `sessions/<session_id>/history.json`
- 还原用户一次任务的完整过程
- 看用户消息、assistant 消息、工具调用、工具结果、上下文压缩
- 适合回答“这次任务到底做到了哪一步”
- 适合路径问题、文件发送问题、中断继续、历史丢失、多 session/多 agent 协作、步骤不一致

参考仓库：

- `https://github.com/gaotong132/tools`

已知入口：

- `ha`
- `ha <json_file> -o report.html`

使用前提：

- 已找到对应 `history.json`
- 最好已确认 session_id 是本次问题的主会话，而不是后台 memory update
- 如果是效果/路径/过程类问题，可以先从采集包 `sessions/` 下候选 session 中按时间和 metadata 筛选目标

输出使用方式：

- 不要把 HTML 全部贴给用户。
- 提炼关键阶段：输入、工具调用、失败点、最后一条有效输出。

### 8.2 history 工具调用速度分析脚本

用途：

- 解析 `sessions/<session_id>/history.json` 中的 `chat.tool_calls.delta`、`chat.tool_call`、`chat.tool_update`、`chat.tool_result`。
- 区分两段耗时：LLM 流式生成工具调用参数的耗时，以及工具真实执行的耗时。
- 适合回答“模型生成工具调用是否慢”“工具调用参数流中是否有明显停顿”“工具开始后是不是工具本身慢”。
- 适合长时间未回复、工具调用卡住、继续执行很慢、subagent/search/fetch 等工具耗时分布不清的场景。

已知入口示例：

```powershell
python "<skill-dir>\scripts\analyze_history_tool_call_speed.py" "<history.json>" --top 10 --gap-threshold-ms 1000
```

如果需要给其他工具二次处理：

```powershell
python "<skill-dir>\scripts\analyze_history_tool_call_speed.py" "<history.json>" --json-out "tool_speed.json"
```

输出使用方式：

- 先看 `Slowest LLM Tool-Call Delta Streams`，判断慢在模型流式吐出工具调用参数。
- 再看 `Slowest Tool Executions`，判断慢在工具真实执行。
- `chat.tool_calls.delta` 是 LLM 返回工具调用 JSON 参数的流式片段，不是工具执行日志；不要把每条 delta 当作一次工具调用。
- `history.json` 的 `timestamp` 是 history 观测时间，不等同于模型网关服务端耗时；如果存在 `[LLM_IO_TRACE]`，再用 trace Timing 交叉验证。
- 报告只用于性能分段和证据补充，不替代 API / jiuwenclaw / 前端边界判断。

### 8.3 LLM debug / trace 分析工具

用途：

- 解析包含 `[LLM_IO_TRACE]` 的 debug 日志
- 还原 LLM 请求体、响应、tool_calls、工具耗时、上下文输入
- 适合回答“模型为什么这么调用工具/为什么上下文不对/耗时在哪里”
- 适合 PPT 效果、图表错乱、长时间未回复、继续执行很慢、工具结果为空、多 agent 看不到上下文等问题

参考仓库：

- `https://gitcode.com/xiaowenzihwl/log_parse`
- 如果本地使用 `gaotong132/tools` 中的 `llm_trace_analyzer`，也可用其 `lt` 命令分析 `full.log`

已知入口示例：

- `lt <log_file_path> -o trace_report`
- `lt --session <session_id>`

使用前提：

- 日志级别开启 DEBUG。
- `full.log` 中存在 `[LLM_IO_TRACE]`。
- 最好已确定目标 session_id；如果无法确定，先用时间窗口和 request_id 缩小范围。

输出使用方式：

- 优先看 Timing Overview、iteration、request body、response、tool_calls。
- 用于解释模型上下文和工具链，不用于替代上层 API 定界。

---

## 9. 输出格式

默认用以下结构回答用户：

```markdown
## 结论
一句话说明问题边界和直接原因。

## 证据链
- 时间 A：日志文件 X，出现 Y，说明 Z。
- 时间 B：日志文件 X，出现 Y，说明 Z。

## 为什么不是另一层
说明为什么不是 jiuwenclaw / 为什么不是上层 / 为什么不是前端。

## 详细时间线
| 本地时间 | 层级 | 日志 | 事件 | 含义 |
|---|---|---|---|---|

## 修正策略
- 立即规避：
- 研发修复：
- 测试补充：

## 还需要补充的日志
如果证据不足，列出最少需要的文件或时间窗口。
```

如果用户说“没看懂”，改用更短的解释：

```markdown
一句话结论：
问题出在 <层级>，不是 <容易误解的层级>。

为什么：
<日志原文> 表示 <含义>。

前端为什么显示那句话：
因为 <调用链失败/错误被兜底包装>。
```

---

## 10. 案例模板：定时任务唤醒后显示“这次处理没有顺利完成”

已知日志指纹：

- API/launcher：
  - `[scheduler] dyn-...: tick completed`
  - `Created invocation`
  - `Huawei MaaS session not found`
  - `invokeSingleCat crashed before fallback error emission`
- jiuwenclaw：
  - 可能只有零散 `Before create openai client`
  - 或出现与主 invocation 不完全对应的 `session_memory_update`
  - 可能看不到同一次业务调用的完整错误栈

正确结论：

> 这次前端显示“这次处理没有顺利完成”，根因在上层 RelayClaw/OfficeClaw API。定时任务触发后，API 在调用 office 智能体时发现 Huawei MaaS session 不存在，`invokeSingleCat` 在 fallback 错误消息生成前崩溃，所以前端只能显示笼统失败。jiuwenclaw full.log 中没有完整错误是合理的，因为这次失败主要发生在上层 API 登录态/MaaS 会话校验阶段。

不要误判为：

- “jiuwenclaw 没日志，所以 jiuwenclaw 卡住了”
- “看到 `Before create openai client`，所以一定是 jiuwenclaw 模型问题”
- “凌晨有 3 次 Connection error，所以早上这次也是同一个模型错误”

需要区分：

- 凌晨模型连接错误：jiuwenclaw LLM 调用链路，`Connection error`，3 次重试。
- 早上首次触发失败：上层 API MaaS session 缺失，`Huawei MaaS session not found`。

---

## 11. 案例模板：模型调用异常，3 次重试后终止

已知日志指纹：

- `LLM invoke 失败 [连接错误]`
- `model call failed`
- `openAI API async invoke error: Connection error`
- `将在 10.0s / 20.0s / 40.0s 后重试`
- `第 1 次重试 / 共 3 次`
- `retry notification sent successfully`

正确结论：

> 请求已经进入 jiuwenclaw，并且失败发生在 jiuwenclaw 调模型阶段。直接原因是模型兼容接口连接失败，系统按配置进行了 3 次重试，最后仍失败。需要排查模型网关、网络、代理、服务端可用性和休眠恢复后的连接状态。

注意：

- 如果 session_id 是 `session_memory_update_...`，说明失败可能发生在后台记忆更新，不一定是主任务。
- 要用主会话的 invocationId/request_id 判断用户前端失败是否由这条错误直接导致。

---

## 12. 常见误区

- 只看 `full.log`，忽略 `jiuwen.log`。`full.log` 偏 LLM trace / E2A 视角，可能不足以解释 ReAct 迭代、SessionMemory 更新、工具执行等执行流。
- 把 `full.log` 无 invoke_output 等同于“LLM 调用挂死”。更稳妥的做法是回到 `jiuwen.log`、`llm.log` 和同一时间窗口的后续事件交叉验证。
- 把 keepalive 持续等同于“SSE 流挂死”。keepalive 只能说明连接或进程仍有活动，不能单独证明模型流、执行循环或上层 invocation 的具体状态。
- 只看 `full.log`，忽略 `api.*.log`。很多前端失败实际在上层 API。
- 忽略 UTC 与本地时间，导致把 10:05 的 API 日志误读为凌晨 02:05。
- 看到同一个 session_id 就认为是同一次业务调用。必须同时看 request_id、invocationId、时间窗口。
- 把 session memory update 的模型错误当成用户主任务失败。
- 把 `No bindings found for thread` 当成模型失败。它更多是投递/渠道绑定问题。
- 在没有 session_id 前运行 history/trace 工具，生成大量无关报告。
- 一上来拉源码。日志指纹足够时，应先给边界结论。

---

## 13. 给研发的修复建议库

### 上层 API / RelayClaw

- 对 Huawei MaaS session 做恢复前置校验：
  - 定时任务触发前检查 session 是否存在。
  - 休眠恢复后刷新或重新建立 session。
  - session 缺失时返回明确错误码。
- 修复 `invokeSingleCat crashed before fallback error emission`：
  - 所有异常进入统一 fallback。
  - fallback 失败也要有兜底错误事件。
  - 日志中记录 invocationId、threadId、catId、sessionId、错误分类。
- 前端文案分级：
  - 登录态失效：提示重新登录/刷新授权。
  - 模型连接失败：提示模型服务或网络异常。
  - 工具失败：提示具体工具。
  - 投递失败：提示消息处理成功但投递失败。
- 定时任务状态：
  - 区分 tick completed、invocation created、agent completed、delivery completed。
  - 不要仅因 scheduler tick completed 就认为业务处理成功。

### jiuwenclaw

- LLM 连接错误增加更明确的错误分类：
  - DNS/timeout/connection reset/APIConnectionError/rate limit/auth。
- 对 session memory update 后台任务降低用户可见失败影响。
- 在 E2A 入口和出口统一记录 request_id/session_id/response_kind/is_final。
- DEBUG trace 支持按 session_id 过滤，避免超大日志难读。
- 工具调用失败时记录 tool name、args 摘要、耗时、错误分类。

### 测试补充

- Windows 休眠后首次定时任务触发。
- 隔夜登录态过期后定时任务触发。
- 模型网关断网恢复后的首次调用。
- API session 存在但 jiuwenclaw WebSocket 断开。
- Agent 成功但 outbound binding 缺失。
- session memory update 失败是否影响主任务状态。

---

## 14. 最小证据标准

不要在证据不足时给过度确定结论。至少需要：

- 用户现象和时间点。
- 上层 API 或 launcher 日志中的同一时间窗口。
- jiuwenclaw full.log 中同一时间窗口。
- 至少一个关联 ID：session_id、request_id、invocationId、task_id。

如果缺少某类日志，明确说：

> 目前能确定 `<已有结论>`，但还不能确认 `<缺失部分>`。需要补充 `<文件>` 中 `<时间窗口>` 的日志。

---

## 14.1 Harness 维护准则

本 skill 按 harness 架构维护，详细原则见 `references/harness-principles.md`。执行和演进时遵守：

- `SKILL.md` 只保留入口流程、定界矩阵和高频判断，不继续堆大量事故细节。
- 稳定知识放 `references/`，真实已定位案例放 `cases/`，确定性操作放 `scripts/`，评估 prompts 放 `evals/`。
- 新案例优先追加到 `cases/log_case_repo.md`；只有变成通用规律后，才提升到 `references/fingerprints.md` 或 `references/issue-taxonomy.md`。
- 版本频繁变化时，`references/` 只沉淀已验证、可复用的架构和日志语义；发布基线切换、关键日志/状态机/fallback/文案映射变化、同一指纹语义变化时再更新。具体规则见 `references/harness-principles.md`。
- 外部工具和源码是深入分析手段，不是入口。先完成最小定界，再决定是否跑 history/trace 或拉源码。
- 所有新增问题沉淀都要保留复用边界：哪些日志条件满足时可套用，哪些相似现象不能套用。

---

## 15. 最终回答风格

- 先给结论，不要先堆日志。
- 用“问题出在 X，不是 Y”帮助用户建立边界。
- 只引用关键日志原文，不贴大段日志。
- 如果用户是测试/产品视角，用通俗解释。
- 如果用户是研发视角，再补代码路径、修复策略和测试建议。
- 对不确定点明确标注“需要补日志确认”。

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
