# 外部分析工具使用说明

两套工具不只是“报错定位工具”，更是还原 Agent 执行流的核心工具。实际排障时，不管是功能问题、效果问题还是性能问题，只要能对应到一次具体 Agent 执行，都应尽早尝试用它们还原完整过程。

原则：

- 先做“最小定界”：确认采集包时间范围、问题时间、候选 `session_id` / `request_id` / `invocationId`。这一步不是为了马上判断根因，而是为了避免跑错 session。
- 只要 Agent 曾经开始处理任务，就默认优先用 history 还原“这次任务怎么走的”。
- 如果存在 `[LLM_IO_TRACE]`，再用 trace 还原“模型看到了什么上下文、为什么这么调用工具、耗时在哪里”。
- 外部工具适用于功能失败、效果问题、性能问题、路径错误、工具异常、任务中断、多 agent 协作、UI 与实际输出不一致等问题。
- 如果暂时没有 `session_id`，先从采集包的 `sessions/` 候选、API 日志的 `cliSessionId`、full.log 的 `session_id` 里找。不要无目标地批量跑所有 session。
- 如果日志已明确显示问题发生在 Agent 之前，例如安装失败、启动失败、API 登录态/MaaS session 缺失且没有 jiuwen request，则先完成上层定界；只有需要确认用户任务上下文时再用 history/trace。

## 1. session history 分析工具

仓库：

- `https://github.com/gaotong132/tools`

核心用途：

- 解析 `sessions/<session_id>/history.json`
- 还原一次用户任务的完整过程
- 查看用户消息、assistant 输出、工具调用、工具结果、上下文压缩
- 判断“任务做到哪一步停了”
- 判断输出路径、文件发送路径、工具参数是否正确
- 比较中断前后历史是否丢失
- 复盘多 agent / 多 session 的消息流

典型命令：

```powershell
ha "<path-to-history.json>" -o "history_report.html" -v
```

如果工具未安装：

```powershell
git clone https://github.com/gaotong132/tools
cd tools
uv venv
uv pip install -e .[dev]
ha "<path-to-history.json>" -o "history_report.html" -v
```

什么时候优先用：

- 任何能对应到一次具体 Agent 执行的问题，需要先看完整执行过程
- 用户问“这次任务过程是什么？”
- PPT 生成提示成功但最终产物异常
- 输出路径错误、发送不存在文件、最终产物不在 output 目录
- 任务中断后继续、历史丢失、步骤总数不一致
- 多 session / 多智能体协作上下文不一致
- 工具调用结果为空，需要确认工具是否真的返回了空
- 日志中已确定 `session_id=officeclaw_...`
- 需要证明某一步工具调用是否执行过
- 需要区分主任务与 session memory update

不适合单独替代：

- API 层定界
- MaaS session/auth 错误分析
- 模型连接错误根因分析

## 2. history 工具调用速度分析脚本

脚本位置：

- `scripts/analyze_history_tool_call_speed.py`

核心用途：

- 解析 `sessions/<session_id>/history.json`。
- 专门统计 `chat.tool_calls.delta` 的片段间隔、首尾耗时、最大停顿和长间隔数量。
- 同时统计 `chat.tool_call` -> `chat.tool_update` -> `chat.tool_result` 的工具启动与执行耗时。
- 把“LLM 正在流式生成工具调用参数”和“工具已经开始真实执行”拆开，避免把 delta 误判成工具执行。

典型命令：

```powershell
python "<skill-dir>\scripts\analyze_history_tool_call_speed.py" "<path-to-history.json>" --top 10 --gap-threshold-ms 1000
```

输出机器可读 JSON：

```powershell
python "<skill-dir>\scripts\analyze_history_tool_call_speed.py" "<path-to-history.json>" --json-out "tool_speed.json"
```

什么时候优先用：

- 用户问“工具调用时 LLM 返回速度是不是慢？”
- 前端长时间停在工具调用前后，但还不清楚慢在模型输出、工具排队还是工具执行。
- history 中能看到 `chat.tool_calls.delta`，需要看片段间隔、最大停顿、p95 间隔。
- 工具调用较多，需要找最慢的 tool call、缺失 `tool_result` 或异常时间线。
- 需要在没有 `[LLM_IO_TRACE]` 的情况下，先用 history 侧观测时间做性能分段。

不要用它替代：

- LLM 服务端真实首 token / 总 latency 结论。history 的 `timestamp` 是观测时间，需要 trace 或网关日志交叉验证。
- 工具为什么被模型选择。这个问题仍然看 LLM trace 的 request/response/tool_calls。
- API 层 invocation、scheduler、MaaS/session 定界。

指标解读：

- `LLM delta`：第一条到最后一条 `chat.tool_calls.delta` 的耗时，表示模型流式吐出工具调用参数的 history 观测时长。
- `Max gap` / `Long gaps`：delta 片段之间的最大停顿和超过阈值的次数，用于发现流式返回抖动。
- `Emit delay`：最后一条 delta 到正式 `chat.tool_call` 的延迟。
- `Start delay`：`chat.tool_call` 到 `chat.tool_update status=in_progress` 的延迟。
- `Tool exec`：`chat.tool_update status=in_progress` 到 `chat.tool_result` 的耗时；没有 update 时退化为 `chat.tool_call` 到 `chat.tool_result`。

## 3. LLM trace 分析工具

仓库：

- `https://gitcode.com/xiaowenzihwl/log_parse`
- 如果本地已使用 `gaotong132/tools`，其中也包含 `llm_trace_analyzer`，可用 `lt` 命令。

核心用途：

- 解析 `full.log` 中的 `[LLM_IO_TRACE]`
- 合并分片请求体
- 还原 LLM messages/tools/response/tool_calls
- 分析 LLM 时间、工具耗时、上下文输入
- 判断模型是否收到了工具结果、前序 agent 回复、历史上下文
- 判断效果问题是模型输出、工具转换，还是 UI/产物链路导致

典型命令：

```powershell
lt "<path-to-full.log>" -o "trace_report" --session "<session_id>" -v
```

如果工具支持自动打开：

```powershell
lt "<path-to-full.log>" -o "trace_report" --session "<session_id>" -O
```

什么时候优先用：

- `full.log` 中存在 `[LLM_IO_TRACE]`
- 需要理解 Agent 某一轮为什么这么做、为什么这么慢、为什么生成效果不对
- 用户问“模型当时看到了什么上下文？”
- 用户问“工具调用耗时在哪里？”
- PPT 效果问题：排版超出页面、字体颜色差、图表错乱、HTML 与 PPT 不一致
- 性能问题：长时间未回复、继续执行重新识别需求耗时过长
- 多智能体问题：后续 Agent 看不到前面 Agent 回复
- 工具调用问题：工具结果为空、工具选择不合理、tool_calls 异常
- 需要分析 prompt、tools、tool_calls、reasoning_content
- 需要确认模型输出是否为空、是否只输出 tool call、是否异常 finish

不要用它替代：

- 上层 API 的 `invocationId` 定界
- 前端文案映射分析
- scheduler 是否触发的判断

## 4. 推荐组合

### 效果问题：PPT、图表、排版、字体、符号

1. 用 `history.json` 确认用户需求、生成路径、工具调用顺序。
2. 用 trace 看模型生成的 HTML/PPT 指令、图表数据、样式约束。
3. 查 workspace/output 中是否存在多份 HTML/PPT 或旧产物复用。
4. 若 HTML 预览 OK 但 PPT 空白/错乱，重点转向转换器、资源路径、打包链路。

### 性能问题：长时间未回复、继续执行很慢

1. 先用 API/connection 查是否有中断、取消、unknown/expired request。
2. 用 history 工具调用速度脚本拆分 `chat.tool_calls.delta` 流式返回耗时和工具执行耗时。
3. 用 trace Timing Overview 交叉验证 LLM 服务端时间、工具时间、总时间。
4. 用 history 看是否重复识别需求、重复规划、上下文丢失。

### 过程问题：中断、继续、多 session、多 agent

1. 从 API 日志提取所有相关 `invocationId`、`requestId`、`cliSessionId`。
2. 用 history 比较中断前后消息是否连续。
3. 用 trace 检查模型实际收到的上下文是否包含历史/前序 agent 回复。

### 定时任务失败

1. 先看 `api*.log` 和 `desktop-launcher.log`，确定 `invocationId`、`task_id`、`session_id`。
2. 再看 `full.log` 是否有同一 `request_id`。
3. 如果请求进入 jiuwenclaw，再用 trace 工具。
4. 如果需要还原用户任务过程，再用 history 工具。

### 模型调用异常

1. 先从 `full.log` 找 `LLM invoke 失败`、`request_id`、`session_id`。
2. 用 trace 工具看最后一次 `invoke_request` 和是否有 `invoke_output`。
3. 如果前端表现和 jiuwenclaw 错误不一致，再回查 API。

### 工具调用卡住或结果不对

1. 用 history 工具还原工具调用过程。
2. 用 history 工具调用速度脚本判断慢在 `chat.tool_calls.delta`、工具启动，还是 `tool_result` 返回。
3. 用 trace 工具看模型为什么选择该工具。
4. 在 `full.log` 中找 `Executing tool:` 和工具错误。

