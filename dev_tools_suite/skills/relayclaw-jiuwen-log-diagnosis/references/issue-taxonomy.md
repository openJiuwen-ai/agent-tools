# 测试问题类型经验库

这份参考把历史测试标题归类为定位策略。标题信息不足时，不要直接断言根因；先按类型决定要看哪些日志、哪些 session、是否需要外部工具。

已确认根因的具体问题不要继续堆在本文件中，应沉淀到 `cases/log_case_repo.md`。本文件只保留“看到某类现象时先查什么”的分流策略。

## 1. PPT 生成与渲染效果类

历史标题：

- `PPT效果：排版超出页面`
- `ppt字体颜色选择不佳，字样模糊；文字超出框`
- `提示ppt生成成功了，预览和html也是OK的，但是打开ppt之后没有内容`
- `html中不存在的图表，ppt上存在`
- `生成的图表内容错乱`
- `横杠符号显示有问题`

定位重点：

- 这是“效果问题”优先，不一定是异常失败。
- 需要确认最终产物链路：用户需求 -> HTML/page 生成 -> 预览 -> PPT 转换 -> 最终 pptx。
- 同时看 `history.json` 和 LLM trace：
  - `history.json` 看用户原始需求、产物路径、工具调用顺序、是否生成成功。
  - trace 看模型给工具的 HTML/PPT 指令、图表数据、样式约束。
- 如果预览 OK 但 pptx 空白，优先怀疑 HTML -> PPT 转换/打包/路径/资源加载，而不是模型内容本身。
- 如果 HTML 不存在图表但 PPT 有，优先查转换器默认模板、缓存、旧产物复用、输出目录污染。

关键日志/文件：

- `sessions/<session_id>/history.json`
- `runtime_logs/full.log` 中 `Executing tool:`、`write_file`、PPT 相关工具
- `[LLM_IO_TRACE]` 中生成 HTML/PPT 的 request/output
- workspace/output 目录路径记录

## 2. 输出路径与产物归档类

历史标题：

- `输出路径在skill目录下`
- `HTML文件在workspace目录下存在两份，最终产物在非output目录`
- `发送了一个不存在的文件`

定位重点：

- 这是文件路径/产物管理问题。
- 需要区分模型选择路径、工具实际写入路径、上层发送文件路径。
- 优先查 history 中工具参数的 `file_path`、`output_dir`、`send_file_to_user` 或类似发送工具。
- 如果路径在 skill 目录下，重点看 system prompt / skill 指令是否错误要求写入技能目录，或模型误把 skill reference 当工作区。
- 如果发送不存在文件，重点看最终产物路径是否被后续步骤移动、清理、覆盖，或路径转义/Windows 盘符问题。

## 3. 会话中断、继续执行与历史丢失类

历史标题：

- `同一会话多次中断后，会话历史丢失`
- `任务中断后继续，未生成ppt就结束了`
- `任务中断未生成ppt，继续执行识别需求花费10分钟`
- `任务运行较长时间后，发送新的对话会中止这些任务。`
- `多session执行中，新开session发送对话，原session中断(回归未通过)`

定位重点：

- 这是 session 生命周期、取消/中断、恢复策略问题。
- 必须确定：
  - 原 session_id
  - 新 session_id
  - request_id 是否被取消
  - invocation 状态是否被新请求覆盖
  - history.json 是否写入完整
  - session memory 是否更新
- 使用 history 工具看过程，用 API 日志看 invocation/cancel，用 trace 看 LLM 是否重新识别需求。
- 如果继续执行花费很久，先用 `scripts/analyze_history_tool_call_speed.py` 拆分 `chat.tool_calls.delta` 流式返回耗时和工具执行耗时，再判断是否缺少历史上下文导致重新规划。

关键日志：

- `Created invocation`
- `chat.final`
- `unknown/expired request`
- `session_memory_update`
- `history.json`
- `metadata.json`

## 4. 多智能体协作与上下文可见性类

历史标题：

- `多智能体协作：成语接龙后面Agent看不到前面Agent的回复？`
- `同时执行多个session，有一个session，没有思考过程`

定位重点：

- 先确定是否为 subagent/fork/multi-agent 模式。
- 看各 agent 的 session_id 命名、父子关系、上下文传递、消息转发。
- trace 工具价值高：检查后续 agent 的 messages 是否包含前序 agent 输出。
- history 工具用于验证 UI 层是否显示了前序回复。

## 5. 安装、技能与工具调用类

历史标题：

- `在线安装技能，有技能安装失败`
- `工具调用结果为空`

定位重点：

- 技能安装失败：查安装工具输出、网络、权限、目标目录、manifest/SKILL.md 格式。
- 工具调用结果为空：区分工具真实空结果、工具异常被吞、模型没有使用结果、上层展示丢失。
- history 工具看 tool_call 和 tool_result；trace 工具看模型是否接收到了 tool_result。
- 工具调用卡住或很慢：用 `scripts/analyze_history_tool_call_speed.py` 看 `chat.tool_calls.delta` 是否慢、正式 `chat.tool_call` 是否延迟、`tool_update` 到 `tool_result` 是否耗时过长。

## 6. 对话行为与回复质量类

历史标题：

- `回复内容不同，但意思相同，重复了`
- `未回复，直接确认需求`
- `成语接龙长时间未回复`
- `会话显示已结束，共识总结结果却未显示完全`

定位重点：

- 这是行为/状态展示/流式输出类问题。
- 需要同时看：
  - history 中最终 assistant 消息
  - API/connection 的 streaming chunk / final
  - UI 展示状态
  - LLM trace 的实际输出
- 如果模型实际输出完整但 UI 缺失，偏上层展示/stream 聚合。
- 如果模型本身没输出或只输出确认，偏 prompt/上下文/中断策略。

## 7. 定时任务与系统恢复类

历史标题：

- `周期定时任务中断后首次触发报这次处理没有顺利完成`

定位重点：

- 先定界上层 scheduler/API/MaaS session，再看 jiuwenclaw。
- 休眠恢复、登录态失效、MaaS session 丢失是高频原因。
- 重点查 `api*.log` 中 `scheduler`、`Created invocation`、`Huawei MaaS session not found`、`invokeSingleCat crashed before fallback error emission`。

## 8. 步骤、状态与实际执行不一致类

历史标题：

- `步骤的总数不一致`
- `步骤与实际执行不符`

定位重点：

- 这是计划/进度展示与真实执行之间的一致性问题。
- 需要比较：
  - 模型计划输出
  - todo/step 工具调用
  - 实际工具执行
  - 前端步骤 UI
- history 工具用于还原步骤变化。
- trace 工具用于看模型每轮上下文中的 step/todo 状态。

## 9. 快速分流表

| 问题类型 | 首查日志/文件 | 外部工具优先级 |
|---|---|---|
| 前端失败/登录态/定时任务 | `install_logs/api*.log` | history/trace 次要 |
| PPT 效果/内容错乱 | `history.json` + `[LLM_IO_TRACE]` | history + trace 高 |
| 性能慢/长时间未回复 | `history.json` + tool-call speed + trace Timing | history speed + trace 高 |
| 中断/继续/多 session | API invocation + `history.json` | history 高，trace 中 |
| 工具空结果/路径错误 | `history.json` + tool-call speed + `full.log` | history 高 |
| UI 显示不完整 | API streaming + history | history 中 |

