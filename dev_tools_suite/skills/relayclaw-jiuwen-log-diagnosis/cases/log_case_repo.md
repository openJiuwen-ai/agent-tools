# log_case_repo.md（日志定位历史案例库）

说明：本文件用于沉淀 RelayClaw / JiuWenClaw 日志解析与问题定位技能的历史案例，供 Agent 排查时类比复用，统一定位、定界标准。

格式规范：每条案例按「案例标题 → 现象 → 关键日志特征 → 根因 → 定界 → 排查结论 → 补充说明」填写。关键字要清晰，方便 Agent 直接匹配；新增案例按序号递增，可按异常类型分类。

适用场景：OfficeClaw / RelayClaw / JiuWenClaw 的前端失败、定时任务异常、模型调用异常、工具调用异常、会话中断、投递失败、安装启动失败等所有日志定位相关场景。

维护原则：

- 先定界：每个案例的「定界」必须明确写出是不是 jiuwenclaw 问题，以及归属模块。
- 少结构：团队成员新增案例时只追加到本文件，不需要维护多文件索引。
- 可检索：关键日志特征中必须写可搜索关键字，如 `Huawei MaaS session not found`、`LLM invoke 失败`。
- 可复用：补充说明要写清楚“什么条件下可以类比”和“什么条件下不能套用”。
- 要脱敏：日志片段必须去掉 token、账号、手机号、内网地址、完整用户内容、个人路径等敏感信息。
- 不过度确定：证据不足时，在「定界」或「排查结论」里写明“暂不能确认”，不要写成已确认根因。

## 一、上层 API / 登录态 / MaaS 异常类案例

### 案例1：唤醒后首次定时任务因 MaaS session 缺失失败

**【现象】**周期定时任务在 Windows 休眠、隔夜或唤醒后首次触发时失败，前端显示“这次处理没有顺利完成”。jiuwenclaw `full.log` 中可能没有同一次主业务调用的完整错误栈。

**【关键日志特征】**

- 日志级别：ERROR
- 关键字：`Created invocation`、`Huawei MaaS session not found`、`invokeSingleCat crashed before fallback error emission`
- 关联信息：上层 API 日志中能看到 `invocationId`；jiuwenclaw 侧缺少同一时间窗口、同一 `request_id/session_id` 的完整 E2A/LLM 执行链路
- 时间特征：常见于休眠恢复、隔夜、登录态过期后的首次触发

**【根因】**RelayClaw / OfficeClaw API 在准备调用 office agent 或恢复会话时发现 Huawei MaaS / ModelArts session 缺失，调用在 fallback 错误消息生成前崩溃，前端只能显示笼统失败文案。

**【定界】**不是 jiuwenclaw 主执行链路问题。问题边界在上层 API 的 MaaS/session 恢复与错误 fallback，归属 RelayClaw / OfficeClaw API。

**【排查结论】**先在 `install_logs/api*.log` 或 `desktop-launcher.log` 查同一时间窗口的 `Created invocation`、`Huawei MaaS session not found`、`invokeSingleCat crashed before fallback error emission`。再用 `runtime_logs/full.log` 验证是否缺少同一主业务调用的完整 jiuwenclaw 链路。修复方向是定时任务触发前校验/刷新 MaaS session，并修复 fallback 前崩溃，让 session 缺失能返回明确错误文案。

**【补充说明】**此类问题容易被误判为 jiuwenclaw 没日志或模型失败。只有看到同一 `request_id/session_id` 下的 `LLM invoke 失败`、模型连接错误和重试链路时，才转向 jiuwenclaw LLM 调用问题。

## 二、jiuwenclaw LLM / 模型调用异常类案例

### 案例2：jiuwenclaw 调模型连接失败并重试后终止

**【现象】**前端显示“模型调用异常”或类似模型失败文案；日志中可见模型调用失败，系统按配置多次重试，通常表现为 1/3、2/3、3/3 重试后仍失败。

**【关键日志特征】**

- 日志级别：ERROR/WARN
- 关键字：`[E2A][in]`、`Before create openai client`、`LLM invoke 失败`、`openAI API async invoke error: Connection error`、`retry notification sent successfully`
- 关联信息：同一 `request_id/session_id` 下能串起 jiuwenclaw E2A 入口、模型调用、失败和重试；上层 API 可能记录 `Agent service emitted error message`
- 时间特征：可能出现在网络波动、模型网关不可达、代理异常、休眠恢复后的首次模型调用

**【根因】**请求已经进入 jiuwenclaw，但 jiuwenclaw 调用 OpenAI 兼容接口或模型网关时发生连接错误，重试后仍未恢复。

**【定界】**这是 jiuwenclaw 内部 LLM 调用链路问题，具体边界在 `jiuwenclaw/llm_invoke`。上层 API 的 `Agent service emitted error message` 只是收到下游错误，不是直接根因。

**【排查结论】**先在 `runtime_logs/full.log` 找同一 `request_id/session_id` 的 `[E2A][in]`、`Before create openai client`、`LLM invoke 失败` 和重试日志。若存在 `[LLM_IO_TRACE]`，再用 trace 工具确认最后一次 `invoke_request`、是否有 `invoke_output`、messages/tools 是否异常。修复方向是排查模型网关、网络、代理、DNS、证书、API key、endpoint 和休眠恢复后的连接健康检查。

**【补充说明】**不能只凭 `Before create openai client` 判定模型连接错误；必须同时看到失败和重试日志。如果 API 日志显示 `Huawei MaaS session not found` 且请求没有完整进入 jiuwenclaw，应优先按上层 MaaS/session 问题处理。

## 三、投递 / 渠道绑定异常类案例

> 暂无已确认案例。新增时按下面模板追加。

## 四、调度器 / 休眠恢复异常类案例

> 暂无已确认案例。若根因是 scheduler tick、next trigger 或补偿触发异常，放入本类；若根因是 tick 后的 API MaaS/session 缺失，放入“一、上层 API / 登录态 / MaaS 异常类案例”。

## 五、工具调用 / 文件路径 / 产物异常类案例

> 暂无已确认案例。适用于工具返回空、发送不存在文件、产物路径错误、PPT/HTML 转换链路异常等。

## 六、会话中断 / 历史丢失 / 多 Agent 异常类案例

> 暂无已确认案例。适用于中断继续、history 丢失、多 session 相互影响、多 agent 上下文不可见等。

## 七、安装 / 启动 / 环境异常类案例

> 暂无已确认案例。适用于安装失败、启动失败、配置缺失、权限或路径异常等。

## 八、sidecar / 进程生命周期异常类案例

### 案例3：sidecar/runtime 变化触发 invocation AbortError

**【现象】**前端显示笼统失败，API 日志中能看到 invocation 创建或执行中断，随后 `invokeSingleCat crashed before fallback error emission`，异常类型可能是 `AbortError`。如果只搜 invocation 或模型调用关键词，容易误判为普通 API fallback 崩溃或 jiuwenclaw 执行失败。

**【关键日志特征】**

- 日志级别：INFO/WARN/ERROR
- 关键字：`AbortError`、`invokeSingleCat crashed before fallback error emission`、`sidecar stop`、`willKill`、`runtime_signature_changed`、`process exit`、`SIGTERM`、`SIGKILL`、`sidecarPid`
- 关联信息：同一时间窗口内 API/launcher 日志存在 sidecar 或进程生命周期事件；jiuwenclaw 侧可能没有同一请求的自然 complete/error chunk
- 时间特征：sidecar/process 事件通常出现在 invocation 崩溃前后 30 秒内

**【根因】**sidecar/runtime 被停止、重启、kill，或 runtime signature 变化触发进程生命周期管理，导致正在执行的 invocation 被 abort。`invokeSingleCat crashed` 是结果，不是最终触发点。

**【定界】**不是 jiuwenclaw 主执行链路的模型/工具问题。问题边界优先在上层 RelayClaw/OfficeClaw 的 sidecar/runtime 进程生命周期管理；是否属于用户取消、超时、runtime 变更、守护进程重启或进程崩溃，需要结合前后 30 秒日志确认。

**【排查结论】**看到 `AbortError` 时，必须在 `install_logs/api*.log` 和 `desktop-launcher.log` 中追加搜索 sidecar/进程生命周期关键词，不能止步于 `invokeSingleCat crashed`。如果找到 `runtime_signature_changed`、`willKill`、`sidecar stop` 或 process kill 事件，应把它作为触发点继续追查；再用 jiuwenclaw `full.log` 验证该请求是否缺少自然完成链路。

**【补充说明】**此案例不能套用于所有 `invokeSingleCat crashed`。如果前置错误是 `Huawei MaaS session not found`，应按 MaaS/session 案例处理；如果请求已进入 jiuwenclaw 并出现完整 `LLM invoke 失败` 与重试链路，应按 jiuwenclaw LLM 调用问题处理。

## 九、新增案例模板（直接复制填写）

### 案例N：[填写案例标题，简洁明了，如：XXX异常（XXX现象）]

**【现象】**[填写具体故障现象，如前端报错、任务失败、模型异常、工具异常、响应超时等，说明影响范围、出现频率。]

**【关键日志特征】**

- 日志级别：[INFO/WARN/ERROR/FATAL]
- 关键字：[填写核心报错关键字、异常信息，便于搜索]
- 异常堆栈关键行：[如有，复制日志中异常堆栈核心行，注意脱敏]
- 关联信息：[填写 invocationId、request_id、session_id、task_id、时间特征、相关日志文件]

**【根因】**[填写故障根本原因，明确具体问题；证据不足时写“暂不能确认”。]

**【定界】**[先写是不是 jiuwenclaw 问题，再写归属模块，如上层 API、scheduler、jiuwenclaw LLM、工具调用、投递、前端展示、安装环境等。]

**【排查结论】**[填写具体排查方法、修复措施、需要补充的日志或可落地动作。]

**【补充说明】**[填写案例复用要点、注意事项、容易误判的相似问题。]

