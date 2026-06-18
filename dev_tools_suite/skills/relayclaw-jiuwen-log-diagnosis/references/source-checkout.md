# 源码拉取与代码定位

日志指纹不足以定界、用户要求代码级修复、需要确认前端文案/错误映射、或分析者无法解释日志对应的代码语义时，必须读取架构参考或源码。不要在日志证据足够且代码语义已知时默认拉远端最新源码。

源码是解释和修复工具，也是避免误判的重要校验手段，但不是日志事实的替代品。OfficeClaw/RelayClaw/JiuWenClaw 的问题通常发生在已经发布的版本上，而远端默认分支可能已经包含后续改动；如果源码版本和发布版本没有对齐，代码只能辅助理解模块结构，不能单独作为本次发布版本的根因证据。

## 0. 代码参与层级

代码理解必须参与诊断，只是参与方式分层：

1. **固化代码语义**：先使用 `references/architecture.md`、`references/fingerprints.md`、`cases/log_case_repo.md` 中已经沉淀的架构、日志语义和历史案例。
2. **轻量源码查证**：当边界不稳、日志语义不熟、候选案例互相冲突、或需要解释错误传播路径时，读取相关函数/模块确认语义。
3. **深度源码定位**：当需要修复点、补丁、PR、测试建议，或 fallback/状态机/投递/前端映射本身异常时，系统定位源码路径。

如果没有做过第 1 层或第 2 层代码语义确认，不要把结论写成 `confirmed`；应写成 `probable`，并说明需要哪些源码或版本信息来确认。

## 1. 源码来源优先级

按下面顺序选择源码来源：

1. **用户 query 中提供的本地源码路径或仓库信息**：优先使用。先确认路径存在、仓库名、当前分支、commit、远端 URL，再决定查 `relay-claw` 还是 `jiuwenclaw`。
2. **采集包或日志能识别的发布版本源码**：如果 `MANIFEST.txt`、安装日志或用户说明里有版本号、tag、commit、构建号，优先请求用户提供对应版本源码；如果用户无法提供，再从远端拉取后 checkout 到对应 tag/commit。
3. **官方远端默认分支**：只有在确实需要源码、且用户没有提供路径、也无法识别发布版本时才使用。结论中必须标注“源码版本未确认与问题发布版本一致”。

开始诊断时先记录 query 里的源码线索。是否立即读取源码取决于边界是否需要代码语义确认；不要为了准备一个可能用不到、且版本可能不一致的远端默认分支而阻塞日志和案例库分析。

## 2. 默认源码仓库与分支

### RelayClaw / OfficeClaw 上层仓库

- 仓库：`https://gitcode.com/openJiuwen/relay-claw.git`
- 分支：`main`
- 作用：上层 API、前端/桌面集成、调度器、invocation、Huawei MaaS session、错误 fallback、outbound delivery。

### JiuWenClaw vendor 仓库

- 仓库：`https://gitcode.com/openJiuwen/jiuwenclaw.git`
- 分支：`enterprise_dev`
- 放置位置：拉取后放到 `relay-claw/vendor/jiuwenclaw`
- 作用：agent runtime、E2A/WebSocket、LLM 调用、工具执行、session memory、`LLM_IO_TRACE`。

## 3. 推荐本地目录

不要硬编码盘符，也不要假设用户有 `E:` 盘。选择一个不会污染当前业务仓库的临时/缓存目录：

- Windows PowerShell：`Join-Path $env:TEMP "relayclaw-source"`
- Linux/macOS：`${TMPDIR:-/tmp}/relayclaw-source` 或 `~/.cache/relayclaw-source`
- 如果用户提供了明确源码路径，直接使用用户路径，不要复制或覆盖。

Windows PowerShell 示例：

```powershell
$sourceRoot = Join-Path $env:TEMP "relayclaw-source"
New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null
Set-Location $sourceRoot
git clone -b main https://gitcode.com/openJiuwen/relay-claw.git relay-claw
New-Item -ItemType Directory -Force -Path "relay-claw\vendor" | Out-Null
git clone -b enterprise_dev https://gitcode.com/openJiuwen/jiuwenclaw.git "relay-claw\vendor\jiuwenclaw"
```

如果目录已存在，先检查分支和远端，不要直接覆盖：

```powershell
Set-Location (Join-Path $sourceRoot "relay-claw")
git status
git branch --show-current
git rev-parse HEAD
git remote -v

Set-Location (Join-Path $sourceRoot "relay-claw\vendor\jiuwenclaw")
git status
git branch --show-current
git rev-parse HEAD
git remote -v
```

需要更新时，先确认是否有用户改动；不要覆盖未提交改动：

```powershell
Set-Location (Join-Path $sourceRoot "relay-claw")
git fetch origin
git checkout main
git pull --ff-only

Set-Location (Join-Path $sourceRoot "relay-claw\vendor\jiuwenclaw")
git fetch origin
git checkout enterprise_dev
git pull --ff-only
```

如果日志或用户提供了 tag/commit，拉取后优先 checkout 到对应版本：

```powershell
git fetch --all --tags
git checkout <tag-or-commit>
git rev-parse HEAD
```

## 4. 什么时候查哪个仓库

### 查 relay-claw

当日志/问题涉及：

- `Huawei MaaS session not found`
- `invokeSingleCat crashed before fallback error emission`
- `Created invocation`
- `Session init: binding session`
- `[scheduler] dyn-...`
- `OutboundDeliveryHook`
- `No bindings found for thread`
- 前端文案：`这次处理没有顺利完成`
- API 错误如何映射到前端
- 定时任务触发、恢复、补触发

优先搜索关键词：

```text
Huawei MaaS session not found
invokeSingleCat
fallback error emission
Created invocation
executeInBackground
OutboundDeliveryHook
No bindings found for thread
scheduler
scheduled task
```

### 查 relay-claw/vendor/jiuwenclaw

当日志/问题涉及：

- `LLM invoke 失败`
- `model call failed`
- `openAI API async invoke error`
- `APIConnectionError`
- `[E2A][in]`
- `[E2A][wire][out]`
- `[LLM_IO_TRACE]`
- `JiuWenClawDeepAdapter`
- `AgentWebSocketServer`
- `session_memory_update`
- 工具调用、ReAct、session memory

优先搜索关键词：

```text
LLM invoke 失败
model call failed
Before create openai client
LLM_IO_TRACE
AgentWebSocketServer
E2A
JiuWenClawDeepAdapter
session_memory_update
retry notification sent successfully
```

## 5. 拉取时机建议

成熟的诊断顺序是：

1. 先用 `cases/log_case_repo.md` 做历史案例候选召回。
2. 再用本次日志的时间窗口、关联 ID 和关键指纹做边界判定。
3. 如果候选边界依赖已沉淀的架构/指纹且证据闭环，可以先给边界结论；如果语义不清或存在竞争边界，必须进入源码或架构参考确认。
4. 以下条件之一成立时，必须进入源码：
   - 日志指纹和历史案例不能解释现象。
   - 分析者不确定某条关键日志在代码中的精确含义。
   - 同一现象同时可能属于上层 API、jiuwenclaw、投递或前端展示中的多个边界。
   - 需要确认错误文案、fallback、状态机或投递逻辑如何映射。
   - 用户要求代码级根因、修复方案、补丁或 PR。
   - 同一日志指纹在不同版本含义可能不同，需要结合版本确认。
5. 如果用户已经给了源码路径，可以在早期做轻量查证，但仍要把代码结论和日志证据分开表述，避免被未验证代码牵引。
6. 如果没有源码路径，先询问用户是否有问题发生时对应的源码版本；用户无法提供且确实需要代码时，再自行拉取官方仓库。

## 6. 代码定位输出要求

拉源码后，结论仍要先基于日志证据。代码只能用于解释“为什么会这样”和“如何修”。

输出时按这个顺序：

1. 日志证据已经证明的问题边界。
2. 本次读取的源码路径、仓库、分支/tag/commit，以及是否与发布版本对齐。
3. 对应源码位置/函数/模块。
4. 代码逻辑如何导致该现象。
5. 最小修复策略。
6. 需要补的测试。

不要只说“代码里可能有问题”。必须把代码路径与日志指纹关联起来。

