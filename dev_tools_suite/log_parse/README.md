# trace-log-parse

解析 jiuwenclaw DEBUG 日志里的 `[LLM_IO_TRACE]`，按 `session_id` 还原一次 agent 执行过程，并生成本地 HTML 性能报告。

报告会展示模型调用、工具调用、子 agent、step 阶段、耗时、token 消耗和可视化柱状图，适合排查“慢在哪里”和“token 花在哪里”。

## 安装

```powershell
cd D:\code\trace_log_parse
uv sync
```

## 基本使用

解析单个日志文件：

```powershell
uv run python -m trace_log_parse test_data\full.log -s officeclaw_a10ac72ea3bcdcfe2741700d
```

解析一个日志目录下的所有 `full*.log`：

```powershell
uv run python -m trace_log_parse D:\path\to\logs -s officeclaw_a10ac72ea3bcdcfe2741700d
```

也可以用脚本入口：

```powershell
uv run trace-llm-report D:\path\to\logs -s officeclaw_xxx
```

如果省略 `--session/-s`，脚本会自动选择出现次数最多的非子会话 `session_id` 作为根会话。

## 输入支持

`log_file` 参数可以是：

- 单个 `.log` 文件，例如 `full.log`
- 一个目录，目录内会自动读取所有文件名满足 `full*.log` 的日志，例如：
  - `full.log`
  - `full_20260427_152914.log`
  - `full_20260428_073852.log`

目录模式下会合并多个文件中的 `[LLM_IO_TRACE]` 记录，再按时间排序分析。

## 输出

默认输出到日志文件所在目录或日志目录下：

```text
out_report_<session_id>.html
```

例如：

```text
out_report_officeclaw_a10ac72ea3bcdcfe2741700d.html
```

如果你显式指定 `-o/--out`，会按指定路径输出：

```powershell
uv run python -m trace_log_parse test_data\full.log -s officeclaw_xxx -o test_data\my_report.html
```

## 报告内容

顶部总览包括：

- 思考轮次
- 总思考时间
- 工具次数
- 工具调用总时间
- 总任务时间
- 总 token 消耗（含 input/output）

报告主体按如下层级组织：

```text
session id
  request_id
    step
      模型调用
      工具调用 / 流程间隙
      子模块（subagent/fork，按时间线穿插）
```

## Step 计算规则

Step 由 `skill_step_complete` / `skill_step_complete_batch` 切分：

- 从上一次 `skill_step_complete` 结束，到下一次 `skill_step_complete` 调用结束，算一个 step。
- `skill_step_complete_batch` 作为批量完成步骤的边界。
- 同时兼容新版统一工具格式：`skill_step`（或兼容拼写 `shill_step`）配合 `action="complete"` / `action="complete_batch"`。
- step 标题会从工具调用的 `arguments` 字段解析，例如 `idx`、`result`。
- 如果最后还有一段内容没有后续 `skill_step_complete` 闭合，会显示为 `收尾阶段`。

每个 step 会展示：

- 总耗时
- 工具耗时
- 工具次数
- input/output/total token
- step 内部事件耗时柱状图

## 可视化

报告包含两类图表：

- 全局 Step 耗时总览：按 step 显示模型耗时、工具耗时、其他耗时。
- Step 内部耗时图：展开 step 后，展示该 step 内模型调用和工具调用的耗时。

鼠标悬停柱状图时，会显示详细信息，包括 step 名称、耗时、工具次数和 token 消耗。

## 明细查看

模型调用包含：

- 模型输入：`stream_request` / `invoke_request`
- 思考过程：`reasoning_delta`
- 模型回复：`stream_output` / `invoke_output`

工具调用包含：

- 工具名
- 工具参数 JSON
- 起止时间与耗时

所有详情代码块右上角都有 `复制` 按钮，便于复制原始输入、输出或工具参数。

没有 `tool_calls` 的 output → next request 间隙会显示为 `流程间隙 / 无工具调用`，不再伪造成工具 JSON。

## 解析规则

- `body_part=i/total` 会按顺序合并为完整 body。
- `stream_request → stream_output` 记为一次模型调用。
- `invoke_request → invoke_output` 记为一次模型调用。
- 一次 output 到下一次 request 之间记为工具调用耗时或流程间隙。
- 子 agent 会话会被纳入根会话，例如 `主session_subagent_xxx`。
- token 从 `usage_metadata` 中解析，支持字符串形式的 `input_tokens` / `output_tokens` / `total_tokens`。

## 示例

```powershell
cd D:\code\trace_log_parse

# 单文件
uv run python -m trace_log_parse test_data\full.log -s officeclaw_a10ac72ea3bcdcfe2741700d

# 目录，读取所有 full*.log
uv run python -m trace_log_parse test_data -s officeclaw_a10ac72ea3bcdcfe2741700d

# 指定输出路径
uv run python -m trace_log_parse test_data -s officeclaw_a10ac72ea3bcdcfe2741700d -o test_data\report.html
```

## 注意事项

- Chart.js 通过 CDN 加载，离线环境下图表可能无法显示，但文本报告仍可查看。
- 多文件目录解析只读取当前目录下的 `full*.log`，不会递归子目录。
- step 依赖 `skill_step_complete` / `skill_step_complete_batch` 标记；如果日志中没有这些工具调用，报告会退化为较粗的阶段展示。
