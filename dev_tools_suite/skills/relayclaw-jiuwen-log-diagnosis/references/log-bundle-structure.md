# OfficeClaw 日志采集包结构

当前推荐使用成熟采集工具：

```text
scripts/office_claw_log_collector_4.2.bat
```

如果用户已经通过该工具采集好日志，通常不需要再引导用户重新找原始日志；直接让用户提供解压后的日志包目录即可。

## 1. 采集工具行为

采集工具会：

1. 询问开始/结束时间。
2. 自动查找 OfficeClaw 安装目录。
3. 从用户目录采集 jiuwenclaw runtime 日志。
4. 从安装目录采集上层 install/API/audit 日志。
5. 从 session 目录采集时间范围内的 `history.json` / `metadata.json`。
6. 生成 `MANIFEST.txt`。
7. 打包为 `office_claw_log_<date>_<time>.zip`。

输出 zip 默认在 bat 所在目录。用户通常会再解压成同名目录。

## 2. 解压后的标准目录

典型结构：

```text
office_claw_log_YYYYMMDD_HHMMSS/
├── MANIFEST.txt
├── runtime_logs/
│   ├── full.log
│   └── openjiuwen/
│       ├── run/
│       │   └── jiuwen.log
│       └── llm.log
├── install_logs/
│   ├── api.YYYY-MM-DD.N.log
│   ├── desktop-launcher.log
│   ├── audit-YYYY-MM-DD.ndjson
│   └── other install logs...
└── sessions/
    └── officeclaw_<session_id>/
        ├── history.json
        └── metadata.json
```

实际文件可能比上面少，原因包括：采集时间窗口过窄、日志级别未开启、session LastWriteTime 不在窗口内、安装目录日志不存在。

## 3. 分析入口顺序

拿到采集包后按这个顺序看：

1. `MANIFEST.txt`
   - 采集版本、时间范围、安装目录、数据目录。
   - 先确认问题发生时间是否落在采集范围内。

2. `install_logs/api*.log`
   - 上层 invocation、scheduler、MaaS session、fallback、API 错误。
   - 定时任务、前端“这次处理没有顺利完成”、登录态问题优先看这里。

3. `install_logs/desktop-launcher.log`
   - 上层运行时聚合日志。
   - 常用于补 API 错误、启动、前端桥接信息。

4. `runtime_logs/full.log`
   - jiuwenclaw 主日志。
   - LLM 调用、E2A、工具、trace、session memory。

5. `runtime_logs/openjiuwen/run/jiuwen.log` / `runtime_logs/openjiuwen/llm.log`
   - 分模块补充日志。
   - LLM 成功/失败、工具细节。

6. `sessions/<session_id>/history.json`
   - 一次任务的用户消息、assistant 消息、工具调用历史。
   - 效果问题、流程问题、继续执行问题、多 agent 协作问题常需要看。

## 4. 时间窗口注意事项

采集工具会按用户输入时间过滤 runtime 日志和 session 目录。分析时要确认：

- 问题发生时间是否在 `MANIFEST.txt` 的时间范围内。
- 如果用户只采集了失败时间点，可能缺少任务开始阶段，需要补采更早窗口。
- 如果问题是“任务中断后继续”，需要覆盖：原任务开始、首次中断、继续触发、结束/失败。
- 如果问题是性能问题，需要覆盖完整耗时区间，不要只采集报错点。

## 5. 采集包缺失时的处理

如果用户还没有采集包，可以让用户运行：

```powershell
"<skill-dir>\scripts\office_claw_log_collector_4.2.bat"
```

让用户输入覆盖问题发生前后的时间范围。建议：

- 一次调用失败：前后各 10-20 分钟。
- 长任务/性能问题：从任务开始前 5 分钟到结束后 5 分钟。
- 定时任务/休眠恢复：从休眠前最后一次正常触发到唤醒后首次失败后 10 分钟。
- 多 session 并发：覆盖所有 session 创建到异常结束。

