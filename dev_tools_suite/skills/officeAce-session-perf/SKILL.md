---
name: officeclaw-session-perf
description: >-
  OfficeClaw 会话性能分析。配置 config.local.ps1 后按 session_id 生成 HTML 报告、
  LLM 时延、流程图 SVG、skill_bundle.json。触发：会话性能、延迟、TTFT、工具墙钟、
  officeclaw、Agent 瓶颈、执行流程。
---

# OfficeClaw 会话性能分析

自包含 Skill（内置 `ocperf/`）。**上传前**运行 `.\scripts\clean_for_upload.ps1` 清除缓存。

## 配置（一次性）

```powershell
Copy-Item config\config.example.ps1 config\config.local.ps1
# 填写 SessionsRoot（history.json 根目录）与 LogsRoot（full*.log 目录）
.\scripts\setup_env.ps1
```

## 运行

```powershell
.\run.ps1 -SessionId <session_id>
# 测试目录：.\run.ps1 -SessionId xxx -DataDir D:\data -OutDir D:\out
# 可选 fusion：.\run.ps1 -SessionId xxx -Fusion
```

输出：`<SessionsRoot>/officeclaw_<id>/perf_reports/`

## 交付物

| 文件 | 说明 |
|------|------|
| `out_history_*.html` | 工具墙钟、Todo、扩展分析 |
| `out_full_*.html` | LLM_IO_TRACE |
| `out_unified_*.html` | 单时间线总览 |
| `out_llm_latency_*.html` | TTFT/推理时延 |
| `execution_flowchart_*` | 流程图 HTML/SVG |
| `skill_bundle_*.json` | Agent 结构化数据 |
| `analysis_report.md` | 报告骨架 |

## 数据口径

- 工具墙钟 → history.json
- 模型墙钟、Token → full 日志
- 未开 fusion 时不做 history/full 逐条对齐

## skill_bundle 要点

- `summary`: task_sec / llm_wall_sec / tool_wall_sec / total_tokens_sum
- `phases[].events[]`: kind=llm|tool, duration_sec, exclude_from_tool_kpi（spawn 不计 KPI）
- `fusion_reconcile`: 仅 `-Fusion` 时有值
- `report_paths`: 各 HTML 绝对路径

## 约束

1. 先跑脚本再写结论；Mermaid 校验 `ok=true` 才交付
2. 以 bundle + HTML 为据，禁止臆测
