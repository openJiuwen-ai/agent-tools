#!/usr/bin/env python3
"""Auto-generate analysis_report.md from skill_bundle.json (KPI + phases + Mermaid)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _phase_duplicates_md(phase: dict) -> str:
    dupes = phase.get("duplicates") or []
    if not dupes:
        return "无"
    lines = [f"  - `{d.get('name')}` ×{d.get('count')}" for d in dupes[:6]]
    return "\n".join(lines)


def _phase_failures_md(phase: dict) -> str:
    failures = phase.get("tool_failures") or []
    if not failures:
        return "无"
    lines = [f"  - `{f.get('tool_name')}`" for f in failures[:6]]
    return "\n".join(lines)


def _pct(part: float, total: float) -> str:
    if not total:
        return "—"
    return f"{100 * part / total:.1f}%"


def _top_slow_events(bundle: dict, n: int = 5) -> list[dict]:
    rows: list[dict] = []
    for ph in bundle.get("phases") or []:
        ptitle = ph.get("title") or ""
        for ev in ph.get("events") or []:
            dur = float(ev.get("duration_sec") or 0)
            if dur <= 0:
                continue
            rows.append(
                {
                    "name": ev.get("name") or "?",
                    "kind": ev.get("kind") or "?",
                    "duration_sec": dur,
                    "phase": ptitle,
                    "is_failure": ev.get("kind") == "tool"
                    and any(
                        f.get("tool_call_id") == ev.get("tool_call_id")
                        for f in ph.get("tool_failures") or []
                    ),
                }
            )
    rows.sort(key=lambda x: x["duration_sec"], reverse=True)
    return rows[:n]


def _collect_failures(bundle: dict) -> list[dict]:
    out: list[dict] = []
    for ph in bundle.get("phases") or []:
        for f in ph.get("tool_failures") or []:
            out.append({**f, "phase": ph.get("title") or ""})
    return out


def _collect_duplicates(bundle: dict) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for ph in bundle.get("phases") or []:
        for d in ph.get("duplicates") or []:
            key = (d.get("name"), d.get("kind"))
            if key in seen:
                continue
            seen.add(key)
            if int(d.get("count") or 0) >= 2:
                out.append({**d, "phase": ph.get("title") or ""})
    return out[:12]


def build_analysis_md(bundle: dict) -> str:
    sid = bundle.get("session_id", "—")
    summary = bundle.get("summary") or {}
    fr = bundle.get("fusion_reconcile") or {}
    ts = bundle.get("tool_stats") or {}
    rp = bundle.get("report_paths") or {}

    task = float(summary.get("task_sec") or fr.get("task_sec") or 0)
    llm = float(summary.get("llm_wall_sec") or fr.get("llm_wall_sec") or 0)
    tool = float(
        summary.get("tool_sec") or summary.get("tool_wall_sec") or fr.get("tool_wall_sec") or 0
    )
    bottleneck = "模型推理" if llm >= tool else "工具执行"
    has_fusion = bool(fr)
    cred = fr.get("credibility", "—" if has_fusion else "history+full 分离（未跑 fusion）")

    html_lines = "\n".join(
        f"| {k} | `{v}` |" for k, v in sorted(rp.items()) if v
    )
    if html_lines:
        html_table = "| 报告 | 路径 |\n|------|------|\n" + html_lines
    else:
        html_table = "_未生成 HTML，请运行 run_officeclaw_analysis.ps1_"

    slow = _top_slow_events(bundle, 5)
    slow_lines = "\n".join(
        f"{i + 1}. **{e['kind'].upper()} {e['name']}** — {e['duration_sec']:.1f}s"
        f"（阶段：{e['phase'] or '—'}）"
        + (" ⚠失败" if e.get("is_failure") else "")
        for i, e in enumerate(slow)
    ) or "_无事件_"

    fails = _collect_failures(bundle)
    fail_lines = (
        "\n".join(
            f"- `{f.get('tool_name')}` {f.get('duration_sec', 0):.1f}s — "
            f"{(f.get('snippet') or '')[:100]}（{f.get('phase', '')}）"
            for f in fails[:8]
        )
        or "- 无工具失败记录"
    )

    dups = _collect_duplicates(bundle)
    dup_lines = (
        "\n".join(
            f"- `{d.get('name')}` ×{d.get('count')} — {d.get('relationship', '—')}"
            f"（{d.get('phase', '')}）"
            for d in dups
        )
        or "- 无显著重复调用"
    )

    phase_blocks: list[str] = []
    for ph in bundle.get("phases") or []:
        phase_blocks.append(
            f"""### 阶段：{ph.get('title', '—')}

- **时间**：{ph.get('start', '—')} — {ph.get('end', '—')}
- **任务**：{ph.get('subtitle', '—')}
- **规模**：LLM {ph.get('llm_count', 0)} · 工具 {ph.get('tool_count', 0)}

#### 工具重复 / 并发
{_phase_duplicates_md(ph)}

#### 工具失败
{_phase_failures_md(ph)}

#### Agent 解读（必填）
<!-- 说明本阶段目标、主要耗时、是否无效重试 -->

"""
        )

    fr_section = ""
    if has_fusion:
        fr_section = f"""
## 附录：fusion 交叉校对

| 项 | 值 |
|----|-----|
| 工具匹配 | {fr.get('tools_matched', '—')}/{fr.get('tools_history', '—')} |
| 可信度 | {cred} |

**Agent 解读（必填）**：说明 fusion 结论是否影响边界。
"""
    else:
        fr_section = """
## 附录：数据说明

本次**未生成 fusion 报告**（默认关闭）。工具墙钟以 history 为准，模型墙钟/Token 以 full 报告为准；勿混用两套口径编造交叉结论。

**Agent 解读（必填）**：若需 fusion，请使用 `ocperf skill ... --fusion` 或 `-Fusion` 开关重新跑流水线。
"""

    mermaid = (
        bundle.get("e2e_mermaid_flowchart")
        or bundle.get("mermaid_flowchart")
        or "flowchart TD\n  empty[无数据]\n"
    )

    flow_svg = rp.get("e2e_flowchart_svg_html") or rp.get("e2e_flowchart_svg") or "—"
    flow_html = rp.get("e2e_flowchart_html", "—")
    mermaid_val = rp.get("e2e_flow_mermaid_validation", "—")
    hist_html = rp.get("history_html", "—")

    return f"""# 时延分析结论 — {sid}

**数据来源**：`{bundle.get('source', '—')}` · 生成于 `{bundle.get('generated_at', _now_iso())}`

## 中间 HTML 报告（必阅）

{html_table}

| 重点 | 路径 |
|------|------|
| History 报告 | `{hist_html}` |
| 向量时间轴（柱高=耗时） | `{flow_html}` |
| SVG 流程图（含重试箭头） | `{flow_svg}` |
| Mermaid 渲染 | `{rp.get('e2e_flow_mermaid_html', '—')}` |
| Mermaid 校验报告 | `{mermaid_val}` |

## 总览

| 指标 | 值 |
|------|-----|
| 总任务时间 | {task:.1f}s |
| 模型墙钟 | {llm:.1f}s ({_pct(llm, task)}) |
| 工具墙钟 | {tool:.1f}s ({_pct(tool, task)}) |
| Token 总计 | {summary.get('total_tokens_sum', fr.get('total_tokens_sum', '—'))} |
| 工具成功/失败 | {ts.get('success', '—')} / {ts.get('failure', '—')} |

**一句话结论（Agent 必填）**：脚本初判主瓶颈为 **{bottleneck}**；数据可信度 **{cred}**。

## 瓶颈排序（脚本初筛 + Agent 必填）

{slow_lines}

请结合流程图 HTML/SVG **核对并改写**上述排序，补充用户可感知的等待原因。

## 工具失败摘要

{fail_lines}

## 重复调用摘要

{dup_lines}

## 分阶段分析

{"".join(phase_blocks) if phase_blocks else "_无阶段数据_"}

## 端到端执行流程图

### 向量时间轴 / SVG（脚本生成，必开浏览器）

1. 打开 `{flow_html}` — 按时间顺序，柱高与耗时成正比
2. 打开 `{flow_svg}` — 完整 SVG，**虚线箭头 +「重试」= 连续重复调用**，红色=失败

### Mermaid（嵌入 MD，可渲染）

```mermaid
{mermaid}
```

图例：绿=LLM · 橙=工具 · 红=失败 · 紫=子Agent · 黄=慢 · 虚线箭头=重复/重试

**Agent 必填**：结合流程图指出 3 个最耗时步骤、失败/重复对用户等待的影响。

## 优化建议（Agent 必填）

1. 
2. 
3. 
{fr_section}
---
_由 generate_analysis_md.py 自动生成；Agent 须补全所有「必填」段落后交付。_
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="默认: bundle 同目录 analysis_report.md")
    args = ap.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    out = args.out or (args.bundle.parent / "analysis_report.md")
    out.write_text(build_analysis_md(bundle), encoding="utf-8")
    logger.info("已生成分析报告: %s", out)
    logger.info("请 Agent 打开 HTML/SVG 流程图，补全「必填」段落后交付。")


if __name__ == "__main__":
    main()
