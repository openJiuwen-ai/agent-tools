"""Standalone HTML report (original LLM IO Trace visual design)."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ocperf.time_util import format_now, local_now

from history_parse.models import HistoryExtras, LLMRound, ToolExecution
from history_parse.report_ui import (
    build_execution_phases,
    collect_session_lanes,
    compute_top_llm,
    compute_top_tools,
    compute_top_tools_by_name,
    enhanced_report_styles,
    partition_by_phase,
    render_phased_timeline_shell,
    render_session_hierarchy_html,
    render_top_consumers_html,
    wrap_item_with_session_depth,
    _item_start,
    _item_end,
)
from history_parse.llm_latency_metrics import llm_round_metrics
from history_parse.timeline import aggregate_stats, merge_timeline
from history_parse.todo_tracker import render_todo_section_html
from history_parse.flow_view import (
    compose_extended_analysis,
    extended_report_scripts,
    extended_report_styles,
    wrap_extended_report,
)
from history_parse.session import is_measurable_tool
from parse_rules_snippets import guide_link_html


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _json_pretty(obj: Any) -> str:
    return _esc(json.dumps(obj, ensure_ascii=False, indent=2)[:12000])


def _copyable_pre(escaped_text: str) -> str:
    return (
        '<div class="pre-box"><button class="copy-btn" onclick="copyPre(this, event)">'
        f'复制</button><pre>{escaped_text}</pre></div>'
    )


def render_history_html(
    root_session: str,
    source_label: str,
    rounds: list[LLMRound],
    tools: list[ToolExecution],
    extras: HistoryExtras,
    *,
    guide_href: str = "GUIDE.md",
    guide_label: str = "解析规则说明",
    extended_analysis: bool = False,
) -> str:
    tot = aggregate_stats(rounds, tools)
    merged = merge_timeline(rounds, tools)

    all_ts_early: list[datetime] = []
    for kind, obj in merged:
        all_ts_early.extend([_item_start(kind, obj), _item_end(kind, obj)])
    session_start_early = min(all_ts_early) if all_ts_early else local_now()
    session_end_early = max(all_ts_early) if all_ts_early else local_now()
    phases_early = build_execution_phases(extras.todo_timeline, session_start_early, session_end_early)
    buckets_early = partition_by_phase(merged, phases_early)
    ext_css = extended_report_styles() if extended_analysis else ""
    body_attr = ' class="extended-report"' if extended_analysis else ""
    tool_calls_count = tot.get("tool_calls") or sum(
        1 for t in tools if is_measurable_tool(t.name)
    )
    input_tokens_sum = tot.get("input_tokens_sum") or 0
    output_tokens_sum = tot.get("output_tokens_sum") or 0
    total_tokens_sum = tot.get("total_tokens_sum") or 0
    cache_tokens_sum = tot.get("cache_tokens_sum") or 0
    task_total_sec = tot.get("task_sec") or 0
    tool_sec = tot.get("tool_sec") or 0
    spawn_tool_sec = tot.get("tool_sec_agent_spawn") or 0
    spawn_tool_calls = tot.get("spawn_tool_calls") or 0

    def _round_block(idx: int, r: LLMRound) -> str:
        label = _esc(r.child_label or "child-session")
        child_badge = f'<span class="badge badge-orange">{label}</span>' if r.is_child_session else ""
        child_cls = " timeline-child" if r.is_child_session else ""
        model_hint = f" · {_esc(r.model_name)}" if r.model_name else ""
        m = llm_round_metrics(r)
        ttft_hint = f'<span class="badge-pill badge-ttft">TTFT {r.ttft_sec:.3f}s</span>'
        infer_hint = f'<span class="badge-pill badge-infer">推理 {r.inference_sec:.3f}s</span>'
        tpot_hint = ""
        if m.get("tpot_sec") is not None:
            tpot_ms = m["tpot_sec"] * 1000
            tpot_hint = f'<span class="badge-pill badge-tpot">TPOT {tpot_ms:.1f}ms</span>'
        tps_hint = ""
        if m.get("tokens_per_sec") is not None:
            tps_hint = f'<span class="badge-pill badge-tps">{m["tokens_per_sec"]:.1f} tok/s</span>'
        cache_line = f" · cache {r.cache_tokens or 0}" if (r.cache_tokens or 0) else ""
        return f"""
<div class="timeline-item llm-item{child_cls}">
  <div class="request-header llm-header" onclick="toggleBlock('req-{idx}', this)">
    <div>
      <div class="request-id">
        <span class="badge-pill badge-time">{r.duration_sec:.3f}s</span>
        {ttft_hint}
        {infer_hint}
        {tpot_hint}
        {tps_hint}
        <span class="badge-pill badge-type">模型调用</span>
        <span class="title-main">token：in {r.input_tokens or 0} · out {r.output_tokens or 0} · total {r.total_tokens or 0}{cache_line}{model_hint}</span>
        <span class="title-time">等待 {_esc(r.request_ts.strftime("%H:%M:%S.%f")[:-3])} → 首token {_esc(r.first_token_ts.strftime("%H:%M:%S.%f")[:-3])} → 结束 {_esc(r.output_ts.strftime("%H:%M:%S.%f")[:-3])}</span>
      </div>
    </div>
    <div class="header-right">
      {child_badge}
      <span class="arrow">▼</span>
    </div>
  </div>
  <div class="request-details" id="req-{idx}">
    <div class="llm-body-box">
    <div class="collapsible-header" onclick="toggleBlock('think-{idx}', this)">
      <span>思考过程</span><span class="arrow">▼</span>
    </div>
    <div class="collapsible-content" id="think-{idx}">{_copyable_pre(_esc(r.reasoning_full or ""))}</div>
    <div class="collapsible-header" onclick="toggleBlock('reply-{idx}', this)">
      <span>模型回复</span><span class="arrow">▼</span>
    </div>
    <div class="collapsible-content" id="reply-{idx}">{_copyable_pre(_esc(r.assistant_text))}</div>
    </div>
  </div>
</div>
"""

    def _tool_block(idx: int, t: ToolExecution) -> str:
        payload = {
            "tool_call_id": t.tool_call_id,
            "name": t.name,
            "arguments": t.arguments,
            "result": t.result,
        }
        child_cls = " timeline-child" if t.is_child_session else ""
        label = _esc(t.child_label or "child-session")
        child_badge = f'<span class="badge badge-orange">{label}</span>' if t.is_child_session else ""
        return f"""
<div class="timeline-item tool-item{child_cls}">
  <div class="request-header tool-header" onclick="toggleBlock('tool-{idx}', this)">
    <div>
      <div class="request-id">
        <span class="badge-pill badge-time">{t.duration_sec:.3f}s</span>
        <span class="badge-pill badge-tool">工具调用</span>
        <span class="title-main">{_esc(t.name)}</span>
        <span class="title-time">{_esc(t.start_ts.strftime("%H:%M:%S.%f")[:-3])} → {_esc(t.end_ts.strftime("%H:%M:%S.%f")[:-3])}</span>
      </div>
      <div class="meta-line">tool_call_id：{_esc(t.tool_call_id)}</div>
    </div>
    <div class="header-right">{child_badge}<span class="arrow">▼</span></div>
  </div>
  <div class="request-details" id="tool-{idx}">
    <div class="tool-body-box">
      <div class="collapsible-header" onclick="toggleBlock('tool-args-{idx}', this)">
        <span>调用参数</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content active" id="tool-args-{idx}">{_copyable_pre(_json_pretty(t.arguments))}</div>
      <div class="collapsible-header" onclick="toggleBlock('tool-res-{idx}', this)">
        <span>执行结果</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="tool-res-{idx}">{_copyable_pre(_esc(t.result))}</div>
    </div>
  </div>
</div>
"""

    head = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Agent Session — {_esc(root_session[:40])}…</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px 15px; text-align: center; border-radius: 8px; margin-bottom: 20px; }}
    .header h1 {{ font-size: 1.6em; margin-bottom: 5px; }}
    .header p {{ font-size: 0.9em; opacity: 0.92; }}
    .guide-link {{ color: #e8eaf6; text-decoration: underline; text-underline-offset: 3px; }}
    .guide-link:hover {{ color: #fff; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .stat-card {{ background: var(--stat-bg, white); border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.08); border-top: 4px solid var(--stat-accent, #667eea); }}
    .stat-card:nth-child(1) {{ --stat-accent:#667eea; --stat-bg:#f4f6ff; }}
    .stat-card:nth-child(2) {{ --stat-accent:#26a69a; --stat-bg:#f0fbf9; }}
    .stat-card:nth-child(3) {{ --stat-accent:#f57c00; --stat-bg:#fff7ed; }}
    .stat-card:nth-child(4) {{ --stat-accent:#ab47bc; --stat-bg:#fcf3ff; }}
    .stat-card:nth-child(5) {{ --stat-accent:#42a5f5; --stat-bg:#eef8ff; }}
    .stat-card:nth-child(6) {{ --stat-accent:#7cb342; --stat-bg:#f5fbef; }}
    .token-card {{ grid-column: span 2; }}
    .stat-value {{ font-size: 1.5em; font-weight: bold; color: var(--stat-accent, #667eea); margin-bottom: 4px; }}
    .stat-label {{ font-size: 0.82em; color: #666; }}
    .token-lines {{ display:flex; justify-content:center; gap:14px; flex-wrap:wrap; color:#555; font-size:0.88em; margin-top:4px; }}
    .section {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
    .section-title {{ font-size: 1.2em; color: #333; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #667eea; }}
    .timeline-legend-bar {{ display:flex; flex-wrap:wrap; gap:16px; margin-bottom:14px; font-size:0.9em; color:#555; }}
    .timeline-legend-bar span {{ display:inline-flex; align-items:center; gap:6px; }}
    .legend-dot {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
    .legend-dot-llm {{ background:#2e7d32; }}
    .legend-dot-tool {{ background:#e65100; }}
    .timeline-item {{ border-left: 5px solid #667eea; padding-left: 14px; margin-bottom: 14px; border-radius: 0 8px 8px 0; }}
    .timeline-item.llm-item {{ border-left-color: #2e7d32; background: #f1f8f4; }}
    .timeline-item.tool-item {{ border-left-color: #e65100; background: #fff8f0; }}
    .tool-body-box {{ border: 1px solid #ffe0b2; border-radius: 8px; background: #fffdf8; padding: 10px; margin-top: 8px; }}
    .todo-section {{ margin-bottom: 24px; }}
    .todo-axis {{ display:flex; justify-content:space-between; color:#888; font-size:0.85em; margin-bottom:10px; }}
    .todo-batch {{ margin-bottom:18px; border:1px solid #e8eaf2; border-radius:8px; padding:12px; background:#fafbff; }}
    .todo-batch-title {{ font-weight:700; color:#3949ab; margin-bottom:10px; }}
    .todo-gantt {{ display:flex; flex-direction:column; gap:10px; }}
    .todo-row {{ display:grid; grid-template-columns:minmax(180px,28%) 1fr; gap:8px; align-items:center; }}
    .todo-row-label {{ font-size:0.88em; color:#333; display:flex; align-items:flex-start; gap:6px; }}
    .todo-status-dot {{ width:8px; height:8px; border-radius:50%; margin-top:5px; flex-shrink:0; }}
    .todo-row-track {{ position:relative; height:22px; background:#eef1f7; border-radius:6px; overflow:hidden; }}
    .todo-bar {{ position:absolute; top:2px; height:18px; border-radius:4px; min-width:4px; }}
    .todo-bar-tip {{ font-size:0.7em; color:#fff; padding:0 6px; line-height:18px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:block; }}
    .todo-row-meta {{ grid-column:1 / -1; display:flex; flex-wrap:wrap; gap:12px; font-size:0.78em; color:#666; margin-left:2px; }}
    .todo-pending {{ background:#bdbdbd; }}
    .todo-active {{ background:#1976d2; }}
    .todo-done {{ background:#2e7d32; }}
    .todo-cancelled {{ background:#e53935; }}
    .todo-table {{ width:100%; border-collapse:collapse; margin-top:16px; font-size:0.86em; }}
    .todo-table th, .todo-table td {{ border:1px solid #e0e4ee; padding:8px 10px; text-align:left; vertical-align:top; }}
    .todo-table th {{ background:#f0f3ff; color:#333; }}
    .todo-td-content {{ max-width:320px; word-break:break-word; }}
    .todo-td-changes {{ color:#555; font-size:0.9em; }}
    .todo-pill {{ display:inline-block; padding:2px 8px; border-radius:12px; color:#fff; font-size:0.82em; }}
    .timeline-item.timeline-child {{ margin-left: 20px; }}
    .llm-header {{ background: #e8f5e9 !important; }}
    .llm-header:hover {{ background: #dcefdc !important; }}
    .tool-header {{ background: #fff3e0 !important; }}
    .tool-header:hover {{ background: #ffe8c7 !important; }}
    .child-group {{ border-left: 6px solid #f57c00; background: #fff8ef; border-radius: 8px; padding: 10px 10px 4px; margin: 12px 0; }}
    .child-group .child-group {{ margin-left: 16px; margin-top: 10px; background: #fffaf5; border-left-color: #ff9800; }}
    .child-group .child-group .child-group {{ border-left-color: #fb8c00; background: #fffbf7; }}
    .child-group-title {{ font-size: 0.95em; font-weight: 700; color: #f57c00; margin: 0 0 8px 4px; }}
    .child-group-header {{ display:flex; justify-content:space-between; align-items:center; gap:8px; background:#fff1dc; border-radius:6px; padding:8px 10px; cursor:pointer; margin-bottom:8px; }}
    .child-group-header:hover {{ background:#ffe6c2; }}
    .fork-group-content {{ display:none; }}
    .fork-group-content.active {{ display:block; }}
    .child-group-continue {{ border-left: 3px solid #ffcc80; padding-left: 8px; margin: 6px 0 6px 0; background: transparent; }}
    .request-header {{ padding: 12px; border-radius: 8px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
    .request-id {{ font-weight: bold; color: #667eea; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
    .badge-pill {{ display:inline-block; border-radius:6px; padding:2px 8px; font-size:0.82em; font-weight:700; border:1px solid transparent; }}
    .badge-time {{ background:#eef2ff; color:#4154c5; border-color:#cfd7ff; }}
    .badge-type {{ background:#e8f5e9; color:#2e7d32; border-color:#c8e6c9; }}
    .badge-ttft {{ background:#e3f2fd; color:#1565c0; border-color:#bbdefb; }}
    .badge-infer {{ background:#f3e5f5; color:#6a1b9a; border-color:#e1bee7; }}
    .badge-tpot {{ background:#fff8e1; color:#f57f17; border-color:#ffe082; }}
    .badge-tps {{ background:#e8f5e9; color:#2e7d32; border-color:#c8e6c9; }}
    .badge-tool {{ background:#fff3e0; color:#e67e22; border-color:#ffe0b2; }}
    .title-main {{ color:#333; }}
    .title-time {{ color:#777; font-size:0.85em; }}
    .meta-line {{ color: #666; font-size: 0.86em; margin-top: 2px; word-break: break-all; }}
    .request-details {{ display: none; padding: 12px; background: #fafafa; border-radius: 8px; margin-top: 8px; }}
    .request-details.active {{ display: block; }}
    .collapsible-header {{ background: #f0f0f0; padding: 9px 10px; cursor: pointer; border-radius: 6px; margin: 10px 0 6px; display: flex; justify-content: space-between; align-items: center; }}
    .collapsible-header:hover {{ background: #e5e8eb; }}
    .collapsible-content {{ display: none; padding: 10px; background: #f7f8fa; border-radius: 6px; max-height: 420px; overflow-y: auto; }}
    .collapsible-content.active {{ display: block; }}
    .request-group-content {{ max-height: none; overflow: visible; }}
    .arrow {{ transition: transform 0.2s; }}
    .arrow.rotated {{ transform: rotate(180deg); }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 20px; font-size: 0.78em; font-weight: 600; }}
    .badge-orange {{ background: #fff3e0; color: #f57c00; }}
    .message-box {{ background: white; border: 1px solid #e0e0e0; border-left: 4px solid #4CAF50; border-radius: 8px; padding: 10px; margin: 8px 0; }}
    .llm-body-box {{ border: 1px solid #dfe3eb; border-radius: 8px; background: #fbfcfe; padding: 10px; margin-top: 8px; }}
    .message-header {{ margin-bottom: 6px; }}
    .message-label {{ font-weight: 700; color: #666; font-size: 0.9em; }}
    .pre-box {{ position: relative; }}
    .copy-btn {{ position: absolute; top: 8px; right: 8px; z-index: 2; border: 1px solid #cfd7ff; background: #eef2ff; color: #4154c5; border-radius: 5px; padding: 3px 8px; font-size: 12px; cursor: pointer; }}
    .copy-btn:hover {{ background: #dfe6ff; }}
    pre {{ background: #101622; color: #e6edf3; padding: 10px; border-radius: 6px; overflow: auto; font-size: 0.8rem; white-space: pre-wrap; word-break: break-word; }}
    code {{ background: #eef2ff; padding: 1px 5px; border-radius: 4px; }}
    .header-right {{ display: flex; align-items: center; gap: 8px; }}
    /* 并行时间线视图样式 */
    .parallel-timeline-container {{ background: #fff; border-radius: 10px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-top: 16px; }}
    .timeline-section-title {{ font-size: 1.1em; font-weight: 600; color: #333; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #667eea; }}
    .timeline-header {{ margin-bottom: 12px; }}
    .timeline-time-axis {{ display: flex; justify-content: space-between; padding: 0 8px; position: sticky; top: 0; background: #fff; z-index: 1; }}
    .time-label {{ font-size: 0.9em; color: #666; font-weight: 600; }}
    .timeline-scroll-container {{ overflow-x: auto; overflow-y: hidden; border-radius: 8px; border: 1px solid #e0e4ee; }}
    .timeline-lanes {{ min-width: 100%; padding: 12px; background: #fafafa; }}
    .timeline-lane {{ margin-bottom: 20px; background: #fff; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .lane-header {{ display: flex; align-items: center; gap: 10px; padding-bottom: 10px; margin-bottom: 10px; border-bottom: 2px solid #f0f0f0; }}
    .lane-header {{ border-left: 4px solid #667eea; padding-left: 10px; }}
    .lane-name {{ font-size: 1em; font-weight: 600; color: #333; }}
    .lane-type {{ font-size: 0.8em; color: #888; padding: 2px 6px; background: #f0f0f0; border-radius: 4px; }}
    .lane-content {{ position: relative; height: 48px; background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px; overflow: visible; }}
    .timeline-event {{ position: absolute; top: 8px; height: 32px; border-radius: 6px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s; min-width: 40px; z-index: 2; }}
    .timeline-event:hover {{ opacity: 0.9; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
    .event-label {{ font-size: 0.85em; color: white; font-weight: 500; text-shadow: 0 1px 2px rgba(0,0,0,0.3); padding: 2px 6px; text-align: center; }}
    .timeline-legend {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 16px; padding-top: 12px; border-top: 1px solid #f0f0f0; }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 0.9em; color: #666; }}
    .legend-color {{ width: 20px; height: 20px; border-radius: 4px; }}
    .event-tooltip {{ display: none; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); background: #333; color: white; padding: 8px 12px; border-radius: 6px; font-size: 0.85em; white-space: nowrap; z-index: 100; margin-bottom: 8px; }}
    .timeline-event:hover .event-tooltip {{ display: block; }}
    .event-tooltip::after {{ content: ''; position: absolute; top: 100%; left: 50%; transform: translateX(-50%); border: 6px solid transparent; border-top-color: #333; }}
    .session-count-badge {{ font-size: 0.8em; background: #667eea; color: white; padding: 2px 8px; border-radius: 10px; }}
    .time-marker-line {{ position: absolute; top: 0; bottom: 0; width: 1px; background: #ddd; }}
    .time-marker-label {{ position: absolute; bottom: -20px; left: 50%; transform: translateX(-50%); font-size: 0.75em; color: #888; white-space: nowrap; }}
    /* Step维度并行时间线样式 */
    .step-parallel-container {{ background: #fff; border-radius: 10px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-top: 16px; }}
    .step-timeline-header {{ margin-bottom: 12px; }}
    .step-timeline-title {{ font-size: 1em; font-weight: 600; color: #333; margin-bottom: 8px; }}
    .step-time-axis {{ position: relative; display: flex; justify-content: space-between; padding: 0 8px; height: 24px; }}
    .step-timeline-scroll {{ overflow-x: auto; overflow-y: hidden; border-radius: 8px; border: 1px solid #e0e4ee; }}
    .step-timeline-grid {{ min-width: 100%; padding: 12px; background: #f8f9fa; }}
    .step-lane {{ margin-bottom: 16px; background: #fff; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .step-lane-header {{ display: flex; align-items: center; gap: 8px; padding-bottom: 8px; margin-bottom: 8px; border-bottom: 1px solid #f0f0f0; }}
    .step-lane-indicator {{ width: 8px; height: 24px; border-radius: 4px; }}
    .step-lane-name {{ font-size: 0.95em; font-weight: 600; color: #333; }}
    .step-lane-type {{ font-size: 0.75em; color: #888; padding: 2px 6px; background: #f0f0f0; border-radius: 4px; }}
    .step-lane-content {{ position: relative; height: 44px; background: linear-gradient(to bottom, #fafafa 0%, #f0f0f0 100%); border-radius: 6px; overflow: visible; }}
    .step-event {{ position: absolute; top: 6px; height: 32px; border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s; min-width: 40px; z-index: 2; }}
    .step-event:hover {{ opacity: 0.9; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
    .step-event-label {{ font-size: 0.75em; color: white; font-weight: 500; text-shadow: 0 1px 2px rgba(0,0,0,0.3); }}
    .step-event-duration {{ font-size: 0.65em; color: rgba(255,255,255,0.9); margin-top: 1px; }}
    .step-timeline-legend {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 16px; padding-top: 12px; border-top: 1px solid #f0f0f0; }}
    .step-event-llm {{ border: 1px solid #388e3c; }}
    .step-event-tool {{ border: 1px solid #e65100; }}
    .step-event-agent {{ border: 1px solid #7b1fa2; }}
    {enhanced_report_styles()}
    {ext_css}
    .extended-report .container {{ max-width: 100%; padding: 20px 28px; }}
  </style>
</head>
<body{body_attr}>
  <div class="container">
  <div class="header">
    <h1>Agent 会话历史报告</h1>
    <p>根 session: {_esc(root_session)} · 生成: {_esc(format_now())} · {guide_link_html(guide_href, guide_label)}</p>
  </div>

  <div class="stats-grid">
    <div class="stat-card"><div class="stat-value">{tot["rounds"]}</div><div class="stat-label">思考轮次</div></div>
    <div class="stat-card"><div class="stat-value">{tot["llm_wall_sec"]}s</div><div class="stat-label">模型墙钟(含TTFT)</div></div>
    <div class="stat-card"><div class="stat-value">{tot.get("llm_ttft_sum_sec", 0)}s</div><div class="stat-label">TTFT累计</div></div>
    <div class="stat-card"><div class="stat-value">{tot.get("llm_inference_sum_sec", 0)}s</div><div class="stat-label">流式推理累计</div></div>
    <div class="stat-card"><div class="stat-value">{tool_calls_count}</div><div class="stat-label">工具调用次数</div></div>
    <div class="stat-card"><div class="stat-value">{tool_sec}s</div><div class="stat-label">工具调用墙钟</div></div>
    <div class="stat-card"><div class="stat-value">{spawn_tool_calls}</div><div class="stat-label">spawn/fork 次数</div></div>
    <div class="stat-card"><div class="stat-value">{spawn_tool_sec}s</div><div class="stat-label">spawn/fork 墙钟（不计入上项）</div></div>
    <div class="stat-card"><div class="stat-value">{task_total_sec}s</div><div class="stat-label">总任务时间</div></div>
    <div class="stat-card token-card">
      <div class="stat-value">{total_tokens_sum or "—"}</div>
      <div class="stat-label">总token消耗</div>
      <div class="token-lines"><span>input：{input_tokens_sum}</span><span>output：{output_tokens_sum}</span><span>cache：{cache_tokens_sum}</span></div>
    </div>
  </div>
"""


    extras_parts: list[str] = []
    if extras.user_turns:
        extras_parts.append('<div class="section"><div class="section-title">用户对话</div>')
        for t in extras.user_turns:
            extras_parts.append(
                f'<div class="message-box"><div class="message-header">'
                f'<span class="message-label">用户</span>'
                f'<span class="title-time">{_esc(t.timestamp.strftime("%H:%M:%S"))}</span></div>'
                f'<pre style="background:#f8fafc;color:#333;border:1px solid #e0e0e0;padding:10px;'
                f'border-radius:6px;white-space:pre-wrap">{_esc(t.content)}</pre></div>'
            )
        extras_parts.append("</div>")
    if extras.context_events:
        extras_parts.append('<div class="section"><div class="section-title">上下文压缩 / 重载</div>')
        for ev in extras.context_events:
            kind = "重载" if ev.kind == "reload" else "压缩"
            extras_parts.append(
                f'<div class="meta-line" style="margin:8px 0">'
                f'<span class="badge-pill badge-type">{_esc(kind)}</span> '
                f'{_esc(ev.summary)} '
                f'<span class="title-time">{_esc(ev.timestamp.strftime("%H:%M:%S"))}</span></div>'
            )
        extras_parts.append("</div>")
    if extras.todo_timeline and getattr(extras.todo_timeline, "tasks", None):
        todo_html = render_todo_section_html(extras.todo_timeline, _esc)
        if todo_html:
            extras_parts.append(todo_html)

    top_section = render_top_consumers_html(
        compute_top_llm(rounds, 5),
        compute_top_tools(tools, 5, skip_predicate=lambda t: not is_measurable_tool(t.name)),
        _esc,
        tool_by_name_rows=compute_top_tools_by_name(
            tools, 5, skip_predicate=lambda t: not is_measurable_tool(t.name)
        ),
    )
    lanes = collect_session_lanes(merged, root_session)
    session_tree_section = render_session_hierarchy_html(lanes, root_session, _esc)

    session_start = session_start_early
    session_end = session_end_early

    def _render_item(idx: int, kind: Literal["llm", "tool"], obj: LLMRound | ToolExecution) -> str:
        block = _round_block(idx, obj) if kind == "llm" else _tool_block(idx, obj)
        depth = len(obj.child_path) if obj.is_child_session else 0
        return wrap_item_with_session_depth(block, depth, obj.is_child_session)

    phases = phases_early
    buckets = buckets_early
    phased_body = render_phased_timeline_shell(phases, buckets, _esc, _render_item)

    blocks: list[str] = [top_section, session_tree_section]
    blocks.append('<div class="section"><div class="section-title">执行时间线（按 Todo 阶段划分）</div>')
    blocks.append(
        '<div class="timeline-legend-bar">'
        '<span><i class="legend-dot legend-dot-llm"></i> 模型（绿）</span>'
        '<span><i class="legend-dot legend-dot-tool"></i> 工具 call→result（橙）</span>'
        '<span>缩进 = 子 Agent 层次</span>'
        "</div>"
    )
    blocks.append(phased_body)
    blocks.append("</div>")

    body_content = "".join(extras_parts) + "\n".join(blocks)
    if extended_analysis:
        ext_html = compose_extended_analysis(phases, buckets, _esc)
        body_content = wrap_extended_report(body_content, ext_html, _esc)

    foot = """
<script>
function toggleBlock(id, headerEl) {
  const el = document.getElementById(id);
  if (!el) return;
  const isOpen = el.classList.contains('active');
  if (isOpen) {
    el.classList.remove('active');
  } else {
    el.classList.add('active');
  }
  const arrow = headerEl.querySelector('.arrow');
  if (arrow) arrow.classList.toggle('rotated', !isOpen);
}

async function copyPre(btn, event) {
  event.stopPropagation();
  const box = btn.closest('.pre-box');
  const pre = box ? box.querySelector('pre') : null;
  if (!pre) return;
  const text = pre.innerText || pre.textContent || '';
  try {
    await navigator.clipboard.writeText(text);
    const old = btn.textContent;
    btn.textContent = '已复制';
    setTimeout(() => { btn.textContent = old; }, 1200);
  } catch (err) {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    const old = btn.textContent;
    btn.textContent = '已复制';
    setTimeout(() => { btn.textContent = old; }, 1200);
  }
}
""" + (extended_report_scripts() if extended_analysis else "") + """
</script>
</div>
</body>
</html>
"""
    return head + body_content + foot



def write_history_report(path: Path, html_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
