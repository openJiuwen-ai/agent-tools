"""HTML latency report for full.json (LLM_IO_TRACE)."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from full_parse.loader import FullSessionData
from ocperf.time_util import format_now, local_now
from full_parse.stats import aggregate_full_stats
from full_parse.timeline import merge_full_timeline
from full_parse.trace_analysis import LLMRound, ToolGap
from history_parse.flow_view import (
    compose_extended_analysis,
    extended_report_scripts,
    extended_report_styles,
    wrap_extended_report,
)
from history_parse.report_ui import (
    agent_timeline_scripts,
    agent_timeline_styles,
    build_execution_phases,
    collect_session_lanes,
    compute_top_llm,
    compute_top_tool_windows,
    enhanced_report_styles,
    partition_by_phase,
    render_phased_timeline_shell,
    render_session_hierarchy_html,
    render_timeline_dual_view,
    TimelineDualViewRequest,
    render_top_consumers_html,
)
from history_parse.todo_tracker import render_todo_section_html
from parse_rules_snippets import guide_link_html


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _json_pretty(obj: Any) -> str:
    return _esc(json.dumps(obj, ensure_ascii=False, indent=2)[:50000])


def _copyable_pre(escaped_text: str) -> str:
    return (
        f'<div class="pre-box"><button class="copy-btn" onclick="copyPre(this, event)">复制</button>'
        f"<pre>{escaped_text}</pre></div>"
    )


def _llm_block(idx: int, r: LLMRound) -> str:
    child_badge = (
        f'<span class="badge badge-orange">{_esc(r.child_label)}</span>' if r.is_child_session else ""
    )
    kind_label = "stream" if r.kind == "stream" else "invoke"
    tools_summary = ""
    if r.tools:
        names = ", ".join(str(t.get("name") or "?") for t in r.tools)
        tools_summary = f'<div class="meta-line"><strong>本轮声明 tool_calls：</strong>{_esc(names)}</div>'
    return f"""
<div class="timeline-item llm-item">
  <div class="request-header llm-header" onclick="toggleBlock('req-{idx}', this)">
    <div>
      <div class="request-id">
        <span class="badge-pill badge-time">{r.duration_sec:.3f}s</span>
        <span class="badge-pill badge-type">模型调用 · {kind_label}</span>
        <span class="badge-pill badge-meta">iter {r.iteration}</span>
        <span class="title-main">{_esc(r.model_name)} · tok in {r.input_tokens or 0} out {r.output_tokens or 0} cache {r.cache_tokens or 0} Σ {r.total_tokens or 0}</span>
        <span class="title-time">{_esc(r.request_ts.strftime("%H:%M:%S.%f")[:-3])} → {_esc(r.output_ts.strftime("%H:%M:%S.%f")[:-3])}</span>
      </div>
      <div class="meta-line">request_id：{_esc(r.request_id)}</div>
      {tools_summary}
    </div>
    <div class="header-right">{child_badge}<span class="arrow">▼</span></div>
  </div>
  <div class="request-details" id="req-{idx}">
    <div class="llm-body-box">
      <div class="collapsible-header" onclick="toggleBlock('in-{idx}', this)">
        <span>模型输入 (stream_request / invoke_request)</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content active" id="in-{idx}">{_copyable_pre(_esc(r.request_body_full or "(无请求体)"))}</div>
      <div class="collapsible-header" onclick="toggleBlock('think-{idx}', this)">
        <span>思考过程 (reasoning_delta × {r.reasoning_batches})</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="think-{idx}">{_copyable_pre(_esc(r.reasoning_full or ""))}</div>
      <div class="collapsible-header" onclick="toggleBlock('out-{idx}', this)">
        <span>模型输出 (stream_output / invoke_output)</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="out-{idx}">{_copyable_pre(_esc(r.output_body_excerpt or ""))}</div>
      <div class="collapsible-header" onclick="toggleBlock('tools-{idx}', this)">
        <span>tool_calls JSON</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="tools-{idx}">{_copyable_pre(_json_pretty(r.tools))}</div>
    </div>
  </div>
</div>
"""


def _tool_window_block(idx: int, g: ToolGap) -> str:
    title = g.tools_triggered or "(no tool_calls)"
    note = "墙钟：上一轮 stream_output 结束 → 下一轮 stream_request 开始"
    if g.duration_sec <= 0:
        note += " · 末轮 tool_calls 无后续 request（记为 0s）"
    blocks = "".join(_copyable_pre(_json_pretty(t)) for t in g.detail_tools)
    if not blocks:
        blocks = '<div class="meta-line">无结构化 tool_calls，可能为流程等待。</div>'
    return f"""
<div class="timeline-item tool-item">
  <div class="request-header tool-header" onclick="toggleBlock('gap-{idx}', this)">
    <div>
      <div class="request-id">
        <span class="badge-pill badge-time">{g.duration_sec:.3f}s</span>
        <span class="badge-pill badge-tool">工具执行窗口</span>
        <span class="title-main">{_esc(title)}</span>
        <span class="title-time">{_esc(g.after_output_ts.strftime("%H:%M:%S.%f")[:-3])} → {_esc(g.next_request_ts.strftime("%H:%M:%S.%f")[:-3])}</span>
      </div>
      <div class="meta-line">{_esc(note)}</div>
    </div>
    <span class="arrow">▼</span>
  </div>
  <div class="request-details" id="gap-{idx}">
    <div class="tool-body-box"><div class="collapsible-content active">{blocks}</div></div>
  </div>
</div>
"""


def render_full_html(
    data: FullSessionData,
    *,
    guide_href: str = "GUIDE.md",
    guide_label: str = "解析规则说明",
    extended_analysis: bool = False,
) -> str:
    tot = aggregate_full_stats(data.rounds, data.gaps)
    merged = merge_full_timeline(data.rounds, data.gaps)
    ext_css = extended_report_styles() if extended_analysis else ""
    body_attr = ' class="extended-report"' if extended_analysis else ""

    files_rows = "".join(
        f"<tr><td>{_esc(Path(s.path).name)}</td><td>{s.trace_lines}</td>"
        f"<td>{_esc(s.first_ts)}</td><td>{_esc(s.last_ts)}</td></tr>"
        for s in data.source_files
    )

    all_ts: list[datetime] = []
    for kind, obj in merged:
        if kind == "llm":
            all_ts.extend([obj.request_ts, obj.output_ts])
        else:
            all_ts.extend([obj.after_output_ts, obj.next_request_ts])
    session_start = min(all_ts) if all_ts else local_now()
    session_end = max(all_ts) if all_ts else local_now()

    def _render_item(idx: int, kind: str, obj: LLMRound | ToolGap) -> str:
        return _llm_block(idx, obj) if kind == "llm" else _tool_window_block(idx, obj)

    phases = build_execution_phases(data.todo_timeline, session_start, session_end)
    buckets = partition_by_phase(merged, phases)
    phased_timeline = render_phased_timeline_shell(phases, buckets, _esc, _render_item)
    lanes = collect_session_lanes(merged, data.root_session)
    timeline_section = render_timeline_dual_view(
        TimelineDualViewRequest(
            chronological_html=phased_timeline,
            lanes=lanes,
            merged=merged,
            root_session=data.root_session,
            esc=_esc,
            render_item=_render_item,
        )
    )
    top_section = render_top_consumers_html(
        compute_top_llm(data.rounds, 5),
        compute_top_tool_windows(data.gaps, 5),
        _esc,
        tool_panel_title="工具执行窗口（full，墙钟区间）",
    )
    session_tree = render_session_hierarchy_html(
        lanes,
        data.root_session,
        _esc,
        timeline_indent_note=False,
    )

    todo_html = ""
    if data.todo_timeline and getattr(data.todo_timeline, "tasks", None):
        todo_html = render_todo_section_html(data.todo_timeline, _esc) or ""

    history_note = ""
    if data.history_source:
        history_note = (
            f'<p class="meta-line">Todo 任务阶段来自 '
            f'<code>{_esc(Path(data.history_source).name)}</code></p>'
        )
    elif not data.todo_timeline:
        history_note = (
            '<p class="meta-line">未找到 <code>history.json</code>（full 日志本身不含 Todo；'
            "需同 session 目录下的 history 或运行 skill 时传入 history 路径）。</p>"
        )

    overview_body = f"""
  {top_section}
  {session_tree}
  {todo_html}

  <div class="section">
    <div class="section-title">数据源文件</div>
    <table class="cmp-table"><thead><tr><th>文件</th><th>TRACE 行数</th><th>首条时间</th><th>末条时间</th></tr></thead>
    <tbody>{files_rows}</tbody></table>
  </div>

  <div class="section">
    <div class="section-title">执行时间线</div>
    {history_note}
    <div class="legend-bar">
      <span><i class="legend-dot" style="background:#2e7d32"></i> 模型调用</span>
      <span style="margin-left:16px"><i class="legend-dot" style="background:#e65100"></i> 工具执行窗口</span>
    </div>
    {timeline_section}
  </div>
"""
    if extended_analysis:
        ext_html = compose_extended_analysis(phases, buckets, _esc)
        overview_body = wrap_extended_report(overview_body, ext_html, _esc)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Full Log 时延分析 — {_esc(data.root_session[:48])}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    .header {{ background: linear-gradient(135deg, #3949ab 0%, #5c6bc0 100%); color: #fff; padding: 22px; border-radius: 10px; margin-bottom: 18px; }}
    .header h1 {{ font-size: 1.5em; margin-bottom: 6px; }}
    .guide-link {{ color: #e8eaf6; text-decoration: underline; text-underline-offset: 3px; }}
    .guide-link:hover {{ color: #fff; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .stat-card {{ background: #fff; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 3px solid #3949ab; }}
    .stat-value {{ font-size: 1.35em; font-weight: bold; color: #3949ab; }}
    .stat-label {{ font-size: 0.8em; color: #666; }}
    .section {{ background: #fff; border-radius: 10px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    .section-title {{ font-size: 1.1em; font-weight: 700; color: #333; border-bottom: 2px solid #3949ab; padding-bottom: 8px; margin-bottom: 12px; }}
    .cmp-table {{ width: 100%; border-collapse: collapse; font-size: 0.86em; }}
    .cmp-table th, .cmp-table td {{ border: 1px solid #e0e4ee; padding: 8px; text-align: left; }}
    .cmp-table th {{ background: #eef0fa; }}
    .timeline-item {{ border-left: 5px solid #3949ab; padding-left: 14px; margin-bottom: 14px; border-radius: 0 8px 8px 0; }}
    .llm-item {{ border-left-color: #2e7d32; background: #f1f8f4; }}
    .tool-item {{ border-left-color: #e65100; background: #fff8f0; }}
    .request-header {{ padding: 12px; border-radius: 8px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }}
    .llm-header {{ background: #e8f5e9; }}
    .tool-header {{ background: #fff3e0; }}
    .request-details {{ display: none; padding: 12px 0; }}
    .request-details.active {{ display: block; }}
    .collapsible-header {{ background: #f0f0f0; padding: 8px 10px; cursor: pointer; border-radius: 6px; margin: 8px 0 4px; display: flex; justify-content: space-between; }}
    .collapsible-content {{ display: none; padding: 8px; background: #f7f8fa; border-radius: 6px; max-height: 480px; overflow: auto; }}
    .collapsible-content.active {{ display: block; }}
    .badge-pill {{ display: inline-block; border-radius: 6px; padding: 2px 8px; font-size: 0.8em; font-weight: 600; margin-right: 4px; }}
    .badge-time {{ background: #eef2ff; color: #4154c5; }}
    .badge-type {{ background: #e8f5e9; color: #2e7d32; }}
    .badge-tool {{ background: #fff3e0; color: #e65100; }}
    .badge-meta {{ background: #f3e5f5; color: #7b1fa2; }}
    .title-time {{ color: #777; font-size: 0.85em; margin-left: 8px; }}
    .meta-line {{ font-size: 0.84em; color: #666; margin-top: 4px; word-break: break-all; }}
    .pre-box {{ position: relative; }}
    .copy-btn {{ position: absolute; top: 8px; right: 8px; z-index: 2; border: 1px solid #cfd7ff; background: #eef2ff; color: #4154c5; border-radius: 4px; padding: 2px 8px; font-size: 12px; cursor: pointer; }}
    pre {{ background: #101622; color: #e6edf3; padding: 10px; border-radius: 6px; white-space: pre-wrap; word-break: break-word; font-size: 0.78rem; }}
    .legend-bar {{ margin-bottom: 12px; font-size: 0.9em; color: #555; }}
    .legend-dot {{ width: 10px; height: 10px; display: inline-block; border-radius: 2px; margin-right: 6px; }}
    .badge-orange {{ background: #fff3e0; color: #f57c00; padding: 2px 8px; border-radius: 12px; font-size: 0.78em; }}
    {enhanced_report_styles()}
    {agent_timeline_styles()}
    {ext_css}
    .extended-report .wrap {{ max-width: 100%; padding: 20px 28px; }}
  </style>
</head>
<body{body_attr}>
<div class="wrap">
  <div class="header">
    <h1>Full Log 时延分析报告</h1>
    <p>session_id：{_esc(data.root_session)} · 生成 {format_now()}</p>
    <p>合并 {len(data.source_files)} 个 full.json · trace 记录 {len(data.records)} 条 · {guide_link_html(guide_href, guide_label)}</p>
  </div>

  <div class="stats-grid">
    <div class="stat-card"><div class="stat-value">{tot['task_sec']}s</div><div class="stat-label">总任务时间</div></div>
    <div class="stat-card"><div class="stat-value">{tot['rounds']}</div><div class="stat-label">模型轮次</div></div>
    <div class="stat-card"><div class="stat-value">{tot['llm_wall_sec']}s</div><div class="stat-label">模型墙钟</div></div>
    <div class="stat-card"><div class="stat-value">{tot['tool_windows']}</div><div class="stat-label">工具窗口数</div></div>
    <div class="stat-card"><div class="stat-value">{tot['tool_wall_sec']}s</div><div class="stat-label">工具墙钟</div></div>
    <div class="stat-card"><div class="stat-value">{tot['agent_wall_sec']}s</div><div class="stat-label">子Agent窗口墙钟</div></div>
    <div class="stat-card"><div class="stat-value">{tot['total_tokens_sum']:,}</div><div class="stat-label">Token 总计</div></div>
    <div class="stat-card"><div class="stat-value">{tot.get('cache_tokens_sum', 0):,}</div><div class="stat-label">cache_token 总计</div></div>
  </div>

  {overview_body}
</div>
<script>
function toggleBlock(id, headerEl) {{
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('active');
  const arrow = headerEl.querySelector('.arrow');
  if (arrow) arrow.classList.toggle('rotated', el.classList.contains('active'));
}}
async function copyPre(btn, event) {{
  event.stopPropagation();
  const pre = btn.closest('.pre-box')?.querySelector('pre');
  if (!pre) return;
  const text = pre.innerText || '';
  try {{ await navigator.clipboard.writeText(text); btn.textContent = '已复制'; }}
  catch (e) {{ btn.textContent = '失败'; }}
  setTimeout(() => {{ btn.textContent = '复制'; }}, 1200);
}}
{agent_timeline_scripts()}
{extended_report_scripts() if extended_analysis else ""}
</script>
</body>
</html>
"""


def write_full_report(path: Path, html_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
