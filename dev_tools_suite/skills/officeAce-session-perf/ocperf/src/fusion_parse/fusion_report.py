"""HTML report for fused history + full latency analysis."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ocperf.time_util import format_now, local_now

from fusion_parse.models import FusedModelRound, FusedTool, FusionSessionData
from history_parse.report_ui import (
    TimelineItem,
    _item_end,
    _item_start,
    _naive,
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
)
from history_parse.flow_view import (
    compose_extended_analysis,
    extended_report_scripts,
    extended_report_styles,
    wrap_extended_report,
)
from history_parse.session import is_measurable_tool
from history_parse.todo_tracker import render_todo_section_html
from parse_rules_snippets import guide_link_html


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _json_pretty(obj: Any, limit: int = 12000) -> str:
    return _esc(json.dumps(obj, ensure_ascii=False, indent=2)[:limit])


def _copyable_pre(escaped_text: str) -> str:
    return (
        f'<div class="pre-box"><button class="copy-btn" onclick="copyPre(this, event)">复制</button>'
        f"<pre>{escaped_text}</pre></div>"
    )


_STATUS_BADGE = {
    "matched": ("已匹配", "#2e7d32", "#e8f5e9"),
    "history_only": ("仅 history", "#1565c0", "#e3f2fd"),
    "full_only": ("仅 full 声明", "#e65100", "#fff3e0"),
    "name_mismatch": ("名称不一致", "#c62828", "#ffebee"),
    "aligned": ("已对齐", "#2e7d32", "#e8f5e9"),
    "history_missing": ("无 history 对照", "#757575", "#f5f5f5"),
    "weak_overlap": ("耗时偏差", "#f57c00", "#fff3e0"),
}


def _badge(label: str, fg: str, bg: str) -> str:
    return (
        f'<span class="badge-pill" style="background:{bg};color:{fg};border:1px solid {fg}33">'
        f"{_esc(label)}</span>"
    )


def _status_badge(status: str) -> str:
    label, fg, bg = _STATUS_BADGE.get(status, (status, "#333", "#eee"))
    return _badge(label, fg, bg)


def _model_block(idx: int, m: FusedModelRound) -> str:
    r = m.full
    child_cls = " timeline-child" if r.is_child_session else ""
    child_badge = (
        f'<span class="badge badge-orange">{_esc(r.child_label)}</span>' if r.is_child_session else ""
    )
    kind = "stream" if r.kind == "stream" else "invoke"
    match_line = ""
    if m.history:
        match_line = (
            f'<div class="meta-line">history 对照：总 {m.history.duration_sec:.3f}s '
            f"(TTFT {m.history.ttft_sec:.3f}s + 推理 {m.history.inference_sec:.3f}s)"
            f"{f' · Δ={m.duration_delta_sec:+.3f}s' if m.duration_delta_sec is not None else ''}"
            f"</div>"
        )
    notes = "".join(f"<div class=\"meta-line\">{_esc(n)}</div>" for n in m.notes)
    return f"""
<div class="timeline-item llm-item{child_cls}">
  <div class="request-header llm-header" onclick="toggleBlock('m-{idx}', this)">
    <div>
      <div class="request-id">
        <span class="badge-pill badge-time">{r.duration_sec:.3f}s</span>
        <span class="badge-pill badge-src">full · 模型</span>
        {_status_badge(m.match_status)}
        <span class="badge-pill badge-type">{kind} · iter {r.iteration}</span>
        <span class="title-main">{_esc(r.model_name)} · tok in {r.input_tokens or 0} out {r.output_tokens or 0}</span>
        <span class="title-time">{_esc(r.request_ts.strftime("%H:%M:%S.%f")[:-3])} → {_esc(r.output_ts.strftime("%H:%M:%S.%f")[:-3])}</span>
      </div>
      <div class="meta-line">request_id：{_esc(r.request_id)} · 重叠 {m.overlap_sec:.2f}s</div>
      {match_line}
      {notes}
    </div>
    <div class="header-right">{child_badge}<span class="arrow">▼</span></div>
  </div>
  <div class="request-details" id="m-{idx}">
    <div class="llm-body-box">
      <div class="collapsible-header" onclick="toggleBlock('min-{idx}', this)">
        <span>模型输入 (full)</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content active" id="min-{idx}">{_copyable_pre(_esc(r.request_body_full or "(无)"))}</div>
      <div class="collapsible-header" onclick="toggleBlock('mthink-{idx}', this)">
        <span>思考 (full reasoning_delta)</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="mthink-{idx}">{_copyable_pre(_esc(r.reasoning_full or ""))}</div>
      <div class="collapsible-header" onclick="toggleBlock('mout-{idx}', this)">
        <span>输出 (full)</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="mout-{idx}">{_copyable_pre(_esc(r.output_body_excerpt or ""))}</div>
    </div>
  </div>
</div>
"""


def _tool_block(idx: int, t: FusedTool) -> str:
    h = t.history
    if t.status == "full_only":
        return f"""
<div class="timeline-item tool-item tool-dimmed">
  <div class="request-header tool-header">
    <div>
      <div class="request-id">
        <span class="badge-pill badge-time">—</span>
        <span class="badge-pill badge-src">full 声明</span>
        {_status_badge(t.status)}
        <span class="title-main">{_esc(h.name)}</span>
      </div>
      <div class="meta-line">{_esc("; ".join(t.notes))}</div>
    </div>
  </div>
</div>
"""
    child_cls = " timeline-child" if h.is_child_session else ""
    notes = "".join(f'<div class="meta-line">{_esc(n)}</div>' for n in t.notes)
    gap_hint = ""
    if t.gap:
        gap_hint = (
            f'<div class="meta-line">落入 full 窗口 {_esc(t.gap.after_output_ts.strftime("%H:%M:%S"))}'
            f" → {_esc(t.gap.next_request_ts.strftime('%H:%M:%S'))}</div>"
        )
    return f"""
<div class="timeline-item tool-item{child_cls}">
  <div class="request-header tool-header" onclick="toggleBlock('t-{idx}', this)">
    <div>
      <div class="request-id">
        <span class="badge-pill badge-time">{h.duration_sec:.3f}s</span>
        <span class="badge-pill badge-src">history · 工具</span>
        {_status_badge(t.status)}
        <span class="title-main">{_esc(h.name)}</span>
        <span class="title-time">{_esc(h.start_ts.strftime("%H:%M:%S.%f")[:-3])} → {_esc(h.end_ts.strftime("%H:%M:%S.%f")[:-3])}</span>
      </div>
      {gap_hint}
      {notes}
    </div>
    <span class="arrow">▼</span>
  </div>
  <div class="request-details" id="t-{idx}">
    <div class="tool-body-box">
      <div class="meta-line">tool_call_id：{_esc(h.tool_call_id)}</div>
      <div class="collapsible-header" onclick="toggleBlock('targ-{idx}', this)">
        <span>参数</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="targ-{idx}">{_copyable_pre(_json_pretty(h.arguments))}</div>
      <div class="collapsible-header" onclick="toggleBlock('tres-{idx}', this)">
        <span>结果</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="tres-{idx}">{_copyable_pre(_esc(h.result[:8000]))}</div>
    </div>
  </div>
</div>
"""


def _reconcile_table(data: FusionSessionData) -> str:
    s = data.summary
    rows = [
        ("模型轮次 (full)", str(s.model_rounds_full), "墙钟与 Token 以此为准"),
        ("↳ history 时间对齐", str(s.model_rounds_history_matched), f"弱偏差 {s.model_weak_overlap}"),
        ("工具执行 (history)", str(s.tools_history), "单次 call→result，墙钟以此为准"),
        ("↳ 与 full 窗口匹配", str(s.tools_matched), f"仅 history {s.tools_history_only} · 仅 full {s.tools_full_only}"),
        ("full 工具窗口数", str(s.gaps_full), "output→下次 request"),
    ]
    body = "".join(
        f"<tr><td>{_esc(a)}</td><td><strong>{_esc(b)}</strong></td><td>{_esc(c)}</td></tr>"
        for a, b, c in rows
    )
    issues = ""
    if s.issues:
        issues = "<ul class=\"issue-list\">" + "".join(f"<li>{_esc(i)}</li>" for i in s.issues) + "</ul>"
    return f"""
<div class="section">
  <div class="section-title">交叉校对摘要</div>
  <table class="cmp-table">
    <thead><tr><th>维度</th><th>数量</th><th>说明</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
  {issues}
</div>
"""


def render_fusion_html(
    data: FusionSessionData,
    *,
    guide_href: str = "GUIDE.md",
    guide_label: str = "解析规则说明",
    extended_analysis: bool = False,
) -> str:
    s = data.summary
    ext_css = extended_report_styles() if extended_analysis else ""
    body_attr = ' class="extended-report"' if extended_analysis else ""
    files_rows = "".join(
        f"<tr><td>{_esc(Path(sf.path).name)}</td><td>{sf.trace_lines}</td>"
        f"<td>{_esc(sf.first_ts)}</td><td>{_esc(sf.last_ts)}</td></tr>"
        for sf in data.full_source_files
    )

    hist_tools = [t.history for t in data.tools if t.status != "full_only"]
    lane_merged: list[TimelineItem] = []
    for m in data.model_rounds:
        lane_merged.append(("llm", m.full))
    for t in data.tools:
        if t.status != "full_only":
            lane_merged.append(("tool", t.history))
    lane_merged.sort(key=lambda x: _naive(_item_start(x[0], x[1])))

    all_ts: list[datetime] = []
    for kind, obj in lane_merged:
        all_ts.extend([_item_start(kind, obj), _item_end(kind, obj)])
    if all_ts:
        all_naive = [_naive(t) for t in all_ts]
        session_start = min(all_naive)
        session_end = max(all_naive)
    else:
        session_start = session_end = local_now()

    llm_by_full = {id(m.full): m for m in data.model_rounds}
    tool_by_hist = {id(t.history): t for t in data.tools if t.status != "full_only"}

    def _render_item(idx: int, kind: str, obj: Any) -> str:
        if kind == "llm":
            m = llm_by_full.get(id(obj))
            block = _model_block(idx, m) if m else ""
            core = obj
        else:
            ft = tool_by_hist.get(id(obj))
            block = _tool_block(idx, ft) if ft else ""
            core = obj
        depth = len(core.child_path) if getattr(core, "is_child_session", False) else 0
        return wrap_item_with_session_depth(block, depth, getattr(core, "is_child_session", False))

    phases = build_execution_phases(data.extras.todo_timeline, session_start, session_end)
    buckets = partition_by_phase(lane_merged, phases)
    phased_timeline = render_phased_timeline_shell(phases, buckets, _esc, _render_item)

    full_only_blocks = "".join(
        _tool_block(9000 + i, t) for i, t in enumerate(data.tools, 1) if t.status == "full_only"
    )

    top_section = render_top_consumers_html(
        compute_top_llm([m.full for m in data.model_rounds], 5),
        compute_top_tools(
            hist_tools, 5, skip_predicate=lambda t: not is_measurable_tool(t.name)
        ),
        _esc,
        tool_panel_title="工具调用（history · 单次，不含 spawn/fork）",
        tool_by_name_rows=compute_top_tools_by_name(
            hist_tools, 5, skip_predicate=lambda t: not is_measurable_tool(t.name)
        ),
    )
    session_tree = render_session_hierarchy_html(
        collect_session_lanes(lane_merged, data.root_session),
        data.root_session,
        _esc,
    )

    todo_html = ""
    if data.extras.todo_timeline and getattr(data.extras.todo_timeline, "tasks", None):
        todo_html = render_todo_section_html(data.extras.todo_timeline, _esc) or ""

    full_only_section = (
        "<div class='section' style='margin-top:16px'><div class='section-title'>"
        "仅 full 声明（无 history 执行，不计墙钟）</div>"
        + full_only_blocks
        + "</div>"
        if full_only_blocks
        else ""
    )
    overview_body = f"""
  {top_section}
  {session_tree}
  {todo_html}

  {_reconcile_table(data)}

  <div class="section">
    <div class="section-title">full 数据源</div>
    <table class="cmp-table"><thead><tr><th>文件</th><th>TRACE 行</th><th>首条</th><th>末条</th></tr></thead>
    <tbody>{files_rows or "<tr><td colspan=4>—</td></tr>"}</tbody></table>
  </div>

  <div class="section">
    <div class="section-title">融合时间线（按 Todo 阶段划分）</div>
    <div class="legend-bar">
      <span>■ 绿色 = 模型 (full)</span> ·
      <span>■ 橙色 = 工具 (history)</span> ·
      <span>缩进 = 子 Agent</span>
    </div>
    {phased_timeline}
    {full_only_section}
  </div>
"""
    if extended_analysis:
        ext_html = compose_extended_analysis(phases, buckets, _esc)
        overview_body = wrap_extended_report(overview_body, ext_html, _esc)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>融合时延报告 — {_esc(data.root_session[:40])}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f8; color: #333; line-height: 1.55; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    .header {{ background: linear-gradient(135deg, #1a237e 0%, #5e35b1 50%, #00838f 100%); color: #fff; padding: 22px; border-radius: 10px; margin-bottom: 16px; }}
    .header h1 {{ font-size: 1.45em; margin-bottom: 8px; }}
    .header p {{ font-size: 0.9em; opacity: 0.95; }}
    .guide-link {{ color: #e1f5fe; text-decoration: underline; }}
    .trust-banner {{ background: #fff8e1; border: 1px solid #ffcc80; border-radius: 8px; padding: 12px 14px; margin-bottom: 16px; font-size: 0.9em; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 16px; }}
    .stat-card {{ background: #fff; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 3px solid #5e35b1; }}
    .stat-value {{ font-size: 1.3em; font-weight: bold; color: #4527a0; }}
    .stat-label {{ font-size: 0.78em; color: #666; margin-top: 4px; }}
    .section {{ background: #fff; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
    .section-title {{ font-size: 1.05em; font-weight: 700; border-bottom: 2px solid #5e35b1; padding-bottom: 8px; margin-bottom: 12px; }}
    .cmp-table {{ width: 100%; border-collapse: collapse; font-size: 0.86em; }}
    .cmp-table th, .cmp-table td {{ border: 1px solid #e0e4ee; padding: 8px; text-align: left; }}
    .cmp-table th {{ background: #ede7f6; }}
    .issue-list {{ margin: 10px 0 0 18px; color: #e65100; }}
    .timeline-item {{ border-left: 5px solid #5e35b1; padding-left: 14px; margin-bottom: 14px; border-radius: 0 8px 8px 0; }}
    .llm-item {{ border-left-color: #2e7d32; background: #f1f8f4; }}
    .tool-item {{ border-left-color: #e65100; background: #fff8f0; }}
    .tool-dimmed {{ opacity: 0.75; border-left-style: dashed; }}
    .timeline-child {{ margin-left: 16px; }}
    .request-header {{ padding: 12px; border-radius: 8px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
    .llm-header {{ background: #e8f5e9; }}
    .tool-header {{ background: #fff3e0; }}
    .request-details {{ display: none; padding: 10px 0; }}
    .request-details.active {{ display: block; }}
    .collapsible-header {{ background: #f0f0f0; padding: 8px 10px; cursor: pointer; border-radius: 6px; margin: 8px 0 4px; display: flex; justify-content: space-between; }}
    .collapsible-content {{ display: none; padding: 8px; background: #f7f8fa; border-radius: 6px; max-height: 420px; overflow: auto; }}
    .collapsible-content.active {{ display: block; }}
    .badge-pill {{ display: inline-block; border-radius: 6px; padding: 2px 8px; font-size: 0.78em; font-weight: 600; margin: 2px 4px 2px 0; }}
    .badge-time {{ background: #eef2ff; color: #4154c5; }}
    .badge-src {{ background: #ede7f6; color: #4527a0; }}
    .badge-type {{ background: #e8f5e9; color: #2e7d32; }}
    .title-time {{ color: #777; font-size: 0.85em; margin-left: 6px; }}
    .meta-line {{ font-size: 0.84em; color: #666; margin-top: 4px; word-break: break-all; }}
    .pre-box {{ position: relative; }}
    .copy-btn {{ position: absolute; top: 8px; right: 8px; z-index: 2; border: 1px solid #cfd7ff; background: #eef2ff; padding: 2px 8px; font-size: 12px; cursor: pointer; border-radius: 4px; }}
    pre {{ background: #101622; color: #e6edf3; padding: 10px; border-radius: 6px; white-space: pre-wrap; word-break: break-word; font-size: 0.78rem; }}
    .arrow {{ transition: transform 0.2s; }}
    .arrow.rotated {{ transform: rotate(180deg); }}
    .legend-bar {{ font-size: 0.88em; color: #555; margin-bottom: 12px; }}
    .badge-orange {{ background: #fff3e0; color: #f57c00; padding: 2px 8px; border-radius: 12px; font-size: 0.78em; }}
    {enhanced_report_styles()}
    {ext_css}
    .extended-report .wrap {{ max-width: 100%; padding: 20px 28px; }}
  </style>
</head>
<body{body_attr}>
<div class="wrap">
  <div class="header">
    <h1>融合时延性能分析报告</h1>
    <p>session：{_esc(data.root_session)} · 目录：{_esc(data.log_dir)}</p>
    <p>{_esc(data.history_label)} + {data.full_file_count} 个 full 文件 · {guide_link_html(guide_href, guide_label)}</p>
    <p>生成时间：{format_now()}</p>
  </div>

  <div class="trust-banner">
    <strong>可信数据源：</strong>模型轮次与 Token、模型输入取自 <code>full.json</code>（LLM_IO_TRACE）；
    工具调用起止与墙钟取自 <code>history.json</code>（tool_call → tool_result）。
    下表为两者时间轴交叉校对结果。
  </div>

  <div class="stats-grid">
    <div class="stat-card"><div class="stat-value">{s.task_sec}s</div><div class="stat-label">总任务时间</div></div>
    <div class="stat-card"><div class="stat-value">{s.llm_wall_sec}s</div><div class="stat-label">模型墙钟 (full)</div></div>
    <div class="stat-card"><div class="stat-value">{s.tool_wall_sec}s</div><div class="stat-label">工具墙钟 (history)</div></div>
    <div class="stat-card"><div class="stat-value">{s.model_rounds_full}</div><div class="stat-label">模型轮次</div></div>
    <div class="stat-card"><div class="stat-value">{s.tools_matched}/{s.tools_history}</div><div class="stat-label">工具匹配率</div></div>
    <div class="stat-card"><div class="stat-value">{s.total_tokens_sum:,}</div><div class="stat-label">Token Σ (full)</div></div>
  </div>

  {overview_body}
</div>
<script>
function toggleBlock(id, header) {{
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('active');
  const arrow = header.querySelector('.arrow');
  if (arrow) arrow.classList.toggle('rotated');
}}
function copyPre(btn, ev) {{
  ev.stopPropagation();
  const pre = btn.parentElement.querySelector('pre');
  if (!pre) return;
  navigator.clipboard.writeText(pre.textContent || '').then(() => {{
    btn.textContent = '已复制';
    setTimeout(() => {{ btn.textContent = '复制'; }}, 1200);
  }});
}}
{extended_report_scripts() if extended_analysis else ""}
</script>
</body>
</html>
"""


def write_fusion_report(path: Path, html_content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_content, encoding="utf-8")
