"""Unified session HTML — full.json layout + history Todo phases + TTFT/TPOT."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ocperf.time_util import format_now, local_now

from full_parse.loader import FullSessionData
from full_parse.stats import aggregate_full_stats
from full_parse.timeline import merge_full_timeline
from full_parse.trace_analysis import LLMRound as FullLLMRound
from full_parse.trace_analysis import ToolGap
from history_parse.flow_view import (
    compose_extended_analysis,
    extended_report_scripts,
    extended_report_styles,
    wrap_extended_report,
)
from history_parse.llm_latency_metrics import (
    aggregate_llm_latency,
    compute_tpot,
    compute_tokens_per_sec,
)
from history_parse.models import HistoryExtras, LLMRound, ToolExecution
from history_parse.report_ui import (
    ExecutionPhase,
    TimelineKind,
    _item_end,
    _item_start,
    _pick_phase,
    build_execution_phases,
    collect_session_lanes,
    compute_top_llm,
    compute_top_tool_windows,
    enhanced_report_styles,
    partition_by_phase,
    render_phased_timeline_shell,
    render_session_hierarchy_html,
    render_top_consumers_html,
    wrap_item_with_session_depth,
)
from history_parse.timeline import aggregate_stats, merge_timeline
from history_parse.todo_tracker import render_todo_section_html
from parse_rules_snippets import guide_link_html

FullItem = tuple[Literal["llm", "tool_window"], FullLLMRound | ToolGap]
_MIN_OVERLAP_SEC = 0.5


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _json_pretty(obj: Any, limit: int = 50000) -> str:
    return _esc(json.dumps(obj, ensure_ascii=False, indent=2)[:limit])


def _copyable_pre(escaped_text: str) -> str:
    return (
        f'<div class="pre-box"><button class="copy-btn" onclick="copyPre(this, event)">复制</button>'
        f"<pre>{escaped_text}</pre></div>"
    )


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _overlap_sec(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    start = max(_naive(a0), _naive(b0))
    end = min(_naive(a1), _naive(b1))
    if end <= start:
        return 0.0
    return (end - start).total_seconds()


def _fmt_tpot(tpot: float | None) -> str:
    if tpot is None:
        return "—"
    if tpot < 0.01:
        return f"{tpot * 1000:.1f} ms"
    return f"{tpot:.3f} s"


def _fmt_tps(tps: float | None) -> str:
    return f"{tps:.1f}" if tps is not None else "—"


def _coalesce_token(full_val: int | None, hist_val: int | None) -> int:
    """Prefer history usage_metadata when matched; full may omit or zero-fill cache."""
    if hist_val is not None:
        return int(hist_val)
    if full_val is not None:
        return int(full_val)
    return 0


def _merged_task_sec(
    hist_rounds: list[LLMRound],
    hist_tools: list[ToolExecution],
    full_data: FullSessionData,
) -> float:
    """Wall span across history tools + full/history LLM timestamps."""
    all_ts: list[datetime] = []
    for r in hist_rounds:
        all_ts.extend([r.request_ts, r.output_ts])
    for t in hist_tools:
        all_ts.extend([t.start_ts, t.end_ts])
    for r in full_data.rounds:
        all_ts.extend([r.request_ts, r.output_ts])
    for g in full_data.gaps:
        all_ts.extend([g.after_output_ts, g.next_request_ts])
    naive_ts = [_naive(t) for t in all_ts]
    if not naive_ts:
        return 0.0
    return round((max(naive_ts) - min(naive_ts)).total_seconds(), 3)


@dataclass
class _HistMatch:
    history: LLMRound | None
    overlap_sec: float = 0.0
    note: str = ""


def match_history_to_full(
    fr: FullLLMRound,
    hist_rounds: list[LLMRound],
    used: set[int],
) -> _HistMatch:
    rid_hits: list[tuple[float, int]] = []
    for i, hr in enumerate(hist_rounds):
        if i in used:
            continue
        if hr.request_id and hr.request_id == fr.request_id:
            ov = _overlap_sec(hr.request_ts, hr.output_ts, fr.request_ts, fr.output_ts)
            rid_hits.append((ov, i))
    if rid_hits:
        best_ov, best_i = max(rid_hits, key=lambda x: x[0])
        used.add(best_i)
        note = "request_id 匹配"
        if best_ov >= _MIN_OVERLAP_SEC:
            note += f" · 重叠 {best_ov:.1f}s"
        return _HistMatch(history=hist_rounds[best_i], overlap_sec=best_ov, note=note)
    best_i = -1
    best_ov = 0.0
    for i, hr in enumerate(hist_rounds):
        if i in used:
            continue
        ov = _overlap_sec(hr.request_ts, hr.output_ts, fr.request_ts, fr.output_ts)
        if ov > best_ov:
            best_ov = ov
            best_i = i
    if best_i >= 0 and best_ov >= _MIN_OVERLAP_SEC:
        used.add(best_i)
        return _HistMatch(
            history=hist_rounds[best_i],
            overlap_sec=best_ov,
            note=f"时间重叠 {best_ov:.1f}s",
        )
    return _HistMatch(history=None, note="无 history 对照")


def _sum_matched_cache(
    full_rounds: list[FullLLMRound],
    matches: dict[int, _HistMatch],
) -> int:
    total = 0
    for i, fr in enumerate(full_rounds):
        hm = matches.get(i, _HistMatch(None))
        h = hm.history
        total += _coalesce_token(fr.cache_tokens, h.cache_tokens if h else None)
    return total


def diagnose_zero_inference(h: LLMRound | None, f: FullLLMRound) -> str:
    """Explain why history inference_sec ≈ 0 (no output vs non-streaming)."""
    if h is None:
        if f.duration_sec <= 0.001:
            return "无 history 对照；full 墙钟≈0（output 未与 request 配对或同刻结束）"
        return "无 history 对照，无法从 history 计算 TTFT/推理分段"

    if h.inference_sec > 0.001:
        return ""

    parts: list[str] = []
    same_moment = abs((h.output_ts - h.first_token_ts).total_seconds()) < 0.005
    out_tok = int(h.output_tokens or 0)
    has_text = bool((h.reasoning_full or "").strip() or (h.assistant_text or "").strip())
    has_tools = bool(f.tools)

    if same_moment:
        parts.append("首 token 与 usage_metadata 同刻 → 流式输出时间窗为 0")

    if out_tok == 0:
        parts.append("output_tokens=0")
    elif same_moment:
        parts.append(f"output_tokens={out_tok} 已在 usage 瞬间入账，history 无后续 delta 间隔")

    if not has_text:
        parts.append("history 无 reasoning/delta 文本")
    if has_tools and not (f.assistant_preview or "").strip():
        parts.append("full 输出以 tool_calls 为主")

    if f.kind == "invoke":
        parts.append("full=invoke（非 stream 流式）")
    elif f.kind == "stream" and f.reasoning_batches == 0 and not (f.reasoning_full or "").strip():
        parts.append("full 无 reasoning_delta 批次")

    if out_tok == 0 and has_tools:
        parts.append("【结论】无 output token，模型直接返回 tool_calls")
    elif out_tok > 0 and same_moment:
        parts.append("【结论】有输出 token，但非流式间隔——一次性返回/同刻记账，非“无输出”")
    elif out_tok == 0:
        parts.append("【结论】无 output token")
    else:
        parts.append("【结论】推理为 0，请结合上方条目判断")

    return " · ".join(parts)


def partition_full_by_phase(
    merged: list[FullItem],
    phases: list[ExecutionPhase],
) -> dict[str, list[tuple[int, str, Any]]]:
    buckets: dict[str, list] = {p.phase_id: [] for p in phases}
    for idx, (kind, obj) in enumerate(merged, 1):
        tl_kind: TimelineKind = "llm" if kind == "llm" else "tool"
        start = _item_start(tl_kind, obj)
        end = _item_end(tl_kind, obj)
        mid = start + (end - start) / 2
        pid = _pick_phase(mid, phases)
        buckets.setdefault(pid, []).append((idx, kind, obj))
    return buckets


def _llm_block(
    idx: int,
    fr: FullLLMRound,
    hm: _HistMatch,
) -> str:
    h = hm.history
    child_cls = " timeline-child" if fr.is_child_session else ""
    child_badge = (
        f'<span class="badge badge-orange">{_esc(fr.child_label)}</span>' if fr.is_child_session else ""
    )
    kind_label = "stream" if fr.kind == "stream" else "invoke"
    tools_summary = ""
    if fr.tools:
        names = ", ".join(str(t.get("name") or "?") for t in fr.tools)
        tools_summary = f'<div class="meta-line"><strong>tool_calls：</strong>{_esc(names)}</div>'

    ttft_badge = infer_badge = tpot_badge = tps_badge = hist_line = zero_warn = ""

    if h:
        tpot = compute_tpot(h.inference_sec, h.output_tokens)
        tps = compute_tokens_per_sec(h.output_tokens, h.inference_sec)
        ttft_badge = f'<span class="badge-pill badge-ttft">TTFT {h.ttft_sec:.3f}s</span>'
        infer_badge = f'<span class="badge-pill badge-infer">推理 {h.inference_sec:.3f}s</span>'
        tpot_badge = f'<span class="badge-pill badge-tpot">TPOT {_fmt_tpot(tpot)}</span>'
        tps_badge = f'<span class="badge-pill badge-tps">{_fmt_tps(tps)} tok/s</span>'
        hist_line = (
            f'<div class="meta-line"><span class="src-tag hist">history</span> '
            f"等待→首token→结束 · {_esc(hm.note)}</div>"
        )
        diag = diagnose_zero_inference(h, fr)
        if diag:
            zero_warn = f'<div class="meta-line infer-zero-warn">⚠ 推理=0：{_esc(diag)}</div>'
    else:
        hist_line = f'<div class="meta-line muted"><span class="src-tag hist">history</span> {_esc(hm.note)}</div>'

    tin = _coalesce_token(fr.input_tokens, h.input_tokens if h else None)
    tout = _coalesce_token(fr.output_tokens, h.output_tokens if h else None)
    tcache = _coalesce_token(fr.cache_tokens, h.cache_tokens if h else None)
    ttot = _coalesce_token(fr.total_tokens, h.total_tokens if h else None) or (tin + tout)

    hist_think = ""
    if h and (h.reasoning_full or h.assistant_text):
        hist_body = (h.reasoning_full or "") + "\n" + (h.assistant_text or "")
        hist_think = f"""
      <div class="collapsible-header" onclick="toggleBlock('uhist-{idx}', this)">
        <span>思考/回复 (history reasoning+delta)</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="uhist-{idx}">{_copyable_pre(_esc(hist_body))}</div>
"""

    return f"""
<div class="timeline-item llm-item{child_cls}">
  <div class="request-header llm-header" onclick="toggleBlock('req-{idx}', this)">
    <div>
      <div class="request-id">
        <span class="badge-pill badge-time">{fr.duration_sec:.3f}s</span>
        {ttft_badge}{infer_badge}{tpot_badge}{tps_badge}
        <span class="badge-pill badge-type">模型 · {kind_label}</span>
        <span class="badge-pill badge-meta">iter {fr.iteration}</span>
        <span class="title-main">{_esc(fr.model_name)} · in {tin} · out {tout} · cache {tcache} · Σ {ttot}</span>
        <span class="title-time">{_esc(fr.request_ts.strftime("%H:%M:%S.%f")[:-3])} → {_esc(fr.output_ts.strftime("%H:%M:%S.%f")[:-3])}</span>
      </div>
      <div class="meta-line">request_id：{_esc(fr.request_id)}</div>
      {tools_summary}
      {hist_line}
      {zero_warn}
    </div>
    <div class="header-right">{child_badge}<span class="arrow">▼</span></div>
  </div>
  <div class="request-details" id="req-{idx}">
    <div class="llm-body-box">
      <div class="collapsible-header" onclick="toggleBlock('in-{idx}', this)">
        <span>模型输入 (stream_request / invoke_request)</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content active" id="in-{idx}">{_copyable_pre(_esc(fr.request_body_full or "(无请求体)"))}</div>
      <div class="collapsible-header" onclick="toggleBlock('think-{idx}', this)">
        <span>思考过程 (full reasoning_delta × {fr.reasoning_batches})</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="think-{idx}">{_copyable_pre(_esc(fr.reasoning_full or ""))}</div>
      <div class="collapsible-header" onclick="toggleBlock('out-{idx}', this)">
        <span>模型输出 (stream_output / invoke_output)</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="out-{idx}">{_copyable_pre(_esc(fr.output_body_excerpt or ""))}</div>
      {hist_think}
      <div class="collapsible-header" onclick="toggleBlock('tools-{idx}', this)">
        <span>tool_calls JSON</span><span class="arrow">▼</span>
      </div>
      <div class="collapsible-content" id="tools-{idx}">{_copyable_pre(_json_pretty(fr.tools))}</div>
    </div>
  </div>
</div>
"""


def _tool_window_block(idx: int, g: ToolGap) -> str:
    child_cls = " timeline-child" if g.is_child_session else ""
    title = g.tools_triggered or "(no tool_calls)"
    note = "墙钟：上一轮 output 结束 → 下一轮 request 开始（full 工具窗口）"
    if g.duration_sec <= 0:
        note += " · 末轮无后续 request"
    blocks = "".join(_copyable_pre(_json_pretty(t)) for t in g.detail_tools)
    if not blocks:
        blocks = '<div class="meta-line">无结构化 tool_calls。</div>'
    return f"""
<div class="timeline-item tool-item{child_cls}">
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


def _zero_inference_table(
    full_rounds: list[FullLLMRound],
    matches: dict[int, _HistMatch],
) -> str:
    rows: list[str] = []
    for i, fr in enumerate(full_rounds):
        h = matches.get(i, _HistMatch(None)).history
        if h is None or h.inference_sec > 0.001:
            continue
        diag = diagnose_zero_inference(h, fr)
        rows.append(
            "<tr>"
            f"<td>{i + 1}</td>"
            f"<td class='mono'>{_esc(fr.request_ts.strftime('%H:%M:%S'))}</td>"
            f"<td>{_esc(fr.model_name)}</td>"
            f"<td class='num'>{h.ttft_sec:.3f}</td>"
            f"<td class='num'>{h.inference_sec:.3f}</td>"
            f"<td class='num'>{h.output_tokens or 0}</td>"
            f"<td>{_esc(fr.kind)}</td>"
            f"<td>{_esc(diag)}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
<div class="section">
  <div class="section-title">推理时间 = 0 的模型调用（诊断）</div>
  <p class="meta-line" style="margin-bottom:10px">
    推理时间 = usage_metadata 时刻 − 首条 reasoning/delta 时刻。为 0 通常表示<strong>无流式间隔</strong>（同刻结束）或<strong>无 output token</strong>，而非一定“模型无响应”。
  </p>
  <table class="cmp-table">
    <thead><tr>
      <th>#</th><th>时间</th><th>模型</th><th>TTFT</th><th>推理</th><th>out tok</th><th>full</th><th>原因</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>
"""


def _report_links(report_paths: dict[str, str]) -> str:
    labels = {
        "history_html": "History",
        "full_html": "Full",
        "llm_latency_html": "LLM 时延",
        "fusion_html": "交叉校对",
        "e2e_flowchart_html": "泳道图",
    }
    items = []
    for key, label in labels.items():
        path = report_paths.get(key)
        if path and Path(path).is_file():
            name = Path(path).name
            items.append(f'<li><a href="{_esc(name)}">{_esc(label)}</a> <code>{_esc(name)}</code></li>')
    if not items:
        return ""
    return (
        '<div class="section"><div class="section-title">独立分报告</div>'
        f"<ul class=\"report-links\">{''.join(items)}</ul></div>"
    )


def render_unified_html(
    session_id: str,
    rounds: list[LLMRound],
    tools: list[ToolExecution],
    extras: HistoryExtras,
    *,
    full_data: FullSessionData,
    report_paths: dict[str, str] | None = None,
    guide_href: str = "GUIDE.md",
    guide_label: str = "解析规则说明",
    extended_analysis: bool = False,
) -> str:
    if not full_data:
        raise ValueError("融合总览需要 full 日志数据")

    tot_full = aggregate_full_stats(full_data.rounds, full_data.gaps)
    tot_hist = aggregate_stats(rounds, tools)
    lat = aggregate_llm_latency(rounds)

    merged = merge_full_timeline(full_data.rounds, full_data.gaps)
    used: set[int] = set()
    matches: dict[int, _HistMatch] = {}
    for i, fr in enumerate(full_data.rounds):
        matches[i] = match_history_to_full(fr, rounds, used)

    task_sec = _merged_task_sec(rounds, tools, full_data)
    cache_sum = _sum_matched_cache(full_data.rounds, matches)

    all_ts: list[datetime] = []
    for r in rounds:
        all_ts.extend([r.request_ts, r.output_ts])
    for t in tools:
        all_ts.extend([t.start_ts, t.end_ts])
    for r in full_data.rounds:
        all_ts.extend([r.request_ts, r.output_ts])
    for g in full_data.gaps:
        all_ts.extend([g.after_output_ts, g.next_request_ts])
    naive_ts = [_naive(t) for t in all_ts]
    session_start = min(naive_ts) if naive_ts else local_now()
    session_end = max(naive_ts) if naive_ts else local_now()

    phases = build_execution_phases(extras.todo_timeline, session_start, session_end)
    buckets = partition_full_by_phase(merged, phases)

    full_by_id = {id(r): i for i, r in enumerate(full_data.rounds)}

    def _render_item(idx: int, kind: str, obj: Any) -> str:
        if kind == "llm":
            fi = full_by_id.get(id(obj), 0)
            block = _llm_block(idx, obj, matches.get(fi, _HistMatch(None)))
        else:
            block = _tool_window_block(idx, obj)
        depth = len(obj.child_path) if obj.is_child_session else 0
        return wrap_item_with_session_depth(block, depth, obj.is_child_session)

    phased_timeline = render_phased_timeline_shell(phases, buckets, _esc, _render_item)

    files_rows = "".join(
        f"<tr><td>{_esc(Path(s.path).name)}</td><td>{s.trace_lines}</td>"
        f"<td>{_esc(s.first_ts)}</td><td>{_esc(s.last_ts)}</td></tr>"
        for s in full_data.source_files
    )

    top_section = render_top_consumers_html(
        compute_top_llm(full_data.rounds, 5),
        compute_top_tool_windows(full_data.gaps, 5),
        _esc,
        tool_panel_title="工具执行窗口（full）",
    )
    session_tree = render_session_hierarchy_html(
        collect_session_lanes(merged, session_id),
        session_id,
        _esc,
    )

    todo_html = ""
    if extras.todo_timeline and getattr(extras.todo_timeline, "tasks", None):
        todo_html = render_todo_section_html(extras.todo_timeline, _esc) or ""

    zero_table = _zero_inference_table(full_data.rounds, matches)
    links = _report_links(report_paths or {})
    ext_css = extended_report_styles() if extended_analysis else ""
    body_attr = ' class="extended-report"' if extended_analysis else ""

    overview = f"""
  {links}
  {top_section}
  {session_tree}
  {todo_html}

  <div class="section">
    <div class="section-title">数据源文件</div>
    <table class="cmp-table"><thead><tr><th>文件</th><th>TRACE 行</th><th>首条</th><th>末条</th></tr></thead>
    <tbody>{files_rows}</tbody></table>
  </div>

  <div class="section">
    <div class="section-title">融合执行时间线（full 事件 · history Todo 阶段）</div>
    <p class="meta-line" style="margin-bottom:10px">
      时间线主体与 <code>out_full</code> 一致（模型轮次 + 工具窗口）；阶段划分来自 <code>history</code> Todo。
      每条模型附 history 的 TTFT / 推理 / TPOT / tok/s；推理=0 见黄色告警与文末诊断表。
    </p>
    <div class="legend-bar">
      <span><i class="legend-dot" style="background:#2e7d32"></i> 模型 (full)</span>
      <span style="margin-left:14px"><i class="legend-dot" style="background:#e65100"></i> 工具窗口 (full)</span>
      <span style="margin-left:14px">TTFT/TPOT 来自 history</span>
    </div>
    {phased_timeline}
  </div>
  {zero_table}
"""

    if extended_analysis:
        hist_merged = merge_timeline(rounds, tools)
        hb = partition_by_phase(hist_merged, phases)
        ext_html = compose_extended_analysis(phases, hb, _esc)
        overview = wrap_extended_report(overview, ext_html, _esc)

    html_out = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>融合总览 — {_esc(session_id[:48])}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    .header {{ background: linear-gradient(135deg, #1a237e 0%, #3949ab 50%, #00838f 100%); color: #fff; padding: 22px; border-radius: 10px; margin-bottom: 18px; }}
    .header h1 {{ font-size: 1.5em; margin-bottom: 6px; }}
    .guide-link {{ color: #e1f5fe; text-decoration: underline; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .stat-card {{ background: #fff; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 3px solid #3949ab; }}
    .stat-value {{ font-size: 1.25em; font-weight: bold; color: #3949ab; }}
    .stat-label {{ font-size: 0.78em; color: #666; }}
    .section {{ background: #fff; border-radius: 10px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    .section-title {{ font-size: 1.1em; font-weight: 700; color: #333; border-bottom: 2px solid #3949ab; padding-bottom: 8px; margin-bottom: 12px; }}
    .cmp-table {{ width: 100%; border-collapse: collapse; font-size: 0.86em; }}
    .cmp-table th, .cmp-table td {{ border: 1px solid #e0e4ee; padding: 8px; text-align: left; vertical-align: top; }}
    .cmp-table th {{ background: #eef0fa; }}
    .cmp-table .num {{ text-align: right; }}
    .cmp-table .mono {{ font-family: ui-monospace, monospace; font-size: 0.9em; }}
    .timeline-item {{ border-left: 5px solid #3949ab; padding-left: 14px; margin-bottom: 14px; border-radius: 0 8px 8px 0; }}
    .llm-item {{ border-left-color: #2e7d32; background: #f1f8f4; }}
    .tool-item {{ border-left-color: #e65100; background: #fff8f0; }}
    .timeline-child {{ margin-left: 18px; }}
    .request-header {{ padding: 12px; border-radius: 8px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
    .llm-header {{ background: #e8f5e9; }}
    .llm-header:hover {{ background: #dcefdc; }}
    .tool-header {{ background: #fff3e0; }}
    .request-details {{ display: none; padding: 12px 0; }}
    .request-details.active {{ display: block; }}
    .request-id {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }}
    .collapsible-header {{ background: #f0f0f0; padding: 8px 10px; cursor: pointer; border-radius: 6px; margin: 8px 0 4px; display: flex; justify-content: space-between; }}
    .collapsible-content {{ display: none; padding: 8px; background: #f7f8fa; border-radius: 6px; max-height: 480px; overflow: auto; }}
    .collapsible-content.active {{ display: block; }}
    .badge-pill {{ display: inline-block; border-radius: 6px; padding: 2px 8px; font-size: 0.78em; font-weight: 600; margin-right: 4px; border: 1px solid transparent; }}
    .badge-time {{ background: #eef2ff; color: #4154c5; }}
    .badge-type {{ background: #e8f5e9; color: #2e7d32; }}
    .badge-tool {{ background: #fff3e0; color: #e65100; }}
    .badge-meta {{ background: #f3e5f5; color: #7b1fa2; }}
    .badge-ttft {{ background: #e3f2fd; color: #1565c0; }}
    .badge-infer {{ background: #f3e5f5; color: #6a1b9a; }}
    .badge-tpot {{ background: #fff8e1; color: #f57f17; }}
    .badge-tps {{ background: #e8f5e9; color: #2e7d32; }}
    .title-main {{ font-weight: 600; color: #333; }}
    .title-time {{ color: #777; font-size: 0.85em; }}
    .meta-line {{ font-size: 0.84em; color: #666; margin-top: 4px; word-break: break-all; }}
    .meta-line.muted {{ color: #999; }}
    .infer-zero-warn {{ background: #fff8e1; border-left: 3px solid #f9a825; padding: 6px 8px; border-radius: 4px; color: #5d4037; }}
    .src-tag {{ font-size: 0.75em; padding: 1px 6px; border-radius: 4px; margin-right: 6px; }}
    .src-tag.hist {{ background: #e8f5e9; color: #2e7d32; }}
    .pre-box {{ position: relative; }}
    .copy-btn {{ position: absolute; top: 8px; right: 8px; z-index: 2; border: 1px solid #cfd7ff; background: #eef2ff; color: #4154c5; border-radius: 4px; padding: 2px 8px; font-size: 12px; cursor: pointer; }}
    pre {{ background: #101622; color: #e6edf3; padding: 10px; border-radius: 6px; white-space: pre-wrap; word-break: break-word; font-size: 0.78rem; }}
    .legend-bar {{ margin-bottom: 12px; font-size: 0.9em; color: #555; }}
    .legend-dot {{ width: 10px; height: 10px; display: inline-block; border-radius: 2px; margin-right: 6px; }}
    .badge-orange {{ background: #fff3e0; color: #f57c00; padding: 2px 8px; border-radius: 12px; font-size: 0.78em; }}
    .report-links {{ list-style: none; padding: 0; }}
    .report-links li {{ margin: 6px 0; }}
    .report-links code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }}
    {enhanced_report_styles()}
    {ext_css}
    .extended-report .wrap {{ max-width: 100%; padding: 20px 28px; }}
  </style>
</head>
<body{body_attr}>
<div class="wrap">
  <div class="header">
    <h1>会话融合总览</h1>
    <p>session：{_esc(session_id)} · full 时间线 + history Todo 阶段 + TTFT/TPOT</p>
    <p>生成 {format_now()} · {guide_link_html(guide_href, guide_label)}</p>
  </div>

  <div class="stats-grid">
    <div class="stat-card"><div class="stat-value">{task_sec}s</div><div class="stat-label">总任务时间</div></div>
    <div class="stat-card"><div class="stat-value">{tot_full['rounds']}</div><div class="stat-label">模型轮次 (full)</div></div>
    <div class="stat-card"><div class="stat-value">{lat.get('llm_ttft_sum_sec', 0)}s</div><div class="stat-label">TTFT Σ (history)</div></div>
    <div class="stat-card"><div class="stat-value">{lat.get('llm_inference_sum_sec', 0)}s</div><div class="stat-label">推理 Σ (history)</div></div>
    <div class="stat-card"><div class="stat-value">{_fmt_tpot(lat.get('avg_tpot_sec'))}</div><div class="stat-label">平均 TPOT</div></div>
    <div class="stat-card"><div class="stat-value">{lat.get('avg_tokens_per_sec') or '—'}</div><div class="stat-label">平均 tok/s</div></div>
    <div class="stat-card"><div class="stat-value">{tot_full.get('input_tokens_sum', 0):,}</div><div class="stat-label">input Σ</div></div>
    <div class="stat-card"><div class="stat-value">{tot_full.get('output_tokens_sum', 0):,}</div><div class="stat-label">output Σ</div></div>
    <div class="stat-card"><div class="stat-value">{cache_sum:,}</div><div class="stat-label">cache Σ</div></div>
    <div class="stat-card"><div class="stat-value">{tot_hist.get('tool_calls', 0)}</div><div class="stat-label">工具 (history)</div></div>
    <div class="stat-card"><div class="stat-value">{tot_hist.get('tool_sec', 0)}s</div><div class="stat-label">工具墙钟 (history)</div></div>
  </div>

  {overview}
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
  try {{ await navigator.clipboard.writeText(pre.innerText || ''); btn.textContent = '已复制'; }}
  catch (e) {{ btn.textContent = '失败'; }}
  setTimeout(() => {{ btn.textContent = '复制'; }}, 1200);
}}
{extended_report_scripts() if extended_analysis else ""}
</script>
</body>
</html>"""
    return html_out


def write_unified_report(path: Path, html_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
