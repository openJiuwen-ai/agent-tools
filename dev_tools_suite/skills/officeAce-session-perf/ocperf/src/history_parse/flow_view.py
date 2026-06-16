"""Swimlane flow view: vertical (time-accurate) and horizontal (legacy) layouts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from history_parse.pattern_analysis import (
    EventRef,
    PhasePatternReport,
    SessionPatternSummary,
    assign_flow_tracks,
    event_ref_from_item,
)
from history_parse.report_ui import ExecutionPhase, TimelineKind, _naive

# 纵向时间轴：像素/秒（越大越易读，过长会话会限高并滚动）
_BASE_PX_PER_SEC = 18.0
_MIN_CANVAS_PX = 720
_MAX_CANVAS_PX = 18000
_RULER_W = 88
_LABEL_COL_W = 200


@dataclass(frozen=True)
class _HorizontalLaneContext:
    t0: datetime
    span_sec: float
    esc: Callable[[str], str]
    mode: str


def _rel_label(rel: str) -> str:
    return {"serial": "串行", "concurrent": "并发", "mixed": "混合"}[rel]


def _sec_offset(t: datetime, t0: datetime) -> float:
    return max(0.0, (_naive(t) - _naive(t0)).total_seconds())


def _px_per_sec(span_sec: float, event_count: int) -> float:
    if span_sec <= 0:
        return _BASE_PX_PER_SEC
    ideal_h = max(_MIN_CANVAS_PX, event_count * 44)
    capped_h = min(_MAX_CANVAS_PX, max(ideal_h, span_sec * _BASE_PX_PER_SEC))
    return capped_h / span_sec


def _format_ts(t: datetime) -> str:
    return t.strftime("%H:%M:%S.%f")[:-3]


def _event_tooltip(e: EventRef) -> str:
    tip = (
        f"{e.name} | {_format_ts(e.start)} → {_format_ts(e.end)} | "
        f"{e.duration_sec:.3f}s | {e.session_label}"
    )
    if e.exclude_from_tool_time:
        tip += " | 不计入工具墙钟"
    if e.detail:
        tip += f" | {e.detail}"
    return tip


# --- 横向泳道（旧版：时间从左到右，可横向滚动）---


def _pct_horizontal(
    start: datetime, end: datetime, t0: datetime, span_sec: float
) -> tuple[float, float]:
    if span_sec <= 0:
        return 0.0, 100.0
    left = max(0.0, (_naive(start) - _naive(t0)).total_seconds() / span_sec * 100)
    width = max(0.35, (_naive(end) - _naive(start)).total_seconds() / span_sec * 100)
    width = min(width, 100.0 - left)
    return left, width


def _render_horizontal_time_axis(
    t0: datetime, t1: datetime, esc: Callable[[str], str], n: int = 6
) -> str:
    span = max(1.0, (_naive(t1) - _naive(t0)).total_seconds())
    parts = ['<div class="flow-time-axis">']
    for i in range(n + 1):
        pct = i / n * 100
        ts = _naive(t0).timestamp() + span * i / n
        label = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        parts.append(
            f'<span class="flow-tick" style="left:{pct:.1f}%"><i></i>{esc(label)}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


def _render_horizontal_lane(
    lane_title: str,
    lane_class: str,
    events: list[EventRef],
    ctx: _HorizontalLaneContext,
) -> str:
    esc = ctx.esc
    mode = ctx.mode
    t0 = ctx.t0
    span_sec = ctx.span_sec
    if not events:
        return (
            f'<div class="flow-lane {lane_class}">'
            f'<div class="flow-lane-label">{esc(lane_title)}</div>'
            '<div class="flow-lane-body empty">（无）</div></div>'
        )

    if mode == "concurrent":
        tracks = assign_flow_tracks(events)
        max_track = max(tracks.values()) if tracks else 0
    else:
        tracks = {e.idx: 0 for e in sorted(events, key=lambda x: _naive(x.start))}
        max_track = 0

    track_count = max_track + 1
    body_h = max(72, track_count * 52 + 16)
    parts = [
        f'<div class="flow-lane {lane_class}">',
        f'<div class="flow-lane-label">{esc(lane_title)}</div>',
        f'<div class="flow-lane-body" style="min-height:{body_h}px">',
    ]
    for ti in range(track_count):
        parts.append(f'<div class="flow-track" data-track="{ti}">')
        track_events = [e for e in events if tracks.get(e.idx, 0) == ti]
        if mode == "serial":
            track_events = sorted(track_events, key=lambda e: _naive(e.start))
        for e in track_events:
            left, width = _pct_horizontal(e.start, e.end, t0, span_sec)
            bar_cls = "flow-bar-llm" if e.kind == "llm" else "flow-bar-tool"
            if e.exclude_from_tool_time:
                bar_cls += " flow-bar-agent"
            child = " flow-bar-child" if e.is_child else ""
            parts.append(
                f'<div class="flow-bar {bar_cls}{child}" style="left:{left:.2f}%;width:{width:.2f}%" '
                f'title="{esc(_event_tooltip(e))}">'
                f'<span class="flow-bar-label">{esc(e.name[:28])}</span>'
                f'<span class="flow-bar-time">{e.duration_sec:.1f}s</span></div>'
            )
        if mode == "serial" and len(track_events) > 1:
            parts.append('<div class="flow-serial-hint">串行顺序 →</div>')
        parts.append("</div>")
    parts.append("</div></div>")
    return "".join(parts)


def _render_horizontal_phase_body(
    events: list[EventRef],
    t0: datetime,
    t1: datetime,
    span_sec: float,
    esc: Callable[[str], str],
) -> str:
    llm_events = [e for e in events if e.kind == "llm"]
    tool_events = [e for e in events if e.kind == "tool"]
    parts: list[str] = []
    for mode in ("concurrent", "serial"):
        disp = "" if mode == "concurrent" else ' style="display:none"'
        mode_label = "并发" if mode == "concurrent" else "串行"
        parts.append(
            f'<div class="flow-h-mode-body" data-h-mode="{mode}"{disp}>'
            f'<p class="muted flow-h-mode-hint">横向 · {mode_label}视图</p>'
            '<div class="flow-h-scroll-wrap">'
        )
        parts.append(_render_horizontal_time_axis(t0, t1, esc))
        lane_ctx = _HorizontalLaneContext(t0=t0, span_sec=span_sec, esc=esc, mode=mode)
        parts.append(
            _render_horizontal_lane("模型 LLM", "flow-lane-llm", llm_events, lane_ctx)
        )
        parts.append(
            _render_horizontal_lane("工具 Tool", "flow-lane-tool", tool_events, lane_ctx)
        )
        parts.append("</div></div>")
    return "".join(parts)


def render_pattern_stats_html(
    summary: SessionPatternSummary,
    esc: Callable[[str], str],
) -> str:
    parts: list[str] = [
        '<div class="section extended-section" id="pattern-stats">',
        '<div class="section-title">扩展分析 · 重复 / 循环 / 并发与串行</div>',
        '<p class="section-desc">按 Todo 阶段统计（<code>spawn_subagent</code> / '
        '<code>fork_agent</code> 不计入工具墙钟 KPI；下表含 spawn 次数供对照）。'
        "「并发」指时间区间重叠，「串行」指先后不重叠；"
        "流程图「并发分轨」列数为峰值并行轨，不是 spawn 总次数。</p>",
    ]
    if not any(p.has_issues for p in summary.phases):
        parts.append('<p class="muted">各阶段未发现明显的重复或循环模式。</p>')
    for pr in summary.phases:
        if not pr.has_issues and pr.event_count == 0:
            continue
        parts.append(
            f'<div class="pattern-phase-card" id="pat-{esc(pr.phase_id)}">'
            f'<h3>{esc(pr.phase_title)}</h3>'
            f'<p class="muted">事件 {pr.event_count}（LLM {pr.llm_count} · 工具 {pr.tool_count}）</p>'
        )
        if pr.duplicates:
            parts.append('<table class="pattern-table"><thead><tr>')
            parts.append(
                "<th>类型</th><th>名称</th><th>次数</th><th>关系</th><th>说明</th></tr></thead><tbody>"
            )
            for d in pr.duplicates:
                kind_lbl = "工具" if d.kind == "tool" else "LLM"
                rel = _rel_label(d.relationship)
                parts.append(
                    f"<tr><td>{esc(kind_lbl)}</td><td><code>{esc(d.name)}</code></td>"
                    f"<td>{d.count}</td><td><span class=\"rel-badge rel-{esc(d.relationship)}\">"
                    f"{esc(rel)}</span></td><td>{esc(d.note)}</td></tr>"
                )
            parts.append("</tbody></table>")
        if pr.cycles:
            parts.append('<ul class="cycle-list">')
            for c in pr.cycles:
                seq = " → ".join(esc(x) for x in c.example_sequence)
                parts.append(
                    f"<li><strong>{esc(c.note)}</strong><br>"
                    f'<span class="muted">{seq}</span></li>'
                )
            parts.append("</ul>")
        if not pr.duplicates and not pr.cycles:
            parts.append('<p class="muted">本阶段无重复/循环告警。</p>')
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _render_time_ruler(t0: datetime, span_sec: float, px_per_sec: float, esc: Callable[[str], str]) -> str:
    canvas_h = int(span_sec * px_per_sec) + 8
    tick_step = span_sec / 12 if span_sec > 120 else max(5.0, span_sec / 8)
    if tick_step < 1:
        tick_step = 1.0
    parts = [f'<div class="flow-vruler" style="height:{canvas_h}px">']
    t = 0.0
    while t <= span_sec + 0.01:
        top = int(t * px_per_sec)
        ts = _naive(t0).timestamp() + t
        label = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        parts.append(
            f'<div class="flow-vtick" style="top:{top}px">'
            f'<span>{esc(label)}</span></div>'
        )
        t += tick_step
    parts.append("</div>")
    return "".join(parts)


def _render_vertical_canvas(
    events: list[EventRef],
    t0: datetime,
    span_sec: float,
    esc: Callable[[str], str],
    mode: str,
) -> str:
    if not events:
        return '<p class="muted">（无事件）</p>'

    px = _px_per_sec(span_sec, len(events))
    canvas_h = int(span_sec * px) + 12

    if mode == "concurrent":
        tracks = assign_flow_tracks(events)
        n_tracks = max(tracks.values(), default=0) + 1
    else:
        tracks = {e.idx: 0 for e in events}
        n_tracks = 1

    track_w_pct = 100.0 / n_tracks
    inner_w = f"calc(100% - {_LABEL_COL_W}px)"

    parts = [
        f'<div class="flow-vchart" style="min-height:{canvas_h + 40}px">',
        f'<div class="flow-vchart-grid" style="height:{canvas_h}px">',
        f'<div class="flow-vchart-ruler-col" style="width:{_RULER_W}px">',
        _render_time_ruler(t0, span_sec, px, esc),
        "</div>",
        f'<div class="flow-vchart-body-col" style="width:{inner_w}">',
        f'<div class="flow-vcanvas" style="height:{canvas_h}px">',
    ]

    for e in sorted(events, key=lambda x: (_naive(x.start), x.idx)):
        top = int(_sec_offset(e.start, t0) * px)
        raw_h = max(2.0, e.duration_sec * px)
        height = int(raw_h)
        ti = tracks.get(e.idx, 0)
        left_pct = ti * track_w_pct
        width_pct = track_w_pct - 0.4

        kind_cls = "flow-vblock-llm" if e.kind == "llm" else "flow-vblock-tool"
        if e.exclude_from_tool_time:
            kind_cls += " flow-vblock-agent"
        child_cls = " flow-vblock-child" if e.is_child else ""
        compact = " flow-vblock-compact" if height < 26 else ""

        tip = _event_tooltip(e)

        inner = ""
        if height >= 22:
            inner = (
                f'<span class="flow-vblock-name">{esc(e.name)}</span>'
                f'<span class="flow-vblock-dur">{e.duration_sec:.2f}s</span>'
            )
        else:
            inner = f'<span class="flow-vblock-name flow-vblock-name-sm">{esc(e.name)}</span>'

        parts.append(
            f'<div class="flow-vblock {kind_cls}{child_cls}{compact}" '
            f'style="top:{top}px;height:{height}px;left:{left_pct:.3f}%;width:{width_pct:.3f}%" '
            f'title="{esc(tip)}">{inner}</div>'
        )

    parts.append("</div></div></div>")
    parts.append(
        '<div class="flow-vlegend">'
        '<span class="flow-vleg flow-vleg-llm">■ LLM</span>'
        '<span class="flow-vleg flow-vleg-tool">■ 工具</span>'
        '<span class="flow-vleg flow-vleg-agent">■ spawn/fork（仅展示）</span>'
        '<span class="muted">纵轴 = 实际时间；块高度 ∝ 耗时；无时间重叠则不叠在同一时刻</span>'
        "</div>"
    )
    parts.append("</div>")
    return "".join(parts)


def render_flow_view_html(
    phases: list[ExecutionPhase],
    buckets: dict[str, list[tuple[int, TimelineKind, Any]]],
    pattern_summary: SessionPatternSummary,
    esc: Callable[[str], str],
) -> str:
    phase_nav: list[str] = ['<div class="flow-nav" id="flow-phase-nav">']
    panels: list[str] = ['<div class="flow-panels" id="flow-panels">']

    pat_by_id = {p.phase_id: p for p in pattern_summary.phases}
    rendered = 0

    for phase in phases:
        items = buckets.get(phase.phase_id, [])
        if not items:
            continue
        events = [event_ref_from_item(idx, kind, obj) for idx, kind, obj in items]
        if not events:
            continue

        t0 = min(_naive(e.start) for e in events)
        t1 = max(_naive(e.end) for e in events)
        span = max(1.0, (_naive(t1) - _naive(t0)).total_seconds())
        pid = esc(phase.phase_id)
        active = " active" if rendered == 0 else ""
        rendered += 1

        phase_nav.append(
            f'<button type="button" class="flow-phase-btn{active}" '
            f'data-flow-phase="{pid}" onclick="flowSelectPhase(this)">{esc(phase.title)}</button>'
        )

        pr = pat_by_id.get(phase.phase_id)
        badge = ' <span class="flow-badge-warn">!</span>' if pr and pr.has_issues else ""
        spawn_n = sum(1 for e in events if e.kind == "tool" and e.exclude_from_tool_time)
        spawn_note = f" · 子Agent派发 ×{spawn_n}" if spawn_n else ""

        panels.append(
            f'<div class="flow-panel{active}" id="flow-panel-{pid}" data-phase="{pid}">'
            f'<h3 class="flow-panel-title">{esc(phase.title)}{badge}</h3>'
            f'<p class="muted flow-panel-range">'
            f'{esc(_format_ts(t0))} – {esc(_format_ts(t1))} · 跨度 {span:.1f}s · '
            f"{len(events)} 项{spawn_note} · 纵向 {int(span * _px_per_sec(span, len(events)))}px 时间轴</p>"
        )
        # 纵向（默认）
        panels.append('<div class="flow-orient-wrap flow-orient-vertical" data-orient="vertical">')
        panels.append(
            '<div class="flow-mode-tabs flow-v-mode-tabs">'
            '<button type="button" class="flow-mode-btn active" data-mode="timeline" '
            f'onclick="flowSelectVMode(this, \'{pid}\')">时间轴（严格按执行时间）</button>'
            '<button type="button" class="flow-mode-btn" data-mode="concurrent" '
            f'onclick="flowSelectVMode(this, \'{pid}\')">并发分轨（仅重叠时分列）</button>'
            "</div>"
        )
        for mode in ("timeline", "concurrent"):
            disp = "" if mode == "timeline" else ' style="display:none"'
            panels.append(f'<div class="flow-v-mode-body" data-v-mode="{mode}"{disp}>')
            panels.append(
                '<div class="flow-vchart-scroll">'
                + _render_vertical_canvas(
                    events,
                    t0,
                    span,
                    esc,
                    "serial" if mode == "timeline" else "concurrent",
                )
                + "</div></div>"
            )
        panels.append("</div>")

        # 横向（旧版泳道）
        panels.append(
            '<div class="flow-orient-wrap flow-orient-horizontal" data-orient="horizontal" '
            'style="display:none">'
        )
        panels.append(
            '<div class="flow-mode-tabs flow-h-mode-tabs">'
            '<button type="button" class="flow-h-mode-btn active" data-h-mode="concurrent" '
            f'onclick="flowSelectHMode(this, \'{pid}\')">并发视图（重叠分轨）</button>'
            '<button type="button" class="flow-h-mode-btn" data-h-mode="serial" '
            f'onclick="flowSelectHMode(this, \'{pid}\')">串行视图（时间顺序）</button>'
            "</div>"
        )
        panels.append(_render_horizontal_phase_body(events, t0, t1, span, esc))
        panels.append("</div>")

        if pr and (pr.duplicates or pr.cycles):
            panels.append('<div class="flow-inline-summary">')
            for d in pr.duplicates[:8]:
                panels.append(
                    f'<span class="flow-chip rel-{esc(d.relationship)}">'
                    f'{esc(d.name)} ×{d.count} {_rel_label(d.relationship)}</span>'
                )
            panels.append("</div>")
        panels.append("</div>")

    if rendered == 0:
        phase_nav.append('<span class="muted">本报告无可用阶段事件。</span>')
    phase_nav.append("</div>")
    panels.append("</div>")

    layout_tabs = (
        '<div class="flow-layout-tabs" id="flow-layout-tabs">'
        '<span class="flow-layout-label">流程图方向：</span>'
        '<button type="button" class="flow-layout-btn active" data-layout="vertical" '
        'onclick="flowSetLayout(\'vertical\')">纵向（严格时间）</button>'
        '<button type="button" class="flow-layout-btn" data-layout="horizontal" '
        'onclick="flowSetLayout(\'horizontal\')">横向（泳道概览）</button>'
        "</div>"
        '<p class="section-desc flow-layout-desc" id="flow-layout-desc">'
        "<strong>纵向</strong>：自上而下为真实时间，块高度与位置按耗时比例；适合看先后与细节。"
        "<strong>横向</strong>：时间从左到右，LLM/工具分行展示，可横向滚动；适合概览并发与阶段跨度。"
        "</p>"
    )

    return (
        '<div class="section extended-section flow-section" id="flow-view">'
        '<div class="section-title">扩展分析 · 执行流程</div>'
        + layout_tabs
        + "".join(phase_nav)
        + "".join(panels)
        + "</div>"
    )


def extended_report_styles() -> str:
    return """
.extended-section { margin-top: 2rem; }
.section-desc { color: #64748b; font-size: 0.92rem; margin: 0.4rem 0 1rem; line-height: 1.5; }
.pattern-phase-card {
  background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
  padding: 1rem 1.25rem; margin-bottom: 1rem;
}
.pattern-phase-card h3 { margin: 0 0 0.35rem; font-size: 1.05rem; }
.pattern-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 0.5rem; }
.pattern-table th, .pattern-table td {
  border: 1px solid #e2e8f0; padding: 0.45rem 0.6rem; text-align: left;
}
.rel-badge { padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
.rel-serial { background: #dbeafe; color: #1e40af; }
.rel-concurrent { background: #ffedd5; color: #c2410c; }
.rel-mixed { background: #fce7f3; color: #9d174d; }
.cycle-list { margin: 0.5rem 0 0 1.2rem; }
.cycle-list li { margin-bottom: 0.5rem; }

.report-view-nav {
  display: flex; gap: 0.5rem; flex-wrap: wrap;
  position: sticky; top: 0; z-index: 100;
  background: #fff; padding: 0.75rem 0; margin-bottom: 1rem;
  border-bottom: 2px solid #e2e8f0;
}
.report-view-btn {
  padding: 0.55rem 1.1rem; border: 1px solid #cbd5e1; border-radius: 8px;
  background: #f8fafc; cursor: pointer; font-size: 0.95rem; font-weight: 500;
}
.report-view-btn.active { background: #0f766e; color: #fff; border-color: #0f766e; }
.report-view-pane { display: none; }
.report-view-pane.active { display: block; }

.flow-section { min-width: 0; }
.flow-nav {
  display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1rem;
  max-height: 160px; overflow-y: auto; padding: 0.25rem;
}
.flow-phase-btn {
  padding: 0.4rem 0.75rem; border-radius: 6px; border: 1px solid #cbd5e1;
  background: #fff; cursor: pointer; font-size: 0.85rem;
}
.flow-phase-btn.active { background: #0369a1; color: #fff; border-color: #0369a1; }
.flow-badge-warn { color: #dc2626; font-weight: bold; }
.flow-panels { width: 100%; }
.flow-panel { display: none; }
.flow-panel.active { display: block; }
.flow-panel-title { margin: 0 0 0.25rem; }
.flow-panel-range { margin: 0 0 0.75rem; }
.flow-mode-tabs { display: flex; gap: 0.5rem; margin-bottom: 0.75rem; flex-wrap: wrap; }
.flow-mode-btn {
  padding: 0.45rem 0.9rem; border: 1px solid #94a3b8; border-radius: 6px;
  background: #f1f5f9; cursor: pointer; font-size: 0.88rem;
}
.flow-mode-btn.active { background: #1e293b; color: #fff; }

.flow-layout-tabs {
  display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem;
  margin-bottom: 0.75rem; padding: 0.6rem 0.85rem;
  background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px;
}
.flow-layout-label { font-weight: 600; font-size: 0.9rem; color: #0c4a6e; }
.flow-layout-btn {
  padding: 0.45rem 0.95rem; border: 1px solid #7dd3fc; border-radius: 6px;
  background: #fff; cursor: pointer; font-size: 0.88rem;
}
.flow-layout-btn.active { background: #0369a1; color: #fff; border-color: #0369a1; }
.flow-layout-desc { margin-top: -0.25rem; margin-bottom: 1rem; }

.flow-orient-wrap { margin-top: 0.5rem; }
.flow-h-mode-tabs, .flow-v-mode-tabs { margin-bottom: 0.5rem; }
.flow-h-mode-btn {
  padding: 0.45rem 0.9rem; border: 1px solid #94a3b8; border-radius: 6px;
  background: #f1f5f9; cursor: pointer; font-size: 0.88rem;
}
.flow-h-mode-btn.active { background: #1e293b; color: #fff; }
.flow-h-mode-hint { margin: 0 0 0.5rem; font-size: 0.82rem; }

.flow-h-scroll-wrap {
  overflow-x: auto; overflow-y: visible;
  padding-bottom: 1rem; border: 1px solid #e2e8f0;
  border-radius: 8px; background: #fafbfc; padding: 12px;
}
.flow-time-axis {
  position: relative; height: 28px; margin: 0 100px 0.5rem 0;
  min-width: 900px; border-bottom: 1px solid #cbd5e1;
}
.flow-tick {
  position: absolute; transform: translateX(-50%);
  font-size: 0.75rem; color: #64748b; white-space: nowrap;
}
.flow-tick i {
  display: block; width: 1px; height: 8px; background: #94a3b8; margin: 0 auto 2px;
}
.flow-lane {
  display: grid; grid-template-columns: 100px 1fr;
  gap: 0.75rem; margin-bottom: 1.25rem; min-width: 1000px;
}
.flow-lane-label {
  font-weight: 600; font-size: 0.9rem; padding-top: 0.5rem;
  color: #334155; text-align: right;
}
.flow-lane-body {
  position: relative; background: #f8fafc;
  border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 4px;
}
.flow-lane-body.empty { padding: 12px; color: #94a3b8; min-height: 48px; }
.flow-track {
  position: relative; height: 44px; margin-bottom: 6px;
  border-bottom: 1px dashed #e2e8f0;
}
.flow-track:last-child { border-bottom: none; margin-bottom: 0; }
.flow-bar {
  position: absolute; top: 4px; height: 36px; border-radius: 6px;
  padding: 4px 8px; box-sizing: border-box; overflow: hidden;
  display: flex; flex-direction: column; justify-content: center;
  font-size: 0.72rem; line-height: 1.2; cursor: default;
  border: 1px solid rgba(0,0,0,0.12); min-width: 48px;
}
.flow-bar-llm { background: linear-gradient(135deg, #86efac, #22c55e); color: #14532d; }
.flow-bar-tool { background: linear-gradient(135deg, #fdba74, #f97316); color: #7c2d12; }
.flow-bar-agent {
  background: linear-gradient(135deg, #e9d5ff, #c4b5fd); color: #4c1d95;
  border-style: dashed;
}
.flow-bar-child { box-shadow: inset 0 0 0 2px #6366f1; }
.flow-bar-label { font-weight: 600; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; }
.flow-bar-time { opacity: 0.85; font-size: 0.68rem; }
.flow-serial-hint {
  position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
  font-size: 0.75rem; color: #64748b; pointer-events: none;
}

#view-flow .flow-scroll-wrap {
  overflow: visible; max-width: none; padding: 0;
}
.flow-vchart-scroll {
  overflow: auto; max-height: 85vh; border: 1px solid #e2e8f0;
  border-radius: 10px; background: #fafbfc; padding: 12px;
}
.flow-vchart { width: 100%; min-width: 720px; }
.flow-vchart-grid {
  display: flex; flex-direction: row; align-items: flex-start;
  position: relative;
}
.flow-vchart-ruler-col { flex-shrink: 0; position: relative; }
.flow-vruler { position: relative; border-right: 2px solid #94a3b8; }
.flow-vtick {
  position: absolute; left: 0; right: 4px;
  transform: translateY(-50%);
  font-size: 0.72rem; color: #64748b; text-align: right; padding-right: 6px;
}
.flow-vchart-body-col { flex: 1; position: relative; min-width: 0; }
.flow-vcanvas {
  position: relative; width: 100%;
  background: repeating-linear-gradient(
    to bottom,
    transparent,
    transparent 59px,
    #e8ecf1 59px,
    #e8ecf1 60px
  );
  border-radius: 6px;
}
.flow-vblock {
  position: absolute; box-sizing: border-box;
  border-radius: 5px; border: 1px solid rgba(0,0,0,0.15);
  padding: 3px 8px; overflow: hidden;
  display: flex; flex-direction: column; justify-content: center;
  font-size: 0.78rem; line-height: 1.25; cursor: default;
  z-index: 2;
}
.flow-vblock-llm {
  background: linear-gradient(90deg, #bbf7d0, #4ade80); color: #14532d;
}
.flow-vblock-tool {
  background: linear-gradient(90deg, #fed7aa, #fb923c); color: #7c2d12;
}
.flow-vblock-agent {
  background: linear-gradient(90deg, #e9d5ff, #c4b5fd); color: #4c1d95;
  border-style: dashed;
}
.flow-vblock-child { box-shadow: inset 0 0 0 2px #4f46e5; }
.flow-vblock-compact { padding: 1px 4px; font-size: 0.68rem; }
.flow-vblock-name { font-weight: 700; word-break: break-all; }
.flow-vblock-name-sm { font-size: 0.65rem; }
.flow-vblock-dur { opacity: 0.9; font-size: 0.72rem; }
.flow-vlegend {
  margin-top: 10px; display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.82rem;
}
.flow-vleg-llm { color: #15803d; }
.flow-vleg-tool { color: #c2410c; }
.flow-vleg-agent { color: #6d28d9; }
.flow-inline-summary { margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.35rem; }
.flow-chip {
  padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.78rem;
  background: #e2e8f0;
}
.flow-chip.rel-concurrent { background: #ffedd5; }
.flow-chip.rel-serial { background: #dbeafe; }
.flow-chip.rel-mixed { background: #fce7f3; }
"""


def extended_report_scripts() -> str:
    return """
function reportShowView(name) {
  document.querySelectorAll('.report-view-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.report-view-btn').forEach(b => b.classList.remove('active'));
  const pane = document.getElementById('view-' + name);
  const btn = document.querySelector('.report-view-btn[data-view="' + name + '"]');
  if (pane) pane.classList.add('active');
  if (btn) btn.classList.add('active');
}
function flowSelectPhase(btn) {
  const id = btn.getAttribute('data-flow-phase');
  document.querySelectorAll('.flow-phase-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.flow-panel').forEach(p => {
    p.classList.toggle('active', p.getAttribute('data-phase') === id);
  });
}
function flowSetLayout(layout) {
  document.querySelectorAll('.flow-layout-btn').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-layout') === layout);
  });
  document.querySelectorAll('#flow-view .flow-orient-wrap').forEach(w => {
    w.style.display = w.getAttribute('data-orient') === layout ? 'block' : 'none';
  });
  const desc = document.getElementById('flow-layout-desc');
  if (desc) {
    desc.style.display = layout === 'vertical' ? 'block' : 'block';
  }
}
function flowSelectVMode(btn, phaseId) {
  const mode = btn.getAttribute('data-mode');
  const panel = document.getElementById('flow-panel-' + phaseId);
  if (!panel) return;
  const wrap = panel.querySelector('.flow-orient-vertical');
  if (!wrap) return;
  wrap.querySelectorAll('.flow-v-mode-tabs .flow-mode-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  wrap.querySelectorAll('.flow-v-mode-body').forEach(body => {
    body.style.display = body.getAttribute('data-v-mode') === mode ? 'block' : 'none';
  });
}
function flowSelectHMode(btn, phaseId) {
  const mode = btn.getAttribute('data-h-mode');
  const panel = document.getElementById('flow-panel-' + phaseId);
  if (!panel) return;
  const wrap = panel.querySelector('.flow-orient-horizontal');
  if (!wrap) return;
  wrap.querySelectorAll('.flow-h-mode-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  wrap.querySelectorAll('.flow-h-mode-body').forEach(body => {
    body.style.display = body.getAttribute('data-h-mode') === mode ? 'block' : 'none';
  });
}
"""


def compose_extended_analysis(
    phases: list[ExecutionPhase],
    buckets: dict[str, list[tuple[int, TimelineKind, Any]]],
    esc: Callable[[str], str],
) -> str:
    from history_parse.pattern_analysis import analyze_session_patterns

    summary = analyze_session_patterns(phases, buckets)
    return render_pattern_stats_html(summary, esc) + render_flow_view_html(
        phases, buckets, summary, esc
    )


def wrap_extended_report(
    overview_html: str,
    extended_parts: str,
    esc: Callable[[str], str],
) -> str:
    nav = (
        '<nav class="report-view-nav">'
        '<button type="button" class="report-view-btn active" data-view="overview" '
        'onclick="reportShowView(\'overview\')">总览报告</button>'
        '<button type="button" class="report-view-btn" data-view="flow" '
        'onclick="reportShowView(\'flow\')">执行流程</button>'
        "</nav>"
    )
    return (
        nav
        + '<div id="view-overview" class="report-view-pane active">'
        + overview_html
        + "</div>"
        + '<div id="view-flow" class="report-view-pane">'
        + extended_parts
        + "</div>"
    )
