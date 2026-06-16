"""End-to-end session flowchart artifacts (Mermaid + vector-style HTML timeline)."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from history_parse.mermaid_validate import sanitize_mermaid_label
from history_parse.pattern_analysis import EventRef, assign_flow_tracks
from history_parse.report_ui import TimelineKind

_MAX_MERMAID_NODES = 180
_PX_PER_SEC = 2.2
_MIN_BAR_PX = 6
_MAX_BAR_PX = 320
_TRACK_W = 168
_TIME_PX_PER_SEC = 2.8
_MIN_EVT_H = 22
_CANVAS_PAD = 56

FlowLink = Literal["start", "serial", "parallel", "retry"]


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _event_kind(raw: object) -> TimelineKind:
    return "llm" if raw == "llm" else "tool"


def _llm_metrics_subtitle(ev: dict[str, Any]) -> str:
    """Short LLM latency line for flowchart blocks."""
    if ev.get("kind") != "llm":
        return ""
    lines: list[str] = []
    if ev.get("ttft_sec") is not None:
        lines.append(f"TTFT {float(ev['ttft_sec']):.1f}s")
    if ev.get("inference_sec") is not None:
        lines.append(f"推 {float(ev['inference_sec']):.1f}s")
    if ev.get("tpot_sec") is not None:
        lines.append(f"TPOT {float(ev['tpot_sec']) * 1000:.0f}ms")
    if ev.get("tokens_per_sec") is not None:
        lines.append(f"{float(ev['tokens_per_sec']):.0f} tok/s")
    tok = f"in{ev.get('input_tokens') or 0}/out{ev.get('output_tokens') or 0}"
    if ev.get("cache_tokens"):
        tok += f"/c{ev['cache_tokens']}"
    lines.append(tok)
    return " · ".join(lines)


def _safe_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)[:48]


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def collect_e2e_events(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten phases[].events in chronological order with phase + flags."""
    out: list[dict[str, Any]] = []
    for phase in bundle.get("phases") or []:
        fail_ids = {f.get("tool_call_id") for f in phase.get("tool_failures") or []}
        dup_names = {d.get("name") for d in phase.get("duplicates") or [] if d.get("count", 0) >= 2}
        for ev in phase.get("events") or []:
            item = dict(ev)
            item["phase_title"] = phase.get("title") or ""
            item["phase_id"] = phase.get("phase_id") or ""
            item["is_failure"] = (
                ev.get("kind") == "tool"
                and (ev.get("tool_call_id") in fail_ids or _tool_failed(ev))
            )
            item["is_duplicate_name"] = ev.get("name") in dup_names
            item["is_spawn"] = bool(ev.get("exclude_from_tool_kpi"))
            out.append(item)
    out.sort(key=lambda e: (e.get("start") or "", e.get("idx", 0)))
    return out


def _tool_failed(ev: dict[str, Any]) -> bool:
    d = (ev.get("detail") or "").lower()
    return any(k in d for k in ("error", "failed", "exception", "timeout"))


def _event_start(ev: dict[str, Any]) -> datetime | None:
    return _parse_iso(ev.get("start") or "")


def _event_end(ev: dict[str, Any]) -> datetime | None:
    end = _parse_iso(ev.get("end") or "")
    if end:
        return end
    start = _event_start(ev)
    if start is None:
        return None
    return start + timedelta(seconds=float(ev.get("duration_sec") or 0))


def _events_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    sa, ea = _event_start(a), _event_end(a)
    sb, eb = _event_start(b), _event_end(b)
    if not all((sa, ea, sb, eb)):
        return False
    sa, ea, sb, eb = (_naive_dt(sa), _naive_dt(ea), _naive_dt(sb), _naive_dt(eb))
    return sa < eb and sb < ea


def _naive_dt(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _annotate_consecutive_repeat(events: list[dict[str, Any]]) -> None:
    prev_name = ""
    prev_kind = ""
    streak = 0
    for ev in events:
        name = ev.get("name") or ""
        kind = ev.get("kind") or ""
        if kind == prev_kind and name == prev_name and name:
            streak += 1
            ev["repeat_index"] = streak
        else:
            streak = 0
            ev["repeat_index"] = 0
        prev_name, prev_kind = name, kind


def annotate_parallel_serial(events: list[dict[str, Any]]) -> None:
    """Assign swimlane track + link kind (serial / parallel / retry) from timestamps."""
    refs: list[EventRef] = []
    for i, ev in enumerate(events):
        start = _event_start(ev)
        end = _event_end(ev)
        if not start or not end:
            continue
        refs.append(
            EventRef(
                idx=i,
                kind=_event_kind(ev.get("kind")),
                name=ev.get("name") or "?",
                start=start,
                end=end,
                duration_sec=float(ev.get("duration_sec") or 0),
                exclude_from_tool_time=bool(ev.get("exclude_from_tool_kpi")),
            )
        )
    tracks = assign_flow_tracks(refs) if refs else {}
    for r in refs:
        events[r.idx]["flow_track"] = tracks.get(r.idx, 0)

    for i, ev in enumerate(events):
        if i == 0:
            ev["flow_link"] = "start"
            continue
        prev = events[i - 1]
        if _events_overlap(prev, ev):
            ev["flow_link"] = "parallel"
        elif ev.get("repeat_index", 0) > 0:
            ev["flow_link"] = "retry"
        else:
            ev["flow_link"] = "serial"

    max_track = max((e.get("flow_track", 0) for e in events), default=0)
    for ev in events:
        ev["flow_max_track"] = max_track


def prepare_e2e_events(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    events = collect_e2e_events(bundle)
    _annotate_consecutive_repeat(events)
    annotate_parallel_serial(events)
    return events


def _cluster_by_parallel(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group events: overlapping window = one parallel cluster, else singleton."""
    if not events:
        return []
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [events[0]]
    for ev in events[1:]:
        if any(_events_overlap(ev, x) for x in current):
            current.append(ev)
        else:
            clusters.append(current)
            current = [ev]
    clusters.append(current)
    return clusters


def _link_mermaid(prev_id: str | None, curr_id: str, ev: dict[str, Any]) -> str | None:
    if not prev_id:
        return None
    link = ev.get("flow_link") or "serial"
    if link == "parallel":
        return f"  {prev_id} -.并行.-> {curr_id}"
    if link == "retry":
        return f"  {prev_id} -.重复.-> {curr_id}"
    return f"  {prev_id} --> {curr_id}"


def build_e2e_mermaid(bundle: dict[str, Any], *, max_nodes: int = _MAX_MERMAID_NODES) -> str:
    events = prepare_e2e_events(bundle)
    if not events:
        return "flowchart TD\n  empty[无事件]\n"

    lines = [
        "flowchart TD",
        "  classDef llm fill:#bbf7d0,stroke:#15803d,stroke-width:2px",
        "  classDef tool fill:#fed7aa,stroke:#c2410c,stroke-width:2px",
        "  classDef fail fill:#fecaca,stroke:#dc2626,stroke-width:3px",
        "  classDef agent fill:#e9d5ff,stroke:#7c3aed,stroke-dasharray:5 5",
        "  classDef slow fill:#fff9c4,stroke:#f9a825,stroke-width:3px",
        "  classDef phase fill:#e3f2fd,stroke:#1565c0",
        "  classDef parCluster fill:#e3f2fd,stroke:#1565c0,stroke-dasharray:4 4",
    ]

    durations = [float(e.get("duration_sec") or 0) for e in events]
    slow_thresh = sorted(durations, reverse=True)[min(4, len(durations) - 1)] if durations else 9999.0
    if slow_thresh < 30:
        slow_thresh = 30.0

    truncated = len(events) > max_nodes
    if truncated:
        lines.append(f'  note_trunc["显示前 {max_nodes} / {len(events)} 个事件 — 完整时间轴见 execution_flowchart.html"]')

    show = events[:max_nodes]
    clusters = _cluster_by_parallel(show)
    prev_exit: str | None = None
    prev_phase: str | None = None
    phase_subgraph_open = False
    node_seq = 0

    def _emit_node(ev: dict[str, Any], local_i: int) -> str:
        nonlocal node_seq
        nid = f"e{node_seq}"
        node_seq += 1
        dur = float(ev.get("duration_sec") or 0)
        name = sanitize_mermaid_label(ev.get("name") or "?", max_len=24)
        kind = ev.get("kind") or "?"
        tags: list[str] = []
        if ev.get("flow_link") == "parallel":
            tags.append("并发")
        elif ev.get("flow_link") == "serial":
            tags.append("串行")
        if ev.get("is_failure"):
            tags.append("失败")
        if ev.get("repeat_index", 0) > 0:
            tags.append(f"重复×{ev['repeat_index'] + 1}")
        if ev.get("is_spawn"):
            tags.append("子Agent")
        if dur >= slow_thresh:
            tags.append("慢")
        label = f"{kind.upper()} {name}<br/>{dur:.1f}s"
        if kind == "llm":
            sub = _llm_metrics_subtitle(ev)
            if sub:
                label += f"<br/><small>{sanitize_mermaid_label(sub, max_len=48)}</small>"
        if tags:
            label += f"<br/><small>{' · '.join(tags)}</small>"
        lines.append(f'  {nid}["{label}"]')
        if kind == "llm":
            cls = "slow" if dur >= slow_thresh else "llm"
        elif ev.get("is_spawn"):
            cls = "agent"
        elif ev.get("is_failure"):
            cls = "fail"
        elif dur >= slow_thresh:
            cls = "slow"
        else:
            cls = "tool"
        lines.append(f"  class {nid} {cls}")
        return nid

    for ci, cluster in enumerate(clusters):
        phase = sanitize_mermaid_label(cluster[0].get("phase_title") or "", max_len=36)
        if phase and cluster[0].get("phase_id") != prev_phase:
            if phase_subgraph_open:
                lines.append("  end")
            pid = _safe_id(cluster[0].get("phase_id") or f"p{ci}")
            lines.append(f'  subgraph sg_{pid}["{phase}"]')
            phase_subgraph_open = True
            prev_phase = cluster[0].get("phase_id")
            prev_exit = None

        if len(cluster) > 1:
            lines.append(f'  subgraph par_{ci}["并发 ×{len(cluster)}"]')
            lines.append("    direction LR")
            par_ids: list[str] = []
            for ev in cluster:
                par_ids.append(_emit_node(ev, ci))
            lines.append("  end")
            lines.append(f"  class par_{ci} parCluster")
            cluster_entry = par_ids[0]
            if prev_exit:
                ln = _link_mermaid(prev_exit, cluster_entry, cluster[0])
                if ln:
                    lines.append(ln)
            ends = [_event_end(e) for e in cluster]
            exit_i = ends.index(max(e for e in ends if e))
            prev_exit = par_ids[exit_i]
        else:
            ev = cluster[0]
            nid = _emit_node(ev, ci)
            ln = _link_mermaid(prev_exit, nid, ev)
            if ln:
                lines.append(ln)
            prev_exit = nid

    if phase_subgraph_open:
        lines.append("  end")

    if truncated and events:
        lines.append("  note_trunc --> e0")

    return "\n".join(lines) + "\n"


def _layout_time_canvas(events: list[dict[str, Any]]) -> tuple[float, float, int]:
    """Return (canvas_width, canvas_height, max_track)."""
    starts = [_event_start(e) for e in events]
    ends = [_event_end(e) for e in events]
    valid = [(s, e) for s, e in zip(starts, ends) if s and e]
    if not valid:
        return 800.0, 400.0, 0
    t0 = min(_naive_dt(s) for s, _ in valid)
    t1 = max(_naive_dt(e) for _, e in valid)
    span = max(1.0, (t1 - t0).total_seconds())
    max_track = max((e.get("flow_track", 0) for e in events), default=0)
    w = _CANVAS_PAD * 2 + (max_track + 1) * _TRACK_W
    h = _CANVAS_PAD * 2 + span * _TIME_PX_PER_SEC
    h = min(max(h, 320.0), 24000.0)
    return w, h, max_track


def _connector_svg(
    x1: float, y1: float, x2: float, y2: float, link: str
) -> str:
    if link == "parallel":
        return (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#1565c0" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#arr-par)"/>'
            f'<text x="{(x1+x2)/2:.1f}" y="{(y1+y2)/2-4:.1f}" font-size="9" fill="#1565c0">并发</text>'
        )
    if link == "retry":
        return (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#e65100" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#arr-retry)"/>'
        )
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="#546e7a" stroke-width="2" marker-end="url(#arr-ser)"/>'
    )


def build_timeline_html(bundle: dict[str, Any]) -> str:
    events = prepare_e2e_events(bundle)
    sid = bundle.get("session_id", "—")
    summary = bundle.get("summary") or {}

    if not events:
        body = '<p class="muted">无事件</p>'
        canvas_w, canvas_h = 800, 200
    else:
        starts = [_event_start(e) for e in events]
        ends = [_event_end(e) for e in events]
        t0 = min(_naive_dt(s) for s in starts if s)
        t1 = max(_naive_dt(e) for e in ends if e)
        span = max(1.0, (t1 - t0).total_seconds())
        canvas_w, canvas_h, max_track = _layout_time_canvas(events)

        blocks: list[str] = []
        centers: list[tuple[float, float, float, float, dict]] = []

        for i, ev in enumerate(events):
            start = _event_start(ev)
            end = _event_end(ev)
            if not start or not end:
                continue
            top = _CANVAS_PAD + (_naive_dt(start) - t0).total_seconds() * _TIME_PX_PER_SEC
            h = max(_MIN_EVT_H, (_naive_dt(end) - _naive_dt(start)).total_seconds() * _TIME_PX_PER_SEC)
            left = _CANVAS_PAD + ev.get("flow_track", 0) * _TRACK_W
            dur = float(ev.get("duration_sec") or 0)
            kind = ev.get("kind") or "?"
            name = _esc(ev.get("name") or "?")
            link = ev.get("flow_link") or "serial"

            cls = "blk-llm" if kind == "llm" else "blk-tool"
            if ev.get("is_failure"):
                cls += " blk-fail"
            if ev.get("is_spawn"):
                cls += " blk-spawn"
            if link == "parallel":
                cls += " blk-parallel"
            if ev.get("repeat_index", 0) > 0:
                cls += " blk-retry"

            badges = []
            if link == "parallel":
                badges.append('<span class="badge par">并发</span>')
            elif link == "serial" and i > 0:
                badges.append('<span class="badge ser">串行</span>')
            if ev.get("is_failure"):
                badges.append('<span class="badge fail">失败</span>')
            if ev.get("repeat_index", 0) > 0:
                badges.append('<span class="badge dup">重复</span>')

            cx, cy = left + _TRACK_W * 0.45, top + h / 2
            centers.append((cx, cy, left, top + h, ev))

            sub = _llm_metrics_subtitle(ev) if kind == "llm" else ""
            title = f"{dur:.2f}s"
            if sub:
                title += f" · {sub}"
            blocks.append(
                f'<div class="evt-block {cls}" style="left:{left:.0f}px;top:{top:.0f}px;'
                f'width:{_TRACK_W - 12}px;height:{h:.0f}px" title="{_esc(title)}">'
                f'<div class="blk-title">{"🤖" if kind == "llm" else "🔧"} {name}</div>'
                f'<div class="blk-dur">{dur:.1f}s</div>'
                + (f'<div class="blk-meta">{_esc(sub)}</div>' if sub else "")
                + f'<div class="blk-badges">{"".join(badges)}</div></div>'
            )

        conn: list[str] = []
        for i in range(1, len(centers)):
            cx1, cy1, _, y1b, _prev = centers[i - 1]
            cx2, cy2, _, _, curr_ev = centers[i]
            link = curr_ev.get("flow_link") or "serial"
            conn.append(_connector_svg(cx1, y1b, cx2, cy2 - 8, link))

        track_labels = "".join(
            f'<div class="track-hdr" style="left:{_CANVAS_PAD + t * _TRACK_W}px">泳道 {t + 1}</div>'
            for t in range(max_track + 1)
        )

        body = (
            f'<div class="canvas-wrap" style="width:{canvas_w:.0f}px;height:{canvas_h:.0f}px">'
            f"{track_labels}"
            f'<svg class="connectors" width="{canvas_w:.0f}" height="{canvas_h:.0f}">'
            '<defs>'
            '<marker id="arr-ser" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
            '<path d="M0,0 L6,3 L0,6 Z" fill="#546e7a"/></marker>'
            '<marker id="arr-par" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
            '<path d="M0,0 L6,3 L0,6 Z" fill="#1565c0"/></marker>'
            '<marker id="arr-retry" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
            '<path d="M0,0 L6,3 L0,6 Z" fill="#e65100"/></marker>'
            "</defs>"
            + "".join(conn)
            + "</svg>"
            + "".join(blocks)
            + "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>端到端执行流 — {_esc(sid)}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f0f2f8; color: #222; }}
.header {{ background: linear-gradient(135deg,#1a237e,#5e35b1); color:#fff; padding:20px 24px; }}
.header h1 {{ margin:0 0 8px; font-size:1.35em; }}
.stats {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; font-size:0.88em; opacity:0.95; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; padding:12px 24px; background:#fff; border-bottom:1px solid #e0e4ee; font-size:0.85em; }}
.legend i {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; vertical-align:middle; }}
.wrap {{ max-width: 100%; margin: 0 auto; padding: 16px 20px 40px; overflow:auto; }}
.timeline {{ background:#fff; border-radius:10px; padding:16px; box-shadow:0 2px 12px rgba(0,0,0,.06); overflow:auto; }}
.canvas-wrap {{ position:relative; background:#fafbfc; border:1px solid #e0e4ee; border-radius:8px; }}
.track-hdr {{ position:absolute; top:8px; font-size:10px; color:#78909c; width:{_TRACK_W}px; text-align:center; }}
.connectors {{ position:absolute; left:0; top:0; pointer-events:none; z-index:1; }}
.evt-block {{ position:absolute; z-index:2; border-radius:6px; padding:4px 6px; border:2px solid rgba(0,0,0,.1); font-size:11px; overflow:hidden; box-sizing:border-box; }}
.blk-llm {{ background:linear-gradient(135deg,#c8e6c9,#81c784); }}
.blk-tool {{ background:linear-gradient(135deg,#ffe0b2,#ffb74d); }}
.blk-fail {{ border-color:#c62828; background:linear-gradient(135deg,#ffcdd2,#e57373); }}
.blk-spawn {{ border-style:dashed; border-color:#7b1fa2; }}
.blk-parallel {{ border-color:#1565c0; box-shadow:0 0 0 1px #90caf9; }}
.blk-retry {{ outline:2px dotted #e65100; outline-offset:1px; }}
.blk-title {{ font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.blk-dur {{ font-size:10px; color:#333; }}
.blk-meta {{ font-size:9px; color:#444; line-height:1.2; margin-top:1px; word-break:break-all; }}
.blk-badges {{ margin-top:2px; }}
.badge {{ display:inline-block; font-size:9px; padding:1px 4px; border-radius:3px; margin-right:3px; }}
.badge.par {{ background:#e3f2fd; color:#1565c0; }}
.badge.ser {{ background:#eceff1; color:#455a64; }}
.badge.fail {{ background:#ffebee; color:#c62828; }}
.badge.dup {{ background:#fff3e0; color:#e65100; }}
.muted {{ color:#888; }}
a {{ color:#3949ab; }}
</style>
</head>
<body>
<div class="header">
  <h1>端到端执行流程（泳道时间轴）</h1>
  <p>Session: {_esc(sid)} · 事件 {len(events)} 个 · 纵轴=时间 · 横轴=并发泳道</p>
  <div class="stats">
    <span>总任务 {summary.get('task_sec', '—')}s</span>
    <span>模型墙钟 {summary.get('llm_wall_sec', '—')}s</span>
    <span>TTFT Σ {summary.get('llm_ttft_sum_sec', '—')}s</span>
    <span>推理 Σ {summary.get('llm_inference_sum_sec', '—')}s</span>
    <span>tok in/out/cache {summary.get('input_tokens_sum', '—')}/{summary.get('output_tokens_sum', '—')}/{summary.get('cache_tokens_sum', 0)}</span>
    <span>工具墙钟 {summary.get('tool_sec', summary.get('tool_wall_sec', '—'))}s</span>
  </div>
</div>
<div class="legend">
  <span><i style="background:#66bb6a"></i>LLM</span>
  <span><i style="background:#fb8c00"></i>工具</span>
  <span><i style="background:#e53935"></i>失败</span>
  <span><i style="background:#ab47bc;border:1px dashed #7b1fa2"></i>子Agent</span>
  <span><i style="background:#fff;border:2px solid #1565c0"></i>并发（时间重叠·虚线蓝箭头）</span>
  <span><i style="background:#fff;border:2px solid #546e7a"></i>串行（实线灰箭头）</span>
  <span><i style="background:#fff;border:2px dotted #e65100"></i>重复调用</span>
</div>
<div class="wrap">
  <p>纵轴按真实时间排布；横轴泳道由时间重叠自动分配：<strong>同一时刻并列=并发</strong>，<strong>先后不重叠=串行</strong>。</p>
  <div class="timeline">{body}</div>
</div>
</body>
</html>"""


def build_mermaid_html(bundle: dict[str, Any], mermaid_src: str) -> str:
    sid = bundle.get("session_id", "—")
    escaped = json.dumps(mermaid_src)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>Mermaid 流程图 — {_esc(sid)}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #fafafa; }}
h1 {{ color: #1a237e; }}
#diagram {{ background: #fff; padding: 16px; border-radius: 8px; overflow: auto; }}
</style>
</head>
<body>
<h1>端到端 Mermaid 流程图</h1>
<p>Session: {_esc(sid)} · 若节点过多可横向滚动</p>
<div id="diagram" class="mermaid"></div>
<script>
mermaid.initialize({{ startOnLoad: false, theme: 'neutral', flowchart: {{ useMaxWidth: false, htmlLabels: true }} }});
const src = {escaped};
document.getElementById('diagram').textContent = src;
mermaid.run({{ nodes: [document.getElementById('diagram')] }});
</script>
</body>
</html>"""


def write_flowchart_artifacts(
    out_dir: Path,
    bundle: dict[str, Any],
    safe_name: str,
    *,
    validate: bool = True,
) -> dict[str, str]:
    """Write Mermaid + HTML flow files; return paths map."""
    from history_parse.mermaid_validate import (
        validate_mermaid,
        validate_mermaid_file,
        write_validation_report,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    mermaid = build_e2e_mermaid(bundle)
    static = validate_mermaid(mermaid)
    if validate and not static["ok"]:
        raise ValueError("Mermaid 静态校验失败: " + "; ".join(static["errors"]))

    mmd_path = out_dir / f"execution_flow_{safe_name}.mmd"
    mmd_path.write_text(mermaid, encoding="utf-8")

    validation = validate_mermaid_file(mmd_path, try_cli=True)
    validation["static"] = static
    val_path = write_validation_report(out_dir, safe_name, validation)
    if validate and not validation.get("ok", True):
        raise ValueError("Mermaid 校验失败: " + "; ".join(validation.get("errors", [])))
    if validate and validation.get("render_ok") is False:
        raise ValueError(f"Mermaid 渲染不可用: {validation.get('render_note', '')}")

    timeline_path = out_dir / f"execution_flowchart_{safe_name}.html"
    timeline_path.write_text(build_timeline_html(bundle), encoding="utf-8")

    mermaid_html_path = out_dir / f"execution_flow_mermaid_{safe_name}.html"
    mermaid_html_path.write_text(build_mermaid_html(bundle, mermaid), encoding="utf-8")

    return {
        "e2e_flow_mmd": str(mmd_path.resolve()),
        "e2e_flowchart_html": str(timeline_path.resolve()),
        "e2e_flow_mermaid_html": str(mermaid_html_path.resolve()),
        "e2e_flow_mermaid_validation": str(val_path.resolve()),
    }


def augment_bundle_file(bundle_path: Path, flow_paths: dict[str, str]) -> None:
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    data["e2e_mermaid_flowchart"] = build_e2e_mermaid(data)
    rp = data.get("report_paths") or {}
    rp.update(flow_paths)
    data["report_paths"] = rp
    bundle_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# Public helpers for standalone scripts (avoid protected-member access).
event_start = _event_start
event_end = _event_end
naive_dt = _naive_dt
llm_metrics_subtitle = _llm_metrics_subtitle
