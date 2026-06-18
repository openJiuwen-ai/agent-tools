#!/usr/bin/env python3
"""Generate complete E2E flowchart: HTML timeline + Mermaid + SVG (duration/fail/retry)."""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import sys
from pathlib import Path


def _resolve_ocperf_src() -> Path:
    scripts = Path(__file__).resolve().parent
    env = os.environ.get("OCPERF_HOME")
    if env:
        src = Path(env) / "src"
        if (src / "ocperf" / "cli.py").is_file():
            return src
    bundled = scripts.parent / "ocperf" / "src"
    if (bundled / "ocperf" / "cli.py").is_file():
        return bundled
    raise FileNotFoundError(
        "未找到 ocperf 引擎。请确认 skill/ocperf 存在，或设置 OCPERF_HOME 指向引擎根目录。"
    )


def _flowchart_export():
    src = str(_resolve_ocperf_src())
    if src not in sys.path:
        sys.path.append(src)
    from history_parse import flowchart_export

    return flowchart_export


logger = logging.getLogger(__name__)

_PX_PER_SEC = 2.8
_MIN_H = 22
_MARGIN = 48
_TRACK_W = 150


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _safe_name(session_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)


def build_e2e_svg(bundle: dict, *, max_events: int = 2500) -> str:
    """Swimlane SVG: Y=time, X=track; blue dashed=parallel, gray solid=serial."""
    fe = _flowchart_export()
    events = fe.prepare_e2e_events(bundle)
    if not events:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80">'
            '<text x="20" y="40" font-size="14">无事件</text></svg>'
        )

    truncated = len(events) > max_events
    show = events[:max_events]
    durations = [float(e.get("duration_sec") or 0) for e in show]
    slow_thresh = sorted(durations, reverse=True)[min(4, len(durations) - 1)] if durations else 9999.0
    if slow_thresh < 30:
        slow_thresh = 30.0

    sid = bundle.get("session_id", "—")
    starts = [fe.event_start(e) for e in show]
    ends = [fe.event_end(e) for e in show]
    valid = [(s, e) for s, e in zip(starts, ends) if s and e]
    if not valid:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80">'
            '<text x="20" y="40">无事件</text></svg>'
        )
    t0 = min(fe.naive_dt(s) for s, _ in valid)
    t1 = max(fe.naive_dt(e) for _, e in valid)
    span = max(1.0, (t1 - t0).total_seconds())
    max_track = max((e.get("flow_track", 0) for e in show), default=0)
    w = _MARGIN * 2 + (max_track + 1) * _TRACK_W
    total_h = _MARGIN * 2 + span * _PX_PER_SEC
    total_h = min(max(total_h, 320), 24000)

    body: list[str] = [
        f'<text x="{_MARGIN}" y="28" font-size="16" font-weight="bold" fill="#1a237e">'
        f"端到端执行流 — {_esc(sid)}</text>",
        f'<text x="{_MARGIN}" y="44" font-size="11" fill="#555">'
        f"事件 {len(show)}{f'/{len(events)}' if truncated else ''} · "
        f"纵轴=时间 横轴=泳道 · 蓝虚线=并发 灰实线=串行</text>",
    ]
    centers: list[tuple[float, float, float, dict]] = []

    for ev in show:
        start = fe.event_start(ev)
        end = fe.event_end(ev)
        if not start or not end:
            continue
        top = _MARGIN + (fe.naive_dt(start) - t0).total_seconds() * _PX_PER_SEC
        h = max(_MIN_H, (fe.naive_dt(end) - fe.naive_dt(start)).total_seconds() * _PX_PER_SEC)
        left = _MARGIN + ev.get("flow_track", 0) * _TRACK_W
        dur = float(ev.get("duration_sec") or 0)
        kind = ev.get("kind") or "?"
        name = _esc((ev.get("name") or "?")[:28])
        link = ev.get("flow_link") or "serial"

        if ev.get("is_failure"):
            fill, stroke = "#ffcdd2", "#c62828"
        elif ev.get("is_spawn"):
            fill, stroke = "#e1bee7", "#7b1fa2"
        elif kind == "llm":
            fill, stroke = "#c8e6c9", "#2e7d32"
        else:
            fill, stroke = "#ffe0b2", "#ef6c00"
        if dur >= slow_thresh:
            stroke = "#f9a825"
        sw = 3 if link == "parallel" else 2

        body.append(
            f'<rect x="{left:.0f}" y="{top:.0f}" width="{_TRACK_W - 10:.0f}" height="{h:.0f}" rx="5" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        icon = "LLM" if kind == "llm" else "TOOL"
        rel = "并发" if link == "parallel" else ("重复" if link == "retry" else "串行")
        body.append(
            f'<text x="{left + 6:.0f}" y="{top + 16:.0f}" font-size="11" font-weight="600">'
            f"{icon} {name}</text>"
        )
        sub = ""
        if kind == "llm":
            sub = fe.llm_metrics_subtitle(ev)
        line2 = f"{dur:.1f}s · {_esc(rel)}"
        if sub:
            line2 = _esc(sub[:42])
        body.append(
            f'<text x="{left + 6:.0f}" y="{top + 30:.0f}" font-size="10" fill="#333">'
            f"{line2}</text>"
        )
        if sub and h > 36:
            body.append(
                f'<text x="{left + 6:.0f}" y="{top + 42:.0f}" font-size="9" fill="#555">'
                f"{dur:.1f}s · {_esc(rel)}</text>"
            )
        cx, cy = left + (_TRACK_W - 10) / 2, top + h
        centers.append((cx, cy, top, ev))

    for i in range(1, len(centers)):
        cx1, y1b, _, prev = centers[i - 1]
        cx2, _, top2, curr = centers[i]
        link = curr.get("flow_link") or "serial"
        if link == "parallel":
            body.append(
                f'<line x1="{cx1:.0f}" y1="{y1b:.0f}" x2="{cx2:.0f}" y2="{top2:.0f}" '
                f'stroke="#1565c0" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#arr-p)"/>'
            )
        elif link == "retry":
            body.append(
                f'<line x1="{cx1:.0f}" y1="{y1b:.0f}" x2="{cx2:.0f}" y2="{top2:.0f}" '
                f'stroke="#e65100" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#arr-r)"/>'
            )
        else:
            body.append(
                f'<line x1="{cx1:.0f}" y1="{y1b:.0f}" x2="{cx2:.0f}" y2="{top2:.0f}" '
                f'stroke="#546e7a" stroke-width="2" marker-end="url(#arr-s)"/>'
            )

    defs = (
        "<defs>"
        '<marker id="arr-s" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#546e7a"/></marker>'
        '<marker id="arr-p" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#1565c0"/></marker>'
        '<marker id="arr-r" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#e65100"/></marker>'
        "</defs>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{total_h:.0f}" '
        f'viewBox="0 0 {w} {total_h:.0f}">{defs}{"".join(body)}</svg>'
    )


def build_svg_wrapper_page(bundle: dict, svg_body: str) -> str:
    sid = bundle.get("session_id", "—")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>SVG 端到端流程 — {_esc(sid)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #f5f5f5; }}
.wrap {{ background: #fff; padding: 12px; border-radius: 8px; overflow: auto; max-width: 100%; }}
svg {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>端到端流程图（SVG 向量）</h1>
<p>Session: {_esc(sid)} · 泳道=并发 · 蓝虚线=并行 · 灰实线=串行 · 橙虚线=重复</p>
<div class="wrap">{svg_body}</div>
</body>
</html>"""


def write_svg_artifacts(out_dir: Path, bundle: dict, safe: str) -> dict[str, str]:
    svg = build_e2e_svg(bundle)
    svg_path = out_dir / f"execution_flowchart_{safe}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    html_path = out_dir / f"execution_flowchart_{safe}_svg.html"
    html_path.write_text(build_svg_wrapper_page(bundle, svg), encoding="utf-8")
    return {
        "e2e_flowchart_svg": str(svg_path.resolve()),
        "e2e_flowchart_svg_html": str(html_path.resolve()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="生成完整端到端流程图（HTML+Mermaid+SVG）")
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    if not args.bundle.is_file():
        raise FileNotFoundError(f"bundle 不存在: {args.bundle}")

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    sid = bundle.get("session_id") or "session"
    safe = _safe_name(sid)
    out_dir = (args.out_dir or args.bundle.parent).resolve()

    fe = _flowchart_export()
    paths = fe.write_flowchart_artifacts(out_dir, bundle, safe, validate=True)
    svg_paths = write_svg_artifacts(out_dir, bundle, safe)
    paths.update(svg_paths)
    fe.augment_bundle_file(args.bundle, paths)

    val_path = paths.get("e2e_flow_mermaid_validation")
    if val_path:
        rep = json.loads(Path(val_path).read_text(encoding="utf-8"))
        status = "OK" if rep.get("ok") else "FAIL"
        logger.info("Mermaid validation: %s", status)
        for w in rep.get("warnings") or []:
            logger.info("  warn: %s", w)
        if rep.get("render_note"):
            logger.info("  render: %s", rep["render_note"])

    logger.info("已生成端到端流程图:")
    for k, v in paths.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        main()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
