"""Dedicated HTML: LLM inference latency timeline (TTFT / TPOT / tok/s / tokens)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ocperf.time_util import format_now

from history_parse.llm_latency_metrics import (
    aggregate_llm_latency,
    compute_tokens_per_sec,
    compute_tpot,
    llm_round_metrics,
)
from history_parse.models import LLMRound


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _copyable_pre(escaped_text: str) -> str:
    return (
        f'<div class="pre-box"><button class="copy-btn" onclick="copyPre(this, event)">复制</button>'
        f"<pre>{escaped_text}</pre></div>"
    )


def _detail_row(idx: int, r: LLMRound) -> str:
    if not (r.reasoning_full or r.assistant_text):
        return ""
    think = _copyable_pre(_esc(r.reasoning_full)) if r.reasoning_full else '<p class="muted">无思考过程</p>'
    out = _copyable_pre(_esc(r.assistant_text)) if r.assistant_text else '<p class="muted">无输出文本</p>'
    return f"""
<tr class="detail-row"><td colspan="14">
  <div class="llm-detail">
    <div class="collapsible-header" onclick="toggleBlock('lat-think-{idx}', this)"><span>思考过程</span><span class="arrow">▼</span></div>
    <div class="collapsible-content" id="lat-think-{idx}">{think}</div>
    <div class="collapsible-header" onclick="toggleBlock('lat-out-{idx}', this)"><span>模型输出</span><span class="arrow">▼</span></div>
    <div class="collapsible-content active" id="lat-out-{idx}">{out}</div>
  </div>
</td></tr>"""


def _fmt_tpot(tpot: float | None) -> str:
    if tpot is None:
        return "—"
    if tpot < 0.01:
        return f"{tpot * 1000:.1f} ms"
    return f"{tpot:.3f} s"


def _fmt_tps(tps: float | None) -> str:
    return f"{tps:.1f}" if tps is not None else "—"


def render_llm_latency_html(
    session_id: str,
    rounds: list[LLMRound],
    *,
    source_label: str = "history.json",
) -> str:
    agg = aggregate_llm_latency(rounds)
    if not rounds:
        empty = '<p class="muted">无 LLM 调用记录</p>'
        return _page_shell(session_id, source_label, agg, empty, "")

    t0 = min(r.request_ts for r in rounds)
    t1 = max(r.output_ts for r in rounds)
    span = max(1.0, (t1 - t0).total_seconds())
    chart_w = 920
    bar_area_h = 28

    rows: list[str] = []
    chart_rows: list[str] = []
    cum_in = cum_out = cum_cache = 0

    for i, r in enumerate(rounds, 1):
        m = llm_round_metrics(r)
        cum_in += r.input_tokens or 0
        cum_out += r.output_tokens or 0
        cum_cache += r.cache_tokens or 0
        rel_start = (r.request_ts - t0).total_seconds()
        rel_ttft_end = rel_start + r.ttft_sec
        rel_end = rel_start + r.duration_sec
        left_pct = 100.0 * rel_start / span
        ttft_w = max(0.4, 100.0 * r.ttft_sec / span)
        infer_w = max(0.4, 100.0 * r.inference_sec / span)
        model = _esc(r.model_name or "LLM")
        t_start = _esc(r.request_ts.strftime("%H:%M:%S.%f")[:-3])

        chart_rows.append(
            f'<div class="llm-row">'
            f'<div class="llm-row-label"><span class="idx">#{i}</span> {model}'
            f'<span class="ts">{t_start}</span></div>'
            f'<div class="llm-bar-track" style="width:{chart_w}px">'
            f'<div class="seg-ttft" style="left:{left_pct:.2f}%;width:{ttft_w:.2f}%" '
            f'title="TTFT {r.ttft_sec:.3f}s"></div>'
            f'<div class="seg-infer" style="left:{left_pct + ttft_w:.2f}%;width:{infer_w:.2f}%" '
            f'title="推理 {r.inference_sec:.3f}s"></div>'
            f"</div></div>"
        )

        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td class='mono'>{t_start}</td>"
            f"<td>{model}</td>"
            f"<td class='num'>{r.ttft_sec:.3f}</td>"
            f"<td class='num'>{r.inference_sec:.3f}</td>"
            f"<td class='num'>{r.duration_sec:.3f}</td>"
            f"<td class='num'>{_fmt_tpot(m['tpot_sec'])}</td>"
            f"<td class='num'>{_fmt_tps(m['tokens_per_sec'])}</td>"
            f"<td class='num'>{r.input_tokens or 0}</td>"
            f"<td class='num'>{r.output_tokens or 0}</td>"
            f"<td class='num'>{r.cache_tokens or 0}</td>"
            f"<td class='num muted'>{cum_in}</td>"
            f"<td class='num muted'>{cum_out}</td>"
            f"<td class='num muted'>{cum_cache}</td>"
            "</tr>"
        )
        rows.append(_detail_row(i, r))

    timeline_block = (
        f'<div class="timeline-chart">'
        f'<div class="axis"><span>{_esc(t0.strftime("%H:%M:%S"))}</span>'
        f'<span>{_esc(t1.strftime("%H:%M:%S"))}</span></div>'
        f'<div class="chart-legend">'
        f'<span><i class="lg ttft"></i>TTFT（首 token 前）</span>'
        f'<span><i class="lg infer"></i>推理（首 token → 结束）</span>'
        f"</div>"
        + "".join(chart_rows)
        + "</div>"
    )

    table_block = (
        '<div class="table-wrap"><table class="metrics-table">'
        "<thead><tr>"
        "<th>#</th><th>开始</th><th>模型</th>"
        "<th>TTFT(s)</th><th>推理(s)</th><th>总(s)</th>"
        "<th>TPOT</th><th>tok/s</th>"
        "<th>input</th><th>output</th><th>cache</th>"
        "<th>Σ input</th><th>Σ output</th><th>Σ cache</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )

    body = timeline_block + table_block
    return _page_shell(session_id, source_label, agg, body, "")


def _page_shell(
    session_id: str,
    source_label: str,
    agg: dict[str, Any],
    body: str,
    extra: str,
) -> str:
    avg_tpot = agg.get("avg_tpot_sec")
    avg_tps = agg.get("avg_tokens_per_sec")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>LLM 推理时延 — {_esc(session_id[:48])}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #eef1f8; color: #222; }}
.header {{ background: linear-gradient(135deg,#0d47a1,#5e35b1); color: #fff; padding: 22px 28px; }}
.header h1 {{ margin: 0 0 6px; font-size: 1.45em; }}
.header p {{ margin: 0; opacity: 0.92; font-size: 0.9em; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; padding: 20px 28px; }}
.stat {{ background: #fff; border-radius: 8px; padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,.06); border-top: 3px solid #3949ab; }}
.stat-val {{ font-size: 1.35em; font-weight: 700; color: #1a237e; }}
.stat-lbl {{ font-size: 0.8em; color: #666; margin-top: 4px; }}
.wrap {{ max-width: 1280px; margin: 0 auto; padding: 0 28px 40px; }}
.card {{ background: #fff; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 12px rgba(0,0,0,.06); }}
.card h2 {{ margin: 0 0 14px; font-size: 1.1em; color: #1a237e; border-bottom: 2px solid #c5cae9; padding-bottom: 8px; }}
.timeline-chart {{ overflow-x: auto; }}
.axis {{ display: flex; justify-content: space-between; font-size: 0.82em; color: #666; margin-bottom: 8px; }}
.chart-legend {{ display: flex; gap: 18px; font-size: 0.85em; margin-bottom: 12px; }}
.chart-legend .lg {{ display: inline-block; width: 14px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}
.lg.ttft {{ background: #42a5f5; }}
.lg.infer {{ background: #66bb6a; }}
.llm-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 6px; min-height: 28px; }}
.llm-row-label {{ width: 200px; flex-shrink: 0; font-size: 0.82em; }}
.llm-row-label .idx {{ color: #5c6bc0; font-weight: 700; }}
.llm-row-label .ts {{ display: block; color: #888; font-size: 0.78em; }}
.llm-bar-track {{ position: relative; height: 22px; background: #f5f5f5; border-radius: 4px; flex: 1; min-width: 400px; }}
.seg-ttft, .seg-infer {{ position: absolute; top: 2px; height: 18px; border-radius: 3px; min-width: 2px; }}
.seg-ttft {{ background: #42a5f5; }}
.seg-infer {{ background: #43a047; }}
.table-wrap {{ overflow-x: auto; }}
.metrics-table {{ width: 100%; border-collapse: collapse; font-size: 0.86em; }}
.metrics-table th, .metrics-table td {{ border: 1px solid #e0e4ee; padding: 8px 10px; text-align: left; }}
.metrics-table th {{ background: #e8eaf6; color: #333; white-space: nowrap; }}
.metrics-table .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.metrics-table .mono {{ font-family: ui-monospace, monospace; font-size: 0.9em; }}
.metrics-table .muted {{ color: #78909c; }}
.muted {{ color: #888; }}
.detail-row td {{ background: #fafbff; border-top: none; padding-top: 0; }}
.llm-detail {{ padding: 8px 4px 12px; }}
.collapsible-header {{ background: #f0f0f0; padding: 8px 10px; cursor: pointer; border-radius: 6px; margin: 6px 0 4px; display: flex; justify-content: space-between; font-size: .88em; }}
.collapsible-content {{ display: none; padding: 8px; background: #f7f8fa; border-radius: 6px; max-height: 360px; overflow: auto; }}
.collapsible-content.active {{ display: block; }}
.pre-box {{ position: relative; }}
.copy-btn {{ position: absolute; top: 8px; right: 8px; z-index: 2; border: 1px solid #cfd7ff; background: #eef2ff; color: #4154c5; border-radius: 5px; padding: 3px 8px; font-size: 12px; cursor: pointer; }}
pre {{ background: #101622; color: #e6edf3; padding: 10px; border-radius: 6px; overflow: auto; font-size: .78rem; white-space: pre-wrap; word-break: break-word; margin: 0; }}
.arrow {{ font-size: .75em; color: #888; }}
a {{ color: #3949ab; }}
</style>
</head>
<body>
<div class="header">
  <h1>LLM 推理时延时间线</h1>
  <p>Session: {_esc(session_id)} · 数据源: {_esc(source_label)} · 生成: {_esc(format_now())}</p>
</div>
<div class="stats">
  <div class="stat"><div class="stat-val">{agg.get('llm_calls', 0)}</div><div class="stat-lbl">LLM 调用次数</div></div>
  <div class="stat"><div class="stat-val">{agg.get('llm_ttft_sum_sec', 0)}s</div><div class="stat-lbl">TTFT 累计</div></div>
  <div class="stat"><div class="stat-val">{agg.get('llm_inference_sum_sec', 0)}s</div><div class="stat-lbl">推理累计</div></div>
  <div class="stat"><div class="stat-val">{_fmt_tpot(avg_tpot)}</div><div class="stat-lbl">平均 TPOT</div></div>
  <div class="stat"><div class="stat-val">{_fmt_tps(avg_tps)}</div><div class="stat-lbl">平均 tok/s</div></div>
  <div class="stat"><div class="stat-val">{agg.get('input_tokens_sum', 0)}</div><div class="stat-lbl">input 总量</div></div>
  <div class="stat"><div class="stat-val">{agg.get('output_tokens_sum', 0)}</div><div class="stat-lbl">output 总量</div></div>
  <div class="stat"><div class="stat-val">{agg.get('cache_tokens_sum', 0)}</div><div class="stat-lbl">cache_token 总量</div></div>
</div>
<div class="wrap">
  <div class="card">
    <h2>按时间线：TTFT / 推理 / TPOT / 吞吐</h2>
    <p style="font-size:0.88em;color:#555;margin-bottom:14px">
      横轴为会话内相对时间；每条 LLM 调用展示 TTFT（蓝）与推理阶段（绿）。
      下表含单次 input / output / cache_token 及累计 Σ。
    </p>
    {body}
  </div>
  {extra}
</div>
<script>
function toggleBlock(id, headerEl) {{
  const el = document.getElementById(id);
  if (!el) return;
  const isOpen = el.classList.contains('active');
  el.classList.toggle('active', !isOpen);
}}
async function copyPre(btn, event) {{
  event.stopPropagation();
  const pre = btn.closest('.pre-box')?.querySelector('pre');
  if (!pre) return;
  const text = pre.innerText || pre.textContent || '';
  try {{ await navigator.clipboard.writeText(text); btn.textContent = '已复制'; }}
  catch {{ document.execCommand('copy'); btn.textContent = '已复制'; }}
  setTimeout(() => {{ btn.textContent = '复制'; }}, 1200);
}}
</script>
</body>
</html>"""


def write_llm_latency_report(path: Path, html_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
