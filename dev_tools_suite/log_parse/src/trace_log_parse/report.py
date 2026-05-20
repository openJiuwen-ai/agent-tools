"""Render analysis to a self-contained HTML file."""

from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from trace_log_parse.analysis import LLMRound, ToolGap, aggregate_totals, interval_union_sec

_GAP_EMPTY_MSG = (
    '<div class="meta-line">'
    "这段时间没有 tool_calls，"
    "表示模型输出后到下一次请求前的流程等待/框架处理间隙。"
    "</div>"
)


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _json_pretty(obj: Any) -> str:
    return _esc(json.dumps(obj, ensure_ascii=False, indent=2)[:12000])


def _copyable_pre(escaped_text: str) -> str:
    pre_open = '<div class="pre-box">'
    copy_btn = (
        '<button class="copy-btn" onclick="copyPre(this, event)">复制</button>'
    )
    return f"{pre_open}{copy_btn}<pre>{escaped_text}</pre></div>"


def _js_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _item_start(kind: Literal["llm", "gap"], obj: Any) -> datetime:
    return obj.request_ts if kind == "llm" else obj.after_output_ts


def _item_end(kind: Literal["llm", "gap"], obj: Any) -> datetime:
    return obj.output_ts if kind == "llm" else obj.next_request_ts


def _parse_tool_arguments(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _step_boundary_label(gap: "ToolGap") -> str | None:
    for tool in gap.detail_tools:
        tool_name = tool.get("name")
        args = _parse_tool_arguments(tool.get("arguments"))
        action = args.get("action") if isinstance(args, dict) else None

        is_old_complete = tool_name in {"skill_step_complete", "skill_step_complete_batch"}
        is_new_complete = tool_name in {"skill_step", "shill_step"} and action in {"complete", "complete_batch"}
        if not (is_old_complete or is_new_complete):
            continue

        if isinstance(args, dict):
            if tool_name == "skill_step_complete_batch" or action == "complete_batch":
                return f"批量完成步骤：{json.dumps(args, ensure_ascii=False)}"
            idx = args.get("idx", "")
            result = args.get("result", "")
            return f"完成步骤 {idx}：{result}".strip("：")
        return f"{tool_name}: {args}"
    return None


def merge_timeline_for_display(
    rounds: list[LLMRound], gaps: list[ToolGap]
) -> list[tuple[Literal["llm", "gap"], Any]]:
    """Chronological: each LLM round, then tool gap(s) that follow that output.

    轮次按 **output_ts** 主序（再 request_ts、session_id），避免并行多会话时
    「先结束的一轮」因 request 较晚而排在后面，导致主/子时间线倒序、树状分组错位。
    """
    remaining: list[ToolGap] = sorted(gaps, key=lambda g: g.next_request_ts)

    def take_gaps_for_round(r: LLMRound) -> list[ToolGap]:
        taken: list[ToolGap] = []
        rest: list[ToolGap] = []
        for g in remaining:
            dt = abs((g.after_output_ts - r.output_ts).total_seconds())
            # 并行子会话可能共用同一 request_id，且输出在同一毫秒内完成；仅按时间戳会把
            # 其它 session 的 tool gap 挂到本轮下（例如 A 轮只有 edit_file，间隙却含 spawn_subagent）。
            if dt < 1e-3 and g.session_id == r.session_id:
                taken.append(g)
            else:
                rest.append(g)
        remaining[:] = rest
        taken.sort(key=lambda x: x.next_request_ts)
        return taken

    out: list[tuple[Literal["llm", "gap"], Any]] = []
    for r in sorted(rounds, key=lambda x: (x.output_ts, x.request_ts, x.session_id)):
        out.append(("llm", r))
        for g in take_gaps_for_round(r):
            out.append(("gap", g))
    return out


def render_html(
    root_session: str,
    log_path: str,
    rounds: list[LLMRound],
    gaps: list[ToolGap],
) -> str:
    post_final_rounds = [r for r in rounds if r.is_post_final]
    post_final_gaps = [g for g in gaps if g.is_post_final]
    rounds = [r for r in rounds if not r.is_post_final]
    gaps = [g for g in gaps if not g.is_post_final]

    tot = aggregate_totals(rounds)
    gap_sec = interval_union_sec([(g.after_output_ts, g.next_request_ts) for g in gaps])
    merged = merge_timeline_for_display(rounds, gaps)
    post_final_merged = merge_timeline_for_display(post_final_rounds, post_final_gaps)
    post_final_tot = aggregate_totals(post_final_rounds)
    post_final_gap_sec = interval_union_sec([(g.after_output_ts, g.next_request_ts) for g in post_final_gaps])
    tool_calls_count = sum(len(r.tools) for r in rounds)
    input_tokens_sum = tot.get("input_tokens_sum") or 0
    output_tokens_sum = tot.get("output_tokens_sum") or 0
    total_tokens_sum = tot.get("total_tokens_sum") or 0
    all_points: list[datetime] = []
    all_points.extend(r.request_ts for r in rounds)
    all_points.extend(r.output_ts for r in rounds)
    all_points.extend(g.after_output_ts for g in gaps)
    all_points.extend(g.next_request_ts for g in gaps)
    task_total_sec = round((max(all_points) - min(all_points)).total_seconds(), 3) if all_points else 0.0

    # 按 request_id 分组：gap 继承“前一个 LLM round”的 request_id（同一时间线）
    request_groups: dict[str, list[tuple[Literal["llm", "gap"], Any]]] = {}
    current_rid = ""
    for kind, obj in merged:
        if kind == "llm":
            current_rid = obj.request_id
            request_groups.setdefault(current_rid, []).append((kind, obj))
            continue

        # gap item
        rid_for_gap = current_rid or "(unknown-request)"
        request_groups.setdefault(rid_for_gap, []).append((kind, obj))

    def _round_block(idx: int, r: LLMRound) -> str:
        label = _esc(r.child_label or "child-session")
        child_badge = f'<span class="badge badge-orange">{label}</span>' if r.is_child_session else ""
        tool_summary_html = ""
        if r.tools:
            counter = Counter(str(t.get("name") or "?") for t in r.tools)
            summary = ", ".join(f"{name}×{count}" if count > 1 else name for name, count in counter.items())
            tool_summary_html = f'<div class="meta-line"><strong>工具调用：</strong>{_esc(summary)}</div>'
        tools_html = ""
        if r.tools:
            tool_blocks = "".join(_copyable_pre(_json_pretty(item)) for item in r.tools)
            tools_html = f"""
    <div class="collapsible-header" onclick="toggleBlock('tools-{idx}', this)">
      <span>本轮工具调用</span><span class="arrow">▼</span>
    </div>
    <div class="collapsible-content" id="tools-{idx}">{tool_blocks}</div>
"""
        return f"""
<div class="timeline-item llm-item">
  <div class="request-header" onclick="toggleBlock('req-{idx}', this)">
    <div>
      <div class="request-id">
        <span class="badge-pill badge-time">{r.duration_sec:.3f}s</span>
        <span class="badge-pill badge-type">模型调用</span>
        <span class="title-main">token消耗：input：{r.input_tokens or 0} output：{r.output_tokens or 0} total：{r.total_tokens or 0}</span>
        <span class="title-time">{_esc(r.request_ts.strftime("%H:%M:%S.%f")[:-3])} → {_esc(r.output_ts.strftime("%H:%M:%S.%f")[:-3])}</span>
      </div>
      {tool_summary_html}
    </div>
    <div class="header-right">
      {child_badge}
      <span class="arrow">▼</span>
    </div>
  </div>
  <div class="request-details" id="req-{idx}">
    <div class="llm-body-box">
    <div class="collapsible-header" onclick="toggleBlock('in-{idx}', this)">
      <span>模型输入</span><span class="arrow">▼</span>
    </div>
    <div class="collapsible-content" id="in-{idx}">{_copyable_pre(_esc(r.request_body_full))}</div>
    <div class="collapsible-header" onclick="toggleBlock('think-{idx}', this)">
      <span>思考过程</span><span class="arrow">▼</span>
    </div>
    <div class="collapsible-content" id="think-{idx}">{_copyable_pre(_esc(r.reasoning_full or ""))}</div>
    <div class="collapsible-header" onclick="toggleBlock('reply-{idx}', this)">
      <span>模型回复</span><span class="arrow">▼</span>
    </div>
    <div class="collapsible-content" id="reply-{idx}">{_copyable_pre(_esc(r.output_body_excerpt))}</div>
    {tools_html}
    </div>
  </div>
</div>
"""

    def _gap_block(idx: int, g: ToolGap) -> str:
        tool_name = g.tools_triggered or "(no tool_calls)"
        has_tools = bool(g.detail_tools)
        title_badge = "工具调用" if has_tools else "流程间隙"
        title_name = tool_name if has_tools else "无工具调用"
        details = g.detail_tools
        detail_blocks = []
        for j, item in enumerate(details, 1):
            detail_blocks.append(
                _copyable_pre(_json_pretty(item))
            )
        details_html = "".join(detail_blocks) if detail_blocks else _GAP_EMPTY_MSG
        return f"""
<div class="timeline-item gap-item">
  <div class="request-header" onclick="toggleBlock('gap-{idx}', this)">
    <div>
      <div class="request-id">
        <span class="badge-pill badge-time">{g.duration_sec:.3f}s</span>
        <span class="badge-pill badge-tool">{title_badge}</span>
        <span class="title-main">{_esc(title_name)}</span>
        <span class="title-time">{_esc(g.after_output_ts.strftime("%H:%M:%S.%f")[:-3])} → {_esc(g.next_request_ts.strftime("%H:%M:%S.%f")[:-3])}</span>
      </div>
    </div>
    <span class="arrow">▼</span>
  </div>
  <div class="request-details" id="gap-{idx}">
    <div class="collapsible-content active">{details_html}</div>
  </div>
</div>
"""

    def _append_timeline_items(items: list[tuple[Literal["llm", "gap"], Any]]) -> None:
        """Render as a tree by child_path.

        目标：forkagent 不再和主流程交叉展示；每个 forkagent session 是一个大块，
        块内按自身时间线排序展示，避免重复/错位。
        """
        nonlocal idx

        def _event_ts(kind: Literal["llm", "gap"], obj: Any) -> datetime:
            return obj.output_ts if kind == "llm" else obj.after_output_ts

        # Group by full child_path
        by_path: dict[tuple[tuple[str, str, str], ...], list[tuple[Literal["llm", "gap"], Any]]] = {}
        for kind, obj in items:
            path = tuple(getattr(obj, "child_path", ()) or ())
            by_path.setdefault(path, []).append((kind, obj))
        for path_items in by_path.values():
            path_items.sort(key=lambda it: (_event_ts(it[0], it[1]), _item_end(it[0], it[1])))

        # Build parent -> children paths
        children: dict[tuple[tuple[str, str, str], ...], list[tuple[tuple[str, str, str], ...]]] = {}
        for p in by_path.keys():
            parent = p[:-1]
            if not p:
                continue
            children.setdefault(parent, []).append(p)
        path_start_ts: dict[tuple[tuple[str, str, str], ...], datetime] = {}
        for p, path_items in by_path.items():
            if not path_items:
                continue
            path_start_ts[p] = min(
                _event_ts(kind, item) for kind, item in path_items
            )
        for parent, kids in children.items():
            kids.sort(key=lambda p: (path_start_ts.get(p, datetime.max), p[-1][1]))

        def _render_path(path: tuple[tuple[str, str, str], ...]) -> None:
            nonlocal idx, group_idx
            path_items = by_path.get(path, [])
            pending_children = list(children.get(path, []))

            def _render_child_block(child: tuple[tuple[str, str, str], ...]) -> None:
                nonlocal group_idx
                _, label, session_kind = child[-1]
                title = "ForkAgent" if session_kind == "forkagent" else "子模块"
                if session_kind == "forkagent":
                    gid = f"forkgrp-{group_idx}"
                    group_idx += 1
                    blocks.append('<div class="child-group">')
                    blocks.append(
                        f'<div class="child-group-header" onclick="toggleBlock(\'{gid}\', this)">'
                        f'<div class="child-group-title">ForkAgent：{_esc(label)}</div>'
                        '<span class="arrow">▼</span></div>'
                    )
                    blocks.append(f'<div class="fork-group-content" id="{gid}">')
                    _render_path(child)
                    blocks.append("</div></div>")
                else:
                    blocks.append(
                        f'<div class="child-group"><div class="child-group-title">{title}：{_esc(label)}</div>'
                    )
                    _render_path(child)
                    blocks.append("</div>")

            # Render items directly under this path; insert child block right after
            # the nearest parent event whose time reaches that child start time.
            for kind, obj in path_items:
                is_redundant_fork_gap = False
                is_fork_trigger_gap = False
                if kind == "gap":
                    tools_text = (obj.tools_triggered or "").lower()
                    # 某些轮次会重复吐出 fork_agent tool_calls，但并未真正产生新的 fork session。
                    # 当此处已没有待插入的 fork 子块时，隐藏该重复间隙，避免“fork_agent×3”重复噪声。
                    if "fork_agent" in tools_text and pending_children:
                        is_fork_trigger_gap = True
                    if "fork_agent" in tools_text and not pending_children:
                        is_redundant_fork_gap = True

                if kind == "llm":
                    blocks.append(_round_block(idx, obj))
                elif not is_redundant_fork_gap:
                    blocks.append(_gap_block(idx, obj))
                else:
                    # Skip duplicated fork-only gap without any new child sessions.
                    pass
                idx += 1

                # 当父层出现一次 fork_agent 触发时，将所有 fork 子块整体挂载在其后，
                # 避免同批并行 fork 因首个输出时间不同被拆到后续重复 gap 之后。
                if is_fork_trigger_gap:
                    while pending_children:
                        _render_child_block(pending_children.pop(0))
                    continue

                event_end = _item_end(kind, obj)
                while pending_children and path_start_ts.get(pending_children[0], datetime.max) <= event_end:
                    _render_child_block(pending_children.pop(0))

            # If any child starts after all parent items (or parent has no direct items), append remaining.
            while pending_children:
                _render_child_block(pending_children.pop(0))

        _render_path(())

    def _step_stats(items: list[tuple[Literal["llm", "gap"], Any]]) -> dict[str, Any]:
        if not items:
            return {
                "duration": 0.0,
                "input": 0,
                "output": 0,
                "total": 0,
                "tool_sec": 0.0,
                "tool_calls": 0,
            }
        starts = [_item_start(kind, item) for kind, item in items]
        ends = [_item_end(kind, item) for kind, item in items]
        llm_rounds = [item for kind, item in items if kind == "llm"]
        gap_items = [item for kind, item in items if kind == "gap"]
        duration = round((max(ends) - min(starts)).total_seconds(), 3)
        model_sec = round(
            interval_union_sec([(r.request_ts, r.output_ts) for r in llm_rounds]),
            3,
        )
        tool_sec = round(
            interval_union_sec(
                [(g.after_output_ts, g.next_request_ts) for g in gap_items]
            ),
            3,
        )
        other_sec = round(max(0.0, duration - model_sec - tool_sec), 3)
        return {
            "duration": duration,
            "model_sec": model_sec,
            "input": sum(r.input_tokens or 0 for r in llm_rounds),
            "output": sum(r.output_tokens or 0 for r in llm_rounds),
            "total": sum(r.total_tokens or 0 for r in llm_rounds),
            "tool_sec": tool_sec,
            "other_sec": other_sec,
            "tool_calls": sum(len(g.detail_tools) for g in gap_items),
        }

    def _step_event_chart_data(items: list[tuple[Literal["llm", "gap"], Any]]) -> dict[str, Any]:
        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        details: list[str] = []
        for kind, obj in items:
            if kind == "llm":
                labels.append("模型调用")
                values.append(round(obj.duration_sec, 3))
                colors.append("#4caf50")
                token_line = (
                    f"token input={obj.input_tokens or 0}, "
                    f"output={obj.output_tokens or 0}, "
                    f"total={obj.total_tokens or 0}"
                )
                details.append(
                    f"模型调用 | {obj.request_ts.strftime('%H:%M:%S.%f')[:-3]} → "
                    f"{obj.output_ts.strftime('%H:%M:%S.%f')[:-3]} | "
                    f"{token_line}"
                )
            else:
                label = obj.tools_triggered or "(no tool_calls)"
                if label == "(no tool_calls)":
                    label = "流程间隙"
                labels.append(label)
                values.append(round(obj.duration_sec, 3))
                colors.append("#f57c00")
                details.append(
                    f"{label} | {obj.after_output_ts.strftime('%H:%M:%S.%f')[:-3]} → "
                    f"{obj.next_request_ts.strftime('%H:%M:%S.%f')[:-3]}"
                )
        return {
            "labels": labels,
            "values": values,
            "colors": colors,
            "details": details,
        }

    step_items = list[tuple[Literal["llm", "gap"], Any]]
    step_group = tuple[str, step_items]

    def _split_steps(items: step_items) -> list[step_group]:
        steps: list[step_group] = []
        current: step_items = []
        unnamed_idx = 1
        for kind, obj in items:
            current.append((kind, obj))
            label = _step_boundary_label(obj) if kind == "gap" else None
            if label is None:
                continue
            steps.append((label, current))
            current = []
        if current:
            steps.append(("收尾阶段", current))
        return steps

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    head = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>LLM IO Trace — {_esc(root_session[:40])}…</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px 15px; text-align: center; border-radius: 8px; margin-bottom: 20px; }}
    .header h1 {{ font-size: 1.6em; margin-bottom: 5px; }}
    .header p {{ font-size: 0.9em; opacity: 0.92; }}
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
    .chart-container {{ background:white; border-radius:10px; padding:18px; margin-bottom:20px; box-shadow:0 2px 10px rgba(0,0,0,0.08); }}
    .chart-title {{ font-size:1.1em; font-weight:700; color:#333; margin-bottom:12px; padding-bottom:8px; border-bottom:2px solid #667eea; }}
    .chart-wrapper {{ position:relative; height:320px; }}
    .step-chart-wrapper {{ position:relative; height:520px; margin-bottom:12px; background:#fff; border:1px solid #e0e4ee; border-radius:8px; padding:10px; }}
    .timeline-item {{ border-left: 4px solid #667eea; padding-left: 14px; margin-bottom: 14px; }}
    .gap-item {{ border-left-color: #8e6bf2; }}
    .child-group {{ border-left: 6px solid #f57c00; background: #fff8ef; border-radius: 8px; padding: 10px 10px 4px; margin: 12px 0; }}
    .child-group .child-group {{ margin-left: 16px; margin-top: 10px; background: #fffaf5; border-left-color: #ff9800; }}
    .child-group .child-group .child-group {{ border-left-color: #fb8c00; background: #fffbf7; }}
    .child-group-title {{ font-size: 0.95em; font-weight: 700; color: #f57c00; margin: 0 0 8px 4px; }}
    .child-group-header {{ display:flex; justify-content:space-between; align-items:center; gap:8px; background:#fff1dc; border-radius:6px; padding:8px 10px; cursor:pointer; margin-bottom:8px; }}
    .child-group-header:hover {{ background:#ffe6c2; }}
    .fork-group-content {{ display:none; }}
    .fork-group-content.active {{ display:block; }}
    .child-group-continue {{ border-left: 3px solid #ffcc80; padding-left: 8px; margin: 6px 0 6px 0; background: transparent; }}
    .step-group {{ border: 1px solid #e0e4ee; border-left: 5px solid var(--step-accent, #667eea); border-radius: 8px; background: var(--step-bg, #ffffff); margin: 10px 0; overflow: hidden; }}
    .step-header {{ background: var(--step-head, #f3f5fb); padding: 10px 12px; cursor: pointer; display:flex; justify-content:space-between; align-items:center; gap:10px; }}
    .step-header:hover {{ filter: brightness(0.985); }}
    .step-color-1 {{ --step-accent:#667eea; --step-bg:#fbfcff; --step-head:#f0f3ff; }}
    .step-color-2 {{ --step-accent:#26a69a; --step-bg:#f8fffd; --step-head:#e9f8f6; }}
    .step-color-3 {{ --step-accent:#f57c00; --step-bg:#fffaf3; --step-head:#fff1dc; }}
    .step-color-4 {{ --step-accent:#ab47bc; --step-bg:#fff9ff; --step-head:#f7eafa; }}
    .step-color-5 {{ --step-accent:#42a5f5; --step-bg:#f8fcff; --step-head:#e8f4ff; }}
    .step-color-6 {{ --step-accent:#7cb342; --step-bg:#fbfff7; --step-head:#edf7e6; }}
    .step-title {{ font-weight:700; color:#333; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
    .step-meta {{ color:#666; font-size:0.84em; margin-top:3px; }}
    .badge-step {{ background:#eef7ff; color:#1976d2; border-color:#bbdefb; }}
    .step-content {{ display:none; padding:12px; }}
    .step-content.active {{ display:block; }}
    .request-header {{ background: #f8f9fa; padding: 12px; border-radius: 8px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
    .request-header:hover {{ background: #eceff3; }}
    .request-id {{ font-weight: bold; color: #667eea; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
    .badge-pill {{ display:inline-block; border-radius:6px; padding:2px 8px; font-size:0.82em; font-weight:700; border:1px solid transparent; }}
    .badge-time {{ background:#eef2ff; color:#4154c5; border-color:#cfd7ff; }}
    .badge-type {{ background:#e8f5e9; color:#2e7d32; border-color:#c8e6c9; }}
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
  </style>
</head>
<body>
  <div class="container">
  <div class="header">
    <h1>LLM IO Trace 性能报告</h1>
    <p>根 session: {_esc(root_session)} · 生成: {_esc(generated_at)}</p>
  </div>

  <div class="stats-grid">
    <div class="stat-card"><div class="stat-value">{tot["rounds"]}</div><div class="stat-label">思考轮次</div></div>
    <div class="stat-card"><div class="stat-value">{tot["llm_wall_sec"]}s</div><div class="stat-label">总思考时间</div></div>
    <div class="stat-card"><div class="stat-value">{tool_calls_count}</div><div class="stat-label">工具次数</div></div>
    <div class="stat-card"><div class="stat-value">{round(gap_sec, 3)}s</div><div class="stat-label">工具调用总时间</div></div>
    <div class="stat-card"><div class="stat-value">{task_total_sec}s</div><div class="stat-label">总任务时间</div></div>
    <div class="stat-card token-card">
      <div class="stat-value">{total_tokens_sum or "—"}</div>
      <div class="stat-label">总token消耗</div>
      <div class="token-lines"><span>input：{input_tokens_sum}</span><span>output：{output_tokens_sum}</span></div>
    </div>
  </div>
  <div class="chart-container">
    <div class="chart-title">Step 耗时总览</div>
    <div class="chart-wrapper"><canvas id="global-step-duration-chart"></canvas></div>
  </div>
"""

    blocks: list[str] = []
    idx = 1
    group_idx = 1
    step_chart_rows: list[dict[str, Any]] = []
    step_detail_charts: list[dict[str, Any]] = []
    blocks.append(f'<div class="section"><div class="section-title">session id：{_esc(root_session)}</div>')

    # 外层按 request_id 分组（同一个 session 可有多个 request_id）
    for rid, items_for_rid in request_groups.items():
        rid_toggle = f"toggleBlock('rid-{_esc(rid)}', this)"
        rid_hdr_open = f'<div class="collapsible-header" onclick="{rid_toggle}">'
        rid_hdr_body = (
            f"<span><strong>request_id:</strong> {_esc(rid)}</span>"
            '<span class="arrow">▼</span>'
        )
        blocks.append(f"{rid_hdr_open}{rid_hdr_body}</div>")
        blocks.append(f'<div class="collapsible-content active request-group-content" id="rid-{_esc(rid)}">')
        for step_no, (step_label, step_items) in enumerate(_split_steps(items_for_rid), 1):
            stats = _step_stats(step_items)
            step_id = f"step-{idx}-{step_no}"
            step_chart_id = f"step-chart-{idx}-{step_no}"
            color_cls = f"step-color-{((step_no - 1) % 6) + 1}"
            display_label = f"Step {len(step_chart_rows) + 1}"
            step_chart_rows.append(
                {
                    "label": display_label,
                    "title": step_label,
                    "model": stats["model_sec"],
                    "tool": stats["tool_sec"],
                    "other": stats["other_sec"],
                    "total": stats["duration"],
                    "input": stats["input"],
                    "output": stats["output"],
                    "tokenTotal": stats["total"],
                    "toolCalls": stats["tool_calls"],
                }
            )
            step_detail_charts.append(
                {
                    "id": step_chart_id,
                    "title": step_label,
                    **_step_event_chart_data(step_items),
                }
            )
            step_meta = (
                f"工具耗时：{stats['tool_sec']}s · 工具次数：{stats['tool_calls']} · "
                f"token消耗：input：{stats['input']} output：{stats['output']} "
                f"total：{stats['total']}"
            )
            blocks.append(
                f"""
<div class="step-group {color_cls}">
  <div class="step-header" onclick="toggleBlock('{step_id}', this)">
    <div>
      <div class="step-title">
        <span class="badge-pill badge-time">{stats["duration"]}s</span>
        <span class="badge-pill badge-step">Step {step_no}</span>
        <span class="title-main">{_esc(step_label)}</span>
      </div>
      <div class="step-meta">{step_meta}</div>
    </div>
    <span class="arrow">▼</span>
  </div>
  <div class="step-content" id="{step_id}">
    <div class="step-chart-wrapper"><canvas id="{step_chart_id}"></canvas></div>
"""
            )
            _append_timeline_items(step_items)
            blocks.append("</div></div>")
        blocks.append("</div>")
    blocks.append("</div>")

    if post_final_merged:
        post_input = post_final_tot.get("input_tokens_sum") or 0
        post_output = post_final_tot.get("output_tokens_sum") or 0
        post_total = post_final_tot.get("total_tokens_sum") or 0
        post_gap_sec = round(post_final_gap_sec, 3)
        post_meta = (
            f"模型轮次：{post_final_tot['rounds']} · "
            f"模型耗时：{post_final_tot['llm_wall_sec']}s · "
            f"流程间隙：{post_gap_sec}s · "
            f"token：input {post_input} / output {post_output} / total {post_total}"
        )
        blocks.append(
            f"""
<div class="section">
  <div class="section-title">chat.final 后后台流程（不计入主流程总览）</div>
  <div class="meta-line">{post_meta}</div>
"""
        )
        _append_timeline_items(post_final_merged)
        blocks.append("</div>")

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

const globalStepRows = __GLOBAL_STEP_ROWS__;
const stepDetailCharts = __STEP_DETAIL_CHARTS__;

function renderGlobalStepChart() {
  const el = document.getElementById('global-step-duration-chart');
  if (!el || !window.Chart) return;
  new Chart(el, {
    type: 'bar',
    data: {
      labels: globalStepRows.map(x => x.label),
      datasets: [
        { label: '模型调用耗时', data: globalStepRows.map(x => x.model), backgroundColor: '#4caf50' },
        { label: '工具调用耗时', data: globalStepRows.map(x => x.tool), backgroundColor: '#f57c00' },
        { label: '其他耗时', data: globalStepRows.map(x => x.other), backgroundColor: '#9e9e9e' }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: {
          callbacks: {
            afterTitle: (items) => globalStepRows[items[0].dataIndex]?.title || '',
            afterBody: (items) => {
              const row = globalStepRows[items[0].dataIndex] || {};
              return [
                '总耗时：' + (row.total || 0) + 's',
                '模型耗时：' + (row.model || 0) + 's',
                '工具耗时：' + (row.tool || 0) + 's',
                '其他耗时：' + (row.other || 0) + 's',
                '工具次数：' + (row.toolCalls || 0),
                'token input：' + (row.input || 0),
                'token output：' + (row.output || 0),
                'token total：' + (row.tokenTotal || 0)
              ];
            }
          }
        }
      },
      scales: {
        x: { stacked: true },
        y: { stacked: true, title: { display: true, text: '耗时（秒）' } }
      }
    }
  });
}

function renderStepDetailCharts() {
  if (!window.Chart) return;
  stepDetailCharts.forEach((cfg) => {
    const el = document.getElementById(cfg.id);
    if (!el) return;
    new Chart(el, {
      type: 'bar',
      data: {
        labels: cfg.labels,
        datasets: [{ label: '耗时（秒）', data: cfg.values, backgroundColor: cfg.colors }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: cfg.title, align: 'start' },
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => cfg.labels[items[0].dataIndex] || '',
              label: (item) => '耗时：' + item.raw + 's',
              afterLabel: (item) => cfg.details[item.dataIndex] || ''
            }
          }
        },
        scales: {
          x: { title: { display: true, text: '耗时（秒）' } },
          y: { ticks: { autoSkip: false, font: { size: 11 } } }
        }
      }
    });
  });
}

renderGlobalStepChart();
renderStepDetailCharts();
</script>
</div>
</body>
</html>
""".replace("__GLOBAL_STEP_ROWS__", _js_json(step_chart_rows)).replace(
        "__STEP_DETAIL_CHARTS__", _js_json(step_detail_charts)
    )
    return head + "\n".join(blocks) + foot


def write_report(path: Path, html_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
