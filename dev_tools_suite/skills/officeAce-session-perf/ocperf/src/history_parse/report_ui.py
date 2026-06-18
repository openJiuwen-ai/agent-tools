"""Shared HTML fragments: session tree, top consumers, todo-phased timeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Sequence

from history_parse.models import LLMRound, SessionPath, ToolExecution
from history_parse.session import child_info, is_measurable_tool, session_path
from history_parse.todo_tracker import TodoTimeline

TimelineKind = Literal["llm", "tool"]
TimelineItem = tuple[TimelineKind, Any]


def _naive(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _item_start(kind: TimelineKind, obj: Any) -> datetime:
    if kind == "llm":
        return obj.request_ts
    if hasattr(obj, "start_ts"):
        return obj.start_ts
    return obj.after_output_ts  # ToolGap (full.json)


def _item_end(kind: TimelineKind, obj: Any) -> datetime:
    if kind == "llm":
        return obj.output_ts
    if hasattr(obj, "end_ts"):
        return obj.end_ts
    return obj.next_request_ts  # ToolGap


def _item_duration(kind: TimelineKind, obj: Any) -> float:
    return float(getattr(obj, "duration_sec", 0.0))


def _item_session_id(kind: TimelineKind, obj: Any) -> str:
    return str(obj.session_id or "")


def _item_child_meta(kind: TimelineKind, obj: Any) -> tuple[bool, str, SessionPath]:
    return bool(obj.is_child_session), str(obj.child_label or ""), obj.child_path or ()


@dataclass
class TopConsumerRow:
    rank: int
    kind: TimelineKind
    title: str
    duration_sec: float
    detail: str
    session_label: str = ""
    is_child: bool = False


@dataclass
class SessionLaneStats:
    session_id: str
    label: str
    depth: int
    path: SessionPath
    kind: str = "root"  # root | subagent | forkagent | child
    llm_count: int = 0
    tool_count: int = 0
    llm_sec: float = 0.0
    tool_sec: float = 0.0


@dataclass
class SessionTreeNode:
    session_id: str
    label: str
    kind: str
    stats: SessionLaneStats | None = None
    children: list["SessionTreeNode"] = field(default_factory=list)


@dataclass
class ExecutionPhase:
    phase_id: str
    title: str
    subtitle: str
    start: datetime
    end: datetime
    level: int  # 0=batch, 1=task, 2=unassigned
    parent_id: str | None = None


def compute_top_llm(rounds: Sequence[Any], limit: int = 5) -> list[TopConsumerRow]:
    ranked = sorted(rounds, key=lambda r: -float(r.duration_sec))
    out: list[TopConsumerRow] = []
    for i, r in enumerate(ranked[:limit], 1):
        is_child, label, _ = _item_child_meta("llm", r)
        tok = f"in {r.input_tokens or 0} · out {r.output_tokens or 0}"
        cache = getattr(r, "cache_tokens", None) or 0
        if cache:
            tok += f" · cache {cache}"
        model = getattr(r, "model_name", "") or ""
        detail = f"{model} · {tok}".strip(" ·")
        if hasattr(r, "ttft_sec"):
            detail += f" · TTFT {r.ttft_sec:.2f}s + 推理 {r.inference_sec:.2f}s"
        rid = getattr(r, "request_id", "") or ""
        title = f"模型 · {rid[:16]}…" if len(rid) > 16 else (f"模型 · {rid}" if rid else f"模型轮次 #{i}")
        out.append(
            TopConsumerRow(
                rank=i,
                kind="llm",
                title=title,
                duration_sec=float(r.duration_sec),
                detail=detail,
                session_label=label if is_child else "根会话",
                is_child=is_child,
            )
        )
    return out


def compute_top_tool_windows(
    gaps: Sequence[Any],
    limit: int = 5,
) -> list[TopConsumerRow]:
    """Rank full.json tool execution windows (not single-tool precision)."""
    ranked = sorted(gaps, key=lambda g: -float(g.duration_sec))
    out: list[TopConsumerRow] = []
    for i, g in enumerate(ranked[:limit], 1):
        is_child = bool(getattr(g, "is_child_session", False))
        label = str(getattr(g, "child_label", "") or "")
        title = str(getattr(g, "tools_triggered", None) or "工具窗口")
        out.append(
            TopConsumerRow(
                rank=i,
                kind="tool",
                title=title[:80],
                duration_sec=float(g.duration_sec),
                detail="full 工具执行窗口（output → 下次 request）",
                session_label=label if is_child else "根会话",
                is_child=is_child,
            )
        )
    return out


def compute_top_tools_by_name(
    tools: Sequence[Any],
    limit: int = 5,
    *,
    skip_predicate: Callable[[Any], bool] | None = None,
) -> list[TopConsumerRow]:
    """Aggregate wall time by tool name (sum of durations, sorted)."""
    pool = [t for t in tools if not (skip_predicate and skip_predicate(t))]
    by_name: dict[str, dict[str, Any]] = {}
    for t in pool:
        name = str(getattr(t, "name", "?") or "?")
        entry = by_name.setdefault(
            name,
            {"duration_sec": 0.0, "count": 0, "max_single": 0.0},
        )
        d = float(t.duration_sec)
        entry["duration_sec"] += d
        entry["count"] += 1
        entry["max_single"] = max(entry["max_single"], d)
    ranked = sorted(by_name.items(), key=lambda x: -x[1]["duration_sec"])
    out: list[TopConsumerRow] = []
    for i, (name, agg) in enumerate(ranked[:limit], 1):
        out.append(
            TopConsumerRow(
                rank=i,
                kind="tool",
                title=name,
                duration_sec=round(agg["duration_sec"], 3),
                detail=f"{agg['count']} 次 · 单次最大 {agg['max_single']:.3f}s",
                session_label="按名称汇总",
                is_child=False,
            )
        )
    return out


def compute_top_tools(
    tools: Sequence[Any],
    limit: int = 5,
    *,
    skip_predicate: Callable[[Any], bool] | None = None,
) -> list[TopConsumerRow]:
    pool = [t for t in tools if not (skip_predicate and skip_predicate(t))]
    ranked = sorted(pool, key=lambda t: -float(t.duration_sec))
    out: list[TopConsumerRow] = []
    for i, t in enumerate(ranked[:limit], 1):
        is_child, label, _ = _item_child_meta("tool", t)
        name = getattr(t, "name", "?")
        out.append(
            TopConsumerRow(
                rank=i,
                kind="tool",
                title=str(name),
                duration_sec=float(t.duration_sec),
                detail=f"tool_call_id: {getattr(t, 'tool_call_id', '—')[:24]}",
                session_label=label if is_child else "根会话",
                is_child=is_child,
            )
        )
    return out


def render_top_consumers_html(
    llm_rows: list[TopConsumerRow],
    tool_rows: list[TopConsumerRow],
    esc: Callable[[str], str],
    *,
    tool_panel_title: str = "工具调用",
    tool_by_name_rows: list[TopConsumerRow] | None = None,
) -> str:
    def _highlights() -> str:
        parts: list[str] = []
        if llm_rows:
            r = llm_rows[0]
            parts.append(
                f'<div class="top-highlight llm"><span class="top-highlight-label">最慢模型</span>'
                f'<span class="top-highlight-val">{r.duration_sec:.3f}s</span>'
                f'<span class="top-highlight-sub">{esc(r.detail[:60])}</span></div>'
            )
        if tool_rows:
            r = tool_rows[0]
            parts.append(
                f'<div class="top-highlight tool"><span class="top-highlight-label">最慢工具(单次)</span>'
                f'<span class="top-highlight-val">{r.duration_sec:.3f}s</span>'
                f'<span class="top-highlight-sub">{esc(r.title)}</span></div>'
            )
        if tool_by_name_rows:
            r = tool_by_name_rows[0]
            parts.append(
                f'<div class="top-highlight tool-sum"><span class="top-highlight-label">工具累计最高</span>'
                f'<span class="top-highlight-val">{r.duration_sec:.3f}s</span>'
                f'<span class="top-highlight-sub">{esc(r.title)} · {esc(r.detail)}</span></div>'
            )
        if not parts:
            return ""
        return f'<div class="top-highlights">{"".join(parts)}</div>'

    def _table(rows: list[TopConsumerRow], kind_label: str, accent: str) -> str:
        if not rows:
            return f"<p class='meta-line'>无 {kind_label} 记录。</p>"
        body = ""
        for r in rows:
            child = f"<span class='lane-tag child'>{esc(r.session_label)}</span>" if r.is_child else ""
            body += (
                f"<tr><td>{r.rank}</td>"
                f"<td><strong>{esc(r.title)}</strong>{child}</td>"
                f"<td class='top-dur' style='color:{accent}'>{r.duration_sec:.3f}s</td>"
                f"<td class='meta-line'>{esc(r.detail)}</td></tr>"
            )
        return (
            f"<table class='top-table'><thead><tr><th>#</th><th>{kind_label}</th>"
            f"<th>耗时</th><th>说明</th></tr></thead><tbody>{body}</tbody></table>"
        )

    by_name_block = ""
    if tool_by_name_rows:
        by_name_block = f"""
    <div class="top-panel top-panel-wide">
      <h4 class="top-panel-title">工具按名称累计 Top</h4>
      {_table(tool_by_name_rows, "工具名", "#bf360c")}
    </div>
"""

    return f"""
<div class="section top-consumers-section">
  <div class="section-title">耗时 Top 排行</div>
  {_highlights()}
  <div class="top-grid">
    <div class="top-panel">
      <h4 class="top-panel-title">模型调用（LLM）· 单次最慢</h4>
      {_table(llm_rows, "模型", "#2e7d32")}
    </div>
    <div class="top-panel">
      <h4 class="top-panel-title">{esc(tool_panel_title)}</h4>
      {_table(tool_rows, "工具", "#e65100")}
    </div>
  </div>
  {by_name_block}
</div>
"""


def collect_session_lanes(
    merged: list[TimelineItem],
    root_session: str,
) -> list[SessionLaneStats]:
    lanes: dict[str, SessionLaneStats] = {}

    def _lane(sid: str, is_child: bool, label: str, path: SessionPath) -> SessionLaneStats:
        if sid not in lanes:
            depth = len(path) if path else 0
            display = "根会话 · " + root_session[:20] + ("…" if len(root_session) > 20 else "")
            if is_child:
                display = " → ".join(p[1] for p in path) if path else label
            node_kind = path[-1][2] if path else "root"
            lanes[sid] = SessionLaneStats(
                session_id=sid,
                label=display,
                depth=depth,
                path=path,
                kind=node_kind,
            )
        return lanes[sid]

    for kind, obj in merged:
        sid = _item_session_id(kind, obj)
        is_child, label, path = _item_child_meta(kind, obj)
        lane = _lane(sid, is_child, label, path)
        dur = _item_duration(kind, obj)
        if kind == "llm":
            lane.llm_count += 1
            lane.llm_sec += dur
        else:
            if is_measurable_tool(getattr(obj, "name", None)):
                lane.tool_count += 1
                lane.tool_sec += dur

    if root_session not in lanes:
        lanes[root_session] = SessionLaneStats(
            session_id=root_session,
            label="根会话",
            depth=0,
            path=(),
            kind="root",
        )

    def _sort_key(lane: SessionLaneStats) -> tuple:
        if lane.session_id == root_session:
            return (-1, 0, lane.session_id)
        return (lane.depth, len(lane.session_id), lane.session_id)

    return sorted(lanes.values(), key=_sort_key)


def _parent_session_id(session_id: str, root_session: str, known: set[str]) -> str:
    if session_id == root_session:
        return ""
    best = root_session
    for cand in known:
        if cand == session_id:
            continue
        if not session_id.startswith(cand):
            continue
        if len(cand) <= len(best):
            continue
        if len(cand) == len(root_session) or session_id[len(cand)] == "_":
            best = cand
    return best


def build_session_tree(
    lanes: list[SessionLaneStats],
    root_session: str,
) -> SessionTreeNode | None:
    if not lanes:
        return None
    by_id = {lane.session_id: lane for lane in lanes}
    known = set(by_id)
    nodes: dict[str, SessionTreeNode] = {}
    for sid, lane in by_id.items():
        kind = lane.kind if sid == root_session else lane.kind
        label = lane.label if sid == root_session else lane.label
        nodes[sid] = SessionTreeNode(
            session_id=sid,
            label=label,
            kind=kind,
            stats=lane,
        )
    if root_session not in nodes:
        nodes[root_session] = SessionTreeNode(
            session_id=root_session,
            label="根会话",
            kind="root",
            stats=None,
        )
    for sid in sorted(by_id, key=lambda s: (by_id[s].depth, len(s), s)):
        if sid == root_session:
            continue
        parent_sid = _parent_session_id(sid, root_session, known)
        parent = nodes.get(parent_sid) or nodes[root_session]
        if sid not in [c.session_id for c in parent.children]:
            parent.children.append(nodes[sid])
    for node in nodes.values():
        node.children.sort(key=lambda c: (c.stats.depth if c.stats else 0, c.session_id))
    return nodes[root_session]


def _render_tree_node(node: SessionTreeNode, esc: Callable[[str], str], depth: int) -> str:
    lane = node.stats
    kind_cls = f"tree-kind-{node.kind}"
    stats_html = ""
    if lane:
        stats_html = (
            f'<span class="tree-stat">模型 {lane.llm_count}× · {lane.llm_sec:.2f}s</span>'
            f'<span class="tree-stat">工具 {lane.tool_count}× · {lane.tool_sec:.2f}s</span>'
        )
    sid_short = esc(node.session_id[:56] + ("…" if len(node.session_id) > 56 else ""))
    children_html = ""
    if node.children:
        inner = "".join(_render_tree_node(c, esc, depth + 1) for c in node.children)
        children_html = f'<ul class="session-tree-children">{inner}</ul>'
    return f"""
<li class="session-tree-node {kind_cls}" data-depth="{depth}">
  <div class="session-tree-node-card">
    <span class="tree-connector">{"└─" if depth else "●"}</span>
    <div class="tree-node-main">
      <span class="session-lane-title">{esc(node.label)}</span>
      <code class="session-lane-id">{sid_short}</code>
      <div class="session-lane-stats">{stats_html}</div>
    </div>
  </div>
  {children_html}
</li>
"""


def render_session_hierarchy_html(
    lanes: list[SessionLaneStats],
    root_session: str,
    esc: Callable[[str], str],
    *,
    timeline_indent_note: bool = True,
) -> str:
    child_lanes = [lane for lane in lanes if lane.session_id != root_session or lane.depth > 0]
    if not child_lanes and len(lanes) <= 1:
        root_lane = next((lane for lane in lanes if lane.session_id == root_session), None)
        if not root_lane or (root_lane.llm_count == 0 and root_lane.tool_count == 0):
            return ""

    root = build_session_tree(lanes, root_session)
    if not root:
        return ""
    tree_body = _render_tree_node(root, esc, 0)
    timeline_note = (
        "下方时间线按深度缩进对应。"
        if timeline_indent_note
        else "下方时间线以橙色标签区分子 Agent，不再额外缩进。"
    )
    return f"""
<div class="section session-tree-section">
  <div class="section-title">子 Agent / 会话层次结构</div>
  <p class="meta-line">树形展示 <code>session_id</code> 嵌套（<span class="tree-legend subagent">subagent</span>
    <span class="tree-legend forkagent">fork</span>）；{timeline_note}</p>
  <ul class="session-tree-root">{tree_body}</ul>
</div>
"""


def build_execution_phases(
    todo: TodoTimeline | None,
    session_start: datetime,
    session_end: datetime,
) -> list[ExecutionPhase]:
    if not todo or not todo.batches:
        return [
            ExecutionPhase(
                phase_id="all",
                title="全时段",
                subtitle="无 Todo 路线图",
                start=session_start,
                end=session_end,
                level=2,
            )
        ]

    phases: list[ExecutionPhase] = []
    batches = sorted(todo.batches, key=lambda b: b.created_at)
    seen_task_phases: set[str] = set()
    for i, batch in enumerate(batches):
        batch_end = batches[i + 1].created_at if i + 1 < len(batches) else session_end
        bid = f"batch-{batch.index}"
        phases.append(
            ExecutionPhase(
                phase_id=bid,
                title=f"路线图 {batch.index}",
                subtitle=batch.label[:80],
                start=batch.created_at,
                end=batch_end,
                level=0,
            )
        )
        for tid in batch.task_ids:
            task = todo.tasks.get(tid)
            if not task:
                continue
            phase_id = f"task-{tid}"
            if phase_id in seen_task_phases:
                continue
            seen_task_phases.add(phase_id)
            t_start = task.started_at or task.created_at
            t_end = task.completed_at or batch_end
            if _naive(t_end) <= _naive(t_start):
                t_end = batch_end
            if _naive(t_end) <= _naive(t_start):
                continue
            content = (task.content or tid)[:70].replace("\n", " ")
            phases.append(
                ExecutionPhase(
                    phase_id=phase_id,
                    title=f"任务 · {content}",
                    subtitle=f"状态 {task.current_status}",
                    start=t_start,
                    end=t_end,
                    level=1,
                    parent_id=bid,
                )
            )

    phases.append(
        ExecutionPhase(
            phase_id="unassigned",
            title="未归属 Todo 阶段",
            subtitle="不在任何任务执行窗口内的事件",
            start=session_start,
            end=session_end,
            level=2,
        )
    )
    return phases


def _pick_phase(ts: datetime, phases: list[ExecutionPhase]) -> str:
    t = _naive(ts)
    task_hits: list[ExecutionPhase] = []
    batch_hits: list[ExecutionPhase] = []
    for p in phases:
        if p.phase_id == "unassigned":
            continue
        if _naive(p.start) <= t < _naive(p.end):
            if p.level == 1:
                task_hits.append(p)
            elif p.level == 0:
                batch_hits.append(p)
    if task_hits:
        return min(
            task_hits,
            key=lambda p: (_naive(p.end) - _naive(p.start)).total_seconds(),
        ).phase_id
    if batch_hits:
        return max(batch_hits, key=lambda p: _naive(p.start)).phase_id
    all_phase = next((p for p in phases if p.phase_id == "all"), None)
    if all_phase:
        return "all"
    return "unassigned"


def _pick_phase_for_item(kind: TimelineKind, obj: Any, phases: list[ExecutionPhase]) -> str:
    start = _item_start(kind, obj)
    end = _item_end(kind, obj)
    mid = start + (end - start) / 2
    return _pick_phase(mid, phases)


def partition_by_phase(
    merged: list[TimelineItem],
    phases: list[ExecutionPhase],
) -> dict[str, list[tuple[int, TimelineKind, Any]]]:
    buckets: dict[str, list] = {p.phase_id: [] for p in phases}
    for idx, (kind, obj) in enumerate(merged, 1):
        pid = _pick_phase_for_item(kind, obj, phases)
        buckets.setdefault(pid, []).append((idx, kind, obj))
    return buckets


def render_phased_timeline_shell(
    phases: list[ExecutionPhase],
    buckets: dict[str, list[tuple[int, TimelineKind, Any]]],
    esc: Callable[[str], str],
    render_item: Callable[[int, TimelineKind, Any], str],
    *,
    default_open: bool = True,
) -> str:
    phase_by_id = {p.phase_id: p for p in phases}
    batches = [p for p in phases if p.level == 0]
    tasks_by_parent: dict[str, list[ExecutionPhase]] = {}
    for p in phases:
        if p.level == 1 and p.parent_id:
            tasks_by_parent.setdefault(p.parent_id, []).append(p)

    def _render_items(items: list[tuple[int, TimelineKind, Any]]) -> str:
        inner = "".join(render_item(idx, kind, obj) for idx, kind, obj in items)
        return inner or "<p class='meta-line'>本阶段无模型/工具事件。</p>"

    def _phase_block(
        p: ExecutionPhase,
        inner_html: str,
        item_count: int,
        open_cls: str,
        extra_cls: str = "",
    ) -> str:
        dur = max(0.0, (_naive(p.end) - _naive(p.start)).total_seconds())
        return f"""
<div class="todo-phase-block {extra_cls}">
  <div class="todo-phase-header" onclick="toggleBlock('phase-{esc(p.phase_id)}', this)">
    <div>
      <strong>{esc(p.title)}</strong>
      <span class="phase-sub">{esc(p.subtitle)}</span>
      <span class="title-time">{esc(p.start.strftime("%H:%M:%S"))} – {esc(p.end.strftime("%H:%M:%S"))} · {dur:.1f}s · {item_count} 项</span>
    </div>
    <span class="arrow {'rotated' if default_open else ''}">▼</span>
  </div>
  <div class="todo-phase-body request-details {open_cls}" id="phase-{esc(p.phase_id)}">
    {inner_html}
  </div>
</div>
"""

    parts: list[str] = []
    open_cls = "active" if default_open else ""

    if batches:
        for batch in batches:
            batch_items = buckets.get(batch.phase_id, [])
            task_phases = tasks_by_parent.get(batch.phase_id, [])
            task_inner_parts: list[str] = []
            for tp in task_phases:
                t_items = buckets.get(tp.phase_id, [])
                task_inner_parts.append(
                    _phase_block(tp, _render_items(t_items), len(t_items), open_cls, "todo-task-phase phase-level-1")
                )
            batch_direct = ""
            if batch_items:
                batch_direct = (
                    '<div class="todo-batch-direct"><p class="meta-line">路线图级事件（未落入具体任务窗口）</p>'
                    + _render_items(batch_items)
                    + "</div>"
                )
            inner = "".join(task_inner_parts) + batch_direct
            if not inner:
                inner = "<p class='meta-line'>本路线图阶段无模型/工具事件。</p>"
            total = len(batch_items) + sum(len(buckets.get(t.phase_id, [])) for t in task_phases)
            parts.append(_phase_block(batch, inner, total, open_cls, "todo-batch-phase phase-level-0"))
    else:
        for p in phases:
            if p.phase_id == "unassigned":
                continue
            items = buckets.get(p.phase_id, [])
            parts.append(
                _phase_block(p, _render_items(items), len(items), open_cls, f"phase-level-{p.level}")
            )

    unassigned = phase_by_id.get("unassigned")
    if unassigned:
        u_items = buckets.get("unassigned", [])
        if u_items:
            parts.append(
                _phase_block(unassigned, _render_items(u_items), len(u_items), open_cls, "phase-level-2")
            )
    return "".join(parts)


def _safe_dom_id(session_id: str) -> str:
    sid = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
    return sid[:80] if sid else "agent_unknown"


_TIMELINE_BLOCK_ID_STEMS = ("req-", "in-", "think-", "out-", "tools-", "gap-")


def prefix_timeline_block_ids(html: str, prefix: str) -> str:
    """Give timeline block ids a unique prefix (agent pane vs global chrono pane)."""
    if not prefix:
        return html
    for stem in _TIMELINE_BLOCK_ID_STEMS:
        html = html.replace(f'id="{stem}', f'id="{prefix}{stem}')
        html = html.replace(f"toggleBlock('{stem}", f"toggleBlock('{prefix}{stem}")
    return html


def partition_by_agent(
    merged: list[TimelineItem],
) -> dict[str, list[tuple[int, TimelineKind, Any]]]:
    buckets: dict[str, list] = {}
    for idx, (kind, obj) in enumerate(merged, 1):
        sid = _item_session_id(kind, obj)
        buckets.setdefault(sid, []).append((idx, kind, obj))
    return buckets


def render_agent_timeline_shell(
    lanes: list[SessionLaneStats],
    agent_buckets: dict[str, list[tuple[int, TimelineKind, Any]]],
    esc: Callable[[str], str],
    render_item: Callable[[int, TimelineKind, Any], str],
    *,
    root_session: str,
) -> str:
    active_lanes = [lane for lane in lanes if agent_buckets.get(lane.session_id)]
    if not active_lanes:
        return '<p class="muted">无按 Agent 可分组的事件。</p>'
    if len(active_lanes) <= 1 and active_lanes[0].session_id == root_session:
        return (
            '<p class="muted">本会话无子 Agent，仅根会话有事件；请使用「时间顺序」视图。</p>'
        )

    nav: list[str] = ['<div class="agent-timeline-nav" role="tablist">']
    panels: list[str] = ['<div class="agent-timeline-panels">']

    first_panel = True
    for lane in active_lanes:
        items = agent_buckets.get(lane.session_id, [])
        if not items:
            continue
        dom_id = esc(_safe_dom_id(lane.session_id))
        active = " active" if first_panel else ""
        first_panel = False
        kind_badge = ""
        if lane.kind == "subagent":
            kind_badge = ' <span class="agent-kind-badge subagent">subagent</span>'
        elif lane.kind == "forkagent":
            kind_badge = ' <span class="agent-kind-badge fork">fork</span>'
        elif lane.session_id == root_session:
            kind_badge = ' <span class="agent-kind-badge root">根</span>'

        llm_n = sum(1 for _, k, _ in items if k == "llm")
        tool_n = sum(1 for _, k, _ in items if k == "tool")
        nav.append(
            f'<button type="button" class="agent-timeline-btn{active}" role="tab" '
            f'data-agent-dom="{dom_id}" onclick="agentTimelineSelect(this)">'
            f'{esc(lane.label)}{kind_badge}'
            f'<span class="agent-btn-stats">LLM {llm_n} · 工具 {tool_n}</span></button>'
        )

        block_prefix = f"agent-{dom_id}-"
        inner = "".join(
            prefix_timeline_block_ids(render_item(idx, kind, obj), block_prefix)
            for idx, kind, obj in items
        )
        if not inner:
            inner = "<p class='meta-line'>本 Agent 无模型/工具事件。</p>"

        panels.append(
            f'<div class="agent-timeline-panel{active}" id="agent-panel-{dom_id}" '
            f'data-agent-dom="{dom_id}" role="tabpanel">'
            f'<div class="agent-panel-header">'
            f'<h4>{esc(lane.label)}</h4>'
            f'<code class="session-lane-id">{esc(lane.session_id[:72])}'
            f'{"…" if len(lane.session_id) > 72 else ""}</code>'
            f'<p class="meta-line">模型 {lane.llm_count} 次 · {lane.llm_sec:.2f}s · '
            f"工具/窗口 {lane.tool_count} 次 · {lane.tool_sec:.2f}s</p>"
            f"</div>"
            f'<div class="agent-panel-body">{inner}</div>'
            f"</div>"
        )

    nav.append("</div>")
    panels.append("</div>")
    return "".join(nav) + "".join(panels)


@dataclass(frozen=True)
class TimelineDualViewRequest:
    chronological_html: str
    lanes: list[SessionLaneStats]
    merged: list[TimelineItem]
    root_session: str
    esc: Callable[[str], str]
    render_item: Callable[[int, TimelineKind, Any], str]


def render_timeline_dual_view(request: TimelineDualViewRequest) -> str:
    agent_buckets = partition_by_agent(request.merged)
    agent_html = render_agent_timeline_shell(
        request.lanes,
        agent_buckets,
        request.esc,
        request.render_item,
        root_session=request.root_session,
    )
    chronological_html = request.chronological_html
    return f"""
<div class="timeline-dual-view">
  <div class="timeline-view-tabs">
    <span class="timeline-view-label">时间线视图：</span>
    <button type="button" class="timeline-view-btn active" data-timeline-view="chrono"
      onclick="timelineSetView('chrono')">时间顺序（全局）</button>
    <button type="button" class="timeline-view-btn" data-timeline-view="agent"
      onclick="timelineSetView('agent')">按 Agent 查看</button>
  </div>
  <p class="meta-line timeline-view-hint">
    「时间顺序」保留原有按时间戳排列的全局时间线；「按 Agent」可切换子 Agent 标签页，点击条目可展开模型输入、思考、输出与 tool_calls（与全局视图相同）。
  </p>
  <div id="timeline-pane-chrono" class="timeline-view-pane active">{chronological_html}</div>
  <div id="timeline-pane-agent" class="timeline-view-pane">{agent_html}</div>
</div>
"""


def agent_timeline_styles() -> str:
    return """
    .timeline-dual-view { margin-top: 4px; }
    .timeline-view-tabs {
      display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem;
      margin-bottom: 0.5rem; padding: 0.5rem 0.75rem;
      background: #eef0fa; border-radius: 8px; border: 1px solid #c5cae9;
    }
    .timeline-view-label { font-weight: 600; font-size: 0.9rem; color: #3949ab; }
    .timeline-view-btn {
      padding: 0.45rem 0.95rem; border: 1px solid #9fa8da; border-radius: 6px;
      background: #fff; cursor: pointer; font-size: 0.88rem;
    }
    .timeline-view-btn.active { background: #3949ab; color: #fff; border-color: #3949ab; }
    .timeline-view-hint { margin-bottom: 0.75rem; }
    .timeline-view-pane { display: none; }
    .timeline-view-pane.active { display: block; }
    .agent-timeline-nav {
      display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1rem;
      max-height: 200px; overflow-y: auto; padding: 0.25rem;
      border-bottom: 2px solid #e8eaf2;
    }
    .agent-timeline-btn {
      display: flex; flex-direction: column; align-items: flex-start;
      padding: 0.5rem 0.75rem; border: 1px solid #cbd5e1; border-radius: 8px;
      background: #fff; cursor: pointer; font-size: 0.85rem; text-align: left;
      min-width: 140px; max-width: 280px;
    }
    .agent-timeline-btn.active {
      background: #3949ab; color: #fff; border-color: #3949ab;
    }
    .agent-timeline-btn.active .agent-btn-stats { color: #e8eaf6; }
    .agent-btn-stats { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
    .agent-kind-badge {
      display: inline-block; font-size: 0.7rem; padding: 1px 5px;
      border-radius: 4px; margin-left: 4px; vertical-align: middle;
    }
    .agent-kind-badge.root { background: #e8eaf6; color: #3949ab; }
    .agent-kind-badge.subagent { background: #fff3e0; color: #e65100; }
    .agent-kind-badge.fork { background: #f3e5f5; color: #7b1fa2; }
    .agent-timeline-btn.active .agent-kind-badge { opacity: 0.9; }
    .agent-timeline-panel { display: none; }
    .agent-timeline-panel.active { display: block; }
    .agent-panel-header {
      padding: 0.75rem 1rem; margin-bottom: 0.75rem;
      background: #f8f9fc; border-radius: 8px; border-left: 4px solid #3949ab;
    }
    .agent-panel-header h4 { margin: 0 0 0.25rem; font-size: 1rem; }
    .agent-panel-body { padding: 0 0.25rem 1rem; }
    """


def agent_timeline_scripts() -> str:
    return """
function timelineSetView(mode) {
  document.querySelectorAll('.timeline-view-btn').forEach(function(b) {
    b.classList.toggle('active', b.getAttribute('data-timeline-view') === mode);
  });
  var chrono = document.getElementById('timeline-pane-chrono');
  var agent = document.getElementById('timeline-pane-agent');
  if (chrono) chrono.classList.toggle('active', mode === 'chrono');
  if (agent) agent.classList.toggle('active', mode === 'agent');
}
function agentTimelineSelect(btn) {
  var domId = btn.getAttribute('data-agent-dom');
  document.querySelectorAll('.agent-timeline-btn').forEach(function(b) {
    b.classList.remove('active');
  });
  btn.classList.add('active');
  document.querySelectorAll('.agent-timeline-panel').forEach(function(p) {
    p.classList.toggle('active', p.getAttribute('data-agent-dom') === domId);
  });
}
"""


def enhanced_report_styles() -> str:
    return """
    .top-consumers-section .top-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 900px) { .top-consumers-section .top-grid { grid-template-columns: 1fr; } }
    .top-panel-title { font-size: 0.95em; color: #444; margin-bottom: 8px; }
    .top-table { width: 100%; border-collapse: collapse; font-size: 0.86em; }
    .top-table th, .top-table td { border: 1px solid #e0e4ee; padding: 8px; text-align: left; vertical-align: top; }
    .top-table th { background: #f5f7fa; }
    .top-dur { font-weight: 700; white-space: nowrap; }
    .lane-tag.child { display: inline-block; margin-left: 6px; font-size: 0.78em; background: #fff3e0; color: #e65100; padding: 1px 6px; border-radius: 4px; }
    .top-highlights { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
    .top-highlight { flex: 1; min-width: 200px; padding: 12px 14px; border-radius: 8px; border: 1px solid #e0e4ee; }
    .top-highlight.llm { background: #f1f8f4; border-left: 4px solid #2e7d32; }
    .top-highlight.tool { background: #fff8f0; border-left: 4px solid #e65100; }
    .top-highlight.tool-sum { background: #fff3e0; border-left: 4px solid #bf360c; }
    .top-highlight-label { display: block; font-size: 0.78em; color: #666; margin-bottom: 4px; }
    .top-highlight-val { font-size: 1.4em; font-weight: 800; color: #333; }
    .top-highlight-sub { display: block; font-size: 0.8em; color: #777; margin-top: 4px; }
    .top-panel-wide { grid-column: 1 / -1; margin-top: 8px; }
    .session-tree-root, .session-tree-children { list-style: none; margin: 0; padding: 0; }
    .session-tree-children { margin-left: 20px; padding-left: 12px; border-left: 2px dashed #e0e4ee; }
    .session-tree-node { margin: 8px 0; }
    .session-tree-node-card { display: flex; align-items: flex-start; gap: 8px; padding: 10px 12px; border-radius: 8px; border: 1px solid #e8eaf2; background: #fafbff; }
    .session-tree-node.tree-kind-root .session-tree-node-card { border-left: 4px solid #667eea; background: #f4f6ff; }
    .session-tree-node.tree-kind-subagent .session-tree-node-card { border-left: 4px solid #f57c00; background: #fffaf5; }
    .session-tree-node.tree-kind-forkagent .session-tree-node-card { border-left: 4px solid #7b1fa2; background: #faf5ff; }
    .tree-connector { font-family: monospace; color: #999; flex-shrink: 0; }
    .tree-node-main { flex: 1; min-width: 0; }
    .tree-stat { display: inline-block; margin-right: 14px; font-size: 0.84em; color: #555; }
    .tree-legend.subagent { color: #f57c00; font-weight: 600; margin-right: 8px; }
    .tree-legend.forkagent { color: #7b1fa2; font-weight: 600; }
    .session-lane-title { font-weight: 700; color: #333; display: block; }
    .session-lane-id { font-size: 0.78em; color: #888; word-break: break-all; }
    .session-lane-stats { margin-top: 4px; }
    .todo-batch-phase { border: 1px solid #c5cae9; border-radius: 10px; margin-bottom: 16px; overflow: hidden; border-left: 5px solid #3949ab; }
    .todo-batch-children { padding: 8px 12px 12px; background: #fafbff; }
    .todo-task-phase { border: 1px solid #e3f2fd; border-radius: 8px; margin: 10px 0; overflow: hidden; border-left: 4px solid #1976d2; background: #fff; }
    .todo-batch-direct { margin-top: 8px; padding-top: 8px; border-top: 1px dashed #dfe3eb; }
    .todo-phase-block { border: 1px solid #dfe3eb; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }
    .todo-phase-block.phase-level-2 { border-left: 4px solid #9e9e9e; }
    .todo-phase-header { background: #f0f3ff; padding: 10px 12px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
    .todo-phase-header:hover { background: #e8ecff; }
    .todo-phase-body { padding: 10px 12px 4px; }
    .phase-sub { color: #666; font-size: 0.85em; margin-left: 8px; }
    .session-group-wrap { margin-bottom: 16px; }
    .session-group-title { font-size: 0.92em; font-weight: 700; color: #f57c00; margin: 8px 0; padding: 6px 10px; background: #fff8ef; border-radius: 6px; }
    """


def wrap_item_with_session_depth(
    html_block: str,
    depth: int,
    is_child: bool,
) -> str:
    if not is_child or depth <= 0:
        return html_block
    margin = 16 + depth * 14
    return f'<div class="timeline-nested" style="margin-left:{margin}px">{html_block}</div>'
