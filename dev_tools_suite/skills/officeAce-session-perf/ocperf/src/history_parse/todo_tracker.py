"""Reconstruct task timelines from todo_create / todo_modify tool executions."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from history_parse.models import ToolExecution

_TASK_LINE_RE = re.compile(
    r"\[\s*([>xX✓\s]?)\s*\]\s*task_id:\s*"
    r"([a-f0-9-]{8}-[a-f0-9-]{4}-[a-f0-9-]{4}-[a-f0-9-]{4}-[a-f0-9-]{12})\s*,\s*content:\s*"
    r"(.+?)(?=\n\s*\[|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_TODO_TOOL_NAMES = frozenset({"todo_create", "todo_modify", "todo_update"})


@dataclass
class TodoStatusChange:
    status: str
    timestamp: datetime
    tool_name: str
    tool_call_id: str


@dataclass
class TodoTaskRecord:
    task_id: str
    content: str
    batch_index: int
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    current_status: str = "pending"
    changes: list[TodoStatusChange] = field(default_factory=list)

    def execution_sec(self) -> float | None:
        start = self.started_at or self.created_at
        end = self.completed_at
        if end is None:
            return None
        return round(max(0.0, (end - start).total_seconds()), 3)

    def bar_start(self) -> datetime:
        return self.started_at or self.created_at

    def bar_end(self, fallback: datetime) -> datetime:
        return self.completed_at or fallback


@dataclass
class TodoBatch:
    index: int
    created_at: datetime
    tool_call_id: str
    label: str
    task_ids: list[str] = field(default_factory=list)


@dataclass
class TodoTimeline:
    batches: list[TodoBatch] = field(default_factory=list)
    tasks: dict[str, TodoTaskRecord] = field(default_factory=dict)

    def ordered_tasks(self) -> list[TodoTaskRecord]:
        out: list[TodoTaskRecord] = []
        for batch in self.batches:
            for tid in batch.task_ids:
                if tid in self.tasks:
                    out.append(self.tasks[tid])
        for tid, rec in sorted(self.tasks.items(), key=lambda x: x[1].created_at):
            if rec not in out:
                out.append(rec)
        return out

    def time_range(self) -> tuple[datetime, datetime] | None:
        tasks = self.ordered_tasks()
        if not tasks:
            return None
        starts = [t.created_at for t in tasks]
        ends = [t.bar_end(t.created_at) for t in tasks]
        return min(starts), max(ends)


def _parse_tool_args(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                return raw
    return raw


def _normalize_status(status: str) -> str:
    s = (status or "pending").strip().lower()
    if s in {"done", "complete", "completed", "success"}:
        return "completed"
    if s in {"running", "in_progress", "in-progress", "active", "started"}:
        return "in_progress"
    if s in {"pending", "todo", "waiting"}:
        return "pending"
    if s in {"cancelled", "canceled", "failed", "error"}:
        return "cancelled"
    return s


def _mark_to_status(mark: str) -> str:
    if mark in {">", "x", "X", "✓"}:
        return "in_progress" if mark == ">" else "completed"
    return "pending"


def _normalize_result_text(result: str) -> str:
    text = result or ""
    if "\\n" in text:
        text = text.replace("\\n", "\n").replace("\\t", "\t")
    return text


def _parse_create_tasks(result: str, arguments: Any) -> list[tuple[str, str, str]]:
    """Return list of (task_id, content, initial_status)."""
    found: list[tuple[str, str, str]] = []
    text = _normalize_result_text(result)
    for mark, task_id, content in _TASK_LINE_RE.findall(text):
        content = re.sub(r"\n\nNext step:.*", "", content, flags=re.DOTALL).strip()
        found.append((task_id, content, _mark_to_status(mark)))

    if found:
        return found

    args = _parse_tool_args(arguments)
    if isinstance(args, dict):
        raw_tasks = args.get("tasks") or args.get("task") or ""
        if isinstance(raw_tasks, list):
            parts = [str(x).strip() for x in raw_tasks if str(x).strip()]
        else:
            parts = [p.strip() for p in re.split(r"[;\n]", str(raw_tasks)) if p.strip()]
        return [(f"pending-{i}", p, "pending") for i, p in enumerate(parts, 1)]
    return []


def _apply_status(
    rec: TodoTaskRecord,
    status: str,
    ts: datetime,
    tool_name: str,
    tool_call_id: str,
) -> None:
    status = _normalize_status(status)
    if rec.current_status == status and rec.changes and rec.changes[-1].status == status:
        return
    rec.changes.append(TodoStatusChange(status, ts, tool_name, tool_call_id))
    rec.current_status = status
    if status == "in_progress" and rec.started_at is None:
        rec.started_at = ts
    if status == "completed" and rec.completed_at is None:
        rec.completed_at = ts


def _batch_label(arguments: Any, tasks: list[tuple[str, str, str]]) -> str:
    args = _parse_tool_args(arguments)
    if isinstance(args, dict) and args.get("tasks"):
        raw = str(args["tasks"]).replace("<arg_value>", "").strip()
        first = raw.split(";")[0].strip() if raw else ""
        if first:
            return first[:80] + ("…" if len(first) > 80 else "")
    if tasks:
        return tasks[0][1][:80] + ("…" if len(tasks[0][1]) > 80 else "")
    return "任务路线图"


def build_todo_timeline(tools: list[ToolExecution]) -> TodoTimeline:
    timeline = TodoTimeline()
    batch_index = 0

    todo_tools = sorted(
        [t for t in tools if t.name in _TODO_TOOL_NAMES or t.name.startswith("todo_")],
        key=lambda t: t.start_ts,
    )

    for t in todo_tools:
        if t.name == "todo_create":
            batch_index += 1
            parsed = _parse_create_tasks(t.result, t.arguments)
            label = _batch_label(t.arguments, parsed)
            batch = TodoBatch(
                index=batch_index,
                created_at=t.end_ts,
                tool_call_id=t.tool_call_id,
                label=label,
            )
            for task_id, content, initial_status in parsed:
                rec = TodoTaskRecord(
                    task_id=task_id,
                    content=content,
                    batch_index=batch_index,
                    created_at=t.end_ts,
                    started_at=t.end_ts if initial_status == "in_progress" else None,
                    current_status=initial_status,
                    changes=[
                        TodoStatusChange(initial_status, t.end_ts, "todo_create", t.tool_call_id)
                    ],
                )
                timeline.tasks[task_id] = rec
                batch.task_ids.append(task_id)
            timeline.batches.append(batch)
            continue

        if t.name in {"todo_modify", "todo_update"}:
            args = _parse_tool_args(t.arguments)
            if not isinstance(args, dict):
                continue
            items = args.get("todos") or args.get("todo") or []
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                task_id = str(item.get("id") or item.get("task_id") or "").strip()
                if not task_id:
                    continue
                status = _normalize_status(str(item.get("status") or "pending"))
                if task_id not in timeline.tasks:
                    content = str(item.get("content") or item.get("title") or f"任务 {task_id[:8]}")
                    timeline.tasks[task_id] = TodoTaskRecord(
                        task_id=task_id,
                        content=content,
                        batch_index=batch_index or 0,
                        created_at=t.end_ts,
                        current_status=status,
                        changes=[],
                    )
                _apply_status(timeline.tasks[task_id], status, t.end_ts, t.name, t.tool_call_id)

    return timeline


def _normalize_task_event_id(raw: str) -> str:
    tid = (raw or "").strip()
    if tid.lower().startswith("todo:"):
        return tid[5:]
    return tid


def _task_event_ts(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _task_sort_key(task_id: str, content: str, task_index: int | None) -> tuple[int, str]:
    m = re.match(r"^\s*Stage\s+(\d+)\s*:", content, re.IGNORECASE)
    if m:
        return (int(m.group(1)), task_id)
    if task_index is not None:
        return (task_index + 1, task_id)
    return (9999, task_id)


def build_todo_timeline_from_task_events(
    events: list[dict[str, Any]],
    root_session: str,
) -> TodoTimeline | None:
    """Build TodoTimeline from history task.start / task.update events (main Skill stages)."""
    timeline = TodoTimeline()
    batch: TodoBatch | None = None
    batch_index = 0
    task_order: list[str] = []
    first_ts: datetime | None = None

    filtered: list[dict[str, Any]] = []
    for event in events:
        sid = str(event.get("session_id") or "")
        etype = str(event.get("event_type") or "")
        if sid.startswith(root_session) and etype in {"task.start", "task.update"}:
            filtered.append(event)
    task_events = sorted(
        filtered,
        key=lambda e: (float(e.get("timestamp") or 0), str(e.get("id") or "")),
    )
    if not task_events:
        return None

    def _ensure_batch(ts: datetime, label_hint: str = "") -> TodoBatch:
        nonlocal batch, batch_index, first_ts
        if first_ts is None:
            first_ts = ts
        if batch is not None:
            return batch
        batch_index += 1
        label = label_hint or "主 Skill 路线图"
        batch = TodoBatch(
            index=batch_index,
            created_at=ts,
            tool_call_id="task-events",
            label=label[:80] + ("…" if len(label) > 80 else ""),
        )
        timeline.batches.append(batch)
        return batch

    def _ensure_task(
        task_id: str,
        content: str,
        ts: datetime,
        *,
        task_index: int | None = None,
    ) -> TodoTaskRecord:
        if batch is None:
            raise RuntimeError("todo batch must be initialized before adding tasks")
        norm_id = _normalize_task_event_id(task_id)
        if norm_id not in timeline.tasks:
            timeline.tasks[norm_id] = TodoTaskRecord(
                task_id=norm_id,
                content=content or f"任务 {norm_id[:8]}",
                batch_index=batch.index,
                created_at=first_ts or ts,
                current_status="pending",
                changes=[],
            )
            task_order.append(norm_id)
        elif content and not timeline.tasks[norm_id].content.startswith("Stage"):
            timeline.tasks[norm_id].content = content
        if norm_id not in batch.task_ids:
            batch.task_ids.append(norm_id)
        return timeline.tasks[norm_id]

    for event in task_events:
        et = str(event.get("event_type") or "")
        ts = datetime.fromtimestamp(float(event.get("timestamp") or 0)).astimezone()

        if et == "task.start":
            raw_id = str(event.get("task_id") or "")
            content = str(event.get("task_content") or "")
            task_index = event.get("task_index")
            idx = int(task_index) if isinstance(task_index, int) else None
            _ensure_batch(ts, content or "主 Skill 路线图")
            rec = _ensure_task(raw_id, content, ts, task_index=idx)
            _apply_status(rec, "in_progress", ts, "task.start", "task-events")
            continue

        if et == "task.update":
            items = event.get("tasks") or []
            if not isinstance(items, list) or not items:
                continue
            first_content = str(items[0].get("task_content") or items[0].get("content") or "")
            total_tasks = event.get("total_tasks")
            label = first_content
            if isinstance(total_tasks, int) and total_tasks > 1:
                label = f"主 Skill · Stage 1–{total_tasks}"
            current_batch = _ensure_batch(ts, label)
            if isinstance(total_tasks, int) and total_tasks > 1:
                current_batch.label = label[:80] + ("…" if len(label) > 80 else "")
            for item in items:
                if not isinstance(item, dict):
                    continue
                raw_id = str(item.get("task_id") or item.get("id") or "")
                content = str(item.get("task_content") or item.get("content") or "")
                status = _normalize_status(str(item.get("status") or "pending"))
                task_index = item.get("task_index")
                idx = int(task_index) if isinstance(task_index, int) else None
                rec = _ensure_task(raw_id, content, ts, task_index=idx)
                start_epoch = _task_event_ts(item.get("start_time"))
                if status == "in_progress":
                    start_ts = (
                        datetime.fromtimestamp(start_epoch).astimezone()
                        if start_epoch is not None
                        else ts
                    )
                    _apply_status(rec, status, start_ts, "task.update", "task-events")
                else:
                    _apply_status(rec, status, ts, "task.update", "task-events")

    if not timeline.tasks:
        return None

    if batch is None:
        raise RuntimeError("todo batch missing after task event processing")
    batch.task_ids = sorted(
        batch.task_ids,
        key=lambda tid: _task_sort_key(
            tid,
            timeline.tasks[tid].content,
            None,
        ),
    )
    return timeline


_STATUS_RANK = {"pending": 0, "in_progress": 1, "completed": 2, "cancelled": 3}


def _copy_task(rec: TodoTaskRecord) -> TodoTaskRecord:
    return TodoTaskRecord(
        task_id=rec.task_id,
        content=rec.content,
        batch_index=rec.batch_index,
        created_at=rec.created_at,
        started_at=rec.started_at,
        completed_at=rec.completed_at,
        current_status=rec.current_status,
        changes=list(rec.changes),
    )


def _enrich_tasks_from_tools(merged: TodoTimeline, tool_tl: TodoTimeline) -> None:
    """Apply todo_modify/create timing from tool timeline onto authoritative task events."""
    for tid, src in tool_tl.tasks.items():
        if tid not in merged.tasks:
            continue
        dst = merged.tasks[tid]
        if src.content and not dst.content.startswith("Stage"):
            dst.content = src.content
        if src.started_at and (
            dst.started_at is None or _naive_ts(src.started_at) < _naive_ts(dst.started_at)
        ):
            dst.started_at = src.started_at
        if src.completed_at and (
            dst.completed_at is None or _naive_ts(src.completed_at) > _naive_ts(dst.completed_at)
        ):
            dst.completed_at = src.completed_at
        if _STATUS_RANK.get(src.current_status, 0) >= _STATUS_RANK.get(dst.current_status, 0):
            dst.current_status = src.current_status
        seen = {(c.status, c.timestamp, c.tool_name) for c in dst.changes}
        for ch in src.changes:
            key = (ch.status, ch.timestamp, ch.tool_name)
            if key not in seen:
                dst.changes.append(ch)
                seen.add(key)
        dst.changes.sort(key=lambda c: c.timestamp)


def _naive_ts(ts: datetime) -> datetime:
    return ts.replace(tzinfo=None) if ts.tzinfo else ts


def merge_todo_timelines(
    primary: TodoTimeline | None,
    secondary: TodoTimeline | None,
) -> TodoTimeline:
    """Merge event-based (primary) and tool-based (secondary) todo timelines.

    When primary comes from task.start/task.update (OfficeClaw Skill stages), it is
    authoritative: do not append duplicate todo_create batches for the same task IDs.
    """
    if primary is None or not primary.tasks:
        return secondary or TodoTimeline()
    if secondary is None or not secondary.tasks:
        return primary

    has_event_batch = any(b.tool_call_id == "task-events" for b in primary.batches)
    primary_ids = set(primary.tasks)
    secondary_new = [tid for tid in secondary.tasks if tid not in primary_ids]

    if has_event_batch:
        merged = TodoTimeline(
            batches=list(primary.batches),
            tasks={tid: _copy_task(rec) for tid, rec in primary.tasks.items()},
        )
        _enrich_tasks_from_tools(merged, secondary)
        return merged

    merged = TodoTimeline()
    offset = len(primary.batches)
    merged.batches.extend(primary.batches)
    merged.tasks = {tid: _copy_task(rec) for tid, rec in primary.tasks.items()}

    for batch in secondary.batches:
        batch_task_ids = [tid for tid in batch.task_ids if tid in secondary_new]
        if has_event_batch and not batch_task_ids:
            continue
        new_index = batch.index + offset
        new_batch = TodoBatch(
            index=new_index,
            created_at=batch.created_at,
            tool_call_id=batch.tool_call_id,
            label=batch.label,
            task_ids=list(batch_task_ids or batch.task_ids),
        )
        merged.batches.append(new_batch)
        for tid in new_batch.task_ids:
            if tid in secondary.tasks and tid not in merged.tasks:
                rec = secondary.tasks[tid]
                merged.tasks[tid] = TodoTaskRecord(
                    task_id=rec.task_id,
                    content=rec.content,
                    batch_index=new_index,
                    created_at=rec.created_at,
                    started_at=rec.started_at,
                    completed_at=rec.completed_at,
                    current_status=rec.current_status,
                    changes=list(rec.changes),
                )
            elif tid in merged.tasks:
                merged.tasks[tid].batch_index = new_index

    for tid in secondary_new:
        if tid in merged.tasks:
            continue
        rec = secondary.tasks[tid]
        merged.tasks[tid] = TodoTaskRecord(
            task_id=rec.task_id,
            content=rec.content,
            batch_index=rec.batch_index + offset,
            created_at=rec.created_at,
            started_at=rec.started_at,
            completed_at=rec.completed_at,
            current_status=rec.current_status,
            changes=list(rec.changes),
        )

    _enrich_tasks_from_tools(merged, secondary)
    return merged


_STATUS_LABEL = {
    "pending": "待开始",
    "in_progress": "进行中",
    "completed": "已完成",
    "cancelled": "已取消",
}

_STATUS_CLASS = {
    "pending": "todo-pending",
    "in_progress": "todo-active",
    "completed": "todo-done",
    "cancelled": "todo-cancelled",
}


def render_todo_section_html(timeline: TodoTimeline, esc: Any) -> str:
    if not timeline.tasks:
        return ""

    tr = timeline.time_range()
    if not tr:
        return ""
    axis_start, axis_end = tr
    span_sec = max((axis_end - axis_start).total_seconds(), 1.0)

    def _pct(ts: datetime) -> float:
        return max(0.0, min(100.0, ((ts - axis_start).total_seconds() / span_sec) * 100))

    def _fmt_ts(ts: datetime | None) -> str:
        return ts.strftime("%H:%M:%S") if ts else "—"

    def _fmt_dur(sec: float | None) -> str:
        return f"{sec:.2f}s" if sec is not None else "—"

    parts = [
        '<div class="section todo-section">',
        '<div class="section-title">任务路线图（Todo）</div>',
        '<div class="todo-axis">',
        f'<span>{esc(_fmt_ts(axis_start))}</span>',
        f'<span>{esc(_fmt_ts(axis_end))}</span>',
        "</div>",
    ]

    for batch in timeline.batches:
        parts.append(
            f'<div class="todo-batch">'
            f'<div class="todo-batch-title">路线图 {batch.index} · {esc(batch.label)} '
            f'<span class="title-time">{esc(_fmt_ts(batch.created_at))}</span></div>'
        )
        parts.append('<div class="todo-gantt">')
        for tid in batch.task_ids:
            task = timeline.tasks.get(tid)
            if not task:
                continue
            bar_start = task.bar_start()
            bar_end = task.bar_end(axis_end)
            left = _pct(bar_start)
            width = max(_pct(bar_end) - left, 0.8)
            status_cls = _STATUS_CLASS.get(task.current_status, "todo-pending")
            status_label = _STATUS_LABEL.get(task.current_status, task.current_status)
            parts.append(
                f'<div class="todo-row">'
                f'<div class="todo-row-label" title="{esc(task.task_id)}">'
                f'<span class="todo-status-dot {status_cls}"></span>'
                f'{esc(task.content[:72])}{"…" if len(task.content) > 72 else ""}'
                f"</div>"
                f'<div class="todo-row-track">'
                f'<div class="todo-bar {status_cls}" style="left:{left:.2f}%;width:{width:.2f}%">'
                f'<span class="todo-bar-tip">{esc(status_label)} · {_fmt_dur(task.execution_sec())}</span>'
                f"</div></div>"
                f'<div class="todo-row-meta">'
                f'<span>创建 {_fmt_ts(task.created_at)}</span>'
                f'<span>开始 {_fmt_ts(task.started_at)}</span>'
                f'<span>完成 {_fmt_ts(task.completed_at)}</span>'
                f'<span>耗时 {_fmt_dur(task.execution_sec())}</span>'
                f"</div></div>"
            )
        parts.append("</div></div>")

    parts.append('<table class="todo-table"><thead><tr>')
    parts.append(
        "<th>任务</th><th>状态</th><th>创建</th><th>开始</th><th>完成</th><th>执行耗时</th><th>状态变更</th></tr></thead><tbody>"
    )
    for task in timeline.ordered_tasks():
        changes = " → ".join(
            f'{_STATUS_LABEL.get(c.status, c.status)}@{c.timestamp.strftime("%H:%M:%S")}'
            for c in task.changes
        )
        status_cls = _STATUS_CLASS.get(task.current_status, "")
        status_lbl = _STATUS_LABEL.get(task.current_status, task.current_status)
        parts.append(
            f"<tr>"
            f"<td class='todo-td-content'>{esc(task.content)}</td>"
            f"<td><span class='todo-pill {status_cls}'>{esc(status_lbl)}</span></td>"
            f"<td>{esc(_fmt_ts(task.created_at))}</td>"
            f"<td>{esc(_fmt_ts(task.started_at))}</td>"
            f"<td>{esc(_fmt_ts(task.completed_at))}</td>"
            f"<td>{esc(_fmt_dur(task.execution_sec()))}</td>"
            f"<td class='todo-td-changes'>{esc(changes)}</td>"
            f"</tr>"
        )
    parts.append("</tbody></table></div>")
    return "\n".join(parts)
