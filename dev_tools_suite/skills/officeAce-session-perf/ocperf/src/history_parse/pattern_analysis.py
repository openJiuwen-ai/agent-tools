"""Detect duplicate / cyclic / concurrent vs serial tool & LLM patterns per phase."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from history_parse.session import is_measurable_tool
from history_parse.report_ui import (
    ExecutionPhase,
    TimelineKind,
    _item_child_meta,
    _item_duration,
    _item_end,
    _item_start,
    _naive,
)

Relationship = Literal["serial", "concurrent", "mixed"]


@dataclass
class EventRef:
    idx: int
    kind: TimelineKind
    name: str
    start: datetime
    end: datetime
    duration_sec: float
    session_label: str = ""
    is_child: bool = False
    detail: str = ""
    exclude_from_tool_time: bool = False


@dataclass
class DuplicateGroup:
    name: str
    kind: TimelineKind
    count: int
    relationship: Relationship
    events: list[EventRef] = field(default_factory=list)
    note: str = ""


@dataclass
class CycleHint:
    pattern: list[str]
    repeats: int
    span_events: int
    example_sequence: list[str]
    note: str = ""


@dataclass
class PhasePatternReport:
    phase_id: str
    phase_title: str
    event_count: int
    llm_count: int
    tool_count: int
    duplicates: list[DuplicateGroup] = field(default_factory=list)
    cycles: list[CycleHint] = field(default_factory=list)
    has_issues: bool = False


@dataclass
class SessionPatternSummary:
    phases: list[PhasePatternReport] = field(default_factory=list)
    global_duplicates: list[DuplicateGroup] = field(default_factory=list)


def _overlap(a: EventRef, b: EventRef) -> bool:
    return _naive(a.start) < _naive(b.end) and _naive(b.start) < _naive(a.end)


def _tool_name(obj: Any, kind: TimelineKind) -> str:
    if kind == "tool":
        if hasattr(obj, "name") and getattr(obj, "name", None):
            return str(obj.name).strip()
        if hasattr(obj, "tools_triggered"):
            return str(getattr(obj, "tools_triggered", None) or "tool_window").strip()
        return "?"
    model = str(getattr(obj, "model_name", "") or "").strip()
    return model or "LLM"


def event_ref_from_item(idx: int, kind: TimelineKind, obj: Any) -> EventRef:
    is_child, label, _ = _item_child_meta(kind, obj)
    name = _tool_name(obj, kind)
    detail = ""
    if kind == "llm" and hasattr(obj, "ttft_sec"):
        from history_parse.llm_latency_metrics import llm_round_metrics

        detail = llm_round_metrics(obj)["detail"]
    elif kind == "tool":
        detail = str(getattr(obj, "tool_call_id", ""))[:32]
    exclude = kind == "tool" and not is_measurable_tool(name)
    return EventRef(
        idx=idx,
        kind=kind,
        name=name,
        start=_item_start(kind, obj),
        end=_item_end(kind, obj),
        duration_sec=_item_duration(kind, obj),
        session_label=label if is_child else "根会话",
        is_child=is_child,
        detail=detail,
        exclude_from_tool_time=exclude,
    )


def _classify_duplicate_group(events: list[EventRef]) -> Relationship:
    if len(events) < 2:
        return "serial"
    has_overlap = False
    has_serial = False
    ordered = sorted(events, key=lambda e: _naive(e.start))
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if _overlap(a, b):
                has_overlap = True
            else:
                if _naive(a.end) <= _naive(b.start) or _naive(b.end) <= _naive(a.start):
                    has_serial = True
    if has_overlap and has_serial:
        return "mixed"
    if has_overlap:
        return "concurrent"
    return "serial"


def _find_cycles(tool_names: list[str], min_period: int = 1, max_period: int = 6) -> list[CycleHint]:
    hints: list[CycleHint] = []
    n = len(tool_names)
    if n < 4:
        return hints
    max_period = min(max_period, n // 2)
    for period in range(min_period, max_period + 1):
        if n < period * 2:
            continue
        pattern = tool_names[:period]
        repeats = 1
        i = period
        while i + period <= n and tool_names[i:i + period] == pattern:
            repeats += 1
            i += period
        if repeats >= 2:
            hints.append(
                CycleHint(
                    pattern=pattern,
                    repeats=repeats,
                    span_events=repeats * period,
                    example_sequence=tool_names[: repeats * period],
                    note=f"工具序列「{' → '.join(pattern)}」连续重复 {repeats} 次",
                )
            )
    # adjacent same-name ping-pong
    for i in range(n - 3):
        a, b, c, d = tool_names[i], tool_names[i + 1], tool_names[i + 2], tool_names[i + 3]
        if a == c and b == d and a != b:
            hints.append(
                CycleHint(
                    pattern=[a, b],
                    repeats=2,
                    span_events=4,
                    example_sequence=tool_names[i:i + 4],
                    note=f"交替循环：{a} ↔ {b}",
                )
            )
    # dedupe by pattern string
    seen: set[str] = set()
    out: list[CycleHint] = []
    for h in hints:
        key = "→".join(h.pattern) + f"×{h.repeats}"
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out[:8]


def analyze_phase_events(
    phase: ExecutionPhase,
    items: list[tuple[int, TimelineKind, Any]],
) -> PhasePatternReport:
    events = [event_ref_from_item(idx, kind, obj) for idx, kind, obj in items]
    llm_n = sum(1 for e in events if e.kind == "llm")
    tool_n = sum(1 for e in events if e.kind == "tool")
    report = PhasePatternReport(
        phase_id=phase.phase_id,
        phase_title=phase.title,
        event_count=len(events),
        llm_count=llm_n,
        tool_count=tool_n,
    )
    if not events:
        return report

    by_key: dict[tuple[TimelineKind, str], list[EventRef]] = defaultdict(list)
    for e in events:
        if e.exclude_from_tool_time:
            continue
        by_key[(e.kind, e.name.lower())].append(e)

    duplicates: list[DuplicateGroup] = []
    for (kind, _lname), group in by_key.items():
        if len(group) < 2:
            continue
        rel = _classify_duplicate_group(group)
        note_parts = []
        if kind == "tool" and rel == "serial":
            note_parts.append("同工具多次串行执行，可能存在重试或分步调用")
        elif kind == "tool" and rel == "concurrent":
            note_parts.append("同工具时间重叠，可能为并行子会话或日志交叉")
        elif kind == "tool" and rel == "mixed":
            note_parts.append("同工具既有重叠又有先后，请结合子 Agent 时间线判断")
        if kind == "llm":
            note_parts.append("同阶段多次模型调用")
        duplicates.append(
            DuplicateGroup(
                name=group[0].name,
                kind=kind,
                count=len(group),
                relationship=rel,
                events=sorted(group, key=lambda e: _naive(e.start)),
                note="；".join(note_parts),
            )
        )

    spawn_events = sorted(
        [e for e in events if e.kind == "tool" and e.exclude_from_tool_time],
        key=lambda e: _naive(e.start),
    )
    if spawn_events:
        spawn_by_name: dict[str, list[EventRef]] = defaultdict(list)
        for e in spawn_events:
            spawn_by_name[e.name.lower()].append(e)
        for group in spawn_by_name.values():
            rel = _classify_duplicate_group(group) if len(group) >= 2 else "serial"
            duplicates.append(
                DuplicateGroup(
                    name=group[0].name,
                    kind="tool",
                    count=len(group),
                    relationship=rel,
                    events=group,
                    note="子 Agent 派发（不计入工具墙钟 KPI；并发视图分轨数≠调用次数）",
                )
            )

    tool_seq = [
        e.name
        for e in sorted(events, key=lambda e: _naive(e.start))
        if e.kind == "tool" and not e.exclude_from_tool_time
    ]
    cycles = _find_cycles(tool_seq)

    report.duplicates = sorted(
        duplicates,
        key=lambda d: (-d.count, d.kind == "tool", d.name),
    )
    report.cycles = cycles
    report.has_issues = bool(duplicates or cycles)
    return report


def analyze_session_patterns(
    phases: list[ExecutionPhase],
    buckets: dict[str, list[tuple[int, TimelineKind, Any]]],
) -> SessionPatternSummary:
    phase_reports = []
    for phase in phases:
        items = buckets.get(phase.phase_id, [])
        if not items:
            continue
        phase_reports.append(analyze_phase_events(phase, items))

    global_by_tool: dict[str, list[EventRef]] = defaultdict(list)
    for pr in phase_reports:
        for d in pr.duplicates:
            if d.kind == "tool":
                global_by_tool[d.name.lower()].extend(d.events)
    global_dups = []
    for name, evs in global_by_tool.items():
        if len(evs) < 2:
            continue
        rel = _classify_duplicate_group(evs)
        global_dups.append(
            DuplicateGroup(
                name=evs[0].name,
                kind="tool",
                count=len(evs),
                relationship=rel,
                events=sorted(evs, key=lambda e: _naive(e.start))[:20],
                note="跨阶段汇总",
            )
        )

    return SessionPatternSummary(phases=phase_reports, global_duplicates=global_dups)


def assign_flow_tracks(events: list[EventRef]) -> dict[int, int]:
    """Greedy track assignment for non-overlapping layout (concurrent = different tracks)."""
    tracks: dict[int, int] = {}
    track_ends: list[datetime] = []
    for e in sorted(events, key=lambda x: _naive(x.start)):
        placed = False
        for ti, end in enumerate(track_ends):
            if _naive(end) <= _naive(e.start):
                tracks[e.idx] = ti
                track_ends[ti] = e.end
                placed = True
                break
        if not placed:
            tracks[e.idx] = len(track_ends)
            track_ends.append(e.end)
    return tracks
