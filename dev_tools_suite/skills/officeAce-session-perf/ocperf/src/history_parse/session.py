"""Session tree helpers and timing utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from history_parse.models import SessionPath

# 子 Agent 拉起类工具：单独统计次数/墙钟，不计入「工具调用时间」KPI
AGENT_SPAWN_TOOL_NAMES = frozenset({"spawn_subagent", "fork_agent"})


def is_agent_spawn_tool(name: str | None) -> bool:
    return (name or "").strip().lower() in AGENT_SPAWN_TOOL_NAMES


def is_measurable_tool(name: str | None) -> bool:
    return not is_agent_spawn_tool(name)


def _short_suffix(value: str) -> str:
    return value[:12] if len(value) > 12 else value


def session_path(session_id: str, root: str) -> SessionPath:
    if session_id == root:
        return ()
    if not session_id.startswith(root):
        return ((session_id, "child-session", "child"),)

    i = len(root)
    path: list[tuple[str, str, str]] = []
    lowered = session_id.lower()
    sub_m = "_subagent_"
    fork_m = "_fork_agent_"

    while i < len(session_id):
        if session_id[i] != "_":
            break
        if lowered.startswith(sub_m, i):
            marker_len = len(sub_m)
            kind: Literal["subagent", "forkagent"] = "subagent"
        elif lowered.startswith(fork_m, i):
            marker_len = len(fork_m)
            kind = "forkagent"
        else:
            break
        j = i + marker_len
        if j >= len(session_id):
            break
        k = j
        while k < len(session_id) and session_id[k] != "_":
            k += 1
        suffix_id = session_id[j:k]
        if not suffix_id:
            break
        i = k
        node_sid = session_id[:i]
        label = (
            f"subagent · {_short_suffix(suffix_id)}"
            if kind == "subagent"
            else f"fork · {_short_suffix(suffix_id)}"
        )
        path.append((node_sid, label, kind))

    if path:
        return tuple(path)
    return ((session_id, "child-session", "child"),)


def child_info(session_id: str, root: str) -> tuple[bool, str, SessionPath]:
    path = session_path(session_id, root)
    if not path:
        return False, "", ()
    return True, path[-1][1], path


def summarize_tools(tools: list[dict[str, Any]]) -> str:
    names = [str(t.get("name") or "?") for t in tools]
    return ", ".join(names) if names else "—"


def classify_tools(tools: list[dict[str, Any]]) -> tuple[bool, bool]:
    has_spawn = False
    has_fork = False
    for t in tools:
        name = str(t.get("name", "")).lower()
        if name == "spawn_subagent":
            has_spawn = True
        elif name == "fork_agent":
            has_fork = True
    return has_spawn, has_fork


def interval_union_sec(intervals: list[tuple[datetime, datetime]]) -> float:
    normalized = sorted((start, end) for start, end in intervals if end >= start)
    if not normalized:
        return 0.0
    total = 0.0
    cur_start, cur_end = normalized[0]
    for start, end in normalized[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
            continue
        total += (cur_end - cur_start).total_seconds()
        cur_start, cur_end = start, end
    total += (cur_end - cur_start).total_seconds()
    return total
