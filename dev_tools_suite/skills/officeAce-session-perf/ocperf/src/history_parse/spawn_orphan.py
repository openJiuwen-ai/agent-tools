"""Detect orphan spawn tool calls and backfill timing from full.log spawn_concurrency lines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from history_parse.models import ToolExecution
from history_parse.session import child_info, is_agent_spawn_tool

_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
_LOG_PREFIX = "[tool_concurrency]"
_LEGACY_LOG_PREFIX = "[spawn_concurrency]"
_SPAWN_START_RE = re.compile(
    r"\[(?:tool_concurrency|spawn_concurrency)\]\s+(?:slot acquire tool=\S+ id=|spawn start id=)(\S+)"
)
_SPAWN_DONE_RE = re.compile(
    r"\[(?:tool_concurrency|spawn_concurrency)\]\s+"
    r"(?:slot release tool=\S+ id=|spawn done id=)(\S+)\s+elapsed=([\d.]+)s"
)
_TRACE_SESSION_RE = re.compile(r"session_id='([^']+)'")
_TASK_ID_RE = re.compile(r"['\"]task_id['\"]\s*:\s*['\"](subagent_[a-f0-9]+)['\"]")


@dataclass
class SpawnConcurrencyRecord:
    tool_call_id: str
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    elapsed_sec: float | None = None


def _parse_log_timestamp(raw: str) -> datetime | None:
    """Parse full.log wall-clock timestamps as naive local time (no tz suffix in logs)."""
    m = _LOG_TS_RE.match(raw.strip())
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def _align_log_ts(ts: datetime, ref: datetime) -> datetime:
    """Attach history timezone to naive log timestamps, or convert when both are aware."""
    if ref.tzinfo is None:
        return ts.replace(tzinfo=None) if ts.tzinfo else ts
    if ts.tzinfo is None:
        return ts.replace(tzinfo=ref.tzinfo)
    return ts.astimezone(ref.tzinfo)


def _resolve_log_files(paths: list[Path]) -> list[Path]:
    """Expand log directories to full*.log / full*.txt files."""
    files: list[Path] = []
    for inp in paths:
        p = inp.resolve()
        if p.is_dir():
            for pattern in ("full*.log", "full*.txt", "full*.json"):
                files.extend(sorted(p.glob(pattern)))
        elif p.is_file():
            files.append(p)
    seen: set[str] = set()
    out: list[Path] = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def parse_spawn_concurrency_logs(paths: list[Path]) -> dict[str, SpawnConcurrencyRecord]:
    """Parse [spawn_concurrency] spawn start/done lines from full*.log files."""
    paths = _resolve_log_files(paths)
    records: dict[str, SpawnConcurrencyRecord] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if _LOG_PREFIX not in line and _LEGACY_LOG_PREFIX not in line:
                continue
            ts = _parse_log_timestamp(line)
            m_start = _SPAWN_START_RE.search(line)
            if m_start and ts is not None:
                call_id = m_start.group(1)
                rec = records.setdefault(call_id, SpawnConcurrencyRecord(tool_call_id=call_id))
                rec.start_ts = ts
                continue
            m_done = _SPAWN_DONE_RE.search(line)
            if m_done and ts is not None:
                call_id = m_done.group(1)
                rec = records.setdefault(call_id, SpawnConcurrencyRecord(tool_call_id=call_id))
                rec.end_ts = ts
                try:
                    rec.elapsed_sec = float(m_done.group(2))
                except ValueError:
                    pass
    return records


def _infer_start_ts(rec: SpawnConcurrencyRecord, ref: datetime) -> datetime | None:
    if rec.start_ts is not None:
        return _align_log_ts(rec.start_ts, ref)
    if rec.end_ts is not None and rec.elapsed_sec is not None:
        end = _align_log_ts(rec.end_ts, ref)
        return end - timedelta(seconds=rec.elapsed_sec)
    return None


def parse_subagent_windows_from_full_log(
    paths: list[Path],
    root_session: str,
) -> dict[str, tuple[datetime, datetime]]:
    """Map subagent session_id -> (first_trace_ts, last_trace_ts) from full.log LLM_IO_TRACE."""
    paths = _resolve_log_files(paths)
    prefix = root_session + "_subagent_"
    bounds: dict[str, list[datetime]] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if root_session not in line or "[LLM_IO_TRACE]" not in line or prefix not in line:
                continue
            ts = _parse_log_timestamp(line)
            m = _TRACE_SESSION_RE.search(line)
            if ts is None or m is None:
                continue
            sid = m.group(1)
            if not sid.startswith(prefix):
                continue
            bounds.setdefault(sid, []).append(ts)
    return {sid: (min(ts_list), max(ts_list)) for sid, ts_list in bounds.items()}


def _claimed_subagent_sessions(
    root_session: str,
    existing_tools: list[ToolExecution] | None,
) -> set[str]:
    claimed: set[str] = set()
    for tool in existing_tools or []:
        if not is_agent_spawn_tool(tool.name):
            continue
        m = _TASK_ID_RE.search(tool.result or "")
        if m:
            claimed.add(f"{root_session}_{m.group(1)}")
    return claimed


def _match_subagent_window(
    orphan_start: datetime,
    windows: dict[str, tuple[datetime, datetime]],
    claimed: set[str],
    *,
    max_delta_sec: float = 120.0,
) -> tuple[str, datetime, datetime] | None:
    ref = orphan_start
    best_sid = ""
    best: tuple[datetime, datetime] | None = None
    best_delta = max_delta_sec
    for sid, (win_start, win_end) in windows.items():
        if sid in claimed:
            continue
        aligned_start = _align_log_ts(win_start, ref)
        delta = abs((aligned_start - ref).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best_sid = sid
            best = (aligned_start, _align_log_ts(win_end, ref))
    if best is None or not best_sid:
        return None
    return best_sid, best[0], best[1]


def _lookup_spawn_record(
    records: dict[str, SpawnConcurrencyRecord],
    call_id: str,
) -> SpawnConcurrencyRecord | None:
    if call_id in records:
        return records[call_id]
    for rid, rec in records.items():
        if call_id.startswith(rid) or rid.startswith(call_id):
            return rec
    return None


def resolve_orphan_spawns(
    pending_calls: dict[str, dict[str, Any]],
    root_session: str,
    *,
    full_log_paths: list[Path] | None = None,
    existing_tools: list[ToolExecution] | None = None,
) -> tuple[list[dict[str, Any]], list[ToolExecution]]:
    """Return orphan metadata and synthetic ToolExecution rows backfilled from full.log."""
    spawn_records = parse_spawn_concurrency_logs(full_log_paths or [])
    subagent_windows = parse_subagent_windows_from_full_log(full_log_paths or [], root_session)
    claimed_subagents = _claimed_subagent_sessions(root_session, existing_tools)
    orphans: list[dict[str, Any]] = []
    backfilled: list[ToolExecution] = []

    for call_id, pending in sorted(pending_calls.items(), key=lambda x: x[1].get("start_ts") or datetime.min):
        name = str(pending.get("name") or "")
        if not is_agent_spawn_tool(name):
            continue

        session_id = str(pending.get("session_id") or root_session)
        history_start = pending.get("start_ts")
        if not isinstance(history_start, datetime):
            continue
        start_ts = history_start

        rec = _lookup_spawn_record(spawn_records, call_id)
        end_ts: datetime | None = None
        duration_sec: float | None = None
        backfill_source = ""
        matched_subagent = ""

        if rec is not None:
            if rec.end_ts is not None:
                end_ts = _align_log_ts(rec.end_ts, history_start)
            if rec.elapsed_sec is not None:
                duration_sec = rec.elapsed_sec
            elif end_ts is not None:
                duration_sec = max(0.0, (end_ts - start_ts).total_seconds())
            if end_ts is not None:
                backfill_source = "spawn_concurrency"

        if end_ts is None and subagent_windows:
            matched = _match_subagent_window(
                history_start,
                subagent_windows,
                claimed_subagents,
            )
            if matched is not None:
                matched_subagent, _win_start, win_end = matched
                end_ts = win_end
                duration_sec = max(0.0, (end_ts - start_ts).total_seconds())
                backfill_source = "subagent_window"
                claimed_subagents.add(matched_subagent)

        orphan_entry: dict[str, Any] = {
            "tool_call_id": call_id,
            "name": name,
            "session_id": session_id,
            "start_ts": start_ts.isoformat(),
            "backfilled": bool(end_ts is not None),
            "backfill_source": backfill_source or None,
        }
        if matched_subagent:
            orphan_entry["subagent_session_id"] = matched_subagent
        if end_ts is not None:
            orphan_entry["end_ts"] = end_ts.isoformat()
        if duration_sec is not None:
            orphan_entry["duration_sec"] = round(duration_sec, 3)
        orphans.append(orphan_entry)

        if end_ts is None:
            continue

        is_child, label, path = child_info(session_id, root_session)
        duration = duration_sec if duration_sec is not None else max(0.0, (end_ts - start_ts).total_seconds())
        result_suffix = matched_subagent.split("_subagent_")[-1][:8] if matched_subagent else ""
        result_note = f"subagent_{result_suffix}" if result_suffix else backfill_source
        backfilled.append(
            ToolExecution(
                session_id=session_id,
                request_id=str(pending.get("request_id") or ""),
                tool_call_id=call_id,
                name=name,
                arguments=pending.get("arguments"),
                start_ts=start_ts,
                end_ts=end_ts,
                duration_sec=round(duration, 6),
                result=f"success=True [orphan backfill from {backfill_source}; {result_note}]",
                is_child_session=is_child,
                child_label=label,
                child_path=path,
                is_orphan_backfill=True,
            )
        )

    return orphans, backfilled
