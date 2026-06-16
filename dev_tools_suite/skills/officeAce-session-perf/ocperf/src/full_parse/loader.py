"""Load and merge multiple full log files (JSON NDJSON or plain-text) for one session."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from full_parse.trace_analysis import LLMRound, ToolGap, build_timeline
from full_parse.trace_parser import (
    TraceRecord,
    assemble_records,
    filter_sessions,
    parse_json_log_object,
    parse_trace_line,
)


@dataclass
class SourceFileStats:
    path: str
    trace_lines: int
    first_ts: str
    last_ts: str


@dataclass
class FullSessionData:
    root_session: str
    rounds: list[LLMRound]
    gaps: list[ToolGap]
    records: list[TraceRecord]
    source_files: list[SourceFileStats]
    todo_timeline: Any = None
    history_source: str | None = None


_FULL_GLOBS = ("full*.json", "full*.txt", "full*.log")
_HISTORY_NAMES = ("history.json",)


def resolve_full_log_paths(inputs: list[Path]) -> list[Path]:
    """Resolve full log files from paths or directories (json / txt / log)."""
    paths: list[Path] = []
    for inp in inputs:
        p = inp.resolve()
        if p.is_dir():
            found: list[Path] = []
            for pattern in _FULL_GLOBS:
                found.extend(sorted(p.glob(pattern)))
            if not found:
                found = sorted(p.glob("**/*.json")) + sorted(p.glob("**/*.txt"))
            paths.extend(found)
        elif p.is_file():
            paths.append(p)
    seen: set[Path] = set()
    out: list[Path] = []
    for path in sorted(paths, key=lambda x: (x.stat().st_mtime, x.name)):
        rp = path.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


# Backward-compatible alias
resolve_full_json_paths = resolve_full_log_paths


def resolve_history_path(
    inputs: list[Path],
    *,
    history_path: Path | None = None,
) -> Path | None:
    """Locate history.json beside full logs, session dir, or explicit path."""
    if history_path is not None:
        hp = history_path.resolve()
        if hp.is_file():
            return hp
    for inp in inputs:
        p = inp.resolve()
        if p.is_file() and p.suffix == ".json" and "history" in p.name.lower():
            return p
        roots: list[Path] = []
        if p.is_dir():
            roots.append(p)
        else:
            roots.append(p.parent)
        for root in roots:
            for name in _HISTORY_NAMES:
                candidate = root / name
                if candidate.is_file():
                    return candidate
            matches = sorted(root.glob("history*.json"))
            if matches:
                return matches[0]
            # OfficeClaw: full logs in .logs while history.json lives in sessions/<sid>/
            cur = root
            for _ in range(5):
                cur = cur.parent
                for name in _HISTORY_NAMES:
                    candidate = cur / name
                    if candidate.is_file():
                        return candidate
                matches = sorted(cur.glob("history*.json"))
                if matches:
                    return matches[0]
    return None


def load_todo_timeline_from_inputs(
    session_id: str,
    inputs: list[Path],
    *,
    history_path: Path | None = None,
) -> tuple[Any, str | None]:
    """Build TodoTimeline from co-located history.json, if present."""
    hist_path = resolve_history_path(inputs, history_path=history_path)
    if hist_path is None:
        return None, None
    try:
        from history_parse.analysis import build_timeline_from_history
        from history_parse.parser import load_history

        events = load_history(hist_path)
        _, _, extras = build_timeline_from_history(
            events,
            session_id,
            full_log_paths=resolve_full_log_paths(inputs),
        )
        todo = extras.todo_timeline
        if todo and getattr(todo, "tasks", None):
            return todo, str(hist_path)
    except (OSError, ValueError, KeyError):
        pass
    return None, None


def _parse_log_line(
    line: str,
    line_no: int,
    *,
    source_file: str,
    file_order: int,
) -> TraceRecord | None:
    line = line.strip()
    if not line or "[LLM_IO_TRACE]" not in line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        rec = parse_trace_line(line, line_no)
        if rec is not None:
            rec.source_file = source_file
            rec.file_order = file_order
        return rec
    if not isinstance(obj, dict):
        return None
    rec = parse_json_log_object(
        obj,
        line_no,
        source_file=source_file,
        file_order=file_order,
    )
    if rec is None and "message" not in obj:
        msg = obj.get("log") or obj.get("content") or ""
        if isinstance(msg, str) and "[LLM_IO_TRACE]" in msg:
            rec = parse_json_log_object(
                {"message": msg, "timestamp": obj.get("timestamp", "")},
                line_no,
                source_file=source_file,
                file_order=file_order,
            )
    return rec


def load_session_from_paths(
    session_id: str,
    inputs: list[Path],
    *,
    history_path: Path | None = None,
) -> FullSessionData:
    paths = resolve_full_log_paths(inputs)
    if not paths:
        raise FileNotFoundError("未找到 full*.json / full*.txt / full*.log 文件")

    raw: list[TraceRecord] = []
    file_stats: list[SourceFileStats] = []

    for file_order, path in enumerate(paths):
        trace_count = 0
        first_ts = ""
        last_ts = ""
        with path.open(encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                rec = _parse_log_line(
                    line,
                    line_no,
                    source_file=path.name,
                    file_order=file_order,
                )
                if rec is None:
                    continue
                if rec.session_id != session_id and not rec.session_id.startswith(session_id + "_"):
                    continue
                raw.append(rec)
                trace_count += 1
                ts_s = rec.ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                if not first_ts:
                    first_ts = ts_s
                last_ts = ts_s
        file_stats.append(
            SourceFileStats(
                path=str(path),
                trace_lines=trace_count,
                first_ts=first_ts or "—",
                last_ts=last_ts or "—",
            )
        )

    if not raw:
        raise ValueError(f"session {session_id!r} 在指定文件中无 LLM_IO_TRACE 记录")

    raw.sort(key=lambda r: (r.ts, r.file_order, r.raw_line_no))
    assembled = assemble_records(raw)
    filtered = filter_sessions(assembled, session_id)
    rounds, gaps = build_timeline(filtered, session_id)
    todo_tl, hist_src = load_todo_timeline_from_inputs(
        session_id, inputs, history_path=history_path
    )

    return FullSessionData(
        root_session=session_id,
        rounds=rounds,
        gaps=gaps,
        records=filtered,
        source_files=file_stats,
        todo_timeline=todo_tl,
        history_source=hist_src,
    )
