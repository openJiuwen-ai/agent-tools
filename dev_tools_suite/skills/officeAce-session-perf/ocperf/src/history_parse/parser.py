"""Load JSONL history files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_history(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    events.sort(key=lambda e: (float(e.get("timestamp") or 0), str(e.get("id") or "")))
    return events


def filter_sessions(events: list[dict[str, Any]], root_session: str) -> list[dict[str, Any]]:
    return [e for e in events if str(e.get("session_id") or "").startswith(root_session)]


def pick_root_session(events: list[dict[str, Any]]) -> str | None:
    counts = Counter(str(e.get("session_id") or "") for e in events if e.get("session_id"))
    if not counts:
        return None
    roots = [
        s
        for s in counts
        if "_subagent_" not in s and "_fork_agent_" not in s.lower() and "_fork_" not in s.lower()
    ]
    pool = roots if roots else list(counts.keys())
    return Counter({s: counts[s] for s in pool}).most_common(1)[0][0]
