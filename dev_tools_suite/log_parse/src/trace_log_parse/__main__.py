"""CLI: trace-llm-report LOG_PATH --session SESSION_ID [--out PATH]"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from trace_log_parse.analysis import build_timeline
from trace_log_parse.parser import (
    TraceRecord,
    assemble_records,
    filter_sessions,
    parse_trace_line,
)
from trace_log_parse.report import render_html, write_report


def parse_log_file(path: Path) -> list[TraceRecord]:
    raw: list[TraceRecord] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            rec = parse_trace_line(line.rstrip("\n"), line_no)
            if rec:
                raw.append(rec)
    return assemble_records(raw)


def resolve_log_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Log path not found: {path}")
    files = sorted(
        p for p in path.iterdir() if p.is_file() and p.name.startswith("full") and p.suffix.lower() == ".log"
    )
    if not files:
        raise FileNotFoundError(f"No full*.log files found in directory: {path}")
    return files


def parse_log_path(path: Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    for file_path in resolve_log_files(path):
        records.extend(parse_log_file(file_path))
    return assemble_records(records)


def pick_default_session(records: list[TraceRecord]) -> str | None:
    """Prefer the most frequent non-child session_id; else most frequent overall."""
    from collections import Counter

    counts = Counter(r.session_id for r in records)
    if not counts:
        return None
    roots = [s for s in counts if "_subagent_" not in s and "_fork_" not in s.lower()]
    if roots:
        sub = Counter({s: counts[s] for s in roots})
        return sub.most_common(1)[0][0]
    return counts.most_common(1)[0][0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse [LLM_IO_TRACE] logs → HTML timeline.")
    ap.add_argument("log_file", type=Path, help="Path to a full .log file or a directory containing full*.log files")
    ap.add_argument(
        "--session",
        "-s",
        default=None,
        help="Root session_id (child sessions matching prefix are included). Default: auto-detect.",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: same dir as log, out_report_<session_id>.html)",
    )
    args = ap.parse_args()

    log_path = args.log_file.resolve()

    try:
        log_files = resolve_log_files(log_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    all_recs = parse_log_path(log_path)
    root = args.session or pick_default_session(all_recs)
    if not root:
        raise SystemExit("No [LLM_IO_TRACE] records found.")

    filtered = filter_sessions(all_recs, root)
    if not filtered:
        raise SystemExit(f"No records for session prefix {root!r}")

    rounds, gaps = build_timeline(filtered, root)

    out_path = args.out
    if out_path is None:
        session_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in root)
        out_dir = log_path if log_path.is_dir() else log_path.parent
        out_path = out_dir / f"out_report_{session_name}.html"

    source_label = str(log_path)
    if len(log_files) > 1:
        source_label = f"{log_path} ({len(log_files)} files: {', '.join(p.name for p in log_files)})"
    html_text = render_html(root, source_label, rounds, gaps)
    write_report(out_path.resolve(), html_text)
    logger.info("Wrote %s (%s LLM rounds, %s tool gaps)", out_path, len(rounds), len(gaps))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
