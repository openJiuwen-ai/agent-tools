#!/usr/bin/env python3
"""Analyze tool-call streaming and execution timing from OfficeClaw history.json.

The input is a JSONL history file. The script separates two different timing
surfaces that are easy to confuse:

1. chat.tool_calls.delta: LLM streaming of tool-call name/arguments.
2. chat.tool_update/chat.tool_result: actual tool execution observed by history.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    seq: int
    request_id: str
    session_id: str | None = None
    task_id: str | None = None
    index: int | None = None
    tool_call_id: str | None = None
    name: str | None = None

    delta_timestamps: list[float] = field(default_factory=list)
    delta_lines: list[int] = field(default_factory=list)
    delta_argument_parts: list[str] = field(default_factory=list)

    formal_time: float | None = None
    formal_line: int | None = None
    formal_arguments: str | None = None

    update_time: float | None = None
    update_line: int | None = None
    update_status: str | None = None

    result_time: float | None = None
    result_line: int | None = None
    result_len: int | None = None
    result_status_hint: str | None = None

    anomalies: list[str] = field(default_factory=list)

    def add_delta(self, line_no: int, timestamp: float, delta: dict[str, Any]) -> None:
        self.delta_timestamps.append(timestamp)
        self.delta_lines.append(line_no)
        tool_call_id = first_non_empty(delta.get("tool_call_id"), delta.get("id"))
        if tool_call_id and not self.tool_call_id:
            self.tool_call_id = tool_call_id
        if delta.get("name") and not self.name:
            self.name = str(delta["name"])
        if self.index is None and delta.get("index") is not None:
            self.index = safe_int(delta.get("index"))
        arguments = delta.get("arguments")
        if arguments:
            self.delta_argument_parts.append(str(arguments))

    @property
    def argument_text(self) -> str:
        if self.formal_arguments is not None:
            return self.formal_arguments
        return "".join(self.delta_argument_parts)

    def to_metrics(self, gap_threshold_ms: float) -> dict[str, Any]:
        delta_start = self.delta_timestamps[0] if self.delta_timestamps else None
        delta_end = self.delta_timestamps[-1] if self.delta_timestamps else None
        gaps_ms = [
            (self.delta_timestamps[i] - self.delta_timestamps[i - 1]) * 1000
            for i in range(1, len(self.delta_timestamps))
        ]
        llm_delta_duration_ms = diff_ms(delta_start, delta_end)
        tool_call_emit_delay_ms = diff_ms(delta_end, self.formal_time)
        tool_start_delay_ms = diff_ms(self.formal_time, self.update_time)

        exec_start = self.update_time if self.update_time is not None else self.formal_time
        tool_exec_duration_ms = diff_ms(exec_start, self.result_time)

        total_start = delta_start if delta_start is not None else self.formal_time
        total_observed_ms = diff_ms(total_start, self.result_time)
        total_from_first_delta_to_result_ms = diff_ms(delta_start, self.result_time)

        arguments = self.argument_text
        argument_keys = extract_json_keys(arguments)

        return {
            "seq": self.seq,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "tool_call_id": self.tool_call_id,
            "index": self.index,
            "name": self.name or "(unknown)",
            "delta_count": len(self.delta_timestamps),
            "llm_delta_start": delta_start,
            "llm_delta_end": delta_end,
            "llm_delta_duration_ms": llm_delta_duration_ms,
            "avg_delta_gap_ms": avg(gaps_ms),
            "p95_delta_gap_ms": percentile(gaps_ms, 95),
            "max_delta_gap_ms": max(gaps_ms) if gaps_ms else None,
            "long_gap_count": sum(1 for gap in gaps_ms if gap >= gap_threshold_ms),
            "tool_call_time": self.formal_time,
            "tool_call_emit_delay_ms": tool_call_emit_delay_ms,
            "tool_update_time": self.update_time,
            "tool_start_delay_ms": tool_start_delay_ms,
            "tool_result_time": self.result_time,
            "tool_exec_duration_ms": tool_exec_duration_ms,
            "total_from_first_delta_to_result_ms": total_from_first_delta_to_result_ms,
            "total_observed_ms": total_observed_ms,
            "arguments_len": len(arguments),
            "argument_keys": argument_keys,
            "result_len": self.result_len,
            "result_status_hint": self.result_status_hint,
            "first_line": self.delta_lines[0] if self.delta_lines else self.formal_line,
            "last_line": self.result_line or self.formal_line or (self.delta_lines[-1] if self.delta_lines else None),
            "anomalies": list(self.anomalies),
        }


@dataclass
class ReportContext:
    path: Path
    summary: dict[str, Any]
    warnings: list[str]
    records: list[ToolCallRecord]
    gap_threshold_ms: float
    top: int


def first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value:
            return str(value)
    return None


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def diff_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000


def avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil((pct / 100) * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


def extract_json_keys(text: str) -> list[str]:
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    return [str(key) for key in parsed.keys()]


def status_hint(result: Any) -> str:
    text = str(result)
    lowered = text.lower()
    if "success=true" in lowered or "'success': true" in lowered or '"success": true' in lowered:
        return "success"
    if "success=false" in lowered or "'success': false" in lowered or '"success": false' in lowered:
        return "failed"
    if "error" in lowered:
        return "has_error_text"
    return "unknown"


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.0f}ms"


def fmt_time(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except (OSError, OverflowError, ValueError):
        return str(value)


def read_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    previous_timestamp: float | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                warnings.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            event["_line_no"] = line_no
            timestamp = event.get("timestamp")
            if not isinstance(timestamp, (int, float)):
                warnings.append(f"line {line_no}: missing numeric timestamp")
            elif previous_timestamp is not None and timestamp < previous_timestamp:
                warnings.append(f"line {line_no}: timestamp goes backwards")
            if isinstance(timestamp, (int, float)):
                previous_timestamp = timestamp
            events.append(event)
    return events, warnings


def analyze_events(
    events: list[dict[str, Any]],
    request_filter: str | None = None,
) -> tuple[list[ToolCallRecord], dict[str, Any], list[str]]:
    records: list[ToolCallRecord] = []
    warnings: list[str] = []
    by_call_id: dict[str, ToolCallRecord] = {}
    active_by_request_index: dict[tuple[str, int], ToolCallRecord] = {}
    last_formal_by_request: dict[str, ToolCallRecord] = {}
    session_ids: set[str] = set()
    request_ids: set[str] = set()
    event_counts: dict[str, int] = {}

    def new_record(request_id: str, event: dict[str, Any], index: int | None = None) -> ToolCallRecord:
        record = ToolCallRecord(
            seq=len(records) + 1,
            request_id=request_id,
            session_id=event.get("session_id"),
            task_id=event.get("task_id"),
            index=index,
        )
        records.append(record)
        return record

    for event in events:
        request_id = str(event.get("request_id") or "")
        if request_filter and request_id != request_filter:
            continue
        event_type = event.get("event_type")
        if not event_type:
            continue
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        if request_id:
            request_ids.add(request_id)
        if event.get("session_id"):
            session_ids.add(str(event["session_id"]))

        timestamp = event.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            continue
        line_no = int(event["_line_no"])

        if event_type == "chat.tool_calls.delta":
            for delta in event.get("tool_calls") or []:
                if not isinstance(delta, dict):
                    continue
                index = safe_int(delta.get("index")) or 0
                tool_call_id = first_non_empty(delta.get("tool_call_id"), delta.get("id"))
                record = by_call_id.get(tool_call_id) if tool_call_id else None
                if record is None:
                    active_key = (request_id, index)
                    record = active_by_request_index.get(active_key)
                    if needs_new_delta_record(record, tool_call_id):
                        record = new_record(request_id, event, index)
                        active_by_request_index[active_key] = record
                record.add_delta(line_no, timestamp, delta)
                if record.tool_call_id:
                    by_call_id[record.tool_call_id] = record

        elif event_type == "chat.tool_call":
            tool_call = event.get("tool_call") or {}
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = first_non_empty(tool_call.get("tool_call_id"), tool_call.get("id"))
            record = by_call_id.get(tool_call_id) if tool_call_id else None
            if record is None:
                record = find_active_for_formal(active_by_request_index, request_id, tool_call.get("name"))
            if record is None:
                record = new_record(request_id, event)
            record.formal_time = timestamp
            record.formal_line = line_no
            if tool_call_id:
                record.tool_call_id = tool_call_id
                by_call_id[tool_call_id] = record
            if tool_call.get("name"):
                record.name = str(tool_call["name"])
            if tool_call.get("arguments") is not None:
                record.formal_arguments = str(tool_call["arguments"])
            if event.get("task_id") and not record.task_id:
                record.task_id = str(event["task_id"])
            last_formal_by_request[request_id] = record
            if record.index is not None:
                active_by_request_index.pop((request_id, record.index), None)

        elif event_type == "chat.tool_update":
            tool_call_id = first_non_empty(event.get("tool_call_id"))
            record = by_call_id.get(tool_call_id) if tool_call_id else last_formal_by_request.get(request_id)
            if record is None:
                record = new_record(request_id, event)
                if tool_call_id:
                    record.tool_call_id = tool_call_id
                    by_call_id[tool_call_id] = record
            if event.get("tool_name") and not record.name:
                record.name = str(event["tool_name"])
            if event.get("task_id") and not record.task_id:
                record.task_id = str(event["task_id"])
            if event.get("status") == "in_progress" and record.update_time is None:
                record.update_time = timestamp
                record.update_line = line_no
                record.update_status = str(event["status"])

        elif event_type == "chat.tool_result":
            tool_call_id = first_non_empty(event.get("tool_call_id"))
            record = by_call_id.get(tool_call_id) if tool_call_id else last_formal_by_request.get(request_id)
            if record is None:
                record = new_record(request_id, event)
                if tool_call_id:
                    record.tool_call_id = tool_call_id
                    by_call_id[tool_call_id] = record
            if event.get("tool_name") and not record.name:
                record.name = str(event["tool_name"])
            if event.get("task_id") and not record.task_id:
                record.task_id = str(event["task_id"])
            result = event.get("result")
            record.result_time = timestamp
            record.result_line = line_no
            record.result_len = len(str(result)) if result is not None else 0
            record.result_status_hint = status_hint(result)

    for record in records:
        if record.delta_timestamps and record.formal_time is None:
            record.anomalies.append("missing chat.tool_call after delta stream")
        if record.formal_time is not None and record.result_time is None:
            record.anomalies.append("missing chat.tool_result")
        if record.result_time is not None and record.formal_time is None:
            record.anomalies.append("tool_result without chat.tool_call")
        if record.delta_timestamps and record.tool_call_id is None:
            record.anomalies.append("delta stream has no tool_call_id")

    summary = {
        "session_ids": sorted(session_ids),
        "request_count": len(request_ids),
        "tool_call_count": len(records),
        "records_with_delta": sum(1 for record in records if record.delta_timestamps),
        "records_with_result": sum(1 for record in records if record.result_time is not None),
        "records_with_anomalies": sum(1 for record in records if record.anomalies),
        "event_counts": event_counts,
    }
    return records, summary, warnings


def find_active_for_formal(
    active_by_request_index: dict[tuple[str, int], ToolCallRecord],
    request_id: str,
    name: Any,
) -> ToolCallRecord | None:
    candidates = [
        record
        for (candidate_request_id, _), record in active_by_request_index.items()
        if candidate_request_id == request_id
    ]
    if not candidates:
        return None
    if name:
        named = [record for record in candidates if record.name == str(name)]
        if named:
            return named[-1]
    return candidates[-1]


def needs_new_delta_record(
    record: ToolCallRecord | None,
    tool_call_id: str | None,
) -> bool:
    if record is None:
        return True
    if not tool_call_id or not record.tool_call_id:
        return False
    return record.tool_call_id != tool_call_id


def as_json_payload(
    path: Path,
    summary: dict[str, Any],
    warnings: list[str],
    records: list[ToolCallRecord],
    gap_threshold_ms: float,
) -> dict[str, Any]:
    return {
        "source": str(path),
        "summary": summary,
        "warnings": warnings,
        "tool_calls": [record.to_metrics(gap_threshold_ms) for record in records],
    }


def render_markdown(ctx: ReportContext) -> str:
    metrics = [record.to_metrics(ctx.gap_threshold_ms) for record in ctx.records]
    lines: list[str] = []
    lines.append("# History Tool Call Speed Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Source: `{ctx.path}`")
    lines.append(f"- Sessions: {', '.join(ctx.summary['session_ids']) or '-'}")
    lines.append(f"- Requests: {ctx.summary['request_count']}")
    lines.append(f"- Tool calls observed: {ctx.summary['tool_call_count']}")
    lines.append(f"- Tool calls with `chat.tool_calls.delta`: {ctx.summary['records_with_delta']}")
    lines.append(f"- Tool calls with `chat.tool_result`: {ctx.summary['records_with_result']}")
    lines.append(f"- Tool calls with anomalies: {ctx.summary['records_with_anomalies']}")
    lines.append(f"- Long delta gap threshold: {fmt_ms(ctx.gap_threshold_ms)}")
    lines.append("")
    if ctx.warnings:
        lines.append("## File Warnings")
        lines.append("")
        for warning in ctx.warnings[:20]:
            lines.append(f"- {warning}")
        if len(ctx.warnings) > 20:
            lines.append(f"- ... {len(ctx.warnings) - 20} more")
        lines.append("")

    lines.extend(render_top_table("Slowest Total Observed Tool Calls", metrics, "total_observed_ms", ctx.top))
    lines.extend(render_top_table("Slowest LLM Tool-Call Delta Streams", metrics, "llm_delta_duration_ms", ctx.top))
    lines.extend(render_top_table("Slowest Tool Executions", metrics, "tool_exec_duration_ms", ctx.top))

    lines.append("## Per-Request Timeline")
    lines.append("")
    by_request: dict[str, list[dict[str, Any]]] = {}
    for item in metrics:
        by_request.setdefault(item["request_id"], []).append(item)
    for request_id, items in by_request.items():
        lines.append(f"### Request `{request_id}`")
        lines.append("")
        lines.append(
            "| # | Tool | Delta | LLM delta | Max gap | Long gaps | "
            "Emit delay | Start delay | Tool exec | Total | Args | Result | Lines |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for item in items:
            lines.append(format_timeline_row(item))
        lines.append("")

    anomalies = [item for item in metrics if item["anomalies"]]
    if anomalies:
        lines.append("## Anomalies")
        lines.append("")
        for item in anomalies:
            anomaly_text = ", ".join(item["anomalies"])
            lines.append(
                f"- #{item['seq']} `{item['name']}` request `{item['request_id']}`: "
                f"{anomaly_text}"
            )
        lines.append("")

    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append(
        "- `chat.tool_calls.delta` measures history-observed streaming of "
        "tool-call arguments, not actual tool execution."
    )
    lines.append(
        "- `Tool exec` is measured from `chat.tool_update status=in_progress` to "
        "`chat.tool_result`; if no update exists, it falls back to `chat.tool_call` "
        "to `chat.tool_result`."
    )
    lines.append(
        "- History timestamps are client/runtime observation times. Use LLM trace "
        "timing to confirm model service-side latency when available."
    )
    return "\n".join(lines)


def render_top_table(title: str, metrics: list[dict[str, Any]], key: str, top: int) -> list[str]:
    rows = [item for item in metrics if item.get(key) is not None]
    rows.sort(key=lambda item: item[key], reverse=True)
    lines = [f"## {title}", ""]
    if not rows:
        lines.extend(["No data.", ""])
        return lines
    lines.append("| # | Tool | Request | Value | LLM delta | Tool exec | Delta count | Args | Result |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for item in rows[:top]:
        lines.append(format_top_table_row(item, key))
    lines.append("")
    return lines


def format_timeline_row(item: dict[str, Any]) -> str:
    result_len = item["result_len"] if item["result_len"] is not None else "-"
    return (
        "| {seq} | `{name}` | {delta_count} | {llm_delta_duration} | {max_gap} | "
        "{long_gaps} | {emit_delay} | {start_delay} | {tool_exec} | {total} | "
        "{args_len} | {result_len} | {lines_ref} |"
    ).format(
        seq=item["seq"],
        name=item["name"],
        delta_count=item["delta_count"],
        llm_delta_duration=fmt_ms(item["llm_delta_duration_ms"]),
        max_gap=fmt_ms(item["max_delta_gap_ms"]),
        long_gaps=item["long_gap_count"],
        emit_delay=fmt_ms(item["tool_call_emit_delay_ms"]),
        start_delay=fmt_ms(item["tool_start_delay_ms"]),
        tool_exec=fmt_ms(item["tool_exec_duration_ms"]),
        total=fmt_ms(item["total_observed_ms"]),
        args_len=item["arguments_len"],
        result_len=result_len,
        lines_ref=line_ref(item),
    )


def format_top_table_row(item: dict[str, Any], value_key: str) -> str:
    result_len = item["result_len"] if item["result_len"] is not None else "-"
    return (
        "| {seq} | `{name}` | `{request}` | {value} | {llm_delta} | {tool_exec} | "
        "{delta_count} | {args_len} | {result_len} |"
    ).format(
        seq=item["seq"],
        name=item["name"],
        request=short_id(item["request_id"]),
        value=fmt_ms(item[value_key]),
        llm_delta=fmt_ms(item["llm_delta_duration_ms"]),
        tool_exec=fmt_ms(item["tool_exec_duration_ms"]),
        delta_count=item["delta_count"],
        args_len=item["arguments_len"],
        result_len=result_len,
    )


def short_id(value: str) -> str:
    if len(value) <= 12:
        return value
    return value[:8] + "..."


def line_ref(item: dict[str, Any]) -> str:
    first_line = item.get("first_line")
    last_line = item.get("last_line")
    if first_line and last_line and first_line != last_line:
        return f"{first_line}-{last_line}"
    if first_line:
        return str(first_line)
    return "-"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze LLM tool-call delta streaming and tool execution timing from history.json.",
    )
    parser.add_argument("history_json", type=Path, help="Path to sessions/<session_id>/history.json")
    parser.add_argument("--top", type=int, default=10, help="Number of rows to show in each slow-call table")
    parser.add_argument(
        "--gap-threshold-ms",
        type=float,
        default=1000.0,
        help="Threshold used to count long gaps between chat.tool_calls.delta events",
    )
    parser.add_argument("--request-id", help="Only analyze one request_id")
    parser.add_argument("--json-out", type=Path, help="Optional path for machine-readable JSON output")
    args = parser.parse_args()

    events, read_warnings = read_events(args.history_json)
    records, summary, analyze_warnings = analyze_events(events, request_filter=args.request_id)
    warnings = read_warnings + analyze_warnings

    if args.json_out:
        payload = as_json_payload(
            args.history_json,
            summary,
            warnings,
            records,
            args.gap_threshold_ms,
        )
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_ctx = ReportContext(
        path=args.history_json,
        summary=summary,
        warnings=warnings,
        records=records,
        gap_threshold_ms=args.gap_threshold_ms,
        top=args.top,
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("%s", render_markdown(report_ctx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
