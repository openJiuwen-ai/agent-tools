"""Parse raw log lines into assembled LLM_IO_TRACE records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


_TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\[\d+\]"
)

_HDR_RE = re.compile(
    r"\[LLM_IO_TRACE\]\s+event=(?P<event>[\w.]+)\s+"
    # event_id 为可选字段：新版日志会在 event 之后、session_id 之前打印
    # event_id='xxx'，用于在同一 (session_id, request_id) 下并发调用时唯一标识
    # 一次 request/output 配对。旧日志不带该字段，仍走原有兜底逻辑。
    r"(?:event_id=(?P<eid>(?:'[^']*'|\"[^\"]*\"))\s+)?"
    r"session_id=(?P<sid>(?:'[^']*'|\"[^\"]*\"))\s+"
    r"request_id=(?P<rid>(?:'[^']*'|\"[^\"]*\"))\s+"
    r"iteration=(?P<iter>\S+)\s+"
    r"model_name=(?P<model>(?:'[^']*'|\"[^\"]*\"))"
)

_REASON_SEQ = re.compile(r"reasoning_seq=(?P<seq>\d+)\s+")


def _strip_q(x: str) -> str:
    x = x.strip()
    if len(x) >= 2 and x[0] in "'\"" and x[-1] == x[0]:
        return x[1:-1]
    return x


def parse_ts(line: str) -> datetime | None:
    m = _TS_RE.match(line)
    if not m:
        return None
    return datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S.%f")


@dataclass
class TraceRecord:
    ts: datetime
    event: str
    session_id: str
    request_id: str
    iteration: str
    model_name: str
    event_id: str | None = None  # 用于同一 (sid, rid) 下并发调用的唯一配对标记
    body: str = ""
    reasoning_seq: int | None = None
    body_part: tuple[int, int] | None = None  # (i, total) if multipart line
    raw_line_no: int = 0


def _extract_body_segment(line: str) -> tuple[tuple[int, int] | None, int | None, str]:
    """Return (body_part i/total or None, reasoning_seq or None, body text)."""
    seq_val: int | None = None
    mseq = _REASON_SEQ.search(line)
    if mseq:
        seq_val = int(mseq.group("seq"))

    bp: tuple[int, int] | None = None
    mbp = re.search(r"\sbody_part=(\d+)/(\d+)\s+body=", line)
    if mbp:
        bp = (int(mbp.group(1)), int(mbp.group(2)))
        body = line[mbp.end():]
        return bp, seq_val, body

    mbody = re.search(r"\sbody=", line)
    if mbody:
        return None, seq_val, line[mbody.end():]

    return None, seq_val, ""


def parse_trace_line(line: str, line_no: int) -> TraceRecord | None:
    if "[LLM_IO_TRACE]" not in line:
        return None
    ts = parse_ts(line)
    if ts is None:
        return None
    m = _HDR_RE.search(line)
    if not m:
        return None
    bp, rseq, body = _extract_body_segment(line)
    eid_raw = m.group("eid")
    return TraceRecord(
        ts=ts,
        event=m.group("event"),
        event_id=_strip_q(eid_raw) if eid_raw else None,
        session_id=_strip_q(m.group("sid")),
        request_id=_strip_q(m.group("rid")),
        iteration=m.group("iter"),
        model_name=_strip_q(m.group("model")),
        body=body,
        reasoning_seq=rseq,
        body_part=bp,
        raw_line_no=line_no,
    )


def assemble_records(raw_records: list[TraceRecord]) -> list[TraceRecord]:
    """Merge body_part=i/total chunks into single records (first timestamp kept).

    Chunks are merged in file order; each multipart group shares the same
    (session_id, request_id, iteration, event, model_name, total) until `total`
    parts are collected. Consecutive LLM rounds therefore form independent groups.
    """
    out: list[TraceRecord] = []
    idx = 0
    n = len(raw_records)
    while idx < n:
        rec = raw_records[idx]
        start_idx = idx
        if rec.body_part is None:
            out.append(rec)
            idx += 1
            continue
        pi, pt = rec.body_part
        meta = (rec.session_id, rec.request_id, rec.iteration, rec.event, rec.model_name, rec.event_id, pt)
        parts: dict[int, str] = {pi: rec.body}
        start_ts = rec.ts
        start_ln = rec.raw_line_no
        idx += 1
        while idx < n and len(parts) < pt:
            nx = raw_records[idx]
            if not nx.body_part:
                break
            qi, qt = nx.body_part
            nm = (nx.session_id, nx.request_id, nx.iteration, nx.event, nx.model_name, nx.event_id, qt)
            if nm != meta or qt != pt:
                break
            parts[qi] = nx.body
            idx += 1
        if len(parts) == pt:
            merged_body = "".join(parts[j] for j in range(1, pt + 1))
            out.append(
                TraceRecord(
                    ts=start_ts,
                    event=rec.event,
                    event_id=rec.event_id,
                    session_id=rec.session_id,
                    request_id=rec.request_id,
                    iteration=rec.iteration,
                    model_name=rec.model_name,
                    body=merged_body,
                    reasoning_seq=None,
                    body_part=None,
                    raw_line_no=start_ln,
                )
            )
        else:
            # Incomplete group: emit first slice only and advance one line (avoid deadlock)
            out.append(
                TraceRecord(
                    ts=start_ts,
                    event=rec.event,
                    event_id=rec.event_id,
                    session_id=rec.session_id,
                    request_id=rec.request_id,
                    iteration=rec.iteration,
                    model_name=rec.model_name,
                    body=rec.body,
                    body_part=None,
                    raw_line_no=start_ln,
                )
            )
            idx = start_idx + 1

    return sorted(out, key=lambda r: (r.ts, r.raw_line_no))


def filter_sessions(records: list[TraceRecord], root_session: str) -> list[TraceRecord]:
    """Include root session and child sessions (prefix root + _)."""

    def keep(sid: str) -> bool:
        if sid == root_session:
            return True
        if sid.startswith(root_session + "_"):
            return True
        return False

    return [r for r in records if keep(r.session_id)]


def extract_tokens_from_body(body: str) -> dict[str, int | None]:
    """usage_metadata in logs is often a stringified struct; also try JSON."""
    out: dict[str, int | None] = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        m = re.search(rf"{name}[=:]?\s*(\d+)", body)
        if m:
            out[name] = int(m.group(1))
    if body.strip().startswith("{"):
        try:
            data = json.loads(body)
            um = data.get("usage_metadata")
            if isinstance(um, dict):
                for k in out:
                    if k in um and isinstance(um[k], int):
                        out[k] = um[k]
            elif isinstance(um, str):
                for name in out:
                    m = re.search(rf"{name}[=:]?\s*(\d+)", um)
                    if m:
                        out[name] = int(m.group(1))
        except json.JSONDecodeError:
            pass
    return out


def extract_tool_calls(body: str) -> list[dict[str, Any]]:
    if not body.strip().startswith("{"):
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    raw = data.get("tool_calls") or []
    tools: list[dict[str, Any]] = []
    for tc in raw:
        if isinstance(tc, dict):
            tools.append(
                {
                    "name": tc.get("name"),
                    "arguments": tc.get("arguments"),
                    "id": tc.get("id"),
                }
            )
    return tools


def summarize_tools(tools: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for t in tools:
        n = t.get("name") or "?"
        parts.append(str(n))
    return ", ".join(parts) if parts else "(no tool_calls)"
