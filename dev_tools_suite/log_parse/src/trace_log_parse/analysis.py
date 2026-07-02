"""Build per-request timeline: LLM rounds, tool gaps, subagent markers, tokens."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from trace_log_parse.parser import (
    TraceRecord,
    extract_tool_calls,
    extract_tokens_from_body,
    summarize_tools,
)

SessionPath = tuple[tuple[str, str, str], ...]  # (session_id, label, kind)
PairKey = tuple[str, str, str]  # (kind, sid_or_eid, rid_or_empty) for LLM round pairing


@dataclass
class LLMRound:
    kind: Literal["stream", "invoke"]
    session_id: str
    request_id: str
    iteration: str
    model_name: str
    request_ts: datetime
    output_ts: datetime
    duration_sec: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    reasoning_chars: int = 0
    reasoning_batches: int = 0
    reasoning_full: str = ""
    assistant_preview: str = ""
    request_body_full: str = ""
    output_body_excerpt: str = ""
    is_child_session: bool = False
    child_label: str = ""
    child_path: SessionPath = ()
    is_post_final: bool = False

    def token_summary(self) -> str:
        parts = []
        if self.input_tokens is not None:
            parts.append(f"in={self.input_tokens}")
        if self.output_tokens is not None:
            parts.append(f"out={self.output_tokens}")
        if self.total_tokens is not None:
            parts.append(f"Σ={self.total_tokens}")
        return ", ".join(parts) if parts else "—"


@dataclass
class ToolGap:
    """Time between LLM output (tool_calls emitted) and next LLM request."""

    session_id: str
    after_output_ts: datetime
    next_request_ts: datetime
    duration_sec: float
    tools_triggered: str
    detail_tools: list[dict[str, Any]]
    from_kind: Literal["stream", "invoke"]
    is_child_session: bool = False
    child_label: str = ""
    child_path: SessionPath = ()
    is_post_final: bool = False


def _short_suffix(value: str) -> str:
    return value[:12] if len(value) > 12 else value


def _session_path(session_id: str, root: str) -> SessionPath:
    """Parse session_id suffix into a chain: root → subagent → fork → fork → …

    Matches SubagentSessionProxy nesting: each segment appends ``_<subagent_id>`` or
    ``_<fork_agent_id>`` (ids are hex-ish tokens without underscores). A single
    ``_fork_agent_`` occurrence = one tree level, so nested forks render as nested groups.
    """
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
            f"subagent ⋯{_short_suffix(suffix_id)}"
            if kind == "subagent"
            else f"forkagent ⋯{_short_suffix(suffix_id)}"
        )
        path.append((node_sid, label, kind))

    if path:
        return tuple(path)

    return ((session_id, "child-session", "child"),)


def _child_info(session_id: str, root: str) -> tuple[bool, str, SessionPath]:
    path = _session_path(session_id, root)
    if not path:
        return False, "", ()
    return True, path[-1][1], path


def interval_union_sec(intervals: list[tuple[datetime, datetime]]) -> float:
    """Wall-clock duration covered by intervals, without double-counting overlaps."""
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


def build_timeline(records: list[TraceRecord], root_session: str) -> tuple[list[LLMRound], list[ToolGap]]:
    """Pair request→output within each session; attach reasoning deltas; derive tool gaps.

    配对键策略：
    - 新版日志带 ``event_id``，request 与对应 output（以及期间的 reasoning_delta）共享同
      一 event_id，可严格唯一配对，支持同一 (session_id, request_id) 下的并发调用。
    - 旧版日志没有 ``event_id``，退化为 ``(session_id, request_id)``，与之前行为一致。

    Tool gap 推导（output → next request）必须按 ``(session_id, request_id)`` 维度排队，
    因为下一次 request 并不知道它该接续哪个 event。为支持并发，``last_outputs`` 改为
    FIFO 队列，按到达顺序匹配，避免被后到的 output 直接覆盖。
    """

    def _pair_key(rec: TraceRecord) -> PairKey:
        """Pairing key prefers event_id (concurrency-safe), else (sid, rid)."""
        if rec.event_id:
            return ("eid", rec.event_id, "")
        return ("sr", rec.session_id, rec.request_id)

    # Accumulate reasoning_delta sizes between pairing windows;
    # 同 pair_key 累积，配对成 round 时一次性冲刷
    pending_reasoning_chars: dict[PairKey, int] = {}
    pending_reasoning_batches: dict[PairKey, int] = {}
    pending_reasoning_text: dict[PairKey, list[str]] = {}

    rounds: list[LLMRound] = []
    gaps: list[ToolGap] = []

    pending_stream: dict[PairKey, TraceRecord] = {}
    pending_invoke: dict[PairKey, TraceRecord] = {}

    # FIFO queue per (sid, rid)：支持同 request_id 下并发产生的多次 output，
    # 后到的 next_request 会按到达顺序消费最早的未匹配 output。
    last_outputs: dict[
        tuple[str, str],
        list[tuple[datetime, list[dict[str, Any]], Literal["stream", "invoke"], str, str]],
    ] = {}

    # tool_call_request / tool_call_output：执行侧成对事件，通过 event_id 严格配对。
    # 它们独立于 LLM request↔output 配对流程：不入 pending_* 队列，不消费 last_outputs，
    # 否则会破坏原本的 stream/invoke 配对节奏。处理完毕后再按 tool_call_id 把真实
    # 执行耗时 / 结果体回填到 LLM gap 子项中。
    tool_call_events: dict[str, dict[str, Any]] = {}

    def _record_tool_call(rec: TraceRecord) -> None:
        """Pair tool_call_request/output by event_id (mandatory for these events)."""
        if not rec.event_id:
            return
        slot = tool_call_events.setdefault(
            rec.event_id,
            {
                "tool_call_id": rec.tool_call_id,
                "tool_name": rec.tool_name,
                "agent": rec.agent,
                "session_id": rec.session_id,
                "request_id": rec.request_id,
                "request_ts": None,
                "request_body": "",
                "output_ts": None,
                "output_body": "",
            },
        )
        # 后到的元数据若缺失，沿用首次写入；这里只在仍为 None/空时补充。
        for k in ("tool_call_id", "tool_name", "agent"):
            if not slot.get(k):
                slot[k] = getattr(rec, k)
        if rec.event == "tool_call_request":
            slot["request_ts"] = rec.ts
            slot["request_body"] = rec.body
        elif rec.event == "tool_call_output":
            slot["output_ts"] = rec.ts
            slot["output_body"] = rec.body

    # Sort for safety
    recs = sorted(records, key=lambda r: (r.ts, r.raw_line_no))
    final_at: dict[tuple[str, str], datetime] = {}
    for rec in recs:
        if rec.event == "chat.final":
            sr_key = (rec.session_id, rec.request_id)
            final_at.setdefault(sr_key, rec.ts)

    def _is_post_final(session_id: str, request_id: str, ts: datetime) -> bool:
        cutoff = final_at.get((session_id, request_id))
        return cutoff is not None and ts >= cutoff

    for rec in recs:
        # tool_call_* 走独立通道，不影响 LLM 的 request/output 配对
        if rec.event in ("tool_call_request", "tool_call_output"):
            _record_tool_call(rec)
            continue

        key = _pair_key(rec)
        sr_key = (rec.session_id, rec.request_id)

        if rec.event == "reasoning_delta":
            pending_reasoning_chars[key] = pending_reasoning_chars.get(key, 0) + len(rec.body)
            pending_reasoning_batches[key] = pending_reasoning_batches.get(key, 0) + 1
            pending_reasoning_text.setdefault(key, []).append(rec.body)
            continue

        if rec.event == "chat.final":
            continue

        if rec.event in ("stream_request", "invoke_request"):
            queue = last_outputs.get(sr_key)
            last_output = queue.pop(0) if queue else None
            if queue is not None and not queue:
                last_outputs.pop(sr_key, None)
            if last_output is not None:
                last_output_ts, last_tools, last_kind, last_session_for_output, last_request_for_output = last_output
            if last_output is not None and rec.ts >= last_output_ts:
                cutoff = final_at.get(sr_key)
                is_post_gap = _is_post_final(rec.session_id, rec.request_id, rec.ts)
                gap_start = last_output_ts
                if cutoff is not None and rec.ts >= cutoff and gap_start < cutoff:
                    gap_start = cutoff
                gap_sec = (rec.ts - gap_start).total_seconds()
                if gap_sec >= 0:
                    gap_is_child, gap_label, gap_path = _child_info(last_session_for_output, root_session)
                    gaps.append(
                        ToolGap(
                            session_id=last_session_for_output,
                            after_output_ts=gap_start,
                            next_request_ts=rec.ts,
                            duration_sec=gap_sec,
                            tools_triggered=summarize_tools(last_tools) if not is_post_gap else "(post-final)",
                            detail_tools=[] if is_post_gap else list(last_tools),
                            from_kind=last_kind or "stream",
                            is_child_session=gap_is_child,
                            child_label=gap_label,
                            child_path=gap_path,
                            is_post_final=is_post_gap,
                        )
                    )

            if rec.event == "stream_request":
                pending_stream[key] = rec
            else:
                pending_invoke[key] = rec
            continue

        if rec.event == "stream_output":
            req = pending_stream.pop(key, None)
            if req is None:
                rk, rch = pending_reasoning_batches.get(key, 0), pending_reasoning_chars.get(key, 0)
                pending_reasoning_batches[key] = 0
                pending_reasoning_chars[key] = 0
                rtext = "".join(pending_reasoning_text.get(key, []))
                pending_reasoning_text[key] = []

                tokens = extract_tokens_from_body(rec.body)
                tools = extract_tool_calls(rec.body)
                assistant_text = ""
                if rec.body.strip().startswith("{"):
                    try:
                        data = json.loads(rec.body)
                        assistant_text = str(data.get("content") or "")[:500]
                    except json.JSONDecodeError:
                        assistant_text = rec.body[:400]

                is_child, clabel, cpath = _child_info(rec.session_id, root_session)
                rounds.append(
                    LLMRound(
                        kind="stream",
                        session_id=rec.session_id,
                        request_id=rec.request_id,
                        iteration=rec.iteration,
                        model_name=rec.model_name,
                        request_ts=rec.ts,
                        output_ts=rec.ts,
                        duration_sec=0.0,
                        input_tokens=tokens.get("input_tokens"),
                        output_tokens=tokens.get("output_tokens"),
                        total_tokens=tokens.get("total_tokens"),
                        tools=tools,
                        reasoning_chars=rch,
                        reasoning_batches=rk,
                        reasoning_full=rtext,
                        assistant_preview=assistant_text,
                        request_body_full="",
                        output_body_excerpt=rec.body,
                        is_child_session=is_child,
                        child_label=clabel,
                        child_path=cpath,
                        is_post_final=_is_post_final(rec.session_id, rec.request_id, rec.ts),
                    )
                )
                last_outputs.setdefault(sr_key, []).append(
                    (rec.ts, tools, "stream", rec.session_id, rec.request_id)
                )
                continue
            rk, rch = pending_reasoning_batches.get(key, 0), pending_reasoning_chars.get(key, 0)
            pending_reasoning_batches[key] = 0
            pending_reasoning_chars[key] = 0
            rtext = "".join(pending_reasoning_text.get(key, []))
            pending_reasoning_text[key] = []

            tokens = extract_tokens_from_body(rec.body)
            tools = extract_tool_calls(rec.body)
            assistant_text = ""
            if rec.body.strip().startswith("{"):
                try:
                    data = json.loads(rec.body)
                    assistant_text = str(data.get("content") or "")[:500]
                except json.JSONDecodeError:
                    assistant_text = rec.body[:400]

            is_child, clabel, cpath = _child_info(rec.session_id, root_session)
            dur = (rec.ts - req.ts).total_seconds()
            rounds.append(
                LLMRound(
                    kind="stream",
                    session_id=rec.session_id,
                    request_id=rec.request_id,
                    iteration=rec.iteration,
                    model_name=rec.model_name,
                    request_ts=req.ts,
                    output_ts=rec.ts,
                    duration_sec=dur,
                    input_tokens=tokens.get("input_tokens"),
                    output_tokens=tokens.get("output_tokens"),
                    total_tokens=tokens.get("total_tokens"),
                    tools=tools,
                    reasoning_chars=rch,
                    reasoning_batches=rk,
                    reasoning_full=rtext,
                    assistant_preview=assistant_text,
                    request_body_full=req.body,
                    output_body_excerpt=rec.body,
                    is_child_session=is_child,
                    child_label=clabel,
                    child_path=cpath,
                    is_post_final=_is_post_final(rec.session_id, rec.request_id, req.ts),
                )
            )
            # Always bind gaps to the current output's own tool_calls.
            # Reusing previous non-empty tools causes phantom tool gaps
            # (e.g. a later round without spawn_subagent still shows it).
            last_outputs.setdefault(sr_key, []).append(
                (rec.ts, list(tools), "stream", rec.session_id, rec.request_id)
            )
            continue

        if rec.event == "invoke_output":
            req = pending_invoke.pop(key, None)
            if req is None:
                rk_i = pending_reasoning_batches.pop(key, 0)
                rch_i = pending_reasoning_chars.pop(key, 0)
                rtext_i = "".join(pending_reasoning_text.pop(key, []))

                tokens = extract_tokens_from_body(rec.body)
                tools = extract_tool_calls(rec.body)
                assistant_text = ""
                if rec.body.strip().startswith("{"):
                    try:
                        data = json.loads(rec.body)
                        assistant_text = str(data.get("content") or "")[:500]
                    except json.JSONDecodeError:
                        assistant_text = rec.body[:400]

                is_child, clabel, cpath = _child_info(rec.session_id, root_session)
                rounds.append(
                    LLMRound(
                        kind="invoke",
                        session_id=rec.session_id,
                        request_id=rec.request_id,
                        iteration=rec.iteration,
                        model_name=rec.model_name,
                        request_ts=rec.ts,
                        output_ts=rec.ts,
                        duration_sec=0.0,
                        input_tokens=tokens.get("input_tokens"),
                        output_tokens=tokens.get("output_tokens"),
                        total_tokens=tokens.get("total_tokens"),
                        tools=tools,
                        reasoning_chars=rch_i,
                        reasoning_batches=rk_i,
                        reasoning_full=rtext_i,
                        assistant_preview=assistant_text,
                        request_body_full="",
                        output_body_excerpt=rec.body,
                        is_child_session=is_child,
                        child_label=clabel,
                        child_path=cpath,
                        is_post_final=_is_post_final(rec.session_id, rec.request_id, rec.ts),
                    )
                )
                last_outputs.setdefault(sr_key, []).append(
                    (rec.ts, tools, "invoke", rec.session_id, rec.request_id)
                )
                continue
            rk_i = pending_reasoning_batches.pop(key, 0)
            rch_i = pending_reasoning_chars.pop(key, 0)
            rtext_i = "".join(pending_reasoning_text.pop(key, []))

            tokens = extract_tokens_from_body(rec.body)
            tools = extract_tool_calls(rec.body)
            assistant_text = ""
            if rec.body.strip().startswith("{"):
                try:
                    data = json.loads(rec.body)
                    assistant_text = str(data.get("content") or "")[:500]
                except json.JSONDecodeError:
                    assistant_text = rec.body[:400]

            is_child, clabel, cpath = _child_info(rec.session_id, root_session)
            dur = (rec.ts - req.ts).total_seconds()
            rounds.append(
                LLMRound(
                    kind="invoke",
                    session_id=rec.session_id,
                    request_id=rec.request_id,
                    iteration=rec.iteration,
                    model_name=rec.model_name,
                    request_ts=req.ts,
                    output_ts=rec.ts,
                    duration_sec=dur,
                    input_tokens=tokens.get("input_tokens"),
                    output_tokens=tokens.get("output_tokens"),
                    total_tokens=tokens.get("total_tokens"),
                    tools=tools,
                    reasoning_chars=rch_i,
                    reasoning_batches=rk_i,
                    reasoning_full=rtext_i,
                    assistant_preview=assistant_text,
                    request_body_full=req.body,
                    output_body_excerpt=rec.body,
                    is_child_session=is_child,
                    child_label=clabel,
                    child_path=cpath,
                    is_post_final=_is_post_final(rec.session_id, rec.request_id, req.ts),
                )
            )
            # Keep only current output's tool_calls; do not inherit from prior output.
            last_outputs.setdefault(sr_key, []).append(
                (rec.ts, list(tools), "invoke", rec.session_id, rec.request_id)
            )

    # Handle terminal outputs that emitted tool_calls but have no following request.
    # We still surface these tool calls in the report as zero-duration tool gaps.
    for _sr_key, queue in last_outputs.items():
        for out_ts, out_tools, out_kind, out_sid, out_rid in queue:
            if not out_tools:
                continue
            is_child, label, path = _child_info(out_sid, root_session)
            gaps.append(
                ToolGap(
                    session_id=out_sid,
                    after_output_ts=out_ts,
                    next_request_ts=out_ts,
                    duration_sec=0.0,
                    tools_triggered=summarize_tools(out_tools),
                    detail_tools=list(out_tools),
                    from_kind=out_kind,
                    is_child_session=is_child,
                    child_label=label,
                    child_path=path,
                    is_post_final=_is_post_final(out_sid, out_rid, out_ts),
                )
            )

    # ── tool_call 注入通 ──────────────────────────────────────────
    # 将 tool_call_request / tool_call_output 成对数据回填到 LLM gap
    # 的 detail_tools 子项中，使报告能展示真实执行耗时与结果。
    # 未命中任何 gap 的孤立 tool_call 对兜底为独立 ToolGap。
    #
    # Build index: tool_call_id → paired data
    paired_tool_calls: dict[str, dict[str, Any]] = {}
    for slot in tool_call_events.values():
        tcid = slot.get("tool_call_id")
        if not tcid or slot.get("request_ts") is None:
            continue
        paired_tool_calls[tcid] = slot

    orphan_tcids: set[str] = set(paired_tool_calls.keys())

    # Inject into existing gap detail_tools
    for gap in gaps:
        for dt in gap.detail_tools:
            dt_id = dt.get("id")
            if not dt_id or dt_id not in paired_tool_calls:
                continue
            pc = paired_tool_calls[dt_id]
            # 注入额外元数据（不会破坏现有渲染，report.py 只会读取 name/arguments/id）
            dt["_start_ts"] = pc["request_ts"].isoformat()
            dt["_end_ts"] = pc.get("output_ts").isoformat() if pc.get("output_ts") else ""
            dt["_duration_ms"] = round(
                (
                    (pc["output_ts"] - pc["request_ts"]).total_seconds() * 1000
                )
                if pc.get("output_ts") else 0,
                1,
            )
            # 从 output body 解析状态与结果片段
            out_body = pc.get("output_body", "")
            if out_body.strip().startswith("{"):
                try:
                    od = json.loads(out_body)
                    dt["_tool_status"] = od.get("status", "")
                    dt["_tool_duration_ms"] = od.get("duration_ms")
                    # 提取结果短摘要供报告展示（完整 body 有独立的可展开渲染）
                    result = od.get("result") or od.get("error") or ""
                    if isinstance(result, dict):
                        result = json.dumps(result, ensure_ascii=False)[:500]
                    dt["_tool_result"] = str(result) if result else ""
                except json.JSONDecodeError:
                    pass
            orphan_tcids.discard(dt_id)

    # 孤立 tool_call 对 → 兜底 ToolGap（日志中有 tool_call 但无 LLM gap 覆盖时可见）
    # 每个 tool_call 独立为一个 gap，携带真实起止时间，并发调用在时间线上自然重叠。
    for tcid in orphan_tcids:
        pc = paired_tool_calls[tcid]
        sk_sid = pc["session_id"]
        sk_rid = pc["request_id"]
        gap_is_child, gap_label, gap_path = _child_info(sk_sid, root_session)
        out_ts = pc.get("output_ts") or pc["request_ts"]
        duration = (out_ts - pc["request_ts"]).total_seconds()

        item: dict[str, Any] = {
            "name": pc.get("tool_name") or "?",
            "tool_call_id": pc.get("tool_call_id"),
            "request_body": pc.get("request_body", ""),
            "_start_ts": pc["request_ts"].isoformat(),
            "_end_ts": out_ts.isoformat(),
        }
        out_body = pc.get("output_body", "")
        if out_body.strip().startswith("{"):
            try:
                od = json.loads(out_body)
                item["_tool_status"] = od.get("status", "")
                item["_tool_duration_ms"] = od.get("duration_ms")
                result = od.get("result") or od.get("error") or ""
                if isinstance(result, dict):
                    result = json.dumps(result, ensure_ascii=False)[:500]
                item["_tool_result"] = str(result) if result else ""
            except json.JSONDecodeError:
                pass

        gaps.append(
            ToolGap(
                session_id=sk_sid,
                after_output_ts=pc["request_ts"],
                next_request_ts=out_ts,
                duration_sec=duration,
                tools_triggered=pc.get("tool_name") or "?",
                detail_tools=[item],
                from_kind="invoke",
                is_child_session=gap_is_child,
                child_label=gap_label,
                child_path=gap_path,
                is_post_final=_is_post_final(sk_sid, sk_rid, pc["request_ts"]),
            )
        )

    # ── 流程间隙裁剪 ────────────────────────────────────────────
    # LLM output→next_request 之间派生的"流程间隙" gap（detail_tools 为空）
    # 本质上代表"LLM 输出后到下次请求前的等待"。当中间发生了工具调用 *或者
    # 其它并发 LLM round* 时，原始时间窗会包住实际忙碌的区间，导致显示一段
    # 假的"空闲"。这里用 "工具调用 + 其它 LLM round" 的并集对 idle gap 做减法，
    # 仅保留真正未被任何活动覆盖的残段，过短残段直接丢弃。
    busy_intervals: list[tuple[datetime, datetime]] = sorted(
        [
            (g.after_output_ts, g.next_request_ts)
            for g in gaps
            if g.detail_tools and (g.tools_triggered or "") not in ("(no tool_calls)", "(post-final)")
        ]
        + [
            # 并发 LLM round 自身的活动区间，覆盖时也算"忙碌"，避免被误判为 idle
            (r.request_ts, r.output_ts)
            for r in rounds
            if r.output_ts > r.request_ts
        ]
    )

    def _subtract_intervals(
        start: datetime, end: datetime, covers: list[tuple[datetime, datetime]]
    ) -> list[tuple[datetime, datetime]]:
        """Return parts of [start, end] not covered by any interval in covers."""
        out_parts: list[tuple[datetime, datetime]] = []
        cur = start
        for cs, ce in covers:
            if ce <= cur or cs >= end:
                continue
            if cs > cur:
                out_parts.append((cur, cs))
            if ce > cur:
                cur = ce
            if cur >= end:
                break
        if cur < end:
            out_parts.append((cur, end))
        return out_parts

    idle_min_sec = 0.05  # 50ms：低于此值的残段视为日志抖动，丢弃

    trimmed_gaps: list[ToolGap] = []
    for g in gaps:
        is_idle = (not g.detail_tools) or (g.tools_triggered or "") in ("(no tool_calls)", "(post-final)")
        if not is_idle:
            trimmed_gaps.append(g)
            continue
        # Find covering busy intervals (tool calls + concurrent LLM rounds) overlapping this idle gap
        covers = [
            (cs, ce) for cs, ce in busy_intervals
            if ce > g.after_output_ts and cs < g.next_request_ts
        ]
        if not covers:
            # 无忙碌覆盖：保留原间隙
            if g.duration_sec >= idle_min_sec:
                trimmed_gaps.append(g)
            continue
        residuals = _subtract_intervals(g.after_output_ts, g.next_request_ts, covers)
        for r_start, r_end in residuals:
            sec = (r_end - r_start).total_seconds()
            if sec < idle_min_sec:
                continue
            trimmed_gaps.append(
                ToolGap(
                    session_id=g.session_id,
                    after_output_ts=r_start,
                    next_request_ts=r_end,
                    duration_sec=sec,
                    tools_triggered=g.tools_triggered,
                    detail_tools=list(g.detail_tools),
                    from_kind=g.from_kind,
                    is_child_session=g.is_child_session,
                    child_label=g.child_label,
                    child_path=g.child_path,
                    is_post_final=g.is_post_final,
                )
            )
    gaps = trimmed_gaps

    # ── idle gap 去重 ───────────────────────────────────────────
    # 同一 (sid, rid) 下并发 LLM 调用时，多个 (output_i → request_j) 会派生
    # 时间窗口完全相同的"流程间隙"。它们语义上是同一段等待，渲染时三连重复展示，
    # 这里按 (child_path, session_id, after_output_ts, next_request_ts) 去重。
    seen_idle_keys: set[tuple[Any, ...]] = set()
    dedup_gaps: list[ToolGap] = []
    for g in gaps:
        is_idle = (not g.detail_tools) or (g.tools_triggered or "") in ("(no tool_calls)", "(post-final)")
        if is_idle:
            dedup_key = (
                tuple(g.child_path),
                g.session_id,
                g.after_output_ts,
                g.next_request_ts,
            )
            if dedup_key in seen_idle_keys:
                continue
            seen_idle_keys.add(dedup_key)
        dedup_gaps.append(g)
    gaps = dedup_gaps

    gaps.sort(key=lambda g: (g.after_output_ts, g.next_request_ts))
    return rounds, gaps


def aggregate_totals(rounds: list[LLMRound]) -> dict[str, Any]:
    tin = sum((r.input_tokens or 0) for r in rounds)
    tout = sum((r.output_tokens or 0) for r in rounds)
    ttot = sum((r.total_tokens or 0) for r in rounds)
    llm_sec = interval_union_sec([(r.request_ts, r.output_ts) for r in rounds])
    return {
        "rounds": len(rounds),
        "llm_wall_sec": round(llm_sec, 3),
        "input_tokens_sum": tin or None,
        "output_tokens_sum": tout or None,
        "total_tokens_sum": ttot or None,
    }
