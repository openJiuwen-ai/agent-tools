"""Build per-request timeline: LLM rounds, tool gaps, subagent markers, tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from full_parse.trace_parser import (
    TraceRecord,
    extract_tool_calls,
    extract_tokens_from_body,
    summarize_tools,
)
from history_parse.session import child_info as _child_info, interval_union_sec

SessionPath = tuple[tuple[str, str, str], ...]  # (session_id, label, kind)


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
    cache_tokens: int | None = None
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
        if self.cache_tokens:
            parts.append(f"cache={self.cache_tokens}")
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
    has_spawn_subagent: bool = False
    has_fork_agent: bool = False
    is_pure_spawn_fork: bool = False


def _short_suffix(value: str) -> str:
    return value[:12] if len(value) > 12 else value


def build_timeline(records: list[TraceRecord], root_session: str) -> tuple[list[LLMRound], list[ToolGap]]:
    """Pair request→output within each session; attach reasoning deltas; derive tool gaps."""

    # Accumulate reasoning_delta sizes between pairing windows by (sid,rid) roughly:
    # assign to next output by flushing when we pin an output timestamp
    pending_reasoning_chars: dict[tuple[str, str], int] = {}
    pending_reasoning_batches: dict[tuple[str, str], int] = {}
    pending_reasoning_text: dict[tuple[str, str], list[str]] = {}

    rounds: list[LLMRound] = []
    gaps: list[ToolGap] = []

    pending_stream: dict[tuple[str, str], TraceRecord] = {}
    pending_invoke: dict[tuple[str, str], TraceRecord] = {}

    last_outputs: dict[
        tuple[str, str],
        tuple[datetime, list[dict[str, Any]], Literal["stream", "invoke"], str, str],
    ] = {}

    # Sort for safety
    recs = sorted(records, key=lambda r: (r.ts, r.raw_line_no))
    final_at: dict[tuple[str, str], datetime] = {}
    for rec in recs:
        if rec.event == "chat.final":
            key = (rec.session_id, rec.request_id)
            final_at.setdefault(key, rec.ts)

    def _is_post_final(session_id: str, request_id: str, ts: datetime) -> bool:
        cutoff = final_at.get((session_id, request_id))
        return cutoff is not None and ts >= cutoff

    for rec in recs:
        key = (rec.session_id, rec.request_id)

        if rec.event == "reasoning_delta":
            pending_reasoning_chars[key] = pending_reasoning_chars.get(key, 0) + len(rec.body)
            pending_reasoning_batches[key] = pending_reasoning_batches.get(key, 0) + 1
            pending_reasoning_text.setdefault(key, []).append(rec.body)
            continue

        if rec.event == "chat.final":
            continue

        if rec.event in ("stream_request", "invoke_request"):
            last_output = last_outputs.pop(key, None)
            if last_output is not None:
                last_output_ts, last_tools, last_kind, last_session_for_output, last_request_for_output = last_output
            if last_output is not None and rec.ts >= last_output_ts:
                cutoff = final_at.get(key)
                is_post_gap = _is_post_final(rec.session_id, rec.request_id, rec.ts)
                gap_start = last_output_ts
                if cutoff is not None and rec.ts >= cutoff and gap_start < cutoff:
                    gap_start = cutoff
                gap_sec = (rec.ts - gap_start).total_seconds()
                if gap_sec >= 0:
                    gap_is_child, gap_label, gap_path = _child_info(last_session_for_output, root_session)
                    gap_tools = [] if is_post_gap else list(last_tools)
                    has_spawn, has_fork, is_pure = classify_tools(gap_tools)
                    gaps.append(
                        ToolGap(
                            session_id=last_session_for_output,
                            after_output_ts=gap_start,
                            next_request_ts=rec.ts,
                            duration_sec=gap_sec,
                            tools_triggered=summarize_tools(last_tools) if not is_post_gap else "(post-final)",
                            detail_tools=gap_tools,
                            from_kind=last_kind or "stream",
                            is_child_session=gap_is_child,
                            child_label=gap_label,
                            child_path=gap_path,
                            is_post_final=is_post_gap,
                            has_spawn_subagent=has_spawn,
                            has_fork_agent=has_fork,
                            is_pure_spawn_fork=is_pure,
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
                        import json

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
                        cache_tokens=tokens.get("cache_tokens"),
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
                last_outputs[key] = (rec.ts, tools, "stream", rec.session_id, rec.request_id)
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
                    import json

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
                    cache_tokens=tokens.get("cache_tokens"),
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
            last_outputs[key] = (rec.ts, list(tools), "stream", rec.session_id, rec.request_id)
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
                        import json

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
                        cache_tokens=tokens.get("cache_tokens"),
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
                last_outputs[key] = (rec.ts, tools, "invoke", rec.session_id, rec.request_id)
                continue
            rk_i = pending_reasoning_batches.pop(key, 0)
            rch_i = pending_reasoning_chars.pop(key, 0)
            rtext_i = "".join(pending_reasoning_text.pop(key, []))

            tokens = extract_tokens_from_body(rec.body)
            tools = extract_tool_calls(rec.body)
            assistant_text = ""
            if rec.body.strip().startswith("{"):
                try:
                    import json

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
                    cache_tokens=tokens.get("cache_tokens"),
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
            last_outputs[key] = (rec.ts, list(tools), "invoke", rec.session_id, rec.request_id)

    # Handle terminal outputs that emitted tool_calls but have no following request.
    # We still surface these tool calls in the report as zero-duration tool gaps.
    for (_, _), (out_ts, out_tools, out_kind, out_sid, out_rid) in last_outputs.items():
        if not out_tools:
            continue
        is_child, label, path = _child_info(out_sid, root_session)
        gap_tools = list(out_tools)
        has_spawn, has_fork, is_pure = classify_tools(gap_tools)
        gaps.append(
            ToolGap(
                session_id=out_sid,
                after_output_ts=out_ts,
                next_request_ts=out_ts,
                duration_sec=0.0,
                tools_triggered=summarize_tools(out_tools),
                detail_tools=gap_tools,
                from_kind=out_kind,
                is_child_session=is_child,
                child_label=label,
                child_path=path,
                is_post_final=_is_post_final(out_sid, out_rid, out_ts),
                has_spawn_subagent=has_spawn,
                has_fork_agent=has_fork,
                is_pure_spawn_fork=is_pure,
            )
        )

    gaps.sort(key=lambda g: (g.after_output_ts, g.next_request_ts))
    return rounds, gaps


def has_only_spawn_or_fork(tools: list[dict[str, Any]]) -> bool:
    """Check if tools only contain spawn_subagent or fork_agent."""
    tool_names = {str(t.get("name", "")).lower() for t in tools}
    non_agent_tools = tool_names - {"spawn_subagent", "fork_agent"}
    return len(non_agent_tools) == 0 and len(tool_names) > 0


def classify_tools(tools: list[dict[str, Any]]) -> tuple[bool, bool, bool]:
    """Return (has_spawn_subagent, has_fork_agent, is_pure_spawn_fork)."""
    has_spawn = False
    has_fork = False
    other_tools = False
    for t in tools:
        name = str(t.get("name", "")).lower()
        if name == "spawn_subagent":
            has_spawn = True
        elif name == "fork_agent":
            has_fork = True
        elif name:
            other_tools = True
    return has_spawn, has_fork, not other_tools and (has_spawn or has_fork)


def aggregate_totals(rounds: list[LLMRound], gaps: list[ToolGap]) -> dict[str, Any]:
    tin = sum((r.input_tokens or 0) for r in rounds)
    tout = sum((r.output_tokens or 0) for r in rounds)
    ttot = sum((r.total_tokens or 0) for r in rounds)
    llm_sec = interval_union_sec([(r.request_ts, r.output_ts) for r in rounds])
    
    # 排除spawn_subagent和fork_agent的工具调用时间
    # 只要gap中有spawn_subagent或fork_agent，就排除这个gap的时间
    tool_gaps_excluding_agents = [
        g for g in gaps 
        if not g.has_spawn_subagent and not g.has_fork_agent and not g.is_post_final
    ]
    tool_sec_excluding_agents = interval_union_sec(
        [(g.after_output_ts, g.next_request_ts) for g in tool_gaps_excluding_agents]
    )
    
    # 统计工具重试次数
    tool_retry_stats = analyze_tool_retries(gaps)
    
    return {
        "rounds": len(rounds),
        "llm_wall_sec": round(llm_sec, 3),
        "input_tokens_sum": tin or None,
        "output_tokens_sum": tout or None,
        "total_tokens_sum": ttot or None,
        "tool_sec_excluding_agents": round(tool_sec_excluding_agents, 3),
        "tool_retry_count": tool_retry_stats["retry_count"],
        "tool_retry_rounds": tool_retry_stats["retry_rounds"],
        "tool_retry_details": tool_retry_stats["details"],
    }


def analyze_tool_retries(gaps: list[ToolGap]) -> dict[str, Any]:
    """分析工具重试次数和轮次。"""
    retry_count = 0
    retry_rounds = 0
    details: list[dict[str, Any]] = []
    
    # 按session分组
    sessions: dict[str, list[ToolGap]] = {}
    for g in gaps:
        if g.is_post_final or g.is_pure_spawn_fork:
            continue
        sessions.setdefault(g.session_id, []).append(g)
    
    for session_id, session_gaps in sessions.items():
        # 按request_id分组（从工具调用中提取）
        request_tools: dict[str, list[tuple[datetime, list[dict[str, Any]]]]] = {}
        
        for g in session_gaps:
            for tool in g.detail_tools:
                tool_name = str(tool.get("name", ""))
                if not tool_name or tool_name.lower() in ["spawn_subagent", "fork_agent"]:
                    continue
                # 使用工具名+参数哈希作为key来检测重试
                args = tool.get("arguments", "")
                tool_key = f"{tool_name}_{str(args)[:200]}"
                request_tools.setdefault(tool_key, []).append((g.after_output_ts, tool))
        
        for tool_key, calls in request_tools.items():
            if len(calls) > 1:
                retry_count += len(calls) - 1
                retry_rounds += 1
                details.append({
                    "tool_key": tool_key,
                    "call_count": len(calls),
                    "first_call": calls[0][0].strftime("%H:%M:%S.%f")[:-3],
                    "last_call": calls[-1][0].strftime("%H:%M:%S.%f")[:-3],
                })
    
    return {
        "retry_count": retry_count,
        "retry_rounds": retry_rounds,
        "details": details,
    }
