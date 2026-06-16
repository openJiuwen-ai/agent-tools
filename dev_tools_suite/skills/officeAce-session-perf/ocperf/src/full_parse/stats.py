"""Aggregate stats for full.json timeline."""

from __future__ import annotations

from typing import Any

from full_parse.trace_analysis import LLMRound, ToolGap
from history_parse.session import interval_union_sec


def aggregate_full_stats(rounds: list[LLMRound], gaps: list[ToolGap]) -> dict[str, Any]:
    all_ts = []
    for r in rounds:
        all_ts.extend([r.request_ts, r.output_ts])
    for g in gaps:
        all_ts.extend([g.after_output_ts, g.next_request_ts])
    task_sec = round((max(all_ts) - min(all_ts)).total_seconds(), 3) if all_ts else 0.0

    llm_sec = interval_union_sec([(r.request_ts, r.output_ts) for r in rounds])
    gap_sec = interval_union_sec([(g.after_output_ts, g.next_request_ts) for g in gaps])
    agent_gaps = [g for g in gaps if not g.has_spawn_subagent and not g.has_fork_agent]
    spawn_gaps = [g for g in gaps if g.has_spawn_subagent or g.has_fork_agent]

    stream_n = sum(1 for r in rounds if r.kind == "stream")
    invoke_n = sum(1 for r in rounds if r.kind == "invoke")
    gaps_with_tools = sum(1 for g in gaps if g.detail_tools)

    return {
        "task_sec": task_sec,
        "rounds": len(rounds),
        "stream_rounds": stream_n,
        "invoke_rounds": invoke_n,
        "llm_wall_sec": round(llm_sec, 3),
        "tool_windows": len(gaps),
        "tool_windows_with_calls": gaps_with_tools,
        "tool_wall_sec": round(gap_sec, 3),
        "tool_wall_sec_no_agent": round(
            interval_union_sec([(g.after_output_ts, g.next_request_ts) for g in agent_gaps]), 3
        ),
        "agent_window_count": len(spawn_gaps),
        "agent_wall_sec": round(
            interval_union_sec([(g.after_output_ts, g.next_request_ts) for g in spawn_gaps]), 3
        ),
        "input_tokens_sum": sum(r.input_tokens or 0 for r in rounds),
        "output_tokens_sum": sum(r.output_tokens or 0 for r in rounds),
        "total_tokens_sum": sum(r.total_tokens or 0 for r in rounds),
        "cache_tokens_sum": sum(r.cache_tokens or 0 for r in rounds),
    }
