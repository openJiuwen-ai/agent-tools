"""Merge model rounds / tool executions and aggregate stats."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from history_parse.models import LLMRound, ToolExecution
from history_parse.llm_latency_metrics import aggregate_llm_latency
from history_parse.session import interval_union_sec, is_measurable_tool

TimelineItem = tuple[Literal["llm", "tool"], LLMRound | ToolExecution]


def merge_timeline(rounds: list[LLMRound], tools: list[ToolExecution]) -> list[TimelineItem]:
    items: list[TimelineItem] = [("llm", r) for r in rounds]
    items.extend(("tool", t) for t in tools)

    def _sort_key(item: TimelineItem) -> tuple[datetime, datetime, int]:
        kind, obj = item
        if kind == "llm":
            return (obj.request_ts, obj.output_ts, 0)
        return (obj.start_ts, obj.end_ts, 1)

    items.sort(key=_sort_key)
    return items


def item_start(kind: Literal["llm", "tool"], obj: Any) -> datetime:
    return obj.request_ts if kind == "llm" else obj.start_ts


def item_end(kind: Literal["llm", "tool"], obj: Any) -> datetime:
    return obj.output_ts if kind == "llm" else obj.end_ts


def aggregate_stats(rounds: list[LLMRound], tools: list[ToolExecution]) -> dict[str, Any]:
    all_ts: list[datetime] = []
    for r in rounds:
        all_ts.extend([r.request_ts, r.output_ts])
    for t in tools:
        all_ts.extend([t.start_ts, t.end_ts])
    task_sec = round((max(all_ts) - min(all_ts)).total_seconds(), 3) if all_ts else 0.0
    llm_sec = interval_union_sec([(r.request_ts, r.output_ts) for r in rounds])
    measurable = [t for t in tools if is_measurable_tool(t.name)]
    spawn_tools = [t for t in tools if not is_measurable_tool(t.name)]
    tool_sec = interval_union_sec([(t.start_ts, t.end_ts) for t in measurable])
    lat = aggregate_llm_latency(rounds)
    ttft_sum = lat["llm_ttft_sum_sec"]
    inference_sum = lat["llm_inference_sum_sec"]
    return {
        "rounds": len(rounds),
        "llm_wall_sec": round(llm_sec, 3),
        "llm_ttft_sum_sec": ttft_sum,
        "llm_inference_sum_sec": inference_sum,
        "cache_tokens_sum": lat["cache_tokens_sum"],
        "avg_tpot_sec": lat["avg_tpot_sec"],
        "avg_tokens_per_sec": lat["avg_tokens_per_sec"],
        "tool_sec": round(tool_sec, 3),
        "tool_sec_excluding_agents": round(tool_sec, 3),
        "tool_sec_all": round(
            interval_union_sec([(t.start_ts, t.end_ts) for t in tools]), 3
        ),
        "tool_sec_agent_spawn": round(
            interval_union_sec([(t.start_ts, t.end_ts) for t in spawn_tools]), 3
        ),
        "tool_calls": len(measurable),
        "tool_calls_all": len(tools),
        "spawn_tool_calls": len(spawn_tools),
        "input_tokens_sum": sum(r.input_tokens or 0 for r in rounds),
        "output_tokens_sum": sum(r.output_tokens or 0 for r in rounds),
        "total_tokens_sum": sum(r.total_tokens or 0 for r in rounds),
        "task_sec": task_sec,
    }
