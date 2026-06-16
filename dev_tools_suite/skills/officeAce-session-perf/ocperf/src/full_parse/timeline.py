"""Merge full.json LLM rounds and tool execution windows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from full_parse.trace_analysis import LLMRound, ToolGap

FullItem = tuple[Literal["llm", "tool_window"], LLMRound | ToolGap]


def merge_full_timeline(rounds: list[LLMRound], gaps: list[ToolGap]) -> list[FullItem]:
    items: list[FullItem] = [("llm", r) for r in rounds]
    items.extend(("tool_window", g) for g in gaps)

    def _key(item: FullItem) -> tuple[datetime, datetime, int]:
        kind, obj = item
        if kind == "llm":
            return (obj.request_ts, obj.output_ts, 0)
        return (obj.after_output_ts, obj.next_request_ts, 1)

    items.sort(key=_key)
    return items
