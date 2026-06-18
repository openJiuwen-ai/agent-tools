"""Timeline data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

SessionPath = tuple[tuple[str, str, str], ...]


@dataclass
class LLMRound:
    session_id: str
    request_id: str
    model_name: str
    """思考/等待起点（上次工具结束，或上轮模型结束，或用户消息）。"""
    request_ts: datetime
    """首条 reasoning/delta，即首 token 时刻。"""
    first_token_ts: datetime
    output_ts: datetime
    """首 token 前等待（TTFT）。"""
    ttft_sec: float
    """首 token 到 usage_metadata 的流式输出耗时。"""
    inference_sec: float
    """ttft_sec + inference_sec。"""
    duration_sec: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_tokens: int | None = None
    reasoning_full: str = ""
    assistant_text: str = ""
    is_child_session: bool = False
    child_label: str = ""
    child_path: SessionPath = ()


@dataclass
class ToolExecution:
    """一次真实工具执行：由 chat.tool_call 到 chat.tool_result。"""

    session_id: str
    request_id: str
    tool_call_id: str
    name: str
    arguments: Any
    start_ts: datetime
    end_ts: datetime
    duration_sec: float
    result: str = ""
    is_child_session: bool = False
    child_label: str = ""
    child_path: SessionPath = ()
    is_orphan_backfill: bool = False


@dataclass
class UserTurn:
    request_id: str
    timestamp: datetime
    content: str
    session_id: str


@dataclass
class ContextEvent:
    kind: Literal["compress", "reload"]
    timestamp: datetime
    tool_name: str
    summary: str
    detail: str = ""


@dataclass
class HistoryExtras:
    user_turns: list[UserTurn] = field(default_factory=list)
    context_events: list[ContextEvent] = field(default_factory=list)
    tool_stats: dict[str, Any] = field(default_factory=dict)
    todo_timeline: Any = None  # TodoTimeline | None, avoid circular import
    orphan_spawn_calls: list[dict[str, Any]] = field(default_factory=list)
    spawn_orphan_backfilled: int = 0
