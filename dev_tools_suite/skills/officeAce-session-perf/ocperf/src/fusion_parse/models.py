"""Fused timeline models (history tools + full model rounds)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from full_parse.trace_analysis import LLMRound as FullLLMRound
from full_parse.trace_analysis import ToolGap
from history_parse.models import HistoryExtras, LLMRound as HistoryLLMRound, ToolExecution

ToolReconcileStatus = Literal[
    "matched",
    "history_only",
    "full_only",
    "name_mismatch",
]

ModelMatchStatus = Literal[
    "aligned",
    "history_missing",
    "weak_overlap",
]


@dataclass
class FusedModelRound:
    """Authoritative model timing from full.json."""

    full: FullLLMRound
    history: HistoryLLMRound | None = None
    match_status: ModelMatchStatus = "history_missing"
    overlap_sec: float = 0.0
    duration_delta_sec: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class FusedTool:
    """Authoritative tool timing from history.json."""

    history: ToolExecution
    gap: ToolGap | None = None
    declared_in_full: bool = False
    status: ToolReconcileStatus = "history_only"
    notes: list[str] = field(default_factory=list)


@dataclass
class FusedGap:
    gap: ToolGap
    tools: list[FusedTool]
    declared_names: list[str]
    history_names: list[str]


@dataclass
class ReconcileSummary:
    model_rounds_full: int = 0
    model_rounds_history_matched: int = 0
    model_weak_overlap: int = 0
    tools_history: int = 0
    tools_matched: int = 0
    tools_history_only: int = 0
    tools_full_only: int = 0
    tools_name_mismatch: int = 0
    gaps_full: int = 0
    llm_wall_sec: float = 0.0
    tool_wall_sec: float = 0.0
    task_sec: float = 0.0
    input_tokens_sum: int = 0
    output_tokens_sum: int = 0
    total_tokens_sum: int = 0
    issues: list[str] = field(default_factory=list)


@dataclass
class FusionSessionData:
    root_session: str
    log_dir: str
    history_label: str
    full_file_count: int
    model_rounds: list[FusedModelRound]
    tools: list[FusedTool]
    gaps: list[FusedGap]
    extras: HistoryExtras
    summary: ReconcileSummary
    full_source_files: list[Any] = field(default_factory=list)
