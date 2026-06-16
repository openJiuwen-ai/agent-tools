"""Match and merge history tools with full.json model rounds."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from full_parse.loader import load_session_from_paths
from full_parse.trace_analysis import ToolGap
from history_parse.analysis import build_timeline_from_history
from history_parse.models import HistoryExtras, LLMRound as HistoryLLMRound, ToolExecution
from history_parse.parser import load_history
from history_parse.session import interval_union_sec

from fusion_parse.discovery import DiscoveredLogs, discover_session_logs
from fusion_parse.models import (
    FusedGap,
    FusedModelRound,
    FusedTool,
    FusionSessionData,
    ModelMatchStatus,
    ReconcileSummary,
    ToolReconcileStatus,
)

_BOUNDARY_TOL = timedelta(seconds=2.0)
_MIN_OVERLAP_SEC = 0.5


def _naive(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _norm_name(name: str) -> str:
    return (name or "").strip().lower()


def _overlap_sec(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    a0, a1, b0, b1 = _naive(a0), _naive(a1), _naive(b0), _naive(b1)
    start = max(a0, b0)
    end = min(a1, b1)
    if end <= start:
        return 0.0
    return (end - start).total_seconds()


def _load_history_events(paths: list[Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        events.extend(load_history(path))
    events.sort(key=lambda e: (float(e.get("timestamp") or 0), str(e.get("id") or "")))
    return events


def _history_round_span(r: HistoryLLMRound) -> tuple[datetime, datetime]:
    return r.request_ts, r.output_ts


def _full_round_span(r) -> tuple[datetime, datetime]:
    return r.request_ts, r.output_ts


def _match_history_model(
    full_round,
    history_rounds: list[HistoryLLMRound],
    used: set[int],
) -> tuple[HistoryLLMRound | None, ModelMatchStatus, float, list[str]]:
    f0, f1 = _full_round_span(full_round)
    best_idx = -1
    best_overlap = 0.0
    for i, hr in enumerate(history_rounds):
        if i in used:
            continue
        h0, h1 = _history_round_span(hr)
        ov = _overlap_sec(f0, f1, h0, h1)
        if ov > best_overlap:
            best_overlap = ov
            best_idx = i
    notes: list[str] = []
    if best_idx < 0 or best_overlap < _MIN_OVERLAP_SEC:
        notes.append("history 中无足够时间重叠的模型轮次")
        return None, "history_missing", best_overlap, notes

    hr = history_rounds[best_idx]
    used.add(best_idx)
    full_dur = full_round.duration_sec
    hist_dur = hr.duration_sec
    delta = round(full_dur - hist_dur, 3)
    notes.append(f"history 总耗时 {hist_dur:.3f}s（含 TTFT 定义）vs full {full_dur:.3f}s，Δ={delta:+.3f}s")
    if abs(delta) > 5.0:
        status: ModelMatchStatus = "weak_overlap"
        notes.append("耗时差异较大，报告以 full 墙钟为准")
    else:
        status = "aligned"
    return hr, status, best_overlap, notes


def _tool_in_gap(tool: ToolExecution, gap: ToolGap) -> bool:
    g0 = _naive(gap.after_output_ts) - _BOUNDARY_TOL
    g1 = _naive(gap.next_request_ts) + _BOUNDARY_TOL
    return _naive(tool.start_ts) >= g0 and _naive(tool.end_ts) <= g1


def _find_gap_for_tool(tool: ToolExecution, gaps: list[ToolGap]) -> ToolGap | None:
    best: ToolGap | None = None
    best_ov = 0.0
    for gap in gaps:
        if not _tool_in_gap(tool, gap):
            continue
        ov = _overlap_sec(
            tool.start_ts,
            tool.end_ts,
            gap.after_output_ts,
            gap.next_request_ts,
        )
        if ov > best_ov:
            best_ov = ov
            best = gap
    return best


def _declared_tool_names(gap: ToolGap) -> list[str]:
    names: list[str] = []
    for t in gap.detail_tools:
        n = str(t.get("name") or "").strip()
        if n:
            names.append(n)
    if not names and gap.tools_triggered and gap.tools_triggered != "(no tool_calls)":
        for part in gap.tools_triggered.split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names


def build_fusion_session(
    session_id: str,
    log_dir: Path,
    *,
    history_files: list[Path] | None = None,
    full_files: list[Path] | None = None,
) -> FusionSessionData:
    if history_files is not None and full_files is not None:
        discovered = DiscoveredLogs(log_dir=log_dir.resolve(), history_files=history_files, full_files=full_files)
    else:
        discovered = discover_session_logs(log_dir)
    events = _load_history_events(discovered.history_files)
    hist_rounds, hist_tools, extras = build_timeline_from_history(
        events,
        session_id,
        full_log_paths=discovered.full_files,
    )
    full_data = load_session_from_paths(session_id, discovered.full_files)

    used_hist_models: set[int] = set()
    fused_models: list[FusedModelRound] = []
    for fr in sorted(full_data.rounds, key=lambda r: r.request_ts):
        hr, status, ov, notes = _match_history_model(fr, hist_rounds, used_hist_models)
        delta = None
        if hr is not None:
            delta = round(fr.duration_sec - hr.duration_sec, 3)
        fused_models.append(
            FusedModelRound(
                full=fr,
                history=hr,
                match_status=status,
                overlap_sec=round(ov, 3),
                duration_delta_sec=delta,
                notes=notes,
            )
        )

    gaps_sorted = sorted(full_data.gaps, key=lambda g: g.after_output_ts)
    gap_by_id = {id(g): g for g in gaps_sorted}
    tools_by_gap: dict[int, list[FusedTool]] = {id(g): [] for g in gaps_sorted}
    unmatched_tools: list[FusedTool] = []

    matched_hist_ids: set[str] = set()
    for tool in sorted(hist_tools, key=lambda t: t.start_ts):
        gap = _find_gap_for_tool(tool, gaps_sorted)
        declared = _declared_tool_names(gap) if gap else []
        declared_norm = {_norm_name(n) for n in declared}
        tnorm = _norm_name(tool.name)
        status: ToolReconcileStatus
        notes: list[str] = []
        declared_in_full = tnorm in declared_norm if gap else False

        if gap is None:
            status = "history_only"
            notes.append("未落入任一 full 工具窗口，仍以 history 时间为准")
            unmatched_tools.append(
                FusedTool(
                    history=tool,
                    gap=None,
                    declared_in_full=False,
                    status=status,
                    notes=notes,
                )
            )
            continue

        if declared_in_full:
            status = "matched"
            matched_hist_ids.add(tool.tool_call_id)
        elif declared_norm:
            status = "name_mismatch"
            notes.append(f"窗口声明工具 {declared}，history 为 {tool.name!r}")
        else:
            status = "history_only"
            notes.append("full 窗口未声明 tool_calls，但 history 有执行记录")

        ft = FusedTool(
            history=tool,
            gap=gap,
            declared_in_full=declared_in_full,
            status=status,
            notes=notes,
        )
        tools_by_gap[id(gap)].append(ft)

    # full_only: declared in gap but no history tool matched
    fused_tools: list[FusedTool] = list(unmatched_tools)
    fused_gaps: list[FusedGap] = []
    tools_full_only = 0
    for gap in gaps_sorted:
        declared = _declared_tool_names(gap)
        attached = tools_by_gap.get(id(gap), [])
        hist_names = [t.history.name for t in attached]
        declared_norm = {_norm_name(n) for n in declared}
        matched_norm = {_norm_name(t.history.name) for t in attached if t.status == "matched"}
        for dn in declared:
            if _norm_name(dn) not in matched_norm:
                tools_full_only += 1
                pseudo = ToolExecution(
                    session_id=gap.session_id,
                    request_id="—",
                    tool_call_id=f"full-only-{dn}-{gap.after_output_ts.timestamp()}",
                    name=dn,
                    arguments=None,
                    start_ts=gap.after_output_ts,
                    end_ts=gap.next_request_ts,
                    duration_sec=gap.duration_sec,
                    result="(仅在 full output tool_calls 中声明，history 无 call/result)",
                )
                fused_tools.append(
                    FusedTool(
                        history=pseudo,
                        gap=gap,
                        declared_in_full=True,
                        status="full_only",
                        notes=["history 无对应 tool_call/tool_result，不可计入墙钟"],
                    )
                )
        fused_gaps.append(
            FusedGap(
                gap=gap,
                tools=attached,
                declared_names=declared,
                history_names=hist_names,
            )
        )
        fused_tools.extend(attached)

    fused_tools.sort(key=lambda t: _naive(t.history.start_ts))

    all_ts: list[datetime] = []
    for fm in fused_models:
        all_ts.extend([fm.full.request_ts, fm.full.output_ts])
    for ft in fused_tools:
        if ft.status != "full_only":
            all_ts.extend([ft.history.start_ts, ft.history.end_ts])
    if all_ts:
        all_naive = [_naive(t) for t in all_ts]
        task_sec = round((max(all_naive) - min(all_naive)).total_seconds(), 3)
    else:
        task_sec = 0.0

    from history_parse.session import is_measurable_tool

    real_tools = [t for t in fused_tools if t.status != "full_only"]
    measurable_tools = [
        t for t in real_tools if is_measurable_tool(t.history.name)
    ]
    summary = ReconcileSummary(
        model_rounds_full=len(fused_models),
        model_rounds_history_matched=sum(1 for m in fused_models if m.history),
        model_weak_overlap=sum(1 for m in fused_models if m.match_status == "weak_overlap"),
        tools_history=len(hist_tools),
        tools_matched=sum(1 for t in fused_tools if t.status == "matched"),
        tools_history_only=sum(1 for t in fused_tools if t.status == "history_only"),
        tools_full_only=tools_full_only,
        tools_name_mismatch=sum(1 for t in fused_tools if t.status == "name_mismatch"),
        gaps_full=len(gaps_sorted),
        llm_wall_sec=round(
            interval_union_sec([(m.full.request_ts, m.full.output_ts) for m in fused_models]),
            3,
        ),
        tool_wall_sec=round(
            interval_union_sec(
                [(t.history.start_ts, t.history.end_ts) for t in measurable_tools]
            ),
            3,
        ),
        task_sec=task_sec,
        input_tokens_sum=sum(m.full.input_tokens or 0 for m in fused_models),
        output_tokens_sum=sum(m.full.output_tokens or 0 for m in fused_models),
        total_tokens_sum=sum(m.full.total_tokens or 0 for m in fused_models),
    )
    if summary.tools_history_only:
        summary.issues.append(f"{summary.tools_history_only} 次工具仅在 history 有记录（已计入墙钟）")
    if summary.tools_full_only:
        summary.issues.append(f"{summary.tools_full_only} 个工具仅在 full 声明（未计入墙钟）")
    if summary.model_weak_overlap:
        summary.issues.append(f"{summary.model_weak_overlap} 轮模型与 history 耗时偏差较大")

    return FusionSessionData(
        root_session=session_id,
        log_dir=str(discovered.log_dir),
        history_label=discovered.history_label,
        full_file_count=len(discovered.full_files),
        model_rounds=fused_models,
        tools=fused_tools,
        gaps=fused_gaps,
        extras=extras,
        summary=summary,
        full_source_files=full_data.source_files,
    )
