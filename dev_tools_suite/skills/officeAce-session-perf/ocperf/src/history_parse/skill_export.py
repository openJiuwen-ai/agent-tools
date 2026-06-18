"""Export structured session data for Agent Skill / LLM analysis (no HTML)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ocperf.time_util import local_now

from history_parse.llm_latency_metrics import llm_round_metrics
from history_parse.models import HistoryExtras, LLMRound, ToolExecution
from history_parse.pattern_analysis import (
    SessionPatternSummary,
    analyze_session_patterns,
    event_ref_from_item,
)
from history_parse.report_ui import (
    TimelineKind,
    build_execution_phases,
    partition_by_phase,
)
from history_parse.session import is_measurable_tool
from history_parse.timeline import merge_timeline

SourceKind = Literal["history", "full", "fusion"]


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _safe_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)[:48]


def is_tool_failure(tool: ToolExecution) -> bool:
    r = tool.result or ""
    if "success=False" in r:
        return True
    if "success=True" in r:
        return False
    low = r.lower()
    if any(k in low for k in ("error", "exception", "failed", "failure", "traceback")):
        return True
    return False


def _tool_failure_snippet(tool: ToolExecution, max_len: int = 200) -> str:
    r = (tool.result or "").strip()
    if not r:
        return ""
    return r[:max_len] + ("…" if len(r) > max_len else "")


@dataclass
class PhaseToolFailure:
    tool_name: str
    tool_call_id: str
    duration_sec: float
    start: str
    end: str
    snippet: str
    session_label: str = ""


@dataclass
class PhaseExport:
    phase_id: str
    title: str
    subtitle: str
    start: str
    end: str
    llm_count: int
    tool_count: int
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    cycles: list[dict[str, Any]] = field(default_factory=list)
    tool_failures: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SkillBundle:
    session_id: str
    source: SourceKind
    generated_at: str
    history_path: str | None = None
    full_paths: list[str] = field(default_factory=list)
    log_dir: str | None = None
    report_paths: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    fusion_reconcile: dict[str, Any] = field(default_factory=dict)
    tool_stats: dict[str, Any] = field(default_factory=dict)
    phases: list[dict[str, Any]] = field(default_factory=list)
    pattern_global: list[dict[str, Any]] = field(default_factory=list)
    mermaid_flowchart: str = ""
    analysis_prompt: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _event_dict(idx: int, kind: TimelineKind, obj: Any) -> dict[str, Any]:
    ref = event_ref_from_item(idx, kind, obj)
    out: dict[str, Any] = {
        "idx": ref.idx,
        "kind": ref.kind,
        "name": ref.name,
        "start": _iso(ref.start),
        "end": _iso(ref.end),
        "duration_sec": ref.duration_sec,
        "session_label": ref.session_label,
        "is_child": ref.is_child,
        "exclude_from_tool_kpi": ref.exclude_from_tool_time,
        "detail": ref.detail,
        "tool_call_id": getattr(obj, "tool_call_id", "") if kind == "tool" else "",
    }
    if kind == "llm" and isinstance(obj, LLMRound):
        out.update(llm_round_metrics(obj))
    return out


def _phase_exports(
    phases: list,
    buckets: dict[str, list],
    pattern: SessionPatternSummary,
    merged_tools: list[ToolExecution],
) -> list[PhaseExport]:
    pat_by_id = {p.phase_id: p for p in pattern.phases}
    tool_by_idx = {i + 1: t for i, t in enumerate(merged_tools)}

    out: list[PhaseExport] = []
    for phase in phases:
        items = buckets.get(phase.phase_id, [])
        if not items:
            continue
        pr = pat_by_id.get(phase.phase_id)
        failures: list[PhaseToolFailure] = []
        events: list[dict[str, Any]] = []
        for idx, kind, obj in items:
            events.append(_event_dict(idx, kind, obj))
            if kind == "tool" and is_measurable_tool(getattr(obj, "name", None)):
                if is_tool_failure(obj):
                    is_child = bool(obj.is_child_session)
                    failures.append(
                        PhaseToolFailure(
                            tool_name=obj.name,
                            tool_call_id=obj.tool_call_id,
                            duration_sec=obj.duration_sec,
                            start=_iso(obj.start_ts),
                            end=_iso(obj.end_ts),
                            snippet=_tool_failure_snippet(obj),
                            session_label=obj.child_label if is_child else "根会话",
                        )
                    )

        dupes = []
        cycles = []
        if pr:
            dupes = [
                {
                    "name": d.name,
                    "kind": d.kind,
                    "count": d.count,
                    "relationship": d.relationship,
                    "note": d.note,
                }
                for d in pr.duplicates
            ]
            cycles = [
                {
                    "pattern": c.pattern,
                    "repeats": c.repeats,
                    "note": c.note,
                    "example_sequence": c.example_sequence,
                }
                for c in pr.cycles
            ]

        out.append(
            PhaseExport(
                phase_id=phase.phase_id,
                title=phase.title,
                subtitle=phase.subtitle,
                start=_iso(phase.start),
                end=_iso(phase.end),
                llm_count=sum(1 for _, k, _ in items if k == "llm"),
                tool_count=sum(
                    1 for _, k, tool_obj in items if k == "tool" and is_measurable_tool(tool_obj.name)
                ),
                duplicates=dupes,
                cycles=cycles,
                tool_failures=[asdict(f) for f in failures],
                events=events,
            )
        )
    return out


def build_mermaid_flowchart(phases_export: list[PhaseExport]) -> str:
    """Skeleton flowchart TD for LLM refinement."""
    if not phases_export:
        return "flowchart TD\n  empty[无阶段数据]\n"

    lines = ["flowchart TD", "  classDef phase fill:#e0f2fe,stroke:#0369a1"]
    lines.append("  classDef llm fill:#bbf7d0,stroke:#15803d")
    lines.append("  classDef tool fill:#fed7aa,stroke:#c2410c")
    lines.append("  classDef fail fill:#fecaca,stroke:#dc2626")
    lines.append("  classDef agent fill:#e9d5ff,stroke:#7c3aed,stroke-dasharray:5 5")

    prev_phase: str | None = None
    for pe in phases_export:
        pid = _safe_id(pe.phase_id)
        lines.append(f'  subgraph phase_{pid}["{pe.title[:60]}"]')
        lines.append(f'    P_{pid}["阶段: {pe.title[:40]}<br/>LLM×{pe.llm_count} 工具×{pe.tool_count}"]')
        lines.append(f"    class P_{pid} phase")

        fail_ids = {f["tool_call_id"] for f in pe.tool_failures}
        last_node: str | None = f"P_{pid}"
        for ev in pe.events:
            eid = f"n_{pid}_{ev['idx']}"
            label = ev["name"].replace('"', "'")[:24]
            dur = ev["duration_sec"]
            if ev["kind"] == "llm":
                lines.append(f'    {eid}["🤖 {label}<br/>{dur:.1f}s"]')
                lines.append(f"    class {eid} llm")
            elif ev.get("exclude_from_tool_kpi"):
                lines.append(f'    {eid}["⚙ {label} spawn"]')
                lines.append(f"    class {eid} agent")
            else:
                tcid = ev.get("tool_call_id") or ""
                cls = "fail" if tcid in fail_ids else "tool"
                lines.append(f'    {eid}["🔧 {label}<br/>{dur:.1f}s"]')
                lines.append(f"    class {eid} {cls}")

            if last_node:
                lines.append(f"    {last_node} --> {eid}")
            last_node = eid

        if pe.cycles:
            for i, cy in enumerate(pe.cycles[:2]):
                cid = f"cycle_{pid}_{i}"
                pat = "→".join(cy.get("pattern", [])[:3])
                lines.append(f'    {cid}[["循环: {pat}×{cy.get("repeats", 2)}"]]')
                if last_node:
                    lines.append(f"    {last_node} -.-> {cid}")

        lines.append("  end")
        if prev_phase:
            lines.append(f"  {prev_phase} --> P_{pid}")
        prev_phase = f"P_{pid}"

    return "\n".join(lines) + "\n"


ANALYSIS_PROMPT_TEMPLATE = """\
请基于 skill_bundle.json 完成以下分析（勿编造未出现在 JSON 中的工具或耗时）：

1. **分阶段综述**：按 phases[] 顺序，说明每阶段任务目标（title/subtitle）、LLM 与工具数量、主要耗时。
2. **工具重复与循环**：引用 duplicates[]、cycles[]，区分并发(concurrent)/串行(serial)/混合(mixed)，判断是否可能存在无效重试或死循环。
3. **工具失败**：列出各阶段 tool_failures[]；若无失败，明确说明；结合 result snippet 给出可能原因。
4. **交叉校对（fusion_reconcile）**：若存在 fusion_reconcile，逐项解释 matched / history_only / full_only / name_mismatch / weak_overlap；说明哪些工具耗时计入墙钟、哪些仅为 full 声明。
5. **流程图**：在 mermaid_flowchart 骨架上增删节点，标注阶段任务、模型调用、工具调用；箭头表示时间顺序；循环用虚线或注释标出。
6. **建议**：可操作的优化项（减少重复工具、修复失败工具、合并并发、缩短 TTFT 等）。

数据可信度：spawn_subagent/fork_agent 不计入工具墙钟（exclude_from_tool_kpi=true），但可在流程中展示。
模型墙钟以 full 为准；工具墙钟以 history 为准；总任务时间取融合时间跨度。
"""


def export_fusion_reconcile(data) -> dict[str, Any]:
    """Serialize fusion ReconcileSummary + tool/model mismatch samples for LLM."""
    s = data.summary
    out: dict[str, Any] = {
        "model_rounds_full": s.model_rounds_full,
        "model_rounds_history_matched": s.model_rounds_history_matched,
        "model_weak_overlap": s.model_weak_overlap,
        "tools_history": s.tools_history,
        "tools_matched": s.tools_matched,
        "tools_history_only": s.tools_history_only,
        "tools_full_only": s.tools_full_only,
        "tools_name_mismatch": s.tools_name_mismatch,
        "llm_wall_sec": s.llm_wall_sec,
        "tool_wall_sec": s.tool_wall_sec,
        "task_sec": s.task_sec,
        "input_tokens_sum": s.input_tokens_sum,
        "output_tokens_sum": s.output_tokens_sum,
        "total_tokens_sum": s.total_tokens_sum,
        "issues": list(s.issues),
        "credibility": (
            "high"
            if s.tools_history and s.tools_matched >= s.tools_history * 0.7
            else "low"
            if s.tools_history and s.tools_matched < s.tools_history * 0.5
            else "medium"
        ),
    }
    if s.tools_history:
        out["tool_match_rate"] = round(s.tools_matched / s.tools_history, 3)

    samples: dict[str, list[dict[str, Any]]] = {
        "history_only": [],
        "full_only": [],
        "name_mismatch": [],
        "weak_overlap_models": [],
    }
    for t in data.tools:
        if t.status == "history_only" and len(samples["history_only"]) < 8:
            samples["history_only"].append(
                {"tool": t.history.name, "duration_sec": t.history.duration_sec, "status": t.status}
            )
        elif t.status == "full_only" and len(samples["full_only"]) < 8:
            samples["full_only"].append({"tool": t.history.name, "status": t.status, "notes": t.notes})
        elif t.status == "name_mismatch" and len(samples["name_mismatch"]) < 8:
            declared = t.gap.tools_triggered if t.gap else ""
            samples["name_mismatch"].append(
                {
                    "history": t.history.name,
                    "declared_in_full": declared,
                    "duration_sec": t.history.duration_sec,
                }
            )
    for m in data.model_rounds:
        if m.match_status == "weak_overlap" and len(samples["weak_overlap_models"]) < 5:
            samples["weak_overlap_models"].append(
                {
                    "full_sec": m.full.duration_sec,
                    "history_sec": m.history.duration_sec if m.history else None,
                    "delta_sec": m.duration_delta_sec,
                    "notes": m.notes,
                }
            )
    out["samples"] = samples
    return out


def build_history_bundle(
    session_id: str,
    rounds: list[LLMRound],
    tools: list[ToolExecution],
    extras: HistoryExtras,
    *,
    history_path: str | None = None,
    full_paths: list[str] | None = None,
    report_paths: dict[str, str] | None = None,
    aggregate: dict[str, Any] | None = None,
    fusion_reconcile: dict[str, Any] | None = None,
    source: SourceKind = "history",
) -> SkillBundle:
    merged = merge_timeline(rounds, tools)
    all_ts = []
    for kind, obj in merged:
        if kind == "llm":
            all_ts.extend([obj.request_ts, obj.output_ts])
        else:
            all_ts.extend([obj.start_ts, obj.end_ts])
    t0 = min(all_ts) if all_ts else local_now()
    t1 = max(all_ts) if all_ts else local_now()

    phases = build_execution_phases(extras.todo_timeline, t0, t1)
    buckets = partition_by_phase(merged, phases)
    pattern = analyze_session_patterns(phases, buckets)
    phase_list = _phase_exports(phases, buckets, pattern, tools)

    global_dups = [
        {
            "name": d.name,
            "kind": d.kind,
            "count": d.count,
            "relationship": d.relationship,
            "note": d.note,
        }
        for d in pattern.global_duplicates
    ]

    mermaid = build_mermaid_flowchart(phase_list)

    summary = aggregate or {}
    return SkillBundle(
        session_id=session_id,
        source=source,
        generated_at=_iso(local_now()),
        history_path=history_path,
        full_paths=full_paths or [],
        report_paths=report_paths or {},
        summary=summary,
        fusion_reconcile=fusion_reconcile or {},
        tool_stats=dict(extras.tool_stats or {}),
        phases=[asdict(p) for p in phase_list],
        pattern_global=global_dups,
        mermaid_flowchart=mermaid,
        analysis_prompt=ANALYSIS_PROMPT_TEMPLATE,
    )


def write_skill_bundle(path: Path, bundle: SkillBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.to_json(), encoding="utf-8")
