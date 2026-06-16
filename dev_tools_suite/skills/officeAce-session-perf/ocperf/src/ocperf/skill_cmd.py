"""`ocperf skill` — generate HTML reports + skill_bundle.json for Agent Skill."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fusion_parse import build_fusion_session, render_fusion_html, write_fusion_report
from full_parse.full_report import render_full_html, write_full_report
from full_parse.loader import load_session_from_paths, resolve_full_log_paths
from full_parse.stats import aggregate_full_stats
from history_parse import (
    build_timeline_from_history,
    load_history,
    pick_root_session,
    render_history_html,
    write_history_report,
)
from history_parse.flowchart_export import augment_bundle_file, write_flowchart_artifacts
from history_parse.llm_latency_report import render_llm_latency_html, write_llm_latency_report
from history_parse.unified_report import render_unified_html, write_unified_report
from history_parse.skill_export import SourceKind, build_history_bundle, export_fusion_reconcile, write_skill_bundle
from history_parse.timeline import aggregate_stats
from ocperf.errors import OcperfError
from ocperf.officeclaw import resolve_officeclaw_session
from parse_rules_snippets import resolve_guide_href

logger = logging.getLogger(__name__)


def _safe_name(session_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)


def _require_html_reports(
    report_paths: dict[str, str],
    session_id: str,
    *,
    require_history: bool = False,
    require_full: bool = False,
) -> None:
    missing: list[str] = []
    if require_history:
        hist = report_paths.get("history_html")
        if not hist or not Path(hist).is_file():
            missing.append("history_html")
    if require_full:
        full = report_paths.get("full_html")
        if not full or not Path(full).is_file():
            missing.append("full_html")
    fus = report_paths.get("fusion_html")
    if fus and not Path(fus).is_file():
        missing.append("fusion_html")
    if missing:
        raise OcperfError(
            f"缺少必需 HTML 报告: {', '.join(missing)}（session {session_id}）。"
            "请勿使用 --no-extended。"
        )


def _emit_flow_artifacts(bundle_path: Path, bundle, session_id: str, out_dir: Path) -> None:
    import json

    data = json.loads(bundle.to_json()) if hasattr(bundle, "to_json") else {}
    if not data:
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
    safe = _safe_name(session_id)
    paths = write_flowchart_artifacts(out_dir, data, safe)
    augment_bundle_file(bundle_path, paths)
    logger.info(f"已生成端到端流程图: {paths.get('e2e_flowchart_html')}")


def _find_history_in_dir(d: Path) -> Path | None:
    for name in ("history.json",):
        p = d / name
        if p.is_file():
            return p
    cands = sorted(d.glob("history*.json"))
    return cands[0] if cands else None


def _find_full_in_dir(d: Path) -> list[Path]:
    return resolve_full_log_paths([d])


def discover_inputs(path: Path) -> tuple[Path | None, list[Path], Path]:
    path = path.resolve()
    if path.is_file():
        low = path.name.lower()
        if "history" in low:
            return path, [], path.parent
        if "full" in low or path.suffix.lower() in (".txt", ".log"):
            return None, [path], path.parent
        raise OcperfError(f"无法识别日志类型: {path.name}（文件名需含 history 或 full）")
    if path.is_dir():
        return _find_history_in_dir(path), _find_full_in_dir(path), path
    raise OcperfError(f"路径不存在: {path}")


def _session_from_history(hist_path: Path) -> str | None:
    events = load_history(hist_path)
    return pick_root_session(events) if events else None


def _run_pipeline(
    *,
    session_id: str | None,
    hist_path: Path | None,
    full_files: list[Path],
    work_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    extended = not args.no_extended
    report_paths: dict[str, str] = {}
    fusion_data = None

    do_fusion = bool(getattr(args, "fusion", False)) and not args.no_fusion
    if hist_path and full_files and do_fusion:
        session_id = session_id or _session_from_history(hist_path)
        if not session_id:
            raise OcperfError("融合分析需要 session_id，请使用 -s")
        try:
            fusion_data = build_fusion_session(
                session_id,
                work_dir,
                history_files=[hist_path],
                full_files=full_files,
            )
        except (ValueError, FileNotFoundError) as e:
            logger.info(f"融合分析失败，将仅生成 history 报告: {e}")
            fusion_data = None
        if fusion_data is not None:
            fus_out = (args.fusion_out or out_dir / f"out_fusion_{_safe_name(session_id)}.html").resolve()
            guide = resolve_guide_href(fus_out, args.guide_link)
            write_fusion_report(
                fus_out,
                render_fusion_html(fusion_data, guide_href=guide, extended_analysis=extended),
            )
            report_paths["fusion_html"] = str(fus_out)
            logger.info(f"已生成融合报告: {fus_out}")

    if hist_path:
        events = load_history(hist_path)
        if not events:
            raise OcperfError("history 文件为空")
        session_id = session_id or pick_root_session(events)
        if not session_id:
            raise OcperfError("无法识别 session_id")

        if full_files:
            try:
                full_data = load_session_from_paths(
                    session_id, full_files, history_path=hist_path
                )
                full_out = (args.full_out or out_dir / f"out_full_{_safe_name(session_id)}.html").resolve()
                guide = resolve_guide_href(full_out, args.guide_link)
                write_full_report(
                    full_out,
                    render_full_html(full_data, guide_href=guide, extended_analysis=extended),
                )
                report_paths["full_html"] = str(full_out)
                logger.info(f"已生成 full 报告: {full_out}")
            except (ValueError, FileNotFoundError) as e:
                logger.info(f"full 报告生成失败（继续 history）: {e}")
                full_data = None
        else:
            full_data = None

        rounds, tools, extras = build_timeline_from_history(
            events, session_id, full_log_paths=full_files or None
        )
        hist_out = (args.history_out or out_dir / f"out_history_{_safe_name(session_id)}.html").resolve()
        guide = resolve_guide_href(hist_out, args.guide_link)
        write_history_report(
            hist_out,
            render_history_html(
                session_id,
                hist_path.name,
                rounds,
                tools,
                extras,
                guide_href=guide,
                extended_analysis=extended,
            ),
        )
        report_paths["history_html"] = str(hist_out)
        logger.info(f"已生成 history 报告: {hist_out}")  # out_history_*.html

        llm_lat_out = (out_dir / f"out_llm_latency_{_safe_name(session_id)}.html").resolve()
        write_llm_latency_report(
            llm_lat_out,
            render_llm_latency_html(
                session_id,
                rounds,
                source_label="history.json（推断时延）",
            ),
        )
        report_paths["llm_latency_html"] = str(llm_lat_out)
        logger.info(f"已生成 LLM 时延报告: {llm_lat_out}")

        if full_data is not None:
            unified_out = (out_dir / f"out_unified_{_safe_name(session_id)}.html").resolve()
            write_unified_report(
                unified_out,
                render_unified_html(
                    session_id,
                    rounds,
                    tools,
                    extras,
                    full_data=full_data,
                    report_paths=report_paths,
                    guide_href=resolve_guide_href(unified_out, args.guide_link),
                    extended_analysis=extended,
                ),
            )
            report_paths["unified_html"] = str(unified_out)
            logger.info(f"已生成融合总览报告: {unified_out}")
        else:
            logger.info("跳过融合总览（需要 full + history）")

        bundle_path = (args.bundle_out or out_dir / f"skill_bundle_{_safe_name(session_id)}.json").resolve()
        tot = aggregate_stats(rounds, tools)
        if full_files and not fusion_data and full_data is not None:
            try:
                fa = aggregate_full_stats(full_data.rounds, full_data.gaps)
                tot = {
                    **tot,
                    "task_sec": fa.get("task_sec") or tot.get("task_sec"),
                    "llm_wall_sec": fa.get("llm_wall_sec"),
                    "tool_wall_sec": fa.get("tool_wall_sec"),
                    "total_tokens_sum": fa.get("total_tokens_sum"),
                    "input_tokens_sum": fa.get("input_tokens_sum"),
                    "output_tokens_sum": fa.get("output_tokens_sum"),
                }
            except (ValueError, FileNotFoundError) as e:
                logger.info(f"full 指标合并跳过: {e}")
        if fusion_data is not None:
            fr = export_fusion_reconcile(fusion_data)
            tot = {
                **tot,
                "task_sec": fusion_data.summary.task_sec,
                "llm_wall_sec": fusion_data.summary.llm_wall_sec,
                "tool_wall_sec": fusion_data.summary.tool_wall_sec,
                "total_tokens_sum": fusion_data.summary.total_tokens_sum,
            }
        else:
            fr = {}

        source: SourceKind = "fusion" if fusion_data is not None else "history"
        bundle = build_history_bundle(
            session_id,
            rounds,
            tools,
            extras,
            history_path=str(hist_path),
            full_paths=[str(p) for p in full_files],
            report_paths=report_paths,
            aggregate=tot,
            fusion_reconcile=fr,
            source=source,
        )
        bundle.log_dir = str(work_dir)
        write_skill_bundle(bundle_path, bundle)
        _require_html_reports(
            report_paths,
            session_id,
            require_history=True,
            require_full=bool(full_files),
        )
        _emit_flow_artifacts(bundle_path, bundle, session_id, out_dir)
        logger.info(f"已生成 Skill 数据包: {bundle_path}")
        logger.info(f"  阶段数 {len(bundle.phases)} · 全局重复模式 {len(bundle.pattern_global)}")
        if fr:
            logger.info(
                f"  融合校对: 工具 {fr.get('tools_matched', 0)}/{fr.get('tools_history', 0)} 匹配 · "
                f"模型弱偏差 {fr.get('model_weak_overlap', 0)}"
            )
        return

    if full_files:
        session_id = (session_id or "").strip()
        if not session_id:
            raise OcperfError("仅 full 日志时必须指定 -s session_id")
        data = load_session_from_paths(session_id, full_files, history_path=hist_path)
        full_out = (args.full_out or out_dir / f"out_full_{_safe_name(session_id)}.html").resolve()
        guide = resolve_guide_href(full_out, args.guide_link)
        write_full_report(
            full_out,
            render_full_html(data, guide_href=guide, extended_analysis=extended),
        )
        report_paths["full_html"] = str(full_out)
        logger.info(f"已生成 full 报告: {full_out}")
        logger.info("提示: 仅 full 无法导出完整 skill_bundle，请提供 history.json。")
        return

    raise OcperfError("未找到 history.json 或 full 日志，请检查路径。")


def cmd_skill(args: argparse.Namespace) -> None:
    out_dir = (args.out_dir or Path.cwd()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.officeclaw:
        sid = (args.session or (str(args.input) if args.input else "")).strip()
        if not sid:
            raise OcperfError("OfficeClaw 模式需要 session_id（-s 或 positional input）")
        layout = resolve_officeclaw_session(
            sid,
            logs_root=args.logs_root,
            sessions_root=args.sessions_root,
        )
        work_dir = layout.session_dir
        if args.out_dir is None:
            out_dir = (layout.session_dir / "perf_reports").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"OfficeClaw 会话: {layout.session_id}")
        logger.info(f"  history: {layout.history_path}")
        logger.info(f"  full 日志: {len(layout.full_files)} 个文件 @ {layout.logs_root}")
        _run_pipeline(
            session_id=layout.session_id,
            hist_path=layout.history_path,
            full_files=layout.full_files,
            work_dir=work_dir,
            out_dir=out_dir,
            args=args,
        )
        return

    if args.input is None:
        raise OcperfError("请提供 input 路径，或使用 --officeclaw -s <session_id>")

    inp = args.input.resolve()
    if not inp.exists():
        raise OcperfError(f"路径不存在: {inp}")

    hist_path, full_files, work_dir = discover_inputs(inp)
    if args.out_dir is None:
        out_dir = work_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    _run_pipeline(
        session_id=args.session,
        hist_path=hist_path,
        full_files=full_files,
        work_dir=work_dir,
        out_dir=out_dir,
        args=args,
    )


def register_skill_parser(sub) -> None:
    p = sub.add_parser(
        "skill",
        help="Skill 工作流：生成 HTML 报告 + skill_bundle.json（供 Agent 解读）",
    )
    p.add_argument("input", type=Path, nargs="?", default=None)
    p.add_argument("-s", "--session", default=None)
    p.add_argument("--officeclaw", action="store_true")
    p.add_argument("--logs-root", type=Path, default=None)
    p.add_argument("--sessions-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--bundle-out", type=Path, default=None)
    p.add_argument("--history-out", type=Path, default=None)
    p.add_argument("--full-out", type=Path, default=None)
    p.add_argument("--fusion-out", type=Path, default=None)
    p.add_argument("--guide-link", default=None)
    p.add_argument("--no-extended", action="store_true")
    p.add_argument(
        "--fusion",
        action="store_true",
        help="生成 fusion 交叉报告（默认关闭，需 history+full）",
    )
    p.add_argument("--no-fusion", action="store_true", help="显式关闭 fusion（默认已关闭）")
    p.set_defaults(func=cmd_skill)
