#!/usr/bin/env python3
"""Compare HIXL and Mooncake Router logs with a closed TTFT breakdown.

The first table uses mutually exclusive request phases, so the component means
sum back to the observed mean client TTFT. The second table contains nested
diagnostics that explain a phase but must not be added to the first table.
"""

from __future__ import annotations

import sys

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
REQUEST_RE = re.compile(r"\brequest(?:_id)?=([^\s]+)")
COMPLETE_RE = re.compile(rf"\[Complete\]\s+(?P<request>\S+).*?ttft=(?P<ttft>{NUMBER})ms")

FIELDS = (
    # Exclusive-boundary inputs.
    "client_ttft_ms",
    "route_ms",
    "router_to_early_d_dispatch_ms",
    "d_selection_to_dispatch_ms",
    "p_complete_to_d_dispatch_ms",
    "worker_dispatch_to_first_byte_ms",
    # Router detail.
    "router_request_parse_ms",
    "router_preprocessor_ms",
    "router_route_ms",
    "router_worker_selection_ms",
    "router_route_attempt_ms",
    "router_capacity_turn_wait_ms",
    "router_capacity_wait_ms",
    "pd_pending_prefix_wait_ms",
    "router_route_build_ms",
    "router_route_unaccounted_ms",
    # P and transfer detail.
    "p_request_ms",
    "p_prefill_admission_wait_ms",
    "p_dispatch_to_scheduler_seen_ms",
    "p_scheduler_seen_to_kv_ready_ms",
    "p_kv_ready_to_export_ready_ms",
    "p_export_ready_to_http_response_ms",
    "p_response_network_ms",
    "pd_begin_ms",
    "pd_export_ms",
    "pd_link_ms",
    "pd_prepare_ms",
    "event_stream_connect_ms",
    "d_gate_wait_ms",
    "pd_import_ms",
    "d_request_seen_to_attach_start_ms",
    "d_attach_complete_to_first_byte_ms",
    "pre_decode_total_ms",
    "delta_plan_ms",
    "delta_inflight_wait_ms",
    "delta_capacity_wait_ms",
    "import_executor_queue_ms",
    "reserve_ms",
    "hixl_pull_ms",
    "attach_task_queue_ms",
    "attach_execution_ms",
    # Native connector dispatch detail.
    "snapshot_callback_ms",
    "d_dispatch_to_headers_ms",
)

RELEVANT_MARKERS = (
    "[Complete]",
    "[KVBMRouterTiming]",
    "[KVBMPDTiming]",
    "[MooncakePDTiming]",
    "[MooncakePDDispatchTiming]",
    "[KVBMPDRequestSummary]",
)
METRIC_PATTERNS = {
    name: (
        re.compile(rf"\b{re.escape(name)}\s*=\s*({NUMBER})"),
        re.compile(rf"['\"]{re.escape(name)}['\"]\s*:\s*({NUMBER})"),
    )
    for name in FIELDS
}

DIAGNOSTICS = (
    ("Router request parse", "router_request_parse_ms", "路由"),
    ("Router preprocessor", "router_preprocessor_ms", "路由"),
    ("Python Router route", "router_route_ms", "路由"),
    ("Router worker selection", "router_worker_selection_ms", "路由"),
    ("Router route attempts", "router_route_attempt_ms", "路由"),
    ("Router capacity turn wait", "router_capacity_turn_wait_ms", "路由"),
    ("Router capacity HTTP wait", "router_capacity_wait_ms", "路由"),
    ("Pending-prefix wait", "pd_pending_prefix_wait_ms", "路由"),
    ("Router route build", "router_route_build_ms", "路由"),
    ("Router unaccounted", "router_route_unaccounted_ms", "路由"),
    ("P request", "p_request_ms", "P/传输"),
    ("P admission wait", "p_prefill_admission_wait_ms", "P/传输"),
    ("P dispatch -> scheduler seen", "p_dispatch_to_scheduler_seen_ms", "P/传输"),
    ("P scheduler seen -> KV ready", "p_scheduler_seen_to_kv_ready_ms", "P/传输"),
    ("P KV ready -> export ready", "p_kv_ready_to_export_ready_ms", "P/传输"),
    ("P response network", "p_response_network_ms", "P/传输"),
    ("PD begin", "pd_begin_ms", "P/传输"),
    ("PD export", "pd_export_ms", "P/传输"),
    ("PD link", "pd_link_ms", "P/传输"),
    ("PD prepare", "pd_prepare_ms", "P/传输"),
    ("PD event-stream connect", "event_stream_connect_ms", "P/传输"),
    ("D gate wait", "d_gate_wait_ms", "D 首字节"),
    ("D request seen -> attach start", "d_request_seen_to_attach_start_ms", "D 首字节"),
    ("PD import", "pd_import_ms", "D 首字节"),
    ("D attach complete -> first byte", "d_attach_complete_to_first_byte_ms", "D 首字节"),
    ("Pre-decode total (inclusive)", "pre_decode_total_ms", "P/传输"),
    ("Delta plan", "delta_plan_ms", "D 首字节"),
    ("Delta in-flight wait", "delta_inflight_wait_ms", "D 首字节"),
    ("Delta capacity wait", "delta_capacity_wait_ms", "D 首字节"),
    ("Import executor queue", "import_executor_queue_ms", "D 首字节"),
    ("D block reserve", "reserve_ms", "D 首字节"),
    ("HIXL pull", "hixl_pull_ms", "D 首字节"),
    ("Attach task queue", "attach_task_queue_ms", "D 首字节"),
    ("Attach execution", "attach_execution_ms", "D 首字节"),
    ("Native D selection -> dispatch", "d_selection_to_dispatch_ms", "派发前"),
    ("P complete -> D dispatch", "p_complete_to_d_dispatch_ms", "派发前"),
    ("Native snapshot callback", "snapshot_callback_ms", "派发前"),
    ("D dispatch -> HTTP headers", "d_dispatch_to_headers_ms", "D 首字节"),
)

COMPONENTS = (
    ("请求进入 -> 路由完成", "route"),
    ("路由完成 -> D HTTP派发", "route_to_dispatch"),
    ("D HTTP派发 -> 首字节", "dispatch_to_first_byte"),
    ("未归因/计时边界", "residual"),
)


def write_report(*values, file=None, flush=False):
    """Write CLI result data without routing it through diagnostic logging."""
    stream = sys.stdout if file is None else file
    stream.write(" ".join(str(value) for value in values) + "\n")
    if flush:
        stream.flush()


@dataclass
class RunBreakdown:
    label: str
    requests: dict[str, dict[str, float]]
    joined_ids: list[str]
    components: dict[str, list[float]]
    sources: Counter[str]

    def mean_component(self, name: str) -> float:
        values = self.components[name]
        return sum(values) / len(values)

    def metric(self, name: str) -> tuple[float | None, int]:
        values = [
            self.requests[request_id][name] for request_id in self.joined_ids if name in self.requests[request_id]
        ]
        if not values:
            return None, 0
        return sum(values) / len(values), len(values)


def clean(line: str) -> str:
    return ANSI_RE.sub("", line)


def extract_metric(line: str, name: str) -> float | None:
    for pattern in METRIC_PATTERNS[name]:
        match = pattern.search(line)
        if match:
            value = float(match.group(1))
            return value if math.isfinite(value) else None
    return None


def parse_router_log(path: Path) -> dict[str, dict[str, float]]:
    requests: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            if not any(marker in raw_line for marker in RELEVANT_MARKERS):
                continue
            line = clean(raw_line)
            complete = COMPLETE_RE.search(line)
            request_match = REQUEST_RE.search(line)
            request_id = complete.group("request") if complete else request_match.group(1) if request_match else None
            if request_id is None:
                continue
            record = requests.setdefault(request_id, {})
            if complete:
                record["client_ttft_ms"] = float(complete.group("ttft"))
            for name in FIELDS:
                value = extract_metric(line, name)
                if value is not None and value >= 0:
                    record[name] = value
    return requests


def _route_ms(record: dict[str, float]) -> tuple[float | None, str]:
    if "route_ms" in record:
        return record["route_ms"], "rust_route_ms"
    if "router_route_ms" in record:
        return record["router_route_ms"], "python_router_route_ms"
    return None, "missing"


def build_breakdown(
    label: str,
    requests: dict[str, dict[str, float]],
    transport: str,
) -> RunBreakdown:
    components = {name: [] for _, name in COMPONENTS}
    components["ttft"] = []
    joined_ids: list[str] = []
    sources: Counter[str] = Counter()

    for request_id, record in requests.items():
        ttft = record.get("client_ttft_ms")
        dispatch_to_first_byte = record.get("worker_dispatch_to_first_byte_ms")
        route, route_source = _route_ms(record)
        if ttft is None or dispatch_to_first_byte is None or route is None:
            continue

        if transport == "hixl":
            dispatch_offset = record.get("router_to_early_d_dispatch_ms")
            if dispatch_offset is not None and dispatch_offset > 0:
                route_to_dispatch = dispatch_offset - route
                sources["router_to_early_d_dispatch_ms"] += 1
            else:
                route_to_dispatch = ttft - dispatch_to_first_byte - route
                sources["derived_from_ttft"] += 1
        else:
            if "p_request_ms" in record:
                route_to_dispatch = record["p_request_ms"] + record.get("p_complete_to_d_dispatch_ms", 0.0)
                sources["p_request_plus_dispatch_gap"] += 1
            elif "d_selection_to_dispatch_ms" in record:
                route_to_dispatch = record["d_selection_to_dispatch_ms"]
                sources["d_selection_to_dispatch_ms"] += 1
            else:
                route_to_dispatch = ttft - dispatch_to_first_byte - route
                sources["derived_from_ttft"] += 1

        residual = ttft - route - route_to_dispatch - dispatch_to_first_byte
        components["route"].append(route)
        components["route_to_dispatch"].append(route_to_dispatch)
        components["dispatch_to_first_byte"].append(dispatch_to_first_byte)
        components["residual"].append(residual)
        components["ttft"].append(ttft)
        joined_ids.append(request_id)
        sources[route_source] += 1

    return RunBreakdown(
        label=label,
        requests=requests,
        joined_ids=joined_ids,
        components=components,
        sources=sources,
    )


def _seconds(value_ms: float) -> str:
    return f"{value_ms / 1000.0:.3f}s"


def _signed_seconds(value_ms: float) -> str:
    return f"{value_ms / 1000.0:+.3f}s"


def _metric_cell(value: float | None, count: int) -> str:
    return "n/a" if value is None else f"{value / 1000.0:.3f}s (n={count})"


def compare_breakdowns(hixl: RunBreakdown, mooncake: RunBreakdown) -> str:
    if not hixl.joined_ids or not mooncake.joined_ids:
        missing = []
        if not hixl.joined_ids:
            missing.append(hixl.label)
        if not mooncake.joined_ids:
            missing.append(mooncake.label)
        raise ValueError(
            "无法形成TTFT闭环；缺少同请求的client_ttft_ms、route_ms/"
            "router_route_ms或worker_dispatch_to_first_byte_ms: " + ", ".join(missing)
        )

    hixl_ttft = hixl.mean_component("ttft")
    mooncake_ttft = mooncake.mean_component("ttft")
    total_saving = mooncake_ttft - hixl_ttft
    lines = [
        "== TTFT互斥阶段拆解（均值，可加总） ==",
        "",
        "| 阶段 | HIXL | Mooncake | HIXL节省（MC-HIXL） | 对总TTFT差值贡献 |",
        "|---|---:|---:|---:|---:|",
    ]

    component_savings = []
    for display_name, name in COMPONENTS:
        hixl_value = hixl.mean_component(name)
        mooncake_value = mooncake.mean_component(name)
        saving = mooncake_value - hixl_value
        component_savings.append((display_name, saving))
        contribution = f"{saving / total_saving:.1%}" if abs(total_saving) > 1e-9 else "n/a"
        lines.append(
            f"| {display_name} | {_seconds(hixl_value)} | "
            f"{_seconds(mooncake_value)} | {_signed_seconds(saving)} | "
            f"{contribution} |"
        )
    lines.append(
        f"| **Client TTFT** | **{_seconds(hixl_ttft)}** | "
        f"**{_seconds(mooncake_ttft)}** | "
        f"**{_signed_seconds(total_saving)}** | **100.0%** |"
    )

    component_sum = sum(saving for _, saving in component_savings)
    lines.extend(
        (
            "",
            "闭环校验：阶段差值合计 "
            f"{_signed_seconds(component_sum)}，TTFT实测差值 "
            f"{_signed_seconds(total_saving)}，误差 "
            f"{_signed_seconds(component_sum - total_saving)}。",
            "正值表示HIXL节省时间，负值表示HIXL额外耗时。",
            "",
            "== 主要收益与代价 ==",
        )
    )
    savings = sorted(
        ((name, value) for name, value in component_savings if value > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    costs = sorted(
        ((name, value) for name, value in component_savings if value < 0),
        key=lambda item: item[1],
    )
    if savings:
        lines.append("主要节省：" + "；".join(f"{name} {_signed_seconds(value)}" for name, value in savings))
    if costs:
        lines.append("主要代价：" + "；".join(f"{name} {_signed_seconds(value)}" for name, value in costs))

    lines.extend(
        (
            "",
            "== 嵌套诊断指标（不可与上表相加） ==",
            "",
            "| 指标 | 所属阶段 | HIXL均值 | Mooncake均值 | MC-HIXL |",
            "|---|---|---:|---:|---:|",
        )
    )
    for display_name, name, phase in DIAGNOSTICS:
        hixl_value, hixl_count = hixl.metric(name)
        mooncake_value, mooncake_count = mooncake.metric(name)
        if hixl_value is None and mooncake_value is None:
            continue
        difference = (
            _signed_seconds(mooncake_value - hixl_value)
            if hixl_value is not None and mooncake_value is not None
            else "n/a"
        )
        lines.append(
            f"| {display_name} | {phase} | "
            f"{_metric_cell(hixl_value, hixl_count)} | "
            f"{_metric_cell(mooncake_value, mooncake_count)} | {difference} |"
        )

    lines.extend(
        (
            "",
            "== 数据覆盖 ==",
            f"{hixl.label}: parsed_requests={len(hixl.requests)} "
            f"closed_requests={len(hixl.joined_ids)} "
            f"sources={dict(hixl.sources)}",
            f"{mooncake.label}: parsed_requests={len(mooncake.requests)} "
            f"closed_requests={len(mooncake.joined_ids)} "
            f"sources={dict(mooncake.sources)}",
            "",
            "说明：HIXL的P请求、import和attach通常与D请求并行，属于D派发到首字节阶段的内部诊断，不能再次加入TTFT总和。",
        )
    )
    return "\n".join(lines)


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"日志文件不存在: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对比prefix-HIXL与prefix-Mooncake Router日志的TTFT组成。")
    parser.add_argument("--hixl", required=True, type=_path, help="HIXL Router日志")
    parser.add_argument("--mooncake", required=True, type=_path, help="Mooncake Router日志")
    parser.add_argument("--hixl-label", default="prefix-hixl")
    parser.add_argument("--mooncake-label", default="prefix-mc")
    parser.add_argument(
        "--output",
        type=Path,
        help="可选：同时把Markdown结果写入文件",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    hixl = build_breakdown(
        args.hixl_label,
        parse_router_log(args.hixl),
        "hixl",
    )
    mooncake = build_breakdown(
        args.mooncake_label,
        parse_router_log(args.mooncake),
        "mooncake",
    )
    try:
        output = compare_breakdowns(hixl, mooncake)
    except ValueError as error:
        parser.error(str(error))
    write_report(output)
    if args.output is not None:
        args.output.expanduser().write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
