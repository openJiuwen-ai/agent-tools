#!/usr/bin/env python3
"""Summarize Router and decode-worker timing for one PD benchmark run."""

from __future__ import annotations

import sys

import argparse
import ast
from bisect import bisect_left
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

ROUTER_METRICS = (
    "router_worker_selection_ms",
    "router_route_attempt_ms",
    "router_capacity_turn_wait_ms",
    "router_capacity_wait_ms",
    "router_route_ms",
    "pd_pending_prefix_wait_ms",
    "p_request_ms",
    "pd_import_ms",
    "d_request_seen_to_attach_start_ms",
    "pre_decode_total_ms",
    "worker_dispatch_to_first_byte_ms",
    "client_ttft_ms",
)
MOONCAKE_WAIT_METRICS = (
    "mooncake_wait_ms",
    "mooncake_transfer_wait_ms",
    "kv_transfer_wait_ms",
    "connector_wait_ms",
    "transfer_wait_ms",
)
MOONCAKE_D_METRICS = (
    "d_http_received_to_engine_arrival_ms",
    "d_engine_arrival_to_connector_wait_ms",
    "connector_kv_wait_ms",
    "d_engine_arrival_to_connector_ready_ms",
    "d_http_received_to_connector_ready_ms",
    "connector_ready_to_first_scheduled_ms",
    "d_engine_arrival_to_first_scheduled_ms",
    "d_http_received_to_first_scheduled_ms",
    "first_scheduled_to_first_token_ms",
    "first_token_to_first_byte_ms",
    "first_scheduled_to_first_byte_ms",
    "d_http_received_to_first_byte_ms",
)

ENGINE_STATS_RE = re.compile(
    rf"Avg generation throughput:\s*({NUMBER})\s+tokens/s.*?"
    rf"Running:\s*({NUMBER})\s+reqs,\s*"
    rf"Waiting:\s*({NUMBER})\s+reqs,\s*"
    rf"GPU KV cache usage:\s*({NUMBER})%"
)
FULL_TIMESTAMP_RE = re.compile(r"(?P<stamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")
SHORT_TIMESTAMP_RE = re.compile(r"(?<!\d)(?P<stamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")
COMPLETE_RE = re.compile(
    rf"\[Complete\]\s+(?P<request>\S+).*?"
    rf"ttft=(?P<ttft>{NUMBER})ms\s+total=(?P<total>{NUMBER})ms\s+"
    rf"tokens=(?P<prompt>\d+)/(?P<completion>\d+)/(?P<tokens>\d+)"
)
REQUEST_RE = re.compile(r"\brequest(?:_id)?=([^\s]+)")
TARGET_RE = re.compile(r"\btarget=([^\s]+)")
CAPACITY_CONTROL_METRICS = (
    "handler_request_read_ms",
    "handler_execution_ms",
    "engine_core_proxy_ms",
    "engine_core_execution_ms",
    "engine_core_queue_and_transport_ms",
)


def write_report(*values, file=None, flush=False):
    """Write CLI result data without routing it through diagnostic logging."""
    stream = sys.stdout if file is None else file
    stream.write(" ".join(str(value) for value in values) + "\n")
    if flush:
        stream.flush()


def clean(line: str) -> str:
    return ANSI_RE.sub("", line)


def read_lines(paths: Iterable[Path]) -> Iterable[tuple[Path, str]]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                yield path, clean(line)


def extract_metric(line: str, name: str) -> float | None:
    patterns = (
        rf"\b{re.escape(name)}\s*=\s*({NUMBER})",
        rf"['\"]{re.escape(name)}['\"]\s*:\s*({NUMBER})",
    )
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return float(match.group(1))
    return None


def parse_timestamp(line: str, default_year: int | None = None) -> datetime | None:
    match = FULL_TIMESTAMP_RE.search(line)
    if match:
        return datetime.fromisoformat(match.group("stamp").replace(",", "."))
    match = SHORT_TIMESTAMP_RE.search(line)
    if match and default_year is not None:
        return datetime.fromisoformat(f"{default_year}-{match.group('stamp').replace(',', '.')}")
    return None


def percentile(values: list[float], quantile: float) -> float:
    index = max(0, math.ceil(len(values) * quantile) - 1)
    return sorted(values)[index]


def format_stats(name: str, values: list[float], unit: str = "ms") -> str:
    if not values:
        return f"{name}: no data"
    ordered = sorted(values)
    return (
        f"{name}: count={len(values)} avg={sum(values) / len(values):.2f}{unit} "
        f"p50={percentile(ordered, 0.50):.2f}{unit} "
        f"p90={percentile(ordered, 0.90):.2f}{unit} "
        f"p99={percentile(ordered, 0.99):.2f}{unit} "
        f"max={ordered[-1]:.2f}{unit}"
    )


def _reuse_class(reused: float, transferred: float) -> str:
    """Bucket one HIXL allocation without hiding the common near-full case."""
    total = reused + transferred
    if total <= 0 or reused <= 0:
        return "zero"
    ratio = reused / total
    if transferred <= 0:
        return "full"
    if ratio >= 0.90:
        return "near_full"
    return "partial"


def _request_tpot_ms(request: dict[str, float | str]) -> float | None:
    """Derive per-request TPOT using the same N-1 interval convention as Router."""
    completion_tokens = int(float(request.get("completion_tokens", 0)))
    if completion_tokens <= 1:
        return None
    generation_ms = max(
        float(request.get("total_ms", 0)) - float(request.get("ttft_ms", 0)),
        0.0,
    )
    return generation_ms / (completion_tokens - 1)


def _request_metric(
    request_ids: Iterable[str],
    requests: dict[str, dict[str, float | str]],
    name: str,
) -> list[float]:
    values = []
    for request_id in request_ids:
        request = requests.get(request_id)
        if request is None:
            continue
        if name == "tpot_ms":
            value = _request_tpot_ms(request)
        elif name == "generation_ms":
            value = max(float(request["total_ms"]) - float(request["ttft_ms"]), 0.0)
        else:
            raw = request.get(name)
            value = float(raw) if raw is not None else None
        if value is not None:
            values.append(value)
    return values


def summarize_decode_routing_diagnostics(
    lines: Iterable[str],
    complete_requests: dict,
    router_timings: dict,
    pd_summaries: dict,
) -> None:
    queue_events: dict[str, list[set[str]]] = defaultdict(list)
    affinities: dict[str, str] = {}
    confirmations: dict[str, list[float]] = defaultdict(list)
    reasons: Counter = Counter()
    rejected_requests: set[str] = set()
    accepted = 0
    legacy = 0
    for line in lines:
        request_match = REQUEST_RE.search(line)
        if not request_match:
            continue
        request_id = request_match.group(1)
        if "[PDDecodeLoadBalance]" in line:
            if "action=queue_relief" not in line:
                legacy += 1
                continue
            excluded_match = re.search(r"\bexcluded=(\[[^\]]*\])", line)
            if excluded_match:
                try:
                    excluded = ast.literal_eval(excluded_match.group(1))
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(excluded, list) or not all(isinstance(x, str) for x in excluded):
                    continue
                queue_events[request_id].append(set(excluded))
                affinity = re.search(r"\baffinity_worker=([^\s]+)", line)
                if affinity and affinity.group(1) != "None":
                    affinities.setdefault(request_id, affinity.group(1))
        elif "[PDDecodeCapacityConfirmation]" in line:
            elapsed = extract_metric(line, "elapsed_ms")
            confirmations[request_id].append(elapsed if elapsed is not None else math.nan)
            admission_match = re.search(r"\badmitted=(True|False)\b", line)
            admission = admission_match.group(1) if admission_match else None
            if admission == "False":
                rejected_requests.add(request_id)
                reason = re.search(r"\breason=([^\s]+)", line)
                reasons[reason.group(1) if reason else "unknown"] += 1
            elif admission == "True":
                accepted += 1

    write_report("\n== HIXL queue-relief decisions ==")
    write_report(f"legacy_load_filter_events={legacy}")
    write_report(f"queue_relief_events={sum(map(len, queue_events.values()))} requests={len(queue_events)}")
    if not queue_events:
        write_report("no queue-relief events found; older logs may not contain this marker")
    else:
        completed = set(queue_events).intersection(complete_requests, router_timings)
        outside = sum(str(router_timings[rid].get("target")) not in queue_events[rid][0] for rid in completed)
        affinity_joined = completed.intersection(affinities)
        changed = sum(str(router_timings[rid].get("target")) != affinities.get(rid) for rid in affinity_joined)
        write_report(f"completed_outside_initial_exclusions={outside}/{len(completed)}")
        write_report(f"explicit_affinity_target_changes={changed}/{len(affinity_joined)}")
        write_report("note: queue-relief decisions are candidate filters, not proof of a dispatched spill")
        for label, ids in (
            ("queue_relief_observed", completed),
            ("no_queue_relief_observed", set(complete_requests) - set(queue_events)),
        ):
            write_report(label + ":")
            write_report("  " + format_stats("ttft", _request_metric(ids, complete_requests, "ttft_ms")))
            transfers = [
                float(pd_summaries[rid]["transferred_blocks"])
                for rid in ids
                if rid in pd_summaries and "transferred_blocks" in pd_summaries[rid]
            ]
            write_report("  " + format_stats("transferred_blocks", transfers, " blocks"))

    write_report("\n== Decode capacity confirmation attempts ==")
    write_report(f"logged_attempts={sum(map(len, confirmations.values()))} requests={len(confirmations)}")
    write_report(f"logged_accepted={accepted} requests_with_rejections={len(rejected_requests)}")
    write_report(f"rejection_reasons={dict(sorted(reasons.items()))}")
    write_report(format_stats("attempts_per_request", [float(len(x)) for x in confirmations.values()], " attempts"))
    finite_elapsed = []
    for elapsed_values in confirmations.values():
        finite_elapsed.extend(value for value in elapsed_values if math.isfinite(value))
    write_report(format_stats("confirmation_elapsed", finite_elapsed))
    write_report(
        "note: counts are logged attempts, not underlying HTTP calls; shared queries may serve multiple requests"
    )


def summarize_router_backpressure(lines: Iterable[str], completes: dict, timings: dict) -> None:
    """Correlate request waits with observed queues, not inferred idle capacity."""
    observations: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    failures: Counter = Counter()
    per_worker: Counter = Counter()
    for line in lines:
        request = REQUEST_RE.search(line)
        if not request:
            continue
        rid = request.group(1)
        if "[PDDecodeCapacityConfirmation]" in line and re.search(r"\badmitted=False\b", line):
            worker = re.search(r"\bworker=(\S+)", line)
            if worker:
                per_worker[worker.group(1)] += 1
            if "reason=decode_capacity_confirmation_failed" in line:
                # Inspect only error fields; elapsed time alone cannot diagnose a timeout.
                error = re.search(r"\berror=(.*)", line)
                text = error.group(1).lower() if error else ""
                error_kinds = (
                    ("timeout", ("timeout", "timed out")),
                    ("connection", ("connecterror", "connection refused", "connection reset")),
                    ("http_status", ("status=", "status_code", "http status")),
                )
                kind = "unclassified"
                for label, terms in error_kinds:
                    if any(term in text for term in terms):
                        kind = label
                        break
                failures[kind] += 1
        if "[PDAdmissionWait]" not in line:
            continue
        match = re.search(r"\bdecode_details=(\{.*\})\s*$", line)
        if not match:
            continue
        try:
            details = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if not isinstance(details, dict):
            continue
        for worker, detail in details.items():
            if isinstance(worker, str) and isinstance(detail, dict):
                observations[rid].append((worker, detail))

    write_report("\n== Router backpressure versus observed D queues ==")
    write_report(
        f"admission_wait_requests={len(observations)} rejected_attempts_by_worker={dict(sorted(per_worker.items()))}"
    )
    write_report(f"confirmation_query_failure_categories={dict(sorted(failures.items()))}")
    low_queue: set[str] = set()
    by_worker: dict[str, set[str]] = defaultdict(set)
    reason_requests: dict[str, set[str]] = defaultdict(set)
    for rid, records in observations.items():
        for worker, detail in records:
            reason_requests[str(detail.get("reason", "unknown"))].add(rid)
            waiting = detail.get("waiting")
            if isinstance(waiting, (int, float)) and math.isfinite(waiting) and 0 <= waiting <= 1:
                low_queue.add(rid)
                by_worker[worker].add(rid)
    write_report(f"requests_with_rejection_while_D_waiting_le_1={len(low_queue)}")
    write_report(f"by_rejection_reason_requests={dict(sorted((k, len(v)) for k, v in reason_requests.items()))}")
    long_routes = set()
    for selected_rid in low_queue:
        if (
            isinstance(timings.get(selected_rid, {}).get("router_route_ms"), (int, float))
            and timings[selected_rid]["router_route_ms"] >= 1000
        ):
            long_routes.add(selected_rid)
    write_report(f"low_queue_rejection_and_route_ge_1s={len(long_routes)}")
    for worker, ids in sorted(by_worker.items()):
        write_report(f"{worker}: low_queue_rejected_requests={len(ids)}")
        write_report("  " + format_stats("route", _request_metric(ids, timings, "router_route_ms")))
        write_report("  " + format_stats("ttft", _request_metric(ids, completes, "ttft_ms")))
    write_report("slow low-queue rejection paths (top 10):")
    for rid in sorted(long_routes, key=lambda x: timings[x]["router_route_ms"], reverse=True)[:10]:
        phase_names = (
            "router_route_ms",
            "router_capacity_turn_wait_ms",
            "router_capacity_wait_ms",
            "pd_pending_prefix_wait_ms",
        )
        metric_text = " ".join(
            f"{name}={float(timings[rid][name]):.2f}ms" if name in timings[rid] else f"{name}=no-data"
            for name in phase_names
        )
        last = {}
        for worker, detail in observations[rid]:
            last[worker] = detail
        detail_text = "; ".join(
            f"{worker}:reason={d.get('reason', 'unknown')},running={d.get('running', 'no-data')},"
            f"waiting={d.get('waiting', 'no-data')},available={d.get('available_blocks', 'no-data')},"
            f"estimated={d.get('estimated_blocks', 'no-data')},pending={d.get('pending_reserved_blocks', 'no-data')}"
            for worker, d in sorted(last.items())
        )
        write_report(
            f"request={rid} target={timings[rid].get('target', 'unknown')} {metric_text} "
            f"last_observed_rejections=[{detail_text}]"
        )
    write_report(
        "note: admission-wait logs record state changes, not continuous queue duration. "
        "A low queue observation may occur before the long route wait; it does not prove affinity caused it "
        "or another D could safely admit the request. Missing fields remain unknown; "
        "summed attempt times are not exclusive TTFT phases."
    )


def summarize_small_delta_confirmations(lines: Iterable[str], completes: dict, timings: dict, summaries: dict) -> None:
    """Inspect small estimated budgets without treating stale estimates as allocations."""
    events: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in lines:
        if "[PDDecodeCapacityConfirmation]" not in line:
            continue
        request = REQUEST_RE.search(line)
        worker = re.search(r"\bworker=(\S+)", line)
        admitted = re.search(r"\badmitted=(True|False)\b", line)
        if not request or not worker or not admitted:
            continue
        reason = re.search(r"\breason=(\S+)", line)
        error = re.search(r"\berror=(.*?)(?:\s+initial_error=|$)", line)
        error_text = error.group(1).lower() if error else ""
        kind = "unclassified"
        if "timed out waiting for shared d worker capacity query" in error_text:
            kind = "shared_query_wait_timeout"
        else:
            for name in ("ReadTimeout", "ConnectTimeout", "PoolTimeout", "WriteTimeout"):
                if name.lower() in error_text:
                    kind = name
                    break
            else:
                if "timeout" in error_text or "timed out" in error_text:
                    kind = "timeout_unspecified"
                elif "connecterror" in error_text or "connection refused" in error_text:
                    kind = "connection_error"
        events[(request.group(1), worker.group(1))].append(
            {
                "stamp": parse_timestamp(line),
                "accepted": admitted.group(1) == "True",
                "reason": reason.group(1) if reason else "unknown",
                "error_kind": kind,
                "estimated": extract_metric(line, "estimated_blocks"),
                "elapsed": extract_metric(line, "elapsed_ms"),
                "shared": bool(re.search(r"\bshared=True\b", line)),
                "available": extract_metric(line, "available_blocks"),
                "margin": extract_metric(line, "margin_blocks"),
            }
        )

    write_report("\n== Small estimated delta capacity confirmations (1-8 blocks) ==")
    failures: dict[str, list[dict]] = defaultdict(list)
    categories: Counter = Counter()
    reasons: Counter = Counter()
    recovery_spans = []
    recovered_pairs = 0
    pairs = 0
    for (rid, _), records in sorted(events.items()):
        small = []
        for selected_r in records:
            if (
                not selected_r["accepted"]
                and selected_r["estimated"] is not None
                and (1 <= selected_r["estimated"] <= 8)
            ):
                small.append(selected_r)
        if not small:
            continue
        pairs += 1
        failures[rid].extend(small)
        for r in small:
            reasons[r["reason"]] += 1
            if r["reason"] == "decode_capacity_confirmation_failed":
                categories[r["error_kind"]] += 1
        # Require a later accepted record on the same Worker, not a success elsewhere.
        stamps = [r["stamp"] for r in small if r["stamp"] is not None]
        if stamps:
            first = min(stamps)
            accepted_stamps = []
            for matched_selected_r in records:
                if (
                    matched_selected_r["accepted"]
                    and matched_selected_r["stamp"] is not None
                    and (matched_selected_r["stamp"] > first)
                ):
                    accepted_stamps.append(matched_selected_r["stamp"])
            if accepted_stamps:
                recovered_pairs += 1
                recovery_spans.append((min(accepted_stamps) - first).total_seconds() * 1000)
    all_failures = [r for records in failures.values() for r in records]
    write_report(f"rejected_logged_attempts={len(all_failures)} requests={len(failures)} request_worker_pairs={pairs}")
    write_report(f"reasons={dict(sorted(reasons.items()))} query_failure_types={dict(sorted(categories.items()))}")
    write_report(f"shared_flagged_attempts={sum(r['shared'] for r in all_failures)}")
    apparent_room = []
    for rejected_attempt in all_failures:
        if (
            rejected_attempt["reason"] == "decode_capacity_confirmation_failed"
            and rejected_attempt["available"] is not None
            and rejected_attempt["margin"] is not None
        ):
            if rejected_attempt["available"] >= rejected_attempt["estimated"] + rejected_attempt["margin"]:
                apparent_room.append(rejected_attempt)
    write_report(f"query_failed_despite_previous_snapshot_room={len(apparent_room)}")
    write_report(
        format_stats("failed_confirmation_elapsed", [r["elapsed"] for r in all_failures if r["elapsed"] is not None])
    )
    for shared in (True, False):
        write_report(
            format_stats(
                f"shared_flag_{shared}_failed_elapsed",
                [r["elapsed"] for r in all_failures if r["shared"] == shared and r["elapsed"] is not None],
            )
        )
    write_report(f"same_worker_later_acceptance={recovered_pairs}/{pairs}")
    write_report(format_stats("first_small_rejection_to_later_acceptance", recovery_spans))
    write_report(format_stats("affected_request_route", _request_metric(failures, timings, "router_route_ms")))
    write_report(format_stats("affected_request_ttft", _request_metric(failures, completes, "ttft_ms")))
    write_report("slow affected requests (top 10):")

    def route_value(rid):
        return float(timings.get(rid, {}).get("router_route_ms", -1))

    for rid in sorted(failures, key=route_value, reverse=True)[:10]:
        elapsed = sum(r["elapsed"] for r in failures[rid] if r["elapsed"] is not None)
        route = f"{route_value(rid):.2f}ms" if route_value(rid) >= 0 else "no-data"
        summary = summaries.get(rid, {})
        write_report(
            f"request={rid} target={timings.get(rid, {}).get('target', 'unknown')} "
            f"small_rejections={len(failures[rid])} route={route} "
            f"logged_failed_elapsed_sum={elapsed:.2f}ms "
            f"final_reused={summary.get('reused_blocks', 'no-data')} "
            f"final_transferred={summary.get('transferred_blocks', 'no-data')}"
        )
    write_report(
        "note: 1-8 blocks is an analysis bucket, not an admission threshold. "
        "Failed-query budgets and available capacity come from previous snapshots, not authoritative allocations. "
        "shared=True does not prove a shared-wait timeout; exception types are classified only when explicit. "
        "Recovery spans include backoff/queueing, and elapsed sums are not exclusive TTFT costs. "
        "HTTP versus Worker execution time is unavailable without dedicated timing fields."
    )


def summarize_capacity_query_timings(lines: Iterable[str]) -> None:
    metrics = {name: [] for name in ("elapsed_ms", "http_ms", *CAPACITY_CONTROL_METRICS)}
    errors: Counter = Counter()
    stages: Counter = Counter()
    retries = recovered = count = 0
    for line in lines:
        if "[PDCapacityQueryTiming]" not in line:
            continue
        count += 1
        attempts = extract_metric(line, "attempts")
        retried = attempts is not None and attempts > 1
        retries += retried
        recovered += retried and bool(re.search(r"\bok=True\b", line))
        for name in metrics:
            value = extract_metric(line, name)
            if value is not None and math.isfinite(value) and value >= 0:
                metrics[name].append(value)
        if re.search(r"\bok=False\b", line):
            for field, counter in (("error_type", errors), ("failure_stage", stages)):
                match = re.search(rf"\b{field}=(\S+)", line)
                counter[match.group(1) if match else "unknown"] += 1
    write_report("\n== Capacity query phases and bounded retry ==")
    write_report(f"logged_confirmations={count} retried={retries} recovered_on_retry={recovered}")
    write_report(f"final_errors={dict(sorted(errors.items()))} failure_stages={dict(sorted(stages.items()))}")
    for name, values in metrics.items():
        write_report(format_stats(name, values))
    write_report(
        "note: elapsed includes both attempts, http_ms describes the final HTTP attempt. "
        "Worker phases are nested, not additive; timed-out HTTP calls may lack Worker timings. "
        "Older logs report no data."
    )


def summarize_router(path: Path) -> tuple[datetime | None, datetime | None]:
    # Store request-scoped records by request id. Router deployments sometimes
    # duplicate a log line through two handlers; deduping here prevents count=2N
    # and biased latency distributions.
    complete_requests: dict[str, dict[str, float | str]] = {}
    router_timings: dict[str, dict[str, float | str]] = {}
    route_targets: dict[str, str] = {}
    pd_summaries: dict[str, dict[str, float | str]] = {}
    pd_phase_timings: dict[str, dict[str, float]] = defaultdict(dict)
    mooncake_metrics = {name: [] for name in MOONCAKE_WAIT_METRICS}
    counters = {
        "router_managed_pd_inactive_fallback": 0,
        "proxy_errors": 0,
        "health_read_timeouts": 0,
        "health_connect_timeouts": 0,
    }
    first_route_at = None
    first_complete_at = None
    last_complete_at = None
    diagnostic_lines: dict[tuple[datetime | None, str], str] = {}

    for _, line in read_lines((path,)):
        timestamp = parse_timestamp(line)
        for marker in (
            "[PDDecodeLoadBalance]",
            "[PDDecodeCapacityConfirmation]",
            "[PDAdmissionWait]",
            "[PDCapacityQueryTiming]",
        ):
            if marker in line:
                # Deduplicate repeated handlers without merging distinct timed retries.
                diagnostic_lines[(timestamp, line[line.index(marker):].strip())] = line
        if "[ROUTER: START]" in line and timestamp is not None:
            first_route_at = first_route_at or timestamp
        if "[Complete]" in line and timestamp is not None:
            first_complete_at = first_complete_at or timestamp
            last_complete_at = timestamp

        complete_match = COMPLETE_RE.search(line)
        if complete_match:
            request_id = complete_match.group("request")
            complete_requests[request_id] = {
                "ttft_ms": float(complete_match.group("ttft")),
                "total_ms": float(complete_match.group("total")),
                "prompt_tokens": float(complete_match.group("prompt")),
                "completion_tokens": float(complete_match.group("completion")),
                "tokens": float(complete_match.group("tokens")),
                "timestamp": timestamp.isoformat() if timestamp else "",
            }

        if "[KVBMRouterTiming]" in line:
            request_match = REQUEST_RE.search(line)
            target_match = TARGET_RE.search(line)
            if request_match:
                request_id = request_match.group(1)
                target = target_match.group(1) if target_match else "unknown"
                if "stage=route" in line:
                    route_targets[request_id] = target
                elif "stage=complete" in line:
                    timing = {"target": target}
                    for name in ROUTER_METRICS:
                        value = extract_metric(line, name)
                        if value is not None:
                            timing[name] = value
                    router_timings[request_id] = timing

        if "[KVBMPDRequestSummary]" in line:
            request_match = REQUEST_RE.search(line)
            target_match = TARGET_RE.search(line)
            if request_match:
                summary: dict[str, float | str] = {"target": target_match.group(1) if target_match else "unknown"}
                for name in (
                    "p_request_ms",
                    "pd_import_ms",
                    "d_request_seen_to_attach_start_ms",
                    "worker_dispatch_to_first_byte_ms",
                    "ttft_ms",
                    "total_ms",
                    "transferred_blocks",
                    "reused_blocks",
                ):
                    value = extract_metric(line, name)
                    if value is not None:
                        summary[name] = value
                pd_summaries[request_match.group(1)] = summary

        if "[KVBMPDTiming]" in line:
            request_match = REQUEST_RE.search(line)
            if request_match:
                for name in ROUTER_METRICS:
                    value = extract_metric(line, name)
                    if value is not None:
                        pd_phase_timings[request_match.group(1)][name] = value
            for name in MOONCAKE_WAIT_METRICS:
                value = extract_metric(line, name)
                if value is not None:
                    mooncake_metrics[name].append(value)
            if 'fallback_mode="local_recompute"' in line:
                counters["router_managed_pd_inactive_fallback"] += 1

        if "Proxy error:" in line:
            counters["proxy_errors"] += 1
        if "health check connection error" in line and "type=ReadTimeout" in line:
            counters["health_read_timeouts"] += 1
        if "health check connection error" in line and "type=ConnectTimeout" in line:
            counters["health_connect_timeouts"] += 1

    # Use per-request fallbacks, including Rust request_id timing records.
    # This retains partial coverage without counting route/complete twice.
    timing_ids = set(router_timings) | set(pd_summaries) | set(pd_phase_timings)
    metrics = {name: [] for name in ROUTER_METRICS}
    for name in ROUTER_METRICS:
        for rid in timing_ids:
            for source in (router_timings, pd_summaries, pd_phase_timings):
                if name in source.get(rid, {}):
                    metrics[name].append(float(source[rid][name]))
                    break

    request_ids = list(complete_requests)
    ttft_values = _request_metric(request_ids, complete_requests, "ttft_ms")
    total_values = _request_metric(request_ids, complete_requests, "total_ms")
    generation_values = _request_metric(request_ids, complete_requests, "generation_ms")
    tpot_values = _request_metric(request_ids, complete_requests, "tpot_ms")
    per_request_output_tps = [1000.0 / value for value in tpot_values if value > 0]

    write_report("== Complete requests ==")
    write_report(format_stats("ttft", ttft_values))
    write_report(format_stats("generation_after_first_token", generation_values))
    write_report(format_stats("tpot", tpot_values, "ms/token"))
    write_report(format_stats("per_request_output_tps", per_request_output_tps, " tokens/s"))
    write_report(format_stats("total", total_values))
    write_report(
        format_stats(
            "output_tokens",
            [float(v["completion_tokens"]) for v in complete_requests.values()],
            " tokens",
        )
    )
    if total_values:
        total_latency = sum(total_values)
        write_report(
            "latency_composition: "
            f"ttft={sum(ttft_values) / total_latency:.2%} "
            f"generation_after_first_token={sum(generation_values) / total_latency:.2%}"
        )

    write_report("\n== Router critical path (deduplicated by request) ==")
    for name in ROUTER_METRICS:
        write_report(format_stats(name, metrics[name]))

    write_report("\n== Mooncake transfer wait ==")
    present = False
    for name in MOONCAKE_WAIT_METRICS:
        if mooncake_metrics[name]:
            write_report(format_stats(name, mooncake_metrics[name]))
            present = True
    if not present:
        write_report("no dedicated Mooncake transfer-wait metric in Router log")

    write_report("\n== Router warnings/errors ==")
    for name, count in counters.items():
        write_report(f"{name}: {count}")
    write_report("note: router_managed_pd_inactive_fallback does not prove that Mooncake recomputed the prompt on D")

    window_start = first_route_at or first_complete_at
    if window_start is not None and last_complete_at is not None:
        duration_s = max((last_complete_at - window_start).total_seconds(), 0.0)
        write_report(
            "\nBenchmark window inferred from Router log: "
            f"{window_start.isoformat(sep=' ')} -> "
            f"{last_complete_at.isoformat(sep=' ')}"
        )
        if duration_s > 0 and complete_requests:
            write_report(
                f"throughput={len(complete_requests) / duration_s:.3f} req/s "
                f"requests={len(complete_requests)} duration={duration_s:.3f}s"
            )

    write_report("\n== HIXL reuse classes ==")
    grouped: dict[str, list[str]] = defaultdict(list)
    allocations = []
    for request_id, summary in pd_summaries.items():
        if "reused_blocks" not in summary or "transferred_blocks" not in summary:
            continue
        reused = float(summary["reused_blocks"])
        transferred = float(summary["transferred_blocks"])
        allocations.append((request_id, reused, transferred))
        grouped[_reuse_class(reused, transferred)].append(request_id)
    if not allocations:
        write_report("no request-scoped HIXL allocation data")
    else:
        reused_values = [item[1] for item in allocations]
        transferred_values = [item[2] for item in allocations]
        write_report(format_stats("reused_blocks", reused_values, " blocks"))
        write_report(format_stats("transferred_blocks", transferred_values, " blocks"))
        write_report(
            f"overall_reuse_ratio={sum(reused_values) / max(sum(reused_values) + sum(transferred_values), 1.0):.2%}"
        )
        joined_allocations = sum(request_id in complete_requests for request_id, _, _ in allocations)
        write_report(f"allocation_ttft_joined={joined_allocations}/{len(allocations)}")
        for name in ("full", "near_full", "partial", "zero"):
            request_ids = grouped.get(name, [])
            ttft = _request_metric(request_ids, complete_requests, "ttft_ms")
            tpot = _request_metric(request_ids, complete_requests, "tpot_ms")
            total = _request_metric(request_ids, complete_requests, "total_ms")
            write_report(f"{name}: count={len(request_ids)} ratio={len(request_ids) / len(allocations):.2%}")
            if ttft:
                write_report("  " + format_stats("ttft", ttft))
                write_report("  " + format_stats("tpot", tpot, "ms/token"))
                write_report("  " + format_stats("total", total))

    write_report("\n== Decode target distribution ==")
    by_target: dict[str, list[str]] = defaultdict(list)
    for request_id, timing in router_timings.items():
        by_target[str(timing.get("target", "unknown"))].append(request_id)
    if not by_target:
        write_report("no request-scoped decode target data")
    for target, request_ids in sorted(by_target.items()):
        ttft = _request_metric(request_ids, complete_requests, "ttft_ms")
        tpot = _request_metric(request_ids, complete_requests, "tpot_ms")
        total = _request_metric(request_ids, complete_requests, "total_ms")
        waits = [
            float(router_timings[rid]["router_capacity_wait_ms"])
            for rid in request_ids
            if "router_capacity_wait_ms" in router_timings[rid]
        ]
        turn_waits = [
            float(router_timings[rid]["router_capacity_turn_wait_ms"])
            for rid in request_ids
            if "router_capacity_turn_wait_ms" in router_timings[rid]
        ]
        reused = [
            float(pd_summaries[rid]["reused_blocks"])
            for rid in request_ids
            if rid in pd_summaries and "reused_blocks" in pd_summaries[rid]
        ]
        avg_ttft = f"{sum(ttft) / len(ttft):.2f}ms" if ttft else "no-data"
        avg_tpot = f"{sum(tpot) / len(tpot):.2f}ms/token" if tpot else "no-data"
        avg_total = f"{sum(total) / len(total):.2f}ms" if total else "no-data"
        avg_wait = f"{sum(waits) / len(waits):.2f}ms" if waits else "no-data"
        avg_turn_wait = f"{sum(turn_waits) / len(turn_waits):.2f}ms" if turn_waits else "no-data"
        avg_reused = f"{sum(reused) / len(reused):.2f}" if reused else "no-data"
        write_report(
            f"{target}: requests={len(request_ids)} avg_ttft={avg_ttft} "
            f"avg_tpot={avg_tpot} avg_total={avg_total} "
            f"avg_capacity_turn_wait={avg_turn_wait} "
            f"avg_capacity_wait={avg_wait} avg_reused_blocks={avg_reused}"
        )

    changed_target = []
    for selected_request_id, selected_route_target in route_targets.items():
        if selected_request_id in router_timings and selected_route_target != str(
            router_timings[selected_request_id].get("target")
        ):
            changed_target.append(selected_request_id)
    comparable_targets = set(route_targets).intersection(router_timings)
    write_report(f"route_to_complete_target_changes={len(changed_target)}/{len(comparable_targets)}")

    summarize_decode_routing_diagnostics(
        diagnostic_lines.values(),
        complete_requests,
        router_timings,
        pd_summaries,
    )
    summarize_router_backpressure(diagnostic_lines.values(), complete_requests, router_timings)
    summarize_small_delta_confirmations(
        diagnostic_lines.values(),
        complete_requests,
        router_timings,
        pd_summaries,
    )
    summarize_capacity_query_timings(diagnostic_lines.values())

    write_report("\n== Request correlation coverage ==")
    write_report(f"complete_requests={len(complete_requests)}")
    write_report(f"router_complete_timings={len(router_timings)}")
    write_report(f"pd_request_summaries={len(pd_summaries)}")
    write_report(f"route_targets={len(route_targets)}")
    write_report(f"complete_and_router_joined={len(set(complete_requests).intersection(router_timings))}")
    write_report(f"complete_and_pd_joined={len(set(complete_requests).intersection(pd_summaries))}")

    write_report("\n== Slowest request paths ==")
    slowest = sorted(
        complete_requests,
        key=lambda rid: float(complete_requests[rid]["ttft_ms"]),
        reverse=True,
    )[:10]
    if not slowest:
        write_report("no complete request data")
    for request_id in slowest:
        complete = complete_requests[request_id]
        timing = router_timings.get(request_id, {})
        summary = pd_summaries.get(request_id, {})
        tpot_ms = _request_tpot_ms(complete)
        target = timing.get("target", summary.get("target", "unknown"))
        tpot_text = f"{tpot_ms:.2f}ms/token" if tpot_ms is not None else "no-data"
        attach_wait = timing.get(
            "d_request_seen_to_attach_start_ms", summary.get("d_request_seen_to_attach_start_ms", 0.0)
        )
        dispatch_wait = timing.get(
            "worker_dispatch_to_first_byte_ms", summary.get("worker_dispatch_to_first_byte_ms", 0.0)
        )
        write_report(
            f"request={request_id} target={target} "
            f"ttft={float(complete['ttft_ms']):.2f}ms "
            f"tpot={tpot_text} "
            f"route={float(timing.get('router_route_ms', 0.0)):.2f}ms "
            f"capacity_turn_wait={float(timing.get('router_capacity_turn_wait_ms', 0.0)):.2f}ms "
            f"capacity_wait={float(timing.get('router_capacity_wait_ms', 0.0)):.2f}ms "
            f"attach_wait={float(attach_wait):.2f}ms "
            f"dispatch_to_first_byte={float(dispatch_wait):.2f}ms "
            f"reused={int(float(summary.get('reused_blocks', 0)))} "
            f"transferred={int(float(summary.get('transferred_blocks', 0)))}"
        )
    return window_start, last_complete_at


def summarize_decode(
    paths: list[Path],
    window: tuple[datetime | None, datetime | None],
    *,
    alignment_tolerance: float = 3.0,
    sample_interval: float = 10.0,
    router_path: Path | None = None,
    worker_logs: dict[str, str] | None = None,
) -> None:
    write_report("\n== Decode worker load ==")
    all_values = {"gen_tps": [], "running": [], "waiting": [], "kv_usage": []}
    active_values = {
        "gen_tps": [],
        "running": [],
        "waiting": [],
        "kv_usage": [],
    }
    request_timings: dict[str, dict[str, float]] = {}
    capacity_timings = {}
    model_steps = defaultdict(list)
    work_windows = defaultdict(list)
    timed_samples: dict[str, dict[datetime, dict[str, float]]] = {str(path): {} for path in paths}
    window_start, window_end = window
    if window_start is not None:
        window_start -= timedelta(seconds=10)
    if window_end is not None:
        window_end += timedelta(seconds=10)

    for path in paths:
        values = {"gen_tps": [], "running": [], "waiting": [], "kv_usage": []}
        worker_active = {
            "gen_tps": [],
            "running": [],
            "waiting": [],
            "kv_usage": [],
        }
        for _, line in read_lines((path,)):
            timestamp = parse_timestamp(line, default_year=window_start.year if window_start else None)
            in_window = not (
                window_start is not None
                and timestamp is not None
                and (timestamp < window_start or (window_end and timestamp > window_end))
            )
            if in_window and "[KVBMModelStepTiming]" in line:
                fields = {}
                for selected_name in ("start_unix_s", "end_unix_s", "host_execution_ms"):
                    fields[selected_name] = extract_metric(line, selected_name)
                if all(v is not None for v in fields.values()):
                    model_steps[str(path)].append(fields)
            if in_window and "[KVBMPDWorkWindow]" in line:
                start = extract_metric(line, "start_unix_s")
                end = extract_metric(line, "end_unix_s")
                if start is not None and end is not None and end >= start:
                    work_windows[str(path)].append((start, end))
            if "[KVBMCapacityControlTiming]" in line and in_window:
                body = line[line.index("[KVBMCapacityControlTiming]"):].strip()
                phase_values = {}
                for name in CAPACITY_CONTROL_METRICS:
                    value = extract_metric(line, name)
                    if value is not None:
                        phase_values[name] = value
                capacity_timings[(str(path), timestamp, body)] = phase_values
            if "[MooncakeDRequestTiming]" in line:
                request_match = re.search(r"\brequest=([^\s]+)", line)
                if in_window and request_match and request_match.group(1) != "unavailable":
                    request_id = request_match.group(1)
                    timing = request_timings.setdefault(request_id, {})
                    for name in MOONCAKE_D_METRICS:
                        value = extract_metric(line, name)
                        if value is not None and value > 0:
                            timing[name] = value
            match = ENGINE_STATS_RE.search(line)
            if not match:
                continue
            if not in_window:
                continue

            sample = dict(zip(values, map(float, match.groups()), strict=True))
            if timestamp is not None:
                timed_samples[str(path)][timestamp] = sample
            is_active = sample["running"] + sample["waiting"] > 0
            for name, value in sample.items():
                values.setdefault(name, []).append(value)
                all_values.setdefault(name, []).append(value)
                if is_active:
                    worker_active.setdefault(name, []).append(value)
                    active_values.setdefault(name, []).append(value)

        if not values["gen_tps"]:
            write_report(f"{path.name}: no vLLM engine load samples")
            continue
        write_report(
            f"{path.name}: samples={len(values['gen_tps'])} "
            f"gen_tps={sum(values['gen_tps']) / len(values['gen_tps']):.2f} "
            f"running={sum(values['running']) / len(values['running']):.2f} "
            f"waiting={sum(values['waiting']) / len(values['waiting']):.2f} "
            f"kv={sum(values['kv_usage']) / len(values['kv_usage']):.2f}% "
            f"active_samples={len(worker_active['gen_tps'])} "
            f"active_ratio={len(worker_active['gen_tps']) / len(values['gen_tps']):.2%}"
        )
        if worker_active["gen_tps"]:
            write_report(
                f"  active: gen_tps="
                f"{sum(worker_active['gen_tps']) / len(worker_active['gen_tps']):.2f} "
                f"running="
                f"{sum(worker_active['running']) / len(worker_active['running']):.2f} "
                f"waiting="
                f"{sum(worker_active['waiting']) / len(worker_active['waiting']):.2f} "
                f"kv="
                f"{sum(worker_active['kv_usage']) / len(worker_active['kv_usage']):.2f}%"
            )

    if all_values["gen_tps"]:
        write_report("\n== Decode aggregate distributions ==")
        write_report(format_stats("gen_tps", all_values["gen_tps"], " tokens/s"))
        write_report(format_stats("running", all_values["running"], " reqs"))
        write_report(format_stats("waiting", all_values["waiting"], " reqs"))
        write_report(format_stats("kv_usage", all_values["kv_usage"], "%"))

    if active_values["gen_tps"]:
        write_report("\n== Decode active-sample distributions ==")
        write_report(format_stats("active_gen_tps", active_values["gen_tps"], " tokens/s"))
        write_report(format_stats("active_running", active_values["running"], " reqs"))
        write_report(format_stats("active_waiting", active_values["waiting"], " reqs"))
        write_report(format_stats("active_kv_usage", active_values["kv_usage"], "%"))

    write_report("\n== Worker capacity control phases (including late responses) ==")
    write_report(f"logged_capacity_responses={len(capacity_timings)}")
    for name in CAPACITY_CONTROL_METRICS:
        write_report(format_stats(name, [v[name] for v in capacity_timings.values() if name in v]))
    write_report(
        "note: Worker responses can finish after Router HTTP timeout; "
        "these are response observations, not unique requests or exclusive phases."
    )
    write_report("\n== Sampled model host execution versus pull/attach windows ==")
    overlapping, outside = [], []
    for path, steps in model_steps.items():
        for step in steps:
            overlap = any(
                start < step["end_unix_s"] and end > step["start_unix_s"] for start, end in work_windows[path]
            )
            (overlapping if overlap else outside).append(step["host_execution_ms"])
    write_report(
        f"model_step_samples={len(overlapping) + len(outside)} work_windows={sum(map(len, work_windows.values()))}"
    )
    write_report(format_stats("overlapping_host_execution", overlapping))
    write_report(format_stats("outside_logged_windows_host_execution", outside))
    write_report(
        "note: opt-in bounded host-call samples, not synchronized device execution or ITL. "
        "Missing/failed pull windows and unlogged phases remain unknown; overlap is not causation."
    )
    summarize_decode_balance(
        timed_samples,
        window,
        alignment_tolerance,
        sample_interval,
    )
    if router_path is not None:
        summarize_generation_load(router_path, timed_samples, worker_logs or {}, sample_interval)

    for timing in request_timings.values():
        http_to_engine_ms = timing.get("d_http_received_to_engine_arrival_ms")
        engine_to_ready_ms = timing.get("d_engine_arrival_to_connector_ready_ms")
        if http_to_engine_ms is not None and engine_to_ready_ms is not None:
            timing["d_http_received_to_connector_ready_ms"] = http_to_engine_ms + engine_to_ready_ms
        engine_to_scheduled_ms = timing.get("d_engine_arrival_to_first_scheduled_ms")
        if http_to_engine_ms is not None and engine_to_scheduled_ms is not None:
            timing["d_http_received_to_first_scheduled_ms"] = http_to_engine_ms + engine_to_scheduled_ms
        if "first_scheduled_to_first_byte_ms" in timing:
            continue
        first_scheduled_ms = timing.get("d_http_received_to_first_scheduled_ms")
        first_byte_ms = timing.get("d_http_received_to_first_byte_ms")
        if first_scheduled_ms is not None and first_byte_ms is not None:
            timing["first_scheduled_to_first_byte_ms"] = max(0.0, first_byte_ms - first_scheduled_ms)

    write_report("\n== Mooncake D request phases ==")
    if not request_timings:
        write_report("no [MooncakeDRequestTiming] records")
    else:
        for name in MOONCAKE_D_METRICS:
            values = [timing[name] for timing in request_timings.values() if name in timing]
            write_report(format_stats(name, values))


def summarize_decode_balance(
    samples: dict[str, dict[datetime, dict[str, float]]],
    window: tuple[datetime | None, datetime | None],
    tolerance: float = 3.0,
    interval: float = 10.0,
) -> None:
    """Compare near-simultaneous engine snapshots without filling missing data."""
    write_report("\n== Decode time-aligned balance ==")
    if len(samples) < 2:
        write_report("requires logs from at least two decode workers")
        return
    start, end = window
    samples = {
        path: {
            stamp: value
            for stamp, value in records.items()
            if (start is None or stamp >= start) and (end is None or stamp <= end)
        }
        for path, records in samples.items()
    }
    reference = max(samples, key=lambda path: len(samples[path]))
    stamps_by_path = {path: sorted(records) for path, records in samples.items()}
    reference_stamps = stamps_by_path[reference]
    if not reference_stamps:
        write_report("no timestamped engine samples in benchmark window")
        return
    used: dict[str, set[datetime]] = defaultdict(set)
    metrics = {name: [] for name in ("running_gap", "running_cv", "waiting_gap", "kv_gap")}
    rows = []
    incomplete = idle = active = skewed = queued_low = queued_low_kv = 0
    streak = longest = 0
    streak_start = previous = None
    longest_span = 0.0
    for stamp in reference_stamps:
        selected = {reference: (stamp, samples[reference][stamp])}
        for path, stamps in stamps_by_path.items():
            if path == reference:
                continue
            position = bisect_left(stamps, stamp)
            choices = [
                stamps[i] for i in (position - 1, position) if 0 <= i < len(stamps) and stamps[i] not in used[path]
            ]
            if choices:
                nearest = min(choices, key=lambda candidate: abs((candidate - stamp).total_seconds()))
                if abs((nearest - stamp).total_seconds()) <= tolerance:
                    selected[path] = (nearest, samples[path][nearest])
        spread = (
            max(value[0] for value in selected.values()) - min(value[0] for value in selected.values())
        ).total_seconds()
        if len(selected) != len(samples) or spread > tolerance:
            incomplete += 1
            streak = 0
            previous = None
            continue
        for path, (chosen, _) in selected.items():
            used[path].add(chosen)
        population = [value[1] for value in selected.values()]
        if not any(value["running"] + value["waiting"] > 0 for value in population):
            idle += 1
            streak = 0
            previous = None
            continue
        active += 1
        running = [value["running"] for value in population]
        mean = sum(running) / len(running)
        gap = max(running) - min(running)
        cv = math.sqrt(sum((value - mean) ** 2 for value in running) / len(running)) / mean if mean else 0.0
        metrics["running_gap"].append(gap)
        metrics["running_cv"].append(cv)
        metrics["waiting_gap"].append(max(x["waiting"] for x in population) - min(x["waiting"] for x in population))
        metrics["kv_gap"].append(max(x["kv_usage"] for x in population) - min(x["kv_usage"] for x in population))
        is_skewed = gap >= 6 and max(running) >= 2 * max(min(running), 1)
        has_queued_low = any(
            a != b and population[a]["waiting"] >= 2 and population[b]["running"] <= 2
            for a in range(len(population))
            for b in range(len(population))
        )
        has_queued_low_kv = any(
            a != b
            and population[a]["waiting"] >= 2
            and population[b]["running"] <= 2
            and population[b]["kv_usage"] < 90
            for a in range(len(population))
            for b in range(len(population))
        )
        queued_low += has_queued_low
        queued_low_kv += has_queued_low_kv
        if is_skewed:
            skewed += 1
            if streak == 0 or previous is None or (stamp - previous).total_seconds() > interval + tolerance:
                streak = 1
                streak_start = stamp
            else:
                streak += 1
            span = (stamp - streak_start).total_seconds()
            if streak > longest or (streak == longest and span > longest_span):
                longest = streak
                longest_span = span
            previous = stamp
        else:
            streak = 0
            previous = None
        rows.append((gap, stamp, spread, selected))

    write_report(
        f"workers={len(samples)} reference={Path(reference).name} "
        f"tolerance={tolerance:.1f}s expected_interval={interval:.1f}s"
    )
    write_report(
        f"reference_samples={len(reference_stamps)} aligned_samples={active + idle} "
        f"incomplete_or_misaligned={incomplete} active_aligned={active} idle_aligned={idle}"
    )
    if not active:
        write_report("no complete active snapshots; missing workers are not treated as idle")
        return
    for name, values in metrics.items():
        unit = " reqs" if name in {"running_gap", "waiting_gap"} else (" percentage-points" if name == "kv_gap" else "")
        write_report(format_stats(name, values, unit))
    write_report(f"skewed_samples={skewed}/{active} ({skewed / active:.2%}) threshold=gap>=6_and_max>=2*max(min,1)")
    write_report(f"longest_skewed_streak={longest} samples observed_span={longest_span:.2f}s")
    write_report(
        f"queued_with_low_running={queued_low}/{active} queued_with_low_running_and_kv_below90={queued_low_kv}/{active}"
    )
    write_report("note: sample ratios are not time fractions; observed spans do not prove continuous skew.")
    write_report(
        "note: Running/Waiting omit router reservations and import gates; KV usage does not prove admission capacity."
    )
    write_report("Worst aligned snapshots:")
    for gap, stamp, spread, selected in sorted(rows, key=lambda row: row[0], reverse=True)[:5]:
        detail = "; ".join(
            f"{Path(path).name}@{chosen.time()} running={value['running']:.0f} "
            f"waiting={value['waiting']:.0f} kv={value['kv_usage']:.1f}% gen_tps={value['gen_tps']:.1f}"
            for path, (chosen, value) in sorted(selected.items())
        )
        write_report(f"{stamp.isoformat(sep=' ')} gap={gap:.0f} spread={spread:.2f}s {detail}")


def summarize_generation_load(router_path, samples, worker_logs, interval):
    """Join per-request generation windows to observed worker snapshots only."""
    write_report("\n== Request generation-window D load ==")
    requests = {}
    for _, line in read_lines((router_path,)):
        match = COMPLETE_RE.search(line)
        stamp = parse_timestamp(line)
        if not match or stamp is None:
            continue
        workers = re.search(r"workers=(\[[^\]]*\])", line)
        if not workers:
            continue
        try:
            targets = ast.literal_eval(workers.group(1))
        except (ValueError, SyntaxError):
            continue
        decode = [w for w in targets if isinstance(w, str) and w.endswith("-decode")]
        output = int(match.group("completion"))
        duration = float(match.group("total")) - float(match.group("ttft"))
        if len(decode) != 1 or output <= 1 or duration <= 0:
            continue
        requests[match.group("request")] = (decode[0], stamp, duration, duration / (output - 1))
    counts = Counter()
    rows = []
    for target, end, duration, tpot in requests.values():
        path = worker_logs.get(target)
        if path not in samples:
            counts["unmapped"] += 1
            continue
        start = end - timedelta(milliseconds=duration)
        observed = sorted((t, v) for t, v in samples[path].items() if start <= t <= end)
        counts["joined"] += bool(observed)
        gaps = []
        for selected_a, selected_b in zip([start] + [t for (t, _) in observed], [t for (t, _) in observed] + [end]):
            gaps.append((selected_b - selected_a).total_seconds())
        adequate = len(observed) >= 2 and max(gaps) <= interval * 1.5
        counts["adequate" if adequate else "insufficient"] += 1
        if observed:
            rows.append(
                (
                    target,
                    adequate,
                    len(observed),
                    tpot,
                    sum(v["running"] for _, v in observed) / len(observed),
                    sum(v["waiting"] for _, v in observed) / len(observed),
                )
            )
    total = len(requests)
    write_report(
        f"eligible_requests={total} joined={counts['joined']} adequate={counts['adequate']} "
        f"insufficient={counts['insufficient']} unmapped={counts['unmapped']} "
        f"adequate_coverage={counts['adequate'] / total if total else 0:.2%}"
    )
    write_report(format_stats("snapshots_per_joined_request", [float(r[2]) for r in rows], " samples"))
    for target in sorted({r[0] for r in rows}):
        selected = [r for r in rows if r[0] == target and r[1]]
        write_report(f"{target}: adequately_covered_requests={len(selected)}")
        for name, index, unit in (
            ("observed_running", 4, " reqs"),
            ("observed_waiting", 5, " reqs"),
            ("tpot", 3, "ms/token"),
        ):
            write_report("  " + format_stats(name, [r[index] for r in selected], unit))
    for lower, upper in ((0, 4), (4, 8), (8, 12), (12, math.inf)):
        selected = [r for r in rows if r[1] and lower <= r[4] < upper]
        write_report(format_stats(f"TPOT at observed_running [{lower},{upper})", [r[3] for r in selected], "ms/token"))
    write_report(
        "note: coarse snapshot means, not batch size or time-weighted trajectories; "
        "correlation is not causation. Short/sparsely sampled requests are excluded from buckets. "
        "Completion timestamps approximate stream completion; worker clocks must be aligned."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one Prefix/RR Mooncake PD benchmark run.")
    parser.add_argument("--label", default="run", help="Treatment name")
    parser.add_argument("--router-log", type=Path, required=True)
    parser.add_argument(
        "--alignment-tolerance",
        type=float,
        default=3.0,
        help="Maximum timestamp spread across D snapshots in seconds (default: 3)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=10.0,
        help="Expected engine logging interval for observed streaks (default: 10)",
    )
    parser.add_argument(
        "--worker-log",
        action="append",
        default=[],
        metavar="WORKER=PATH",
        help="Explicit D worker to log mapping for generation-window joins; repeatable",
    )
    parser.add_argument(
        "--decode-log",
        type=Path,
        nargs="+",
        default=[],
        help="Decode worker logs, for example: --decode-log d*.log",
    )
    args = parser.parse_args()
    args.worker_logs = {}
    for mapping in args.worker_log:
        worker, separator, path = mapping.partition("=")
        if not separator or not worker or not path:
            parser.error("--worker-log requires unique WORKER=PATH mappings")
        if worker in args.worker_logs:
            parser.error("--worker-log requires unique WORKER=PATH mappings")
        if Path(path) not in args.decode_log:
            parser.error("--worker-log PATH must also be provided in --decode-log")
        args.worker_logs[worker] = str(Path(path))
    if not math.isfinite(args.alignment_tolerance) or args.alignment_tolerance < 0:
        parser.error("--alignment-tolerance must be finite and nonnegative")
    if not math.isfinite(args.sample_interval) or args.sample_interval <= 0:
        parser.error("--sample-interval must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.router_log.is_file():
        raise SystemExit(f"Router log does not exist: {args.router_log}")
    missing = [path for path in args.decode_log if not path.is_file()]
    if missing:
        raise SystemExit(f"Decode log does not exist: {missing[0]}")

    write_report(f"Treatment: {args.label}")
    window = summarize_router(args.router_log)
    summarize_decode(
        args.decode_log,
        window,
        alignment_tolerance=args.alignment_tolerance,
        sample_interval=args.sample_interval,
        router_path=args.router_log,
        worker_logs=args.worker_logs,
    )


if __name__ == "__main__":
    main()
