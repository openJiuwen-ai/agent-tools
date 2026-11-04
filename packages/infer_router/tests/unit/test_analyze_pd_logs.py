# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
"""Tests for the standalone PD benchmark log analyzer."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "tools" / "analyze_pd_logs.py"
SPEC = importlib.util.spec_from_file_location("analyze_pd_logs", SCRIPT)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def test_generation_load_uses_own_worker_and_reports_sparse_coverage(tmp_path, capsys):
    path = tmp_path / "router.log"
    path.write_text(
        "2026-09-17 10:00:30 [Complete] r1 workers=['worker-2-decode'] "
        "ttft=1000ms total=31000ms tokens=100/4/104\n"
        "2026-09-17 10:00:30 [Complete] r2 workers=['worker-2-decode'] "
        "ttft=1000ms total=2000ms tokens=100/4/104\n"
        "2026-09-17 10:00:30 [Complete] r3 workers=['worker-4-decode'] "
        "ttft=1000ms total=31000ms tokens=100/4/104\n"
    )
    base = datetime(2026, 9, 17, 10)
    samples = {"d1.log": {base + timedelta(seconds=i): {"running": 6.0, "waiting": 1.0} for i in (10, 20)}}
    analyzer.summarize_generation_load(path, samples, {"worker-2-decode": "d1.log"}, 10.0)
    output = capsys.readouterr().out
    assert "eligible_requests=3 joined=1 adequate=1 insufficient=1 unmapped=1" in output
    assert "observed_running: count=1 avg=6.00" in output
    assert "TPOT at observed_running [4,8): count=1 avg=10000.00" in output


def test_generation_load_does_not_fill_large_sample_gaps(tmp_path, capsys):
    path = tmp_path / "router.log"
    path.write_text(
        "2026-09-17 10:01:00 [Complete] r1 workers=['worker-2-decode'] ttft=0ms total=60000ms tokens=100/4/104\n"
    )
    base = datetime(2026, 9, 17, 10)
    samples = {"d1.log": {base + timedelta(seconds=i): {"running": 6.0, "waiting": 0.0} for i in (10, 50)}}
    analyzer.summarize_generation_load(path, samples, {"worker-2-decode": "d1.log"}, 10.0)
    assert "adequate=0 insufficient=1" in capsys.readouterr().out


def test_model_step_overlap_is_scoped_to_same_worker(tmp_path, capsys):
    first = tmp_path / "d1.log"
    second = tmp_path / "d2.log"
    first.write_text(
        "2026-09-17 10:00:00 [KVBMPDWorkWindow] phase=pull "
        "start_unix_s=100 end_unix_s=102\n"
        "2026-09-17 10:00:00 [KVBMModelStepTiming] "
        "start_unix_s=101 end_unix_s=103 host_execution_ms=40\n"
    )
    second.write_text(
        "2026-09-17 10:00:00 [KVBMModelStepTiming] start_unix_s=101 end_unix_s=103 host_execution_ms=10\n"
    )
    analyzer.summarize_decode([first, second], (None, None))
    output = capsys.readouterr().out
    assert "model_step_samples=2 work_windows=1" in output
    assert "overlapping_host_execution: count=1 avg=40.00" in output
    assert "outside_logged_windows_host_execution: count=1 avg=10.00" in output


def test_router_summary_deduplicates_and_correlates_hixl_requests(tmp_path, capsys):
    router_log = tmp_path / "router.log"
    router_log.write_text(
        "\n".join(
            (
                "2026-09-09 10:00:00 [ROUTER: START] request=r1",
                "2026-09-09 10:00:01 [KVBMRouterTiming] request=r1 stage=route "
                "target=d0 phase_ms={'router_route_ms': 10}",
                "2026-09-09 10:00:02 [KVBMPDRequestSummary] request_index=1 "
                "request=r1 target=d0 ttft_ms=100 total_ms=200 "
                "d_request_seen_to_attach_start_ms=5 "
                "worker_dispatch_to_first_byte_ms=80 transferred_blocks=5 "
                "reused_blocks=156",
                "2026-09-09 10:00:02 [Complete] r1 workers=['p0', 'd0'] "
                "ttft=100.0ms total=200.0ms tokens=1000/256/1256 overlap=[]",
                "2026-09-09 10:00:02 [KVBMRouterTiming] request=r1 stage=complete "
                "target=d0 phase_ms={'router_route_ms': 10, "
                "'router_capacity_turn_wait_ms': 2, "
                "'router_capacity_wait_ms': 0, 'client_ttft_ms': 100}",
                # Duplicate logger output must not double the request count.
                "2026-09-09 10:00:02 [KVBMRouterTiming] request=r1 stage=complete "
                "target=d0 phase_ms={'router_route_ms': 10, "
                "'router_capacity_turn_wait_ms': 2, "
                "'router_capacity_wait_ms': 0, 'client_ttft_ms': 100}",
                "2026-09-09 10:00:03 [KVBMRouterTiming] request=r2 stage=route "
                "target=d0 phase_ms={'router_route_ms': 20}",
                "2026-09-09 10:00:04 [KVBMPDRequestSummary] request_index=2 "
                "request=r2 target=d1 ttft_ms=300 total_ms=500 "
                "d_request_seen_to_attach_start_ms=20 "
                "worker_dispatch_to_first_byte_ms=250 transferred_blocks=161 "
                "reused_blocks=0",
                "2026-09-09 10:00:04 [Complete] r2 workers=['p0', 'd1'] "
                "ttft=300.0ms total=500.0ms tokens=1000/256/1256 overlap=[]",
                "2026-09-09 10:00:04 [KVBMRouterTiming] request=r2 stage=complete "
                "target=d1 phase_ms={'router_route_ms': 20, "
                "'router_capacity_turn_wait_ms': 6, "
                "'router_capacity_wait_ms': 4, 'client_ttft_ms': 300}",
            )
        ),
        encoding="utf-8",
    )

    analyzer.summarize_router(router_log)

    output = capsys.readouterr().out
    assert "ttft: count=2 avg=200.00ms" in output
    assert "generation_after_first_token: count=2 avg=150.00ms" in output
    assert "tpot: count=2 avg=0.59ms/token" in output
    assert "latency_composition: ttft=57.14% generation_after_first_token=42.86%" in output
    assert "router_route_ms: count=2 avg=15.00ms" in output
    assert "router_capacity_turn_wait_ms: count=2 avg=4.00ms" in output
    assert "d0: requests=1" in output
    assert "avg_capacity_turn_wait=2.00ms" in output
    assert "d1: requests=1" in output
    assert "avg_capacity_turn_wait=6.00ms" in output
    assert "near_full: count=1 ratio=50.00%" in output
    assert "zero: count=1 ratio=50.00%" in output
    assert "avg_tpot=0.39ms/token" in output
    assert "route_to_complete_target_changes=1/2" in output
    assert "throughput=0.500 req/s requests=2 duration=4.000s" in output


def test_reuse_class_keeps_near_full_separate_from_partial():
    assert analyzer._reuse_class(161, 0) == "full"
    assert analyzer._reuse_class(156, 5) == "near_full"
    assert analyzer._reuse_class(80, 81) == "partial"
    assert analyzer._reuse_class(0, 161) == "zero"


def test_request_tpot_uses_completion_intervals():
    request = {"ttft_ms": 100, "total_ms": 2650, "completion_tokens": 256}
    assert analyzer._request_tpot_ms(request) == 10
    assert analyzer._request_tpot_ms({**request, "completion_tokens": 1}) is None


def test_queue_relief_diagnostics_distinguishes_decisions_from_dispatch(tmp_path, capsys):
    path = tmp_path / "router.log"
    decision = (
        "2026-09-17 10:00:00 [PDDecodeLoadBalance] request=r1 action=queue_relief "
        "loads={} excluded=['d0'] affinity_worker=d0"
    )
    rejected = (
        "2026-09-17 10:00:01 [PDDecodeCapacityConfirmation] request=r1 worker=d1 "
        "admitted=False reason=decode_free_block_budget elapsed_ms=50"
    )
    path.write_text(
        "\n".join(
            (
                decision,
                decision,
                rejected,
                rejected,
                "2026-09-17 10:00:02 [PDDecodeCapacityConfirmation] request=r1 worker=d0 "
                "admitted=True elapsed_ms=10 shared=True",
                "2026-09-17 10:00:03 [Complete] r1 ttft=100ms total=200ms tokens=1000/256/1256",
                "2026-09-17 10:00:03 [KVBMRouterTiming] request=r1 stage=complete target=d0",
                "2026-09-17 10:00:03 [KVBMPDRequestSummary] request=r1 target=d0 "
                "transferred_blocks=5 reused_blocks=156",
            )
        ),
        encoding="utf-8",
    )
    analyzer.summarize_router(path)
    output = capsys.readouterr().out
    assert "queue_relief_events=1 requests=1" in output
    assert "completed_outside_initial_exclusions=0/1" in output
    assert "explicit_affinity_target_changes=0/1" in output
    assert "logged_attempts=2 requests=1" in output
    assert "rejection_reasons={'decode_free_block_budget': 1}" in output
    assert "confirmation_elapsed: count=2 avg=30.00ms" in output


def test_queue_relief_diagnostics_handles_legacy_and_malformed_records(capsys):
    analyzer.summarize_decode_routing_diagnostics(
        [
            "[PDDecodeLoadBalance] request=r1 loads={} excluded=['d0']",
            "[PDDecodeLoadBalance] request=r2 action=queue_relief excluded=[invalid]",
        ],
        {},
        {},
        {},
    )
    output = capsys.readouterr().out
    assert "legacy_load_filter_events=1" in output
    assert "queue_relief_events=0 requests=0" in output
    assert "older logs may not contain this marker" in output


def test_confirmation_admitted_field_is_not_ticket_admitted(capsys):
    analyzer.summarize_decode_routing_diagnostics(
        [
            "[PDDecodeCapacityConfirmation] request=r1 worker=d0 admitted=True ticket_admitted=False elapsed_ms=50",
            "[PDDecodeCapacityConfirmation] request=r2 worker=d1 admitted=False "
            "reason=decode_free_block_budget ticket_admitted=True elapsed_ms=60",
        ],
        {},
        {},
        {},
    )
    output = capsys.readouterr().out
    assert "logged_accepted=1 requests_with_rejections=1" in output
    assert "rejection_reasons={'decode_free_block_budget': 1}" in output


def test_router_backpressure_correlates_waits_without_filling_missing_queues(tmp_path, capsys):
    path = tmp_path / "router.log"
    wait = (
        "2026-09-17 10:00:00 [PDAdmissionWait] request=r1 blocker=decode "
        "decode_details={'d0': {'reason': 'decode_free_block_budget', 'running': 17, "
        "'waiting': 0, 'available_blocks': 20, 'estimated_blocks': 161}, "
        "'d1': {'reason': 'decode_block_budget'}}"
    )
    path.write_text(
        "\n".join(
            (
                wait,
                wait,
                "[PDAdmissionWait] request=r2 decode_details={'d1': {'waiting': -1}}",
                "[PDAdmissionWait] request=r3 decode_details={bad}",
                "[PDDecodeCapacityConfirmation] request=r1 worker=d0 admitted=False "
                "reason=decode_capacity_confirmation_failed elapsed_ms=50 error=ReadTimeout",
                "[PDDecodeCapacityConfirmation] request=r1 worker=d0 admitted=True ticket_admitted=False",
                "[Complete] r1 ttft=2500ms total=4000ms tokens=1000/256/1256",
                "[KVBMRouterTiming] request=r1 stage=complete target=d0 "
                "phase_ms={'router_route_ms': 2000, 'pd_pending_prefix_wait_ms': 30}",
            )
        ),
        encoding="utf-8",
    )
    analyzer.summarize_router(path)
    output = capsys.readouterr().out
    assert "admission_wait_requests=2 rejected_attempts_by_worker={'d0': 1}" in output
    assert "confirmation_query_failure_categories={'timeout': 1}" in output
    assert "requests_with_rejection_while_D_waiting_le_1=1" in output
    assert "low_queue_rejection_and_route_ge_1s=1" in output
    assert "d0: low_queue_rejected_requests=1" in output
    assert "d1: low_queue_rejected_requests" not in output
    assert "pd_pending_prefix_wait_ms: count=1 avg=30.00ms" in output
    assert "request=r1 target=d0 router_route_ms=2000.00ms" in output
    assert "router_capacity_wait_ms=no-data" in output
    assert "waiting=no-data" in output


def test_backpressure_classifies_only_explicit_query_errors(capsys):
    analyzer.summarize_router_backpressure(
        [
            "[PDDecodeCapacityConfirmation] request=r1 worker=d0 admitted=False "
            "reason=decode_free_block_budget error=ReadTimeout",
            "[PDDecodeCapacityConfirmation] request=r2 worker=d0 admitted=False "
            "reason=decode_capacity_confirmation_failed error=ConnectError",
            "[PDDecodeCapacityConfirmation] request=r3 worker=d1 admitted=False "
            "reason=decode_capacity_confirmation_failed error=status=503",
            "[PDDecodeCapacityConfirmation] request=r4 worker=d1 admitted=False "
            "reason=decode_capacity_confirmation_failed elapsed_ms=5000",
        ],
        {},
        {},
    )
    output = capsys.readouterr().out
    assert "{'connection': 1, 'http_status': 1, 'unclassified': 1}" in output
    assert "low_queue_rejection_and_route_ge_1s=0" in output


def test_small_delta_confirmation_error_types_and_recovery(capsys):
    lines = [
        "2026-09-17 10:00:00 [PDDecodeCapacityConfirmation] request=r1 worker=d0 "
        "admitted=False reason=decode_capacity_confirmation_failed estimated_blocks=6 "
        "available_blocks=30 margin_blocks=8 elapsed_ms=50 shared=True error=ReadTimeout initial_error=",
        "2026-09-17 10:00:01 [PDDecodeCapacityConfirmation] request=r1 worker=d0 "
        "admitted=False reason=decode_capacity_confirmation_failed estimated_blocks=6 "
        "elapsed_ms=60 shared=True error=timed out waiting for shared D Worker capacity query",
        "2026-09-17 10:00:02 [PDDecodeCapacityConfirmation] request=r1 worker=d1 admitted=True ticket_admitted=False",
        "2026-09-17 10:00:03 [PDDecodeCapacityConfirmation] request=r1 worker=d0 admitted=True",
        "[PDDecodeCapacityConfirmation] request=r2 worker=d0 admitted=False "
        "reason=decode_free_block_budget estimated_blocks=162 elapsed_ms=70",
        "[PDDecodeCapacityConfirmation] request=r3 worker=d0 admitted=False "
        "reason=decode_capacity_confirmation_failed elapsed_ms=80",
    ]
    analyzer.summarize_small_delta_confirmations(
        lines,
        {"r1": {"ttft_ms": 5000}},
        {"r1": {"router_route_ms": 4000, "target": "d0"}},
        {"r1": {"reused_blocks": 156, "transferred_blocks": 5}},
    )
    output = capsys.readouterr().out
    assert "rejected_logged_attempts=2 requests=1 request_worker_pairs=1" in output
    assert "query_failure_types={'ReadTimeout': 1, 'shared_query_wait_timeout': 1}" in output
    assert "shared_flagged_attempts=2" in output
    assert "query_failed_despite_previous_snapshot_room=1" in output
    assert "same_worker_later_acceptance=1/1" in output
    assert "first_small_rejection_to_later_acceptance: count=1 avg=3000.00ms" in output
    assert "logged_failed_elapsed_sum=110.00ms final_reused=156 final_transferred=5" in output


def test_small_delta_does_not_infer_timeouts_or_cross_worker_recovery(capsys):
    analyzer.summarize_small_delta_confirmations(
        [
            "2026-09-17 10:00:00 [PDDecodeCapacityConfirmation] request=r1 worker=d0 "
            "admitted=False reason=decode_capacity_confirmation_failed estimated_blocks=5 "
            "elapsed_ms=5000 shared=True error=unknown initial_error=ReadTimeout",
            "2026-09-17 10:00:01 [PDDecodeCapacityConfirmation] request=r1 worker=d1 admitted=True",
            "[PDDecodeCapacityConfirmation] request=r2 worker=d0 admitted=False "
            "reason=decode_free_block_budget estimated_blocks=8",
            "[PDDecodeCapacityConfirmation] request=r2 worker=d0 admitted=True",
        ],
        {},
        {},
        {},
    )
    output = capsys.readouterr().out
    assert "query_failure_types={'unclassified': 1}" in output
    assert "same_worker_later_acceptance=0/2" in output
    assert "first_small_rejection_to_later_acceptance: no data" in output
    assert "route=no-data" in output


def test_capacity_query_timing_distinguishes_retry_and_missing_worker_phases(capsys):
    analyzer.summarize_capacity_query_timings(
        [
            "[PDCapacityQueryTiming] request=r1 worker=d0 ok=True attempts=2 "
            "elapsed_ms=130 http_ms=80 control_timing={'handler_execution_ms': 70, "
            "'engine_core_proxy_ms': 60, 'engine_core_execution_ms': 10}",
            "[PDCapacityQueryTiming] request=r2 worker=d0 ok=False attempts=2 "
            "error_type=ReadTimeout failure_stage=http_request elapsed_ms=200 "
            "http_ms=unknown control_timing={}",
        ]
    )
    output = capsys.readouterr().out
    assert "logged_confirmations=2 retried=2 recovered_on_retry=1" in output
    assert "final_errors={'ReadTimeout': 1}" in output
    assert "failure_stages={'http_request': 1}" in output
    assert "http_ms: count=1 avg=80.00ms" in output
    assert "engine_core_execution_ms: count=1 avg=10.00ms" in output
    assert "engine_core_queue_and_transport_ms: no data" in output


def test_router_metrics_fill_rust_phases_by_request_without_duplicates(tmp_path, capsys):
    path = tmp_path / "router.log"
    path.write_text(
        "\n".join(
            (
                "[KVBMPDTiming] request_id=r1 pd_import_ms=300 pre_decode_total_ms=800",
                "[KVBMPDTiming] request_id=r1 pd_import_ms=300 pre_decode_total_ms=800",
                "[KVBMPDRequestSummary] request=r1 p_request_ms=500",
                "[KVBMRouterTiming] request=r1 stage=complete target=d0 "
                "phase_ms={'worker_dispatch_to_first_byte_ms': 700}",
                "[KVBMPDTiming] request_id=r2 worker_dispatch_to_first_byte_ms=900 pd_import_ms=400",
            )
        ),
        encoding="utf-8",
    )
    analyzer.summarize_router(path)
    output = capsys.readouterr().out
    assert "pd_import_ms: count=2 avg=350.00ms" in output
    assert "pre_decode_total_ms: count=1 avg=800.00ms" in output
    assert "worker_dispatch_to_first_byte_ms: count=2 avg=800.00ms" in output


def test_decode_alignment_exposes_skew_hidden_by_equal_means(capsys):
    start = datetime(2026, 9, 17, 10)

    def sample(running, waiting=0, kv=70):
        return {"running": running, "waiting": waiting, "kv_usage": kv, "gen_tps": 100}

    analyzer.summarize_decode_balance(
        {
            "d1.log": {start: sample(17, 2), start + timedelta(seconds=10): sample(2)},
            "d2.log": {start + timedelta(seconds=1): sample(2), start + timedelta(seconds=11): sample(17)},
        },
        (start, start + timedelta(seconds=20)),
    )
    output = capsys.readouterr().out
    assert "aligned_samples=2" in output
    assert "running_gap: count=2 avg=15.00 reqs" in output
    assert "skewed_samples=2/2 (100.00%)" in output
    assert "longest_skewed_streak=2 samples observed_span=10.00s" in output
    assert "queued_with_low_running=1/2" in output
    assert "d1.log@10:00:00 running=17" in output


def test_decode_missing_samples_break_streak_and_are_not_zero_filled(capsys):
    start = datetime(2026, 9, 17, 10)
    hot = {"running": 17, "waiting": 0, "kv_usage": 80, "gen_tps": 120}
    low = {"running": 2, "waiting": 0, "kv_usage": 60, "gen_tps": 50}
    analyzer.summarize_decode_balance(
        {
            "d1.log": {start + timedelta(seconds=i): hot for i in (0, 10, 20)},
            "d2.log": {start + timedelta(seconds=i): low for i in (1, 21)},
        },
        (start, start + timedelta(seconds=30)),
    )
    output = capsys.readouterr().out
    assert "aligned_samples=2 incomplete_or_misaligned=1" in output
    assert "running_gap: count=2" in output
    assert "longest_skewed_streak=1 samples observed_span=0.00s" in output


def test_decode_alignment_rejects_excess_total_timestamp_spread(capsys):
    start = datetime(2026, 9, 17, 10)
    sample = {"running": 5, "waiting": 0, "kv_usage": 60, "gen_tps": 100}
    analyzer.summarize_decode_balance(
        {
            "d1.log": {start: sample},
            "d2.log": {start - timedelta(seconds=2): sample},
            "d3.log": {start + timedelta(seconds=2): sample},
        },
        (None, None),
        tolerance=3,
    )
    output = capsys.readouterr().out
    assert "aligned_samples=0 incomplete_or_misaligned=1" in output
    assert "no complete active snapshots" in output


def test_decode_alignment_does_not_reuse_a_sample(capsys):
    start = datetime(2026, 9, 17, 10)
    sample = {"running": 5, "waiting": 0, "kv_usage": 60, "gen_tps": 100}
    analyzer.summarize_decode_balance(
        {
            "d1.log": {start: sample, start + timedelta(seconds=4): sample},
            "d2.log": {start + timedelta(seconds=2): sample},
        },
        (None, None),
        tolerance=3,
    )
    output = capsys.readouterr().out
    assert "aligned_samples=1 incomplete_or_misaligned=1" in output


def test_decode_reader_keeps_timestamps_for_balance(tmp_path, capsys):
    start = datetime(2026, 9, 17, 10)
    paths = [tmp_path / "d1.log", tmp_path / "d2.log"]
    for index, path in enumerate(paths):
        path.write_text(
            f"INFO 09-17 10:00:0{index} [loggers.py:259] Engine 000: "
            f"Avg generation throughput: 100 tokens/s, Running: {17 if index == 0 else 2} reqs, "
            "Waiting: 0 reqs, GPU KV cache usage: 70%",
            encoding="utf-8",
        )
    analyzer.summarize_decode(paths, (start, start + timedelta(seconds=20)))
    output = capsys.readouterr().out
    assert "== Decode time-aligned balance ==" in output
    assert "skewed_samples=1/1 (100.00%)" in output
