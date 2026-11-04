from unittest.mock import Mock, patch

import pytest

from openjiuwentools.infer_router.kv_cache.event import CacheLocation
from openjiuwentools.infer_router.routing.scorer import (
    CacheLocationScorer,
    InflightScorer,
    PluginScorerAlgorithm,
    ScoringContext,
)
from openjiuwentools.infer_router.routing.load_balance import PrefixAwareAlgorithm
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint, WorkerInfo, WorkerType


def _route_hint(precomputed_overlap_scores=None, precomputed_prefix_snapshots=None):
    return RouteHint(
        priority=0,
        estimated_output_tokens=64,
        next_turn_prefill=False,
        request_id="request-1",
        model="test-model",
        token_ids=list(range(16)),
        precomputed_overlap_scores=precomputed_overlap_scores,
        precomputed_prefix_snapshots=precomputed_prefix_snapshots,
    )


def _workers():
    prefill = [
        WorkerInfo(worker_id="p0", model="test-model", worker_type=WorkerType.PREFILL),
        WorkerInfo(worker_id="p1", model="test-model", worker_type=WorkerType.PREFILL),
    ]
    decode = [
        WorkerInfo(worker_id="d0", model="test-model", worker_type=WorkerType.DECODE, queue_depth=0),
        WorkerInfo(worker_id="d1", model="test-model", worker_type=WorkerType.DECODE, queue_depth=1),
    ]
    return prefill, decode


def _algorithm(kv_cache_manager):
    return PluginScorerAlgorithm(
        kv_cache_manager,
        plugin_configs=[{"type": "prefix-cache-scorer", "weight": 1.0}],
        prefill_plugin_configs=[{"type": "prefix-cache-scorer", "weight": 1.0}],
        decode_plugin_configs=[{"type": "queue-scorer", "weight": 1.0}],
    )


def test_pd_reuses_precomputed_overlap_and_skips_decode_match():
    kv_cache_manager = Mock()
    algorithm = _algorithm(kv_cache_manager)
    prefill, decode = _workers()

    selected_prefill, selected_decode = algorithm.select_worker_pair(
        prefill,
        decode,
        _route_hint({"p0": 16, "p1": 0}),
    )

    assert selected_prefill.worker_id == "p0"
    assert selected_decode.worker_id == "d0"
    kv_cache_manager.find_matches.assert_not_called()


def test_pd_logs_request_scoped_scorer_details():
    kv_cache_manager = Mock()
    prefill, decode = _workers()
    excluded_decode = WorkerInfo(
        worker_id="d2",
        model="test-model",
        worker_type=WorkerType.DECODE,
        current_load=95.0,
        queue_depth=42,
        running_requests=22,
    )
    worker_manager = Mock()
    worker_manager.get_all_healthy_prefill_workers.return_value = prefill
    worker_manager.get_all_healthy_decode_workers.return_value = [
        *decode,
        excluded_decode,
    ]
    algorithm = PluginScorerAlgorithm(
        kv_cache_manager,
        worker_manager,
        plugin_configs=[{"type": "prefix-cache-scorer", "weight": 1.0}],
        prefill_plugin_configs=[{"type": "prefix-cache-scorer", "weight": 1.0}],
        decode_plugin_configs=[{"type": "queue-scorer", "weight": 1.0}],
    )
    inflight_tracker = Mock()

    def worker_inflight(worker_id):
        return {"p0": 1, "p1": 2, "d0": 3, "d1": 4, "d2": 69}[worker_id]

    inflight_tracker.get.side_effect = worker_inflight
    algorithm.set_inflight_tracker(inflight_tracker)

    with patch("openjiuwentools.infer_router.routing.scorer.logger.info") as info:
        algorithm.select_worker_pair(
            prefill,
            decode,
            _route_hint({"p0": 16, "p1": 0}),
        )

    detail_calls = [
        log_call
        for log_call in info.call_args_list
        if log_call.args and log_call.args[0].startswith("[PluginScorerDetail]")
    ]
    assert len(detail_calls) == 2
    assert detail_calls[0].args[1:4] == ("request-1", "prefill", True)
    assert detail_calls[1].args[1:4] == ("request-1", "decode", False)
    assert "p0(selected=True,total=1.000" in detail_calls[0].args[4]
    assert "prefix-cache-scorer:1.000*1.0=1.000" in detail_calls[0].args[4]
    assert "d0(selected=True,total=1.000" in detail_calls[1].args[4]
    assert "queue-scorer:1.000*1.0=1.000" in detail_calls[1].args[4]
    assert "d2(eligible=false,load=95.000,queue=42,running=22,inflight=69)" in detail_calls[1].args[5]


def test_pd_cold_prefill_prefers_minimum_inflight_before_lru_score():
    """With no prefix hit, current Router inflight must bound cold LRU spreading."""
    kv_cache_manager = Mock()
    prefill, decode = _workers()
    algorithm = PluginScorerAlgorithm(
        kv_cache_manager,
        prefill_plugin_configs=[{"type": "no-hit-lru-scorer", "weight": 100.0}],
        decode_plugin_configs=[{"type": "queue-scorer", "weight": 1.0}],
    )
    inflight_tracker = Mock()

    def worker_inflight(worker_id):
        return {"p0": 0, "p1": 3, "d0": 0, "d1": 0}[worker_id]

    inflight_tracker.get.side_effect = worker_inflight
    algorithm.set_inflight_tracker(inflight_tracker)

    selected_prefill, _ = algorithm.select_worker_pair(
        prefill,
        decode,
        _route_hint({"p0": 0, "p1": 0}),
    )

    assert selected_prefill.worker_id == "p0"


def test_pd_prefill_hit_is_not_overridden_by_cold_load_guard():
    """A real prefix hit keeps its configured score even at higher inflight."""
    kv_cache_manager = Mock()
    prefill, decode = _workers()
    algorithm = _algorithm(kv_cache_manager)
    inflight_tracker = Mock()

    def worker_inflight(worker_id):
        return {"p0": 10, "p1": 0, "d0": 0, "d1": 0}[worker_id]

    inflight_tracker.get.side_effect = worker_inflight
    algorithm.set_inflight_tracker(inflight_tracker)

    selected_prefill, _ = algorithm.select_worker_pair(
        prefill,
        decode,
        _route_hint({"p0": 16, "p1": 0}),
    )

    assert selected_prefill.worker_id == "p0"


def test_legacy_prefix_aware_reuses_precomputed_overlap():
    """内置 PrefixAware worker-pair 不应重复执行 data-plane KV 匹配。"""

    kv_cache_manager = Mock()
    algorithm = PrefixAwareAlgorithm(kv_cache_manager)
    prefill, decode = _workers()

    selected_prefill, selected_decode = algorithm.select_worker_pair(
        prefill,
        decode,
        _route_hint({"p0": 16, "p1": 0, "d0": 0, "d1": 0}),
    )

    assert selected_prefill.worker_id == "p0"
    assert selected_decode.worker_id in {"d0", "d1"}
    kv_cache_manager.find_matches.assert_not_called()


def test_pd_computes_overlap_only_for_prefill_when_not_precomputed():
    kv_cache_manager = Mock()
    kv_cache_manager.find_matches.return_value = {"p0": 16, "p1": 0}
    algorithm = _algorithm(kv_cache_manager)
    prefill, decode = _workers()

    selected_prefill, selected_decode = algorithm.select_worker_pair(
        prefill,
        decode,
        _route_hint(),
    )

    assert selected_prefill.worker_id == "p0"
    assert selected_decode.worker_id == "d0"
    assert kv_cache_manager.find_matches.call_count == 1


def test_cache_location_scorer_applies_configured_tier_weights():
    """location scorer 应按 NPU、RAM、DISK 权重折算命中 token 数。"""
    scorer = CacheLocationScorer(parameters={"npu_weight": 1.0, "ram_weight": 0.5, "disk_weight": 0.1})
    worker = WorkerInfo(worker_id="p0", model="test-model", worker_type=WorkerType.PREFILL)
    context = ScoringContext(
        overlap_scores={},
        total_tokens=100,
        any_hit=True,
        worker_ids=["p0"],
        location_token_counts={
            "p0": {
                CacheLocation.NPU: 40,
                CacheLocation.CPU: 20,
                CacheLocation.REMOTE: 10,
            }
        },
    )

    assert scorer.score(worker, [worker], _route_hint(), context) == 0.51


def test_inflight_scorer_uses_smooth_absolute_score():
    scorer = InflightScorer(parameters={"scale": 8})
    workers = [
        WorkerInfo(worker_id="d0", model="test-model", worker_type=WorkerType.DECODE),
        WorkerInfo(worker_id="d1", model="test-model", worker_type=WorkerType.DECODE),
    ]
    context = ScoringContext(
        overlap_scores={},
        total_tokens=16,
        any_hit=False,
        worker_ids=[worker.worker_id for worker in workers],
        inflight_counts={"d0": 23, "d1": 24},
    )

    score_23 = scorer.score(workers[0], workers, _route_hint(), context)
    score_24 = scorer.score(workers[1], workers, _route_hint(), context)

    assert score_23 == pytest.approx(8 / 31)
    assert score_24 == pytest.approx(0.25)
    assert score_23 - score_24 < 0.01


def test_cache_location_scorer_selects_npu_hit_over_ram_hit():
    """前缀长度相同时，独立 location scorer 应优先选择 NPU 命中 Worker。"""
    kv_cache_manager = Mock()

    def worker_cache_locations(worker_id, _token_ids):
        return {CacheLocation.CPU: 16} if worker_id == "p0" else {CacheLocation.NPU: 16}

    kv_cache_manager.get_matched_block_location_token_counts.side_effect = worker_cache_locations
    algorithm = PluginScorerAlgorithm(
        kv_cache_manager,
        plugin_configs=[
            {
                "type": "cache-location-scorer",
                "weight": 1.0,
                "parameters": {
                    "npu_weight": 1.0,
                    "ram_weight": 0.6,
                    "disk_weight": 0.2,
                },
            }
        ],
    )
    workers, _decode = _workers()

    selected = algorithm.select_worker(workers, _route_hint())

    assert selected.worker_id == "p1"
    kv_cache_manager.find_matches.assert_not_called()
    assert kv_cache_manager.get_matched_block_location_token_counts.call_count == 2


def test_cache_location_scorer_reuses_request_prefix_snapshots():
    """location scorer 应传递首次 KV 匹配生成的 Worker 快照。"""

    kv_cache_manager = Mock()
    snapshots = {"p0": object(), "p1": object()}
    kv_cache_manager.get_matched_block_location_token_counts.side_effect = [
        {CacheLocation.CPU: 16},
        {CacheLocation.NPU: 16},
    ]
    algorithm = PluginScorerAlgorithm(
        kv_cache_manager,
        plugin_configs=[{"type": "cache-location-scorer", "weight": 1.0}],
    )
    workers, _decode = _workers()

    algorithm.select_worker(
        workers,
        _route_hint(precomputed_prefix_snapshots=snapshots),
    )

    observed = {
        call.args[0]: call.kwargs.get("snapshot")
        for call in kv_cache_manager.get_matched_block_location_token_counts.call_args_list
    }
    assert observed == snapshots
