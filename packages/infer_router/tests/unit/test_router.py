# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
"""测试路由模块"""

import threading
import time
from unittest.mock import Mock, call, patch

import pytest

from openjiuwentools.infer_router.config.config import Settings
from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.routing import router as router_module
from openjiuwentools.infer_router.routing.load_balance import RoundRobinAlgorithm
from openjiuwentools.infer_router.routing.router import (
    PendingPrefixAffinityTracker,
    Router,
)
from openjiuwentools.infer_router.routing.scorer import PluginScorerAlgorithm
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint, WorkerInfo, WorkerType
from openjiuwentools.infer_router.worker.worker_manager import WorkerManager


@pytest.fixture
def kvcache_manager():
    """创建KV缓存管理器mock"""
    mock = Mock(spec=KVCacheManager)
    mock.find_matches.return_value = {}
    return mock


@pytest.fixture
def worker_manager():
    """创建工作器管理器mock"""
    manager = Mock(spec=WorkerManager)
    manager.has_disaggregated_workers.return_value = False
    return manager


@pytest.fixture
def router(kvcache_manager, worker_manager):
    """创建路由器实例"""
    return Router(kvcache_manager, worker_manager)


@pytest.fixture
def route_hint():
    """创建路由提示"""
    return RouteHint(
        priority=5,
        estimated_output_tokens=100,
        next_turn_prefill=False,
        request_id="test-1",
        model="test-model",
        token_ids=[72, 101, 108, 108, 111],
    )


def test_phase_algorithms_support_prefix_prefill_and_round_robin_decode(
    kvcache_manager,
    worker_manager,
    monkeypatch,
):
    prefill_workers = [
        WorkerInfo(
            worker_id="prefill-1",
            model="test-model",
            worker_type=WorkerType.PREFILL,
        ),
        WorkerInfo(
            worker_id="prefill-2",
            model="test-model",
            worker_type=WorkerType.PREFILL,
        ),
    ]
    decode_workers = [
        WorkerInfo(
            worker_id="decode-1",
            model="test-model",
            worker_type=WorkerType.DECODE,
        ),
        WorkerInfo(
            worker_id="decode-2",
            model="test-model",
            worker_type=WorkerType.DECODE,
        ),
    ]
    worker_manager.get_all_healthy_prefill_workers.return_value = prefill_workers
    worker_manager.get_all_healthy_decode_workers.return_value = decode_workers
    monkeypatch.setattr(
        router_module.settings,
        "prefill_load_balancing_algorithm",
        "prefix_aware",
    )
    monkeypatch.setattr(
        router_module.settings,
        "decode_load_balancing_algorithm",
        "round_robin",
    )
    monkeypatch.setattr(
        router_module.settings,
        "prefill_scoring_plugins",
        [{"type": "prefix-cache-scorer", "weight": 1.0}],
    )
    phase_router = Router(kvcache_manager, worker_manager)

    assert isinstance(phase_router.prefill_algorithm, PluginScorerAlgorithm)
    assert isinstance(phase_router.decode_algorithm, RoundRobinAlgorithm)
    hint = RouteHint(
        priority=5,
        estimated_output_tokens=100,
        next_turn_prefill=False,
        request_id="phase-test",
        model="test-model",
        token_ids=list(range(16)),
        precomputed_overlap_scores={
            "prefill-1": 0,
            "prefill-2": 16,
        },
    )

    selections = [
        phase_router._select_pd_worker_pair(
            prefill_workers,
            decode_workers,
            hint,
        )
        for _ in range(3)
    ]

    assert [prefill.worker_id for prefill, _ in selections] == [
        "prefill-2",
        "prefill-2",
        "prefill-2",
    ]
    assert [decode.worker_id for _, decode in selections] == [
        "decode-1",
        "decode-2",
        "decode-1",
    ]


def test_phase_algorithms_default_to_legacy_pair_selector(
    kvcache_manager,
    worker_manager,
    route_hint,
    monkeypatch,
):
    monkeypatch.setattr(
        router_module.settings,
        "prefill_load_balancing_algorithm",
        None,
    )
    monkeypatch.setattr(
        router_module.settings,
        "decode_load_balancing_algorithm",
        None,
    )
    legacy_router = Router(kvcache_manager, worker_manager)
    legacy_router.current_algorithm.select_worker_pair = Mock(return_value=("prefill", "decode"))

    selected = legacy_router._select_pd_worker_pair(
        ["prefill"],
        ["decode"],
        route_hint,
    )

    assert selected == ("prefill", "decode")
    legacy_router.current_algorithm.select_worker_pair.assert_called_once()


@pytest.mark.asyncio
async def test_router_route_single_worker(router):
    """测试单个工作器的情况（组合型worker）"""
    router.kvcache_manager.find_matches.return_value = {"worker-1": 0.5}
    router.kvcache_manager.generate_events.return_value = []

    worker = WorkerInfo(
        worker_id="worker-1",
        model="test-model",
        available_memory=1000000,
        current_load=10,
        cached_prefixes=[],
        url="http://localhost:8001/v1",
        worker_type=WorkerType.COMBINED,
        group="group-single",
    )
    route_hint = RouteHint(
        priority=5,
        estimated_output_tokens=100,
        next_turn_prefill=False,
        request_id="test-1",
        model="test-model",
        token_ids=[72, 101, 108, 108, 111],
    )
    router.worker_manager.get_all_healthy_prefill_workers.return_value = []
    router.worker_manager.get_all_healthy_decode_workers.return_value = []
    router.worker_manager.get_all_healthy_combined_workers.return_value = [worker]

    prefill_id, decode_id = router.route_to_workers(route_hint)

    assert prefill_id == "worker-1"
    assert decode_id is None


@pytest.fixture
def workers_with_groups_for_router():
    """创建用于路由测试的带group的工作器列表"""
    return [
        WorkerInfo(
            worker_id="prefill-1",
            model="test-model",
            url="http://localhost:8001/v1",
            available_memory=1000000,
            current_load=10,
            cached_prefixes=["prefix-1"],
            worker_type=WorkerType.PREFILL,
            group="group-a",
        ),
        WorkerInfo(
            worker_id="decode-1",
            model="test-model",
            url="http://localhost:8002/v1",
            available_memory=800000,
            current_load=5,
            cached_prefixes=["prefix-2"],
            worker_type=WorkerType.DECODE,
            group="group-a",
        ),
        WorkerInfo(
            worker_id="combined-1",
            model="test-model",
            url="http://localhost:8003/v1",
            available_memory=1200000,
            current_load=3,
            cached_prefixes=["prefix-3"],
            worker_type=WorkerType.COMBINED,
            group="group-b",
        ),
        WorkerInfo(
            worker_id="combined-2",
            model="test-model",
            url="http://localhost:8004/v1",
            available_memory=1500000,
            current_load=8,
            cached_prefixes=["prefix-4"],
            worker_type=WorkerType.COMBINED,
            group="group-b",
        ),
    ]


@pytest.mark.asyncio
async def test_router_route_to_workers_returns_pair(router, route_hint, workers_with_groups_for_router):
    """测试路由到worker对"""
    router.kvcache_manager.find_matches.return_value = {
        "prefill-1": 0.5,
        "decode-1": 0.3,
    }

    prefill_workers = [w for w in workers_with_groups_for_router if w.worker_type == WorkerType.PREFILL]
    decode_workers = [w for w in workers_with_groups_for_router if w.worker_type == WorkerType.DECODE]

    router.worker_manager.get_all_healthy_prefill_workers.return_value = prefill_workers
    router.worker_manager.get_all_healthy_decode_workers.return_value = decode_workers
    router.worker_manager.is_combined_group.return_value = False

    prefill_id, decode_id = router.route_to_workers(route_hint)

    assert prefill_id is not None
    assert decode_id is not None


@pytest.mark.asyncio
async def test_router_route_to_workers_returns_combined(router, route_hint, workers_with_groups_for_router):
    """测试路由到组合型worker"""
    router.kvcache_manager.find_matches.return_value = {
        "combined-1": 0.8,
        "combined-2": 0.4,
    }

    router.worker_manager.get_all_healthy_prefill_workers.return_value = []
    router.worker_manager.get_all_healthy_decode_workers.return_value = []

    combined_workers = [w for w in workers_with_groups_for_router if w.worker_type == WorkerType.COMBINED]
    router.worker_manager.get_all_healthy_combined_workers.return_value = combined_workers

    prefill_id, decode_id = router.route_to_workers(route_hint)

    assert prefill_id is not None


@pytest.mark.asyncio
async def test_router_route_to_workers_records_metrics(router, route_hint, workers_with_groups_for_router):
    """测试路由到worker对时记录指标"""

    router.kvcache_manager.find_matches.return_value = {
        "prefill-1": 0.5,
        "decode-1": 0.3,
    }

    prefill_workers = [w for w in workers_with_groups_for_router if w.worker_type == WorkerType.PREFILL]
    decode_workers = [w for w in workers_with_groups_for_router if w.worker_type == WorkerType.DECODE]

    router.worker_manager.get_all_healthy_prefill_workers.return_value = prefill_workers
    router.worker_manager.get_all_healthy_decode_workers.return_value = decode_workers
    router.worker_manager.is_combined_group.return_value = False

    def _find_worker(worker_id):
        return next((w for w in workers_with_groups_for_router if w.worker_id == worker_id), None)

    router.worker_manager.get_worker.side_effect = _find_worker

    with patch("openjiuwentools.infer_router.routing.router.metrics") as mock_metrics:
        prefill_id, decode_id = router.route_to_workers(route_hint)

        assert prefill_id is not None
        assert decode_id is not None

        assert mock_metrics.record_routed_request.call_count >= 2
        mock_metrics.record_routed_request.assert_any_call("prefill-1", "test-model")
        mock_metrics.record_routed_request.assert_any_call("decode-1", "test-model")


def test_route_decision_snapshot_reuses_precomputed_overlap(router, route_hint):
    """路由诊断日志不能为已选 Worker 再次扫描全部 KV RadixTree。"""

    worker = WorkerInfo(
        worker_id="prefill-1",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    router.worker_manager.get_worker.return_value = worker
    route_hint.precomputed_overlap_scores = {"prefill-1": 4}

    snapshot = router._worker_route_snapshot("prefill-1", route_hint)

    assert snapshot["overlap_tokens"] == 4
    router.kvcache_manager.find_matches.assert_not_called()


def test_pending_prefix_affinity_waits_for_hixl_attach():
    """同一公共前缀应等待首个 HIXL import 完成后再复用目标 D。"""
    tracker = PendingPrefixAffinityTracker(ttl_seconds=10.0)
    shared = list(range(2048))
    leader_tokens = shared + list(range(2048, 2560))
    follower_tokens = shared + list(range(9000, 9512))

    assert tracker.register(
        request_id="leader",
        model="test-model",
        worker_id="decode-0",
        token_ids=leader_tokens,
        block_size=128,
        min_tokens=1024,
        min_ratio=0.8,
    )
    assert (
        tracker.wait_for(
            request_id="follower",
            model="test-model",
            token_ids=follower_tokens,
            min_tokens=1024,
            min_ratio=0.8,
            timeout=0.0,
        )
        is None
    )

    tracker.mark_ready("leader", "decode-0", published_blocks=16)
    match = tracker.wait_for(
        request_id="follower",
        model="test-model",
        token_ids=follower_tokens,
        min_tokens=1024,
        min_ratio=0.8,
        timeout=0.0,
    )

    assert match is not None
    assert match.worker_id == "decode-0"
    assert match.state == "ready"
    assert match.matched_tokens == 2048


def test_pending_prefix_affinity_cleanup_releases_waiters():
    """PD fallback或请求完成后，pending prefix必须可重新登记。"""
    tracker = PendingPrefixAffinityTracker(ttl_seconds=10.0)
    tokens = list(range(2048))
    kwargs = {
        "model": "test-model",
        "worker_id": "decode-0",
        "token_ids": tokens,
        "block_size": 128,
        "min_tokens": 1024,
        "min_ratio": 0.8,
    }

    assert tracker.register(request_id="failed", **kwargs)
    tracker.remove("failed", "decode-0")
    assert tracker.register(request_id="replacement", **kwargs)


def test_pending_prefix_affinity_can_replicate_away_from_rejected_worker():
    """容量不足的 affinity D 不应阻止在其他 D 创建同前缀副本。"""
    tracker = PendingPrefixAffinityTracker(ttl_seconds=10.0)
    tokens = list(range(2048))
    assert tracker.register(
        request_id="leader",
        model="test-model",
        worker_id="decode-hot",
        token_ids=tokens,
        block_size=128,
        min_tokens=1024,
        min_ratio=0.8,
    )
    tracker.mark_ready("leader", "decode-hot", published_blocks=16)

    registered, existing = tracker.claim(
        request_id="replica",
        model="test-model",
        worker_id="decode-cool",
        token_ids=tokens,
        block_size=128,
        min_tokens=1024,
        min_ratio=0.8,
        excluded_worker_ids={"decode-hot"},
    )

    assert registered is True
    assert existing is None


def test_pending_prefix_affinity_hashes_once_and_uses_coarse_index():
    """同block size的多条entry只哈希一次输入并只比较索引候选。"""
    tracker = PendingPrefixAffinityTracker(ttl_seconds=10.0)
    entries = [[offset + token for token in range(2048)] for offset in (0, 10000, 20000)]
    for index, token_ids in enumerate(entries):
        assert tracker.register(
            request_id=f"leader-{index}",
            model="test-model",
            worker_id=f"decode-{index}",
            token_ids=token_ids,
            block_size=128,
            min_tokens=1024,
            min_ratio=0.8,
        )

    with (
        patch.object(
            tracker,
            "_fingerprints",
            wraps=tracker._fingerprints,
        ) as fingerprints,
        patch.object(
            tracker,
            "_common_prefix_blocks",
            wraps=tracker._common_prefix_blocks,
        ) as comparisons,
    ):
        match = tracker.find(
            request_id="follower",
            model="test-model",
            token_ids=entries[0],
            min_tokens=1024,
            min_ratio=0.8,
        )

    assert match is not None
    assert match.request_id == "leader-0"
    assert fingerprints.call_count == 1
    assert comparisons.call_count == 1


def test_pd_pending_prefix_claim_precedes_worker_budget_commit(router):
    """同前缀follower不得重复提交P/D inflight或block预算。"""
    prefill = WorkerInfo(
        worker_id="prefill-pending",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-pending",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    shared = list(range(2048))
    leader = RouteHint(
        priority=5,
        estimated_output_tokens=128,
        next_turn_prefill=False,
        request_id="leader",
        model="test-model",
        token_ids=shared + list(range(2048, 2304)),
        precomputed_overlap_scores={decode.worker_id: 0},
    )
    follower = leader.model_copy(
        update={
            "request_id": "follower",
            "token_ids": shared + list(range(9000, 9256)),
        }
    )
    router._pd_pending_prefix_min_tokens = 1024
    router._pd_pending_prefix_min_ratio = 0.8
    router.kvcache_manager.get_worker_block_size.return_value = 128
    prefill_details = {"estimated_tokens": 2304, "estimated_blocks": 18}
    decode_details = {
        "estimated_blocks": 19,
        "steady_state_blocks": 1,
        "full_estimated_blocks": 19,
    }

    with router._routing_lock:
        assert (
            router._commit_pd_worker_pair_locked(
                prefill,
                decode,
                leader,
                prefill_details,
                decode_details,
            )
            is None
        )
        pending = router._commit_pd_worker_pair_locked(
            prefill,
            decode,
            follower,
            prefill_details,
            decode_details,
        )

    assert pending is not None
    assert pending.request_id == leader.request_id
    assert router.inflight.get(prefill.worker_id) == 1
    assert router.inflight.get(decode.worker_id) == 1
    assert router.pd_prefill_budget.get(prefill.worker_id) == (2304, 18)
    assert router.pd_decode_blocks.get_blocks(decode.worker_id) == 19


@pytest.mark.parametrize(
    "populations, expected",
    [
        ([(17, 0, 17), (3, 0, 3)], {"decode-0", "decode-1"}),
        ([(5, 0, 5), (3, 0, 3)], {"decode-0", "decode-1"}),
        ([(1, 0, 17), (3, 0, 3)], {"decode-0", "decode-1"}),
        ([(3, 4, 7), (5, 0, 5)], {"decode-1"}),
        ([(20, 0, 20), (24, 0, 24)], {"decode-0", "decode-1"}),
        ([(17, 1, 18), (3, 0, 3)], {"decode-0", "decode-1"}),
        ([(17, 4, 21), (3, 0, 3)], {"decode-0", "decode-1"}),
        ([(6, 4, 10), (6, 3, 9)], {"decode-0", "decode-1"}),
    ],
)
def test_pd_decode_spill_requires_queue_and_relief(router, route_hint, populations, expected):
    router._requires_decode_capacity_confirmation = Mock(return_value=True)
    workers = []
    for index, (running, waiting, inflight) in enumerate(populations):
        worker = WorkerInfo(
            worker_id=f"decode-{index}",
            model=route_hint.model,
            worker_type=WorkerType.DECODE,
            running_requests=running,
            queue_depth=waiting,
        )
        workers.append(worker)
        for _ in range(inflight):
            router.inflight.increment(worker.worker_id)
    candidates, excluded = router._queue_relief_pd_decode_workers(workers, route_hint)
    assert {worker.worker_id for worker in candidates} == expected
    assert excluded == {worker.worker_id for worker in workers} - expected


@pytest.fixture
def load_balanced_pd(router, route_hint):
    prefill = WorkerInfo(
        worker_id="prefill",
        model=route_hint.model,
        worker_type=WorkerType.PREFILL,
    )
    hot = WorkerInfo(
        worker_id="hot",
        model=route_hint.model,
        worker_type=WorkerType.DECODE,
        running_requests=17,
        queue_depth=10,
    )
    idle = WorkerInfo(
        worker_id="idle",
        model=route_hint.model,
        worker_type=WorkerType.DECODE,
        running_requests=3,
    )
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [hot, idle]
    router.worker_manager.get_all_workers.return_value = {worker.worker_id: worker for worker in (prefill, hot, idle)}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._prefill_admission_decision = Mock(return_value=(True, {"estimated_tokens": 2304, "estimated_blocks": 18}))
    router._decode_admission_decision = Mock(return_value=(True, {"estimated_blocks": 19, "steady_state_blocks": 1}))
    router._needs_decode_capacity_confirmation = Mock(return_value=False)
    router._requires_decode_capacity_confirmation = Mock(return_value=True)
    router._confirm_decode_capacity = Mock(return_value=(True, {"admission_ticket_id": "confirmed-ticket"}))
    router.current_algorithm.select_worker_pair = Mock(
        side_effect=lambda **kwargs: (kwargs["prefill_workers"][0], kwargs["decode_workers"][0])
    )
    route_hint.token_ids = list(range(2304))
    router._pd_pending_prefix_min_tokens = 1024
    router._pd_pending_prefix_min_ratio = 0.8
    return prefill, hot, idle


@pytest.mark.parametrize("pending", [False, True])
def test_pd_queued_prefix_affinity_spill_requires_confirmation(router, route_hint, load_balanced_pd, pending):
    prefill, hot, idle = load_balanced_pd
    if pending:
        assert router._pd_pending_prefix_affinity.register(
            request_id="leader",
            model=route_hint.model,
            worker_id=hot.worker_id,
            token_ids=route_hint.token_ids,
            block_size=128,
            min_tokens=1024,
            min_ratio=0.8,
        )
    else:
        route_hint.pd_decode_affinity_worker_id = hot.worker_id
        route_hint.precomputed_overlap_scores = {hot.worker_id: 2048}
    assert router.find_best_worker_pair(route_hint.model, route_hint) == (prefill.worker_id, idle.worker_id)
    assert router.inflight.get(hot.worker_id) == 0
    assert router.inflight.get(idle.worker_id) == 1
    assert router.pd_decode_blocks.get_blocks(idle.worker_id) == 19
    assert router.current_algorithm.select_worker_pair.call_args.kwargs["decode_workers"] == [idle]
    router._confirm_decode_capacity.assert_called_once()
    assert router._confirm_decode_capacity.call_args.args[0] == idle


def test_pd_close_load_keeps_prefix_affinity(router, route_hint, load_balanced_pd):
    prefill, hot, idle = load_balanced_pd
    hot.running_requests = 5
    hot.queue_depth = 0
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    assert router.find_best_worker_pair(route_hint.model, route_hint) == (prefill.worker_id, hot.worker_id)


def test_pd_large_running_gap_without_queue_preserves_reuse(router, route_hint, load_balanced_pd):
    prefill, hot, _ = load_balanced_pd
    hot.queue_depth = 0
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    assert router.find_best_worker_pair(route_hint.model, route_hint) == (prefill.worker_id, hot.worker_id)
    router._confirm_decode_capacity.assert_not_called()


def test_pd_hotspot_spill_requires_recent_slow_tpot_and_confirms_capacity(
    router,
    route_hint,
    load_balanced_pd,
):
    prefill, hot, idle = load_balanced_pd
    hot.queue_depth = 0
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    route_hint.precomputed_overlap_scores = {hot.worker_id: 2048}
    router._pd_hotspot_spill_enabled = True
    for _ in range(4):
        router.record_decode_tpot(hot.worker_id, 90.0)
        router.record_decode_tpot(idle.worker_id, 50.0)

    assert router.find_best_worker_pair(route_hint.model, route_hint) == (
        prefill.worker_id,
        idle.worker_id,
    )
    router._confirm_decode_capacity.assert_called_once()
    assert router._confirm_decode_capacity.call_args.args[0] == idle
    assert hot.worker_id in router._pd_hotspot_last_spill_at


def test_pd_hotspot_spill_thresholds_load_from_router_yaml(
    tmp_path,
    kvcache_manager,
    worker_manager,
):
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        "router_pd_decode_hotspot_spill_hot_min_running: 18\n"
        "router_pd_decode_hotspot_spill_cold_max_running: 6\n"
        "router_pd_decode_hotspot_spill_running_gap: 99\n"
        "router_pd_decode_hotspot_spill_tpot_ratio: 1.4\n",
        encoding="utf-8",
    )
    with patch.object(router_module, "settings", Settings.from_yaml(str(config_path))):
        configured_router = Router(kvcache_manager, worker_manager)
    assert configured_router._pd_hotspot_hot_min_running == 18
    assert configured_router._pd_hotspot_cold_max_running == 6
    assert configured_router._pd_hotspot_running_gap is None
    assert configured_router._pd_hotspot_tpot_ratio == 1.4


def test_pd_hotspot_spill_gap_only_and_defaults_load_from_router_yaml(
    tmp_path,
    kvcache_manager,
    worker_manager,
):
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        "router_pd_decode_hotspot_spill_running_gap: 12\n",
        encoding="utf-8",
    )
    with patch.object(router_module, "settings", Settings.from_yaml(str(config_path))):
        gap_router = Router(kvcache_manager, worker_manager)
    assert gap_router._pd_hotspot_hot_min_running is None
    assert gap_router._pd_hotspot_cold_max_running is None
    assert gap_router._pd_hotspot_running_gap == 12

    with patch.object(router_module, "settings", Settings()):
        default_router = Router(kvcache_manager, worker_manager)
    assert default_router._pd_hotspot_hot_min_running == 16
    assert default_router._pd_hotspot_cold_max_running == 8
    assert default_router._pd_hotspot_running_gap is None


def test_pd_hotspot_spill_uses_configured_thresholds(
    router,
    route_hint,
    load_balanced_pd,
):
    _, hot, idle = load_balanced_pd
    hot.queue_depth = 0
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    route_hint.precomputed_overlap_scores = {hot.worker_id: 2048}
    router._pd_hotspot_spill_enabled = True
    for _ in range(4):
        router.record_decode_tpot(hot.worker_id, 90.0)
        router.record_decode_tpot(idle.worker_id, 50.0)

    router._pd_hotspot_hot_min_running = 18
    router._pd_hotspot_cold_max_running = None
    router._pd_hotspot_running_gap = None
    candidates, source = router._hotspot_spill_pd_decode_workers([hot, idle], route_hint)
    assert candidates == [hot, idle]
    assert source is None

    router._pd_hotspot_hot_min_running = None
    router._pd_hotspot_cold_max_running = 2
    candidates, source = router._hotspot_spill_pd_decode_workers([hot, idle], route_hint)
    assert candidates == [hot, idle]
    assert source is None

    router._pd_hotspot_cold_max_running = None
    router._pd_hotspot_running_gap = 15
    candidates, source = router._hotspot_spill_pd_decode_workers([hot, idle], route_hint)
    assert candidates == [hot, idle]
    assert source is None

    router._pd_hotspot_running_gap = 14
    router._pd_hotspot_tpot_ratio = 2.0
    candidates, source = router._hotspot_spill_pd_decode_workers([hot, idle], route_hint)
    assert candidates == [hot, idle]
    assert source is None

    router._pd_hotspot_tpot_ratio = 1.5
    candidates, source = router._hotspot_spill_pd_decode_workers([hot, idle], route_hint)
    assert candidates == [idle]
    assert source == hot.worker_id

    router._pd_hotspot_hot_min_running = 17
    router._pd_hotspot_cold_max_running = 3
    router._pd_hotspot_running_gap = None
    candidates, source = router._hotspot_spill_pd_decode_workers([hot, idle], route_hint)
    assert candidates == [idle]
    assert source == hot.worker_id


def test_pd_hotspot_spill_gap_only_does_not_require_default_running_bounds(
    router,
    route_hint,
    load_balanced_pd,
):
    _, hot, idle = load_balanced_pd
    hot.running_requests = 10
    hot.queue_depth = 0
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    route_hint.precomputed_overlap_scores = {hot.worker_id: 2048}
    router._pd_hotspot_spill_enabled = True
    router._pd_hotspot_hot_min_running = None
    router._pd_hotspot_cold_max_running = None
    router._pd_hotspot_running_gap = 7
    for _ in range(4):
        router.record_decode_tpot(hot.worker_id, 90.0)
        router.record_decode_tpot(idle.worker_id, 50.0)

    candidates, source = router._hotspot_spill_pd_decode_workers([hot, idle], route_hint)
    assert candidates == [idle]
    assert source == hot.worker_id


def test_pd_hotspot_spill_cooldown_keeps_affinity(router, route_hint, load_balanced_pd):
    prefill, hot, idle = load_balanced_pd
    hot.queue_depth = 0
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    route_hint.precomputed_overlap_scores = {hot.worker_id: 2048}
    router._pd_hotspot_spill_enabled = True
    for _ in range(4):
        router.record_decode_tpot(hot.worker_id, 90.0)
        router.record_decode_tpot(idle.worker_id, 50.0)
    router._pd_hotspot_last_spill_at[hot.worker_id] = time.monotonic()

    assert router.find_best_worker_pair(route_hint.model, route_hint) == (
        prefill.worker_id,
        hot.worker_id,
    )
    router._confirm_decode_capacity.assert_not_called()


def test_pd_hotspot_spill_without_peer_tpot_keeps_affinity(
    router,
    route_hint,
    load_balanced_pd,
):
    prefill, hot, _ = load_balanced_pd
    hot.queue_depth = 0
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    route_hint.precomputed_overlap_scores = {hot.worker_id: 2048}
    router._pd_hotspot_spill_enabled = True
    for _ in range(4):
        router.record_decode_tpot(hot.worker_id, 90.0)

    assert router.find_best_worker_pair(route_hint.model, route_hint) == (
        prefill.worker_id,
        hot.worker_id,
    )
    router._confirm_decode_capacity.assert_not_called()


def test_pd_hotspot_spill_failed_confirmation_keeps_affinity(
    router,
    route_hint,
    load_balanced_pd,
):
    prefill, hot, idle = load_balanced_pd
    hot.queue_depth = 0
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    route_hint.precomputed_overlap_scores = {hot.worker_id: 2048}
    router._pd_hotspot_spill_enabled = True
    for _ in range(4):
        router.record_decode_tpot(hot.worker_id, 90.0)
        router.record_decode_tpot(idle.worker_id, 50.0)
    router._confirm_decode_capacity.return_value = (
        False,
        {"reason": "decode_free_block_budget", "capacity_confirmed": True},
    )

    with patch.object(router, "_release_pd_decode_admission_ticket"):
        assert router.find_best_worker_pair(route_hint.model, route_hint) == (
            prefill.worker_id,
            hot.worker_id,
        )
    assert hot.worker_id not in router._pd_hotspot_last_spill_at


def test_pd_spill_disabled_without_authoritative_confirmation(router, route_hint, load_balanced_pd):
    prefill, hot, _ = load_balanced_pd
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    router._requires_decode_capacity_confirmation.return_value = False
    assert router.find_best_worker_pair(route_hint.model, route_hint) == (prefill.worker_id, hot.worker_id)
    router._confirm_decode_capacity.assert_not_called()


def test_pd_failed_spill_confirmation_falls_back_to_reuse(router, route_hint, load_balanced_pd):
    prefill, hot, idle = load_balanced_pd
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    route_hint.precomputed_overlap_scores = {hot.worker_id: 2048}

    def confirm(worker, *_args):
        assert worker.worker_id == idle.worker_id
        return False, {
            "reason": "decode_free_block_budget",
            "estimated_blocks": 19,
            "free_blocks": 4,
            "available_blocks": 4,
            "capacity_confirmed": True,
        }

    router._confirm_decode_capacity.side_effect = confirm
    with patch.object(router, "_release_pd_decode_admission_ticket"):
        assert router.find_best_worker_pair(route_hint.model, route_hint) == (prefill.worker_id, hot.worker_id)
    assert router.inflight.get(idle.worker_id) == 0
    assert router.pd_decode_blocks.get_blocks(idle.worker_id) == 0
    assert router.inflight.get(hot.worker_id) == 1


def test_pd_close_load_still_coalesces_pending_prefix(router, route_hint, load_balanced_pd):
    _, hot, _ = load_balanced_pd
    hot.running_requests = 5
    hot.queue_depth = 0
    assert router._pd_pending_prefix_affinity.register(
        request_id="leader",
        model=route_hint.model,
        worker_id=hot.worker_id,
        token_ids=route_hint.token_ids,
        block_size=128,
        min_tokens=1024,
        min_ratio=0.8,
    )
    assert router.find_best_worker_pair(route_hint.model, route_hint) == (None, None)
    router.current_algorithm.select_worker_pair.assert_not_called()
    assert router.inflight.get(hot.worker_id) == 0


def test_pd_busy_leader_created_during_confirmation_does_not_recapture_follower(
    router,
    route_hint,
    load_balanced_pd,
):
    prefill, hot, idle = load_balanced_pd
    router._needs_decode_capacity_confirmation.return_value = True

    def confirm(worker, *_args):
        assert worker.worker_id == idle.worker_id
        assert router._pd_pending_prefix_affinity.register(
            request_id="leader",
            model=route_hint.model,
            worker_id=hot.worker_id,
            token_ids=route_hint.token_ids,
            block_size=128,
            min_tokens=1024,
            min_ratio=0.8,
        )
        return True, {"admission_ticket_id": "idle-ticket"}

    router._confirm_decode_capacity = Mock(side_effect=confirm)
    with patch.object(router, "_release_pd_decode_admission_ticket") as release:
        assert router.find_best_worker_pair(route_hint.model, route_hint) == (prefill.worker_id, idle.worker_id)
    release.assert_not_called()
    assert router.inflight.get(idle.worker_id) == 1
    assert router.pd_decode_blocks.get_blocks(idle.worker_id) == 19


def test_pd_load_band_ignores_capacity_rejected_idle_worker(router, route_hint, load_balanced_pd):
    prefill, hot, idle = load_balanced_pd

    def worker_inflight5(worker, *_args, **_kwargs):
        return (
            (True, {"estimated_blocks": 19, "steady_state_blocks": 1})
            if worker.worker_id == hot.worker_id
            else (False, {"reason": "worker_admission"})
        )

    router._decode_admission_decision.side_effect = worker_inflight5
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    assert router.find_best_worker_pair(route_hint.model, route_hint) == (prefill.worker_id, hot.worker_id)


def test_pd_queue_changed_during_confirmation_releases_ticket(router, route_hint, load_balanced_pd):
    prefill, hot, idle = load_balanced_pd
    hot.running_requests = 4
    hot.queue_depth = 0
    route_hint.pd_decode_affinity_worker_id = hot.worker_id
    router._needs_decode_capacity_confirmation.return_value = True

    def confirm(worker, *_args):
        if worker.worker_id == hot.worker_id:
            hot.running_requests = 17
            hot.queue_depth = 10
        return True, {"admission_ticket_id": f"ticket-{worker.worker_id}"}

    router._confirm_decode_capacity = Mock(side_effect=confirm)
    with patch.object(router, "_release_pd_decode_admission_ticket") as release:
        assert router.find_best_worker_pair(route_hint.model, route_hint) == (prefill.worker_id, idle.worker_id)
    release.assert_called_once_with(hot.worker_id, route_hint.request_id, "ticket-hot", background=True)
    assert router.inflight.get(hot.worker_id) == 0
    assert router.pd_decode_blocks.get_blocks(hot.worker_id) == 0


def test_pd_pending_prefix_race_releases_uncommitted_capacity_ticket(
    router,
    route_hint,
):
    """容量确认期间出现leader时应放弃提交并释放当前D ticket。"""
    prefill = WorkerInfo(
        worker_id="prefill-confirm-race",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-confirm-race",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=0,
        kvbm_block_pool_updated_at=time.monotonic() - 60.0,
    )
    route_hint.token_ids = list(range(2304))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router._pd_pending_prefix_min_tokens = 1024
    router._pd_pending_prefix_min_ratio = 0.8
    router._pd_decode_admission_mode = "auto"
    router._pd_decode_block_admission_enabled = True
    router._pd_decode_capacity_confirmation_enabled = True
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }
    router.current_algorithm.select_worker_pair = Mock(return_value=(prefill, decode))

    def confirm_capacity(*_args, **_kwargs):
        assert router._pd_pending_prefix_affinity.register(
            request_id="concurrent-leader",
            model=route_hint.model,
            worker_id=decode.worker_id,
            token_ids=list(route_hint.token_ids),
            block_size=128,
            min_tokens=1024,
            min_ratio=0.8,
        )
        return {
            "ok": True,
            "free_blocks": 200,
            "admission_ticket_id": "race-ticket",
            "elapsed_ms": 1.0,
        }

    router.worker_manager.confirm_kvbm_block_pool_capacity.side_effect = confirm_capacity

    with patch.object(router, "_release_pd_decode_admission_ticket") as release:
        selected = router.find_best_worker_pair("test-model", route_hint)

    assert selected == (None, None)
    assert router.inflight.get(prefill.worker_id) == 0
    assert router.inflight.get(decode.worker_id) == 0
    assert router.pd_prefill_budget.get(prefill.worker_id) == (0, 0)
    assert router.pd_decode_blocks.get_blocks(decode.worker_id) == 0
    release.assert_called_once_with(
        decode.worker_id,
        route_hint.request_id,
        "race-ticket",
        background=True,
    )


def test_pd_pending_prefix_affinity_requires_actual_overlap_ratio(router):
    """正式Router KV索引命中比例不足时不得强制绑定leader的D。"""
    route_hint = RouteHint(
        priority=5,
        estimated_output_tokens=128,
        next_turn_prefill=False,
        request_id="follower",
        model="test-model",
        token_ids=list(range(2048)),
    )
    router._pd_pending_prefix_min_tokens = 1024
    router._pd_pending_prefix_min_ratio = 0.8
    router.kvcache_manager.find_matches.return_value = {"decode-0": 1024}
    assert router._pd_pending_prefix_affinity.register(
        request_id="leader",
        model=route_hint.model,
        worker_id="decode-0",
        token_ids=list(route_hint.token_ids),
        block_size=128,
        min_tokens=1024,
        min_ratio=0.8,
    )
    router._pd_pending_prefix_affinity.mark_ready(
        "leader",
        "decode-0",
        published_blocks=16,
    )
    match = router._pd_pending_prefix_affinity.wait_for(
        request_id=route_hint.request_id,
        model=route_hint.model,
        token_ids=list(route_hint.token_ids),
        min_tokens=1024,
        min_ratio=0.8,
        timeout=0.0,
    )
    assert match is not None

    assert router.refresh_pd_prefix_affinity(route_hint, match) is False
    assert route_hint.pd_decode_affinity_worker_id is None
    assert (
        router._pd_pending_prefix_affinity.wait_for(
            request_id=route_hint.request_id,
            model=route_hint.model,
            token_ids=list(route_hint.token_ids),
            min_tokens=1024,
            min_ratio=0.8,
            timeout=0.0,
        )
        is None
    )


@pytest.mark.asyncio
async def test_router_route_to_workers_no_workers(router, route_hint):
    """测试无可用worker的情况"""
    router.worker_manager.get_all_healthy_prefill_workers.return_value = []
    router.worker_manager.get_all_healthy_decode_workers.return_value = []
    router.worker_manager.get_all_healthy_combined_workers.return_value = []

    prefill_id, decode_id = router.route_to_workers(route_hint)

    assert prefill_id is None
    assert decode_id is None


@pytest.mark.asyncio
async def test_router_pd_mode_does_not_warn_when_combined_pool_is_empty(router, route_hint):
    """PD 分离部署不应为不存在的 COMBINED 池打印告警。"""

    router.worker_manager.get_all_healthy_prefill_workers.return_value = []
    router.worker_manager.get_all_healthy_decode_workers.return_value = []
    router.worker_manager.get_all_healthy_combined_workers.return_value = []
    router.worker_manager.has_disaggregated_workers.return_value = True

    with (
        patch("openjiuwentools.infer_router.routing.router.logger.warning") as warning,
        patch("openjiuwentools.infer_router.routing.router.logger.error") as error,
    ):
        prefill_id, decode_id = router.route_to_workers(route_hint)

    assert prefill_id is None
    assert decode_id is None
    assert not any(call.args and call.args[0] == "No valid combined workers found" for call in warning.call_args_list)
    error.assert_not_called()


@pytest.mark.asyncio
async def test_router_find_best_worker_pair(router, route_hint):
    """测试查找最佳worker对"""
    router.kvcache_manager.find_matches.return_value = {
        "prefill-high-load": 0.2,
        "prefill-low-load": 0.8,
        "decode-high-load": 0.1,
        "decode-low-load": 0.9,
    }

    prefill_workers = [
        WorkerInfo(
            worker_id="prefill-high-load",
            model="test-model",
            url="http://localhost:8001/v1",
            available_memory=500000,
            current_load=80,
            cached_prefixes=[],
            worker_type=WorkerType.PREFILL,
            group="group-a",
        ),
        WorkerInfo(
            worker_id="prefill-low-load",
            model="test-model",
            url="http://localhost:8002/v1",
            available_memory=1500000,
            current_load=10,
            cached_prefixes=[],
            worker_type=WorkerType.PREFILL,
            group="group-a",
        ),
    ]

    decode_workers = [
        WorkerInfo(
            worker_id="decode-high-load",
            model="test-model",
            url="http://localhost:8003/v1",
            available_memory=400000,
            current_load=70,
            cached_prefixes=[],
            worker_type=WorkerType.DECODE,
            group="group-a",
        ),
        WorkerInfo(
            worker_id="decode-low-load",
            model="test-model",
            url="http://localhost:8004/v1",
            available_memory=1200000,
            current_load=5,
            cached_prefixes=[],
            worker_type=WorkerType.DECODE,
            group="group-a",
        ),
    ]

    router.worker_manager.get_all_healthy_prefill_workers.return_value = prefill_workers
    router.worker_manager.get_all_healthy_decode_workers.return_value = decode_workers
    router.worker_manager.is_combined_group.return_value = False

    prefill_id, decode_id = router.find_best_worker_pair("test-model", route_hint)

    assert prefill_id is not None
    assert decode_id is not None


def test_worker_admission_snapshot_identifies_decode_blocker(router):
    """PD 准入快照应明确指出无可用 D Worker。"""
    prefill = WorkerInfo(
        worker_id="prefill-0",
        model="test-model",
        url="http://localhost:8001/v1",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-0",
        model="test-model",
        url="http://localhost:8002/v1",
        worker_type=WorkerType.DECODE,
    )
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router._is_worker_admitted = Mock(side_effect=lambda worker: worker is prefill)

    snapshot = router.get_worker_admission_snapshot("test-model")

    summary_fields = (
        "blocker",
        "prefill_admitted",
        "prefill_healthy",
        "prefill_configured",
        "decode_admitted",
        "decode_healthy",
        "decode_configured",
    )
    assert {key: snapshot[key] for key in summary_fields} == {
        "blocker": "decode",
        "prefill_admitted": 1,
        "prefill_healthy": 1,
        "prefill_configured": 1,
        "decode_admitted": 0,
        "decode_healthy": 1,
        "decode_configured": 1,
    }
    assert snapshot["decode_details"]["decode-0"]["reason"] == ("worker_admission")


def test_worker_admission_snapshot_uses_current_request_block_budget(
    router,
    route_hint,
):
    """排队超时快照不得以空请求把容量不足的D计为可准入。"""
    prefill = WorkerInfo(
        worker_id="prefill-request-budget",
        model="test-model",
        worker_type=WorkerType.PREFILL,
        num_gpu_blocks=1000,
    )
    decode = WorkerInfo(
        worker_id="decode-request-budget",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=80,
        kvbm_block_pool_updated_at=time.monotonic(),
    )
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router.kvcache_manager.get_worker_block_size.return_value = 128
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {
        prefill.worker_id: 5888,
        decode.worker_id: 5888,
    }
    router.pd_decode_blocks.reserve(
        "existing-request",
        decode.worker_id,
        30,
        0,
    )

    snapshot = router.get_worker_admission_snapshot("test-model", route_hint)

    assert snapshot["blocker"] == "decode"
    assert snapshot["prefill_admitted"] == 1
    assert snapshot["decode_admitted"] == 0


def test_pd_decode_admission_rejects_worker_at_inflight_limit(router, route_hint):
    """D Worker达到PD请求上限后必须退出候选集合。"""
    decode = WorkerInfo(
        worker_id="decode-full",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    router._pd_decode_max_inflight = 2
    router.inflight.increment(decode.worker_id)
    router.inflight.increment(decode.worker_id)
    router.kvcache_manager.get_worker_block_size.return_value = 128

    admitted, diagnostics = router._decode_admission_decision(decode, route_hint)

    assert admitted is False
    assert diagnostics["reason"] == "decode_inflight_limit"
    assert diagnostics["inflight"] == 2


def test_pd_decode_admission_rejects_worker_without_block_headroom(router, route_hint):
    """D Worker实时KV占用超过安全预算时不得继续接收PD请求。"""
    decode = WorkerInfo(
        worker_id="decode-block-full",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        active_decode_blocks=843,
        kv_cache_usage=0.843,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 512
    route_hint.precomputed_overlap_scores = {decode.worker_id: 5888}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._pd_decode_block_headroom_ratio = 0.15

    admitted, diagnostics = router._decode_admission_decision(decode, route_hint)

    assert admitted is False
    assert diagnostics["reason"] == "decode_block_budget"
    assert diagnostics["estimated_blocks"] == 54
    assert diagnostics["steady_state_blocks"] == 2
    assert diagnostics["projected_blocks"] == 897
    assert diagnostics["usable_blocks"] == 850


def test_pd_decode_admission_prefers_fresh_engine_core_free_blocks(
    router,
    route_hint,
):
    """真实空闲block不足时应在Router排队，不得把请求送入D gate。"""
    decode = WorkerInfo(
        worker_id="decode-real-capacity",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        active_decode_blocks=100,
        kvbm_free_blocks=21,
        kvbm_block_pool_updated_at=time.monotonic(),
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128

    admitted, diagnostics = router._decode_admission_decision(decode, route_hint)

    assert admitted is False
    assert diagnostics["reason"] == "decode_free_block_budget"
    assert diagnostics["estimated_blocks"] == 51
    assert diagnostics["free_blocks"] == 21
    assert diagnostics["capacity_source"] == "engine_core"


def test_pd_decode_admission_uses_npu_hint_then_confirms_reuse(
    router,
    route_hint,
):
    """完整预算不足时应确认D真实复用并只预留缺失blocks。"""
    prefill = WorkerInfo(
        worker_id="prefill-delta-admission",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-delta-admission",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=10,
        kvbm_block_pool_updated_at=time.monotonic(),
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    blocks = []
    expected_hashes = []
    for index in range(46):
        block_hash = f"hash-{index}"
        expected_hashes.append(block_hash)
        alias = Mock(block_hash=block_hash)
        block = Mock(token_end=(index + 1) * 128)
        block.select.return_value = alias
        blocks.append(block)
    route_hint.precomputed_prefix_snapshots = {decode.worker_id: Mock(blocks=tuple(blocks))}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }
    router.current_algorithm.select_worker_pair = Mock(return_value=(prefill, decode))
    router._pd_decode_capacity_confirmation_enabled = True

    initial_admitted, initial = router._decode_admission_decision(
        decode,
        route_hint,
    )
    assert initial_admitted is True
    assert initial["estimated_blocks"] == 5
    assert initial["candidate_reusable_prompt_blocks"] == 46
    assert initial["provisional_reusable_prompt_blocks"] == 46
    assert initial["confirmed_reusable_prompt_blocks"] == 0
    assert router._needs_decode_capacity_confirmation(initial_admitted, initial) is True

    def confirm_capacity(worker_id, **kwargs):
        assert worker_id == decode.worker_id
        assert kwargs["block_hashes"] == expected_hashes
        assert kwargs["prompt_blocks"] == 50
        assert kwargs["decode_blocks"] == 1
        decode.kvbm_free_blocks = 10
        decode.kvbm_block_pool_updated_at = time.monotonic()
        return {
            "ok": True,
            "free_blocks": 10,
            "confirmed_reusable_blocks": 46,
            "required_new_blocks": 5,
            "admission_ticket_id": "ticket-1",
            "admission_ticket_pinned_blocks": 46,
            "elapsed_ms": 1.0,
        }

    router.worker_manager.confirm_kvbm_block_pool_capacity.side_effect = confirm_capacity

    assert router.find_best_worker_pair("test-model", route_hint) == (
        prefill.worker_id,
        decode.worker_id,
    )
    assert router.pd_decode_blocks.get_blocks(decode.worker_id) == 5
    assert (
        router.take_pd_decode_admission_ticket(
            route_hint.request_id,
            decode.worker_id,
        )
        == "ticket-1"
    )


def test_pd_decode_fresh_full_budget_fast_path_skips_capacity_http(
    router,
    route_hint,
):
    """新鲜容量足够完整预算时应直接原子预占，不增加同步HTTP开销。"""
    prefill = WorkerInfo(
        worker_id="prefill-full-fast-path",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-full-fast-path",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=200,
        kvbm_block_pool_updated_at=time.monotonic(),
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }
    router.current_algorithm.select_worker_pair = Mock(return_value=(prefill, decode))
    router._pd_decode_capacity_confirmation_enabled = True

    assert router.find_best_worker_pair("test-model", route_hint) == (
        prefill.worker_id,
        decode.worker_id,
    )
    router.worker_manager.confirm_kvbm_block_pool_capacity.assert_not_called()
    assert router.pd_decode_blocks.get_blocks(decode.worker_id) == 51


def test_pd_decode_admission_falls_back_when_engine_core_metric_is_stale(
    router,
    route_hint,
):
    """真实容量指标过期后应回退旧估算，避免Worker永久退出候选集。"""
    decode = WorkerInfo(
        worker_id="decode-stale-capacity",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        active_decode_blocks=100,
        kvbm_free_blocks=0,
        kvbm_block_pool_updated_at=time.monotonic() - 60.0,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128

    admitted, diagnostics = router._decode_admission_decision(decode, route_hint)

    assert admitted is True
    assert diagnostics["free_blocks"] == -1
    assert diagnostics["capacity_source"] == "estimated"


def test_pd_decode_admission_deducts_reservations_after_capacity_sample(
    router,
    route_hint,
):
    """同一容量采样后新准入的请求预算必须防止后续请求并发穿透。"""
    decode = WorkerInfo(
        worker_id="decode-capacity-race",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=200,
        kvbm_block_pool_updated_at=time.monotonic(),
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128

    first_admitted, first_diagnostics = router._decode_admission_decision(
        decode,
        route_hint,
    )
    router.pd_decode_blocks.reserve(
        "request-after-sample",
        decode.worker_id,
        first_diagnostics["estimated_blocks"],
        0,
    )
    second_admitted, second_diagnostics = router._decode_admission_decision(
        decode,
        route_hint,
    )

    assert first_admitted is True
    assert second_admitted is True
    assert second_diagnostics["pending_reserved_blocks"] == 51
    assert second_diagnostics["available_blocks"] == 149
    assert second_diagnostics["capacity_headroom_blocks"] == 0
    assert second_diagnostics["fallback_headroom_blocks"] == 100


def test_pd_decode_final_capacity_confirmation_rejects_stale_selection(
    router,
    route_hint,
):
    """最终查询发现D容量下降时应拒绝旧快照选中结果。"""
    decode = WorkerInfo(
        worker_id="decode-final-confirmation",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=200,
        kvbm_block_pool_updated_at=time.monotonic() - 60.0,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._pd_decode_capacity_confirmation_enabled = True

    initially_admitted, initial = router._decode_admission_decision(
        decode,
        route_hint,
    )

    def confirm_capacity(worker_id, *, timeout_ms, **kwargs):
        assert worker_id == decode.worker_id
        assert timeout_ms > 0
        decode.kvbm_free_blocks = 39
        decode.kvbm_block_pool_updated_at = time.monotonic()
        return {"ok": True, "free_blocks": 39, "elapsed_ms": 1.25}

    router.worker_manager.confirm_kvbm_block_pool_capacity.side_effect = confirm_capacity
    finally_admitted, final = router._confirm_decode_capacity(
        decode,
        route_hint,
        initial,
    )

    assert initially_admitted is True
    assert initial["capacity_source"] == "estimated"
    assert finally_admitted is False
    assert final["reason"] == "decode_free_block_budget"
    assert final["free_blocks"] == 39
    assert final["available_blocks"] == 39
    assert final["estimated_blocks"] == 51
    assert final["capacity_confirmed"] is True
    assert final["capacity_confirmation_ms"] == 1.25


def test_pd_decode_stale_estimate_is_confirmed_before_rejection(
    router,
    route_hint,
):
    """过期容量导致估算拒绝时应查询真实BlockPool后再决定准入。"""
    prefill = WorkerInfo(
        worker_id="prefill-stale-confirmation",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-stale-confirmation",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        active_decode_blocks=900,
        kvbm_free_blocks=0,
        kvbm_block_pool_updated_at=time.monotonic() - 60.0,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }
    router.current_algorithm.select_worker_pair = Mock(return_value=(prefill, decode))
    router._pd_decode_capacity_confirmation_enabled = True
    router._pd_decode_capacity_confirmation_timeout_ms = 50
    router._pd_decode_stale_capacity_confirmation_timeout_ms = 150

    initially_admitted, initial = router._decode_admission_decision(
        decode,
        route_hint,
    )
    assert initially_admitted is False
    assert initial["reason"] == "decode_block_budget"

    def confirm_capacity(worker_id, *, timeout_ms, **kwargs):
        assert worker_id == decode.worker_id
        assert timeout_ms == 150
        decode.kvbm_free_blocks = 200
        decode.kvbm_block_pool_updated_at = time.monotonic()
        return {"ok": True, "free_blocks": 200, "elapsed_ms": 1.0}

    router.worker_manager.confirm_kvbm_block_pool_capacity.side_effect = confirm_capacity

    assert router.find_best_worker_pair("test-model", route_hint) == (
        prefill.worker_id,
        decode.worker_id,
    )
    router.worker_manager.confirm_kvbm_block_pool_capacity.assert_called_once()


def test_pd_decode_stale_confirmation_failure_excludes_worker_once(
    router,
    route_hint,
):
    """实时容量复核失败后应排除Worker且不得重复查询。"""
    prefill = WorkerInfo(
        worker_id="prefill-stale-failed",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-stale-failed",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        active_decode_blocks=900,
        kvbm_free_blocks=0,
        kvbm_block_pool_updated_at=time.monotonic() - 60.0,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }
    router.current_algorithm.select_worker_pair = Mock(return_value=(prefill, decode))
    router._pd_decode_capacity_confirmation_enabled = True
    router.worker_manager.confirm_kvbm_block_pool_capacity.return_value = {
        "ok": False,
        "error": "capacity endpoint timed out",
        "elapsed_ms": 150.0,
    }

    assert router.find_best_worker_pair("test-model", route_hint) == (None, None)
    assert router.find_best_worker_pair("test-model", route_hint) == (None, None)
    router.worker_manager.confirm_kvbm_block_pool_capacity.assert_called_once_with(
        decode.worker_id,
        timeout_ms=router._pd_decode_stale_capacity_confirmation_timeout_ms,
        request_id=route_hint.request_id,
        handoff_id=f"pd-{route_hint.request_id}",
        margin_blocks=2,
        pending_reserved_blocks=0,
    )


def test_pd_decode_cached_rejection_allows_affinity_fallback(router, route_hint):
    """Affinity D 被当前容量代际拒绝后应允许改选其他 D。"""
    prefill = WorkerInfo(
        worker_id="prefill-affinity-fallback",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    affinity_decode = WorkerInfo(
        worker_id="decode-affinity-full",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=20,
        kvbm_block_pool_updated_at=time.monotonic(),
        kvbm_capacity_generation=7,
    )
    fallback_decode = WorkerInfo(
        worker_id="decode-affinity-fallback",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=500,
        kvbm_block_pool_updated_at=time.monotonic(),
        kvbm_capacity_generation=9,
    )
    route_hint.token_ids = list(range(128))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {
        affinity_decode.worker_id: 128,
        fallback_decode.worker_id: 0,
    }
    route_hint.pd_decode_affinity_worker_id = affinity_decode.worker_id
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [
        affinity_decode,
        fallback_decode,
    ]
    router.worker_manager.get_all_workers.return_value = {
        worker.worker_id: worker for worker in (prefill, affinity_decode, fallback_decode)
    }
    router.current_algorithm.select_worker_pair = Mock(
        side_effect=lambda *, prefill_workers, decode_workers, route_hint: (prefill_workers[0], decode_workers[0])
    )
    router._record_pd_decode_confirmation_rejection(
        route_hint.request_id,
        affinity_decode,
        {
            "reason": "decode_free_block_budget",
            "capacity_confirmed": True,
            "capacity_confirmation_generation": 7,
            "free_blocks": 20,
        },
        affinity_worker_id=affinity_decode.worker_id,
    )

    assert router.find_best_worker_pair("test-model", route_hint) == (None, None)
    router.current_algorithm.select_worker_pair.assert_not_called()
    snapshot = router.get_worker_admission_snapshot("test-model", route_hint)
    assert snapshot["decode_admitted"] == 0
    assert snapshot["decode_details"][fallback_decode.worker_id]["reason"] == ("decode_affinity_grace_wait")

    with router._pd_decode_confirmation_rejection_lock:
        router._pd_decode_affinity_rejected_at[route_hint.request_id][affinity_decode.worker_id] -= 1.0

    selected = router.find_best_worker_pair("test-model", route_hint)

    assert selected == (prefill.worker_id, fallback_decode.worker_id)
    candidates = router.current_algorithm.select_worker_pair.call_args.kwargs["decode_workers"]
    assert [worker.worker_id for worker in candidates] == [fallback_decode.worker_id]


def test_pd_decode_cached_rejection_revalidates_on_generation_change(router):
    """有效代际变化时，即使free阈值未跨也应重新执行一次权威复核。"""
    decode = WorkerInfo(
        worker_id="decode-rejection-generation",
        model="test-model",
        worker_type=WorkerType.DECODE,
        kvbm_free_blocks=20,
        kvbm_capacity_generation=11,
    )
    router._record_pd_decode_confirmation_rejection(
        "request-generation",
        decode,
        {
            "reason": "decode_free_block_budget",
            "capacity_confirmed": True,
            "capacity_confirmation_generation": 11,
            "free_blocks": 20,
            "available_blocks": 20,
            "estimated_blocks": 51,
            "dynamic_margin_blocks": 8,
        },
    )
    with router._pd_decode_confirmation_rejection_lock:
        entry = router._pd_decode_confirmation_rejections["request-generation"][decode.worker_id]
        assert entry.retry_at is None
        assert entry.failure_count == 0
        assert entry.probe_at is None
        assert entry.diagnostics["capacity_confirmation_wait_class"] == ("threshold_wait")

    assert router._pd_decode_confirmation_retry_delay("request-generation") is None

    cached = router._get_pd_decode_confirmation_rejections(
        "request-generation",
        [decode],
    )
    assert set(cached) == {decode.worker_id}

    decode.kvbm_capacity_generation = 12
    decode.kvbm_free_blocks = 30

    cached = router._get_pd_decode_confirmation_rejections(
        "request-generation",
        [decode],
    )
    assert cached == {}


def test_pd_decode_reuse_refresh_probe_releases_without_metrics(router):
    """候选复用尚未确认时，即使容量样本不变也应低频权威复核。"""
    decode = WorkerInfo(
        worker_id="decode-probe-release",
        model="test-model",
        worker_type=WorkerType.DECODE,
        kvbm_free_blocks=20,
        kvbm_capacity_generation=11,
    )
    router._record_pd_decode_confirmation_rejection(
        "request-probe-release",
        decode,
        {
            "reason": "decode_free_block_budget",
            "capacity_confirmed": True,
            "capacity_confirmation_generation": 11,
            "free_blocks": 20,
            "available_blocks": 20,
            "estimated_blocks": 51,
            "dynamic_margin_blocks": 8,
            "candidate_reusable_prompt_blocks": 46,
            "confirmed_reusable_prompt_blocks": 0,
        },
    )

    # 异步指标从未更新：free/generation 保持拒绝时的值，阈值不释放。
    assert set(
        router._get_pd_decode_confirmation_rejections(
            "request-probe-release",
            [decode],
        )
    ) == {decode.worker_id}

    # 复核到期：不再阻塞该 worker（允许重发确认 HTTP），条目保留以
    # 便下一次权威拒绝刷新基线。
    with router._pd_decode_confirmation_rejection_lock:
        router._pd_decode_confirmation_rejections["request-probe-release"][decode.worker_id].probe_at = (
            time.monotonic() - 0.001
        )

    assert (
        router._get_pd_decode_confirmation_rejections(
            "request-probe-release",
            [decode],
        )
        == {}
    )
    with router._pd_decode_confirmation_rejection_lock:
        assert decode.worker_id in router._pd_decode_confirmation_rejections["request-probe-release"]

    # 再次权威拒绝：基线刷新并获得新的复核时间点。
    router._record_pd_decode_confirmation_rejection(
        "request-probe-release",
        decode,
        {
            "reason": "decode_free_block_budget",
            "capacity_confirmed": True,
            "capacity_confirmation_generation": 11,
            "free_blocks": 25,
            "available_blocks": 25,
            "estimated_blocks": 51,
            "dynamic_margin_blocks": 8,
            "candidate_reusable_prompt_blocks": 46,
            "confirmed_reusable_prompt_blocks": 0,
        },
    )
    with router._pd_decode_confirmation_rejection_lock:
        refreshed = router._pd_decode_confirmation_rejections["request-probe-release"][decode.worker_id]
        assert refreshed.probe_at is not None
        assert refreshed.probe_at > time.monotonic()
        assert refreshed.free_blocks == 25
        assert refreshed.diagnostics["capacity_confirmation_wait_class"] == ("reuse_refresh_probe")
    assert set(
        router._get_pd_decode_confirmation_rejections(
            "request-probe-release",
            [decode],
        )
    ) == {decode.worker_id}


def test_pd_decode_generationless_definitive_rejection_probes(router):
    """无generation的确定性拒绝按generationless_probe分类并安排复核。"""
    decode = WorkerInfo(
        worker_id="decode-generationless-probe",
        model="test-model",
        worker_type=WorkerType.DECODE,
        kvbm_free_blocks=20,
        kvbm_capacity_generation=-1,
    )
    router._record_pd_decode_confirmation_rejection(
        "request-generationless-probe",
        decode,
        {
            "reason": "decode_free_block_budget",
            "capacity_confirmed": True,
            "capacity_confirmation_generation": -1,
            "free_blocks": 20,
            "available_blocks": 20,
            "estimated_blocks": 51,
            "dynamic_margin_blocks": 8,
        },
    )
    with router._pd_decode_confirmation_rejection_lock:
        entry = router._pd_decode_confirmation_rejections["request-generationless-probe"][decode.worker_id]
        assert entry.retry_at is None
        assert entry.probe_at is not None
        assert entry.diagnostics["capacity_confirmation_wait_class"] == ("generationless_probe")

    # generation 未变化时继续复用旧拒绝，直到复核到期。
    assert set(
        router._get_pd_decode_confirmation_rejections(
            "request-generationless-probe",
            [decode],
        )
    ) == {decode.worker_id}


def test_pd_decode_transient_confirmation_failure_uses_exponential_backoff(
    router,
):
    """连续瞬时确认失败应保留计数并逐步延长重试间隔。"""
    decode = WorkerInfo(
        worker_id="decode-transient-backoff",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    diagnostics = {
        "reason": "decode_capacity_confirmation_failed",
        "capacity_confirmation_error": "timeout",
    }

    router._record_pd_decode_confirmation_rejection(
        "request-transient-backoff",
        decode,
        diagnostics,
    )
    with router._pd_decode_confirmation_rejection_lock:
        first = router._pd_decode_confirmation_rejections["request-transient-backoff"][decode.worker_id]
        assert first.failure_count == 1
        assert first.retry_at is not None
        assert first.probe_at is None
        assert first.diagnostics["capacity_confirmation_retry_delay_ms"] == 250.0
        first.retry_at = time.monotonic() - 1.0

    assert (
        router._get_pd_decode_confirmation_rejections(
            "request-transient-backoff",
            [decode],
        )
        == {}
    )
    with router._pd_decode_confirmation_rejection_lock:
        assert decode.worker_id in router._pd_decode_confirmation_rejections["request-transient-backoff"]

    router._record_pd_decode_confirmation_rejection(
        "request-transient-backoff",
        decode,
        diagnostics,
    )
    with router._pd_decode_confirmation_rejection_lock:
        second = router._pd_decode_confirmation_rejections["request-transient-backoff"][decode.worker_id]
        assert second.failure_count == 2
        assert second.diagnostics["capacity_confirmation_retry_delay_ms"] == 500.0
        assert 0.40 <= second.retry_at - time.monotonic() <= 0.50


def test_pd_decode_stale_shared_failure_retries_with_long_timeout(
    router,
    route_hint,
):
    """stale请求复用短超时查询失败后应以长超时复核一次。"""
    decode = WorkerInfo(
        worker_id="decode-stale-shared-retry",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        active_decode_blocks=900,
        kvbm_free_blocks=0,
        kvbm_block_pool_updated_at=time.monotonic() - 60.0,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._pd_decode_capacity_confirmation_enabled = True
    router._pd_decode_capacity_confirmation_timeout_ms = 50
    router._pd_decode_stale_capacity_confirmation_timeout_ms = 150
    initial_admitted, initial = router._decode_admission_decision(
        decode,
        route_hint,
    )
    assert initial_admitted is False
    assert initial["capacity_source"] == "estimated"

    def confirm_capacity(worker_id, *, timeout_ms, **kwargs):
        assert worker_id == decode.worker_id
        assert timeout_ms == 150
        if router.worker_manager.confirm_kvbm_block_pool_capacity.call_count == 1:
            return {
                "ok": False,
                "error": "shared 50ms leader timed out",
                "elapsed_ms": 55.0,
                "singleflight_shared": True,
            }
        decode.kvbm_free_blocks = 200
        decode.kvbm_block_pool_updated_at = time.monotonic()
        return {
            "ok": True,
            "free_blocks": 200,
            "capacity_generation": 0,
            "elapsed_ms": 80.0,
            "singleflight_shared": False,
        }

    router.worker_manager.confirm_kvbm_block_pool_capacity.side_effect = confirm_capacity

    admitted, diagnostics = router._confirm_decode_capacity(
        decode,
        route_hint,
        initial,
    )

    assert admitted is True
    assert diagnostics["capacity_confirmed"] is True
    assert diagnostics["capacity_confirmation_attempts"] == 2
    assert diagnostics["capacity_confirmation_retry_count"] == 1
    assert diagnostics["capacity_confirmation_ms"] == 135.0
    assert diagnostics["capacity_confirmation_final_attempt_ms"] == 80.0
    assert diagnostics["capacity_confirmation_generation"] == 0
    assert diagnostics["capacity_confirmation_initial_error"] == ("shared 50ms leader timed out")
    assert router.worker_manager.confirm_kvbm_block_pool_capacity.call_args_list == [
        call(
            decode.worker_id,
            timeout_ms=150,
            request_id=route_hint.request_id,
            handoff_id=f"pd-{route_hint.request_id}",
            margin_blocks=2,
            pending_reserved_blocks=0,
        ),
        call(
            decode.worker_id,
            timeout_ms=150,
            request_id=route_hint.request_id,
            handoff_id=f"pd-{route_hint.request_id}",
            margin_blocks=2,
            pending_reserved_blocks=0,
        ),
    ]


@pytest.mark.parametrize("final_ok,capacity_ok", [(True, True), (True, False), (False, False)])
def test_small_delta_timeout_has_one_bounded_retry(router, route_hint, final_ok, capacity_ok):
    decode = WorkerInfo(worker_id="decode-small-retry", model="test-model", worker_type=WorkerType.DECODE)
    router._pd_decode_capacity_confirmation_enabled = True
    router._pd_decode_capacity_confirmation_timeout_ms = 50
    router._pd_decode_npu_reuse_hint = Mock(return_value=(["hash"], 156))
    router._decode_admission_decision = Mock(return_value=(capacity_ok, {"reason": "decode_free_block_budget"}))
    router.worker_manager.confirm_kvbm_block_pool_capacity.side_effect = [
        {"ok": False, "error_type": "ReadTimeout", "error": "ReadTimeout: ", "elapsed_ms": 50},
        {"ok": final_ok, "error_type": "ReadTimeout", "error": "ReadTimeout: ", "elapsed_ms": 150},
    ]
    admitted, details = router._confirm_decode_capacity(
        decode,
        route_hint,
        {"estimated_blocks": 6, "capacity_source": "engine_core"},
    )
    assert admitted is (final_ok and capacity_ok)
    assert details["capacity_confirmation_attempts"] == 2
    assert details["capacity_confirmation_ms"] == 200
    calls = router.worker_manager.confirm_kvbm_block_pool_capacity.call_args_list
    assert [c.kwargs["timeout_ms"] for c in calls] == [50, 150]
    assert calls[0].kwargs["handoff_id"] == calls[1].kwargs["handoff_id"]
    if not final_ok:
        router._decode_admission_decision.assert_not_called()


@pytest.mark.parametrize(
    "estimated,reusable,error,timeout",
    [
        (162, 0, "timeout", 50),
        (6, 156, "HTTPStatusError", 50),
        (6, 156, "timeout", 150),
        (6, 0, "timeout", 50),
    ],
)
def test_delta_retry_does_not_retry_non_timeout_or_long_confirmation(
    router,
    route_hint,
    estimated,
    reusable,
    error,
    timeout,
):
    decode = WorkerInfo(worker_id="decode-no-retry", model="test-model", worker_type=WorkerType.DECODE)
    router._pd_decode_capacity_confirmation_enabled = True
    router._pd_decode_capacity_confirmation_timeout_ms = timeout
    router._pd_decode_npu_reuse_hint = Mock(return_value=(["hash"], reusable))
    router.worker_manager.confirm_kvbm_block_pool_capacity.return_value = {"ok": False, "error": error}
    assert (
        router._confirm_decode_capacity(
            decode,
            route_hint,
            {
                "estimated_blocks": estimated,
                "capacity_source": "engine_core",
            },
        )[0]
        is False
    )
    router.worker_manager.confirm_kvbm_block_pool_capacity.assert_called_once()


@pytest.mark.parametrize(
    "free,limit,max_inflight,inflight,expected_reason",
    [
        (175, 64, 0, 4, "admitted"),
        (0, 64, 0, 4, "decode_free_block_budget"),
        (175, 4, 0, 4, "decode_concurrency_limit"),
        (175, 64, 4, 4, "decode_inflight_limit"),
    ],
)
def test_hixl_auto_decode_replaces_ratio_with_capacity_and_caps(
    router,
    route_hint,
    free,
    limit,
    max_inflight,
    inflight,
    expected_reason,
):
    decode = WorkerInfo(
        worker_id="decode-ratio",
        model="test-model",
        worker_type=WorkerType.DECODE,
        running_requests=1,
        queue_depth=2,
        num_gpu_blocks=994,
        kvbm_free_blocks=free,
        kvbm_block_pool_updated_at=time.monotonic(),
    )
    router._admission_mode = "dynamic"
    router._saturation_ratio = 2
    router._pd_decode_admission_mode = "auto"
    router._pd_decode_capacity_confirmation_enabled = True
    router._pd_decode_block_admission_enabled = True
    router._pd_decode_max_inflight = max_inflight
    router._concurrency_limit = limit
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._pd_decode_npu_reuse_hint = Mock(return_value=([], 0))
    route_hint.token_ids = list(range(20500))
    route_hint.estimated_output_tokens = 128
    for _ in range(inflight):
        router.inflight.increment(decode.worker_id)
    assert router._is_worker_admitted(decode) is False
    admitted, details = router._decode_admission_decision(decode, route_hint)
    assert admitted is (expected_reason == "admitted")
    assert details["reason"] == expected_reason
    assert details["generic_ratio_bypassed"] is True
    assert details["concurrency_limit"] == limit
    if not admitted:
        assert router._is_decode_capacity_confirmation_candidate(admitted, details) is False


@pytest.mark.parametrize(
    "confirmation,blocks,mode,worker_type",
    [
        (False, True, "auto", WorkerType.DECODE),
        (True, False, "auto", WorkerType.DECODE),
        (True, True, "fixed", WorkerType.DECODE),
        (True, True, "auto", WorkerType.COMBINED),
    ],
)
def test_ratio_bypass_is_scoped_to_hixl_auto_decode(
    router,
    route_hint,
    confirmation,
    blocks,
    mode,
    worker_type,
):
    worker = WorkerInfo(
        worker_id="other-ratio", model="test-model", worker_type=worker_type, running_requests=1, queue_depth=2
    )
    router._admission_mode = "dynamic"
    router._saturation_ratio = 2
    router._pd_decode_admission_mode = mode
    router._pd_decode_capacity_confirmation_enabled = confirmation
    router._pd_decode_block_admission_enabled = blocks
    router._pd_decode_npu_reuse_hint = Mock(return_value=([], 0))
    admitted, details = router._decode_admission_decision(worker, route_hint)
    assert admitted is False
    assert details["reason"] == "worker_admission"
    assert details["generic_ratio_bypassed"] is False
    prefill = WorkerInfo(
        worker_id="prefill-ratio", model="test-model", worker_type=WorkerType.PREFILL, running_requests=1, queue_depth=2
    )
    router._pd_prefill_admission_mode = "auto"
    assert router._prefill_admission_decision(prefill, route_hint)[0] is True
    router._pd_prefill_admission_mode = "fixed"
    assert router._prefill_admission_decision(prefill, route_hint)[1]["reason"] == "worker_admission"
    native = WorkerInfo(
        worker_id="native-ratio", model="test-model", worker_type=WorkerType.DECODE, running_requests=1, queue_depth=2
    )
    assert router._native_decode_admission_decision(native)[1]["reason"] == "native_decode_pending_load"


def test_small_delta_timeout_cooldown_does_not_grow_to_one_second(router):
    decode = WorkerInfo(worker_id="decode-small-cooldown", model="test-model", worker_type=WorkerType.DECODE)
    details = {
        "reason": "decode_capacity_confirmation_failed",
        "estimated_blocks": 6,
        "capacity_confirmation_candidate_reusable_blocks": 156,
        "capacity_confirmation_error_type": "ReadTimeout",
    }
    for _ in range(4):
        result = router._record_pd_decode_confirmation_rejection("small", decode, details)
        assert result["capacity_confirmation_retry_delay_ms"] == 250
        assert result["capacity_confirmation_wait_class"] == "query_retry"
    # A genuine confirmed budget rejection must not inherit network retry handling.
    result = router._record_pd_decode_confirmation_rejection(
        "budget",
        decode,
        {
            **details,
            "reason": "decode_free_block_budget",
            "capacity_confirmed": True,
            "free_blocks": 0,
            "available_blocks": 0,
            "capacity_confirmation_generation": 1,
            "confirmed_reusable_prompt_blocks": 156,
        },
    )
    assert result["capacity_confirmation_wait_class"] == "threshold_wait"
    assert "capacity_confirmation_retry_delay_ms" not in result


def test_pd_decode_capacity_confirmation_runs_outside_routing_lock(
    router,
    route_hint,
):
    """engine-core容量HTTP执行期间不得占用全局路由锁。"""
    prefill = WorkerInfo(
        worker_id="prefill-lock-test",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-lock-test",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=200,
        kvbm_block_pool_updated_at=time.monotonic(),
    )
    route_hint.token_ids = list(range(128))
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._pd_decode_capacity_confirmation_enabled = True
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }
    router.current_algorithm.select_worker_pair = Mock(return_value=(prefill, decode))

    def confirm_capacity(_worker_id, *, timeout_ms, **kwargs):
        assert timeout_ms > 0
        acquired = router._routing_lock.acquire(blocking=False)
        assert acquired is True
        router._routing_lock.release()
        return {"ok": True, "free_blocks": 200, "elapsed_ms": 1.0}

    router.worker_manager.confirm_kvbm_block_pool_capacity.side_effect = confirm_capacity

    assert router.find_best_worker_pair("test-model", route_hint) == (
        prefill.worker_id,
        decode.worker_id,
    )


def test_capacity_generation_wait_observes_update_before_wait(router):
    """先发生的容量更新也必须让后续generation等待立即返回。"""
    generation = router.get_capacity_generation()

    router.notify_metrics_updated()

    started = time.perf_counter()
    assert router.wait_for_capacity_change(generation, timeout=1.0) is True
    assert time.perf_counter() - started < 0.1


def test_capacity_generation_wait_returns_for_confirmation_retry(router):
    """没有容量事件时也应在瞬时确认冷却结束后重新路由。"""
    decode = WorkerInfo(
        worker_id="decode-transient-retry",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    router._record_pd_decode_confirmation_rejection(
        "request-transient-retry",
        decode,
        {
            "reason": "decode_capacity_confirmation_failed",
            "capacity_confirmation_error": "timeout",
        },
    )
    generation = router.get_capacity_generation()
    started = time.perf_counter()

    assert (
        router.wait_for_capacity_change(
            generation,
            timeout=1.0,
            request_id="request-transient-retry",
        )
        is False
    )
    elapsed = time.perf_counter() - started

    assert 0.20 <= elapsed < 0.75


def test_capacity_generation_wait_returns_when_affinity_grace_expires(router):
    """Affinity 恢复窗口结束时应唤醒请求尝试其他 D。"""
    decode = WorkerInfo(
        worker_id="decode-affinity-grace",
        model="test-model",
        worker_type=WorkerType.DECODE,
        kvbm_capacity_generation=3,
    )
    router._record_pd_decode_confirmation_rejection(
        "request-affinity-grace",
        decode,
        {
            "reason": "decode_free_block_budget",
            "capacity_confirmed": True,
            "capacity_confirmation_generation": 3,
        },
        affinity_worker_id=decode.worker_id,
    )
    with router._pd_decode_confirmation_rejection_lock:
        router._pd_decode_affinity_rejected_at["request-affinity-grace"][decode.worker_id] -= 0.70
    generation = router.get_capacity_generation()
    started = time.perf_counter()

    assert (
        router.wait_for_capacity_change(
            generation,
            timeout=0.5,
            request_id="request-affinity-grace",
        )
        is False
    )
    elapsed = time.perf_counter() - started

    assert 0.02 <= elapsed < 0.30


def test_capacity_generation_ignores_duplicate_metric_heartbeat(router):
    """准入状态未变化的重复心跳不得唤醒全部排队请求。"""
    worker = WorkerInfo(
        worker_id="decode-generation-dedup",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=200,
        running_requests=2,
        queue_depth=3,
    )
    router.worker_manager.get_all_workers.return_value = {
        worker.worker_id: worker,
    }

    router.notify_metrics_updated()
    first_generation = router.get_capacity_generation()
    router.notify_metrics_updated()

    assert router.get_capacity_generation() == first_generation

    worker.kvbm_free_blocks = 199
    router.notify_metrics_updated()

    assert router.get_capacity_generation() == first_generation + 1


def test_metric_update_advances_only_changed_worker_capacity_generation(router):
    """Unrelated worker metrics must not reactivate a parked decode waiter."""
    prefill = WorkerInfo(
        worker_id="prefill-generation",
        model="test-model",
        worker_type=WorkerType.PREFILL,
        running_requests=1,
    )
    decode = WorkerInfo(
        worker_id="decode-generation",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=200,
    )
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }
    router.notify_metrics_updated()
    prefill_generation = router._capacity_worker_generations[prefill.worker_id]
    decode_generation = router._capacity_worker_generations[decode.worker_id]

    prefill.running_requests = 2
    router.notify_metrics_updated()

    assert router._capacity_worker_generations[prefill.worker_id] == prefill_generation + 1
    assert router._capacity_worker_generations[decode.worker_id] == decode_generation


def test_capacity_waiters_are_admitted_in_fifo_order(router):
    """容量争用期间后到请求不能越过已等待请求。"""
    model = "test-model"

    assert router.enqueue_capacity_waiter("request-1", model) is True
    assert router.enqueue_capacity_waiter("request-2", model, only_if_contended=True) is True
    assert router.wait_for_capacity_turn("request-1", model, timeout=0.0) is True
    assert router.wait_for_capacity_turn("request-2", model, timeout=0.0) is False

    router.remove_capacity_waiter("request-1", model)

    assert router.wait_for_capacity_turn("request-2", model, timeout=0.0) is True
    router.remove_capacity_waiter("request-2", model)
    assert router.enqueue_capacity_waiter("request-3", model, only_if_contended=True) is False


def test_capacity_waiters_allow_bounded_parallel_turns(router):
    """HIXL可按D数量并行准入，同时保持窗口外请求的FIFO顺序。"""
    model = "test-model"
    request_ids = [f"request-{index}" for index in range(1, 6)]
    for request_id in request_ids:
        assert router.enqueue_capacity_waiter(request_id, model) is True

    for request_id in request_ids[:4]:
        assert (
            router.wait_for_capacity_turn(
                request_id,
                model,
                timeout=0.0,
                window_size=4,
            )
            is True
        )
    assert (
        router.wait_for_capacity_turn(
            request_ids[4],
            model,
            timeout=0.0,
            window_size=4,
        )
        is False
    )

    router.remove_capacity_waiter(request_ids[1], model)

    assert (
        router.wait_for_capacity_turn(
            request_ids[4],
            model,
            timeout=0.0,
            window_size=4,
        )
        is True
    )


def test_parked_capacity_waiter_yields_fifo_slot(router):
    """A failed head waiter must not block a later request until completion."""
    model = "test-model"
    generation = router.get_capacity_generation()
    router.enqueue_capacity_waiter("request-1", model)
    router.enqueue_capacity_waiter("request-2", model)

    assert router.wait_for_capacity_turn("request-1", model, timeout=0.0) is True
    assert (
        router.defer_capacity_waiter(
            "request-1",
            model,
            generation,
            blocked_worker_ids=("decode-0",),
            retry_after=1.0,
        )
        is True
    )

    assert router.wait_for_capacity_turn("request-1", model, timeout=0.0) is False
    assert router.wait_for_capacity_turn("request-2", model, timeout=0.0) is True


def test_capacity_change_reactivates_parked_waiter_in_fifo_order(router):
    """Capacity changes restore the parked request's original FIFO priority."""
    model = "test-model"
    generation = router.get_capacity_generation()
    router.enqueue_capacity_waiter("request-1", model)
    router.enqueue_capacity_waiter("request-2", model)
    router.defer_capacity_waiter(
        "request-1",
        model,
        generation,
        blocked_worker_ids=("decode-0",),
        retry_after=1.0,
    )

    with router._release_cond:
        router._publish_capacity_change_locked(("prefill-0",))

    assert router.wait_for_capacity_turn("request-1", model, timeout=0.0) is False
    assert router.wait_for_capacity_turn("request-2", model, timeout=0.0) is True

    with router._release_cond:
        router._publish_capacity_change_locked(("decode-0",))

    assert router.wait_for_capacity_turn("request-1", model, timeout=0.0) is True
    assert router.wait_for_capacity_turn("request-2", model, timeout=0.0) is False


def test_capacity_change_revalidates_waiter_on_worker_generation_change(router):
    """free阈值未跨时，同一D的新容量代际仍应触发一次权威复核。"""
    model = "test-model"
    request_id = "request-threshold-wait"
    decode = WorkerInfo(
        worker_id="decode-threshold-wait",
        model=model,
        worker_type=WorkerType.DECODE,
        kvbm_free_blocks=20,
        kvbm_capacity_generation=7,
    )
    router.worker_manager.get_all_workers.return_value = {
        decode.worker_id: decode,
    }
    router._record_pd_decode_confirmation_rejection(
        request_id,
        decode,
        {
            "reason": "decode_free_block_budget",
            "capacity_confirmed": True,
            "capacity_confirmation_generation": 7,
            "free_blocks": 20,
            "available_blocks": 20,
            "estimated_blocks": 51,
            "dynamic_margin_blocks": 8,
        },
    )
    generation = router.get_capacity_generation()
    router.enqueue_capacity_waiter(request_id, model)
    router.defer_capacity_waiter(
        request_id,
        model,
        generation,
        blocked_worker_ids=(decode.worker_id,),
    )

    decode.kvbm_free_blocks = 30
    with router._release_cond:
        router._publish_capacity_change_locked((decode.worker_id,))

    assert router.wait_for_capacity_turn(request_id, model, timeout=0.0) is False

    decode.kvbm_capacity_generation = 8
    with router._release_cond:
        router._publish_capacity_change_locked((decode.worker_id,))

    assert router.wait_for_capacity_turn(request_id, model, timeout=0.0) is True


def test_parked_waiter_wakes_on_definitive_rejection_probe(router):
    """确定性拒绝的复核到期应唤醒 parked 请求重发权威确认。"""
    model = "test-model"
    request_id = "request-probe-wake"
    decode = WorkerInfo(
        worker_id="decode-probe-wake",
        model=model,
        worker_type=WorkerType.DECODE,
        kvbm_free_blocks=20,
        kvbm_capacity_generation=7,
    )
    router.worker_manager.get_all_workers.return_value = {
        decode.worker_id: decode,
    }
    router._record_pd_decode_confirmation_rejection(
        request_id,
        decode,
        {
            "reason": "decode_free_block_budget",
            "capacity_confirmed": True,
            "capacity_confirmation_generation": 7,
            "free_blocks": 20,
            "available_blocks": 20,
            "estimated_blocks": 51,
            "dynamic_margin_blocks": 8,
            "candidate_reusable_prompt_blocks": 46,
            "confirmed_reusable_prompt_blocks": 0,
        },
    )
    generation = router.get_capacity_generation()
    router.enqueue_capacity_waiter(request_id, model)
    router.defer_capacity_waiter(
        request_id,
        model,
        generation,
        blocked_worker_ids=(decode.worker_id,),
    )

    # 阈值未跨、复核未到期：保持 parked。
    assert router.wait_for_capacity_turn(request_id, model, timeout=0.0) is False

    # 复核到期（defer 时写入 waiter 的复核时间点）：作为 retry_timer 唤醒，
    # 请求重新进入路由循环并重发权威确认。
    with router._release_cond:
        waiter = router._capacity_waiters[model][request_id]
        assert waiter.retry_at is not None
        waiter.retry_at = time.monotonic() - 0.001

    assert router.wait_for_capacity_turn(request_id, model, timeout=0.0) is True


def test_parked_capacity_waiter_gets_bounded_periodic_retry(router):
    """A parked request must retry even without a capacity metric update."""
    model = "test-model"
    generation = router.get_capacity_generation()
    router.enqueue_capacity_waiter("request-1", model)
    router.defer_capacity_waiter(
        "request-1",
        model,
        generation,
        blocked_worker_ids=("decode-0",),
        retry_after=0.03,
    )
    started = time.perf_counter()

    assert router.wait_for_capacity_turn("request-1", model, timeout=0.5) is True
    elapsed = time.perf_counter() - started

    assert 0.02 <= elapsed < 0.25


def test_capacity_change_wait_returns_for_parked_waiter_retry(router):
    """The production capacity wait must honor a parked waiter's retry time."""
    model = "test-model"
    generation = router.get_capacity_generation()
    router.enqueue_capacity_waiter("request-1", model)
    router.defer_capacity_waiter(
        "request-1",
        model,
        generation,
        blocked_worker_ids=("decode-0",),
        retry_after=0.03,
    )
    started = time.perf_counter()

    assert (
        router.wait_for_capacity_change(
            generation,
            timeout=0.5,
            request_id="request-1",
        )
        is False
    )
    elapsed = time.perf_counter() - started

    assert 0.02 <= elapsed < 0.25


def test_parked_capacity_wait_ignores_unrelated_worker_change(router):
    """The blocking wait itself must ignore capacity changes from other workers."""
    model = "test-model"
    generation = router.get_capacity_generation()
    router.enqueue_capacity_waiter("request-1", model)
    router.defer_capacity_waiter(
        "request-1",
        model,
        generation,
        blocked_worker_ids=("decode-0",),
    )
    result = []

    thread = threading.Thread(
        target=lambda: result.append(
            router.wait_for_capacity_change(
                generation,
                timeout=0.5,
                request_id="request-1",
            )
        )
    )
    thread.start()
    with router._release_cond:
        router._publish_capacity_change_locked(("prefill-0",))
    time.sleep(0.03)
    assert thread.is_alive()

    with router._release_cond:
        router._publish_capacity_change_locked(("decode-0",))
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert result == [True]


def test_pd_decode_admission_rejects_razor_thin_confirmed_capacity(
    router,
    route_hint,
):
    """实时容量仅比请求预算多一个block时应保留动态安全余量。"""
    decode = WorkerInfo(
        worker_id="decode-dynamic-margin",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=6,
        kvbm_block_pool_updated_at=time.monotonic(),
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 5888}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._pd_block_ownership = "legacy"

    admitted, diagnostics = router._decode_admission_decision(decode, route_hint)

    assert diagnostics["estimated_blocks"] == 5
    assert diagnostics["available_blocks"] == 6
    assert diagnostics["dynamic_margin_blocks"] == 2
    assert admitted is False
    assert diagnostics["reason"] == "decode_free_block_budget"


def test_pd_decode_admission_deducts_existing_reservations_from_new_sample(
    router,
    route_hint,
):
    """新容量采样也不能吞掉尚未落入BlockPool的Router在途预算。"""
    decode = WorkerInfo(
        worker_id="decode-new-sample-race",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        kvbm_free_blocks=100,
        kvbm_block_pool_updated_at=time.monotonic(),
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router.pd_decode_blocks.reserve(
        "request-before-sample",
        decode.worker_id,
        51,
        0,
    )

    admitted, diagnostics = router._decode_admission_decision(decode, route_hint)

    assert admitted is False
    assert diagnostics["reason"] == "decode_free_block_budget"
    assert diagnostics["free_blocks"] == 100
    assert diagnostics["pending_reserved_blocks"] == 51
    assert diagnostics["capacity_headroom_blocks"] == 0
    assert diagnostics["fallback_headroom_blocks"] == 100
    assert diagnostics["available_blocks"] == 49


def test_pd_decode_auto_mode_reserves_full_prompt_and_expected_output(
    router,
    route_hint,
):
    """Scheduler所有权应覆盖完整Prompt和输出的目标block布局。"""
    decode = WorkerInfo(
        worker_id="decode-auto",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 4096
    route_hint.precomputed_overlap_scores = {decode.worker_id: 5888}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._pd_decode_initial_window_blocks = 2
    router._pd_decode_max_inflight = 0

    admitted, diagnostics = router._decode_admission_decision(decode, route_hint)

    assert admitted is True
    assert diagnostics["estimated_blocks"] == 82
    assert diagnostics["steady_state_blocks"] == 30
    assert diagnostics["block_ownership"] == "scheduler"
    assert diagnostics["max_inflight"] == 0


def test_pd_decode_legacy_ownership_can_deduct_router_prefix_overlap(
    router,
    route_hint,
):
    """仅legacy所有权允许使用Router前缀命中缩减attach前预算。"""
    decode = WorkerInfo(
        worker_id="decode-legacy-overlap",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 5888}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._pd_block_ownership = "legacy"

    admitted, diagnostics = router._decode_admission_decision(decode, route_hint)

    assert admitted is True
    assert diagnostics["estimated_blocks"] == 5
    assert diagnostics["block_ownership"] == "legacy"


def test_pd_decode_short_output_does_not_force_full_initial_window(
    router,
    route_hint,
):
    """128-token输出只需一个block，attach后不应继续保留增长预算。"""
    decode = WorkerInfo(
        worker_id="decode-short-output",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 128
    route_hint.precomputed_overlap_scores = {decode.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._pd_decode_initial_window_blocks = 2

    admitted, diagnostics = router._decode_admission_decision(decode, route_hint)

    assert admitted is True
    assert diagnostics["estimated_blocks"] == 51
    assert diagnostics["steady_state_blocks"] == 0


def test_pd_decode_attach_waits_for_fresh_capacity_before_reconcile(
    router,
    route_hint,
):
    """Prompt attach预算应等新BlockPool样本确认后再收缩。"""
    worker_id = "decode-reconcile"
    router._pd_decode_initial_window_blocks = 2
    router.inflight.increment(worker_id)
    router.pd_decode_blocks.reserve(route_hint.request_id, worker_id, 51, 2)
    old_capacity_sample = time.monotonic()

    router.reconcile_pd_decode_budget(worker_id, route_hint.request_id)
    router.reconcile_pd_decode_budget(worker_id, route_hint.request_id)

    assert router.pd_decode_blocks.get_blocks(worker_id) == 51
    acknowledged, remaining = router.pd_decode_blocks.acknowledge_capacity(
        worker_id,
        old_capacity_sample,
    )
    assert acknowledged == 0
    assert remaining == 51

    acknowledged, remaining = router.pd_decode_blocks.acknowledge_capacity(
        worker_id,
        time.monotonic() + 1.0,
    )
    assert acknowledged == 0
    assert remaining == 51

    acknowledged, remaining = router.pd_decode_blocks.acknowledge_capacity(
        worker_id,
        time.monotonic() + 2.0,
    )
    assert acknowledged == 1
    assert remaining == 2
    assert router.pd_decode_blocks.get_blocks(worker_id) == 2

    router.release_worker(worker_id, request_id=route_hint.request_id)
    assert router.pd_decode_blocks.get_blocks(worker_id) == 0


def test_pd_decode_attach_uses_engine_capacity_generation(router, route_hint):
    """attach预算只能由不早于目标分配代际的容量样本收缩。"""
    worker_id = "decode-generation"
    router.pd_decode_blocks.reserve(route_hint.request_id, worker_id, 51, 2)
    router.pd_decode_blocks.mark_allocated(
        worker_id,
        route_hint.request_id,
        102,
    )
    router.pd_decode_blocks.reconcile(
        worker_id,
        route_hint.request_id,
        102,
    )

    acknowledged, remaining = router.pd_decode_blocks.acknowledge_capacity(
        worker_id,
        time.monotonic() + 1.0,
        101,
    )
    assert acknowledged == 0
    assert remaining == 51

    acknowledged, remaining = router.pd_decode_blocks.acknowledge_capacity(
        worker_id,
        time.monotonic() + 2.0,
        102,
    )
    assert acknowledged == 1
    assert remaining == 2


def test_pd_decode_delta_reservation_restores_full_fallback_budget(
    router,
    route_hint,
):
    """Delta准入后PD失败应恢复D本地重算的完整预算。"""
    worker_id = "decode-delta-fallback"
    router.pd_decode_blocks.reserve(
        route_hint.request_id,
        worker_id,
        5,
        0,
        fallback_block_count=51,
    )

    assert router.pd_decode_blocks.get_blocks(worker_id) == 5

    router.restore_pd_decode_budget_after_fallback(
        worker_id,
        route_hint.request_id,
    )

    assert router.pd_decode_blocks.get_blocks(worker_id) == 51
    router.release_worker(worker_id, request_id=route_hint.request_id)
    assert router.pd_decode_blocks.get_blocks(worker_id) == 0


def test_pd_decode_scheduler_allocation_retains_budget_until_attach(
    router,
    route_hint,
):
    """Scheduler分配通知不得在attach前收缩Router预算。"""
    worker_id = "decode-allocated"
    router.pd_decode_blocks.reserve(route_hint.request_id, worker_id, 51, 1)

    router.mark_pd_decode_blocks_allocated(worker_id, route_hint.request_id)
    router.mark_pd_decode_blocks_allocated(worker_id, route_hint.request_id)

    assert router.pd_decode_blocks.get_blocks(worker_id) == 51
    acknowledged, remaining = router.pd_decode_blocks.acknowledge_capacity(
        worker_id,
        time.monotonic() + 1.0,
    )
    assert acknowledged == 0
    assert remaining == 51

    acknowledged, remaining = router.pd_decode_blocks.acknowledge_capacity(
        worker_id,
        time.monotonic() + 2.0,
    )
    assert acknowledged == 0
    assert remaining == 51

    router.reconcile_pd_decode_budget(worker_id, route_hint.request_id)
    acknowledged, remaining = router.pd_decode_blocks.acknowledge_capacity(
        worker_id,
        time.monotonic() + 3.0,
    )
    assert acknowledged == 0
    assert remaining == 51

    acknowledged, remaining = router.pd_decode_blocks.acknowledge_capacity(
        worker_id,
        time.monotonic() + 4.0,
    )
    assert acknowledged == 1
    assert remaining == 1

    router.restore_pd_decode_budget_after_fallback(
        worker_id,
        route_hint.request_id,
    )
    assert router.pd_decode_blocks.get_blocks(worker_id) == 51

    # 降级通知可能由多个控制路径重复到达，不得重复增加预算。
    router.restore_pd_decode_budget_after_fallback(
        worker_id,
        route_hint.request_id,
    )
    assert router.pd_decode_blocks.get_blocks(worker_id) == 51

    router.release_worker(worker_id, request_id=route_hint.request_id)
    assert router.pd_decode_blocks.get_blocks(worker_id) == 0


def test_pd_decode_admission_adds_unobserved_growth_to_worker_usage(
    router,
    route_hint,
):
    """尚未反映到Worker指标的Decode增长预算必须与实际占用相加。"""
    decode = WorkerInfo(
        worker_id="decode-future-growth",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        active_decode_blocks=800,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.estimated_output_tokens = 512
    route_hint.precomputed_overlap_scores = {decode.worker_id: 5888}
    router.kvcache_manager.get_worker_block_size.return_value = 128
    router._pd_decode_block_headroom_ratio = 0.10
    router.pd_decode_blocks.reserve("existing-request", decode.worker_id, 100, 2)

    admitted, diagnostics = router._decode_admission_decision(decode, route_hint)

    assert admitted is False
    assert diagnostics["reason"] == "decode_block_budget"
    assert diagnostics["projected_blocks"] == 954
    assert diagnostics["usable_blocks"] == 900


def test_pd_prefill_auto_admission_tracks_and_releases_budget(router, route_hint):
    """P自动准入应按缺失token/block登记，并在P阶段完成时立即归还。"""
    prefill = WorkerInfo(
        worker_id="prefill-auto",
        model="test-model",
        worker_type=WorkerType.PREFILL,
        num_gpu_blocks=1000,
        active_decode_blocks=100,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.precomputed_overlap_scores = {prefill.worker_id: 5888}
    router.kvcache_manager.get_worker_block_size.return_value = 128

    admitted, diagnostics = router._prefill_admission_decision(prefill, route_hint)
    assert admitted is True
    assert diagnostics["estimated_tokens"] == 412
    assert diagnostics["estimated_blocks"] == 4

    router.inflight.increment(prefill.worker_id)
    router.pd_prefill_budget.reserve(
        route_hint.request_id,
        prefill.worker_id,
        diagnostics["estimated_tokens"],
        diagnostics["estimated_blocks"],
    )
    router.release_pd_prefill(prefill.worker_id, route_hint.request_id)

    assert router.inflight.get(prefill.worker_id) == 0
    assert router.pd_prefill_budget.get(prefill.worker_id) == (0, 0)


def test_pd_prefill_uses_tensor_derived_block_size_before_first_kv_event(
    router,
    route_hint,
):
    """冷启动时应使用Worker指标中的真实block size，不能回退到Router的16。"""
    prefill = WorkerInfo(
        worker_id="prefill-cold-start",
        model="test-model",
        worker_type=WorkerType.PREFILL,
        num_gpu_blocks=1000,
        kvbm_block_size=128,
    )
    router.worker_manager.get_worker.return_value = prefill
    router.kvcache_manager.get_worker_block_size.return_value = 16
    route_hint.token_ids = list(range(6300))
    route_hint.precomputed_overlap_scores = {prefill.worker_id: 0}

    admitted, diagnostics = router._prefill_admission_decision(prefill, route_hint)

    assert admitted is True
    assert diagnostics["estimated_blocks"] == 50


def test_pd_prefill_auto_admission_does_not_treat_scheduler_step_as_total_capacity(
    router,
    route_hint,
):
    """max_num_batched_tokens是单步上限，不得把第二个长请求错误拒绝。"""
    prefill = WorkerInfo(
        worker_id="prefill-token-budget",
        model="test-model",
        worker_type=WorkerType.PREFILL,
        num_gpu_blocks=1000,
        max_num_batched_tokens=8192,
    )
    route_hint.token_ids = list(range(6300))
    route_hint.precomputed_overlap_scores = {prefill.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128

    admitted, first = router._prefill_admission_decision(prefill, route_hint)
    assert admitted is True
    router.pd_prefill_budget.reserve("request-first", prefill.worker_id, 6300, 50)

    admitted, second = router._prefill_admission_decision(prefill, route_hint)

    assert admitted is True
    assert second["reason"] == "admitted"
    assert second["projected_tokens"] == 12600
    assert second["token_capacity"] == 8192
    assert first["estimated_tokens"] == 6300


def test_pd_prefill_auto_admission_uses_fallback_until_metrics_arrive(
    router,
    route_hint,
):
    """Worker尚未上报Scheduler容量时，应使用旧并发值防止无限派发。"""
    prefill = WorkerInfo(
        worker_id="prefill-no-metrics",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    router._pd_prefill_auto_fallback_max_inflight = 2
    router.inflight.increment(prefill.worker_id)
    router.inflight.increment(prefill.worker_id)

    admitted, diagnostics = router._prefill_admission_decision(prefill, route_hint)

    assert admitted is False
    assert diagnostics["reason"] == "prefill_metrics_fallback_limit"


def test_pd_prefill_auto_admission_uses_block_budget_when_metrics_exist(
    router,
    route_hint,
):
    """已有block容量指标时，应按真实预算而不是兼容并发值准入。"""
    prefill = WorkerInfo(
        worker_id="prefill-with-metrics",
        model="test-model",
        worker_type=WorkerType.PREFILL,
        num_gpu_blocks=1000,
        max_num_batched_tokens=2048,
    )
    router._pd_prefill_auto_fallback_max_inflight = 2
    router.inflight.increment(prefill.worker_id)
    router.inflight.increment(prefill.worker_id)

    admitted, diagnostics = router._prefill_admission_decision(prefill, route_hint)

    assert admitted is True
    assert diagnostics["reason"] == "admitted"


def test_pd_prefill_auto_admission_ignores_generic_waiting_ratio(
    router,
    route_hint,
):
    """P专属预算可用时，不应因通用waiting/running比值拒绝Worker。"""
    prefill = WorkerInfo(
        worker_id="prefill-generic-saturation",
        model="test-model",
        worker_type=WorkerType.PREFILL,
        num_gpu_blocks=1000,
        running_requests=1,
        queue_depth=3,
    )
    router._admission_mode = "dynamic"
    router._saturation_ratio = 0.2
    router._pd_prefill_auto_fallback_max_inflight = 8
    route_hint.token_ids = list(range(6300))
    route_hint.precomputed_overlap_scores = {prefill.worker_id: 0}
    router.kvcache_manager.get_worker_block_size.return_value = 128

    admitted, diagnostics = router._prefill_admission_decision(prefill, route_hint)

    assert admitted is True
    assert diagnostics["reason"] == "admitted"


def test_pd_decode_admission_uses_another_worker_and_releases_budget(
    router,
    route_hint,
):
    """一个D满时应改选其他D，并在请求结束后归还其block预算。"""
    prefill = WorkerInfo(
        worker_id="prefill-0",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    full_decode = WorkerInfo(
        worker_id="decode-full",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    available_decode = WorkerInfo(
        worker_id="decode-available",
        model="test-model",
        worker_type=WorkerType.DECODE,
        num_gpu_blocks=1000,
        active_decode_blocks=100,
    )
    router._pd_decode_max_inflight = 1
    router.inflight.increment(full_decode.worker_id)
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [
        full_decode,
        available_decode,
    ]
    router.worker_manager.get_all_workers.return_value = {
        worker.worker_id: worker for worker in (prefill, full_decode, available_decode)
    }
    router.worker_manager.get_worker.side_effect = {
        prefill.worker_id: prefill,
        full_decode.worker_id: full_decode,
        available_decode.worker_id: available_decode,
    }.get
    router.kvcache_manager.get_worker_block_size.return_value = 128
    route_hint.precomputed_overlap_scores = {available_decode.worker_id: 0}
    router.current_algorithm.select_worker_pair = Mock(return_value=(prefill, available_decode))

    with (
        patch("openjiuwentools.infer_router.routing.router.logger.info") as info,
        patch.object(
            router,
            "_confirm_decode_capacity",
            wraps=router._confirm_decode_capacity,
        ) as confirm_capacity,
    ):
        prefill_id, decode_id = router.find_best_worker_pair("test-model", route_hint)

    assert (prefill_id, decode_id) == (prefill.worker_id, available_decode.worker_id)
    candidates = router.current_algorithm.select_worker_pair.call_args.kwargs["decode_workers"]
    assert [worker.worker_id for worker in candidates] == [available_decode.worker_id]
    admission_calls = [
        log_call
        for log_call in info.call_args_list
        if log_call.args and log_call.args[0].startswith("[PDAdmissionDetail]")
    ]
    assert len(admission_calls) == 2
    decode_log = admission_calls[1]
    assert decode_log.args[1:3] == (route_hint.request_id, "decode")
    assert "decode-full(selected=False,admitted=False" in decode_log.args[3]
    assert "reason=decode_inflight_limit" in decode_log.args[3]
    assert "decode-available(selected=True,admitted=True" in decode_log.args[3]
    confirm_capacity.assert_not_called()
    assert router.pd_decode_blocks.get_blocks(available_decode.worker_id) > 0

    router.release_worker(available_decode.worker_id, request_id=route_hint.request_id)

    assert router.pd_decode_blocks.get_blocks(available_decode.worker_id) == 0


def test_pd_decode_admission_returns_no_pair_when_all_decode_workers_full(
    router,
    route_hint,
):
    """所有D满载时不得先派发P或创建D gate。"""
    prefill = WorkerInfo(
        worker_id="prefill-0",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-0",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    router._pd_decode_max_inflight = 1
    router.inflight.increment(decode.worker_id)
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }

    assert router.find_best_worker_pair("test-model", route_hint) == (None, None)
    assert router.inflight.get(prefill.worker_id) == 0


def test_native_connector_pair_skips_router_managed_pd_budgets(router, route_hint):
    """Mooncake/native connector must not inherit HIXL block reservations."""
    prefill = WorkerInfo(
        worker_id="prefill-native",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-native",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    router._admission_mode = "static"
    router._concurrency_limit = 8
    router._pd_prefill_max_inflight = 1
    router._pd_decode_max_inflight = 1
    router._pd_prefill_block_admission_enabled = True
    router._pd_decode_block_admission_enabled = True
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router.current_algorithm.select_worker_pair = Mock(return_value=(prefill, decode))

    selected = router.find_best_worker_pair(
        "test-model",
        route_hint,
        router_managed_pd=False,
    )

    assert selected == (prefill.worker_id, decode.worker_id)
    assert router.inflight.get(prefill.worker_id) == 1
    assert router.inflight.get(decode.worker_id) == 1
    assert router.pd_prefill_budget.get(prefill.worker_id) == (0, 0)
    assert router.pd_decode_blocks.get_blocks(decode.worker_id) == 0


def test_native_connector_admission_snapshot_uses_generic_state(router, route_hint):
    """Native connector diagnostics must not report HIXL capacity state."""
    prefill = WorkerInfo(
        worker_id="prefill-native",
        model="test-model",
        worker_type=WorkerType.PREFILL,
    )
    decode = WorkerInfo(
        worker_id="decode-native",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    router._admission_mode = "static"
    router._concurrency_limit = 0
    router.worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    router.worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    router.worker_manager.get_all_workers.return_value = {
        prefill.worker_id: prefill,
        decode.worker_id: decode,
    }

    snapshot = router.get_worker_admission_snapshot(
        "test-model",
        route_hint,
        router_managed_pd=False,
    )

    assert snapshot["blocker"] == "both"
    assert snapshot["prefill_details"][prefill.worker_id]["reason"] == ("native_prefill_inflight_limit")
    assert snapshot["decode_details"][decode.worker_id]["capacity_source"] == ("native_connector")


def test_native_decode_admission_counts_bound_undispatched_requests(router):
    """Dynamic Mooncake admission must include requests still waiting on P."""
    decode = WorkerInfo(
        worker_id="decode-native",
        model="test-model",
        worker_type=WorkerType.DECODE,
        running_requests=10,
        queue_depth=2,
    )
    router._admission_mode = "dynamic"
    router._saturation_ratio = 1.0
    for _ in range(20):
        router.inflight.increment(decode.worker_id)

    admitted, details = router._native_decode_admission_decision(decode)

    assert admitted is False
    assert details["native_pending_dispatch"] == 8
    assert details["effective_waiting"] == 10
    assert details["admission_ratio"] == 1.0
    assert details["reason"] == "native_decode_pending_load"


def test_native_prefill_admission_counts_bound_undispatched_requests(router):
    """Dynamic Mooncake P admission must include requests absent from metrics."""
    prefill = WorkerInfo(
        worker_id="prefill-native",
        model="test-model",
        worker_type=WorkerType.PREFILL,
        running_requests=10,
        queue_depth=2,
    )
    router._admission_mode = "dynamic"
    router._saturation_ratio = 1.0
    router._concurrency_limit = 100
    for _ in range(20):
        router.inflight.increment(prefill.worker_id)

    admitted, details = router._native_connector_admission_decision(prefill)

    assert admitted is False
    assert details["native_pending_dispatch"] == 8
    assert details["effective_waiting"] == 10
    assert details["admission_ratio"] == 1.0
    assert details["reason"] == "native_prefill_pending_load"


@pytest.mark.parametrize(
    ("decision_method", "worker_type", "expected_reason"),
    [
        (
            "_native_connector_admission_decision",
            WorkerType.PREFILL,
            "native_prefill_inflight_limit",
        ),
        (
            "_native_decode_admission_decision",
            WorkerType.DECODE,
            "native_decode_inflight_limit",
        ),
    ],
)
def test_native_dynamic_admission_preserves_hard_inflight_limit(
    router,
    decision_method,
    worker_type,
    expected_reason,
):
    """Dynamic admission must never bypass max_concurrency_per_worker."""
    worker = WorkerInfo(
        worker_id=f"{worker_type.value}-native",
        model="test-model",
        worker_type=worker_type,
        running_requests=10,
        queue_depth=0,
    )
    router._admission_mode = "dynamic"
    router._saturation_ratio = 10.0
    router._concurrency_limit = 20
    for _ in range(20):
        router.inflight.increment(worker.worker_id)

    admitted, details = getattr(router, decision_method)(worker)

    assert admitted is False
    assert details["reason"] == expected_reason


def test_native_decode_admission_preserves_generic_zero_running_fallback(router):
    """Without running metrics, native D admission falls back to inflight."""
    decode = WorkerInfo(
        worker_id="decode-native",
        model="test-model",
        worker_type=WorkerType.DECODE,
        running_requests=0,
        queue_depth=0,
    )
    router._admission_mode = "dynamic"
    router._concurrency_limit = 2
    router.inflight.increment(decode.worker_id)

    admitted, details = router._native_decode_admission_decision(decode)

    assert admitted is True
    assert details["native_pending_dispatch"] == 1


@pytest.mark.asyncio
async def test_router_find_best_combined_worker(router, route_hint):
    """测试查找最佳组合型worker"""
    router.kvcache_manager.find_matches.return_value = {
        "combined-high-load": 0.2,
        "combined-low-load": 0.8,
    }

    combined_workers = [
        WorkerInfo(
            worker_id="combined-high-load",
            model="test-model",
            url="http://localhost:8001/v1",
            available_memory=500000,
            current_load=80,
            cached_prefixes=[],
            worker_type=WorkerType.COMBINED,
            group="group-b",
        ),
        WorkerInfo(
            worker_id="combined-low-load",
            model="test-model",
            url="http://localhost:8002/v1",
            available_memory=1500000,
            current_load=10,
            cached_prefixes=[],
            worker_type=WorkerType.COMBINED,
            group="group-b",
        ),
    ]

    router.worker_manager.get_all_healthy_combined_workers.return_value = combined_workers

    worker_id = router.find_best_combined_worker("test-model", route_hint)

    assert worker_id is not None
