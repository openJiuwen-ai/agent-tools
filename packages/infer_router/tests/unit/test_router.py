"""测试路由模块"""

from unittest.mock import Mock

import pytest

from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.routing.router import Router
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
    return Mock(spec=WorkerManager)


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
    router.worker_manager.get_healthy_workers.return_value = [worker]
    router.worker_manager.get_healthy_groups.return_value = ["group-single"]
    router.worker_manager.is_combined_group.return_value = True
    router.worker_manager.get_workers_in_group.return_value = [worker]

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
async def test_router_route_to_workers_returns_pair(
    router, route_hint, workers_with_groups_for_router
):
    """测试路由到worker对"""
    router.kvcache_manager.find_matches.return_value = {
        "prefill-1": 0.5,
        "decode-1": 0.3,
    }
    router.worker_manager.get_healthy_groups.return_value = ["group-a"]
    router.worker_manager.is_combined_group.return_value = False

    prefill_workers = [
        w for w in workers_with_groups_for_router if w.worker_type == WorkerType.PREFILL
    ]
    decode_workers = [
        w for w in workers_with_groups_for_router if w.worker_type == WorkerType.DECODE
    ]

    router.worker_manager.get_prefill_workers_in_group.return_value = prefill_workers
    router.worker_manager.get_decode_workers_in_group.return_value = decode_workers

    prefill_id, decode_id = router.route_to_workers(route_hint)

    assert prefill_id is not None
    assert decode_id is not None


@pytest.mark.asyncio
async def test_router_route_to_workers_returns_combined(
    router, route_hint, workers_with_groups_for_router
):
    """测试路由到组合型worker"""
    router.kvcache_manager.find_matches.return_value = {
        "combined-1": 0.8,
        "combined-2": 0.4,
    }
    router.worker_manager.get_healthy_groups.return_value = ["group-b"]
    router.worker_manager.is_combined_group.return_value = True

    combined_workers = [
        w for w in workers_with_groups_for_router if w.worker_type == WorkerType.COMBINED
    ]
    router.worker_manager.get_workers_in_group.return_value = combined_workers

    prefill_id, decode_id = router.route_to_workers(route_hint)

    assert prefill_id is not None


@pytest.mark.asyncio
async def test_router_route_to_workers_records_metrics(
    router, route_hint, workers_with_groups_for_router
):
    """测试路由到worker对时记录指标"""
    from unittest.mock import patch

    router.kvcache_manager.find_matches.return_value = {
        "prefill-1": 0.5,
        "decode-1": 0.3,
    }
    router.worker_manager.get_healthy_groups.return_value = ["group-a"]
    router.worker_manager.is_combined_group.return_value = False

    prefill_workers = [
        w for w in workers_with_groups_for_router if w.worker_type == WorkerType.PREFILL
    ]
    decode_workers = [
        w for w in workers_with_groups_for_router if w.worker_type == WorkerType.DECODE
    ]

    router.worker_manager.get_prefill_workers_in_group.return_value = prefill_workers
    router.worker_manager.get_decode_workers_in_group.return_value = decode_workers

    def _find_worker(worker_id):
        return next(
            (w for w in workers_with_groups_for_router if w.worker_id == worker_id), None
        )

    router.worker_manager.get_worker.side_effect = _find_worker

    with patch("openjiuwentools.infer_router.routing.router.metrics") as mock_metrics:
        prefill_id, decode_id = router.route_to_workers(route_hint)

        assert prefill_id is not None
        assert decode_id is not None

        # 验证 record_routed_request 被调用
        assert mock_metrics.record_routed_request.call_count >= 2
        mock_metrics.record_routed_request.assert_any_call("prefill-1", "test-model")
        mock_metrics.record_routed_request.assert_any_call("decode-1", "test-model")


@pytest.mark.asyncio
async def test_router_route_to_workers_no_workers(router, route_hint):
    """测试无可用worker的情况"""
    router.worker_manager.get_healthy_groups.return_value = []

    prefill_id, decode_id = router.route_to_workers(route_hint)

    assert prefill_id is None
    assert decode_id is None


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

    router.worker_manager.get_healthy_groups.return_value = ["group-a"]
    router.worker_manager.is_combined_group.return_value = False
    router.worker_manager.get_prefill_workers_in_group.return_value = prefill_workers
    router.worker_manager.get_decode_workers_in_group.return_value = decode_workers

    prefill_id, decode_id = router.find_best_worker_pair("test-model", route_hint)

    assert prefill_id is not None
    assert decode_id is not None


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

    router.worker_manager.get_healthy_groups.return_value = ["group-b"]
    router.worker_manager.is_combined_group.return_value = True
    router.worker_manager.get_workers_in_group.return_value = combined_workers

    worker_id = router.find_best_combined_worker("test-model", route_hint)

    assert worker_id is not None
