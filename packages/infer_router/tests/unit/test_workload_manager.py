"""测试工作器负载管理器"""

import pytest

from openjiuwentools.infer_router.kv_cache.event import CacheEvent
from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType
from openjiuwentools.infer_router.worker.workload_manager import WorkloadManager


@pytest.fixture
def prefill_worker():
    """创建prefill类型工作器"""
    return WorkerInfo(
        worker_id="prefill-worker-1",
        model="test-model",
        url="http://localhost:8001/v1",
        total_tokens=1000000,
        current_load=0.0,
        cached_prefixes=[],
        worker_type=WorkerType.PREFILL,
        group="group-a",
    )


@pytest.fixture
def decode_worker():
    """创建decode类型工作器"""
    return WorkerInfo(
        worker_id="decode-worker-1",
        model="test-model",
        url="http://localhost:8002/v1",
        total_tokens=1000000,
        current_load=0.0,
        cached_prefixes=[],
        worker_type=WorkerType.DECODE,
        group="group-a",
    )


@pytest.fixture
def combined_worker():
    """创建combined类型工作器"""
    return WorkerInfo(
        worker_id="combined-worker-1",
        model="test-model",
        url="http://localhost:8003/v1",
        total_tokens=1000000,
        current_load=0.0,
        cached_prefixes=[],
        worker_type=WorkerType.COMBINED,
        group="group-b",
    )


def test_workload_manager_enabled_by_default():
    """测试workload manager默认启用"""
    wm = WorkloadManager()
    assert wm.enabled is True


def test_workload_manager_disabled():
    """测试workload manager可以被禁用"""
    wm = WorkloadManager(enabled=False)
    assert wm.enabled is False


def test_process_cache_events():
    """测试批量处理缓存事件"""
    wm = WorkloadManager()

    events = [
        CacheEvent(
            event_type="prefill_start",
            block_hashes=[],
            token_count=500,
            token_ids=[1, 2, 3, 4, 5],
            worker_id="test-worker",
        ),
        CacheEvent(
            event_type="decode_start",
            block_hashes=[],
            token_count=300,
            token_ids=[6, 7, 8],
            worker_id="test-worker",
            engine_specific={"osl": 200},
        ),
    ]
    wm.process_cache_events(events)

    workload = wm.get_workload("test-worker")
    assert workload is not None
    assert workload.pending_tokens == 800
    assert workload.pending_osl == 200


def test_process_cache_event_disabled():
    """测试workload manager禁用时不处理事件"""
    wm = WorkloadManager(enabled=False)

    event = CacheEvent(
        event_type="store",
        block_hashes=["hash1"],
        token_count=10,
        token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        worker_id="test-worker",
    )
    wm.process_cache_event(event)

    workload = wm.get_workload("test-worker")
    assert workload is None


def test_process_prefill_start_event():
    """测试处理prefill_start事件"""
    wm = WorkloadManager()

    event = CacheEvent(
        event_type="prefill_start",
        block_hashes=[],
        token_count=1000,
        token_ids=[1, 2, 3],
        worker_id="test-worker",
    )
    wm.process_cache_event(event)

    workload = wm.get_workload("test-worker")
    assert workload is not None
    assert workload.pending_tokens == 1000


def test_process_prefill_end_event():
    """测试处理prefill_end事件"""
    wm = WorkloadManager()

    start_event = CacheEvent(
        event_type="prefill_start",
        block_hashes=[],
        token_count=1000,
        token_ids=[1, 2, 3],
        worker_id="test-worker",
    )
    wm.process_cache_event(start_event)

    end_event = CacheEvent(
        event_type="prefill_end",
        block_hashes=[],
        token_count=1000,
        token_ids=[1, 2, 3],
        worker_id="test-worker",
    )
    wm.process_cache_event(end_event)

    workload = wm.get_workload("test-worker")
    assert workload is not None
    assert workload.pending_tokens == 0


def test_process_decode_start_event():
    """测试处理decode_start事件"""
    wm = WorkloadManager()

    event = CacheEvent(
        event_type="decode_start",
        block_hashes=[],
        token_count=1000,
        token_ids=[1, 2, 3],
        worker_id="test-worker",
        engine_specific={"osl": 500},
    )
    wm.process_cache_event(event)

    workload = wm.get_workload("test-worker")
    assert workload is not None
    assert workload.pending_tokens == 1000
    assert workload.pending_osl == 500


def test_process_decode_end_event():
    """测试处理decode_end事件"""
    wm = WorkloadManager()

    start_event = CacheEvent(
        event_type="decode_start",
        block_hashes=[],
        token_count=1000,
        token_ids=[1, 2, 3],
        worker_id="test-worker",
        engine_specific={"osl": 500},
    )
    wm.process_cache_event(start_event)

    end_event = CacheEvent(
        event_type="decode_end",
        block_hashes=[],
        token_count=1000,
        token_ids=[1, 2, 3],
        worker_id="test-worker",
        engine_specific={"osl": 500},
    )
    wm.process_cache_event(end_event)

    workload = wm.get_workload("test-worker")
    assert workload is not None
    assert workload.pending_tokens == 0
    assert workload.pending_osl == 0


def test_calculate_load_prefill(prefill_worker):
    """测试计算prefill节点的负载"""
    wm = WorkloadManager()

    prefill_start_event = CacheEvent(
        event_type="prefill_start",
        block_hashes=[],
        token_count=400000,
        token_ids=[],
        worker_id=prefill_worker.worker_id,
    )
    wm.process_cache_event(prefill_start_event)

    load = wm.calculate_load(prefill_worker)

    expected_load = (400000 / 1000000) * 100
    assert load == pytest.approx(expected_load)


def test_calculate_load_decode(decode_worker):
    """测试计算decode节点的负载"""
    wm = WorkloadManager()

    decode_start_event = CacheEvent(
        event_type="decode_start",
        block_hashes=[],
        token_count=100000,
        token_ids=[],
        worker_id=decode_worker.worker_id,
        engine_specific={"osl": 200000},
    )
    wm.process_cache_event(decode_start_event)

    load = wm.calculate_load(decode_worker)

    expected_load = ((100000 + 200000) / 1000000) * 100
    assert load == pytest.approx(expected_load)


def test_calculate_load_combined(combined_worker):
    """测试计算combined节点的负载"""
    wm = WorkloadManager()

    decode_start_event = CacheEvent(
        event_type="decode_start",
        block_hashes=[],
        token_count=200000,
        token_ids=[],
        worker_id=combined_worker.worker_id,
        engine_specific={"osl": 100000},
    )
    wm.process_cache_event(decode_start_event)

    load = wm.calculate_load(combined_worker)

    prefill_load = (200000 / 1000000) * 100
    decode_load = ((200000 + 100000) / 1000000) * 100
    expected_load = (prefill_load + decode_load) / 2
    assert load == pytest.approx(expected_load)


def test_calculate_load_disabled(prefill_worker):
    """测试workload manager禁用时返回原始current_load"""
    prefill_worker.current_load = 25.5
    wm = WorkloadManager(enabled=False)

    load = wm.calculate_load(prefill_worker)

    assert load == 25.5


def test_remove_worker():
    """测试移除工作器的负载信息"""
    wm = WorkloadManager()

    prefill_start_event = CacheEvent(
        event_type="prefill_start",
        block_hashes=[],
        token_count=1000,
        token_ids=[1, 2, 3],
        worker_id="worker-1",
    )
    wm.process_cache_event(prefill_start_event)
    wm.remove_worker("worker-1")

    workload = wm.get_workload("worker-1")
    assert workload is None


def test_reset():
    """测试重置所有负载信息"""
    wm = WorkloadManager()

    prefill_start_event1 = CacheEvent(
        event_type="prefill_start",
        block_hashes=[],
        token_count=1000,
        token_ids=[1, 2, 3],
        worker_id="worker-1",
    )
    prefill_start_event2 = CacheEvent(
        event_type="prefill_start",
        block_hashes=[],
        token_count=2000,
        token_ids=[4, 5, 6],
        worker_id="worker-2",
    )
    wm.process_cache_event(prefill_start_event1)
    wm.process_cache_event(prefill_start_event2)
    wm.reset()

    assert wm.get_workload("worker-1") is None
    assert wm.get_workload("worker-2") is None


def test_get_stats():
    """测试获取负载统计"""
    wm = WorkloadManager()

    prefill_start_event = CacheEvent(
        event_type="prefill_start",
        block_hashes=[],
        token_count=1000,
        token_ids=[],
        worker_id="worker-1",
    )
    decode_start_event = CacheEvent(
        event_type="decode_start",
        block_hashes=[],
        token_count=2000,
        token_ids=[],
        worker_id="worker-2",
        engine_specific={"osl": 500},
    )
    wm.process_cache_events([prefill_start_event, decode_start_event])

    stats = wm.get_stats()

    assert stats["worker-1"] == (1000, 0)
    assert stats["worker-2"] == (2000, 500)


def test_update_throughput():
    """测试更新吞吐速率"""
    wm = WorkloadManager()

    wm.update_throughput("worker-1", 1000, 500)
    wm.update_throughput("worker-1", 2000, 1000)

    workload = wm.get_workload("worker-1")
    assert workload is not None
    assert workload.prompt_ratio > 0
    assert workload.completion_ratio > 0


def test_update_throughput_disabled():
    """测试workload manager禁用时不更新吞吐速率"""
    wm = WorkloadManager(enabled=False)

    wm.update_throughput("worker-1", 1000, 500)

    workload = wm.get_workload("worker-1")
    assert workload is None


def test_throughput_ratios_initial_values():
    """测试吞吐速率初始值"""
    wm = WorkloadManager()

    workload = wm.get_or_create_workload("worker-1")
    assert workload.prompt_ratio == 1000.0
    assert workload.completion_ratio == 100.0


def test_throughput_window_filtering():
    """测试5分钟窗口过滤"""
    import time

    wm = WorkloadManager()

    wm.update_throughput("worker-1", 1000, 500)
    initial_prompt_ratio = wm.get_workload("worker-1").prompt_ratio
    initial_completion_ratio = wm.get_workload("worker-1").completion_ratio

    time.sleep(0.1)

    wm.update_throughput("worker-1", 1000, 500)
    new_prompt_ratio = wm.get_workload("worker-1").prompt_ratio
    new_completion_ratio = wm.get_workload("worker-1").completion_ratio

    assert new_prompt_ratio > initial_prompt_ratio
    assert new_completion_ratio > initial_completion_ratio


def test_throughput_ratios_min_value():
    """测试吞吐速率最小值（当没有记录时使用默认值）"""
    wm = WorkloadManager()

    workload = wm.get_or_create_workload("worker-1")
    workload.throughput_records = []
    wm.update_ratios(workload)

    assert workload.prompt_ratio == 1000.0
    assert workload.completion_ratio == 100.0
