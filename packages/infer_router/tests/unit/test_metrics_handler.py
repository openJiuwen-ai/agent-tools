import time
from unittest.mock import Mock

from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType
from openjiuwentools.infer_router.worker.metrics_handler import MetricsHandler


def test_metrics_handler_records_decode_block_capacity():
    """Worker KV指标应保留D block预算准入需要的容量信息。"""
    worker = WorkerInfo(
        worker_id="decode-0",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    manager = Mock()
    manager.workers = {worker.worker_id: worker}
    updated = Mock()
    handler = MetricsHandler(manager, on_metrics_updated=updated)

    handler(
        worker.worker_id,
        {
            "kv_cache_usage": 0.75,
            "num_gpu_block": 1000,
            "active_decode_blocks": 750,
            "num_running_reqs": 10,
            "num_waiting_reqs": 2,
            "max_num_batched_tokens": 8192,
        },
    )

    assert worker.num_gpu_blocks == 1000
    assert worker.active_decode_blocks == 750
    assert worker.kv_cache_usage == 0.75
    assert worker.running_requests == 10
    assert worker.queue_depth == 2
    assert worker.max_num_batched_tokens == 8192
    updated.assert_called_once_with()


def test_metrics_handler_records_engine_core_free_blocks():
    """KVBM容量指标应独立记录真实BlockPool空闲量及接收时间。"""
    worker = WorkerInfo(
        worker_id="decode-capacity",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    manager = Mock()
    manager.workers = {worker.worker_id: worker}
    handler = MetricsHandler(manager)
    before = time.monotonic()

    handler(
        worker.worker_id,
        {"kvbm_free_blocks": 21, "kvbm_block_size": 128},
    )

    assert worker.kvbm_free_blocks == 21
    assert worker.kvbm_block_size == 128
    assert worker.kvbm_block_pool_updated_at >= before
