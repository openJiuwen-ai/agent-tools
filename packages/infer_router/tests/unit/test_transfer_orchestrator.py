import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from openjiuwentools.infer_router.kvbm.transfer_orchestrator import (
    orchestrate_cross_worker_transfer,
)


def test_transfer_orchestrator_reports_router_phase_timings():
    """跨 Worker 编排应返回事件循环、候选选择和外层总耗时。"""

    candidate = SimpleNamespace(
        source_worker_id="worker-source",
        source_block_ids=(7, 8),
        block_hashes=("hash-7", "hash-8"),
        source_matched_blocks=2,
        source_npu_blocks=2,
        physical_mapped_npu_blocks=2,
        target_matched_blocks=0,
        target_npu_blocks=0,
        first_unselected_reason="",
    )
    cache_manager = Mock()
    target_snapshot = SimpleNamespace(worker_id="worker-target")
    cache_manager.plan_cross_worker_transfer.return_value = SimpleNamespace(
        candidate=candidate,
        diagnostics={},
        target_snapshot=target_snapshot,
    )
    cache_manager.record_cross_worker_transfer.return_value = 2
    worker_manager = Mock()
    worker_manager.get_worker.return_value = SimpleNamespace(model="test-model")
    worker_manager.get_healthy_workers.return_value = [
        SimpleNamespace(worker_id="worker-source")
    ]
    worker_manager.transfer_kv_blocks = AsyncMock(return_value={
        "ok": True,
        "destination_blocks": [17, 18],
        "published_count": 2,
        "phase_ms": {"router_transfer_total": 12.5},
    })
    config = SimpleNamespace(
        kvbm_router_transfer_enabled=True,
        kvbm_router_transfer_max_blocks=8,
        kvbm_router_transfer_timeout_ms=3000,
        kvbm_router_link_timeout_ms=3000,
        kvbm_router_datadist_status_ttl_seconds=10.0,
        kvbm_router_source_lease_ttl_ms=30000,
        kvbm_router_reservation_ttl_ms=120000,
    )

    result = asyncio.run(orchestrate_cross_worker_transfer(
        target_worker_id="worker-target",
        token_ids=list(range(256)),
        request_id="request-timing",
        config=config,
        kvcache_manager=cache_manager,
        worker_manager=worker_manager,
        submitted_at=time.perf_counter() - 0.001,
    ))

    phase_ms = result["phase_ms"]
    assert phase_ms["transfer_event_loop_queue_ms"] >= 1.0
    assert phase_ms["transfer_candidate_ms"] >= 0.0
    assert phase_ms["transfer_orchestration_wall_ms"] >= 0.0
    assert phase_ms["transfer_orchestration_unattributed_ms"] >= 0.0
    assert phase_ms["router_index_publish_ms"] >= 0.0
    assert phase_ms["router_index_lock_wait_ms"] >= 0.0
    assert phase_ms["router_index_update_ms"] >= 0.0
    assert phase_ms["router_transfer_total"] == 12.5
    assert result["router_updated_blocks"] == 2
    assert result["_target_prefix_snapshot"] is target_snapshot
    cache_manager.plan_cross_worker_transfer.assert_called_once()


def test_transfer_orchestrator_reuses_plan_diagnostics_without_rescan():
    """候选为空时应直接复用规划诊断，不能再次扫描 Worker 前缀。

    参数:
        无。

    返回:
        无返回值；调用旧诊断接口或结果不完整时由断言报告失败。
    """

    target_snapshot = SimpleNamespace(worker_id="worker-target")
    diagnostics = {
        "target_matched_blocks": 49,
        "target_npu_blocks": 49,
        "sources": [],
    }
    cache_manager = Mock()
    cache_manager.plan_cross_worker_transfer.return_value = SimpleNamespace(
        candidate=None,
        diagnostics=diagnostics,
        target_snapshot=target_snapshot,
    )
    worker_manager = Mock()
    worker_manager.get_worker.return_value = SimpleNamespace(model="test-model")
    worker_manager.get_healthy_workers.return_value = []
    config = SimpleNamespace(
        kvbm_router_transfer_enabled=True,
        kvbm_router_transfer_max_blocks=8,
    )

    result = asyncio.run(
        orchestrate_cross_worker_transfer(
            target_worker_id="worker-target",
            token_ids=list(range(256)),
            request_id="request-no-transfer",
            config=config,
            kvcache_manager=cache_manager,
            worker_manager=worker_manager,
        )
    )

    assert result["skipped"] is True
    assert result["diagnostics"] is diagnostics
    assert result["_target_prefix_snapshot"] is target_snapshot
    cache_manager.plan_cross_worker_transfer.assert_called_once()
    cache_manager.diagnose_cross_worker_transfer.assert_not_called()
