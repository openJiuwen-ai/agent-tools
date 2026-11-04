"""Router 跨 Worker KV block 传输编排。"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.worker.worker_manager import WorkerManager


async def orchestrate_cross_worker_transfer(
    *,
    target_worker_id: str,
    token_ids: list[int],
    request_id: str,
    config: Any,
    kvcache_manager: KVCacheManager,
    worker_manager: WorkerManager,
    submitted_at: float | None = None,
) -> dict:
    """为一次推理请求执行有界的跨 Worker KV block 传输。

    参数:
        target_worker_id: Router 已选定、即将接收推理请求的目标 Worker ID。
        token_ids: 当前 prompt 的 token ID 列表，用于查找连续前缀 block。
        request_id: 当前推理请求 ID，用于关联 Router 与 Worker 事务日志。
        config: Router 配置对象，提供传输开关、block 上限和超时。
        kvcache_manager: Router KV 索引，用于选择源 Worker 和物理 block。
        worker_manager: Worker 管理器，用于建链并调用目标事务接口。
        submitted_at: 调用方把协程提交到事件循环前记录的 perf_counter；为空时
            表示调用方已在当前事件循环中，用于计算事件循环排队耗时。

    返回:
        成功时返回目标 Worker 的 pull-and-publish 结果；没有候选时返回
        ok=True、skipped=True；失败时返回 ok=False，调用方应继续转发并重算。
    """

    orchestration_started = time.perf_counter()
    phase_ms = {
        "transfer_event_loop_queue_ms": max(
            (orchestration_started - submitted_at) * 1000.0,
            0.0,
        ) if submitted_at is not None else 0.0,
        "transfer_candidate_ms": 0.0,
        "router_index_publish_ms": 0.0,
        "router_index_lock_wait_ms": 0.0,
        "router_index_update_ms": 0.0,
    }
    target_prefix_snapshot = None

    def _finish(result: dict) -> dict:
        """补齐 Router 编排耗时并返回结果。

        参数:
            result: 当前编排分支生成的结果字典。

        返回:
            合并 Worker 事务耗时和 Router 外层耗时后的结果字典。
        """

        worker_phase_ms = result.get("phase_ms")
        if isinstance(worker_phase_ms, dict):
            phase_ms.update(worker_phase_ms)
        orchestration_wall_ms = max(
            (time.perf_counter() - orchestration_started) * 1000.0,
            0.0,
        )
        phase_ms["transfer_orchestration_wall_ms"] = orchestration_wall_ms
        accounted_ms = (
            float(phase_ms.get("transfer_candidate_ms", 0.0))
            + float(phase_ms.get("router_transfer_total", 0.0))
            + float(phase_ms.get("router_index_publish_ms", 0.0))
        )
        phase_ms["transfer_orchestration_unattributed_ms"] = max(
            orchestration_wall_ms - accounted_ms,
            0.0,
        )
        result["phase_ms"] = phase_ms
        if target_prefix_snapshot is not None:
            result["_target_prefix_snapshot"] = target_prefix_snapshot
        return result

    if (
        not getattr(config, "kvbm_router_transfer_enabled", False)
        or not token_ids
    ):
        return _finish({"ok": True, "skipped": True, "reason": "disabled_or_empty"})
    target = worker_manager.get_worker(target_worker_id)
    if target is None:
        return _finish(
            {"ok": False, "skipped": True, "error": "target worker not found"}
        )
    healthy_worker_ids = [
        worker.worker_id
        for worker in worker_manager.get_healthy_workers(target.model)
    ]
    candidate_started = time.perf_counter()
    plan = kvcache_manager.plan_cross_worker_transfer(
        target_worker_id,
        token_ids,
        candidate_worker_ids=healthy_worker_ids,
        limit=max(int(getattr(config, "kvbm_router_transfer_max_blocks", 8)), 0),
    )
    target_prefix_snapshot = plan.target_snapshot
    candidate = plan.candidate
    phase_ms["transfer_candidate_ms"] = max(
        (time.perf_counter() - candidate_started) * 1000.0,
        0.0,
    )
    if candidate is None:
        diagnostics = plan.diagnostics
        logger.info(
            "KVBM cross-worker transfer skipped target={} reason=no_npu_source "
            "diagnostics={}",
            target_worker_id,
            diagnostics,
        )
        return _finish({
            "ok": True,
            "skipped": True,
            "reason": "no_npu_source",
            "diagnostics": diagnostics,
        })

    logger.info(
        "KVBM cross-worker transfer candidate request={} source={} target={} "
        "source_matched_blocks={} source_npu_blocks={} "
        "physical_mapped_npu_blocks={} target_matched_blocks={} "
        "target_npu_blocks={} selected_blocks={} first_unselected_reason={}",
        request_id,
        candidate.source_worker_id,
        target_worker_id,
        candidate.source_matched_blocks,
        candidate.source_npu_blocks,
        candidate.physical_mapped_npu_blocks,
        candidate.target_matched_blocks,
        candidate.target_npu_blocks,
        len(candidate.source_block_ids),
        candidate.first_unselected_reason,
    )
    result = await worker_manager.transfer_kv_blocks(
        target_worker_id,
        candidate.source_worker_id,
        list(candidate.source_block_ids),
        list(candidate.block_hashes),
        request_id=request_id,
        timeout_ms=int(getattr(config, "kvbm_router_transfer_timeout_ms", 3000)),
        link_timeout_ms=int(getattr(config, "kvbm_router_link_timeout_ms", 3000)),
        status_ttl_seconds=float(
            getattr(config, "kvbm_router_datadist_status_ttl_seconds", 10.0)
        ),
        source_lease_ttl_ms=int(
            getattr(config, "kvbm_router_source_lease_ttl_ms", 30000)
        ),
        reservation_ttl_ms=int(
            getattr(config, "kvbm_router_reservation_ttl_ms", 120000)
        ),
    )
    destination_blocks = result.get("destination_blocks") or []
    if result.get("already_published") and not destination_blocks:
        destination_blocks = result.get("existing_block_ids") or []
    router_updated_blocks = 0
    if result.get("ok") and len(destination_blocks) == len(candidate.block_hashes):
        index_phase_ms: dict[str, float] = {}
        index_started = time.perf_counter()
        router_updated_blocks = kvcache_manager.record_cross_worker_transfer(
            candidate.source_worker_id,
            target_worker_id,
            candidate.block_hashes,
            destination_blocks,
            timing=index_phase_ms,
        )
        phase_ms.update(index_phase_ms)
        phase_ms["router_index_publish_ms"] = max(
            (time.perf_counter() - index_started) * 1000.0,
            0.0,
        )
    result["source_worker_id"] = candidate.source_worker_id
    result["target_worker_id"] = target_worker_id
    result["router_updated_blocks"] = router_updated_blocks
    result = _finish(result)
    logger.info(
        "KVBM cross-worker transfer finished request={} source={} target={} "
        "blocks={} ok={} published_count={} router_updated_blocks={} "
        "phase_ms={} error={}",
        request_id,
        candidate.source_worker_id,
        target_worker_id,
        len(candidate.source_block_ids),
        result.get("ok", False),
        result.get("published_count", 0),
        router_updated_blocks,
        result.get("phase_ms", {}),
        result.get("error", ""),
    )
    return result
