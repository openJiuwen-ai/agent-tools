"""Worker 负载指标处理器，作为 WorkerEventSubscriber 的 topic handler 注册。

接收 worker 上报的 kv_metrics 消息，更新 WorkerManager 中对应 worker 的负载信息。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from openjiuwentools.infer_router.worker.worker_manager import WorkerManager


class MetricsHandler:
    """处理 kv_metrics topic 的消息，更新 worker 负载信息。"""

    def __init__(self, worker_manager: WorkerManager, on_metrics_updated: callable = None) -> None:
        self.worker_manager = worker_manager
        self._on_metrics_updated = on_metrics_updated

    def __call__(self, worker_id: str, msg: dict) -> None:
        kv_cache_usage = msg.get("kv_cache_usage", 0.0)

        if worker_id not in self.worker_manager.workers:
            logger.warning(f"[METRICS] Unknown worker {worker_id}, skipping metrics update")
            return

        worker = self.worker_manager.workers[worker_id]
        if "current_load" in msg:
            worker.current_load = msg["current_load"]
        elif "kv_cache_usage" in msg:
            worker.current_load = kv_cache_usage * 100.0
        if "num_waiting_reqs" in msg:
            worker.queue_depth = msg["num_waiting_reqs"]
        if "num_running_reqs" in msg:
            worker.running_requests = msg["num_running_reqs"]
        if "num_gpu_block" in msg:
            worker.num_gpu_blocks = max(int(msg["num_gpu_block"]), 0)
        if "active_decode_blocks" in msg:
            worker.active_decode_blocks = max(int(msg["active_decode_blocks"]), 0)
        if "kv_cache_usage" in msg:
            worker.kv_cache_usage = min(max(float(kv_cache_usage), 0.0), 1.0)
        if "kvbm_free_blocks" in msg:
            generation = int(msg.get("kvbm_capacity_generation", -1))
            current_generation = int(
                getattr(worker, "kvbm_capacity_generation", -1)
            )
            if (
                generation >= current_generation
                and (generation >= 0 or current_generation < 0)
            ):
                worker.kvbm_free_blocks = max(int(msg["kvbm_free_blocks"]), 0)
                worker.kvbm_block_pool_updated_at = time.monotonic()
                if generation >= 0:
                    worker.kvbm_capacity_generation = generation
        if "kvbm_block_size" in msg:
            worker.kvbm_block_size = max(int(msg["kvbm_block_size"]), 0)
        if "max_num_batched_tokens" in msg:
            worker.max_num_batched_tokens = max(
                int(msg["max_num_batched_tokens"]), 0
            )

        if "kv_cache_usage" in msg:
            logger.info(
                f"[METRICS] {worker_id}: load={worker.current_load:.1f}%, "
                f"running={worker.running_requests}, waiting={worker.queue_depth}, "
                f"kv_usage={kv_cache_usage:.3f}, blocks="
                f"{worker.active_decode_blocks}/{worker.num_gpu_blocks}, "
                f"max_batched_tokens={worker.max_num_batched_tokens}"
            )
        elif "kvbm_free_blocks" in msg:
            logger.debug(
                "[KVBM_BLOCK_POOL] {}: free_blocks={}",
                worker_id,
                worker.kvbm_free_blocks,
            )

        if self._on_metrics_updated:
            self._on_metrics_updated()
