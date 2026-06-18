"""Worker 负载指标处理器，作为 WorkerEventSubscriber 的 topic handler 注册。

接收 worker 上报的 kv_metrics 消息，更新 WorkerManager 中对应 worker 的负载信息。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from openjiuwentools.infer_router.worker.worker_manager import WorkerManager


class MetricsHandler:
    """处理 kv_metrics topic 的消息，更新 worker 负载信息。"""

    def __init__(self, worker_manager: WorkerManager) -> None:
        self.worker_manager = worker_manager

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

        logger.info(
            f"[METRICS] {worker_id}: load={worker.current_load:.1f}%, "
            f"running={worker.running_requests}, waiting={worker.queue_depth}, "
            f"kv_usage={kv_cache_usage:.3f}"
        )
