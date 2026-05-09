from loguru import logger

from openjiuwentools.infer_router.config.config import settings
from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.monitoring.metrics import metrics
from openjiuwentools.infer_router.routing.load_balance import (
    RoundRobinAlgorithm,
    WorkloadWeightedAlgorithm,
)
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint
from openjiuwentools.infer_router.worker.worker_manager import WorkerManager


class Router:
    """路由决策层，基于路由提示和KV缓存状态选择最佳推理服务节点"""

    def __init__(self, kvcache_manager: KVCacheManager, worker_manager: WorkerManager):
        self.kvcache_manager = kvcache_manager
        self.worker_manager = worker_manager

        self.algorithms = {
            "round_robin": RoundRobinAlgorithm(),
            "weighted": WorkloadWeightedAlgorithm(kvcache_manager, worker_manager),
        }
        self.current_algorithm = self.algorithms.get(settings.load_balancing_algorithm, RoundRobinAlgorithm())

    def set_algorithm(self, algorithm_name: str):
        """设置负载均衡算法"""
        if algorithm_name in self.algorithms:
            self.current_algorithm = self.algorithms[algorithm_name]
            logger.info(f"Load balancing algorithm changed to {algorithm_name}")
        else:
            logger.warning(f"Unknown algorithm {algorithm_name}, using round_robin instead")
            self.current_algorithm = RoundRobinAlgorithm()

    def _check_session_affinity(self, session_id: str | None) -> str | None:
        """检查会话亲和性"""
        if not session_id:
            return None

        affinity_worker = self.kvcache_manager.get_session_affinity(session_id)
        if not affinity_worker:
            return None

        worker = self.worker_manager.get_worker(affinity_worker)
        if not worker:
            return None

        status = self.worker_manager.get_worker_status(affinity_worker)
        if not status or not status.is_healthy:
            return None

        logger.info(f"Session affinity: routing to worker {affinity_worker}")
        return affinity_worker

    def find_best_worker_pair(self, model: str, route_hint: RouteHint) -> tuple[str | None, str | None]:
        """筛选出可用的非组合型group的workers，由路由算法选择最佳的prefill-decode节点对

        Returns:
            (prefill_worker_id, decode_worker_id) - 最佳工作器对
            (None, None) - 如果没有可用工作器
        """
        healthy_groups = self.worker_manager.get_healthy_groups(model)
        if not healthy_groups:
            return None, None

        filtered_groups = {}
        all_workers = {}

        for group in healthy_groups:
            if self.worker_manager.is_combined_group(group):
                continue

            prefill_workers = self.worker_manager.get_prefill_workers_in_group(group, model)
            decode_workers = self.worker_manager.get_decode_workers_in_group(group, model)

            if not prefill_workers or not decode_workers:
                continue

            worker_ids = []
            for w in prefill_workers:
                all_workers[w.worker_id] = w
                worker_ids.append(w.worker_id)
            for w in decode_workers:
                if w.worker_id not in all_workers:
                    all_workers[w.worker_id] = w
                    worker_ids.append(w.worker_id)

            filtered_groups[group] = worker_ids

        if not filtered_groups:
            logger.warning("No valid non-combined groups found")
            return None, None

        try:
            prefill_worker, decode_worker = self.current_algorithm.select_worker_pair(
                workers=all_workers,
                groups=filtered_groups,
                route_hint=route_hint,
            )
            logger.info(f"Selected best worker pair: {prefill_worker.worker_id} + {decode_worker.worker_id}")
            return prefill_worker.worker_id, decode_worker.worker_id
        except ValueError as e:
            logger.warning(f"No valid worker pair found: {e}")
            return None, None

    def find_best_combined_worker(self, model: str, route_hint: RouteHint) -> str | None:
        """筛选出可用的组合型workers，由路由算法选择最佳的组合型worker"""
        healthy_groups = self.worker_manager.get_healthy_groups(model)
        if not healthy_groups:
            return None

        combined_workers = []

        for group in healthy_groups:
            if not self.worker_manager.is_combined_group(group):
                continue

            workers = self.worker_manager.get_workers_in_group(group)
            model_workers = [w for w in workers if w.model == model]
            combined_workers.extend(model_workers)

        if not combined_workers:
            logger.warning("No valid combined workers found")
            return None

        try:
            best_worker = self.current_algorithm.select_worker(combined_workers, route_hint)
            logger.info(f"Selected best combined worker: {best_worker.worker_id}")
            return best_worker.worker_id
        except ValueError as e:
            logger.warning(f"No valid combined worker found: {e}")
            return None

    def route_to_workers(
        self,
        route_hint: RouteHint,
        session_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        """路由决策逻辑 - 选择最佳的worker对（prefill + decode）或组合型worker

        session_affinity 和 cache_aware 逻辑由负载均衡算法统一处理。

        Returns:
            (prefill_worker_id, decode_worker_id) - 如果是非组合型
            (combined_worker_id, None) - 如果是组合型
            (None, None) - 如果没有可用worker
        """
        logger.info(f"[ROUTER: START] Routing request {route_hint.request_id} to worker pair")

        prefill_id, decode_id = self.find_best_worker_pair(route_hint.model, route_hint)
        if prefill_id and decode_id:
            if session_id:
                self.kvcache_manager.set_session_affinity(session_id, prefill_id)

            prefill_worker = self.worker_manager.get_worker(prefill_id)
            decode_worker = self.worker_manager.get_worker(decode_id)
            if prefill_worker:
                metrics.record_routed_request(prefill_worker.worker_id, route_hint.model)
            if decode_worker:
                metrics.record_routed_request(decode_worker.worker_id, route_hint.model)

            return prefill_id, decode_id

        combined_id = self.find_best_combined_worker(route_hint.model, route_hint)
        if combined_id:
            if session_id:
                self.kvcache_manager.set_session_affinity(session_id, combined_id)

            combined_worker = self.worker_manager.get_worker(combined_id)
            if combined_worker:
                metrics.record_routed_request(combined_worker.worker_id, route_hint.model)

            return combined_id, None

        logger.error(f"No available workers for model {route_hint.model}")
        return None, None
