import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger

from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint


@dataclass
class ScheduledRequest:
    """调度请求数据类"""

    route_hint: RouteHint
    arrival_time: float
    priority: int
    estimated_output_tokens: int
    token_ids: list[int] | None = None
    timeout: float = 30.0
    cancel_event: asyncio.Event | None = None
    overlap_scores: dict[str, int] | None = None

    @property
    def tokens_to_process(self) -> int:
        """计算需要处理的token数 = 输入token数 - KV缓存命中token数"""
        if not self.route_hint.token_ids:
            return 0
        input_tokens = len(self.route_hint.token_ids)
        if not self.overlap_scores:
            return input_tokens
        hit_tokens = max(self.overlap_scores.values()) if self.overlap_scores else 0
        return max(0, input_tokens - hit_tokens)


class SchedulingStrategy(ABC):
    """调度策略抽象基类"""

    @abstractmethod
    def select_next(self, queue: list[ScheduledRequest]) -> ScheduledRequest:
        """选择下一个要执行的请求"""
        pass


class FCFSStrategy(SchedulingStrategy):
    """First-Come-First-Served 策略"""

    def select_next(self, queue: list[ScheduledRequest]) -> ScheduledRequest:
        return min(queue, key=lambda x: x.arrival_time)


class LCFSStrategy(SchedulingStrategy):
    """Last-Come-First-Served 策略"""

    def select_next(self, queue: list[ScheduledRequest]) -> ScheduledRequest:
        return max(queue, key=lambda x: x.arrival_time)


class WSPTStrategy(SchedulingStrategy):
    """Weighted Shortest Processing Time 策略

    收益评估公式：eval = (1 + priority) / 待计算的token数
    priority 来自 route_hint
    待计算的token数 = 用户请求输入token数 - kv_cache中已经命中的token数
    """

    def select_next(self, queue: list[ScheduledRequest]) -> ScheduledRequest:
        def wspt_score(request: ScheduledRequest) -> float:
            tokens = max(1, request.tokens_to_process)
            return (1 + request.priority) / tokens

        return max(queue, key=wspt_score)


class Scheduler:
    """调度系统，应用优先级策略管理全局请求队列"""

    def __init__(self, kvcache_manager: KVCacheManager, strategy: str = None):
        self.kvcache_manager = kvcache_manager
        self.global_queue: list[ScheduledRequest] = []
        self.strategies = {
            "FCFS": FCFSStrategy(),
            "LCFS": LCFSStrategy(),
            "WSPT": WSPTStrategy(),
        }
        from openjiuwentools.infer_router.config.config import settings

        default_strategy = strategy or settings.default_scheduling_strategy
        self.current_strategy = self.strategies.get(default_strategy, FCFSStrategy())

        self.task_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "timed_out_tasks": 0,
        }

    def set_strategy(self, strategy: str):
        """设置调度策略"""
        if strategy in self.strategies:
            self.current_strategy = self.strategies[strategy]
            logger.info(f"Scheduling strategy changed to {strategy}")
        else:
            logger.warning(f"Unknown strategy {strategy}, using FCFS instead")
            self.current_strategy = FCFSStrategy()

    def submit(self, route_hint: RouteHint, overlap_scores: dict[str, int] | None = None) -> None:
        """提交请求到调度队列

        Args:
            route_hint: 路由提示
            overlap_scores: KV缓存重叠分数（工作器ID -> 匹配块数）

        """
        logger.info(f"[SCHEDULER: SUBMIT] Request {route_hint.request_id} submitted to scheduler")
        logger.debug(f"[SCHEDULER: SUBMIT] Route hint: {route_hint.model_dump()}")
        if overlap_scores:
            logger.debug(f"[SCHEDULER: SUBMIT] Overlap scores: {overlap_scores}")

        scheduled_request = ScheduledRequest(
            route_hint=route_hint,
            arrival_time=time.time(),
            priority=route_hint.priority,
            estimated_output_tokens=route_hint.estimated_output_tokens,
            token_ids=route_hint.token_ids,
            overlap_scores=overlap_scores,
        )

        self.global_queue.append(scheduled_request)

        from openjiuwentools.infer_router.monitoring import metrics

        metrics.record_scheduled_request(self.current_strategy.__class__.__name__, "global")
        metrics.update_queue_size("global", len(self.global_queue))

        self.task_stats["total_tasks"] += 1

        logger.info(
            f"[SCHEDULER: QUEUE] Request {route_hint.request_id} "
            f"added to global queue. Queue size: {len(self.global_queue)}"
        )

    def get_next_request(self) -> ScheduledRequest | None:
        """按调度策略获取下一个请求"""
        logger.info(f"[SCHEDULER: CHECK] Checking for next request in queue. Current size: {len(self.global_queue)}")
        if not self.global_queue:
            logger.info("[SCHEDULER: EMPTY] Queue is empty, no request available")
            return None

        logger.info(f"[SCHEDULER: SELECT] Selecting next request using {self.current_strategy.__class__.__name__}")
        next_request = self.current_strategy.select_next(self.global_queue)
        self.global_queue.remove(next_request)

        from openjiuwentools.infer_router.monitoring import metrics

        metrics.update_queue_size("global", len(self.global_queue))

        logger.info(
            f"[SCHEDULER: DISPATCH] Dispatching request {next_request.route_hint.request_id} "
            f"from global queue. Remaining queue size: {len(self.global_queue)}"
        )
        logger.debug(
            f"[SCHEDULER: DISPATCH] Selected request: "
            f"priority={next_request.priority}, "
            f"estimated_tokens={next_request.estimated_output_tokens}"
        )
        return next_request

    def get_queue_size(self) -> int:
        """获取全局队列大小"""
        return len(self.global_queue)

    def cancel_request(self, request_id: str) -> bool:
        """取消指定的请求"""
        for req in self.global_queue:
            if req.route_hint.request_id == request_id:
                if req.cancel_event:
                    req.cancel_event.set()
                self.global_queue.remove(req)
                logger.info(f"Cancelled request {request_id} from global queue")
                return True

        return False

    def get_queue_stats(self) -> dict:
        """获取队列统计信息"""
        if not self.global_queue:
            return {
                "queue_size": 0,
                "avg_priority": 0,
                "avg_estimated_tokens": 0,
                "oldest_request_age": 0,
            }

        current_time = time.time()
        avg_priority = sum(req.priority for req in self.global_queue) / len(self.global_queue)
        avg_estimated_tokens = sum(req.estimated_output_tokens for req in self.global_queue) / len(self.global_queue)
        oldest_request_age = current_time - min(req.arrival_time for req in self.global_queue)

        return {
            "queue_size": len(self.global_queue),
            "avg_priority": avg_priority,
            "avg_estimated_tokens": avg_estimated_tokens,
            "oldest_request_age": oldest_request_age,
        }

    def get_stats(self) -> dict:
        """获取调度器统计信息"""
        return {
            "total_tasks": self.task_stats["total_tasks"],
            "completed_tasks": self.task_stats["completed_tasks"],
            "failed_tasks": self.task_stats["failed_tasks"],
            "timed_out_tasks": self.task_stats["timed_out_tasks"],
            "total_queue_size": len(self.global_queue),
            "current_strategy": self.current_strategy.__class__.__name__,
        }

    def mark_task_completed(self):
        """标记任务完成"""
        self.task_stats["completed_tasks"] += 1

    def mark_task_failed(self):
        """标记任务失败"""
        self.task_stats["failed_tasks"] += 1

    def mark_task_timed_out(self):
        """标记任务超时"""
        self.task_stats["timed_out_tasks"] += 1
