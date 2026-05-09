import time

from loguru import logger

from openjiuwentools.infer_router.config.config import settings
from openjiuwentools.infer_router.kv_cache.event import CacheEvent
from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType


class WorkerWorkload:
    """单个工作器的负载信息"""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.pending_tokens: int = 0  # 当前待服务的请求总 token 数
        self.pending_osl: int = 0  # 当前待服务的请求总 osl 数
        self.prompt_ratio: float = 1000.0  # prompt_tokens的吞吐速率（tokens/秒），默认1000
        self.completion_ratio: float = 100.0  # completion_tokens的吞吐速率（tokens/秒），默认100
        self.throughput_records: list[tuple[float, int, int]] = []  # (timestamp, prompt_tokens, completion_tokens)


class WorkloadManager:
    """工作器负载管理器

    维护 worker 的负载信息，通过处理请求级别的 CacheEvent 来更新。

    通过请求级别的 CacheEvent 来更新待服务请求信息：
    1. prefill_start 事件: pending_tokens += token_count
    2. prefill_end 事件: pending_tokens -= token_count
    3. decode_start 事件: pending_tokens += token_count, pending_osl += osl
    4. decode_end 事件: pending_tokens -= token_count, pending_osl -= osl

    负载计算公式：
    - prefill 节点: current_load = pending_tokens / (4 * total_tokens) * 100
    - decode 节点: current_load = (pending_tokens + pending_osl) /
      (3 * total_tokens) * 100
    - combined 节点: (prefill_load + decode_load) / 2
    """

    def __init__(self, enabled: bool = True):
        self.workloads: dict[str, WorkerWorkload] = {}
        self._enabled = enabled
        logger.info(f"WorkloadManager initialized, enabled={self._enabled}")

    @property
    def enabled(self) -> bool:
        """是否启用负载管理"""
        return self._enabled

    def get_or_create_workload(self, worker_id: str) -> WorkerWorkload:
        """获取或创建工作器的负载信息"""
        if worker_id not in self.workloads:
            self.workloads[worker_id] = WorkerWorkload(worker_id)
        return self.workloads[worker_id]

    def process_cache_event(self, event: CacheEvent) -> None:
        """处理缓存事件，更新负载信息"""
        if not self._enabled:
            return

        if not event.worker_id:
            logger.warning(f"CacheEvent without worker_id: {event}")
            return

        workload = self.get_or_create_workload(event.worker_id)

        if event.event_type == "prefill_start":
            pending_tokens = event.effective_token_count
            workload.pending_tokens += pending_tokens
            logger.debug(f"[WORKLOAD] prefill_start event: {event.worker_id}, pending_tokens={workload.pending_tokens}")
        elif event.event_type == "prefill_end":
            pending_tokens = event.effective_token_count
            workload.pending_tokens = max(0, workload.pending_tokens - pending_tokens)
            self._update_throughput_from_event(workload, prompt_tokens=pending_tokens, completion_tokens=1)
            logger.debug(f"[WORKLOAD] prefill_end event: {event.worker_id}, pending_tokens={workload.pending_tokens}")
        elif event.event_type == "decode_start":
            pending_tokens = event.effective_token_count
            pending_osl = event.engine_specific.get("osl", 0)
            workload.pending_tokens += pending_tokens
            workload.pending_osl += pending_osl
            logger.debug(
                f"[WORKLOAD] decode_start event: {event.worker_id}, "
                f"pending_tokens={workload.pending_tokens}, "
                f"pending_osl={workload.pending_osl}"
            )
        elif event.event_type == "decode_end":
            # 移除 decode_start 时添加的 token 和 osl
            workload.pending_tokens = max(0, workload.pending_tokens - event.effective_token_count)
            workload.pending_osl = max(0, workload.pending_osl - event.engine_specific.get("osl", 0))
            # 从事件中获取 completion_tokens，默认为1
            completion_tokens = event.engine_specific.get("completion_tokens", 1)
            self._update_throughput_from_event(
                workload,
                prompt_tokens=event.effective_token_count,
                completion_tokens=completion_tokens,
            )

            logger.debug(
                f"[WORKLOAD] decode_end event: {event.worker_id}, "
                f"pending_tokens={workload.pending_tokens}, "
                f"pending_osl={workload.pending_osl}"
            )
        else:
            logger.debug(f"[WORKLOAD] unknown event type: {event.event_type}")

    def process_cache_events(self, events: list[CacheEvent]) -> None:
        """批量处理缓存事件"""
        for event in events:
            self.process_cache_event(event)

    def calculate_load(self, worker: WorkerInfo) -> float:
        """计算工作器的 current_load

        Args:
            worker: 工作器信息

        Returns:
            current_load 值（0-100）
        """
        if not self._enabled:
            return worker.current_load

        workload = self.get_or_create_workload(worker.worker_id)

        total_tokens = getattr(worker, "total_tokens", settings.worker_token_capacity)

        if total_tokens <= 0:
            return worker.current_load

        worker_type = getattr(worker, "worker_type", WorkerType.COMBINED)

        if worker_type == WorkerType.PREFILL:
            load = (workload.pending_tokens / total_tokens) * 100
        elif worker_type == WorkerType.DECODE:
            load = ((workload.pending_tokens + workload.pending_osl) / total_tokens) * 100
        else:
            prefill_load = (workload.pending_tokens / total_tokens) * 100
            decode_load = ((workload.pending_tokens + workload.pending_osl) / total_tokens) * 100
            load = (prefill_load + decode_load) / 2

        load = max(0, min(100, load))

        logger.info(
            f"[WORKLOAD] calculated load for {worker.worker_id}: {load:.2f}% "
            f"(type={worker_type}, pending_tokens={workload.pending_tokens}, "
            f"pending_osl={workload.pending_osl}, total_tokens={total_tokens})"
        )

        return load

    def get_workload(self, worker_id: str) -> WorkerWorkload | None:
        """获取工作器的负载信息"""
        workload = self.workloads.get(worker_id)
        if workload:
            self.update_ratios(workload)
        return workload

    def update_throughput(self, worker_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        """更新工作器的吞吐速率

        Args:
            worker_id: 工作器ID
            prompt_tokens: 响应中的prompt_tokens
            completion_tokens: 响应中的completion_tokens
        """
        if not self._enabled:
            return

        workload = self.get_or_create_workload(worker_id)
        timestamp = time.time()
        workload.throughput_records.append((timestamp, prompt_tokens, completion_tokens))
        self.update_ratios(workload)

        logger.debug(
            f"[WORKLOAD] Updated throughput for {worker_id}: "
            f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
            f"prompt_ratio={workload.prompt_ratio:.2f}, "
            f"completion_ratio={workload.completion_ratio:.2f}"
        )

    def _update_throughput_from_event(
        self, workload: WorkerWorkload, prompt_tokens: int, completion_tokens: int
    ) -> None:
        """通过事件更新吞吐速率

        Args:
            workload: 工作器负载信息
            prompt_tokens: prompt token数量
            completion_tokens: completion token数量
        """
        if not self._enabled:
            return

        timestamp = time.time()
        workload.throughput_records.append((timestamp, prompt_tokens, completion_tokens))
        self.update_ratios(workload)

        logger.debug(
            f"[WORKLOAD] Updated throughput from event: {workload.worker_id}, "
            f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
            f"prompt_ratio={workload.prompt_ratio:.2f}, "
            f"completion_ratio={workload.completion_ratio:.2f}"
        )

    @staticmethod
    def update_ratios(workload: WorkerWorkload) -> None:
        """更新吞吐速率计算

        保留5分钟窗口内的请求记录，计算平均吞吐速率。
        """
        current_time = time.time()
        window_seconds = 5 * 60  # 5分钟窗口

        # 过滤掉5分钟之前的记录
        workload.throughput_records = [
            (ts, pt, ct) for ts, pt, ct in workload.throughput_records if current_time - ts <= window_seconds
        ]

        if not workload.throughput_records:
            workload.prompt_ratio = 1000.0
            workload.completion_ratio = 100.0
            return

        # 计算总token数和总时间
        total_prompt_tokens = sum(pt for _, pt, _ in workload.throughput_records)
        total_completion_tokens = sum(ct for _, _, ct in workload.throughput_records)

        # 使用窗口时间作为分母计算平均速率
        workload.prompt_ratio = max(1.0, total_prompt_tokens / window_seconds)
        workload.completion_ratio = max(1.0, total_completion_tokens / window_seconds)

    def remove_worker(self, worker_id: str) -> None:
        """移除工作器的负载信息"""
        if worker_id in self.workloads:
            del self.workloads[worker_id]
            logger.debug(f"[WORKLOAD] worker removed: {worker_id}")

    def reset(self) -> None:
        """重置所有负载信息"""
        self.workloads.clear()
        logger.info("[WORKLOAD] All workloads reset")

    def get_stats(self) -> dict[str, tuple[int, int]]:
        """获取所有工作器的负载统计 (pending_tokens, pending_osl)"""
        stats = {}
        for worker_id, workload in self.workloads.items():
            stats[worker_id] = (
                workload.pending_tokens,
                workload.pending_osl,
            )
        return stats
