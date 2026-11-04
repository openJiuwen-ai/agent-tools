import math
import random
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from loguru import logger

from openjiuwentools.infer_router.config.config import settings
from openjiuwentools.infer_router.kv_cache.kv_cache import CacheLocation, KVCacheManager
from openjiuwentools.infer_router.monitoring.metrics import metrics
from openjiuwentools.infer_router.routing.load_balance import (
    RoundRobinAlgorithm,
    WorkloadWeightedAlgorithm,
)
from openjiuwentools.infer_router.routing.scorer import PluginScorerAlgorithm
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint, WorkerType
from openjiuwentools.infer_router.worker.worker_manager import WorkerManager

_PREFIX_AWARE_DEFAULTS = [
    {"type": "prefix-cache-scorer", "weight": 3.0},
    {"type": "kv-cache-utilization-scorer", "weight": 2.0},
    {"type": "queue-scorer", "weight": 2.0},
    {"type": "no-hit-lru-scorer", "weight": 2.0},
    {"type": "inflight-scorer", "weight": 0.0},
]

_PREFILL_DEFAULTS = [
    {"type": "prefix-cache-scorer", "weight": 3.0},
    {"type": "kv-cache-utilization-scorer", "weight": 2.0},
    {"type": "queue-scorer", "weight": 1.0},
    {"type": "inflight-scorer", "weight": 1.0},
    {"type": "no-hit-lru-scorer", "weight": 2.0},
]

_DECODE_DEFAULTS = [
    {"type": "kv-cache-utilization-scorer", "weight": 1.0},
    {"type": "queue-scorer", "weight": 1.0},
    {"type": "inflight-scorer", "weight": 5.0},
    {"type": "active-request-scorer", "weight": 1.0},
    {"type": "no-hit-lru-scorer", "weight": 2.0},
]

_PD_DECODE_AFFINITY_SPILL_GRACE_SECONDS = 0.75
_PD_CAPACITY_WAITER_LOG_THRESHOLD_SECONDS = 0.1
_PD_DECODE_CONFIRMATION_RETRY_BASE_SECONDS = 0.25
_PD_DECODE_CONFIRMATION_RETRY_MAX_SECONDS = 1.0
_PD_DECODE_CONFIRMATION_PROBE_INTERVAL_SECONDS = 1.0
_PD_DECODE_CONFIRMATION_PROBE_JITTER = 0.2
_PD_DECODE_SPILL_MIN_WAITING = 4
_PD_DECODE_SPILL_QUEUE_RATIO = 0.5
_PD_DECODE_SPILL_RELIEF_REQUESTS = 2
_PD_DECODE_SMALL_DELTA_BLOCKS = 8
_PD_DECODE_SMALL_DELTA_RETRY_TIMEOUT_MS = 150
_PD_HOTSPOT_TPOT_MIN_SAMPLES = 4
_PD_HOTSPOT_TPOT_WINDOW_SECONDS = 60.0
_PD_HOTSPOT_COOLDOWN_SECONDS = 5.0


def _admission_rejection_details(decisions: dict, fields: tuple[str, ...]) -> dict:
    """Select the diagnostic fields for workers rejected by admission."""
    rejected = {}
    for worker_id, (admitted, details) in decisions.items():
        if not admitted:
            rejected[worker_id] = {field: details.get(field) for field in fields}
    return rejected


def _capacity_query_timed_out(result: dict) -> bool:
    error = str(result.get("error", "")).lower()
    return str(result.get("error_type", "")).endswith("Timeout") or ("timeout" in error or "timed out" in error)


class InflightTracker:
    """Router-side inflight request counter per worker.

    Unlike worker-reported metrics (ZMQ), this updates instantly on
    route/complete, giving real-time admission control.
    """

    def __init__(self):
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def increment(self, worker_id: str) -> int:
        with self._lock:
            self._counts[worker_id] += 1
            return self._counts[worker_id]

    def decrement(self, worker_id: str) -> int:
        with self._lock:
            self._counts[worker_id] = max(0, self._counts[worker_id] - 1)
            return self._counts[worker_id]

    def get(self, worker_id: str) -> int:
        return self._counts.get(worker_id, 0)


class PdDecodeBlockTracker:
    """跟踪 Router 已准入但尚未完成的 D Worker 预计新增 KV block。"""

    def __init__(self):
        self._reservations: dict[str, tuple[str, int, int]] = {}
        self._original_blocks: dict[str, int] = {}
        self._worker_requests: dict[str, list[str]] = defaultdict(list)
        self._blocks: dict[str, int] = defaultdict(int)
        self._allocated_at: dict[str, float] = {}
        self._allocated_generation: dict[str, int] = {}
        self._pending_capacity_ack: dict[str, float] = {}
        self._pending_capacity_generation: dict[str, int] = {}
        self._pending_capacity_first_sample: dict[str, float] = {}
        self._lock = threading.Lock()

    def reserve(
        self,
        request_id: str,
        worker_id: str,
        block_count: int,
        steady_state_blocks: int | None = None,
        fallback_block_count: int | None = None,
    ) -> None:
        """登记请求的预计新增 block。

        参数:
            request_id: Router 请求 ID。
            worker_id: 目标 D Worker ID。
            block_count: 预计新增的 Prompt 与 Decode block 数。
            steady_state_blocks: Prompt attach 后仍未落地的 Decode 增长预算。
            fallback_block_count: PD失败后本地重算所需的完整预算；未提供时
                使用block_count保持兼容。

        返回:
            无返回值；同一请求重复登记时先替换旧记录。
        """
        with self._lock:
            self._release_locked(request_id)
            blocks = max(int(block_count), 0)
            steady_blocks = max(
                int(steady_state_blocks if steady_state_blocks is not None else blocks),
                0,
            )
            self._reservations[request_id] = (worker_id, blocks, steady_blocks)
            self._original_blocks[request_id] = max(
                int(fallback_block_count if fallback_block_count is not None else blocks),
                blocks,
            )
            self._worker_requests[worker_id].append(request_id)
            self._blocks[worker_id] += blocks

    def release(self, worker_id: str, request_id: str | None = None) -> int:
        """释放一个请求的 D block 预算。

        参数:
            worker_id: 请求使用的 D Worker ID。
            request_id: 精确请求 ID；旧调用方未提供时释放该 Worker 最早的记录。

        返回:
            释放后该 Worker 仍被 Router 预算占用的 block 数。
        """
        with self._lock:
            target_id = request_id
            if target_id is None or self._reservations.get(target_id, (None, 0, 0))[0] != worker_id:
                requests = self._worker_requests.get(worker_id, [])
                target_id = requests[0] if requests else None
            if target_id is not None:
                self._release_locked(target_id)
            return self._blocks.get(worker_id, 0)

    def get_blocks(self, worker_id: str) -> int:
        """返回 Worker 当前登记的预计新增 block 总数。"""
        with self._lock:
            return self._blocks.get(worker_id, 0)

    def reconcile(
        self,
        worker_id: str,
        request_id: str,
        capacity_generation: int | None = None,
    ) -> tuple[int, int, int, bool]:
        """记录请求已 attach，等待新容量样本确认物理 block 已被计数。

        参数:
            worker_id: 目标 D Worker ID。
            request_id: Router 请求 ID。

        返回:
            ``(previous_blocks, steady_state_blocks, worker_remaining_blocks,
            capacity_ack_pending)``；请求不存在时 previous_blocks 为 -1。
        """
        with self._lock:
            reservation = self._reservations.get(request_id)
            if reservation is None or reservation[0] != worker_id:
                return -1, 0, self._blocks.get(worker_id, 0), False
            _, previous_blocks, steady_state_blocks = reservation
            if previous_blocks == steady_state_blocks:
                return (
                    previous_blocks,
                    steady_state_blocks,
                    self._blocks.get(worker_id, 0),
                    False,
                )
            pending = request_id not in self._pending_capacity_ack
            if pending:
                self._allocated_at.pop(request_id, None)
                self._pending_capacity_ack[request_id] = time.monotonic()
                generations = []
                for selected_value in (self._allocated_generation.get(request_id), capacity_generation):
                    if selected_value is not None and int(selected_value) >= 0:
                        generations.append(selected_value)
                if generations:
                    self._pending_capacity_generation[request_id] = max(int(value) for value in generations)
            return (
                previous_blocks,
                steady_state_blocks,
                self._blocks.get(worker_id, 0),
                pending,
            )

    def mark_allocated(
        self,
        worker_id: str,
        request_id: str,
        capacity_generation: int | None = None,
    ) -> tuple[int, int, int, bool]:
        """记录D Scheduler已经为请求分配目标block。

        参数:
            worker_id: 完成目标block分配的D Worker ID。
            request_id: Router请求ID。

        返回:
            ``(previous_blocks, steady_state_blocks, worker_remaining_blocks,
            capacity_ack_pending)``。请求不存在时previous_blocks为-1。

        说明:
            Scheduler allocation可能只完成KVBM目标预留，尚未反映到
            Router读取的BlockPool free_blocks。因此这里只记录事件，
            必须等attach完成后才能启动容量样本确认。
        """
        with self._lock:
            reservation = self._reservations.get(request_id)
            if reservation is None or reservation[0] != worker_id:
                return -1, 0, self._blocks.get(worker_id, 0), False
            _, previous_blocks, steady_state_blocks = reservation
            newly_recorded = request_id not in self._allocated_at
            if newly_recorded:
                self._allocated_at[request_id] = time.monotonic()
            if capacity_generation is not None and int(capacity_generation) >= 0:
                self._allocated_generation[request_id] = max(
                    int(capacity_generation),
                    self._allocated_generation.get(request_id, -1),
                )
            return (
                previous_blocks,
                steady_state_blocks,
                self._blocks.get(worker_id, 0),
                newly_recorded,
            )

    def acknowledge_capacity(
        self,
        worker_id: str,
        capacity_updated_at: float,
        capacity_generation: int = -1,
    ) -> tuple[int, int]:
        """用attach之后连续两个BlockPool样本确认已落地的物理block。

        参数:
            worker_id: 产生容量样本的 D Worker ID。
            capacity_updated_at: Router 收到该容量样本时的单调时钟时间。

        返回:
            ``(acknowledged_requests, worker_remaining_blocks)``，分别表示本次
            完成确认的请求数和收缩后的 Worker 在途预算。

        说明:
            第一个事件后样本可能是事件前发起、事件后才返回的查询，
            其 free_blocks 尚未包含新分配。等待第二个不同样本可覆盖该竞态。
        """
        with self._lock:
            acknowledged = 0
            for request_id in tuple(self._worker_requests.get(worker_id, ())):
                attached_at = self._pending_capacity_ack.get(request_id)
                if attached_at is None or capacity_updated_at <= attached_at:
                    continue
                required_generation = self._pending_capacity_generation.get(request_id)
                generation_confirmed = bool(
                    required_generation is not None and capacity_generation >= required_generation
                )
                if required_generation is not None and not generation_confirmed:
                    continue
                first_sample_at = self._pending_capacity_first_sample.get(request_id)
                if required_generation is None and first_sample_at is None:
                    self._pending_capacity_first_sample[request_id] = capacity_updated_at
                    continue
                if required_generation is None and capacity_updated_at <= first_sample_at:
                    continue
                reservation = self._reservations.get(request_id)
                if reservation is None or reservation[0] != worker_id:
                    self._pending_capacity_ack.pop(request_id, None)
                    self._pending_capacity_generation.pop(request_id, None)
                    self._pending_capacity_first_sample.pop(request_id, None)
                    continue
                _, previous_blocks, steady_state_blocks = reservation
                self._reservations[request_id] = (
                    worker_id,
                    steady_state_blocks,
                    steady_state_blocks,
                )
                self._blocks[worker_id] = max(
                    0,
                    self._blocks.get(worker_id, 0) - previous_blocks + steady_state_blocks,
                )
                self._pending_capacity_ack.pop(request_id, None)
                self._pending_capacity_generation.pop(request_id, None)
                self._pending_capacity_first_sample.pop(request_id, None)
                acknowledged += 1
            return acknowledged, self._blocks.get(worker_id, 0)

    def restore_after_fallback(
        self,
        worker_id: str,
        request_id: str,
    ) -> tuple[int, int, int, bool]:
        """在Scheduler目标block回滚后恢复请求的完整准入预算。

        参数:
            worker_id: 已降级为本地重算的D Worker ID。
            request_id: Router请求ID。

        返回:
            ``(previous_blocks, restored_blocks, worker_remaining_blocks,
            restored)``。请求不存在或预算已经完整时，``restored``为False。

        说明:
            Scheduler分配目标block后，Router会在容量样本确认后把Prompt临时
            预算收缩为Decode增长预算。如果PD事务随后失败，目标block会被
            Scheduler释放；此时必须恢复原始预算，避免free_blocks尚未反映
            本地重算占用期间继续过量准入。
        """
        with self._lock:
            reservation = self._reservations.get(request_id)
            if reservation is None or reservation[0] != worker_id:
                return -1, 0, self._blocks.get(worker_id, 0), False
            _, previous_blocks, steady_state_blocks = reservation
            original_blocks = self._original_blocks.get(request_id, previous_blocks)
            self._pending_capacity_ack.pop(request_id, None)
            self._pending_capacity_generation.pop(request_id, None)
            self._pending_capacity_first_sample.pop(request_id, None)
            self._allocated_at.pop(request_id, None)
            self._allocated_generation.pop(request_id, None)
            if previous_blocks >= original_blocks:
                return (
                    previous_blocks,
                    original_blocks,
                    self._blocks.get(worker_id, 0),
                    False,
                )
            self._reservations[request_id] = (
                worker_id,
                original_blocks,
                steady_state_blocks,
            )
            self._blocks[worker_id] += original_blocks - previous_blocks
            return (
                previous_blocks,
                original_blocks,
                self._blocks.get(worker_id, 0),
                True,
            )

    def _release_locked(self, request_id: str) -> None:
        self._allocated_at.pop(request_id, None)
        self._allocated_generation.pop(request_id, None)
        self._pending_capacity_ack.pop(request_id, None)
        self._pending_capacity_generation.pop(request_id, None)
        self._pending_capacity_first_sample.pop(request_id, None)
        self._original_blocks.pop(request_id, None)
        reservation = self._reservations.pop(request_id, None)
        if reservation is None:
            return
        worker_id, blocks, _ = reservation
        requests = self._worker_requests.get(worker_id, [])
        try:
            requests.remove(request_id)
        except ValueError:
            pass
        if not requests:
            self._worker_requests.pop(worker_id, None)
        self._blocks[worker_id] = max(0, self._blocks.get(worker_id, 0) - blocks)


class PdPrefillBudgetTracker:
    """跟踪已派发但尚未完成 Prefill 的 token 与 KV block 预算。"""

    def __init__(self):
        self._reservations: dict[str, tuple[str, int, int]] = {}
        self._worker_tokens: dict[str, int] = defaultdict(int)
        self._worker_blocks: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def reserve(
        self,
        request_id: str,
        worker_id: str,
        token_count: int,
        block_count: int,
    ) -> None:
        """登记一个 P 请求尚未完成的计算和内存预算。

        参数:
            request_id: Router 请求 ID。
            worker_id: 目标 P Worker ID。
            token_count: 预计需要实际 Prefill 的 token 数。
            block_count: 预计需要新增的 KV block 数。

        返回:
            无返回值；重复请求会先释放旧预算再登记新预算。
        """
        with self._lock:
            self._release_locked(request_id)
            tokens = max(int(token_count), 0)
            blocks = max(int(block_count), 0)
            self._reservations[request_id] = (worker_id, tokens, blocks)
            self._worker_tokens[worker_id] += tokens
            self._worker_blocks[worker_id] += blocks

    def release(
        self,
        request_id: str | None,
        worker_id: str | None = None,
    ) -> tuple[int, int]:
        """释放一个 P 请求预算并返回该 Worker 的剩余预算。

        参数:
            request_id: Router 请求 ID；旧调用方未提供时释放该 Worker 最早的记录。
            worker_id: 可选的 P Worker ID，用于防止误释放其他 Worker 的记录。

        返回:
            ``(remaining_tokens, remaining_blocks)``。
        """
        with self._lock:
            reservation = self._reservations.get(request_id) if request_id else None
            if reservation is None and worker_id:
                request_id = next(
                    (
                        candidate_id
                        for candidate_id, candidate in self._reservations.items()
                        if candidate[0] == worker_id
                    ),
                    None,
                )
                reservation = self._reservations.get(request_id) if request_id else None
            if reservation is not None and (worker_id is None or reservation[0] == worker_id):
                actual_worker_id = reservation[0]
                self._release_locked(request_id)
            else:
                actual_worker_id = worker_id or ""
            return (
                self._worker_tokens.get(actual_worker_id, 0),
                self._worker_blocks.get(actual_worker_id, 0),
            )

    def get(self, worker_id: str) -> tuple[int, int]:
        """返回 Worker 当前的 ``(token预算, block预算)``。"""
        with self._lock:
            return (
                self._worker_tokens.get(worker_id, 0),
                self._worker_blocks.get(worker_id, 0),
            )

    def _release_locked(self, request_id: str) -> None:
        reservation = self._reservations.pop(request_id, None)
        if reservation is None:
            return
        worker_id, tokens, blocks = reservation
        self._worker_tokens[worker_id] = max(0, self._worker_tokens.get(worker_id, 0) - tokens)
        self._worker_blocks[worker_id] = max(0, self._worker_blocks.get(worker_id, 0) - blocks)


@dataclass(frozen=True)
class PendingPrefixMatch:
    """Router 内部正在完成的 HIXL 前缀匹配。"""

    request_id: str
    model: str
    worker_id: str
    matched_tokens: int
    overlap_ratio: float
    state: str


@dataclass
class _DecodeCapacityRejection:
    """与已观测容量状态绑定的请求级 D 拒绝结果。

    retry_at: 瞬时确认失败（HTTP 超时等）的指数退避重试点。
    probe_at: generation缺失或Router候选复用尚未被Worker确认时的低频
        权威复核点。容量代际有效且复用稳定的拒绝仅等待代际/阈值变化。
    """

    capacity_generation: int
    reserved_blocks: int
    inflight: int
    free_blocks: int
    available_blocks: int
    estimated_blocks: int
    dynamic_margin_blocks: int
    failure_count: int
    retry_at: float | None
    diagnostics: dict
    probe_at: float | None = None


@dataclass
class _CapacityWaiter:
    """A FIFO waiter that can yield its active routing slot temporarily."""

    sequence: int
    enqueued_at: float
    parked_worker_generations: dict[str, int] | None = None
    parked_at: float | None = None
    retry_at: float | None = None


@dataclass
class _PendingPrefixEntry:
    request_id: str
    model: str
    worker_id: str
    block_size: int
    fingerprints: tuple[int, ...]
    index_prefix_lengths: tuple[int, ...]
    token_count: int
    created_at: float
    state: str = "pending"


class PendingPrefixAffinityTracker:
    """合并同一公共前缀的 HIXL 首次导入。

    HIXL 的 P->D import 在 Router KV 索引可见前存在一个窗口。同一前缀在
    这个窗口内到达时，如果立即各自派发，会重复申请完整 Prompt blocks。
    这里仅记录短生命周期的请求级状态；正式 cache location 仍由
    ``KVCacheManager.record_pd_import`` 维护。
    """

    def __init__(self, ttl_seconds: float):
        self._ttl_seconds = max(float(ttl_seconds), 1.0)
        self._entries: dict[str, _PendingPrefixEntry] = {}
        self._prefix_index: dict[
            tuple[str, int, int, tuple[int, ...]],
            set[str],
        ] = defaultdict(set)
        self._model_block_sizes: dict[str, set[int]] = defaultdict(set)
        self._block_size_counts: dict[tuple[str, int], int] = defaultdict(int)
        self._condition = threading.Condition()

    @staticmethod
    def _fingerprints(token_ids: list[int], block_size: int) -> tuple[int, ...]:
        size = max(int(block_size), 1)
        return tuple(
            hash(tuple(token_ids[offset:offset + size])) for offset in range(0, len(token_ids) - size + 1, size)
        )

    @staticmethod
    def _index_prefix_lengths(block_count: int) -> tuple[int, ...]:
        """返回用于粗筛的1、2、4...block前缀长度。"""
        lengths = []
        length = 1
        while length <= max(int(block_count), 0):
            lengths.append(length)
            length *= 2
        return tuple(lengths)

    @staticmethod
    def _common_prefix_blocks(
        left: tuple[int, ...],
        right: tuple[int, ...],
    ) -> int:
        common_blocks = 0
        for left_fingerprint, right_fingerprint in zip(left, right):
            if left_fingerprint != right_fingerprint:
                break
            common_blocks += 1
        return common_blocks

    def _remove_entry_locked(self, request_id: str) -> _PendingPrefixEntry | None:
        entry = self._entries.pop(request_id, None)
        if entry is None:
            return None
        for prefix_length in entry.index_prefix_lengths:
            key = (
                entry.model,
                entry.block_size,
                prefix_length,
                entry.fingerprints[:prefix_length],
            )
            indexed_requests = self._prefix_index.get(key)
            if indexed_requests is None:
                continue
            indexed_requests.discard(request_id)
            if not indexed_requests:
                self._prefix_index.pop(key, None)
        size_key = (entry.model, entry.block_size)
        remaining = self._block_size_counts.get(size_key, 0) - 1
        if remaining > 0:
            self._block_size_counts[size_key] = remaining
        else:
            self._block_size_counts.pop(size_key, None)
            model_sizes = self._model_block_sizes.get(entry.model)
            if model_sizes is not None:
                model_sizes.discard(entry.block_size)
                if not model_sizes:
                    self._model_block_sizes.pop(entry.model, None)
        return entry

    def _add_entry_locked(self, entry: _PendingPrefixEntry) -> None:
        self._entries[entry.request_id] = entry
        for prefix_length in entry.index_prefix_lengths:
            key = (
                entry.model,
                entry.block_size,
                prefix_length,
                entry.fingerprints[:prefix_length],
            )
            self._prefix_index[key].add(entry.request_id)
        self._model_block_sizes[entry.model].add(entry.block_size)
        self._block_size_counts[(entry.model, entry.block_size)] += 1

    def _cleanup_locked(self, now: float) -> None:
        expired = [
            request_id for request_id, entry in self._entries.items() if now - entry.created_at >= self._ttl_seconds
        ]
        for request_id in expired:
            self._remove_entry_locked(request_id)

    def _match_locked(
        self,
        model: str,
        token_ids: list[int],
        request_id: str,
        min_tokens: int,
        min_ratio: float,
        incoming_by_block_size: dict[int, tuple[int, ...]] | None = None,
        excluded_worker_ids: set[str] | None = None,
    ) -> PendingPrefixMatch | None:
        now = time.monotonic()
        self._cleanup_locked(now)
        incoming_cache = incoming_by_block_size if incoming_by_block_size is not None else {}
        candidate_ids: set[str] = set()
        for block_size in self._model_block_sizes.get(model, ()):
            incoming = incoming_cache.get(block_size)
            if incoming is None:
                incoming = self._fingerprints(token_ids, block_size)
                incoming_cache[block_size] = incoming
            required_blocks = max(math.ceil(max(int(min_tokens), 1) / block_size), 1)
            prefix_lengths = self._index_prefix_lengths(min(required_blocks, len(incoming)))
            if not prefix_lengths or len(incoming) < required_blocks:
                continue
            prefix_length = prefix_lengths[-1]
            key = (
                model,
                block_size,
                prefix_length,
                incoming[:prefix_length],
            )
            candidate_ids.update(self._prefix_index.get(key, ()))

        best: PendingPrefixMatch | None = None
        candidates = sorted(
            (self._entries[candidate_id] for candidate_id in candidate_ids if candidate_id in self._entries),
            key=lambda entry: entry.created_at,
        )
        for entry in candidates:
            if entry.request_id == request_id:
                continue
            if excluded_worker_ids and entry.worker_id in excluded_worker_ids:
                continue
            incoming = incoming_cache[entry.block_size]
            common_blocks = self._common_prefix_blocks(entry.fingerprints, incoming)
            matched_tokens = min(
                common_blocks * entry.block_size,
                entry.token_count,
                len(token_ids),
            )
            denominator = min(entry.token_count, len(token_ids))
            overlap_ratio = matched_tokens / denominator if denominator else 0.0
            if matched_tokens < min_tokens or overlap_ratio < min_ratio:
                continue
            candidate = PendingPrefixMatch(
                request_id=entry.request_id,
                model=entry.model,
                worker_id=entry.worker_id,
                matched_tokens=matched_tokens,
                overlap_ratio=overlap_ratio,
                state=entry.state,
            )
            if best is None:
                best = candidate
            elif candidate.state == "ready" and best.state != "ready":
                best = candidate
            elif candidate.state == best.state and candidate.matched_tokens > best.matched_tokens:
                best = candidate
        return best

    def register(
        self,
        *,
        request_id: str,
        model: str,
        worker_id: str,
        token_ids: list[int],
        block_size: int,
        min_tokens: int,
        min_ratio: float,
    ) -> bool:
        """登记一个尚未完成的 HIXL prefix import。"""
        registered, _ = self.claim(
            request_id=request_id,
            model=model,
            worker_id=worker_id,
            token_ids=token_ids,
            block_size=block_size,
            min_tokens=min_tokens,
            min_ratio=min_ratio,
        )
        return registered

    def claim(
        self,
        *,
        request_id: str,
        model: str,
        worker_id: str,
        token_ids: list[int],
        block_size: int,
        min_tokens: int,
        min_ratio: float,
        excluded_worker_ids: set[str] | None = None,
    ) -> tuple[bool, PendingPrefixMatch | None]:
        """原子登记 leader，已有同前缀请求时返回该请求。"""
        if len(token_ids) < min_tokens:
            return False, None
        fingerprints = self._fingerprints(token_ids, block_size)
        if not fingerprints:
            return False, None
        with self._condition:
            self._cleanup_locked(time.monotonic())
            existing = self._match_locked(
                model,
                token_ids,
                request_id,
                min_tokens,
                min_ratio,
                incoming_by_block_size={max(int(block_size), 1): fingerprints},
                excluded_worker_ids=excluded_worker_ids,
            )
            if existing is not None:
                return False, existing
            index_prefix_lengths = self._index_prefix_lengths(len(fingerprints))
            self._add_entry_locked(
                _PendingPrefixEntry(
                    request_id=request_id,
                    model=model,
                    worker_id=worker_id,
                    block_size=max(int(block_size), 1),
                    fingerprints=fingerprints,
                    index_prefix_lengths=index_prefix_lengths,
                    token_count=len(token_ids),
                    created_at=time.monotonic(),
                )
            )
            self._condition.notify_all()
            return True, None

    def find(
        self,
        *,
        request_id: str,
        model: str,
        token_ids: list[int],
        min_tokens: int,
        min_ratio: float,
        excluded_worker_ids: set[str] | None = None,
    ) -> PendingPrefixMatch | None:
        """返回当前同前缀 leader，不创建新记录。"""
        if len(token_ids) < min_tokens:
            return None
        with self._condition:
            return self._match_locked(
                model,
                token_ids,
                request_id,
                min_tokens,
                min_ratio,
                incoming_by_block_size={},
                excluded_worker_ids=excluded_worker_ids,
            )

    def wait_for(
        self,
        *,
        request_id: str,
        model: str,
        token_ids: list[int],
        min_tokens: int,
        min_ratio: float,
        timeout: float,
    ) -> PendingPrefixMatch | None:
        """等待同前缀 leader 完成，失败或超时则返回 None。"""
        deadline = time.monotonic() + max(float(timeout), 0.0)
        incoming_by_block_size: dict[int, tuple[int, ...]] = {}
        with self._condition:
            while True:
                match = self._match_locked(
                    model,
                    token_ids,
                    request_id,
                    min_tokens,
                    min_ratio,
                    incoming_by_block_size=incoming_by_block_size,
                )
                if match is None:
                    return None
                if match.state == "ready":
                    return match
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def mark_ready(self, request_id: str, worker_id: str, published_blocks: int) -> None:
        """将 attach 成功的 leader 标记为可复用；发布失败则清除。"""
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.worker_id != worker_id:
                return
            if int(published_blocks) > 0:
                entry.state = "ready"
            else:
                self._remove_entry_locked(request_id)
            self._condition.notify_all()

    def remove(self, request_id: str, worker_id: str | None = None) -> None:
        """请求回滚或超时后清理 pending 状态。

        已 ready 的 entry 对应 Router 已发布的正式 NPU location，短时间保留
        可以覆盖 leader 完成与 follower 唤醒之间的竞态；它会由 TTL 清理。
        """
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or (worker_id is not None and entry.worker_id != worker_id):
                return
            if entry.state == "ready":
                return
            self._remove_entry_locked(request_id)
            self._condition.notify_all()

    def discard(self, request_id: str, worker_id: str | None = None) -> None:
        """无条件清除失效记录，包括已经标记 ready 的记录。"""
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or (worker_id is not None and entry.worker_id != worker_id):
                return
            self._remove_entry_locked(request_id)
            self._condition.notify_all()


class Router:
    """路由决策层，基于路由提示和KV缓存状态选择最佳推理服务节点"""

    def __init__(self, kvcache_manager: KVCacheManager, worker_manager: WorkerManager):
        self.kvcache_manager = kvcache_manager
        self.worker_manager = worker_manager
        self._admission_mode = settings.admission_mode
        self._concurrency_limit = int(settings.max_concurrency_per_worker * (1 + settings.concurrency_headroom))
        self._saturation_ratio = settings.admission_saturation_ratio
        self._pd_prefill_admission_mode = str(settings.router_pd_prefill_admission_mode).lower()
        self._pd_prefill_max_inflight = max(int(settings.router_pd_prefill_max_inflight or 0), 0)
        self._pd_prefill_auto_fallback_max_inflight = max(int(settings.kvbm_router_pd_prefill_max_inflight), 1)
        self._pd_prefill_block_admission_enabled = bool(settings.router_pd_prefill_block_admission_enabled)
        self._pd_prefill_block_headroom_ratio = min(
            max(float(settings.router_pd_prefill_block_headroom_ratio), 0.0), 0.9
        )
        self._pd_decode_admission_mode = str(settings.router_pd_decode_admission_mode).lower()
        configured_ownership = str(settings.kvbm_pd_block_ownership).strip().lower()
        self._pd_block_ownership = (
            configured_ownership if configured_ownership in {"scheduler", "legacy"} else "scheduler"
        )
        self._pd_decode_max_inflight = max(int(settings.router_pd_decode_max_inflight), 0)
        self._pd_decode_block_admission_enabled = bool(settings.router_pd_decode_block_admission_enabled)
        self._pd_decode_block_headroom_ratio = min(
            max(float(settings.router_pd_decode_block_headroom_ratio), 0.0),
            0.9,
        )
        self._pd_decode_initial_window_blocks = max(int(settings.router_pd_decode_initial_window_blocks), 1)
        self._pd_block_pool_metrics_ttl_seconds = max(float(settings.router_pd_block_pool_metrics_ttl_seconds), 1.0)
        self._pd_decode_capacity_confirmation_enabled = bool(
            settings.kvbm_enabled
            and settings.kvbm_router_pd_enabled
            and settings.router_pd_decode_capacity_confirmation_enabled
        )
        self._pd_decode_capacity_confirmation_timeout_ms = max(
            int(settings.router_pd_decode_capacity_confirmation_timeout_ms), 1
        )
        self._pd_decode_stale_capacity_confirmation_timeout_ms = max(
            int(settings.router_pd_decode_stale_capacity_confirmation_timeout_ms),
            self._pd_decode_capacity_confirmation_timeout_ms,
        )
        self.inflight = InflightTracker()
        self.pd_prefill_budget = PdPrefillBudgetTracker()
        self.pd_decode_blocks = PdDecodeBlockTracker()
        self._routing_lock = threading.Lock()
        self._release_cond = threading.Condition()
        self._capacity_generation = 0
        self._capacity_state_signature: tuple | None = None
        self._capacity_worker_state_signatures: dict[str, tuple] = {}
        self._capacity_worker_generations: dict[str, int] = defaultdict(int)
        self._capacity_waiter_sequence = 0
        self._capacity_waiters: dict[str, dict[str, _CapacityWaiter]] = {}
        self._capacity_waiter_models: dict[str, str] = {}
        self._pd_admission_log_lock = threading.Lock()
        self._pd_admission_last_warning_at = 0.0
        self._pd_admission_suppressed_warnings = 0
        self._pd_decode_admission_tickets: dict[str, tuple[str, str]] = {}
        self._pd_decode_confirmation_rejections: dict[
            str,
            dict[str, _DecodeCapacityRejection],
        ] = defaultdict(dict)
        self._pd_decode_affinity_rejected_at: dict[str, dict[str, float]] = defaultdict(dict)
        self._pd_decode_affinity_spill_ready: dict[str, set[str]] = defaultdict(set)
        self._pd_decode_confirmation_rejection_lock = threading.Lock()
        self._pd_hotspot_spill_enabled = bool(
            settings.kvbm_enabled
            and settings.kvbm_router_pd_enabled
            and settings.router_pd_decode_hotspot_spill_enabled
        )
        configured_hot_min = settings.router_pd_decode_hotspot_spill_hot_min_running
        configured_cold_max = settings.router_pd_decode_hotspot_spill_cold_max_running
        configured_gap = settings.router_pd_decode_hotspot_spill_running_gap
        if configured_hot_min is not None or configured_cold_max is not None:
            self._pd_hotspot_hot_min_running = (
                max(int(configured_hot_min), 1) if configured_hot_min is not None else None
            )
            self._pd_hotspot_cold_max_running = (
                max(int(configured_cold_max), 0) if configured_cold_max is not None else None
            )
            self._pd_hotspot_running_gap = None
        elif configured_gap is not None:
            self._pd_hotspot_hot_min_running = None
            self._pd_hotspot_cold_max_running = None
            self._pd_hotspot_running_gap = max(int(configured_gap), 1)
        else:
            self._pd_hotspot_hot_min_running = 16
            self._pd_hotspot_cold_max_running = 8
            self._pd_hotspot_running_gap = None
        configured_tpot_ratio = float(settings.router_pd_decode_hotspot_spill_tpot_ratio)
        self._pd_hotspot_tpot_ratio = max(configured_tpot_ratio, 1.0) if math.isfinite(configured_tpot_ratio) else 1.25
        self._pd_decode_recent_tpot: dict[str, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=32))
        self._pd_hotspot_last_spill_at: dict[str, float] = {}
        self._pd_pending_prefix_affinity = PendingPrefixAffinityTracker(
            ttl_seconds=max(float(settings.router_queue_timeout), 5.0)
        )
        self._pd_pending_prefix_min_tokens = max(int(settings.kvbm_router_pd_decode_direct_min_npu_tokens), 1)
        self._pd_pending_prefix_min_ratio = min(
            max(float(settings.kvbm_router_pd_decode_direct_min_ratio), 0.0),
            1.0,
        )

        self._algorithm_factories = {
            "round_robin": lambda: RoundRobinAlgorithm(),
            "weighted": lambda: WorkloadWeightedAlgorithm(kvcache_manager, worker_manager),
            "prefix_aware": lambda: PluginScorerAlgorithm(
                kvcache_manager,
                worker_manager,
                plugin_configs=settings.scoring_plugins or _PREFIX_AWARE_DEFAULTS,
                prefill_plugin_configs=settings.prefill_scoring_plugins or _PREFILL_DEFAULTS,
                decode_plugin_configs=settings.decode_scoring_plugins or _DECODE_DEFAULTS,
            ),
            "latency_predict": lambda: PluginScorerAlgorithm(
                kvcache_manager,
                worker_manager,
                plugin_configs=settings.scoring_plugins
                or [
                    {"type": "prefix-cache-scorer", "weight": 3.0},
                    {"type": "kv-cache-utilization-scorer", "weight": 2.0},
                    {"type": "queue-scorer", "weight": 2.0},
                    {
                        "type": "latency-predict-scorer",
                        "weight": 5.0,
                        "parameters": {
                            "prediction_server_url": settings.latency_prediction_server_url,
                            "training_server_url": settings.latency_training_server_url,
                            "predict_timeout": settings.latency_predict_timeout,
                            "training_report_enabled": settings.latency_training_report_enabled,
                        },
                    },
                ],
                prefill_plugin_configs=settings.prefill_scoring_plugins or _PREFILL_DEFAULTS,
                decode_plugin_configs=settings.decode_scoring_plugins or _DECODE_DEFAULTS,
            ),
        }
        self.algorithms = {}
        self._phase_algorithms = {}
        self._prefill_algorithm_override = (
            str(settings.prefill_load_balancing_algorithm).strip()
            if settings.prefill_load_balancing_algorithm
            else None
        )
        self._decode_algorithm_override = (
            str(settings.decode_load_balancing_algorithm).strip() if settings.decode_load_balancing_algorithm else None
        )
        self.current_algorithm = self._get_or_create(settings.load_balancing_algorithm)
        self.prefill_algorithm = self._get_or_create_phase(
            "prefill",
            self._prefill_algorithm_override or settings.load_balancing_algorithm,
        )
        self.decode_algorithm = self._get_or_create_phase(
            "decode",
            self._decode_algorithm_override or settings.load_balancing_algorithm,
        )

    def _get_or_create(self, algorithm_name: str):
        """按需创建算法实例，避免启动时全部实例化。"""
        if algorithm_name in self.algorithms:
            return self.algorithms[algorithm_name]
        factory = self._algorithm_factories.get(algorithm_name)
        if factory is None:
            logger.warning(f"Unknown algorithm {algorithm_name}, using round_robin instead")
            return self._get_or_create("round_robin")
        algo = factory()
        if hasattr(algo, "set_inflight_tracker"):
            algo.set_inflight_tracker(self.inflight)
        self.algorithms[algorithm_name] = algo
        return algo

    def _get_or_create_phase(self, phase: str, algorithm_name: str):
        """Create an algorithm instance whose mutable state belongs to one phase."""
        if not (self._prefill_algorithm_override or self._decode_algorithm_override):
            return self.current_algorithm
        key = (phase, algorithm_name)
        if key in self._phase_algorithms:
            return self._phase_algorithms[key]
        factory = self._algorithm_factories.get(algorithm_name)
        if factory is None:
            logger.warning(
                "Unknown {} algorithm {}, using round_robin instead",
                phase,
                algorithm_name,
            )
            algorithm_name = "round_robin"
            key = (phase, algorithm_name)
            if key in self._phase_algorithms:
                return self._phase_algorithms[key]
            factory = self._algorithm_factories[algorithm_name]
        algorithm = factory()
        if hasattr(algorithm, "set_inflight_tracker"):
            algorithm.set_inflight_tracker(self.inflight)
        self._phase_algorithms[key] = algorithm
        return algorithm

    @property
    def phase_algorithms_enabled(self) -> bool:
        return bool(self._prefill_algorithm_override or self._decode_algorithm_override)

    def get_algorithm_for_phase(self, phase: str):
        if phase == "prefill":
            return self.prefill_algorithm
        if phase == "decode":
            return self.decode_algorithm
        return self.current_algorithm

    def _pd_algorithm_label(self) -> str:
        if not self.phase_algorithms_enabled:
            return type(self.current_algorithm).__name__
        return f"prefill={type(self.prefill_algorithm).__name__},decode={type(self.decode_algorithm).__name__}"

    @staticmethod
    def _select_phase_worker(algorithm, phase, workers, route_hint):
        selector = getattr(algorithm, "select_worker_for_phase", None)
        if selector is not None:
            return selector(workers, route_hint, phase=phase)
        return algorithm.select_worker(workers, route_hint)

    def _select_pd_worker_pair(
        self,
        prefill_workers,
        decode_workers,
        route_hint,
    ):
        """Select P and D independently only when phase algorithms are configured."""
        if not self.phase_algorithms_enabled:
            return self.current_algorithm.select_worker_pair(
                prefill_workers=prefill_workers,
                decode_workers=decode_workers,
                route_hint=route_hint,
            )
        prefill_worker = self._select_phase_worker(
            self.prefill_algorithm,
            "prefill",
            prefill_workers,
            route_hint,
        )
        decode_worker = self._select_phase_worker(
            self.decode_algorithm,
            "decode",
            decode_workers,
            route_hint,
        )
        logger.info(
            "[PhaseAlgorithm] request={} prefill_algorithm={} decode_algorithm={} selected={}+{}",
            route_hint.request_id,
            type(self.prefill_algorithm).__name__,
            type(self.decode_algorithm).__name__,
            prefill_worker.worker_id,
            decode_worker.worker_id,
        )
        return prefill_worker, decode_worker

    def set_algorithm(self, algorithm_name: str):
        """设置负载均衡算法"""
        self.current_algorithm = self._get_or_create(algorithm_name)
        if self._prefill_algorithm_override is None:
            self.prefill_algorithm = self.current_algorithm
        if self._decode_algorithm_override is None:
            self.decode_algorithm = self.current_algorithm
        logger.info(f"Load balancing algorithm changed to {algorithm_name}")

    def release_worker(self, worker_id: str, request_id: str | None = None) -> None:
        """释放 Worker 请求计数及可选的 D block 预算。

        参数:
            worker_id: 完成请求所占用的 Worker ID。
            request_id: Router 请求 ID，用于精确释放 D block 预算；旧调用方可省略。

        返回:
            无返回值，并唤醒等待容量的 Router 请求。
        """
        remaining = self.inflight.decrement(worker_id)
        remaining_prefill_tokens, remaining_prefill_blocks = self.pd_prefill_budget.release(request_id, worker_id)
        tracked_blocks = self.pd_decode_blocks.get_blocks(worker_id)
        remaining_blocks = self.pd_decode_blocks.release(worker_id, request_id)
        logger.debug("[Inflight] {} released, remaining={}", worker_id, remaining)
        if tracked_blocks > 0:
            logger.debug(
                "[PDDecodeAdmission] {} block budget released, remaining_blocks={}",
                worker_id,
                remaining_blocks,
            )
        if remaining_prefill_tokens > 0 or remaining_prefill_blocks > 0:
            logger.debug(
                "[PDPrefillAdmission] {} budget released, remaining_tokens={}, remaining_blocks={}",
                worker_id,
                remaining_prefill_tokens,
                remaining_prefill_blocks,
            )
        if request_id is not None:
            self._pd_pending_prefix_affinity.remove(request_id, worker_id)
            self._clear_pd_decode_confirmation_rejections(request_id)
        with self._release_cond:
            self._publish_capacity_change_locked((worker_id,))

    def release_pd_prefill(self, worker_id: str, request_id: str) -> None:
        """在 P export/响应完成时提前释放 P 请求预算和 inflight。

        参数:
            worker_id: 已完成 Prefill 的 P Worker ID。
            request_id: Router 请求 ID。

        返回:
            无返回值；释放后唤醒等待动态容量的请求。
        """
        self.inflight.decrement(worker_id)
        remaining_tokens, remaining_blocks = self.pd_prefill_budget.release(request_id, worker_id)
        logger.debug(
            "[PDPrefillAdmission] {} phase completed, remaining_tokens={}, remaining_blocks={}",
            worker_id,
            remaining_tokens,
            remaining_blocks,
        )
        with self._release_cond:
            self._publish_capacity_change_locked((worker_id,))

    def reconcile_pd_decode_budget(
        self,
        worker_id: str,
        request_id: str,
        capacity_generation: int | None = None,
    ) -> None:
        """D attach后登记预算收缩意图，等待新BlockPool样本确认。

        参数:
            worker_id: 已接收请求 KV 的 D Worker ID。
            request_id: Router 请求 ID。

        返回:
            无返回值。重复通知是幂等的；新容量样本到达后才实际收缩预算。
        """
        (
            previous_blocks,
            steady_state_blocks,
            remaining_blocks,
            capacity_ack_pending,
        ) = self.pd_decode_blocks.reconcile(
            worker_id,
            request_id,
            capacity_generation,
        )
        if previous_blocks < 0 or not capacity_ack_pending:
            return
        logger.info(
            "[PDDecodeAdmission] request={} worker={} attach observed; "
            "waiting for fresh BlockPool capacity sample previous_blocks={} "
            "remaining_decode_growth_blocks={} "
            "worker_reserved_blocks={}",
            request_id,
            worker_id,
            previous_blocks,
            steady_state_blocks,
            remaining_blocks,
        )

    def mark_pd_decode_blocks_allocated(
        self,
        worker_id: str,
        request_id: str,
        capacity_generation: int | None = None,
    ) -> None:
        """记录D Scheduler已经分配目标block，但继续保留完整预算。

        参数:
            worker_id: 已完成目标block分配的D Worker ID。
            request_id: Router请求ID。

        返回:
            无返回值。allocated时容量尚可能没有反映到BlockPool
            快照；只有attached回调才会启动后续预算收缩。
        """
        (
            previous_blocks,
            steady_state_blocks,
            remaining_blocks,
            allocation_recorded,
        ) = self.pd_decode_blocks.mark_allocated(
            worker_id,
            request_id,
            capacity_generation,
        )
        if previous_blocks < 0 or not allocation_recorded:
            return
        logger.info(
            "[PDDecodeAdmission] request={} worker={} Scheduler target blocks "
            "allocated; retaining full reservation until attach "
            "previous_blocks={} remaining_decode_growth_blocks={} "
            "worker_reserved_blocks={}",
            request_id,
            worker_id,
            previous_blocks,
            steady_state_blocks,
            remaining_blocks,
        )

    def restore_pd_decode_budget_after_fallback(
        self,
        worker_id: str,
        request_id: str,
    ) -> None:
        """PD目标block释放后恢复请求的完整D准入预算。

        参数:
            worker_id: 转为本地重算的D Worker ID。
            request_id: Router请求ID。

        返回:
            无返回值。重复通知是幂等的；请求完成后到达的迟到通知会被忽略。
        """
        (
            previous_blocks,
            restored_blocks,
            remaining_blocks,
            restored,
        ) = self.pd_decode_blocks.restore_after_fallback(worker_id, request_id)
        if previous_blocks < 0 or not restored:
            return
        logger.info(
            "[PDDecodeAdmission] request={} worker={} PD target allocation "
            "rolled back; restored full admission budget previous_blocks={} "
            "restored_blocks={} worker_reserved_blocks={}",
            request_id,
            worker_id,
            previous_blocks,
            restored_blocks,
            remaining_blocks,
        )
        with self._release_cond:
            self._publish_capacity_change_locked((worker_id,))

    def _publish_capacity_change_locked(self, worker_ids=()) -> None:
        """Publish global and per-worker capacity versions under release_cond."""

        self._capacity_generation += 1
        for worker_id in worker_ids:
            if worker_id:
                self._capacity_worker_generations[worker_id] += 1
        self._release_cond.notify_all()

    def notify_metrics_updated(self) -> None:
        """在准入相关Worker状态实际变化时发布新容量版本。

        返回:
            无返回值。状态指纹未变化时不会递增generation或唤醒等待者。
        """
        all_workers = self.worker_manager.get_all_workers()
        workers = tuple(all_workers.values() if isinstance(all_workers, dict) else ())
        worker_signatures = {
            worker.worker_id: (
                str(worker.worker_type),
                int(getattr(worker, "num_gpu_blocks", 0) or 0),
                int(getattr(worker, "kvbm_free_blocks", -1)),
                int(getattr(worker, "active_decode_blocks", 0) or 0),
                int(getattr(worker, "running_requests", 0) or 0),
                int(getattr(worker, "queue_depth", 0) or 0),
                int(getattr(worker, "kvbm_block_size", 0) or 0),
                int(getattr(worker, "kvbm_capacity_generation", -1)),
            )
            for worker in workers
        }
        signature = tuple(sorted(worker_signatures.items()))
        with self._release_cond:
            if signature == self._capacity_state_signature:
                return
            changed_worker_ids = set()
            for matched_selected_worker_id in self._capacity_worker_state_signatures.keys() | worker_signatures.keys():
                if self._capacity_worker_state_signatures.get(matched_selected_worker_id) != worker_signatures.get(
                    matched_selected_worker_id
                ):
                    changed_worker_ids.add(matched_selected_worker_id)
            self._capacity_state_signature = signature
            self._capacity_worker_state_signatures = worker_signatures
            self._publish_capacity_change_locked(changed_worker_ids)

    def wait_for_capacity(self, timeout: float) -> bool:
        """兼容旧调用方，等待任意一次容量通知。"""
        with self._release_cond:
            return self._release_cond.wait(timeout=timeout)

    def enqueue_capacity_waiter(
        self,
        request_id: str,
        model: str,
        *,
        only_if_contended: bool = False,
    ) -> bool:
        """按到达顺序登记容量等待请求。

        快速路径请求不必排队；一旦同模型已有等待者，后到请求必须加入
        队尾，避免容量更新时越过等待更久的请求抢占预算。
        """

        with self._release_cond:
            waiters = self._capacity_waiters.get(model)
            if only_if_contended and not waiters:
                return False
            if waiters is None:
                waiters = {}
                self._capacity_waiters[model] = waiters
            if request_id in waiters:
                return True
            self._capacity_waiter_sequence += 1
            waiters[request_id] = _CapacityWaiter(
                sequence=self._capacity_waiter_sequence,
                enqueued_at=time.monotonic(),
            )
            self._capacity_waiter_models[request_id] = model
            self._release_cond.notify_all()
            return True

    def defer_capacity_waiter(
        self,
        request_id: str,
        model: str,
        generation: int,
        *,
        blocked_worker_ids=(),
        retry_after: float | None = None,
    ) -> bool:
        """Temporarily yield a routing slot after an unsuccessful attempt.

        The original FIFO sequence is retained. Only a capacity update from a
        worker that rejected this request reactivates it. Transient confirmation
        and affinity timers may provide an explicit bounded retry deadline.
        """

        blocked_workers = tuple(dict.fromkeys(worker_id for worker_id in blocked_worker_ids if worker_id))
        if not blocked_workers:
            return False
        rejection_retry_delay = self._pd_decode_confirmation_retry_delay(request_id)
        effective_retry_after = retry_after
        if rejection_retry_delay is not None:
            rejection_retry_delay = max(float(rejection_retry_delay), 0.0)
            effective_retry_after = (
                rejection_retry_delay
                if effective_retry_after is None
                else min(float(effective_retry_after), rejection_retry_delay)
            )
        with self._release_cond:
            waiter = self._capacity_waiters.get(model, {}).get(request_id)
            if waiter is None:
                return False
            observed_generation = int(generation)
            if self._capacity_generation != observed_generation:
                waiter.parked_worker_generations = None
                waiter.parked_at = None
                waiter.retry_at = None
                self._release_cond.notify_all()
                return False
            waiter.parked_worker_generations = {
                worker_id: self._capacity_worker_generations.get(worker_id, 0) for worker_id in blocked_workers
            }
            waiter.parked_at = time.monotonic()
            waiter.retry_at = (
                waiter.parked_at + max(float(effective_retry_after), 0.0) if effective_retry_after is not None else None
            )
            requests_ahead = sum(
                candidate.sequence < waiter.sequence
                for candidate in self._capacity_waiters.get(model, {}).values()
                if candidate.parked_worker_generations is None
            )
            logger.debug(
                "[PDCapacityWaiter] request={} model={} action=park "
                "blocked_workers={} worker_generations={} queue_position={} "
                "retry_ms={:.3f}",
                request_id,
                model,
                ",".join(blocked_workers),
                waiter.parked_worker_generations,
                requests_ahead,
                (max(waiter.retry_at - waiter.parked_at, 0.0) * 1000.0 if waiter.retry_at is not None else -1.0),
            )
            self._release_cond.notify_all()
            return True

    def _refresh_capacity_waiter_locked(
        self,
        request_id: str,
        model: str,
        waiter: _CapacityWaiter,
        now: float,
    ) -> bool:
        """Reactivate a parked waiter when capacity changes or retry is due."""

        parked_generations = waiter.parked_worker_generations
        if parked_generations is None:
            return True
        changed_workers = [
            worker_id
            for worker_id, generation in parked_generations.items()
            if self._capacity_worker_generations.get(worker_id, 0) != generation
        ]
        worker_changed = bool(changed_workers) and (
            self._pd_decode_capacity_change_releases_rejection(
                request_id,
                changed_workers,
            )
        )
        if changed_workers and not worker_changed:
            for worker_id in changed_workers:
                parked_generations[worker_id] = self._capacity_worker_generations.get(worker_id, 0)
        retry_due = waiter.retry_at is not None and waiter.retry_at <= now
        if worker_changed or retry_due:
            parked_ms = max(now - waiter.parked_at, 0.0) * 1000.0 if waiter.parked_at is not None else 0.0
            if parked_ms >= _PD_CAPACITY_WAITER_LOG_THRESHOLD_SECONDS * 1000.0:
                logger.info(
                    "[PDCapacityWaiter] request={} model={} action=wake reason={} "
                    "wake_workers={} blocked_workers={} parked_ms={:.3f}",
                    request_id,
                    model,
                    "worker_generation" if worker_changed else "retry_timer",
                    ",".join(changed_workers),
                    ",".join(parked_generations),
                    parked_ms,
                )
            waiter.parked_worker_generations = None
            waiter.parked_at = None
            waiter.retry_at = None
            return True
        return False

    def wait_for_capacity_turn(
        self,
        request_id: str,
        model: str,
        timeout: float,
        *,
        window_size: int = 1,
    ) -> bool:
        """等待请求进入同模型容量队列的有界准入窗口。"""

        deadline = time.monotonic() + max(float(timeout), 0.0)
        admitted_window = max(int(window_size), 1)
        with self._release_cond:
            while True:
                waiters = self._capacity_waiters.get(model, {})
                waiter = waiters.get(request_id)
                if waiter is None:
                    return False
                now = time.monotonic()
                active_waiters = []
                for candidate_request_id, candidate in waiters.items():
                    if self._refresh_capacity_waiter_locked(
                        candidate_request_id,
                        model,
                        candidate,
                        now,
                    ):
                        active_waiters.append(candidate)
                if waiter.parked_worker_generations is not None:
                    remaining = deadline - now
                    if remaining <= 0:
                        return False
                    if waiter.retry_at is not None:
                        remaining = min(
                            remaining,
                            max(waiter.retry_at - now, 0.0),
                        )
                    self._release_cond.wait(timeout=remaining)
                    continue
                requests_ahead = sum(candidate.sequence < waiter.sequence for candidate in active_waiters)
                if requests_ahead < admitted_window:
                    queue_wait_ms = max(now - waiter.enqueued_at, 0.0) * 1000.0
                    if queue_wait_ms >= _PD_CAPACITY_WAITER_LOG_THRESHOLD_SECONDS * 1000.0:
                        logger.info(
                            "[PDCapacityWaiter] request={} model={} action=turn "
                            "queue_position={} window_size={} queue_wait_ms={:.3f}",
                            request_id,
                            model,
                            requests_ahead,
                            admitted_window,
                            queue_wait_ms,
                        )
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._release_cond.wait(timeout=remaining)

    def remove_capacity_waiter(self, request_id: str, model: str) -> None:
        """注销容量等待票据并唤醒下一位请求。"""

        self._clear_pd_decode_confirmation_rejections(request_id)
        with self._release_cond:
            waiters = self._capacity_waiters.get(model)
            if not waiters or request_id not in waiters:
                self._capacity_waiter_models.pop(request_id, None)
                return
            del waiters[request_id]
            self._capacity_waiter_models.pop(request_id, None)
            if not waiters:
                self._capacity_waiters.pop(model, None)
            self._release_cond.notify_all()

    def _capacity_waiter_wake_delay_locked(
        self,
        request_id: str | None,
    ) -> tuple[bool, float | None]:
        """Return parked state and delay until its next targeted wake-up."""

        if not request_id:
            return False, None
        model = self._capacity_waiter_models.get(request_id)
        if model is None:
            return False, None
        waiter = self._capacity_waiters.get(model, {}).get(request_id)
        if waiter is None or waiter.parked_worker_generations is None:
            return False, None
        if any(
            self._capacity_worker_generations.get(worker_id, 0) != generation
            for worker_id, generation in waiter.parked_worker_generations.items()
        ):
            changed_workers = [
                worker_id
                for worker_id, generation in waiter.parked_worker_generations.items()
                if self._capacity_worker_generations.get(worker_id, 0) != generation
            ]
            if self._pd_decode_capacity_change_releases_rejection(
                request_id,
                changed_workers,
            ):
                return True, 0.0
            for worker_id in changed_workers:
                waiter.parked_worker_generations[worker_id] = self._capacity_worker_generations.get(worker_id, 0)
        if waiter.retry_at is None:
            return True, None
        return True, waiter.retry_at - time.monotonic()

    def get_capacity_generation(self) -> int:
        """返回当前容量状态版本。

        返回:
            每次Worker指标更新或请求预算释放后单调递增的整数版本。
        """

        with self._release_cond:
            return self._capacity_generation

    def wait_for_capacity_change(
        self,
        generation: int,
        timeout: float,
        *,
        request_id: str | None = None,
    ) -> bool:
        """等待容量版本发生变化，避免无变化时重复执行路由。

        参数:
            generation: 本次路由尝试读取到的容量版本。
            timeout: 最长等待秒数。

        返回:
            版本已变化返回True；等待超时且版本未变化返回False。
        """

        deadline = time.monotonic() + max(float(timeout), 0.0)
        with self._release_cond:
            while True:
                waiter_parked, waiter_wake_delay = self._capacity_waiter_wake_delay_locked(request_id)
                if waiter_parked:
                    model = self._capacity_waiter_models.get(request_id)
                    waiter = self._capacity_waiters.get(model, {}).get(request_id)
                    parked_generations = waiter.parked_worker_generations or {} if waiter is not None else {}
                    worker_changed = waiter is not None and any(
                        self._capacity_worker_generations.get(worker_id, 0) != parked_generation
                        for worker_id, parked_generation in parked_generations.items()
                    )
                    if worker_changed:
                        return True
                    if waiter_wake_delay is not None and waiter_wake_delay <= 0:
                        # A retry timer is a wake-up opportunity, not a
                        # capacity generation change.
                        return False
                elif self._capacity_generation != generation:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                retry_delay = self._pd_decode_confirmation_retry_delay(request_id)
                if retry_delay is not None:
                    if retry_delay <= 0:
                        return False
                    remaining = min(remaining, retry_delay)
                if waiter_wake_delay is not None:
                    remaining = min(remaining, waiter_wake_delay)
                self._release_cond.wait(timeout=remaining)

    def _clear_pd_decode_confirmation_rejections(self, request_id: str) -> None:
        if not request_id:
            return
        with self._pd_decode_confirmation_rejection_lock:
            self._pd_decode_confirmation_rejections.pop(request_id, None)
            self._pd_decode_affinity_rejected_at.pop(request_id, None)
            self._pd_decode_affinity_spill_ready.pop(request_id, None)

    def _pd_decode_confirmation_retry_delay(
        self,
        request_id: str | None,
    ) -> float | None:
        """返回距离下一次重试或权威复核唤醒的秒数。"""
        if not request_id:
            return None
        now = time.monotonic()
        with self._pd_decode_confirmation_rejection_lock:
            entries = self._pd_decode_confirmation_rejections.get(request_id, {})
            wake_times = []
            for selected_wake_at in (
                entry.retry_at if entry.retry_at is not None else entry.probe_at for entry in entries.values()
            ):
                if selected_wake_at is not None:
                    wake_times.append(selected_wake_at)
            spill_ready = self._pd_decode_affinity_spill_ready[request_id]
            for worker_id, rejected_at in self._pd_decode_affinity_rejected_at.get(
                request_id,
                {},
            ).items():
                if worker_id in spill_ready:
                    continue
                spill_at = rejected_at + _PD_DECODE_AFFINITY_SPILL_GRACE_SECONDS
                if spill_at <= now:
                    spill_ready.add(worker_id)
                    wake_times.append(now)
                else:
                    wake_times.append(spill_at)
        if not wake_times:
            return None
        return min(wake_times) - now

    def _pd_decode_affinity_spill_allowed(
        self,
        request_id: str,
        worker_id: str,
    ) -> bool:
        """判断 affinity D 拒绝后的短恢复窗口是否已经结束。"""
        now = time.monotonic()
        with self._pd_decode_confirmation_rejection_lock:
            if worker_id in self._pd_decode_affinity_spill_ready.get(
                request_id,
                set(),
            ):
                return True
            rejected_at = self._pd_decode_affinity_rejected_at.get(
                request_id,
                {},
            ).get(worker_id)
            if rejected_at is None:
                return True
            if now - rejected_at >= _PD_DECODE_AFFINITY_SPILL_GRACE_SECONDS:
                self._pd_decode_affinity_spill_ready[request_id].add(worker_id)
                return True
            return False

    def _get_pd_decode_confirmation_rejections(
        self,
        request_id: str,
        decode_workers: list,
    ) -> dict[str, dict]:
        """返回仍与当前 D 容量状态一致的请求级拒绝结果。"""
        if not request_id:
            return {}
        workers = {worker.worker_id: worker for worker in decode_workers}
        now = time.monotonic()
        valid: dict[str, dict] = {}
        with self._pd_decode_confirmation_rejection_lock:
            entries = self._pd_decode_confirmation_rejections.get(request_id, {})
            for worker_id, entry in list(entries.items()):
                worker = workers.get(worker_id)
                retry_due = entry.retry_at is not None and now >= entry.retry_at
                probe_due = entry.probe_at is not None and now >= entry.probe_at
                if worker is None:
                    entries.pop(worker_id, None)
                    continue
                if retry_due or probe_due:
                    # Keep the entry so a subsequent authoritative
                    # rejection refreshes the baseline (backoff/probe
                    # cadence) instead of restarting from scratch.
                    continue
                if self._pd_decode_capacity_rejection_released(entry, worker):
                    entries.pop(worker_id, None)
                    continue
                details = dict(entry.diagnostics)
                details["capacity_confirmation_cached"] = True
                valid[worker_id] = details
            if not entries:
                self._pd_decode_confirmation_rejections.pop(request_id, None)
        return valid

    def _record_pd_decode_confirmation_rejection(
        self,
        request_id: str,
        worker,
        diagnostics: dict,
        *,
        affinity_worker_id: str | None = None,
    ) -> dict:
        """缓存D拒绝，并按瞬时失败、代际和复用状态安排精确唤醒。"""
        if not request_id:
            return diagnostics
        raw_generation = diagnostics.get(
            "capacity_confirmation_generation",
            getattr(worker, "kvbm_capacity_generation", -1),
        )
        capacity_generation = int(raw_generation if raw_generation is not None else -1)
        reason = str(diagnostics.get("reason", ""))
        raw_free_blocks = diagnostics.get("free_blocks", -1)
        raw_available_blocks = diagnostics.get("available_blocks", -1)
        free_blocks = int(raw_free_blocks if raw_free_blocks is not None else -1)
        available_blocks = int(raw_available_blocks if raw_available_blocks is not None else -1)
        has_confirmed_capacity = free_blocks >= 0 and available_blocks >= 0
        candidate_reusable_blocks = max(
            int(
                diagnostics.get(
                    "capacity_confirmation_candidate_reusable_blocks",
                    diagnostics.get("candidate_reusable_prompt_blocks", 0),
                )
                or 0
            ),
            0,
        )
        confirmed_reusable_blocks = max(
            int(diagnostics.get("confirmed_reusable_prompt_blocks", 0) or 0),
            0,
        )
        reuse_refresh_needed = candidate_reusable_blocks > confirmed_reusable_blocks
        definitive_capacity_rejection = bool(
            diagnostics.get("capacity_confirmed")
            and reason in {"decode_free_block_budget", "decode_block_budget"}
            and (has_confirmed_capacity or capacity_generation >= 0)
        )
        now = time.monotonic()
        with self._pd_decode_confirmation_rejection_lock:
            previous = self._pd_decode_confirmation_rejections.get(
                request_id,
                {},
            ).get(worker.worker_id)
            failure_count = 0
            retry_at = None
            probe_at = None
            details = dict(diagnostics)
            if not definitive_capacity_rejection:
                failure_count = (
                    previous.failure_count + 1 if previous is not None and previous.retry_at is not None else 1
                )
                retry_delay = min(
                    _PD_DECODE_CONFIRMATION_RETRY_BASE_SECONDS * (2 ** min(failure_count - 1, 2)),
                    _PD_DECODE_CONFIRMATION_RETRY_MAX_SECONDS,
                )
                is_small_delta_query = (
                    reason == "decode_capacity_confirmation_failed"
                    and 0 < int(diagnostics.get("estimated_blocks", 0) or 0) <= _PD_DECODE_SMALL_DELTA_BLOCKS
                )
                if is_small_delta_query and candidate_reusable_blocks > 0:
                    if _capacity_query_timed_out(
                        {
                            "error": diagnostics.get("capacity_confirmation_error", ""),
                            "error_type": diagnostics.get("capacity_confirmation_error_type", ""),
                        }
                    ):
                        retry_delay = _PD_DECODE_CONFIRMATION_RETRY_BASE_SECONDS
                details["capacity_confirmation_wait_class"] = "query_retry"
                retry_at = now + retry_delay
                details["capacity_confirmation_failure_count"] = failure_count
                details["capacity_confirmation_retry_delay_ms"] = retry_delay * 1000.0
            else:
                generationless = capacity_generation < 0
                if generationless or reuse_refresh_needed:
                    # generation缺失时无法依靠事件判断状态变化；候选复用大于
                    # Worker确认复用时，free阈值也无法反映后续prefix可见性。
                    # 这两类请求才需要低频权威复核。
                    wait_class = "generationless_probe" if generationless else "reuse_refresh_probe"
                    probe_delay = _PD_DECODE_CONFIRMATION_PROBE_INTERVAL_SECONDS * (
                        1.0
                        + random.uniform(
                            -_PD_DECODE_CONFIRMATION_PROBE_JITTER,
                            _PD_DECODE_CONFIRMATION_PROBE_JITTER,
                        )
                    )
                    probe_at = now + probe_delay
                    details["capacity_confirmation_probe_delay_ms"] = probe_delay * 1000.0
                else:
                    wait_class = "threshold_wait"
                details["capacity_confirmation_wait_class"] = wait_class
            entry = _DecodeCapacityRejection(
                capacity_generation=capacity_generation,
                reserved_blocks=self.pd_decode_blocks.get_blocks(worker.worker_id),
                inflight=self.inflight.get(worker.worker_id),
                free_blocks=free_blocks,
                available_blocks=available_blocks,
                estimated_blocks=max(
                    int(diagnostics.get("estimated_blocks", 0) or 0),
                    0,
                ),
                dynamic_margin_blocks=max(
                    int(diagnostics.get("dynamic_margin_blocks", 0) or 0),
                    0,
                ),
                failure_count=failure_count,
                retry_at=retry_at,
                diagnostics=details,
                probe_at=probe_at,
            )
            self._pd_decode_confirmation_rejections[request_id][worker.worker_id] = entry
            if worker.worker_id == affinity_worker_id:
                self._pd_decode_affinity_rejected_at[request_id].setdefault(
                    worker.worker_id,
                    now,
                )
        return details

    def _pd_decode_capacity_rejection_released(
        self,
        entry: _DecodeCapacityRejection,
        worker,
    ) -> bool:
        """Return whether a confirmed rejection can pass with current capacity."""

        if entry.retry_at is not None:
            return time.monotonic() >= entry.retry_at
        raw_current_generation = getattr(worker, "kvbm_capacity_generation", -1)
        current_generation = int(raw_current_generation if raw_current_generation is not None else -1)
        if (
            entry.capacity_generation >= 0
            and current_generation >= 0
            and current_generation != entry.capacity_generation
        ):
            # free阈值可能保持不变，但新的代际可能包含该请求可复用prefix的
            # attach/evict变化。每个新代际只放行一次权威复核；若仍拒绝，
            # 新记录会绑定到最新代际。
            return True
        current_free_blocks = int(getattr(worker, "kvbm_free_blocks", -1))
        if entry.free_blocks < 0 or entry.available_blocks < 0 or current_free_blocks < 0:
            return (
                self.pd_decode_blocks.get_blocks(worker.worker_id) < entry.reserved_blocks
                or self.inflight.get(worker.worker_id) < entry.inflight
                or int(getattr(worker, "kvbm_capacity_generation", -1)) != entry.capacity_generation
            )
        current_reserved_blocks = self.pd_decode_blocks.get_blocks(worker.worker_id)
        current_available_blocks = max(
            entry.available_blocks
            + (current_free_blocks - entry.free_blocks)
            + (entry.reserved_blocks - current_reserved_blocks),
            0,
        )
        current_running = max(
            int(getattr(worker, "running_requests", 0) or 0),
            0,
        )
        current_inflight = self.inflight.get(worker.worker_id)
        current_margin_blocks = max(
            2,
            min(max(current_inflight, current_running), 8),
        )
        required_blocks = entry.estimated_blocks + current_margin_blocks
        return current_available_blocks >= required_blocks

    def _pd_decode_capacity_change_releases_rejection(
        self,
        request_id: str,
        worker_ids,
    ) -> bool:
        """Ignore worker generations that do not make this request admissible."""

        all_workers = self.worker_manager.get_all_workers()
        workers = all_workers if isinstance(all_workers, dict) else {}
        with self._pd_decode_confirmation_rejection_lock:
            entries = self._pd_decode_confirmation_rejections.get(request_id, {})
            for worker_id in worker_ids:
                entry = entries.get(worker_id)
                if entry is None:
                    return True
                worker = workers.get(worker_id)
                if worker is None:
                    return True
                if self._pd_decode_capacity_rejection_released(entry, worker):
                    if entry.retry_at is None:
                        entries.pop(worker_id, None)
                    return True
            if not entries:
                self._pd_decode_confirmation_rejections.pop(request_id, None)
        return False

    def wait_for_pd_prefix_affinity(
        self,
        route_hint: RouteHint,
        timeout: float,
    ) -> PendingPrefixMatch | None:
        """等待同一公共前缀的 HIXL leader 完成 import。

        该等待使用独立条件变量，不加入普通容量 FIFO，避免一个热点 prefix
        把其他 prefix 的请求挡在队首。Mooncake/native connector 不调用此方法。
        """
        return self._pd_pending_prefix_affinity.wait_for(
            request_id=route_hint.request_id,
            model=route_hint.model,
            token_ids=list(route_hint.token_ids or []),
            min_tokens=self._pd_pending_prefix_min_tokens,
            min_ratio=self._pd_pending_prefix_min_ratio,
            timeout=timeout,
        )

    def refresh_pd_prefix_affinity(
        self,
        route_hint: RouteHint,
        match: PendingPrefixMatch,
    ) -> bool:
        """刷新 pending leader 的 D 前缀快照并设置本请求的 D 亲和性。"""
        if match.model != route_hint.model or not route_hint.token_ids:
            return False
        snapshots: dict = {}
        overlap_scores = self.kvcache_manager.find_matches(
            token_ids=route_hint.token_ids,
            model=route_hint.model,
            include_workers={match.worker_id},
            snapshot_output=snapshots,
        )
        overlap_tokens = int(overlap_scores.get(match.worker_id, 0) or 0)
        overlap_ratio = overlap_tokens / max(len(route_hint.token_ids), 1)
        if overlap_tokens < self._pd_pending_prefix_min_tokens or overlap_ratio < self._pd_pending_prefix_min_ratio:
            self._pd_pending_prefix_affinity.discard(
                match.request_id,
                match.worker_id,
            )
            return False
        route_hint.precomputed_overlap_scores = dict(route_hint.precomputed_overlap_scores or {})
        route_hint.precomputed_overlap_scores.update(overlap_scores)
        route_hint.precomputed_prefix_snapshots = dict(route_hint.precomputed_prefix_snapshots or {})
        route_hint.precomputed_prefix_snapshots.update(snapshots)
        route_hint.pd_decode_affinity_worker_id = match.worker_id
        logger.info(
            "[PDPendingPrefixAffinity] request={} action=ready worker={} matched_tokens={} ratio={:.3f}",
            route_hint.request_id,
            match.worker_id,
            overlap_tokens,
            overlap_ratio,
        )
        return True

    def _find_pd_pending_prefix(
        self,
        route_hint: RouteHint,
        excluded_worker_ids: set[str] | None = None,
    ) -> PendingPrefixMatch | None:
        """返回与当前 HIXL 请求匹配的在途或刚完成 prefix leader。"""
        match = self._pd_pending_prefix_affinity.find(
            request_id=route_hint.request_id,
            model=route_hint.model,
            token_ids=list(route_hint.token_ids or []),
            min_tokens=self._pd_pending_prefix_min_tokens,
            min_ratio=self._pd_pending_prefix_min_ratio,
            excluded_worker_ids=excluded_worker_ids,
        )
        return match if match is not None and match.state == "pending" else None

    def _queue_relief_pd_decode_workers(
        self,
        workers: list,
        route_hint: RouteHint,
        *,
        comparison_workers: list | None = None,
    ) -> tuple[list, set[str]]:
        """Attempt a capacity-confirmed spill only for substantial engine queues."""
        if not workers or not self._requires_decode_capacity_confirmation():
            return workers, set()
        loads = {}
        comparison_workers = comparison_workers if comparison_workers is not None else workers
        for worker in comparison_workers:
            inflight = max(self.inflight.get(worker.worker_id), 0)
            running = max(int(getattr(worker, "running_requests", 0) or 0), 0)
            waiting = max(int(getattr(worker, "queue_depth", 0) or 0), 0)
            loads[worker.worker_id] = {
                "inflight": inflight,
                "running": running,
                "waiting": waiting,
                "pressure": max(inflight, running + waiting) + waiting,
            }
        excluded = set()
        for worker_id, load in loads.items():
            # A large shared-prefix batch can run efficiently. Running or
            # gate population alone is not evidence of a scheduler bottleneck.
            if (
                load["waiting"] < _PD_DECODE_SPILL_MIN_WAITING
                or load["waiting"] / max(load["running"], 1) < _PD_DECODE_SPILL_QUEUE_RATIO
            ):
                continue
            for other in workers:
                if other.worker_id == worker_id:
                    continue
                other_load = loads.get(other.worker_id)
                if other_load is None:
                    raise KeyError(other.worker_id)
                if (
                    other_load["waiting"] <= load["waiting"] - _PD_DECODE_SPILL_RELIEF_REQUESTS
                    and other_load["pressure"] <= load["pressure"] - _PD_DECODE_SPILL_RELIEF_REQUESTS
                ):
                    excluded.add(worker_id)
                    break
        if excluded:
            logger.info(
                "[PDDecodeLoadBalance] request={} action=queue_relief "
                "loads={} excluded={} affinity_worker={} capacity_confirmation_required=True",
                route_hint.request_id,
                loads,
                sorted(excluded),
                getattr(route_hint, "pd_decode_affinity_worker_id", None),
            )
        candidates = [worker for worker in workers if worker.worker_id not in excluded]
        return candidates, excluded

    def record_decode_tpot(self, worker_id: str, tpot_ms: float) -> None:
        """Record successful D completions for the optional hot-spot signal."""
        if not self._pd_hotspot_spill_enabled or not math.isfinite(tpot_ms) or tpot_ms <= 0:
            return
        with self._routing_lock:
            self._pd_decode_recent_tpot[worker_id].append((time.monotonic(), tpot_ms))

    def _recent_decode_tpot_locked(self, worker_id: str, now: float) -> float | None:
        samples = self._pd_decode_recent_tpot[worker_id]
        while samples and now - samples[0][0] > _PD_HOTSPOT_TPOT_WINDOW_SECONDS:
            samples.popleft()
        if len(samples) < _PD_HOTSPOT_TPOT_MIN_SAMPLES:
            return None
        return statistics.median(value for _, value in samples)

    def _hotspot_spill_pd_decode_workers(
        self,
        workers: list,
        route_hint: RouteHint,
    ) -> tuple[list, str | None]:
        """Offer one cold D before scoring, without bypassing capacity confirmation."""
        if not self._pd_hotspot_spill_enabled or not self._requires_decode_capacity_confirmation() or len(workers) < 2:
            return workers, None
        if not route_hint.token_ids:
            return workers, None
        overlap = route_hint.precomputed_overlap_scores or {}
        affinity_id = getattr(route_hint, "pd_decode_affinity_worker_id", None)
        if affinity_id is None:
            affinity_id = max(workers, key=lambda worker: int(overlap.get(worker.worker_id, 0) or 0)).worker_id
        hot = next((worker for worker in workers if worker.worker_id == affinity_id), None)
        if hot is None:
            return workers, None
        matched = int(overlap.get(affinity_id, 0) or 0)
        if (
            matched < self._pd_pending_prefix_min_tokens
            or matched / len(route_hint.token_ids) < self._pd_pending_prefix_min_ratio
        ):
            return workers, None
        now = time.monotonic()
        if now - self._pd_hotspot_last_spill_at.get(affinity_id, float("-inf")) < _PD_HOTSPOT_COOLDOWN_SECONDS:
            return workers, None
        hot_running = max(int(getattr(hot, "running_requests", 0) or 0), 0)
        hot_tpot = self._recent_decode_tpot_locked(affinity_id, now)
        if hot_tpot is None or (
            self._pd_hotspot_hot_min_running is not None and hot_running < self._pd_hotspot_hot_min_running
        ):
            return workers, None
        cold = []
        for worker in workers:
            if worker.worker_id == affinity_id:
                continue
            running = max(int(getattr(worker, "running_requests", 0) or 0), 0)
            peer_tpot = self._recent_decode_tpot_locked(worker.worker_id, now)
            if running >= hot_running:
                continue
            if self._pd_hotspot_cold_max_running is not None:
                if running > self._pd_hotspot_cold_max_running:
                    continue
            if self._pd_hotspot_running_gap is not None:
                if hot_running - running < self._pd_hotspot_running_gap:
                    continue
            if peer_tpot is not None and hot_tpot >= peer_tpot * self._pd_hotspot_tpot_ratio:
                cold.append((running, self.inflight.get(worker.worker_id), worker, peer_tpot))
        if not cold:
            return workers, None
        _, _, target, peer_tpot = min(cold, key=lambda entry: entry[:2])
        logger.info(
            "[PDDecodeHotspotSpill] request={} action=candidate source={} target={} "
            "running={} target_running={} tpot_ms={:.2f} target_tpot_ms={:.2f} "
            "matched_tokens={} capacity_confirmation_required=True",
            route_hint.request_id,
            affinity_id,
            target.worker_id,
            hot_running,
            int(getattr(target, "running_requests", 0) or 0),
            hot_tpot,
            peer_tpot,
            matched,
        )
        return [target], affinity_id

    def mark_pd_prefix_ready(
        self,
        request_id: str,
        worker_id: str,
        published_blocks: int,
    ) -> None:
        """在 D attach 后唤醒同前缀 HIXL 请求。"""
        self._pd_pending_prefix_affinity.mark_ready(
            request_id,
            worker_id,
            published_blocks,
        )
        logger.info(
            "[PDPendingPrefixAffinity] request={} action={} worker={} published_blocks={}",
            request_id,
            "ready" if int(published_blocks) > 0 else "released",
            worker_id,
            published_blocks,
        )
        with self._release_cond:
            self._publish_capacity_change_locked((worker_id,))

    def mark_pd_prefix_failed(self, request_id: str, worker_id: str) -> None:
        """PD fallback 时清理 pending leader 并唤醒等待者。"""
        self._pd_pending_prefix_affinity.discard(request_id, worker_id)
        logger.info(
            "[PDPendingPrefixAffinity] request={} action=failed worker={}",
            request_id,
            worker_id,
        )
        with self._release_cond:
            self._publish_capacity_change_locked((worker_id,))

    def _worker_route_snapshot(self, worker_id: str | None, route_hint: RouteHint) -> dict:
        if not worker_id:
            return {}
        worker = self.worker_manager.get_worker(worker_id)
        token_count = len(route_hint.token_ids or [])
        overlap_tokens = 0
        if route_hint.token_ids:
            precomputed = route_hint.precomputed_overlap_scores
            if precomputed is not None:
                overlap_tokens = int(precomputed.get(worker_id, 0))
            else:
                # 兼容未经过 data-plane 预计算的调用方，但仅匹配当前日志
                # Worker，避免诊断信息扫描全部候选 Worker。
                overlap_tokens = self.kvcache_manager.find_matches(
                    token_ids=route_hint.token_ids,
                    model=route_hint.model,
                    include_workers={worker_id},
                ).get(worker_id, 0)
        return {
            "worker_id": worker_id,
            "overlap_tokens": overlap_tokens,
            "overlap_ratio": (overlap_tokens / token_count) if token_count else 0.0,
            "queue_depth": worker.queue_depth if worker else None,
            "running_requests": worker.running_requests if worker else None,
            "current_load": worker.current_load if worker else None,
            "inflight": self.inflight.get(worker_id),
        }

    def _log_route_decision(
        self,
        *,
        route_hint: RouteHint,
        selected_workers: list[str],
        route_reason: str,
        total_ms: float,
    ) -> None:
        snapshots = [self._worker_route_snapshot(worker_id, route_hint) for worker_id in selected_workers if worker_id]
        algorithm_label = (
            self._pd_algorithm_label()
            if route_reason == "prefill_decode_pair"
            else type(self.current_algorithm).__name__
        )
        logger.info(
            "[ROUTE_DECISION] request={} model={} selected={} reason={} "
            "algorithm={} token_count={} workers={} took={:.2f}ms",
            route_hint.request_id,
            route_hint.model,
            selected_workers,
            route_reason,
            algorithm_label,
            len(route_hint.token_ids or []),
            snapshots,
            total_ms,
        )

    def _is_worker_admitted(self, worker) -> bool:
        if self._admission_mode == "dynamic":
            running = worker.running_requests
            waiting = worker.queue_depth
            if running > 0:
                return waiting / running < self._saturation_ratio
            # running=0 时（PD 模式 decode 等待 KV 传输、prefill 刚启动等），
            # worker 上报的 waiting/running 比值无意义，回退到 inflight 计数
            return self.inflight.get(worker.worker_id) < self._concurrency_limit
        return self.inflight.get(worker.worker_id) < self._concurrency_limit

    @staticmethod
    def _log_pd_admission_details(
        request_id: str,
        phase: str,
        workers: list,
        decisions: dict,
        selected_worker_id: str,
    ) -> None:
        """记录本次PD路由中所有健康Worker的最终准入结果。"""
        diagnostic_keys = (
            "reason",
            "mode",
            "generic_ratio_bypassed",
            "concurrency_limit",
            "block_ownership",
            "max_inflight",
            "estimated_tokens",
            "reserved_tokens",
            "projected_tokens",
            "token_capacity",
            "total_blocks",
            "observed_blocks",
            "reserved_blocks",
            "pending_reserved_blocks",
            "estimated_blocks",
            "steady_state_blocks",
            "usable_blocks",
            "projected_blocks",
            "free_blocks",
            "available_blocks",
            "dynamic_margin_blocks",
            "capacity_source",
            "free_blocks_age_ms",
        )
        worker_parts = []
        for worker in workers:
            admitted, details = decisions.get(
                worker.worker_id,
                (False, {"reason": "decision_unavailable"}),
            )
            detail_parts = [f"{key}={details[key]}" for key in diagnostic_keys if key in details]
            worker_parts.append(
                f"{worker.worker_id}(selected="
                f"{worker.worker_id == selected_worker_id},admitted={admitted},"
                f"load={worker.current_load:.3f},queue={worker.queue_depth},"
                f"running={worker.running_requests},"
                f"inflight={details.get('inflight', 0)},"
                f"{','.join(detail_parts)})"
            )
        logger.info(
            "[PDAdmissionDetail] request={} phase={} workers={}",
            request_id,
            phase,
            ";".join(worker_parts),
        )

    def _worker_block_size(self, worker_id: str) -> int:
        """获取 Worker 实际 KV block size，缺失时使用 Router 兼容值。"""
        worker = self.worker_manager.get_worker(worker_id)
        try:
            metric_block_size = int(getattr(worker, "kvbm_block_size", 0) or 0) if worker is not None else 0
        except (TypeError, ValueError):
            metric_block_size = 0
        if metric_block_size > 0:
            return metric_block_size
        try:
            return max(int(self.kvcache_manager.get_worker_block_size(worker_id)), 1)
        except (TypeError, ValueError):
            return max(int(settings.kv_cache_block_size), 1)

    def _estimate_missing_prompt(
        self,
        worker,
        route_hint: RouteHint | None,
    ) -> tuple[int, int]:
        """估算 Worker 需要计算的 Prompt token 和新增 block。

        参数:
            worker: 候选 P/D Worker。
            route_hint: 请求 token 与预计算前缀命中信息；为空时返回零预算。

        返回:
            ``(missing_tokens, missing_blocks)``。
        """
        if route_hint is None:
            return 0, 0
        prompt_tokens = len(route_hint.token_ids or [])
        overlap_tokens = 0
        if route_hint.precomputed_overlap_scores is not None:
            overlap_tokens = int(route_hint.precomputed_overlap_scores.get(worker.worker_id, 0))
        missing_tokens = max(prompt_tokens - min(max(overlap_tokens, 0), prompt_tokens), 0)
        return missing_tokens, math.ceil(missing_tokens / self._worker_block_size(worker.worker_id))

    def _prefill_admission_decision(
        self,
        worker,
        route_hint: RouteHint | None,
    ) -> tuple[bool, dict[str, int | float | str]]:
        """按实时队列、缺失 token 和 KV block 预算判断 P Worker 准入。

        参数:
            worker: 候选 P Worker。
            route_hint: 当前请求路由信息；诊断快照可传 None。

        返回:
            ``(admitted, diagnostics)``，诊断包含实时占用、Router 已预留预算
            以及本请求预计成本。
        """
        worker_id = worker.worker_id
        inflight = self.inflight.get(worker_id)
        estimated_tokens, estimated_blocks = self._estimate_missing_prompt(worker, route_hint)
        reserved_tokens, reserved_blocks = self.pd_prefill_budget.get(worker_id)
        token_capacity = max(int(getattr(worker, "max_num_batched_tokens", 0) or 0), 0)
        projected_tokens = reserved_tokens + estimated_tokens
        total_blocks = max(int(getattr(worker, "num_gpu_blocks", 0) or 0), 0)
        observed_blocks = max(
            int(getattr(worker, "active_decode_blocks", 0) or 0),
            int(total_blocks * float(getattr(worker, "kv_cache_usage", 0.0) or 0.0)),
        )
        usable_blocks = math.floor(total_blocks * (1.0 - self._pd_prefill_block_headroom_ratio))
        projected_blocks = max(observed_blocks, reserved_blocks) + estimated_blocks
        block_budget_exceeded = total_blocks > 0 and projected_blocks > usable_blocks
        # auto模式已有P专属的inflight与BlockPool预算。通用dynamic admission
        # 使用waiting/running比值，running较小时一个waiting请求就会拒绝整台
        # P Worker，因此仅在非auto模式保留该通用限制。
        admitted = True if self._pd_prefill_admission_mode == "auto" else self._is_worker_admitted(worker)
        reason = "admitted"
        if not admitted:
            reason = "worker_admission"
        elif self._pd_prefill_max_inflight > 0 and inflight >= self._pd_prefill_max_inflight:
            admitted = False
            reason = "prefill_inflight_limit"
        elif (
            self._pd_prefill_admission_mode == "auto"
            and total_blocks <= 0
            and inflight >= self._pd_prefill_auto_fallback_max_inflight
        ):
            admitted = False
            reason = "prefill_metrics_fallback_limit"
        elif (
            self._pd_prefill_admission_mode == "auto"
            and self._pd_prefill_block_admission_enabled
            and block_budget_exceeded
        ):
            admitted = False
            reason = "prefill_block_budget"
        return admitted, {
            "reason": reason,
            "mode": self._pd_prefill_admission_mode,
            "inflight": inflight,
            "max_inflight": self._pd_prefill_max_inflight,
            "running": int(getattr(worker, "running_requests", 0) or 0),
            "waiting": int(getattr(worker, "queue_depth", 0) or 0),
            "estimated_tokens": estimated_tokens,
            "reserved_tokens": reserved_tokens,
            "projected_tokens": projected_tokens,
            "token_capacity": token_capacity,
            "fallback_max_inflight": self._pd_prefill_auto_fallback_max_inflight,
            "total_blocks": total_blocks,
            "observed_blocks": observed_blocks,
            "reserved_blocks": reserved_blocks,
            "estimated_blocks": estimated_blocks,
            "usable_blocks": usable_blocks,
            "projected_blocks": projected_blocks,
        }

    def _estimate_pd_decode_budget(
        self,
        worker,
        route_hint: RouteHint | None,
    ) -> tuple[int, int]:
        """估算请求在目标 D Worker 上需要新增的 KV block 数。

        参数:
            worker: 候选 D Worker。
            route_hint: 当前请求的 token、输出长度和前缀匹配信息；为空时返回 0。

        返回:
            ``(attach前总预算, attach后增长预算)``。总预算包含缺失 Prompt
            和完整预计输出；增长预算扣除 Worker 已开始分配的初始窗口。

        说明:
            scheduler所有权下先返回完整保守预算。只有D EngineCore
            在派发前确认连续NPU前缀仍可复用时，准入逻辑才会扣减已确认
            的Prompt blocks。legacy模式仍保留原有的Router前缀估算。
        """
        if route_hint is None:
            return 0, 0
        prompt_tokens = len(route_hint.token_ids or [])
        overlap_tokens = 0
        if self._pd_block_ownership == "legacy" and route_hint.precomputed_overlap_scores is not None:
            overlap_tokens = int(route_hint.precomputed_overlap_scores.get(worker.worker_id, 0))
        overlap_tokens = min(max(overlap_tokens, 0), prompt_tokens)
        block_size = self._worker_block_size(worker.worker_id)
        missing_prompt_blocks = math.ceil(max(prompt_tokens - overlap_tokens, 0) / block_size)
        output_tokens = max(int(route_hint.estimated_output_tokens or 0), 1)
        requested_decode_blocks = math.ceil(output_tokens / block_size)
        full_decode_blocks = requested_decode_blocks
        remaining_decode_growth_blocks = (
            max(full_decode_blocks - self._pd_decode_initial_window_blocks, 0)
            if self._pd_decode_admission_mode == "auto"
            else full_decode_blocks
        )
        return (
            missing_prompt_blocks + full_decode_blocks,
            remaining_decode_growth_blocks,
        )

    def _pd_decode_npu_reuse_hint(
        self,
        worker,
        route_hint: RouteHint | None,
    ) -> tuple[list[str], int]:
        """返回Router快照中D可连续复用的完整NPU prefix blocks。"""
        if route_hint is None or not route_hint.token_ids:
            return [], 0
        snapshots = route_hint.precomputed_prefix_snapshots or {}
        snapshot = snapshots.get(worker.worker_id)
        if snapshot is None:
            return [], 0
        block_size = self._worker_block_size(worker.worker_id)
        full_prompt_tokens = len(route_hint.token_ids) // block_size * block_size
        hashes: list[str] = []
        for block in snapshot.blocks:
            if int(block.token_end) > full_prompt_tokens:
                break
            alias = block.select({CacheLocation.NPU})
            if alias is None:
                break
            hashes.append(str(alias.block_hash))
        return hashes, len(hashes)

    def _estimate_pd_decode_blocks(self, worker, route_hint: RouteHint | None) -> int:
        """返回请求 attach 前需要保障的 D block 总预算。"""
        return self._estimate_pd_decode_budget(worker, route_hint)[0]

    def _decode_admission_decision(
        self,
        worker,
        route_hint: RouteHint | None,
        *,
        confirmed_reusable_prompt_blocks: int | None = None,
    ) -> tuple[bool, dict[str, int | float | str]]:
        """判断 D Worker 是否满足请求数与 KV block 双重准入。

        参数:
            worker: 候选 D Worker。
            route_hint: 当前请求路由信息；诊断快照调用可传 None。

        返回:
            二元组 ``(admitted, diagnostics)``。diagnostics 包含拒绝原因、
            inflight、实时/Router预算 block 和预计新增 block。
        """
        worker_id = worker.worker_id
        inflight = self.inflight.get(worker_id)
        full_estimated_blocks, steady_state_blocks = self._estimate_pd_decode_budget(worker, route_hint)
        reuse_hashes, candidate_reusable_prompt_blocks = self._pd_decode_npu_reuse_hint(worker, route_hint)
        block_size = self._worker_block_size(worker_id)
        output_blocks = (
            math.ceil(max(int(route_hint.estimated_output_tokens or 0), 1) / block_size)
            if route_hint is not None
            else 0
        )
        max_reusable_prompt_blocks = max(full_estimated_blocks - output_blocks, 0)
        confirmed_reuse = (
            min(
                max(int(confirmed_reusable_prompt_blocks), 0),
                max_reusable_prompt_blocks,
            )
            if confirmed_reusable_prompt_blocks is not None
            else 0
        )
        # Router NPU索引只用于缩小候选阶段的预算，避免高复用请求在进入
        # engine-core确认前被完整Prompt预算挡住。最终提交仍必须使用Worker
        # 返回的confirmed_reusable_prompt_blocks重新计算，不能把索引当事实源。
        provisional_reuse = confirmed_reuse
        if confirmed_reusable_prompt_blocks is None and self._requires_decode_capacity_confirmation():
            provisional_reuse = min(
                max(int(candidate_reusable_prompt_blocks), 0),
                max_reusable_prompt_blocks,
            )
        estimated_blocks = full_estimated_blocks - provisional_reuse
        total_blocks = max(int(getattr(worker, "num_gpu_blocks", 0) or 0), 0)
        observed_blocks = max(
            int(getattr(worker, "active_decode_blocks", 0) or 0),
            int(total_blocks * float(getattr(worker, "kv_cache_usage", 0.0) or 0.0)),
        )
        capacity_updated_at = float(getattr(worker, "kvbm_block_pool_updated_at", 0.0) or 0.0)
        capacity_age_seconds = (
            max(time.monotonic() - capacity_updated_at, 0.0) if capacity_updated_at > 0.0 else math.inf
        )
        reported_free_blocks = int(getattr(worker, "kvbm_free_blocks", -1))
        authoritative_free_blocks = (
            max(reported_free_blocks, 0)
            if reported_free_blocks >= 0 and capacity_age_seconds <= self._pd_block_pool_metrics_ttl_seconds
            else None
        )
        capacity_acknowledged_requests = 0
        if authoritative_free_blocks is not None:
            (
                capacity_acknowledged_requests,
                reserved_blocks,
            ) = self.pd_decode_blocks.acknowledge_capacity(
                worker_id,
                capacity_updated_at,
                int(getattr(worker, "kvbm_capacity_generation", -1)),
            )
        else:
            reserved_blocks = self.pd_decode_blocks.get_blocks(worker_id)
        fallback_headroom_blocks = math.ceil(total_blocks * self._pd_decode_block_headroom_ratio)
        # Router已经准入但尚未完成attach的请求可能还没有反映到engine-core
        # free_blocks中，因此必须扣除全部在途预算。attach完成后预算会被协调为
        # steady-state并在请求结束时释放，短暂的保守计算不会永久压低容量。
        pending_reserved_blocks = reserved_blocks if authoritative_free_blocks is not None else 0
        running_requests = max(
            int(getattr(worker, "running_requests", 0) or 0),
            0,
        )
        # engine-core容量快照与真正allocate_slots之间存在短暂竞态。
        # 为每个活跃请求最多预留一个小窗口，并设置上下界，避免仅剩一个
        # block余量时仍贴边放行，也避免在低压场景重复扣除大比例headroom。
        dynamic_margin_blocks = max(2, min(max(inflight, running_requests), 8))
        # 新鲜engine-core free_blocks已经包含真实物理分配；Router再扣除尚未
        # 物化的reservation和动态竞态余量即可。固定比例headroom仅用于缺少
        # 权威容量时的估算兜底，否则994块Worker会被人为限制在约84%利用率。
        capacity_headroom_blocks = 0 if authoritative_free_blocks is not None else (fallback_headroom_blocks)
        available_blocks = (
            max(
                authoritative_free_blocks - pending_reserved_blocks - capacity_headroom_blocks,
                0,
            )
            if authoritative_free_blocks is not None
            else -1
        )
        usable_blocks = math.floor(total_blocks * (1.0 - self._pd_decode_block_headroom_ratio))
        projected_blocks = observed_blocks + reserved_blocks + estimated_blocks
        fallback_block_budget_exceeded = total_blocks > 0 and projected_blocks > usable_blocks
        reason = "admitted"
        generic_ratio_bypassed = bool(
            self._requires_decode_capacity_confirmation()
            and self._admission_mode == "dynamic"
            and worker.worker_type == WorkerType.DECODE
        )
        # HIXL auto already guards incremental KV capacity. A small running
        # denominator must not prevent that check; retain the global inflight cap.
        admitted = inflight < self._concurrency_limit if generic_ratio_bypassed else self._is_worker_admitted(worker)
        if not admitted:
            reason = "decode_concurrency_limit" if generic_ratio_bypassed else "worker_admission"
        elif self._pd_decode_max_inflight > 0 and inflight >= self._pd_decode_max_inflight:
            admitted = False
            reason = "decode_inflight_limit"
        elif (
            self._pd_decode_block_admission_enabled
            and authoritative_free_blocks is not None
            and estimated_blocks + dynamic_margin_blocks > available_blocks
        ):
            admitted = False
            reason = "decode_free_block_budget"
        elif (
            self._pd_decode_block_admission_enabled
            and authoritative_free_blocks is None
            and fallback_block_budget_exceeded
        ):
            admitted = False
            reason = "decode_block_budget"
        return admitted, {
            "reason": reason,
            "mode": self._pd_decode_admission_mode,
            "block_ownership": self._pd_block_ownership,
            "generic_ratio_bypassed": generic_ratio_bypassed,
            "concurrency_limit": self._concurrency_limit,
            "inflight": inflight,
            "running": running_requests,
            "waiting": max(int(getattr(worker, "queue_depth", 0) or 0), 0),
            "max_inflight": self._pd_decode_max_inflight,
            "total_blocks": total_blocks,
            "observed_blocks": observed_blocks,
            "reserved_blocks": reserved_blocks,
            "pending_reserved_blocks": pending_reserved_blocks,
            "capacity_headroom_blocks": capacity_headroom_blocks,
            "fallback_headroom_blocks": fallback_headroom_blocks,
            "dynamic_margin_blocks": dynamic_margin_blocks,
            "capacity_acknowledged_requests": capacity_acknowledged_requests,
            "capacity_generation": int(getattr(worker, "kvbm_capacity_generation", -1)),
            "estimated_blocks": estimated_blocks,
            "full_estimated_blocks": full_estimated_blocks,
            "candidate_reusable_prompt_blocks": candidate_reusable_prompt_blocks,
            "candidate_reuse_hash_count": len(reuse_hashes),
            "confirmed_reusable_prompt_blocks": confirmed_reuse,
            "provisional_reusable_prompt_blocks": provisional_reuse,
            "delta_admission": confirmed_reusable_prompt_blocks is not None,
            "steady_state_blocks": steady_state_blocks,
            "usable_blocks": usable_blocks,
            "projected_blocks": projected_blocks,
            "free_blocks": (authoritative_free_blocks if authoritative_free_blocks is not None else -1),
            "available_blocks": available_blocks,
            "free_blocks_age_ms": (capacity_age_seconds * 1000.0 if math.isfinite(capacity_age_seconds) else -1.0),
            "capacity_source": ("engine_core" if authoritative_free_blocks is not None else "estimated"),
        }

    def _confirm_decode_capacity(
        self,
        worker,
        route_hint: RouteHint,
        previous_diagnostics: dict,
    ) -> tuple[bool, dict]:
        """在最终选中D Worker后使用engine-core实时容量再次准入。

        参数:
            worker: 调度算法选中的D Worker。
            route_hint: 当前请求的Prompt、输出长度和前缀命中信息。
            previous_diagnostics: 使用Router容量快照得到的首次准入诊断。

        返回:
            二元组``(admitted, diagnostics)``。不需要确认时原样返回；
            需要确认时读取最新free_blocks并重新执行准入计算。
        """

        if not self._requires_decode_capacity_confirmation():
            return True, previous_diagnostics
        confirmation_timeout_ms = self._pd_decode_capacity_confirmation_timeout_ms
        if previous_diagnostics.get("capacity_source") == "estimated":
            confirmation_timeout_ms = self._pd_decode_stale_capacity_confirmation_timeout_ms
        reuse_hashes, candidate_reusable_blocks = self._pd_decode_npu_reuse_hint(worker, route_hint)
        confirmation_kwargs = {
            "timeout_ms": confirmation_timeout_ms,
            "request_id": route_hint.request_id,
            "handoff_id": f"pd-{route_hint.request_id}",
            "margin_blocks": max(
                int(previous_diagnostics.get("dynamic_margin_blocks", 0) or 0),
                0,
            ),
            "pending_reserved_blocks": max(
                int(previous_diagnostics.get("pending_reserved_blocks", 0) or 0),
                0,
            ),
        }
        if reuse_hashes:
            block_size = self._worker_block_size(worker.worker_id)
            confirmation_kwargs.update(
                block_hashes=reuse_hashes,
                prompt_blocks=math.ceil(len(route_hint.token_ids or []) / block_size),
                decode_blocks=math.ceil(max(int(route_hint.estimated_output_tokens or 0), 1) / block_size),
            )
        result = self.worker_manager.confirm_kvbm_block_pool_capacity(
            worker.worker_id,
            **confirmation_kwargs,
        )
        confirmation_attempts = 1
        confirmation_total_ms = float(result.get("elapsed_ms", 0.0) or 0.0)
        confirmation_shared = bool(result.get("singleflight_shared"))
        confirmation_initial_error = ""
        small_delta_timeout = bool(
            not result.get("ok")
            and _capacity_query_timed_out(result)
            and candidate_reusable_blocks > 0
            and 0 < int(previous_diagnostics.get("estimated_blocks", 0) or 0) <= _PD_DECODE_SMALL_DELTA_BLOCKS
            and confirmation_timeout_ms < _PD_DECODE_SMALL_DELTA_RETRY_TIMEOUT_MS
        )
        should_retry_shared_stale_failure = bool(
            not result.get("ok")
            and confirmation_shared
            and previous_diagnostics.get("capacity_source") == "estimated"
            and self._pd_decode_stale_capacity_confirmation_timeout_ms
            > self._pd_decode_capacity_confirmation_timeout_ms
        )
        if should_retry_shared_stale_failure or small_delta_timeout:
            confirmation_initial_error = str(result.get("error", ""))
            retry_kwargs = dict(confirmation_kwargs)
            retry_kwargs["timeout_ms"] = (
                self._pd_decode_stale_capacity_confirmation_timeout_ms
                if should_retry_shared_stale_failure
                else _PD_DECODE_SMALL_DELTA_RETRY_TIMEOUT_MS
            )
            result = self.worker_manager.confirm_kvbm_block_pool_capacity(
                worker.worker_id,
                **retry_kwargs,
            )
            confirmation_attempts += 1
            confirmation_total_ms += float(result.get("elapsed_ms", 0.0) or 0.0)
            confirmation_shared = confirmation_shared or bool(result.get("singleflight_shared"))
        control_timing = result.get("control_timing")
        if not isinstance(control_timing, dict):
            control_timing = {}
        logger.info(
            "[PDCapacityQueryTiming] request={} worker={} ok={} error_type={} "
            "failure_stage={} attempts={} initial_timeout_ms={} final_timeout_ms={} "
            "elapsed_ms={:.3f} http_ms={} control_timing={} initial_error={}",
            route_hint.request_id,
            worker.worker_id,
            bool(result.get("ok")),
            result.get("error_type", "none" if result.get("ok") else "unknown"),
            result.get("failure_stage", "none" if result.get("ok") else "unknown"),
            confirmation_attempts,
            confirmation_timeout_ms,
            retry_kwargs["timeout_ms"] if confirmation_attempts > 1 else confirmation_timeout_ms,
            confirmation_total_ms,
            result.get("http_ms", "unknown"),
            control_timing,
            confirmation_initial_error,
        )
        confirmation_metadata = {
            "capacity_confirmation_ms": confirmation_total_ms,
            "capacity_confirmation_final_attempt_ms": float(result.get("elapsed_ms", 0.0) or 0.0),
            "capacity_confirmation_shared": confirmation_shared,
            "capacity_confirmation_attempts": confirmation_attempts,
            "capacity_confirmation_retry_count": confirmation_attempts - 1,
            "capacity_confirmation_timeout_ms": confirmation_timeout_ms,
            "capacity_confirmation_initial_error": confirmation_initial_error,
            "capacity_confirmation_previous_source": str(previous_diagnostics.get("capacity_source", "unknown")),
            "capacity_confirmation_previous_age_ms": float(
                previous_diagnostics.get("free_blocks_age_ms", -1.0) or -1.0
            ),
            "capacity_confirmation_candidate_reusable_blocks": (candidate_reusable_blocks),
            "capacity_confirmation_generation": int(
                result.get("capacity_generation", -1) if result.get("capacity_generation", -1) is not None else -1
            ),
            "admission_ticket_id": str(result.get("admission_ticket_id", "") or ""),
            "admission_ticket_pinned_blocks": int(result.get("admission_ticket_pinned_blocks", 0) or 0),
            "outstanding_ticket_reserved_blocks": int(result.get("outstanding_ticket_reserved_blocks", 0) or 0),
            "router_pending_reserved_blocks": int(result.get("router_pending_reserved_blocks", 0) or 0),
            "committed_reserved_blocks": int(result.get("committed_reserved_blocks", 0) or 0),
            "ticket_capacity_admitted": bool(result.get("ticket_capacity_admitted", False)),
        }
        if not result.get("ok"):
            diagnostics = dict(previous_diagnostics)
            diagnostics.update(
                {
                    "reason": "decode_capacity_confirmation_failed",
                    "capacity_confirmation_error": str(result.get("error", "")),
                    "capacity_confirmation_error_type": str(result.get("error_type", "")),
                    **confirmation_metadata,
                }
            )
            return False, diagnostics
        confirmed_reusable_blocks = result.get("confirmed_reusable_blocks")
        admission_ticket_id = str(result.get("admission_ticket_id", "") or "")
        if confirmed_reusable_blocks is not None and int(confirmed_reusable_blocks) > 0 and not admission_ticket_id:
            # 旧Worker没有前缀ticket能力时不能依赖瞬时命中缩小预算。
            confirmed_reusable_blocks = 0
        admitted, diagnostics = self._decode_admission_decision(
            worker,
            route_hint,
            confirmed_reusable_prompt_blocks=(
                int(confirmed_reusable_blocks) if confirmed_reusable_blocks is not None else None
            ),
        )
        diagnostics["capacity_confirmed"] = True
        diagnostics.update(confirmation_metadata)
        return admitted, diagnostics

    def _requires_decode_capacity_confirmation(self) -> bool:
        """返回当前PD路径是否需要在路由锁外执行实时容量确认。"""
        return bool(
            self._pd_decode_capacity_confirmation_enabled
            and self._pd_decode_block_admission_enabled
            and self._pd_decode_admission_mode == "auto"
        )

    def _commit_pd_worker_pair_locked(
        self,
        prefill_worker,
        decode_worker,
        route_hint: RouteHint,
        prefill_details: dict,
        decode_details: dict,
        excluded_prefix_worker_ids: set[str] | None = None,
    ) -> PendingPrefixMatch | None:
        """原子提交Worker预算；同前缀已有leader时不产生任何副作用。"""
        pending_match = self._claim_pd_pending_prefix(
            route_hint,
            decode_worker,
            excluded_worker_ids=excluded_prefix_worker_ids,
        )
        if pending_match is not None:
            return pending_match
        self.inflight.increment(prefill_worker.worker_id)
        self.inflight.increment(decode_worker.worker_id)
        self.pd_prefill_budget.reserve(
            route_hint.request_id,
            prefill_worker.worker_id,
            int(prefill_details["estimated_tokens"]),
            int(prefill_details["estimated_blocks"]),
        )
        self.pd_decode_blocks.reserve(
            route_hint.request_id,
            decode_worker.worker_id,
            int(decode_details["estimated_blocks"]),
            int(decode_details["steady_state_blocks"]),
            fallback_block_count=int(
                decode_details.get(
                    "full_estimated_blocks",
                    decode_details["estimated_blocks"],
                )
            ),
        )
        ticket_id = str(decode_details.get("admission_ticket_id", "") or "")
        if ticket_id:
            self._pd_decode_admission_tickets[route_hint.request_id] = (
                decode_worker.worker_id,
                ticket_id,
            )
        return None

    def _claim_pd_pending_prefix(
        self,
        route_hint: RouteHint,
        decode_worker,
        excluded_worker_ids: set[str] | None = None,
    ) -> PendingPrefixMatch | None:
        """为冷HIXL路由登记leader；已有同前缀请求时返回该请求。"""
        token_ids = list(route_hint.token_ids or [])
        if len(token_ids) < self._pd_pending_prefix_min_tokens:
            return None
        overlap_scores = route_hint.precomputed_overlap_scores or {}
        overlap_tokens = min(
            max(int(overlap_scores.get(decode_worker.worker_id, 0) or 0), 0),
            len(token_ids),
        )
        if overlap_tokens / len(token_ids) >= self._pd_pending_prefix_min_ratio:
            return None
        registered, existing = self._pd_pending_prefix_affinity.claim(
            request_id=route_hint.request_id,
            model=route_hint.model,
            worker_id=decode_worker.worker_id,
            token_ids=token_ids,
            block_size=self._worker_block_size(decode_worker.worker_id),
            min_tokens=self._pd_pending_prefix_min_tokens,
            min_ratio=self._pd_pending_prefix_min_ratio,
            excluded_worker_ids=excluded_worker_ids,
        )
        if registered:
            logger.info(
                "[PDPendingPrefixAffinity] request={} action=registered worker={} matched_tokens={} ratio={:.3f}",
                route_hint.request_id,
                decode_worker.worker_id,
                overlap_tokens,
                overlap_tokens / len(token_ids),
            )
        return existing

    def take_pd_decode_admission_ticket(
        self,
        request_id: str,
        worker_id: str,
    ) -> str:
        """取出已提交路由对应的D前缀准入ticket。"""

        with self._routing_lock:
            ticket = self._pd_decode_admission_tickets.pop(request_id, None)
        if ticket is None or ticket[0] != worker_id:
            return ""
        return ticket[1]

    def _release_pd_decode_admission_ticket(
        self,
        worker_id: str,
        request_id: str,
        ticket_id: str = "",
        *,
        background: bool = False,
    ) -> None:
        """释放未提交候选的D ticket；锁内调用时异步执行HTTP。"""

        release = self.worker_manager.release_kvbm_decode_admission_ticket
        kwargs = {
            "handoff_id": f"pd-{request_id}",
            "ticket_id": ticket_id,
        }
        if background:
            threading.Thread(
                target=release,
                args=(worker_id,),
                kwargs=kwargs,
                daemon=True,
                name="kvbm-admission-ticket-release",
            ).start()
        else:
            release(worker_id, **kwargs)

    def _needs_decode_capacity_confirmation(
        self,
        admitted: bool,
        diagnostics: dict,
    ) -> bool:
        """判断候选D是否需要使用EngineCore事实源完成最终容量确认。"""
        if not self._requires_decode_capacity_confirmation():
            return False
        if (
            not diagnostics.get("delta_admission")
            and int(diagnostics.get("provisional_reusable_prompt_blocks", 0) or 0) > 0
        ):
            return True
        if diagnostics.get("capacity_source") == "estimated":
            return True
        return bool(
            not admitted
            and diagnostics.get("reason") == "decode_free_block_budget"
            and int(diagnostics.get("candidate_reusable_prompt_blocks", 0) or 0) > 0
        )

    def _is_decode_capacity_confirmation_candidate(
        self,
        admitted: bool,
        diagnostics: dict,
    ) -> bool:
        """判断估算阶段被拒绝的D Worker是否应进入实时容量复核。

        参数:
            admitted: 基于当前Router快照得到的准入结果。
            diagnostics: D Worker准入诊断，包含容量来源和拒绝原因。

        返回:
            已准入或容量快照过期且可实时复核时返回True，否则返回False。

        说明:
            该方法只放宽候选选择，不会直接准入请求。候选仍必须在路由锁外
            查询engine-core真实free_blocks，并在回到锁内后完成原子复核。
        """
        if admitted:
            return True
        return bool(
            self._pd_decode_capacity_confirmation_enabled
            and self._pd_decode_admission_mode == "auto"
            and (
                (
                    diagnostics.get("capacity_source") == "estimated"
                    and diagnostics.get("reason") == "decode_block_budget"
                )
                or (
                    diagnostics.get("capacity_source") == "engine_core"
                    and diagnostics.get("reason") == "decode_free_block_budget"
                    and int(diagnostics.get("candidate_reusable_prompt_blocks", 0) or 0) > 0
                )
            )
        )

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

    def get_worker_admission_snapshot(
        self,
        model: str,
        route_hint: RouteHint | None = None,
        *,
        router_managed_pd: bool = True,
    ) -> dict[str, object]:
        """获取指定模型的 PD Worker 准入快照。

        Args:
            model: 请求使用的模型名称，仅统计承载该模型的 Worker。
            route_hint: 可选的当前请求信息。排队超时诊断应传入，
                使准入数量包含真实Prompt、输出长度和前缀命中成本。
            router_managed_pd: True 使用 HIXL 的 P/D 动态预算；False 仅使用
                原生 Connector 的通用 Worker 准入。

        Returns:
            包含 P、D Worker 的配置数、健康数、准入数及阻塞侧的字典。
            ``blocker`` 为 ``prefill``、``decode``、``both`` 或 ``routing``；
            ``routing`` 表示两侧均有准入 Worker，但仍未选出有效 Worker 对。
        """
        healthy_prefill = self.worker_manager.get_all_healthy_prefill_workers(model) or []
        healthy_decode = self.worker_manager.get_all_healthy_decode_workers(model) or []
        all_workers = self.worker_manager.get_all_workers()
        configured_workers = tuple(all_workers.values()) if isinstance(all_workers, dict) else ()
        configured_prefill = sum(
            worker.model == model and worker.worker_type == WorkerType.PREFILL for worker in configured_workers
        ) or len(healthy_prefill)
        configured_decode = sum(
            worker.model == model and worker.worker_type == WorkerType.DECODE for worker in configured_workers
        ) or len(healthy_decode)
        if router_managed_pd:
            prefill_decisions = {
                worker.worker_id: self._prefill_admission_decision(worker, route_hint) for worker in healthy_prefill
            }
            decode_decisions = {
                worker.worker_id: self._decode_admission_decision(worker, route_hint) for worker in healthy_decode
            }
            cached_rejections = self._get_pd_decode_confirmation_rejections(
                route_hint.request_id if route_hint is not None else "",
                healthy_decode,
            )
            for worker_id, details in cached_rejections.items():
                decode_decisions[worker_id] = (False, details)
            affinity_worker_id = (
                getattr(route_hint, "pd_decode_affinity_worker_id", None) if route_hint is not None else None
            )
            if affinity_worker_id in cached_rejections and not self._pd_decode_affinity_spill_allowed(
                route_hint.request_id,
                affinity_worker_id,
            ):
                for worker in healthy_decode:
                    if worker.worker_id == affinity_worker_id:
                        continue
                    details = dict(decode_decisions[worker.worker_id][1])
                    details["reason"] = "decode_affinity_grace_wait"
                    details["affinity_worker_id"] = affinity_worker_id
                    decode_decisions[worker.worker_id] = (False, details)
        else:
            prefill_decisions = {
                worker.worker_id: self._native_connector_admission_decision(worker) for worker in healthy_prefill
            }
            decode_decisions = {
                worker.worker_id: self._native_decode_admission_decision(worker) for worker in healthy_decode
            }
        admitted_prefill = sum(result[0] for result in prefill_decisions.values())
        admitted_decode = sum(result[0] for result in decode_decisions.values())

        if admitted_prefill == 0 and admitted_decode == 0:
            blocker = "both"
        elif admitted_prefill == 0:
            blocker = "prefill"
        elif admitted_decode == 0:
            blocker = "decode"
        else:
            blocker = "routing"

        return {
            "blocker": blocker,
            "prefill_admitted": admitted_prefill,
            "prefill_healthy": len(healthy_prefill),
            "prefill_configured": configured_prefill,
            "decode_admitted": admitted_decode,
            "decode_healthy": len(healthy_decode),
            "decode_configured": configured_decode,
            "prefill_details": _admission_rejection_details(
                prefill_decisions,
                (
                    "reason",
                    "inflight",
                    "running",
                    "waiting",
                    "estimated_tokens",
                    "reserved_tokens",
                    "projected_tokens",
                    "token_capacity",
                    "estimated_blocks",
                    "reserved_blocks",
                    "projected_blocks",
                    "usable_blocks",
                ),
            ),
            "decode_details": _admission_rejection_details(
                decode_decisions,
                (
                    "reason",
                    "free_blocks",
                    "pending_reserved_blocks",
                    "available_blocks",
                    "estimated_blocks",
                    "dynamic_margin_blocks",
                    "capacity_headroom_blocks",
                    "fallback_headroom_blocks",
                    "inflight",
                    "running",
                    "waiting",
                    "native_pending_dispatch",
                    "effective_waiting",
                    "admission_ratio",
                    "free_blocks_age_ms",
                    "capacity_source",
                ),
            ),
        }

    def _native_connector_admission_decision(self, worker) -> tuple[bool, dict]:
        """Include undispatched native-connector requests in P admission.

        Native connectors such as Mooncake own KV transfer and destination block
        allocation. They must not inherit router-managed HIXL block reservations
        or capacity confirmation state.
        """
        worker_id = worker.worker_id
        inflight = self.inflight.get(worker_id)
        running = max(int(getattr(worker, "running_requests", 0) or 0), 0)
        waiting = max(int(getattr(worker, "queue_depth", 0) or 0), 0)
        pending_dispatch = max(inflight - running - waiting, 0)
        effective_waiting = waiting + pending_dispatch
        admission_ratio = effective_waiting / running if running > 0 else float("inf")
        within_inflight_limit = inflight < self._concurrency_limit
        admitted = within_inflight_limit
        reason = "admitted" if admitted else "native_prefill_inflight_limit"
        if admitted and self._admission_mode == "dynamic" and running > 0:
            admitted = admission_ratio < self._saturation_ratio
            if not admitted:
                reason = "native_prefill_pending_load"
        return admitted, {
            "reason": reason,
            "mode": self._admission_mode,
            "inflight": inflight,
            "running": running,
            "waiting": waiting,
            "native_pending_dispatch": pending_dispatch,
            "effective_waiting": effective_waiting,
            "admission_ratio": admission_ratio,
            "capacity_source": "native_connector",
        }

    def _native_decode_admission_decision(self, worker) -> tuple[bool, dict]:
        """Include bound-but-undispatched Mooncake requests in D admission."""
        worker_id = worker.worker_id
        inflight = self.inflight.get(worker_id)
        running = max(int(getattr(worker, "running_requests", 0) or 0), 0)
        waiting = max(int(getattr(worker, "queue_depth", 0) or 0), 0)
        pending_dispatch = max(inflight - running - waiting, 0)
        effective_waiting = waiting + pending_dispatch
        admission_ratio = effective_waiting / running if running > 0 else float("inf")
        within_inflight_limit = inflight < self._concurrency_limit
        admitted = within_inflight_limit
        reason = "admitted" if admitted else "native_decode_inflight_limit"
        if admitted and self._admission_mode == "dynamic" and running > 0:
            admitted = admission_ratio < self._saturation_ratio
            if not admitted:
                reason = "native_decode_pending_load"
        return admitted, {
            "reason": reason,
            "mode": self._admission_mode,
            "inflight": inflight,
            "running": running,
            "waiting": waiting,
            "native_pending_dispatch": pending_dispatch,
            "effective_waiting": effective_waiting,
            "admission_ratio": admission_ratio,
            "capacity_source": "native_connector",
        }

    def _find_native_connector_worker_pair(
        self,
        model: str,
        route_hint: RouteHint,
    ) -> tuple[str | None, str | None]:
        """Select a P/D pair without router-managed HIXL reservations."""
        prefill_workers = self.worker_manager.get_all_healthy_prefill_workers(model)
        decode_workers = self.worker_manager.get_all_healthy_decode_workers(model)
        if not prefill_workers or not decode_workers:
            return None, None

        try:
            with self._routing_lock:
                prefill_decisions = {
                    worker.worker_id: self._native_connector_admission_decision(worker) for worker in prefill_workers
                }
                decode_decisions = {
                    worker.worker_id: self._native_decode_admission_decision(worker) for worker in decode_workers
                }
                admitted_prefill = [worker for worker in prefill_workers if prefill_decisions[worker.worker_id][0]]
                admitted_decode = [worker for worker in decode_workers if decode_decisions[worker.worker_id][0]]
                if not admitted_prefill or not admitted_decode:
                    return None, None
                prefill_worker, decode_worker = self._select_pd_worker_pair(
                    prefill_workers=admitted_prefill,
                    decode_workers=admitted_decode,
                    route_hint=route_hint,
                )
                self.inflight.increment(prefill_worker.worker_id)
                self.inflight.increment(decode_worker.worker_id)
            logger.info(
                "[NativeConnectorAdmission] request={} selected={}+{} prefill_inflight={} decode_inflight={}",
                route_hint.request_id,
                prefill_worker.worker_id,
                decode_worker.worker_id,
                self.inflight.get(prefill_worker.worker_id),
                self.inflight.get(decode_worker.worker_id),
            )
            return prefill_worker.worker_id, decode_worker.worker_id
        except ValueError as error:
            logger.warning("Native connector worker pair selection failed: {}", error)
            return None, None

    def find_best_worker_pair(
        self,
        model: str,
        route_hint: RouteHint,
        *,
        router_managed_pd: bool = True,
    ) -> tuple[str | None, str | None]:
        """按 PD 后端选择并原子预占 P/D Worker。

        参数:
            model: 当前请求使用的模型名称。
            route_hint: 包含Token、前缀命中和预计输出长度的路由提示。
            router_managed_pd: True 使用 HIXL 容量预算；False 使用原生
                Connector 的通用 inflight 准入。

        返回:
            成功时返回``(prefill_worker_id, decode_worker_id)``；没有满足
            容量约束的Worker对时返回``(None, None)``。

        慢速的engine-core HTTP容量确认在``_routing_lock``外执行。确认后
        重新持锁计算准入并原子更新inflight和block预算，防止并发请求
        重复消费同一份free_blocks快照。
        """
        if not router_managed_pd:
            return self._find_native_connector_worker_pair(model, route_hint)

        prefill_workers = self.worker_manager.get_all_healthy_prefill_workers(model)
        decode_workers = self.worker_manager.get_all_healthy_decode_workers(model)
        all_workers = self.worker_manager.get_all_workers()
        configured_workers = tuple(all_workers.values()) if isinstance(all_workers, dict) else ()
        configured_prefill = sum(
            worker.model == model and worker.worker_type == WorkerType.PREFILL for worker in configured_workers
        ) or len(prefill_workers)
        configured_decode = sum(
            worker.model == model and worker.worker_type == WorkerType.DECODE for worker in configured_workers
        ) or len(decode_workers)

        if not prefill_workers or not decode_workers:
            logger.warning(
                "No valid prefill/decode workers found "
                "(prefill healthy={}/configured={}, decode healthy={}/configured={})",
                len(prefill_workers or []),
                configured_prefill,
                len(decode_workers or []),
                configured_decode,
            )
            return None, None

        total_p, total_d = len(prefill_workers), len(decode_workers)
        t0 = time.perf_counter()
        prefill_worker = None
        decode_worker = None
        prefill_details: dict = {}
        decode_details: dict = {}
        prefill_decisions: dict = {}
        decode_decisions: dict = {}
        admitted_prefill_workers = []
        admitted_decode_workers = []
        excluded_decode_ids: set[str] = set()
        excluded_decode_details: dict[str, dict] = {}
        cached_rejections = self._get_pd_decode_confirmation_rejections(
            route_hint.request_id,
            decode_workers,
        )
        excluded_decode_ids.update(cached_rejections)
        excluded_decode_details.update(cached_rejections)
        deferred_prefix_match: PendingPrefixMatch | None = None
        try:
            while True:
                # 第一阶段只做候选选择，不在路由锁内执行控制面HTTP。
                with self._routing_lock:
                    prefill_decisions = {
                        worker.worker_id: self._prefill_admission_decision(worker, route_hint)
                        for worker in prefill_workers
                    }
                    admitted_prefill_workers = [
                        worker for worker in prefill_workers if prefill_decisions[worker.worker_id][0]
                    ]
                    decode_decisions = {
                        worker.worker_id: self._decode_admission_decision(worker, route_hint)
                        for worker in decode_workers
                    }
                    for worker_id, details in excluded_decode_details.items():
                        decode_decisions[worker_id] = (False, details)
                    admitted_decode_workers = []
                    for selected_worker in decode_workers:
                        if (
                            self._is_decode_capacity_confirmation_candidate(
                                *decode_decisions[selected_worker.worker_id]
                            )
                            and selected_worker.worker_id not in excluded_decode_ids
                        ):
                            admitted_decode_workers.append(selected_worker)
                    admitted_decode_workers, load_excluded_ids = self._queue_relief_pd_decode_workers(
                        admitted_decode_workers,
                        route_hint,
                        comparison_workers=decode_workers,
                    )
                    admitted_decode_workers, hotspot_source_id = self._hotspot_spill_pd_decode_workers(
                        admitted_decode_workers,
                        route_hint,
                    )
                    if hotspot_source_id:
                        load_excluded_ids.add(hotspot_source_id)
                    excluded_prefix_ids = excluded_decode_ids | load_excluded_ids
                    # Keep coalescing unless the leader's D has a substantial
                    # scheduler queue and another capacity candidate offers relief.
                    deferred_prefix_match = self._find_pd_pending_prefix(
                        route_hint,
                        excluded_worker_ids=excluded_prefix_ids,
                    )
                    if deferred_prefix_match is not None:
                        break
                    affinity_worker_id = getattr(
                        route_hint,
                        "pd_decode_affinity_worker_id",
                        None,
                    )
                    if affinity_worker_id:
                        affinity_worker = next(
                            (worker for worker in admitted_decode_workers if worker.worker_id == affinity_worker_id),
                            None,
                        )
                        if affinity_worker is not None:
                            admitted_decode_workers = [affinity_worker]
                        elif (
                            affinity_worker_id in excluded_decode_ids
                            and affinity_worker_id not in load_excluded_ids
                            and not self._pd_decode_affinity_spill_allowed(
                                route_hint.request_id,
                                affinity_worker_id,
                            )
                        ):
                            admitted_decode_workers = []
                    if not admitted_prefill_workers or not admitted_decode_workers:
                        break
                    candidate_prefill, candidate_decode = self._select_pd_worker_pair(
                        prefill_workers=admitted_prefill_workers,
                        decode_workers=admitted_decode_workers,
                        route_hint=route_hint,
                    )

                    # Mooncake/native connector和关闭block admission的路径不需要
                    # 慢速HTTP容量确认。选择与预占必须留在同一锁周期内，避免
                    # 并发请求基于相同inflight快照集中提交到同一Worker。
                    candidate_decode_admitted = decode_decisions[candidate_decode.worker_id][0]
                    candidate_decode_details = decode_decisions[candidate_decode.worker_id][1]
                    if not load_excluded_ids and not self._needs_decode_capacity_confirmation(
                        candidate_decode_admitted,
                        candidate_decode_details,
                    ):
                        prefill_worker = candidate_prefill
                        decode_worker = candidate_decode
                        prefill_details = prefill_decisions[prefill_worker.worker_id][1]
                        decode_details = decode_decisions[decode_worker.worker_id][1]
                        deferred_prefix_match = self._commit_pd_worker_pair_locked(
                            prefill_worker,
                            decode_worker,
                            route_hint,
                            prefill_details,
                            decode_details,
                            excluded_prefix_worker_ids=excluded_prefix_ids,
                        )
                        if deferred_prefix_match is not None:
                            prefill_worker = None
                            decode_worker = None
                        break

                confirmed, candidate_details = self._confirm_decode_capacity(
                    candidate_decode,
                    route_hint,
                    candidate_decode_details,
                )
                if not confirmed:
                    self._release_pd_decode_admission_ticket(
                        candidate_decode.worker_id,
                        route_hint.request_id,
                        str(candidate_details.get("admission_ticket_id", "") or ""),
                    )
                    candidate_details = self._record_pd_decode_confirmation_rejection(
                        route_hint.request_id,
                        candidate_decode,
                        candidate_details,
                        affinity_worker_id=getattr(
                            route_hint,
                            "pd_decode_affinity_worker_id",
                            None,
                        ),
                    )
                    logger.info(
                        "[PDDecodeCapacityConfirmation] request={} worker={} "
                        "admitted=False reason={} free_blocks={} "
                        "available_blocks={} estimated_blocks={} margin_blocks={} "
                        "source={} previous_age_ms={:.3f} timeout_ms={} "
                        "attempts={} retry_count={} elapsed_ms={:.3f} shared={} "
                        "failure_count={} retry_delay_ms={:.0f} wait_class={} "
                        "probe_delay_ms={:.0f} error={} "
                        "initial_error={}",
                        route_hint.request_id,
                        candidate_decode.worker_id,
                        candidate_details.get("reason"),
                        candidate_details.get("free_blocks", -1),
                        candidate_details.get("available_blocks", -1),
                        candidate_details.get("estimated_blocks", 0),
                        candidate_details.get("dynamic_margin_blocks", 0),
                        candidate_details.get(
                            "capacity_confirmation_previous_source",
                            candidate_details.get("capacity_source", "unknown"),
                        ),
                        float(
                            candidate_details.get(
                                "capacity_confirmation_previous_age_ms",
                                candidate_details.get("free_blocks_age_ms", -1.0),
                            )
                            or -1.0
                        ),
                        candidate_details.get("capacity_confirmation_timeout_ms", 0),
                        candidate_details.get("capacity_confirmation_attempts", 1),
                        candidate_details.get("capacity_confirmation_retry_count", 0),
                        float(candidate_details.get("capacity_confirmation_ms", 0.0) or 0.0),
                        bool(candidate_details.get("capacity_confirmation_shared")),
                        candidate_details.get(
                            "capacity_confirmation_failure_count",
                            0,
                        ),
                        float(
                            candidate_details.get(
                                "capacity_confirmation_retry_delay_ms",
                                0.0,
                            )
                            or 0.0
                        ),
                        candidate_details.get(
                            "capacity_confirmation_wait_class",
                            "",
                        ),
                        float(
                            candidate_details.get(
                                "capacity_confirmation_probe_delay_ms",
                                0.0,
                            )
                            or 0.0
                        ),
                        candidate_details.get("capacity_confirmation_error", ""),
                        candidate_details.get("capacity_confirmation_initial_error", ""),
                    )
                    excluded_decode_ids.add(candidate_decode.worker_id)
                    excluded_decode_details[candidate_decode.worker_id] = candidate_details
                    continue

                # 第二阶段在锁内复核最新预算，复核通过后原子提交。
                with self._routing_lock:
                    current_candidates = []
                    for matched_selected_worker in decode_workers:
                        if matched_selected_worker.worker_id not in excluded_decode_ids and (
                            matched_selected_worker.worker_id == candidate_decode.worker_id
                            or self._is_decode_capacity_confirmation_candidate(
                                *self._decode_admission_decision(matched_selected_worker, route_hint)
                            )
                        ):
                            current_candidates.append(matched_selected_worker)
                    _, load_excluded_ids = self._queue_relief_pd_decode_workers(
                        current_candidates,
                        route_hint,
                        comparison_workers=decode_workers,
                    )
                    current_candidates = [
                        worker for worker in current_candidates if worker.worker_id not in load_excluded_ids
                    ]
                    spill_candidates, hotspot_source_id = self._hotspot_spill_pd_decode_workers(
                        current_candidates,
                        route_hint,
                    )
                    if hotspot_source_id:
                        load_excluded_ids.add(hotspot_source_id)
                    excluded_prefix_ids = excluded_decode_ids | load_excluded_ids
                    if candidate_decode.worker_id in load_excluded_ids or candidate_decode not in spill_candidates:
                        self._release_pd_decode_admission_ticket(
                            candidate_decode.worker_id,
                            route_hint.request_id,
                            str(candidate_details.get("admission_ticket_id", "") or ""),
                            background=True,
                        )
                        excluded_decode_ids.add(candidate_decode.worker_id)
                        excluded_decode_details[candidate_decode.worker_id] = {
                            **candidate_details,
                            "reason": "decode_queue_changed",
                        }
                        continue
                    deferred_prefix_match = self._find_pd_pending_prefix(
                        route_hint,
                        excluded_worker_ids=excluded_prefix_ids,
                    )
                    if deferred_prefix_match is not None:
                        self._release_pd_decode_admission_ticket(
                            candidate_decode.worker_id,
                            route_hint.request_id,
                            str(candidate_details.get("admission_ticket_id", "") or ""),
                            background=True,
                        )
                        break
                    prefill_admitted, current_prefill_details = self._prefill_admission_decision(
                        candidate_prefill, route_hint
                    )
                    raw_confirmed_generation = candidate_details.get(
                        "capacity_confirmation_generation",
                        -1,
                    )
                    confirmed_generation = int(raw_confirmed_generation if raw_confirmed_generation is not None else -1)
                    current_generation = int(
                        getattr(
                            candidate_decode,
                            "kvbm_capacity_generation",
                            -1,
                        )
                    )
                    if (
                        confirmed_generation >= 0
                        and current_generation >= 0
                        and current_generation != confirmed_generation
                    ):
                        self._release_pd_decode_admission_ticket(
                            candidate_decode.worker_id,
                            route_hint.request_id,
                            str(candidate_details.get("admission_ticket_id", "") or ""),
                            background=True,
                        )
                        stale_details = dict(candidate_details)
                        stale_details.update(
                            {
                                "reason": "decode_capacity_generation_changed",
                                "capacity_generation": current_generation,
                            }
                        )
                        decode_decisions[candidate_decode.worker_id] = (
                            False,
                            stale_details,
                        )
                        excluded_decode_ids.add(candidate_decode.worker_id)
                        excluded_decode_details[candidate_decode.worker_id] = stale_details
                        continue
                    confirmed_reusable_blocks = candidate_details.get("confirmed_reusable_prompt_blocks")
                    decode_admitted, current_decode_details = self._decode_admission_decision(
                        candidate_decode,
                        route_hint,
                        confirmed_reusable_prompt_blocks=(
                            int(confirmed_reusable_blocks) if confirmed_reusable_blocks is not None else None
                        ),
                    )
                    current_decode_details["capacity_confirmed"] = bool(candidate_details.get("capacity_confirmed"))
                    current_decode_details["capacity_confirmation_ms"] = float(
                        candidate_details.get("capacity_confirmation_ms", 0.0) or 0.0
                    )
                    current_decode_details["capacity_confirmation_shared"] = bool(
                        candidate_details.get("capacity_confirmation_shared")
                    )
                    current_decode_details["admission_ticket_id"] = str(
                        candidate_details.get("admission_ticket_id", "") or ""
                    )
                    current_decode_details["admission_ticket_pinned_blocks"] = int(
                        candidate_details.get("admission_ticket_pinned_blocks", 0) or 0
                    )
                    if not prefill_admitted or not decode_admitted:
                        self._release_pd_decode_admission_ticket(
                            candidate_decode.worker_id,
                            route_hint.request_id,
                            str(candidate_details.get("admission_ticket_id", "") or ""),
                            background=True,
                        )
                        prefill_decisions[candidate_prefill.worker_id] = (
                            prefill_admitted,
                            current_prefill_details,
                        )
                        decode_decisions[candidate_decode.worker_id] = (
                            decode_admitted,
                            current_decode_details,
                        )
                        if not decode_admitted:
                            excluded_decode_ids.add(candidate_decode.worker_id)
                        continue

                    prefill_details = current_prefill_details
                    decode_details = current_decode_details
                    deferred_prefix_match = self._commit_pd_worker_pair_locked(
                        candidate_prefill,
                        candidate_decode,
                        route_hint,
                        prefill_details,
                        decode_details,
                        excluded_prefix_worker_ids=excluded_prefix_ids,
                    )
                    if deferred_prefix_match is not None:
                        self._release_pd_decode_admission_ticket(
                            candidate_decode.worker_id,
                            route_hint.request_id,
                            str(candidate_details.get("admission_ticket_id", "") or ""),
                            background=True,
                        )
                        break
                    if hotspot_source_id:
                        self._pd_hotspot_last_spill_at[hotspot_source_id] = time.monotonic()
                        logger.info(
                            "[PDDecodeHotspotSpill] request={} action=committed "
                            "source={} target={} estimated_blocks={} confirmed=True",
                            route_hint.request_id,
                            hotspot_source_id,
                            candidate_decode.worker_id,
                            decode_details.get("estimated_blocks", 0),
                        )
                    prefill_worker = candidate_prefill
                    decode_worker = candidate_decode
                if decode_details.get("capacity_confirmed"):
                    logger.info(
                        "[PDDecodeCapacityConfirmation] request={} worker={} "
                        "admitted=True free_blocks={} available_blocks={} "
                        "estimated_blocks={} full_estimated_blocks={} "
                        "confirmed_reused_blocks={} margin_blocks={} "
                        "ticket_pinned_blocks={} committed_reserved_blocks={} "
                        "ticket_admitted={} "
                        "elapsed_ms={:.3f} shared={}",
                        route_hint.request_id,
                        decode_worker.worker_id,
                        decode_details.get("free_blocks", -1),
                        decode_details.get("available_blocks", -1),
                        decode_details.get("estimated_blocks", 0),
                        decode_details.get("full_estimated_blocks", 0),
                        decode_details.get("confirmed_reusable_prompt_blocks", 0),
                        decode_details.get("dynamic_margin_blocks", 0),
                        decode_details.get("admission_ticket_pinned_blocks", 0),
                        decode_details.get("committed_reserved_blocks", 0),
                        bool(decode_details.get("ticket_capacity_admitted")),
                        float(decode_details.get("capacity_confirmation_ms", 0.0)),
                        bool(decode_details.get("capacity_confirmation_shared")),
                    )
                break
        except ValueError as exc:
            logger.warning("No valid worker pair found: {}", exc)
            return None, None

        if deferred_prefix_match is not None:
            logger.info(
                "[PDPendingPrefixAffinity] request={} action=deferred "
                "leader={} worker={} state={} matched_tokens={} ratio={:.3f}",
                route_hint.request_id,
                deferred_prefix_match.request_id,
                deferred_prefix_match.worker_id,
                deferred_prefix_match.state,
                deferred_prefix_match.matched_tokens,
                deferred_prefix_match.overlap_ratio,
            )
            return None, None

        if prefill_worker is None or decode_worker is None:
            now = time.monotonic()
            with self._pd_admission_log_lock:
                should_log = now - self._pd_admission_last_warning_at >= 1.0
                if should_log:
                    suppressed_warnings = self._pd_admission_suppressed_warnings
                    self._pd_admission_suppressed_warnings = 0
                    self._pd_admission_last_warning_at = now
                else:
                    self._pd_admission_suppressed_warnings += 1
                    suppressed_warnings = 0
            if should_log:
                logger.warning(
                    "All workers exceed admission limits "
                    "(prefill admitted={}/healthy={}/configured={}, "
                    "decode admitted={}/healthy={}/configured={}, suppressed={}, "
                    "prefill_details={}, decode_details={})",
                    len(admitted_prefill_workers),
                    total_p,
                    configured_prefill,
                    len(admitted_decode_workers),
                    total_d,
                    configured_decode,
                    suppressed_warnings,
                    {
                        worker_id: details
                        for worker_id, (admitted, details) in prefill_decisions.items()
                        if not admitted
                    },
                    {worker_id: details for worker_id, (admitted, details) in decode_decisions.items() if not admitted},
                )
            return None, None

        prefill_decisions[prefill_worker.worker_id] = (True, prefill_details)
        decode_decisions[decode_worker.worker_id] = (True, decode_details)
        self._log_pd_admission_details(
            route_hint.request_id,
            "prefill",
            prefill_workers,
            prefill_decisions,
            prefill_worker.worker_id,
        )
        self._log_pd_admission_details(
            route_hint.request_id,
            "decode",
            decode_workers,
            decode_decisions,
            decode_worker.worker_id,
        )

        if len(admitted_prefill_workers) < total_p or len(admitted_decode_workers) < total_d:
            logger.info(
                "[Admission] prefill {}/{} admitted, decode {}/{} admitted",
                len(admitted_prefill_workers),
                total_p,
                len(admitted_decode_workers),
                total_d,
            )

        algo_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Selected best worker pair: {prefill_worker.worker_id} + {decode_worker.worker_id} "
            f"(algorithm={self._pd_algorithm_label()}, took={algo_ms:.2f}ms, "
            f"prefill_admission={prefill_details}, decode_admission={decode_details})"
        )
        return prefill_worker.worker_id, decode_worker.worker_id

    def find_best_combined_worker(
        self,
        model: str,
        route_hint: RouteHint,
        *,
        warn_if_empty: bool = True,
    ) -> str | None:
        """从全局 COMBINED 池中选择最佳工作器。

        Args:
            model: 请求使用的模型名称。
            route_hint: 当前请求的路由提示信息。
            warn_if_empty: COMBINED 池为空时是否打印告警。

        Returns:
            str | None: 选中的工作器 ID；没有可用工作器时返回 None。
        """
        combined_workers = self.worker_manager.get_all_healthy_combined_workers(model)

        if not combined_workers:
            if warn_if_empty:
                logger.warning("No valid combined workers found")
            return None

        total_before = len(combined_workers)
        combined_workers = [w for w in combined_workers if self._is_worker_admitted(w)]
        if not combined_workers:
            if self._admission_mode == "dynamic":
                logger.warning(
                    "All {} combined workers saturated (waiting/running >= {:.2f})",
                    total_before,
                    self._saturation_ratio,
                )
            else:
                logger.warning(
                    "All {} combined workers exceed concurrency limit ({})",
                    total_before,
                    self._concurrency_limit,
                )
            return None

        try:
            t0 = time.perf_counter()
            with self._routing_lock:
                best_worker = self.current_algorithm.select_worker(combined_workers, route_hint)
                self.inflight.increment(best_worker.worker_id)
            algo_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"Selected best combined worker: {best_worker.worker_id} "
                f"(algorithm={type(self.current_algorithm).__name__}, took={algo_ms:.2f}ms)"
            )
            return best_worker.worker_id
        except ValueError as e:
            logger.warning(f"No valid combined worker found: {e}")
            return None

    def route_to_workers(
        self,
        route_hint: RouteHint,
        session_id: str | None = None,
        *,
        router_managed_pd: bool = True,
    ) -> tuple[str | None, str | None]:
        """路由决策逻辑 - 选择最佳的worker对（prefill + decode）或组合型worker

        session_affinity 和 cache_aware 逻辑由负载均衡算法统一处理。

        Returns:
            (prefill_worker_id, decode_worker_id) - 如果是非组合型
            (combined_worker_id, None) - 如果是组合型
            (None, None) - 如果没有可用worker
        """
        t_start = time.perf_counter()
        logger.info(f"[ROUTER: START] Routing request {route_hint.request_id} to worker pair")

        prefill_id, decode_id = self.find_best_worker_pair(
            route_hint.model,
            route_hint,
            router_managed_pd=router_managed_pd,
        )
        if prefill_id and decode_id:
            if session_id:
                self.kvcache_manager.set_session_affinity(session_id, prefill_id)

            prefill_worker = self.worker_manager.get_worker(prefill_id)
            decode_worker = self.worker_manager.get_worker(decode_id)
            if prefill_worker:
                metrics.record_routed_request(prefill_worker.worker_id, route_hint.model)
            if decode_worker:
                metrics.record_routed_request(decode_worker.worker_id, route_hint.model)

            total_ms = (time.perf_counter() - t_start) * 1000
            self._log_route_decision(
                route_hint=route_hint,
                selected_workers=[prefill_id, decode_id],
                route_reason="prefill_decode_pair",
                total_ms=total_ms,
            )
            logger.info(
                f"[ROUTER: END] Request {route_hint.request_id} routed in {total_ms:.2f}ms "
                f"(inflight: {prefill_id}={self.inflight.get(prefill_id)}, "
                f"{decode_id}={self.inflight.get(decode_id)})"
            )
            return prefill_id, decode_id

        has_disaggregated_workers = self.worker_manager.has_disaggregated_workers(route_hint.model)
        combined_id = self.find_best_combined_worker(
            route_hint.model,
            route_hint,
            warn_if_empty=not has_disaggregated_workers,
        )
        if combined_id:
            if session_id:
                self.kvcache_manager.set_session_affinity(session_id, combined_id)

            combined_worker = self.worker_manager.get_worker(combined_id)
            if combined_worker:
                metrics.record_routed_request(combined_worker.worker_id, route_hint.model)

            total_ms = (time.perf_counter() - t_start) * 1000
            self._log_route_decision(
                route_hint=route_hint,
                selected_workers=[combined_id],
                route_reason="combined_worker",
                total_ms=total_ms,
            )
            logger.info(
                f"[ROUTER: END] Request {route_hint.request_id} routed in {total_ms:.2f}ms "
                f"(inflight: {combined_id}={self.inflight.get(combined_id)})"
            )
            return combined_id, None

        total_ms = (time.perf_counter() - t_start) * 1000
        if has_disaggregated_workers:
            # PD容量不足会由data-plane留在全局队列重试，并在最终超时时输出
            # 包含request_id和准入明细的错误；这里不应为每次重试制造ERROR噪声。
            logger.debug(
                "No PD worker pair currently admitted: request={} model={} took={:.2f}ms",
                route_hint.request_id,
                route_hint.model,
                total_ms,
            )
        else:
            logger.error(
                "No available workers: request={} model={} took={:.2f}ms",
                route_hint.request_id,
                route_hint.model,
                total_ms,
            )
        return None, None
