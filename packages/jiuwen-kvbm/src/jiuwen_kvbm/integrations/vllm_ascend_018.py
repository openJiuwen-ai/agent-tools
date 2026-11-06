from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from jiuwen_kvbm.accessor import VllmAscend018TensorProvider, VllmAscendKVAccessor
from jiuwen_kvbm.benchmark import (
    TcpStagingClient,
    TcpStagingServer,
    run_transfer_benchmark,
    start_tcp_staging_server,
)
from jiuwen_kvbm.integrations.scheduler_kv_bridge import (
    SchedulerKVBridge,
    SchedulerKVOwnershipMode,
)
from jiuwen_kvbm.manager import PrefetchCommitRejectedError
from jiuwen_kvbm.runtime import (
    BlockAlreadyPublishedError,
    BlockHashesInFlightError,
    BlockReservationCapacityError,
    KVBMRuntimeEvent,
    PDHandoffManager,
    PDHandoffState,
    TransferCancelledError,
    TransferCapacityError,
    TransferTransactionError,
    TransferTransactionManager,
    TransferTransactionState,
    VllmBlockPoolReservationManager,
    VllmSourceBlockLeaseManager,
    WorkerKVBMConfig,
    WorkerKVBMRuntime,
)
from jiuwen_kvbm.transfer import (
    LLMDataDistBackend,
    LLMDataDistBlockMapping,
    LLMDataDistCacheDesc,
    LLMDataDistCacheKey,
    LLMDataDistCluster,
    LLMDataDistConfig,
    LLMDataDistEndpoint,
    LLMDataDistRole,
    create_llm_datadist_backend,
)

logger = logging.getLogger("jiuwen.kvbm.vllm_ascend_018")

# Diagnostic-only bounded observations, independent of gate/allocation ownership.
_PD_FALLBACK_OBSERVATIONS: dict[str, dict[str, Any]] = {}


def _pd_prompt_diagnostics_enabled() -> bool:
    return os.getenv("JIUWEN_PD_PROMPT_DIAGNOSTICS", "").lower() in ("1", "true", "yes")


def _pd_token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps([int(t) for t in tokens], ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def _observe_pd_fallback_schedule(scheduler: Any, output: Any, log: logging.Logger) -> None:
    """Observe bounded post-gate lifecycle; leaving Scheduler does not imply success."""
    if not _PD_FALLBACK_OBSERVATIONS:
        return
    now = time.monotonic()
    scheduled = _scheduled_request_ids(output)
    requests = getattr(scheduler, "requests", {})
    with _PD_TARGET_LOCK:
        for request_id, record in list(_PD_FALLBACK_OBSERVATIONS.items()):
            phase = None
            if now - record["started"] > 600:
                phase = "observation_expired"
            elif request_id in scheduled and not record["scheduled"]:
                phase = "first_scheduled_after_gate_release"
                record["scheduled"] = True
            elif request_id not in requests:
                phase = "left_scheduler_not_proof_of_success"
            if phase:
                log.info(
                    "[PDFallbackDiagnostics] handoff_id=%s engine_request_id=%s "
                    "phase=%s elapsed_ms=%.3f scheduled_after_release=%s",
                    record["handoff_id"],
                    request_id,
                    phase,
                    (now - record["started"]) * 1000,
                    record["scheduled"],
                )
            if phase in ("observation_expired", "left_scheduler_not_proof_of_success"):
                _PD_FALLBACK_OBSERVATIONS.pop(request_id, None)


_PATCH_ATTR = "_jiuwen_kvbm_018_hook_installed"
_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_initialize_kv_cache_tensors"
_ENTRYPOINT_PATCH_ATTR = "_jiuwen_kvbm_018_entrypoint_hook_installed"
_WORKER_ENTRYPOINT_PATCH_ATTR = "_jiuwen_kvbm_018_worker_entrypoint_hook_installed"
_KV_EVENT_PUBLISHER_PATCH_ATTR = "_jiuwen_kvbm_018_kv_event_publisher_hook_installed"
_KV_EVENT_PUBLISHER_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_publish"
_BLOCK_POOL_PATCH_ATTR = "_jiuwen_kvbm_018_block_pool_hook_installed"
_BLOCK_POOL_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_insert_block_hash"
_BLOCK_POOL_INIT_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_init"
_BLOCK_POOL_GET_NEW_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_get_new_blocks"
_BLOCK_POOL_FREE_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_free_blocks"
_BLOCK_POOL_CAPACITY_GENERATION_ATTR = "_jiuwen_kvbm_018_capacity_generation"
_BLOCK_POOL_MUTATION_METHODS = ("touch", "reset_prefix_cache")
_KV_CACHE_MANAGER_PATCH_ATTR = "_jiuwen_kvbm_018_kv_cache_manager_hook_installed"
_KV_CACHE_MANAGER_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_take_events"
_KV_CACHE_MANAGER_INIT_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_init"
_KV_CACHE_MANAGER_GET_COMPUTED_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_get_computed_blocks"
_KV_CACHE_MANAGER_FREE_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_free"
_COORDINATOR_CACHE_BLOCKS_PATCH_ATTR = "_jiuwen_kvbm_018_cache_blocks_hook_installed"
_COORDINATOR_CACHE_BLOCKS_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_cache_blocks"
_COORDINATOR_CACHE_MAPPING_PROGRESS_ATTR = "_jiuwen_kvbm_018_cache_mapping_progress"
_SCHEDULER_DIAGNOSTICS_PATCH_ATTR = "_jiuwen_kvbm_018_cache_hit_diagnostics_installed"
_SCHEDULER_DIAGNOSTICS_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_schedule"
_SCHEDULER_PREEMPT_PATCH_ATTR = "_jiuwen_kvbm_018_preempt_hook_installed"
_SCHEDULER_PREEMPT_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_preempt_request"
_SCHEDULER_ADD_REQUEST_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_add_request"
_SCHEDULER_UPDATE_OUTPUT_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_update_from_output"
_RUNTIME_SCHEDULER_PATCH_ATTR = "_jiuwen_kvbm_018_runtime_schedule_hook_installed"
_RUNTIME_SCHEDULER_ORIGINAL_ATTR = "_jiuwen_kvbm_018_runtime_original_schedule"
_NEW_REQUEST_DIAGNOSTICS_PATCH_ATTR = "_jiuwen_kvbm_018_cache_hit_diagnostics_installed"
_NEW_REQUEST_DIAGNOSTICS_ORIGINAL_ATTR = "_jiuwen_kvbm_018_original_from_request"
_PROVIDER_ATTR = "_jiuwen_kvbm_provider"
_DATAPLANE_ATTR = "_jiuwen_kvbm_dataplane"
_RESOLVER_ATTR = "_jiuwen_kvbm_physical_block_resolver"
_DIAGNOSTICS_ATTR = "_jiuwen_kvbm_diagnostics"
_DIAGNOSTICS_ENV = "JIUWEN_KVBM_VLLM_ASCEND_018_DIAGNOSTICS"
_DISK_ROOT_ENV = "JIUWEN_KVBM_WORKER_DISK_ROOT"
_CLEAN_DISK_ON_START_ENV = "JIUWEN_KVBM_CLEAN_DISK_ON_START"
_NPU_HIGH_WATERMARK_ENV = "JIUWEN_KVBM_NPU_HIGH_WATERMARK_BLOCKS"
_RAM_HIGH_WATERMARK_ENV = "JIUWEN_KVBM_RAM_HIGH_WATERMARK_BYTES"
_BLOCK_SIZE_ENV = "JIUWEN_KVBM_BLOCK_SIZE"
_WORKER_ID_ENV = "JIUWEN_KVBM_WORKER_ID"
_DATA_BACKEND_ENV = "JIUWEN_KVBM_DATA_BACKEND"
_LLM_DATADIST_MODE_ENV = "JIUWEN_KVBM_LLM_DATADIST_MODE"
_LLM_DATADIST_ROLE_ENV = "JIUWEN_KVBM_LLM_DATADIST_ROLE"
_LLM_DATADIST_CLUSTER_ID_ENV = "JIUWEN_KVBM_LLM_DATADIST_CLUSTER_ID"
_LLM_DATADIST_LISTEN_IP_ENV = "JIUWEN_KVBM_LLM_DATADIST_LISTEN_IP"
_LLM_DATADIST_LISTEN_PORT_ENV = "JIUWEN_KVBM_LLM_DATADIST_LISTEN_PORT"
_LLM_DATADIST_LISTEN_PORT_BASE_ENV = "JIUWEN_KVBM_LLM_DATADIST_LISTEN_PORT_BASE"
_LLM_DATADIST_TRANSFER_BACKEND_ENV = "JIUWEN_KVBM_LLM_DATADIST_TRANSFER_BACKEND"
_LLM_DATADIST_ENABLE_CACHE_MANAGER_ENV = "JIUWEN_KVBM_LLM_DATADIST_ENABLE_CACHE_MANAGER"
_LLM_DATADIST_ENABLE_REMOTE_CACHE_ACCESSIBLE_ENV = "JIUWEN_KVBM_LLM_DATADIST_ENABLE_REMOTE_CACHE_ACCESSIBLE"
_AUTO_REGISTER_ENV = "JIUWEN_KVBM_AUTO_REGISTER_PHYSICAL_BLOCKS"
_AUTO_ENFORCE_ENV = "JIUWEN_KVBM_AUTO_ENFORCE_WATERMARKS_ON_INIT"
_PREFETCH_HOST_ENV = "JIUWEN_KVBM_ENGINE_CORE_PREFETCH_HOST"
_PREFETCH_PORT_ENV = "JIUWEN_KVBM_ENGINE_CORE_PREFETCH_PORT"
_RESERVATION_PORT_ENV = "JIUWEN_KVBM_ENGINE_CORE_RESERVATION_PORT"
_TRANSFER_MAX_INFLIGHT_ENV = "JIUWEN_KVBM_TRANSFER_MAX_INFLIGHT"
_TRANSFER_RETENTION_SECONDS_ENV = "JIUWEN_KVBM_TRANSFER_RETENTION_SECONDS"
_PD_HANDOFF_MAX_INFLIGHT_ENV = "JIUWEN_KVBM_PD_HANDOFF_MAX_INFLIGHT"
_PD_HANDOFF_RETENTION_SECONDS_ENV = "JIUWEN_KVBM_PD_HANDOFF_RETENTION_SECONDS"
_PD_BLOCK_OWNERSHIP_ENV = "JIUWEN_KVBM_PD_BLOCK_OWNERSHIP"
_BENCHMARK_ENABLED_ENV = "JIUWEN_KVBM_BENCHMARK_ENABLED"
_BENCHMARK_TCP_HOST_ENV = "JIUWEN_KVBM_BENCHMARK_TCP_HOST"
_BENCHMARK_TCP_PORT_ENV = "JIUWEN_KVBM_BENCHMARK_TCP_PORT"
_BENCHMARK_MAX_BLOCKS_ENV = "JIUWEN_KVBM_BENCHMARK_MAX_BLOCKS"
_ENGINE_CORE_DATAPLANE: "VllmAscend018DataPlaneControl | None" = None
_ENGINE_CORE_RESOLVER: "VllmAscend018PhysicalBlockResolver | None" = None
_ENGINE_CORE_LLM_DATADIST_CACHE_ID: int | None = None
_ENGINE_CORE_LLM_DATADIST_LOCK = threading.Lock()


class _KVBMControlHTTPServer(ThreadingHTTPServer):
    """适配高并发PD控制请求的本地HTTP服务。

    默认TCPServer的监听队列很小。PD请求会并发建立事件长连接并提交
    handoff/import控制请求，因此扩大backlog并让请求线程随服务退出。
    """

    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False
    request_queue_size = 256


_ENGINE_CORE_PREFETCH_SERVER: _KVBMControlHTTPServer | None = None
_ENGINE_CORE_PREFETCH_THREAD: threading.Thread | None = None
_ENGINE_CORE_RESERVATION_SERVER: _KVBMControlHTTPServer | None = None
_ENGINE_CORE_RESERVATION_THREAD: threading.Thread | None = None
_ENGINE_CORE_TCP_STAGING_SERVER: TcpStagingServer | None = None
_ENGINE_CORE_TCP_STAGING_THREAD: threading.Thread | None = None
_SOURCE_LEASE_SWEEPER_THREAD: threading.Thread | None = None
_SOURCE_LEASE_SWEEPER_STOP = threading.Event()
_ENGINE_CORE_RUNTIME_EVENT_OUTBOX: list[dict[str, Any]] = []
_ENGINE_CORE_RUNTIME_EVENT_OUTBOX_LOCK = threading.Lock()
_ENGINE_CORE_RUNTIME_EVENT_OUTBOX_LIMIT = 4096
_PENDING_PHYSICAL_BLOCK_MAPPINGS: list[tuple[Any, int]] = []
_OBSERVED_KV_CACHE_MANAGERS: list[Any] = []
_OBSERVED_BLOCK_POOLS: list[Any] = []
_VLLM_BLOCK_POOL_LOCK = threading.RLock()
_PROCESS_BLOCK_RESERVATIONS: VllmBlockPoolReservationManager | None = None
_PROCESS_SOURCE_BLOCK_LEASES: VllmSourceBlockLeaseManager | None = None
_PROCESS_TRANSFER_TRANSACTIONS: TransferTransactionManager | None = None
_PROCESS_PD_HANDOFFS: PDHandoffManager | None = None
_SCHEDULER_KV_BRIDGE: SchedulerKVBridge | None = None
_UNRESOLVED_EVENT_LOG_COUNT = 0
_KV_EVENT_TIMING_SAMPLE_COUNT = 0
_KV_EVENT_TIMING_SAMPLE_EVERY = 64
_CACHE_HIT_DIAGNOSTIC_REQUEST_IDS: set[str] = set()
_CACHE_HIT_DIAGNOSTIC_LOCK = threading.Lock()
_CACHE_HIT_DIAGNOSTIC_LIMIT = 4096
_PD_REQUEST_BLOCK_HASHES: dict[str, tuple[Any, ...]] = {}
_PD_REQUEST_TOKEN_BLOCK_FINGERPRINTS: dict[str, tuple[str, ...]] = {}
_PD_REQUEST_BLOCK_HASHES_LIMIT = 4096
_PD_SOURCE_EXPORT_INTENTS: dict[str, "_PDSourceExportIntent"] = {}
_PD_PREEMPTING_REQUEST_IDS: dict[str, int] = {}
_VLLM_INTERNAL_REQUEST_SUFFIX = re.compile(r"^(?P<parent>.+)-[0-9a-fA-F]{8}$")
_OBSERVED_SCHEDULERS: list[Any] = []
_PD_TARGET_GATES: dict[str, "_PDTargetRequestGate"] = {}
_PD_DECODE_ADMISSION_TICKETS: dict[str, "_PDDecodeAdmissionTicket"] = {}
_PD_TARGET_PLAN_TASKS: list["_PDTargetPlanTask"] = []
_PD_TARGET_ATTACH_TASKS: list["_PDTargetAttachTask"] = []
_PD_FAST_PATH_PENDING: dict[str, dict[str, Any]] = {}
_PD_D_EXECUTION_TIMINGS: dict[str, "_PDTargetExecutionTiming"] = {}
_PD_TARGET_LOCK = threading.RLock()
_PD_SCHEDULER_CYCLE_ATTR = "_jiuwen_kvbm_pd_scheduler_cycle_id"
_PD_SCHEDULER_LAST_CYCLE_AT_ATTR = "_jiuwen_kvbm_pd_scheduler_last_cycle_at"
_PD_SCHEDULER_RECORDED_ATTR = "_jiuwen_kvbm_pd_scheduler_recorded"
_PD_SCHEDULER_SAMPLE_COUNTER_ATTR = "_jiuwen_kvbm_pd_scheduler_sample_counter"
_PD_SCHEDULER_CAPACITY_GENERATION_ATTR = "_jiuwen_kvbm_pd_scheduler_capacity_generation"
_PD_SCHEDULER_IDLE_SAMPLE_EVERY = 128
_PD_FAST_PATH_PENDING_MAX_CYCLES = 8
_ASYNC_METADATA_PUBLISHER: "_AsyncPublishedMetadataPublisher | None" = None
_ASYNC_METADATA_PUBLISHER_LOCK = threading.Lock()
_ASYNC_METADATA_QUEUE_LIMIT = 256
_PD_RECONCILE_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="jiuwen-kvbm-pd-reconcile",
)
_PD_IMPORT_EXECUTOR = ThreadPoolExecutor(
    # import任务多数时间用于等待BlockPool容量或engine-core控制响应，扩大线程池
    # 可避免少量等待者阻塞后续所有请求；真实HIXL传输仍由backend锁控制并行度。
    max_workers=64,
    thread_name_prefix="jiuwen-kvbm-pd-import",
)
_PD_IMPORT_FUTURES: dict[str, Future[dict[str, Any]]] = {}
_PD_IMPORT_FUTURES_LOCK = threading.Lock()
_PD_SCHEDULER_IMPORT_SUBMISSIONS: dict[
    str,
    "_PDSchedulerImportSubmission",
] = {}
_PD_CONTROL_EVENT_HISTORY_LIMIT = 4096


class _PDControlEventBroker:
    """在 Worker 进程内向 Router 长连接广播 PD 状态变化。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[
            str,
            set[queue.Queue[dict[str, Any]]],
        ] = {}
        self._history: list[dict[str, Any]] = []
        self._sequence = 0

    def publish(
        self,
        event_type: str,
        *,
        handoff_id: str,
        request_id: str,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """发布一次可重放的PD控制事件。

        参数:
            event_type: 事件类型，例如REQUEST_SEEN或IMPORT_COMPLETE。
            handoff_id: 当前PD事务ID。
            request_id: 当前engine-core请求ID。
            result: 可选的import结果。

        返回:
            已分配单调序号的完整事件字典。
        """

        with self._lock:
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "event_type": event_type,
                "handoff_id": handoff_id,
                "request_id": request_id,
                "created_at_ms": int(time.time() * 1000),
            }
            if result is not None:
                event["result"] = result
            self._history.append(event)
            if len(self._history) > _PD_CONTROL_EVENT_HISTORY_LIMIT:
                del self._history[:len(self._history) - _PD_CONTROL_EVENT_HISTORY_LIMIT]
            subscribers = tuple(self._subscribers.get(handoff_id, ()))
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                logger.warning(
                    "Jiuwen KVBM PD event subscriber is slow; dropping event: %s",
                    event,
                )
        return event

    def subscribe(
        self,
        *,
        handoff_id: str,
        after_sequence: int = 0,
    ) -> tuple[queue.Queue[dict[str, Any]], list[dict[str, Any]]]:
        """注册事件订阅者并返回历史补发事件。

        参数:
            handoff_id: 只关心的PD事务ID。
            after_sequence: Router已经确认的最后事件序号。

        返回:
            订阅队列与注册前已产生的匹配历史事件。
        """

        subscriber: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=256)
        with self._lock:
            history = []
            for selected_event in self._history:
                if selected_event["handoff_id"] == handoff_id and int(selected_event["sequence"]) > after_sequence:
                    history.append(dict(selected_event))
            self._subscribers.setdefault(handoff_id, set()).add(subscriber)
        return subscriber, history

    def unsubscribe(
        self,
        *,
        handoff_id: str,
        subscriber: queue.Queue[dict[str, Any]],
    ) -> None:
        """注销一个断开的Router事件订阅队列。

        参数:
            handoff_id: 订阅时绑定的PD事务ID。
            subscriber: subscribe返回的事件队列。

        返回:
            无返回值。
        """

        with self._lock:
            subscribers = self._subscribers.get(handoff_id)
            if subscribers is None:
                return
            subscribers.discard(subscriber)
            if not subscribers:
                self._subscribers.pop(handoff_id, None)


_PD_CONTROL_EVENTS = _PDControlEventBroker()


@dataclass
class _PDSourceExportIntent:
    """P scheduler 中一次请求级 KV export 的释放前捕获记录。

    属性:
        handoff_id: Router 创建的 PD handoff ID。
        request_id: Router 传入的 engine request ID，可能对应 vLLM 子请求别名。
        prompt_token_count: 需要交给 D 的 prompt token 数。
        ttl_ms: 捕获后 source lease 的有效期。
        created_at: 注册 export 意图时的单调时钟时间戳。
        source_lease_id: 请求释放前成功 pin 全部源 block 后得到的 lease ID。
        source_blocks: 按逻辑位置、cache group 排列的源物理 block ID。
        block_groups: 与 source_blocks 对齐的完整/partial block 描述。
        block_hashes: 与 source_blocks 对齐；partial 项不包含 packed hash。
        token_block_fingerprints: 完整逻辑 block 的稳定 token 指纹。
        partial_token_fingerprint: 最后一个 partial token span 的稳定指纹。
        semantic_plain_text: 请求是否允许仅依据 token 指纹校验 P/D 等价性。
        scheduler_seen_at: P scheduler首次观察到请求的单调时钟时间戳。
        worker_request_received_to_scheduler_seen_ms: vLLM请求携带可靠到达时间时，
            从Worker收到请求到engine-core add_request的耗时。
        worker_request_received_timing_source: 到达时间使用的请求字段和时钟域。
        scheduler_seen_cycle_id: P scheduler首次观察请求时的调度周期编号。
        scheduler_running_request_count: 首次观察请求后P scheduler的运行请求数。
        scheduler_waiting_request_count: 首次观察请求后P scheduler的等待请求数。
        first_scheduled_at: 请求首次获得scheduler token budget的时间戳。
        first_scheduled_cycle_id: 请求首次获得token budget的调度周期编号。
        scheduler_wait_cycles: 请求进入scheduler后、首次获得budget前跳过的周期数。
        first_scheduled_token_count: 首次调度分配给请求的token数。
        first_scheduled_running_request_count: 首次调度周期的运行请求数。
        first_scheduled_waiting_request_count: 首次调度周期的等待请求数。
        first_scheduled_cycle_ms: 首次调度周期距上个活跃PD周期的墙钟间隔。
        kv_ready_at: P请求block table可供捕获时的单调时钟时间戳。
        kv_ready_cycle_id: block table可供捕获时的调度周期编号。
        capture_ready_at: block捕获完成并可供export读取的单调时钟时间戳。
        capture_complete: P 请求释放 hook 是否已经完成捕获判断。
        capture_ready: 捕获完成或失败时置位，供 Router 长轮询等待 engine-core
            信号，避免等待 P HTTP 响应完成后才开始 export。
        preemption_count: 捕获前经历的 vLLM Scheduler 抢占次数。
        last_preempted_computed_tokens: 最近一次抢占时请求已计算的 token 数。
        error: 捕获失败时的稳定错误文本。
    """

    handoff_id: str
    request_id: str
    prompt_token_count: int
    ttl_ms: int
    created_at: float
    source_lease_id: str = ""
    source_blocks: tuple[int, ...] = ()
    block_groups: tuple[dict[str, Any], ...] = ()
    block_hashes: tuple[dict[str, Any], ...] = ()
    token_block_fingerprints: tuple[str, ...] = ()
    partial_token_fingerprint: str = ""
    semantic_plain_text: bool = False
    scheduler_seen_at: float = 0.0
    worker_request_received_to_scheduler_seen_ms: float = 0.0
    worker_request_received_timing_source: str = "unavailable"
    scheduler_seen_cycle_id: int = 0
    scheduler_running_request_count: int = 0
    scheduler_waiting_request_count: int = 0
    first_scheduled_at: float = 0.0
    first_scheduled_cycle_id: int = 0
    scheduler_wait_cycles: int = 0
    first_scheduled_token_count: int = 0
    first_scheduled_running_request_count: int = 0
    first_scheduled_waiting_request_count: int = 0
    first_scheduled_cycle_ms: float = 0.0
    kv_ready_at: float = 0.0
    kv_ready_cycle_id: int = 0
    capture_ready_at: float = 0.0
    capture_complete: bool = False
    capture_ready: threading.Event = field(default_factory=threading.Event, repr=False)
    preemption_count: int = 0
    last_preempted_computed_tokens: int = 0
    error: str = ""


@dataclass
class _PDTargetExecutionTiming:
    """D请求完成KV attach后的首轮模型执行诊断。

    属性:
        handoff_id: Router创建的PD handoff ID。
        request_id: vLLM engine-core使用的真实请求ID。
        attach_completed_at: Scheduler线程完成KV attach的单调时钟时间戳。
        attach_cycle_id: 完成attach时的KVBM调度周期编号。
        running_request_count: attach时Scheduler中的运行请求数。
        schedule_output_at: Scheduler返回包含本请求的调度结果的时间戳。
        first_scheduled_cycle_id: 请求首次获得token budget的周期编号。
        scheduled_token_count: 首轮分配给请求的token数。
        scheduled_batch_request_count: 同轮获得token budget的请求总数。
        scheduled_batch_token_count: 同轮分配的token总数。
    """

    handoff_id: str
    request_id: str
    attach_completed_at: float
    attach_cycle_id: int
    running_request_count: int
    schedule_output_at: float = 0.0
    first_scheduled_cycle_id: int = 0
    scheduled_token_count: int = 0
    scheduled_batch_request_count: int = 0
    scheduled_batch_token_count: int = 0


@dataclass
class _PDTargetPreplan:
    """D请求进入gate时提前生成的目标BlockPool预留计划。

    属性:
        target_packed_hashes: 按逻辑位置和cache group展开的D本地packed hash；
            request-scoped partial位置使用None。
        block_groups: 根据D请求构造的预期逻辑block/group布局。
        result: ``reserve_delta``返回的JSON安全计划结果。
        created_at: 预计划完成时的单调时钟时间戳。
        hash_prepare_ms: 从D请求提取hash和布局的耗时。
        reserve_ms: BlockPool原子pin和预留耗时。
    """

    target_packed_hashes: tuple[bytes | None, ...]
    block_groups: tuple[dict[str, Any], ...]
    result: dict[str, Any]
    created_at: float
    hash_prepare_ms: float
    reserve_ms: float


@dataclass
class _PDSchedulerTargetAllocation:
    """由 vLLM Scheduler 为 D 请求持有的外部 KV 目标布局。

    属性:
        handoff_id: 当前 PD handoff ID。
        request_id: vLLM engine-core 中的真实请求 ID。
        target_packed_hashes: 与 P/D block group 对齐的 D 端 packed hash。
        block_groups: D 请求自身生成的逻辑 block/group 布局。
        planned_destination_blocks: 请求当前完整物理 block 布局。
        transfer_positions: 需要从 P 端传输的布局位置。
        transfer_destination_blocks: 与 transfer_positions 对齐的目标 block ID。
        local_computed_tokens: vLLM prefix cache 已命中的 token 数。
        external_computed_tokens: Scheduler 为外部 KV 分配的 token 数。
        ready_computed_tokens: HIXL 完成后允许请求直接复用的 token 数。
        full_computed_tokens: 可以发布到通用 prefix cache 的完整 block token 数。
        partial_valid_tokens: request-scoped partial block 中的有效 token 数。
        created_at: Scheduler 完成目标 block 分配的单调时钟时间戳。
        allocation_ms: vLLM 查询本地命中并分配目标 block 的耗时。
        committed: 是否已经把外部 KV 提交给请求。
        released: 失败或取消时是否已经归还请求 block。
    """

    handoff_id: str
    request_id: str
    target_packed_hashes: tuple[bytes | None, ...]
    block_groups: tuple[dict[str, Any], ...]
    planned_destination_blocks: tuple[int, ...]
    transfer_positions: tuple[int, ...]
    transfer_destination_blocks: tuple[int, ...]
    local_computed_tokens: int
    external_computed_tokens: int
    ready_computed_tokens: int
    full_computed_tokens: int
    partial_valid_tokens: int
    created_at: float
    allocation_ms: float
    capacity_generation: int
    committed: bool = False
    released: bool = False


@dataclass
class _PDDecodeAdmissionTicket:
    """路由容量确认到Scheduler分配之间持有D端可复用前缀。"""

    ticket_id: str
    handoff_id: str
    request_id: str
    request_hashes: tuple[str, ...]
    raw_blocks: tuple[Any, ...]
    block_pool: Any
    required_new_blocks: int
    created_at: float
    expires_at: float
    expiry_timer: threading.Timer | None = field(default=None, repr=False)


@dataclass
class _PDTargetRequestGate:
    """D scheduler 中等待 KV import 的请求门控记录。

    属性:
        handoff_id: 当前 PD handoff ID。
        request_id: 需要暂停本地 prefill 的目标请求 ID。
        created_at: 注册门控时的单调时钟时间戳。
        expires_at: 门控自动解除的单调时钟时间戳。
        request_seen: Scheduler.add_request 是否已经观察到该请求。
        engine_request_id: vLLM 实际使用的内部请求 ID，可能带随机子请求后缀。
        request_seen_at: scheduler 观察到请求时的单调时钟时间戳。
        preplan: D请求进入gate时提前建立、尚未被import采用的目标预留。
        preplan_error: 提前预留失败时的诊断文本；同步plan仍可继续兜底。
        preplan_capacity_waiting: 是否已成为容量等待 leader。
        scheduler_allocation: scheduler 模式下由请求原生持有的目标 block 布局。
        last_capacity_retry_generation: 上次尝试Scheduler block分配时的
            BlockPool capacity generation。
    """

    handoff_id: str
    request_id: str
    created_at: float
    expires_at: float
    request_seen: bool = False
    engine_request_id: str = ""
    request_seen_at: float = 0.0
    preplan: _PDTargetPreplan | None = None
    preplan_error: str = ""
    preplan_capacity_waiting: bool = False
    scheduler_allocation: _PDSchedulerTargetAllocation | None = None
    admission_ticket_id: str = ""
    last_capacity_retry_generation: int = -1


@dataclass
class _PDSchedulerImportSubmission:
    """等待vLLM Scheduler分配目标block的异步import提交。

    属性:
        handoff_id: 当前PD handoff ID。
        request_id: Router侧D请求ID。
        control: D Worker数据面控制对象。
        backend: 已初始化的LLM-DataDist backend。
        payload: 后续执行真实HIXL import所需的请求体。
        result_future: 对/result接口暴露的事务Future。
        created_at: 接受提交时的单调时钟时间戳。
        capacity_deadline_at: 等待Scheduler目标block的独立截止时间。
        dispatched: 是否已经在目标block就绪后投递import executor。
    """

    handoff_id: str
    request_id: str
    control: Any
    backend: LLMDataDistBackend
    payload: dict[str, Any]
    result_future: Future[dict[str, Any]]
    created_at: float = field(default_factory=time.monotonic)
    capacity_deadline_at: float = 0.0
    dispatched: bool = False


@dataclass
class _PDTargetAttachTask:
    """由控制线程提交、等待 scheduler 线程执行的 request attach 任务。

    属性:
        request_id: 目标 vLLM 请求 ID。
        handoff_id: 当前 PD handoff ID。
        reservation_id: 已完成 HIXL 写入的目标 reservation ID。
        block_hashes: 与 reservation block 对齐的 packed hash 描述。
        block_groups: P export 返回的逻辑 block/group 布局。
        token_block_fingerprints: P 请求按逻辑 block 计算的稳定 token 指纹。
        partial_token_fingerprint: P 请求最后一个 partial token span 的指纹。
        partial_valid_tokens: partial block 中可读取的 prompt token 数。
        semantic_plain_text: P 请求是否不含 salt、LoRA、多模态和 prompt embeds。
        ownership_mode: D 目标 block 的所有权模式。
        timeout_seconds: 控制线程允许等待 scheduler 提交的时间。
        created_at: attach 任务进入 scheduler mailbox 的单调时钟时间戳。
        completed: scheduler 完成提交或失败时设置的事件。
        result: 成功时的 attach 响应。
        error: 失败时的稳定错误文本。
    """

    request_id: str
    handoff_id: str
    reservation_id: str
    block_hashes: tuple[Any, ...]
    block_groups: tuple[dict[str, Any], ...]
    timeout_seconds: float
    token_block_fingerprints: tuple[str, ...] = ()
    partial_token_fingerprint: str = ""
    partial_valid_tokens: int = 0
    semantic_plain_text: bool = False
    ownership_mode: SchedulerKVOwnershipMode = SchedulerKVOwnershipMode.LEGACY
    created_at: float = field(default_factory=time.monotonic)
    completed: threading.Event = field(default_factory=threading.Event, repr=False)
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class _PDTargetPlanTask:
    """由控制线程提交、等待 scheduler 生成 delta import plan 的任务。

    属性:
        request_id: D scheduler 内的真实请求 ID。
        handoff_id: 当前 PD handoff ID。
        block_groups: P export 返回的逻辑 block/group 布局。
        token_block_fingerprints: P 完整 token blocks 的稳定指纹。
        partial_token_fingerprint: P 尾部 partial token span 指纹。
        partial_valid_tokens: partial block 的有效 token 数。
        semantic_plain_text: 是否允许以 token 指纹确认 P/D KV 等价。
        reservation_ttl_ms: delta plan 的本地 pin 和目标预留 TTL。
        timeout_seconds: 控制线程等待 scheduler 处理的最长时间。
        created_at: plan任务进入scheduler mailbox的单调时钟时间戳。
        completed: scheduler 完成或失败时置位的事件。
        result: 成功时的 delta reservation 快照。
        error: 失败时的稳定错误文本。
        inflight_reservation_ids: 需要等待的同hash leader reservation ID。
        capacity_required_blocks: 容量不足时下次预留所需 block 数。
        ownership_mode: D 目标 block 的所有权模式。
    """

    request_id: str
    handoff_id: str
    block_groups: tuple[dict[str, Any], ...]
    token_block_fingerprints: tuple[str, ...]
    partial_token_fingerprint: str
    partial_valid_tokens: int
    semantic_plain_text: bool
    reservation_ttl_ms: int
    timeout_seconds: float
    created_at: float = field(default_factory=time.monotonic)
    completed: threading.Event = field(default_factory=threading.Event, repr=False)
    result: dict[str, Any] | None = None
    error: str | None = None
    inflight_reservation_ids: tuple[str, ...] = ()
    capacity_required_blocks: int = 0
    ownership_mode: SchedulerKVOwnershipMode = SchedulerKVOwnershipMode.LEGACY


def install_vllm_ascend_018_engine_core_hook(
    *,
    diagnostics: bool = False,
    model_runner_cls: type[Any] | None = None,
    provider_label: str = "0.18",
    log: logging.Logger | None = None,
) -> bool:
    """安装 vLLM-Ascend v0.18.0 engine-core KV cache hook。

    参数:
        diagnostics: 是否在 KV cache 初始化后打印只读诊断信息。
        model_runner_cls: 可选的 model runner 类；为空时自动导入
            vllm_ascend.worker.model_runner_v1.NPUModelRunner。
        log: 可选日志对象；为空时使用模块 logger。

    返回:
        安装成功或已经安装时返回 True；无法导入 vLLM-Ascend 时返回 False。

    说明:
        该 hook 运行在 vLLM engine-core/worker 进程内才有意义。它不主动搬运
        KV cache，只在真实 kv_caches 创建后绑定 VllmAscend018TensorProvider，
        并按需打印 key/value shape、dtype、device、num_blocks 等诊断信息。
    """

    hook_logger = log or logger
    cls = model_runner_cls
    if cls is None:
        try:
            from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

            cls = NPUModelRunner
        except Exception as exc:
            hook_logger.warning(
                "Jiuwen KVBM vLLM-Ascend %s hook skipped: %s",
                provider_label,
                exc,
            )
            return False

    if getattr(cls, _PATCH_ATTR, False):
        hook_logger.info(
            "Jiuwen KVBM vLLM-Ascend %s hook already installed",
            provider_label,
        )
        return True

    original = getattr(cls, "initialize_kv_cache_tensors", None)
    if original is None:
        hook_logger.warning("Jiuwen KVBM hook skipped: initialize_kv_cache_tensors is missing")
        return False

    def _wrapped_initialize_kv_cache_tensors(self: Any, kv_cache_config: Any) -> dict[str, Any]:
        kv_caches = original(self, kv_cache_config)
        _bind_provider_to_model_runner(
            self,
            kv_caches,
            diagnostics=diagnostics,
            log=hook_logger,
        )
        return kv_caches

    _install_model_step_diagnostics(cls, hook_logger)
    setattr(cls, _ORIGINAL_ATTR, original)
    setattr(cls, "initialize_kv_cache_tensors", _wrapped_initialize_kv_cache_tensors)
    setattr(cls, _PATCH_ATTR, True)
    hook_logger.info(
        "Jiuwen KVBM vLLM-Ascend %s engine-core hook installed on %s",
        provider_label,
        cls,
    )
    return True


def _install_model_step_diagnostics(cls: type[Any], log: Any) -> bool:
    """Opt-in bounded host timing; deliberately never synchronize the device."""
    attr = "_jiuwen_kvbm_original_execute_model_timing"
    if hasattr(cls, attr):
        return False
    try:
        every = int(os.environ.get("JIUWEN_KVBM_MODEL_STEP_SAMPLE_EVERY", "0"))
        limit = int(os.environ.get("JIUWEN_KVBM_MODEL_STEP_MAX_SAMPLES", "1000"))
    except ValueError:
        log.warning("Jiuwen KVBM model step diagnostics skipped: invalid sampling settings")
        return False
    original = getattr(cls, "execute_model", None)
    if every <= 0 or limit <= 0 or not callable(original):
        return False

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        count = getattr(self, "_jiuwen_kvbm_model_step_count", 0) + 1
        setattr(self, "_jiuwen_kvbm_model_step_count", count)
        emitted = getattr(self, "_jiuwen_kvbm_model_step_samples", 0)
        if count % every or emitted >= limit:
            return original(self, *args, **kwargs)
        setattr(self, "_jiuwen_kvbm_model_step_samples", emitted + 1)
        output = args[0] if args else kwargs.get("scheduler_output")
        scheduled = getattr(output, "num_scheduled_tokens", {})
        start_unix = time.time()
        start = time.perf_counter()
        ok = False
        try:
            result = original(self, *args, **kwargs)
            ok = True
            return result
        finally:
            duration = (time.perf_counter() - start) * 1000.0
            log.info(
                "[KVBMModelStepTiming] step=%s start_unix_s=%.6f end_unix_s=%.6f "
                "host_execution_ms=%.3f scheduled_requests=%s scheduled_tokens=%s "
                "ok=%s timing_source=host_execute_model_no_device_sync",
                count,
                start_unix,
                time.time(),
                duration,
                len(scheduled) if isinstance(scheduled, dict) else -1,
                sum(scheduled.values()) if isinstance(scheduled, dict) else -1,
                ok,
            )

    setattr(cls, attr, original)
    setattr(cls, "execute_model", wrapped)
    log.info("Jiuwen KVBM model step diagnostics installed every=%s max_samples=%s", every, limit)
    return True


def install_vllm_ascend_018_multiprocess_hooks(
    *,
    diagnostics: bool = False,
    worker_id: str = "",
    disk_root: str = "/tmp/jiuwen-kvbm-worker",
    clean_disk_on_start: bool = True,
    npu_high_watermark_blocks: int = 0,
    ram_high_watermark_bytes: int = 0,
    block_size: int = 16,
    auto_register_physical_blocks: bool = False,
    auto_enforce_watermarks_on_init: bool = False,
    prefetch_host: str = "127.0.0.1",
    prefetch_port: int = 0,
    reservation_port: int = 0,
    transfer_max_inflight: int = 1,
    transfer_retention_seconds: float = 300.0,
    benchmark_enabled: bool = False,
    benchmark_tcp_host: str = "127.0.0.1",
    benchmark_tcp_port: int = 0,
    benchmark_max_blocks: int = 64,
    data_backend: str = "local",
    llm_datadist_mode: str = "python",
    llm_datadist_role: str = "mix",
    llm_datadist_cluster_id: int = 0,
    llm_datadist_listen_ip: str = "127.0.0.1",
    llm_datadist_listen_port: int = 0,
    llm_datadist_listen_port_base: int = 19000,
    llm_datadist_transfer_backend: str = "hixl",
    llm_datadist_enable_cache_manager: bool = True,
    llm_datadist_enable_remote_cache_accessible: bool = False,
    pd_block_ownership: str = "scheduler",
    log: logging.Logger | None = None,
) -> bool:
    """安装 vLLM-Ascend v0.18.0 多进程入口 hook。

    参数:
        diagnostics: 是否让子进程中的 KV cache hook 打印诊断信息。
        worker_id: worker 唯一标识，用于 runtime 事件和统计。
        disk_root: worker 子进程中磁盘卸载目录。
        clean_disk_on_start: 是否在 runtime 启动时清理旧磁盘 payload。
        npu_high_watermark_blocks: NPU block 水位，自动执行水位策略时使用。
        ram_high_watermark_bytes: RAM 字节水位，自动执行水位策略时使用。
        block_size: 默认 block token 数。
        auto_register_physical_blocks: 是否在 KV cache 初始化后注册全部物理 block。
        auto_enforce_watermarks_on_init: 是否在初始化后立即执行一次水位策略。
        prefetch_host: engine-core 预取控制服务监听地址。
        prefetch_port: engine-core 预取控制服务监听端口；小于等于 0 时不启动。
        reservation_port: scheduler 进程 BlockPool reservation 内部端口；小于等于
            0 时按 prefetch_port + 10000 推导。
        transfer_max_inflight: 单个 NPU worker 同时执行的跨 Worker 传输上限。
        transfer_retention_seconds: 终态事务保留时长，供超时调用方查询。
        benchmark_enabled: 是否启用隔离的 HIXL/TCP 数据面基准入口。
        benchmark_tcp_host: TCP staging 源端监听地址。
        benchmark_tcp_port: TCP staging 源端监听端口；小于等于 0 时不启动。
        benchmark_max_blocks: 单次基准请求允许搬运的最大物理 block 数。
        data_backend: worker 数据面 backend 名称，local 或 llm_datadist。
        llm_datadist_mode: LLM-DataDist backend 模式，fake 或 python。
        llm_datadist_role: LLM-DataDist 角色，prompt、decoder 或 mix。
        llm_datadist_cluster_id: LLM-DataDist cluster ID。
        llm_datadist_listen_ip: LLM-DataDist 监听 IP。
        llm_datadist_listen_port: LLM-DataDist 监听端口；小于等于 0 时按
            listen_port_base + device_id 推导。
        llm_datadist_listen_port_base: listen_port 未配置时使用的端口基准。
        llm_datadist_transfer_backend: LLM-DataDist 底层传输后端。
        llm_datadist_enable_cache_manager: 是否启用 DataDist cache manager。
        llm_datadist_enable_remote_cache_accessible: 是否启用远端 cache 可访问。
        pd_block_ownership: PD目标block所有权模式。legacy继续使用KVBM额外
            reservation；scheduler使用vLLM Scheduler为D请求原生分配block。
        log: 可选日志对象；为空时使用模块 logger。

    返回:
        安装成功返回 True；无法导入 vLLM/vLLM-Ascend 入口时返回 False。

    说明:
        vLLM v1 常用 spawn 启动 engine-core/worker 子进程，父进程中直接
        monkey patch NPUModelRunner 不一定会被子进程继承。本函数把
        EngineCoreProc.run_engine_core 替换为本模块顶层函数，使 spawn 子进程
        import 本模块后再安装真正的 KV cache hook。
    """

    hook_logger = log or logger
    ownership_mode = SchedulerKVOwnershipMode(str(pd_block_ownership).strip().lower())
    os.environ[_DIAGNOSTICS_ENV] = "1" if diagnostics else "0"
    os.environ[_WORKER_ID_ENV] = worker_id
    os.environ[_DISK_ROOT_ENV] = disk_root
    os.environ[_CLEAN_DISK_ON_START_ENV] = "1" if clean_disk_on_start else "0"
    os.environ[_NPU_HIGH_WATERMARK_ENV] = str(npu_high_watermark_blocks)
    os.environ[_RAM_HIGH_WATERMARK_ENV] = str(ram_high_watermark_bytes)
    os.environ[_BLOCK_SIZE_ENV] = str(block_size)
    os.environ[_AUTO_REGISTER_ENV] = "1" if auto_register_physical_blocks else "0"
    os.environ[_AUTO_ENFORCE_ENV] = "1" if auto_enforce_watermarks_on_init else "0"
    os.environ[_PREFETCH_HOST_ENV] = prefetch_host
    os.environ[_PREFETCH_PORT_ENV] = str(prefetch_port)
    resolved_reservation_port = reservation_port
    if resolved_reservation_port <= 0 and prefetch_port > 0:
        resolved_reservation_port = prefetch_port + 10000
    os.environ[_RESERVATION_PORT_ENV] = str(resolved_reservation_port)
    os.environ[_TRANSFER_MAX_INFLIGHT_ENV] = str(max(1, transfer_max_inflight))
    os.environ[_TRANSFER_RETENTION_SECONDS_ENV] = str(max(1.0, transfer_retention_seconds))
    os.environ[_BENCHMARK_ENABLED_ENV] = "1" if benchmark_enabled else "0"
    os.environ[_BENCHMARK_TCP_HOST_ENV] = benchmark_tcp_host
    os.environ[_BENCHMARK_TCP_PORT_ENV] = str(benchmark_tcp_port)
    os.environ[_BENCHMARK_MAX_BLOCKS_ENV] = str(max(1, benchmark_max_blocks))
    os.environ[_DATA_BACKEND_ENV] = data_backend
    os.environ[_LLM_DATADIST_MODE_ENV] = llm_datadist_mode
    os.environ[_LLM_DATADIST_ROLE_ENV] = llm_datadist_role
    os.environ[_LLM_DATADIST_CLUSTER_ID_ENV] = str(llm_datadist_cluster_id)
    os.environ[_LLM_DATADIST_LISTEN_IP_ENV] = llm_datadist_listen_ip
    os.environ[_LLM_DATADIST_LISTEN_PORT_ENV] = str(llm_datadist_listen_port)
    os.environ[_LLM_DATADIST_LISTEN_PORT_BASE_ENV] = str(llm_datadist_listen_port_base)
    os.environ[_LLM_DATADIST_TRANSFER_BACKEND_ENV] = llm_datadist_transfer_backend
    os.environ[_LLM_DATADIST_ENABLE_CACHE_MANAGER_ENV] = "1" if llm_datadist_enable_cache_manager else "0"
    os.environ[_LLM_DATADIST_ENABLE_REMOTE_CACHE_ACCESSIBLE_ENV] = (
        "1" if llm_datadist_enable_remote_cache_accessible else "0"
    )
    os.environ[_PD_BLOCK_OWNERSHIP_ENV] = ownership_mode.value
    try:
        import vllm.v1.engine.core as core_module
        from vllm.v1.engine.core import EngineCoreProc
    except Exception as exc:
        hook_logger.warning("Jiuwen KVBM engine-core entrypoint hook skipped: %s", exc)
        return False

    if getattr(EngineCoreProc, _ENTRYPOINT_PATCH_ATTR, False):
        hook_logger.info("Jiuwen KVBM vLLM engine-core entrypoint hook already installed")
        return True

    EngineCoreProc.run_engine_core = staticmethod(jiuwen_kvbm_vllm_ascend_018_run_engine_core)
    setattr(EngineCoreProc, _ENTRYPOINT_PATCH_ATTR, True)
    core_module.EngineCoreProc.run_engine_core = EngineCoreProc.run_engine_core
    hook_logger.info("Jiuwen KVBM vLLM engine-core entrypoint hook installed")
    return True


def _pd_block_ownership_mode() -> SchedulerKVOwnershipMode:
    """读取当前进程使用的PD目标block所有权模式。

    返回:
        环境变量配置对应的SchedulerKVOwnershipMode；未配置时返回scheduler。

    异常:
        环境变量不是legacy或scheduler时抛出ValueError。
    """

    return SchedulerKVOwnershipMode(os.getenv(_PD_BLOCK_OWNERSHIP_ENV, "scheduler").strip().lower())


def jiuwen_kvbm_vllm_ascend_018_run_engine_core(*args: Any, **kwargs: Any) -> Any:
    """vLLM engine-core 子进程入口包装函数。

    参数:
        args: 透传给 vLLM-Ascend 原 run_engine_core 的位置参数。
        kwargs: 透传给 vLLM-Ascend 原 run_engine_core 的关键字参数。

    返回:
        vLLM-Ascend 原 run_engine_core 的返回值。
    """

    _install_child_process_hooks()
    from vllm_ascend.patch.platform.patch_balance_schedule import run_engine_core

    return run_engine_core(*args, **kwargs)


def jiuwen_kvbm_vllm_ascend_018_worker_main(*args: Any, **kwargs: Any) -> Any:
    """vLLM-Ascend worker 子进程入口包装函数。

    参数:
        args: 透传给 vLLM WorkerProc.worker_main 的位置参数。
        kwargs: 透传给 vLLM WorkerProc.worker_main 的关键字参数。

    返回:
        vLLM WorkerProc.worker_main 的返回值。
    """

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    install_vllm_ascend_018_engine_core_hook(diagnostics=_diagnostics_enabled_from_env())
    install_vllm_ascend_018_kv_event_publisher_hook()
    install_vllm_ascend_018_block_pool_hook()
    install_vllm_ascend_018_kv_cache_manager_hook()
    from vllm.v1.executor.multiproc_executor import WorkerProc

    return WorkerProc.worker_main(*args, **kwargs)


def _install_child_process_hooks() -> None:
    """在 engine-core 子进程内安装 worker 入口和 KV cache hook。

    返回:
        无返回值。
    """

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    _install_worker_entrypoint_hook()
    install_vllm_ascend_018_engine_core_hook(diagnostics=_diagnostics_enabled_from_env())
    install_vllm_ascend_018_kv_event_publisher_hook()
    install_vllm_ascend_018_block_pool_hook()
    install_vllm_ascend_018_kv_cache_manager_hook()
    install_vllm_ascend_018_scheduler_diagnostics_hook()


def _install_worker_entrypoint_hook() -> bool:
    """在 engine-core 子进程内安装 vLLM-Ascend worker 入口 hook。

    返回:
        安装成功或已经安装时返回 True；无法导入入口类时返回 False。
    """

    try:
        from vllm_ascend.patch.platform.patch_multiproc_executor import AscendWorkerProc
    except Exception as exc:
        logger.warning("Jiuwen KVBM worker entrypoint hook skipped: %s", exc)
        return False

    if getattr(AscendWorkerProc, _WORKER_ENTRYPOINT_PATCH_ATTR, False):
        return True
    AscendWorkerProc.worker_main = staticmethod(jiuwen_kvbm_vllm_ascend_018_worker_main)
    setattr(AscendWorkerProc, _WORKER_ENTRYPOINT_PATCH_ATTR, True)
    logger.info("Jiuwen KVBM vLLM-Ascend worker entrypoint hook installed")
    return True


def install_vllm_ascend_018_kv_event_publisher_hook(
    *,
    publisher_cls: type[Any] | None = None,
    log: logging.Logger | None = None,
) -> bool:
    """安装 engine-core 内 vLLM KV event publisher hook。

    参数:
        publisher_cls: 可选 publisher 类；为空时自动导入
            vllm.distributed.kv_events.ZmqEventPublisher。
        log: 可选日志对象；为空时使用模块 logger。

    返回:
        安装成功或已经安装时返回 True；无法导入 publisher 时返回 False。

    说明:
        vLLM 的公开 KV 事件通常只包含 block_hash/token_ids，不包含物理
        block_id。本 hook 在事件发布给 router 前，把同一批事件先交给
        engine-core 中的 KVBM 数据面，并尝试通过物理 block resolver 补齐
        block_id 后执行真实 RAM/Disk 数据面逻辑。
    """

    hook_logger = log or logger
    cls = publisher_cls
    if cls is None:
        try:
            from vllm.distributed.kv_events import ZmqEventPublisher

            cls = ZmqEventPublisher
        except Exception as exc:
            hook_logger.warning("Jiuwen KVBM KV event publisher hook skipped: %s", exc)
            return False

    if getattr(cls, _KV_EVENT_PUBLISHER_PATCH_ATTR, False):
        hook_logger.info("Jiuwen KVBM KV event publisher hook already installed")
        return True
    original = getattr(cls, "publish", None)
    if original is None:
        hook_logger.warning("Jiuwen KVBM KV event publisher hook skipped: publish is missing")
        return False

    def _wrapped_publish(self: Any, events: Any) -> Any:
        _apply_engine_core_kv_events(events)
        return original(self, events)

    setattr(cls, _KV_EVENT_PUBLISHER_ORIGINAL_ATTR, original)
    setattr(cls, "publish", _wrapped_publish)
    setattr(cls, _KV_EVENT_PUBLISHER_PATCH_ATTR, True)
    hook_logger.info("Jiuwen KVBM KV event publisher hook installed on %s", cls)
    return True


def install_vllm_ascend_018_block_pool_hook(
    *,
    block_pool_cls: type[Any] | None = None,
    log: logging.Logger | None = None,
) -> bool:
    """安装 vLLM BlockPool hash 插入 hook。

    参数:
        block_pool_cls: 可选 BlockPool 类；为空时自动导入
            vllm.v1.core.block_pool.BlockPool。
        log: 可选日志对象；为空时使用模块 logger。

    返回:
        安装成功或已经安装时返回 True；无法导入 BlockPool 时返回 False。

    说明:
        vLLM 在 BlockPool._insert_block_hash() 中拥有最准确的
        block_hash_with_group_id -> KVCacheBlock.block_id 映射。本 hook 在该点
        记录物理 block id，避免通过 model_runner 对象树盲扫。
    """

    hook_logger = log or logger
    cls = block_pool_cls
    if cls is None:
        try:
            from vllm.v1.core.block_pool import BlockPool

            cls = BlockPool
        except Exception as exc:
            hook_logger.warning("Jiuwen KVBM BlockPool hook skipped: %s", exc)
            return False

    if getattr(cls, _BLOCK_POOL_PATCH_ATTR, False):
        hook_logger.info("Jiuwen KVBM BlockPool hook already installed")
        return True
    original = getattr(cls, "_insert_block_hash", None)
    original_init = getattr(cls, "__init__")
    original_get_new = getattr(cls, "get_new_blocks", None)
    original_free = getattr(cls, "free_blocks", None)
    original_cache_full_blocks = getattr(cls, "cache_full_blocks", None)
    if original is None and original_cache_full_blocks is None:
        hook_logger.warning(
            "Jiuwen KVBM BlockPool hook skipped: both _insert_block_hash and cache_full_blocks are missing"
        )
        return False

    def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        setattr(self, _BLOCK_POOL_CAPACITY_GENERATION_ATTR, time.monotonic_ns())
        _record_block_pool(self)

    def _bump_capacity_generation(block_pool: Any) -> int:
        previous = int(getattr(block_pool, _BLOCK_POOL_CAPACITY_GENERATION_ATTR, -1))
        generation = previous + 1 if previous >= 0 else time.monotonic_ns()
        setattr(block_pool, _BLOCK_POOL_CAPACITY_GENERATION_ATTR, generation)
        return generation

    def _wrapped_insert_block_hash(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not (original is not None):
            raise RuntimeError("Required invariant failed: original is not None")
        with _VLLM_BLOCK_POOL_LOCK:
            _record_block_pool(self)
            result = original(self, *args, **kwargs)
            block_hash = args[0] if args else kwargs.get("block_hash_with_group_id")
            block = args[1] if len(args) > 1 else kwargs.get("block")
            block_id = getattr(block, "block_id", None)
            _record_engine_core_physical_mapping(block_hash, block_id)
            return result

    def _wrapped_get_new_blocks(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not (original_get_new is not None):
            raise RuntimeError("Required invariant failed: original_get_new is not None")
        with _VLLM_BLOCK_POOL_LOCK:
            _record_block_pool(self)
            result = original_get_new(self, *args, **kwargs)
            _bump_capacity_generation(self)
            return result

    def _wrapped_free_blocks(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not (original_free is not None):
            raise RuntimeError("Required invariant failed: original_free is not None")
        with _VLLM_BLOCK_POOL_LOCK:
            _record_block_pool(self)
            result = original_free(self, *args, **kwargs)
            _bump_capacity_generation(self)
            if _PROCESS_BLOCK_RESERVATIONS is not None:
                _PROCESS_BLOCK_RESERVATIONS.notify_capacity_changed()
            return result

    def _wrapped_cache_full_blocks(self: Any, *args: Any, **kwargs: Any) -> Any:
        """在 vLLM 完成 full block 缓存后增量记录物理映射。"""

        if not (original_cache_full_blocks is not None):
            raise RuntimeError("Required invariant failed: original_cache_full_blocks is not None")
        with _VLLM_BLOCK_POOL_LOCK:
            _record_block_pool(self)
            result = original_cache_full_blocks(self, *args, **kwargs)
            recorded = _record_cache_full_block_mappings(args, kwargs)
            resolver = _ENGINE_CORE_RESOLVER
            if resolver is not None and recorded:
                resolver.note_cache_full_mappings(recorded)
            return result

    def _locked_block_pool_method(method: Callable[..., Any]) -> Callable[..., Any]:
        """构造使用 KVBM/vLLM 共用锁的 BlockPool 变更方法 wrapper。"""

        def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            with _VLLM_BLOCK_POOL_LOCK:
                _record_block_pool(self)
                result = method(self, *args, **kwargs)
                _bump_capacity_generation(self)
                return result

        return _wrapped

    setattr(cls, _BLOCK_POOL_INIT_ORIGINAL_ATTR, original_init)
    setattr(cls, "__init__", _wrapped_init)
    if original is not None:
        setattr(cls, _BLOCK_POOL_ORIGINAL_ATTR, original)
        setattr(cls, "_insert_block_hash", _wrapped_insert_block_hash)
    if original_get_new is not None:
        setattr(cls, _BLOCK_POOL_GET_NEW_ORIGINAL_ATTR, original_get_new)
        setattr(cls, "get_new_blocks", _wrapped_get_new_blocks)
    if original_free is not None:
        setattr(cls, _BLOCK_POOL_FREE_ORIGINAL_ATTR, original_free)
        setattr(cls, "free_blocks", _wrapped_free_blocks)
    if original_cache_full_blocks is not None:
        setattr(
            cls,
            "_jiuwen_kvbm_018_original_cache_full_blocks",
            original_cache_full_blocks,
        )
        setattr(cls, "cache_full_blocks", _wrapped_cache_full_blocks)
    for method_name in _BLOCK_POOL_MUTATION_METHODS:
        method = getattr(cls, method_name, None)
        if method is None:
            continue
        setattr(cls, f"_jiuwen_kvbm_018_original_{method_name}", method)
        setattr(cls, method_name, _locked_block_pool_method(method))
    setattr(cls, _BLOCK_POOL_PATCH_ATTR, True)
    hook_logger.info("Jiuwen KVBM BlockPool hook installed on %s", cls)
    return True


def _record_cache_full_block_mappings(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> int:
    """从 cache_full_blocks 实际参数中记录 hash 到物理 block。

    参数:
        args: cache_full_blocks 的位置参数，不包含 self。
        kwargs: cache_full_blocks 的关键字参数。

    返回:
        本次成功记录的 ``block_hash -> block_id`` 数量。

    说明:
        不依赖 vLLM/vLLM-Ascend 的具体函数签名。方法返回后，
        被缓存的 KVCacheBlock 已持有最终 block_hash，因此可以
        直接建立增量映射，避免事件到达后扫描整个 BlockPool。
    """

    recorded = 0
    visited: set[int] = set()
    pending = [*args, *kwargs.values()]
    while pending:
        value = pending.pop()
        value_id = id(value)
        if value_id in visited:
            continue
        visited.add(value_id)
        if isinstance(value, dict):
            pending.extend(value.values())
            continue
        if isinstance(value, list | tuple | set):
            pending.extend(value)
            continue
        try:
            block_id = getattr(value, "block_id", None)
        except Exception:
            logger.debug("Cannot inspect cache object block_id", exc_info=True)
            continue
        if block_id is None:
            continue
        try:
            block_hash = getattr(value, "block_hash", None)
        except Exception:
            block_hash = None
        if block_hash is None:
            try:
                block_hash = getattr(value, "_block_hash", None)
            except Exception:
                block_hash = None
        if block_hash is None:
            continue
        _record_engine_core_physical_mapping(block_hash, block_id)
        recorded += 1
    return recorded


def install_vllm_ascend_018_kv_cache_manager_hook(
    *,
    kv_cache_manager_cls: type[Any] | None = None,
    log: logging.Logger | None = None,
) -> bool:
    """安装 vLLM KVCacheManager 事件读取 hook。

    参数:
        kv_cache_manager_cls: 可选 KVCacheManager 类；为空时自动导入
            vllm.v1.core.kv_cache_manager.KVCacheManager。
        log: 可选日志对象；为空时使用模块 logger。

    返回:
        安装成功或已经安装时返回 True；无法导入 KVCacheManager 时返回 False。

    说明:
        vLLM 的 KV event 由 KVCacheManager.take_events() 向外暴露。该 hook 在
        take_events 后从同一个 manager 的 block_pool 快照同步
        block_hash -> physical block id 映射，覆盖 Ascend patch 绕过
        BlockPool._insert_block_hash 的情况。
    """

    hook_logger = log or logger
    cls = kv_cache_manager_cls
    if cls is None:
        try:
            from vllm.v1.core.kv_cache_manager import KVCacheManager

            cls = KVCacheManager
        except Exception as exc:
            hook_logger.warning("Jiuwen KVBM KVCacheManager hook skipped: %s", exc)
            return False

    if getattr(cls, _KV_CACHE_MANAGER_PATCH_ATTR, False):
        hook_logger.info("Jiuwen KVBM KVCacheManager hook already installed")
        return True
    original = getattr(cls, "take_events", None)
    if original is None:
        hook_logger.warning("Jiuwen KVBM KVCacheManager hook skipped: take_events is missing")
        return False

    original_init = getattr(cls, "__init__")
    original_get_computed = getattr(cls, "get_computed_blocks", None)
    original_free = getattr(cls, "free", None)

    def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _record_kv_cache_manager(self)
        _install_coordinator_cache_blocks_hook(self, hook_logger)
        _start_engine_core_reservation_server(hook_logger)

    def _wrapped_take_events(self: Any, *args: Any, **kwargs: Any) -> Any:
        events = original(self, *args, **kwargs)
        _record_kv_cache_manager(self)
        resolver = _ENGINE_CORE_RESOLVER
        if resolver is not None:
            resolver.reset_snapshot_timing()
        _record_from_events_with_manager(events, self)
        return events

    def _wrapped_get_computed_blocks(
        self: Any,
        request: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not (original_get_computed is not None):
            raise RuntimeError("Required invariant failed: original_get_computed is not None")
        request_hashes = _flatten_request_block_hashes(getattr(request, "block_hashes", None))
        _record_pd_request_block_hashes(
            request,
            block_size=_diagnostic_block_size(
                self,
                prompt_token_count=0,
                request_hash_count=len(request_hashes),
            ),
        )
        result = original_get_computed(self, request, *args, **kwargs)
        if _diagnostics_enabled_from_env():
            _log_prefix_cache_lookup_diagnostics(self, request, result, hook_logger)
        return result

    def _wrapped_free(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        """在 vLLM 释放 P 请求 block table 前捕获请求级 partial KV。"""

        if not (original_free is not None):
            raise RuntimeError("Required invariant failed: original_free is not None")
        with _VLLM_BLOCK_POOL_LOCK:
            if _defer_pd_source_capture_for_preemption(request, hook_logger):
                return original_free(self, request, *args, **kwargs)
            _capture_pd_source_request_blocks(self, request, hook_logger)
            return original_free(self, request, *args, **kwargs)

    setattr(cls, _KV_CACHE_MANAGER_INIT_ORIGINAL_ATTR, original_init)
    setattr(cls, _KV_CACHE_MANAGER_ORIGINAL_ATTR, original)
    setattr(cls, "__init__", _wrapped_init)
    setattr(cls, "take_events", _wrapped_take_events)
    if original_get_computed is not None:
        setattr(
            cls,
            _KV_CACHE_MANAGER_GET_COMPUTED_ORIGINAL_ATTR,
            original_get_computed,
        )
        setattr(cls, "get_computed_blocks", _wrapped_get_computed_blocks)
    if original_free is not None:
        setattr(cls, _KV_CACHE_MANAGER_FREE_ORIGINAL_ATTR, original_free)
        setattr(cls, "free", _wrapped_free)
    setattr(cls, _KV_CACHE_MANAGER_PATCH_ATTR, True)
    hook_logger.info("Jiuwen KVBM KVCacheManager hook installed on %s", cls)
    return True


def _install_coordinator_cache_blocks_hook(
    manager: Any,
    log: logging.Logger,
) -> bool:
    """在实际 KV cache coordinator 实例上安装 cache_blocks hook。

    参数:
        manager: 已初始化的 vLLM KVCacheManager 实例。
        log: 记录 hook 安装结果的日志对象。

    返回:
        安装成功或已安装时返回 True；coordinator 不兼容时
        返回 False。

    说明:
        vLLM-Ascend 可能绕过 BlockPool.cache_full_blocks，改由
        coordinator.cache_blocks 完成 hash 发布。本 hook 在发布后
        从请求 block table 建立增量映射，避免 store 事件触发
        BlockPool 全量快照。
    """

    coordinator = getattr(manager, "coordinator", None)
    original = getattr(coordinator, "cache_blocks", None)
    if coordinator is None or not callable(original):
        return False
    if getattr(coordinator, _COORDINATOR_CACHE_BLOCKS_PATCH_ATTR, False):
        return True

    def _wrapped_cache_blocks(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        request = args[0] if args else kwargs.get("request")
        num_tokens_to_cache = args[1] if len(args) > 1 else kwargs.get("num_tokens_to_cache", kwargs.get("num_tokens"))
        mapping_started_ns = time.perf_counter_ns()
        mapping_result = _record_coordinator_request_block_mappings(
            manager,
            coordinator,
            request,
            num_tokens_to_cache=num_tokens_to_cache,
        )
        mapping_ms = (time.perf_counter_ns() - mapping_started_ns) / 1_000_000.0
        resolver = _ENGINE_CORE_RESOLVER
        if resolver is not None:
            resolver.note_coordinator_cache_mapping_result(
                recorded=int(mapping_result["recorded"]),
                visited_blocks=int(mapping_result["visited_blocks"]),
                skipped_blocks=int(mapping_result["skipped_blocks"]),
                elapsed_ms=mapping_ms,
            )
        return result

    try:
        setattr(coordinator, _COORDINATOR_CACHE_BLOCKS_ORIGINAL_ATTR, original)
        setattr(coordinator, "cache_blocks", _wrapped_cache_blocks)
        setattr(coordinator, _COORDINATOR_CACHE_BLOCKS_PATCH_ATTR, True)
    except (AttributeError, TypeError) as exc:
        log.warning(
            "Jiuwen KVBM coordinator cache_blocks hook skipped: type=%s error=%s",
            type(coordinator).__name__,
            exc,
        )
        return False
    log.info(
        "Jiuwen KVBM coordinator cache_blocks hook installed on %s",
        type(coordinator),
    )
    return True


def _record_coordinator_request_block_mappings(
    manager: Any,
    coordinator: Any,
    request: Any,
    *,
    num_tokens_to_cache: Any,
) -> dict[str, int]:
    """从 coordinator 请求 block table 记录已缓存的物理映射。

    参数:
        manager: 当前 KVCacheManager，用于确定真实 block size。
        coordinator: 完成 cache_blocks 的 coordinator 实例。
        request: vLLM Request，提供 request_id 和 block_hashes。
        num_tokens_to_cache: 本次允许发布到 prefix cache 的 token 数。

    返回:
        包含 recorded、visited_blocks 和 skipped_blocks 的诊断字典。
    """

    if request is None:
        return {"recorded": 0, "visited_blocks": 0, "skipped_blocks": 0}
    request_id = str(getattr(request, "request_id", "") or "").strip()
    request_hashes = _flatten_request_block_hashes(getattr(request, "block_hashes", None))
    if not request_id or not request_hashes:
        return {"recorded": 0, "visited_blocks": 0, "skipped_blocks": 0}
    block_size = _diagnostic_block_size(
        manager,
        prompt_token_count=0,
        request_hash_count=len(request_hashes),
    )
    if block_size <= 0:
        return {"recorded": 0, "visited_blocks": 0, "skipped_blocks": 0}
    try:
        full_block_count = max(int(num_tokens_to_cache), 0) // block_size
    except (TypeError, ValueError):
        full_block_count = len(request_hashes)
    full_block_count = min(full_block_count, len(request_hashes))

    progress = getattr(coordinator, _COORDINATOR_CACHE_MAPPING_PROGRESS_ATTR, None)
    if not isinstance(progress, dict):
        progress = {}
        try:
            setattr(coordinator, _COORDINATOR_CACHE_MAPPING_PROGRESS_ATTR, progress)
        except (AttributeError, TypeError):
            progress = {}
    previous_full_block_count = max(int(progress.get(request_id, 0)), 0)
    if full_block_count < previous_full_block_count:
        previous_full_block_count = 0
    if full_block_count <= previous_full_block_count:
        return {
            "recorded": 0,
            "visited_blocks": 0,
            "skipped_blocks": full_block_count,
        }

    try:
        request_blocks = coordinator.get_blocks(request_id)
    except Exception:
        return {"recorded": 0, "visited_blocks": 0, "skipped_blocks": 0}
    groups = getattr(request_blocks, "blocks", request_blocks)
    if not isinstance(groups, list | tuple) or not groups:
        return {"recorded": 0, "visited_blocks": 0, "skipped_blocks": 0}

    recorded = 0
    visited_blocks = 0
    for logical_index in range(previous_full_block_count, full_block_count):
        request_hash = request_hashes[logical_index]
        for group_id, group_blocks in enumerate(groups):
            if not isinstance(group_blocks, list | tuple):
                group_blocks = getattr(group_blocks, "blocks", group_blocks)
            try:
                block = group_blocks[logical_index]
                block_id = int(getattr(block, "block_id"))
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            visited_blocks += 1
            block_hash = getattr(block, "block_hash", None)
            if block_hash is None:
                block_hash = getattr(block, "_block_hash", None)
            if block_hash is not None:
                _record_engine_core_physical_mapping(block_hash, block_id)
            _record_engine_core_physical_mapping(request_hash, block_id)
            if isinstance(request_hash, bytes | bytearray):
                packed_hash = bytes(request_hash) + group_id.to_bytes(4, "big")
                _record_engine_core_physical_mapping(packed_hash, block_id)
            recorded += 1
    if recorded:
        progress[request_id] = full_block_count
    return {
        "recorded": recorded,
        "visited_blocks": visited_blocks,
        "skipped_blocks": previous_full_block_count,
    }


def _log_prefix_cache_lookup_diagnostics(
    manager: Any,
    request: Any,
    result: Any,
    log: logging.Logger,
) -> None:
    """记录 vLLM 本地 prefix-cache 查询与 BlockPool 索引的对应关系。

    参数:
        manager: 实际执行查询的 vLLM KVCacheManager 实例。
        request: 当前 vLLM 请求，包含 request_id 和 block_hashes。
        result: get_computed_blocks() 的原始返回值。
        log: 用于输出诊断信息的日志对象。

    返回:
        无返回值；诊断字段读取失败时使用空值，不影响 vLLM 查询结果。
    """

    request_id = str(getattr(request, "request_id", "") or "")
    request_hashes = _flatten_request_block_hashes(getattr(request, "block_hashes", None))
    prompt_token_ids = _diagnostic_prompt_token_ids(request)
    diagnostic_block_size = _diagnostic_block_size(
        manager,
        prompt_token_count=len(prompt_token_ids),
        request_hash_count=len(request_hashes),
    )
    token_block_fingerprints = _token_block_fingerprints(
        prompt_token_ids,
        diagnostic_block_size,
    )
    computed_tokens = 0
    computed_block_ids: Any = []
    if isinstance(result, tuple):
        if len(result) > 1:
            computed_tokens = _safe_nonnegative_int(result[1])
        if result:
            get_block_ids = getattr(result[0], "get_block_ids", None)
            if callable(get_block_ids):
                try:
                    computed_block_ids = get_block_ids()
                except Exception:
                    computed_block_ids = []

    block_pool = getattr(manager, "block_pool", None)
    cached_map = getattr(block_pool, "cached_block_hash_to_block", None)
    raw_cache = getattr(cached_map, "_cache", cached_map)
    cached_keys = list(raw_cache.keys()) if isinstance(raw_cache, dict) else []
    request_fingerprints = [_block_hash_fingerprint(value) for value in request_hashes]
    cached_fingerprints = [_block_hash_fingerprint(value, strip_group_id=True) for value in cached_keys]
    cached_fingerprint_set = set(cached_fingerprints)
    matching_request_positions = [
        index for index, fingerprint in enumerate(request_fingerprints) if fingerprint in cached_fingerprint_set
    ]
    cached_aliases = {alias for key in cached_keys for alias in _block_hash_aliases(key)}
    matching_hashes = [
        request_hash
        for request_hash in request_hashes
        if cached_aliases.intersection(_block_hash_aliases(request_hash))
    ]

    log.info(
        "Jiuwen KVBM prefix-cache lookup diagnostics: %s",
        {
            "request_id": request_id,
            "manager_id": hex(id(manager)),
            "block_pool_id": hex(id(block_pool)) if block_pool is not None else "",
            "prefix_cache_enabled": bool(getattr(manager, "enable_caching", False)),
            "skip_reading_prefix_cache": bool(getattr(request, "skip_reading_prefix_cache", False)),
            "cache_salt": _diagnostic_scalar(getattr(request, "cache_salt", None)),
            "lora_request_fingerprint": _diagnostic_object_fingerprint(getattr(request, "lora_request", None)),
            "mm_features_fingerprint": _diagnostic_object_fingerprint(getattr(request, "mm_features", None)),
            "prompt_token_count": len(prompt_token_ids),
            "prompt_token_head": prompt_token_ids[:16],
            "prompt_token_tail": prompt_token_ids[-16:],
            "diagnostic_block_size": diagnostic_block_size,
            "token_block_fingerprints": token_block_fingerprints[:16],
            "request_hash_count": len(request_hashes),
            "request_hash_types": sorted({type(value).__name__ for value in request_hashes}),
            "request_hash_fingerprints": request_fingerprints[:16],
            "cached_hash_count": len(cached_keys),
            "cached_hash_types": sorted({type(value).__name__ for value in cached_keys}),
            "cached_hash_head_fingerprints": cached_fingerprints[:16],
            "cached_hash_tail_fingerprints": cached_fingerprints[-16:],
            "cached_group_ids": _cached_hash_group_ids(cached_keys),
            "matching_request_hashes": len(matching_hashes),
            "matching_fingerprint_count": len(matching_request_positions),
            "matching_request_positions": matching_request_positions[:16],
            "computed_tokens": computed_tokens,
            "computed_block_ids": computed_block_ids,
        },
    )


def _diagnostic_prompt_token_ids(request: Any) -> list[int]:
    """读取请求的 prompt token ID，供跨 worker hash 输入诊断使用。

    参数:
        request: vLLM 内部请求对象。

    返回:
        可安全转换为整数的 prompt token ID 列表；字段缺失或格式异常时返回空列表。
    """

    raw_token_ids = getattr(request, "prompt_token_ids", None)
    if not isinstance(raw_token_ids, list | tuple):
        return []
    token_ids: list[int] = []
    for value in raw_token_ids:
        try:
            token_ids.append(int(value))
        except (TypeError, ValueError):
            return []
    return token_ids


def _diagnostic_block_size(
    manager: Any,
    *,
    prompt_token_count: int,
    request_hash_count: int,
) -> int:
    """解析 prefix-cache 诊断使用的 token block 大小。

    参数:
        manager: vLLM KVCacheManager 实例。
        prompt_token_count: 当前请求的 prompt token 数量。
        request_hash_count: 当前请求已生成的完整 block hash 数量。

    返回:
        优先返回 manager 暴露的真实 block size；无法读取时根据完整 block 数估算，
        仍无法确定时返回 0。
    """

    coordinator = getattr(manager, "coordinator", None)
    single_type_managers = getattr(coordinator, "single_type_managers", ())
    first_group_manager = (
        single_type_managers[0] if isinstance(single_type_managers, list | tuple) and single_type_managers else None
    )
    candidates = (
        getattr(manager, "block_size", None),
        getattr(getattr(manager, "cache_config", None), "block_size", None),
        getattr(getattr(manager, "kv_cache_config", None), "block_size", None),
        getattr(first_group_manager, "block_size", None),
        os.getenv(_BLOCK_SIZE_ENV),
    )
    for candidate in candidates:
        try:
            parsed = int(candidate)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    if prompt_token_count > 0 and request_hash_count > 0:
        estimated = prompt_token_count // request_hash_count
        if estimated > 0:
            return estimated
    return 0


def _token_block_fingerprints(token_ids: list[int], block_size: int) -> list[str]:
    """按 block 边界生成 prompt token 的稳定短指纹。

    参数:
        token_ids: 当前请求的 prompt token ID 列表。
        block_size: 每个 KV block 对应的 token 数量。

    返回:
        每个完整 token block 的 SHA256 前 16 位；输入为空或 block size 无效时返回空列表。
    """

    if not token_ids or block_size <= 0:
        return []
    fingerprints: list[str] = []
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        block = token_ids[start:start + block_size]
        encoded = json.dumps(block, separators=(",", ":")).encode("ascii")
        fingerprints.append(hashlib.sha256(encoded).hexdigest()[:16])
    return fingerprints


def _partial_token_fingerprint(token_ids: list[int], block_size: int) -> str:
    """为 prompt 最后一个未填满 block 生成稳定 token 指纹。

    参数:
        token_ids: 当前请求完整 prompt token ID。
        block_size: 每个 KV block 对应的 token 数。

    返回:
        存在 partial block 时返回其 token span 的 SHA256 前 16 位；prompt 正好
        block 对齐、输入为空或 block_size 无效时返回空字符串。
    """

    if not token_ids or block_size <= 0:
        return ""
    valid_tokens = len(token_ids) % block_size
    if valid_tokens == 0:
        return ""
    encoded = json.dumps(
        token_ids[-valid_tokens:],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _diagnostic_scalar(value: Any) -> Any:
    """把诊断标量转换为长度受控且可直接比较的日志值。

    参数:
        value: 待记录的 cache salt 等标量值。

    返回:
        None、布尔值、数值或最长 128 字符的字符串表示。
    """

    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)[:128]


def _diagnostic_object_fingerprint(value: Any) -> str:
    """为可能较大的请求附加对象生成不可逆诊断指纹。

    参数:
        value: LoRA 请求、多模态特征等可选 hash 输入。

    返回:
        对象为空时返回空字符串，否则返回其稳定表示的 SHA256 前 16 位。
    """

    if value is None:
        return ""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            default=repr,
            separators=(",", ":"),
        ).encode("utf-8", errors="backslashreplace")
    except (TypeError, ValueError):
        encoded = repr(value).encode("utf-8", errors="backslashreplace")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _block_hash_fingerprint(value: Any, *, strip_group_id: bool = False) -> str:
    """生成便于复制和比较的短 block hash 指纹。

    参数:
        value: vLLM 请求 hash 或 BlockPool packed hash。
        strip_group_id: 是否去掉 packed hash 末尾 4 字节 cache group ID。

    返回:
        规范化字节内容的 SHA256 前 16 个十六进制字符。
    """

    if isinstance(value, bytes | bytearray):
        raw = bytes(value)
        if strip_group_id and len(raw) > 4:
            raw = raw[:-4]
    elif isinstance(value, str):
        raw = value.encode("utf-8", errors="backslashreplace")
    elif isinstance(value, int):
        raw = str(value).encode("ascii")
    else:
        raw = repr(value).encode("utf-8", errors="backslashreplace")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cached_hash_group_ids(cached_keys: list[Any]) -> list[int]:
    """提取 BlockPool packed hash 中出现的 cache group ID。

    参数:
        cached_keys: BlockPool hash 索引键列表。

    返回:
        去重并排序后的 group ID；非字节 hash 不参与统计。
    """

    group_ids = {
        int.from_bytes(bytes(value)[-4:], byteorder="big")
        for value in cached_keys
        if isinstance(value, bytes | bytearray) and len(value) > 4
    }
    return sorted(group_ids)


def _flatten_request_block_hashes(value: Any) -> list[Any]:
    """把请求中的单组或多组 block hash 结构拍平成列表。

    参数:
        value: request.block_hashes，可能是列表、按 cache group 分组的元组或空值。

    返回:
        保持原顺序的 block hash 列表；无法识别时返回空列表。
    """

    if value is None:
        return []
    if isinstance(value, bytes | bytearray | str | int):
        return [value]
    if not isinstance(value, list | tuple):
        return [value]
    if isinstance(value, tuple) and _looks_like_hash_tuple(value):
        return [value]
    flattened: list[Any] = []
    for item in value:
        if isinstance(item, list):
            flattened.extend(_flatten_request_block_hashes(item))
        elif isinstance(item, tuple) and not _looks_like_hash_tuple(item):
            flattened.extend(_flatten_request_block_hashes(item))
        else:
            flattened.append(item)
    return flattened


def _looks_like_hash_tuple(value: list[Any] | tuple[Any, ...]) -> bool:
    """判断序列本身是否是 vLLM 的结构化 block hash。

    参数:
        value: 待判断的列表或元组。

    返回:
        序列由标量 hash 组成且长度较小时返回 True，否则返回 False。
    """

    return 0 < len(value) <= 4 and all(_is_scalar(item) for item in value)


def _pd_request_id_aliases(request_id: str) -> tuple[str, ...]:
    """生成 Router 父请求与 vLLM 内部子请求之间的受约束别名。

    参数:
        request_id: Router 或 vLLM 提供的请求 ID。vLLM serving 层可能追加
            `-8位十六进制` 后缀，也可能使用包含 request_id 的 JSON 编码。

    返回:
        去重且保持优先顺序的请求 ID；第一项始终是输入值，随后为可验证的
        父请求 ID。不会对任意短横线后缀进行模糊截断。
    """

    pending = [str(request_id).strip()]
    aliases: list[str] = []
    while pending:
        candidate = pending.pop(0)
        if not candidate or candidate in aliases:
            continue
        aliases.append(candidate)
        suffix_match = _VLLM_INTERNAL_REQUEST_SUFFIX.fullmatch(candidate)
        if suffix_match is not None:
            pending.append(suffix_match.group("parent"))
        try:
            decoded = json.loads(candidate)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, dict):
            embedded = decoded.get("request_id")
            if isinstance(embedded, str) and embedded.strip():
                pending.append(embedded.strip())
    return tuple(aliases)


def _record_pd_request_block_hashes(request: Any, *, block_size: int = 0) -> None:
    """保存 P Worker 请求的完整 block hash 顺序，供 prefill 完成后导出。

    参数:
        request: vLLM Request 对象，需包含 request_id 和 block_hashes。
        block_size: 当前 Scheduler/KVCacheManager 的逻辑 block token 数；未知时
            传 0，并根据完整 token 数和 hash 数量保守推导。

    返回:
        无返回值；请求 ID 或 block hash 为空时不记录。

    说明:
        vLLM 会在请求结束时释放 ``req_to_blocks``，但已发布 block 仍可能留在
        prefix cache。保存逻辑 hash 后，export 可以在请求结束后从 BlockPool
        索引重新定位物理 block，并在同一临界区内完成 pin。
    """

    request_id = str(getattr(request, "request_id", "") or "").strip()
    block_hashes = tuple(_flatten_request_block_hashes(getattr(request, "block_hashes", None)))
    if not request_id or not block_hashes:
        return
    aliases = _pd_request_id_aliases(request_id)
    token_ids = _diagnostic_prompt_token_ids(request)
    resolved_block_size = int(block_size)
    if resolved_block_size <= 0 and token_ids and block_hashes:
        resolved_block_size = len(token_ids) // len(block_hashes)
    token_fingerprints: tuple[str, ...] = ()
    if _pd_request_supports_token_fingerprint(request):
        generated = _token_block_fingerprints(token_ids, resolved_block_size)
        if len(generated) >= len(block_hashes):
            token_fingerprints = tuple(generated[:len(block_hashes)])
    with _VLLM_BLOCK_POOL_LOCK:
        first_record = all(alias not in _PD_REQUEST_BLOCK_HASHES for alias in aliases)
        if first_record and len(_PD_REQUEST_BLOCK_HASHES) + len(aliases) > _PD_REQUEST_BLOCK_HASHES_LIMIT:
            while (
                _PD_REQUEST_BLOCK_HASHES
                and len(_PD_REQUEST_BLOCK_HASHES) + len(aliases) > _PD_REQUEST_BLOCK_HASHES_LIMIT
            ):
                oldest_request_id = next(iter(_PD_REQUEST_BLOCK_HASHES))
                _PD_REQUEST_BLOCK_HASHES.pop(oldest_request_id, None)
                _PD_REQUEST_TOKEN_BLOCK_FINGERPRINTS.pop(oldest_request_id, None)
        for alias in aliases:
            _PD_REQUEST_BLOCK_HASHES[alias] = block_hashes
            if token_fingerprints:
                _PD_REQUEST_TOKEN_BLOCK_FINGERPRINTS[alias] = token_fingerprints
    if first_record:
        logger.info(
            "Jiuwen KVBM recorded P request block hashes: request_id=%s logical_blocks=%d",
            request_id,
            len(block_hashes),
        )


def _local_pd_export_prepare(payload: dict[str, Any]) -> dict[str, Any]:
    """在 P scheduler 注册请求释放前的物理 block 捕获意图。

    参数:
        payload: 包含 handoff_id、request_id、prompt_token_count 和可选 ttl_ms。

    返回:
        包含 created 和 handoff_id 的幂等响应。相同请求只能绑定同一 handoff。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    request_id = str(payload.get("request_id", "")).strip()
    prompt_token_count = int(payload.get("prompt_token_count", 0))
    ttl_ms = int(payload.get("ttl_ms", 120_000))
    if not handoff_id or not request_id:
        raise ValueError("handoff_id and request_id are required")
    if prompt_token_count <= 0:
        raise ValueError("prompt_token_count must be positive")
    if ttl_ms < 1_000 or ttl_ms > 600_000:
        raise ValueError("P export capture ttl_ms must be between 1000 and 600000")
    aliases = _pd_request_id_aliases(request_id)
    with _VLLM_BLOCK_POOL_LOCK:
        existing = next(
            (_PD_SOURCE_EXPORT_INTENTS[alias] for alias in aliases if alias in _PD_SOURCE_EXPORT_INTENTS),
            None,
        )
        if existing is not None:
            if existing.handoff_id != handoff_id:
                raise ValueError(f"P request {request_id} is already bound to handoff {existing.handoff_id}")
            return {
                "ok": True,
                "created": False,
                "handoff_id": handoff_id,
                "captured": bool(existing.source_lease_id),
                "error": existing.error or None,
            }
        intent = _PDSourceExportIntent(
            handoff_id=handoff_id,
            request_id=request_id,
            prompt_token_count=prompt_token_count,
            ttl_ms=ttl_ms,
            created_at=time.monotonic(),
        )
        for alias in aliases:
            _PD_SOURCE_EXPORT_INTENTS[alias] = intent
    return {
        "ok": True,
        "created": True,
        "handoff_id": handoff_id,
        "captured": False,
    }


def _capture_pd_source_request_blocks(
    manager: Any,
    request: Any,
    log: logging.Logger,
) -> None:
    """在 P 请求释放前 pin 完整 prompt block 和最后一个 partial block。

    参数:
        manager: 当前 vLLM KVCacheManager，提供请求 block table 和 BlockPool。
        request: 即将被释放的 vLLM Request。
        log: 用于输出捕获结果和兼容性错误的日志对象。

    返回:
        无返回值。没有 export 意图或 prompt 正好 block 对齐时保持现有 hash
        export 路径；捕获失败会写入 intent.error，由后续 export 明确返回。
    """

    engine_request_id = str(getattr(request, "request_id", "") or "").strip()
    intent = next(
        (
            _PD_SOURCE_EXPORT_INTENTS[alias]
            for alias in _pd_request_id_aliases(engine_request_id)
            if alias in _PD_SOURCE_EXPORT_INTENTS
        ),
        None,
    )
    if intent is None or intent.capture_complete or intent.source_lease_id:
        return
    if intent.error:
        return
    intent.kv_ready_at = time.monotonic()
    intent.kv_ready_cycle_id = _current_pd_scheduler_cycle()
    prompt_token_ids = _diagnostic_prompt_token_ids(request)
    prompt_token_count = min(intent.prompt_token_count, len(prompt_token_ids))
    if _pd_prompt_diagnostics_enabled():
        logger.info(
            "[PDSourceCaptureDiagnostics] handoff_id=%s engine_request_id=%s "
            "intent_prompt_tokens=%d engine_prompt_tokens=%d capture_prompt_tokens=%d "
            "engine_token_sha256=%s capture_token_sha256=%s truncated=%s",
            intent.handoff_id,
            engine_request_id,
            intent.prompt_token_count,
            len(prompt_token_ids),
            prompt_token_count,
            _pd_token_digest(prompt_token_ids),
            _pd_token_digest(prompt_token_ids[:prompt_token_count]),
            prompt_token_count != len(prompt_token_ids),
        )
    request_hashes = tuple(_flatten_request_block_hashes(getattr(request, "block_hashes", None)))
    block_size = _diagnostic_block_size(
        manager,
        prompt_token_count=prompt_token_count,
        request_hash_count=len(request_hashes),
    )
    partial_valid_tokens = prompt_token_count % block_size if block_size > 0 else 0
    if _pd_prompt_diagnostics_enabled():
        logger.info(
            "[PDSourceCaptureDiagnostics] handoff_id=%s engine_request_id=%s "
            "block_size=%d capture_partial_valid_tokens=%d engine_partial_valid_tokens=%d",
            intent.handoff_id,
            engine_request_id,
            block_size,
            partial_valid_tokens,
            len(prompt_token_ids) % block_size if block_size > 0 else 0,
        )
    if partial_valid_tokens == 0:
        _complete_pd_source_capture(intent)
        return
    if not _pd_request_supports_token_fingerprint(request):
        # partial block 没有完整 hash，带 salt/LoRA/多模态语义时无法只凭 token
        # 验证 P/D KV 等价；保留原有完整 block export，D 本地补算尾部。
        _complete_pd_source_capture(intent)
        log.info(
            "Jiuwen KVBM skipped request-scoped P partial capture because the "
            "request has extra KV hash semantics: handoff_id=%s request_id=%s",
            intent.handoff_id,
            engine_request_id,
        )
        return
    try:
        get_blocks = getattr(manager, "get_blocks", None)
        if callable(get_blocks):
            request_blocks = get_blocks(engine_request_id)
            groups = getattr(request_blocks, "blocks", request_blocks)
        else:
            coordinator = getattr(manager, "coordinator", None)
            groups = coordinator.get_blocks(engine_request_id)
        if not isinstance(groups, list | tuple) or not groups:
            raise RuntimeError("vLLM request block table is not available")
        full_logical_blocks = prompt_token_count // block_size
        if len(request_hashes) < full_logical_blocks:
            raise RuntimeError("P request does not expose all complete prompt block hashes")
        source_blocks: list[int] = []
        block_groups: list[dict[str, Any]] = []
        block_hashes: list[dict[str, Any]] = []
        for logical_index in range(full_logical_blocks + 1):
            for group_id, group_blocks in enumerate(groups):
                if logical_index >= len(group_blocks):
                    raise RuntimeError(
                        "P request block table does not contain the prompt tail: "
                        f"group_id={group_id} logical_index={logical_index}"
                    )
                block = group_blocks[logical_index]
                block_id = int(getattr(block, "block_id"))
                source_blocks.append(block_id)
                if logical_index < full_logical_blocks:
                    request_hash = request_hashes[logical_index]
                    if not isinstance(request_hash, bytes | bytearray):
                        raise RuntimeError("P request block hash type is not compatible with vLLM 0.18")
                    packed_hash = bytes(request_hash) + group_id.to_bytes(4, "big")
                    descriptor = _encode_packed_block_hash(packed_hash)
                    group = {
                        "logical_index": logical_index,
                        "group_id": group_id,
                        "block_id": block_id,
                        "request_scoped_partial": False,
                        "block_hash": descriptor,
                    }
                else:
                    descriptor = {
                        "request_scoped_partial": True,
                        "logical_index": logical_index,
                        "group_id": group_id,
                        "valid_token_count": partial_valid_tokens,
                    }
                    group = {
                        **descriptor,
                        "block_id": block_id,
                        "block_hash": None,
                    }
                block_hashes.append(descriptor)
                block_groups.append(group)
        lease = _process_source_block_leases().acquire(
            source_blocks,
            ttl_ms=intent.ttl_ms,
            require_published=False,
        )
        intent.source_lease_id = lease.lease_id
        intent.source_blocks = tuple(source_blocks)
        intent.block_groups = tuple(block_groups)
        intent.block_hashes = tuple(block_hashes)
        intent.token_block_fingerprints = tuple(
            _token_block_fingerprints(prompt_token_ids[:prompt_token_count], block_size)
        )
        intent.partial_token_fingerprint = _partial_token_fingerprint(
            prompt_token_ids[:prompt_token_count],
            block_size,
        )
        intent.semantic_plain_text = True
        _complete_pd_source_capture(intent)
        log.info(
            "Jiuwen KVBM captured request-scoped P partial block before free: "
            "handoff_id=%s request_id=%s full_logical_blocks=%d "
            "partial_valid_tokens=%d physical_blocks=%d lease_id=%s",
            intent.handoff_id,
            engine_request_id,
            full_logical_blocks,
            partial_valid_tokens,
            len(source_blocks),
            lease.lease_id,
        )
    except Exception as exc:
        _complete_pd_source_capture(intent, error=str(exc))
        log.exception(
            "Jiuwen KVBM failed to capture request-scoped P partial block: handoff_id=%s request_id=%s",
            intent.handoff_id,
            engine_request_id,
        )


def _defer_pd_source_capture_for_preemption(
    request: Any,
    log: logging.Logger,
) -> bool:
    """识别 vLLM 抢占触发的临时 block 释放并延后 P export 捕获。

    参数:
        request: 即将由 KVCacheManager 释放 block table 的 vLLM Request。
        log: 用于记录抢占诊断信息的日志对象。

    返回:
        当前释放属于 Scheduler 抢占时返回 True；普通完成或取消释放返回 False。
    """

    engine_request_id = str(getattr(request, "request_id", "") or "").strip()
    aliases = _pd_request_id_aliases(engine_request_id)
    if not any(_PD_PREEMPTING_REQUEST_IDS.get(alias, 0) > 0 for alias in aliases):
        return False
    intent = next(
        (_PD_SOURCE_EXPORT_INTENTS[alias] for alias in aliases if alias in _PD_SOURCE_EXPORT_INTENTS),
        None,
    )
    if intent is None or intent.capture_complete:
        return True
    intent.preemption_count += 1
    intent.last_preempted_computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
    log.info(
        "Jiuwen KVBM deferred P export capture across vLLM preemption: "
        "handoff_id=%s request_id=%s preemption_count=%s computed_tokens=%s",
        intent.handoff_id,
        engine_request_id,
        intent.preemption_count,
        intent.last_preempted_computed_tokens,
    )
    return True


def _complete_pd_source_capture(
    intent: _PDSourceExportIntent,
    *,
    error: str = "",
) -> None:
    """完成一次P请求block捕获，并向Router发布可重放事件。

    参数:
        intent: 当前P请求对应的export捕获意图。
        error: 捕获失败时的稳定错误文本；为空表示捕获已经可以导出。

    返回:
        无返回值。函数会更新capture状态、唤醒本地等待者，并发布
        EXPORT_READY或EXPORT_FAILED事件。
    """

    intent.error = str(error or "")
    intent.capture_complete = True
    intent.capture_ready_at = time.monotonic()
    intent.capture_ready.set()
    _PD_CONTROL_EVENTS.publish(
        "EXPORT_FAILED" if intent.error else "EXPORT_READY",
        handoff_id=intent.handoff_id,
        request_id=intent.request_id,
        result={
            "ok": not bool(intent.error),
            "error": intent.error or None,
        },
    )


def _release_pd_source_export_intent(
    request_id: str,
    handoff_id: str,
) -> dict[str, Any]:
    """删除 P scheduler 的请求级 export 捕获记录。

    参数:
        request_id: Router/engine request ID。
        handoff_id: 用于防止误删新请求记录的 handoff ID。

    返回:
        包含 released 的幂等结果；source lease 由调用方单独释放。
    """

    with _VLLM_BLOCK_POOL_LOCK:
        intent = next(
            (
                _PD_SOURCE_EXPORT_INTENTS[alias]
                for alias in _pd_request_id_aliases(request_id)
                if alias in _PD_SOURCE_EXPORT_INTENTS
            ),
            None,
        )
        if intent is None:
            return {"ok": True, "released": False}
        if intent.handoff_id != handoff_id:
            raise ValueError("handoff_id does not match P export capture")
        intent.capture_ready.set()
        aliases = [alias for alias, candidate in _PD_SOURCE_EXPORT_INTENTS.items() if candidate is intent]
        for alias in aliases:
            _PD_SOURCE_EXPORT_INTENTS.pop(alias, None)
    return {"ok": True, "released": True}


def _expire_pd_source_export_intents(now: float | None = None) -> int:
    """删除超过捕获 TTL 的 P scheduler export 意图。

    参数:
        now: 可选单调时钟时间戳；为空时读取当前时间。

    返回:
        本次删除的不同 intent 数量。对应 source lease 由 lease manager 的同一
        sweeper 独立过期释放，避免在两个清理器之间重复 free block。
    """

    current = time.monotonic() if now is None else float(now)
    with _VLLM_BLOCK_POOL_LOCK:
        expired = {
            id(intent): intent
            for intent in _PD_SOURCE_EXPORT_INTENTS.values()
            if current >= intent.created_at + intent.ttl_ms / 1000.0
        }
        if not expired:
            return 0
        for alias, intent in list(_PD_SOURCE_EXPORT_INTENTS.items()):
            if id(intent) in expired:
                intent.capture_ready.set()
                _PD_SOURCE_EXPORT_INTENTS.pop(alias, None)
    return len(expired)


def _pd_request_supports_token_fingerprint(request: Any) -> bool:
    """判断请求能否仅依靠 token block 指纹安全确认 KV 内容等价。

    参数:
        request: vLLM Request 对象。

    返回:
        请求不含 cache salt、LoRA、多模态特征和 prompt embeddings 时返回
        True；任何附加 KV 语义存在时返回 False。
    """

    return (
        getattr(request, "cache_salt", None) in (None, "")
        and getattr(request, "lora_request", None) is None
        and not getattr(request, "mm_features", None)
        and getattr(request, "prompt_embeds", None) is None
    )


def install_vllm_ascend_018_scheduler_diagnostics_hook(
    *,
    scheduler_cls: type[Any] | None = None,
    new_request_data_cls: type[Any] | None = None,
    log: logging.Logger | None = None,
) -> bool:
    """安装 vLLM Scheduler 请求级 KV cache 命中诊断 hook。

    参数:
        scheduler_cls: 可选 Scheduler 类；为空时自动导入
            vllm.v1.core.sched.scheduler.Scheduler。
        new_request_data_cls: 可选 NewRequestData 类；为空时自动导入
            vllm.v1.core.sched.output.NewRequestData。
        log: 可选日志对象；为空时使用模块 logger。

    返回:
        至少一个诊断挂点安装成功或已经安装时返回 True；两个挂点都无法安装
        时返回 False。

    说明:
        优先 hook 所有调度实现最终构造新请求数据时调用的
        NewRequestData.from_request()，兼容 vLLM-Ascend 替换 Scheduler 类的情况；
        同时保留 Scheduler.schedule() 输出诊断作为兜底。两条路径按请求 ID
        去重，只输出一次真实 prefix cache/BlockPool 命中统计。
    """

    hook_logger = log or logger
    scheduler_type = scheduler_cls
    if scheduler_type is None:
        try:
            from vllm.v1.core.sched.scheduler import Scheduler

            scheduler_type = Scheduler
        except Exception as exc:
            hook_logger.warning(
                "Jiuwen KVBM scheduler cache-hit diagnostics hook skipped: %s",
                exc,
            )

    request_data_type = new_request_data_cls
    if request_data_type is None:
        try:
            from vllm.v1.core.sched.output import NewRequestData

            request_data_type = NewRequestData
        except Exception as exc:
            hook_logger.warning(
                "Jiuwen KVBM request-data cache-hit diagnostics hook skipped: %s",
                exc,
            )

    installed = False
    if request_data_type is not None:
        installed = (
            _install_new_request_data_diagnostics_hook(
                request_data_type,
                hook_logger,
            )
            or installed
        )
    if scheduler_type is not None:
        installed = (
            _install_scheduler_preemption_hook(
                scheduler_type,
                hook_logger,
            )
            or installed
        )
        installed = (
            _install_scheduler_output_diagnostics_hook(
                scheduler_type,
                hook_logger,
            )
            or installed
        )
    return installed


def _install_scheduler_preemption_hook(
    scheduler_cls: type[Any],
    log: logging.Logger,
) -> bool:
    """标记 Scheduler 抢占期间发生的 KVCacheManager.free 调用。

    参数:
        scheduler_cls: vLLM Scheduler 基类或实际运行时子类。
        log: 用于输出安装状态的日志对象。

    返回:
        找到并成功 hook `_preempt_request`，或其所属类已经 hook 时返回 True；
        当前 Scheduler 不提供该方法时返回 False。
    """

    owner = next(
        (candidate for candidate in scheduler_cls.__mro__ if "_preempt_request" in candidate.__dict__),
        None,
    )
    if owner is None:
        return False
    if owner.__dict__.get(_SCHEDULER_PREEMPT_PATCH_ATTR, False):
        return True
    original = owner.__dict__["_preempt_request"]

    def _wrapped_preempt_request(
        self: Any,
        request: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        request_id = str(getattr(request, "request_id", "") or "").strip()
        aliases = _pd_request_id_aliases(request_id)
        for alias in aliases:
            _PD_PREEMPTING_REQUEST_IDS[alias] = _PD_PREEMPTING_REQUEST_IDS.get(alias, 0) + 1
        try:
            return original(self, request, *args, **kwargs)
        finally:
            for alias in aliases:
                remaining = _PD_PREEMPTING_REQUEST_IDS.get(alias, 0) - 1
                if remaining > 0:
                    _PD_PREEMPTING_REQUEST_IDS[alias] = remaining
                else:
                    _PD_PREEMPTING_REQUEST_IDS.pop(alias, None)

    setattr(owner, _SCHEDULER_PREEMPT_ORIGINAL_ATTR, original)
    setattr(owner, "_preempt_request", _wrapped_preempt_request)
    setattr(owner, _SCHEDULER_PREEMPT_PATCH_ATTR, True)
    log.info("Jiuwen KVBM scheduler preemption hook installed on %s", owner)
    return True


def _install_scheduler_output_diagnostics_hook(
    scheduler_cls: type[Any],
    log: logging.Logger,
) -> bool:
    """安装 Scheduler.schedule() 输出诊断兜底挂点。

    参数:
        scheduler_cls: 待 hook 的 vLLM Scheduler 类。
        log: 用于输出安装状态和请求诊断的日志对象。

    返回:
        安装成功或已经安装时返回 True；缺少 schedule 方法时返回 False。
    """

    if getattr(scheduler_cls, _SCHEDULER_DIAGNOSTICS_PATCH_ATTR, False):
        log.info("Jiuwen KVBM scheduler cache-hit diagnostics hook already installed")
        return True
    original = getattr(scheduler_cls, "schedule", None)
    if original is None:
        log.warning("Jiuwen KVBM scheduler cache-hit diagnostics hook skipped: schedule is missing")
        return False

    original_add_request = getattr(scheduler_cls, "add_request", None)
    original_update_from_output = getattr(
        scheduler_cls,
        "update_from_output",
        None,
    )

    def _wrapped_schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _run_pd_scheduler_hook(
            self,
            original,
            args,
            kwargs,
            log,
        )

    def _wrapped_add_request(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        if not (original_add_request is not None):
            raise RuntimeError("Required invariant failed: original_add_request is not None")
        _record_scheduler(self)
        # Hook the concrete subclass, not every instance of the scheduler base.
        runtime_scheduler_cls = type(self)
        if runtime_scheduler_cls is not scheduler_cls:
            _install_scheduler_preemption_hook(runtime_scheduler_cls, log)
            _install_runtime_scheduler_schedule_hook(
                runtime_scheduler_cls,
                log,
                base_scheduler_cls=scheduler_cls,
            )
        _record_pd_request_block_hashes(
            request,
            block_size=int(getattr(self, "block_size", 0) or 0),
        )
        request_id = str(getattr(request, "request_id", "") or "")
        source_request_seen_event: tuple[str, str, int] | None = None
        with _VLLM_BLOCK_POOL_LOCK:
            source_intent = next(
                (
                    _PD_SOURCE_EXPORT_INTENTS[alias]
                    for alias in _pd_request_id_aliases(request_id)
                    if alias in _PD_SOURCE_EXPORT_INTENTS
                ),
                None,
            )
            if source_intent is not None and source_intent.scheduler_seen_at <= 0:
                source_intent.scheduler_seen_at = time.monotonic()
                (
                    source_intent.worker_request_received_to_scheduler_seen_ms,
                    source_intent.worker_request_received_timing_source,
                ) = _request_received_to_scheduler_seen_timing(
                    request,
                    scheduler_seen_monotonic=source_intent.scheduler_seen_at,
                )
                source_intent.scheduler_seen_cycle_id = int(getattr(self, _PD_SCHEDULER_CYCLE_ATTR, 0) or 0)
                source_request_seen_event = (
                    source_intent.handoff_id,
                    source_intent.request_id,
                    source_intent.scheduler_seen_cycle_id,
                )
        result = original_add_request(self, request, *args, **kwargs)
        if source_request_seen_event is not None:
            handoff_id, source_request_id, scheduler_seen_cycle_id = source_request_seen_event
            _PD_CONTROL_EVENTS.publish(
                "P_REQUEST_SEEN",
                handoff_id=handoff_id,
                request_id=source_request_id,
                result={
                    "engine_request_id": request_id,
                    "scheduler_seen_cycle_id": scheduler_seen_cycle_id,
                },
            )
            log.info(
                "Jiuwen KVBM P request entered scheduler: handoff_id=%s "
                "request_id=%s engine_request_id=%s scheduler_seen_cycle_id=%s",
                handoff_id,
                source_request_id,
                request_id,
                scheduler_seen_cycle_id,
            )
        if source_intent is not None:
            with _VLLM_BLOCK_POOL_LOCK:
                source_intent.scheduler_running_request_count = _scheduler_queue_length(getattr(self, "running", None))
                source_intent.scheduler_waiting_request_count = _scheduler_queue_length(getattr(self, "waiting", None))
        observed_gate: tuple[str, str] | None = None
        target_gate: _PDTargetRequestGate | None = None
        with _PD_TARGET_LOCK:
            gate = next(
                (_PD_TARGET_GATES[alias] for alias in _pd_request_id_aliases(request_id) if alias in _PD_TARGET_GATES),
                None,
            )
            if gate is not None:
                gate.request_seen = True
                gate.engine_request_id = request_id
                gate.request_seen_at = time.monotonic()
                observed_gate = (gate.handoff_id, gate.request_id)
                target_gate = gate
                log.info(
                    "Jiuwen KVBM D request entered PD import gate: handoff_id=%s request_id=%s engine_request_id=%s",
                    gate.handoff_id,
                    gate.request_id,
                    gate.engine_request_id,
                )
                if _pd_prompt_diagnostics_enabled():
                    tokens = _diagnostic_prompt_token_ids(request)
                    log.info(
                        "[PDTargetPromptDiagnostics] handoff_id=%s engine_request_id=%s token_count=%d token_sha256=%s",
                        gate.handoff_id,
                        request_id,
                        len(tokens),
                        _pd_token_digest(tokens),
                    )
        if target_gate is not None:
            try:
                if _pd_block_ownership_mode() is SchedulerKVOwnershipMode.SCHEDULER:
                    allocation = _scheduler_owned_pd_target_allocation(
                        self,
                        request,
                        handoff_id=target_gate.handoff_id,
                    )
                    with _PD_TARGET_LOCK:
                        current = _PD_TARGET_GATES.get(target_gate.request_id)
                        if current is not target_gate:
                            getattr(self, "kv_cache_manager").free(request)
                            allocation.released = True
                            raise RuntimeError("D request gate disappeared during Scheduler allocation")
                        target_gate.scheduler_allocation = allocation
                        target_gate.preplan_error = ""
                        target_gate.preplan_capacity_waiting = False
                    log.info(
                        "Jiuwen KVBM D Scheduler-owned blocks allocated: "
                        "handoff_id=%s request_id=%s total_blocks=%s "
                        "reused_blocks=%s transfer_blocks=%s allocation_ms=%.3f",
                        target_gate.handoff_id,
                        target_gate.request_id,
                        len(allocation.planned_destination_blocks),
                        len(allocation.planned_destination_blocks) - len(allocation.transfer_positions),
                        len(allocation.transfer_positions),
                        allocation.allocation_ms,
                    )
                else:
                    preplan = _scheduler_preplan_pd_import(
                        self,
                        request,
                        owner_id=target_gate.handoff_id,
                    )
                    with _PD_TARGET_LOCK:
                        current = _PD_TARGET_GATES.get(target_gate.request_id)
                        if current is target_gate:
                            target_gate.preplan = preplan
                            target_gate.preplan_error = ""
                            target_gate.preplan_capacity_waiting = False
                        else:
                            _rollback_pd_target_preplan(
                                preplan,
                                reason="gate disappeared before preplan publication",
                            )
                    log.info(
                        "Jiuwen KVBM D delta import preplanned: handoff_id=%s "
                        "request_id=%s total_blocks=%s transfer_blocks=%s "
                        "hash_prepare_ms=%.3f reserve_ms=%.3f",
                        target_gate.handoff_id,
                        target_gate.request_id,
                        preplan.result.get("total_blocks"),
                        preplan.result.get("transfer_blocks"),
                        preplan.hash_prepare_ms,
                        preplan.reserve_ms,
                    )
            except BlockHashesInFlightError as exc:
                with _PD_TARGET_LOCK:
                    target_gate.preplan_error = str(exc)
                log.info(
                    "Jiuwen KVBM D delta preplan joined in-flight hashes: "
                    "handoff_id=%s request_id=%s hash_count=%s leader_count=%s",
                    target_gate.handoff_id,
                    target_gate.request_id,
                    exc.hash_count,
                    len(exc.reservation_ids),
                )
            except BlockReservationCapacityError as exc:
                with _PD_TARGET_LOCK:
                    target_gate.preplan_error = str(exc)
                    target_gate.preplan_capacity_waiting = True
                    target_gate.last_capacity_retry_generation = _scheduler_block_pool_capacity_generation(self)
                log.info(
                    "Jiuwen KVBM D delta preplan queued for BlockPool capacity: "
                    "handoff_id=%s request_id=%s requested=%s available=%s reused=%s",
                    target_gate.handoff_id,
                    target_gate.request_id,
                    exc.requested,
                    exc.available,
                    exc.reused,
                )
            except Exception as exc:
                with _PD_TARGET_LOCK:
                    target_gate.preplan_error = str(exc)
                    target_gate.preplan_capacity_waiting = False
                log.warning(
                    "Jiuwen KVBM D delta import preplan unavailable; "
                    "synchronous plan will be used: handoff_id=%s request_id=%s "
                    "error=%s",
                    target_gate.handoff_id,
                    target_gate.request_id,
                    exc,
                )
        if observed_gate is not None:
            preplan_error = target_gate.preplan_error if target_gate is not None else ""
            scheduler_allocation = target_gate.scheduler_allocation if target_gate is not None else None
            _PD_CONTROL_EVENTS.publish(
                "REQUEST_SEEN",
                handoff_id=observed_gate[0],
                request_id=observed_gate[1],
                result={
                    "preplan_ready": bool(
                        target_gate is not None
                        and (target_gate.preplan is not None or target_gate.scheduler_allocation is not None)
                    ),
                    "preplan_error": preplan_error or None,
                    "capacity_unavailable": False,
                    "capacity_waiting": bool(target_gate is not None and target_gate.preplan_capacity_waiting),
                    "scheduler_allocation_ready": scheduler_allocation is not None,
                    "allocated_blocks": (
                        len(scheduler_allocation.planned_destination_blocks) if scheduler_allocation is not None else 0
                    ),
                },
            )
        return result

    def _wrapped_update_from_output(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not (original_update_from_output is not None):
            raise RuntimeError("Required invariant failed: original_update_from_output is not None")
        scheduler_output = args[0] if args else kwargs.get("scheduler_output")
        model_output_received_at = time.monotonic()
        tracked = _pd_d_execution_timings_for_output(scheduler_output)
        update_started = time.perf_counter()
        try:
            return original_update_from_output(self, *args, **kwargs)
        finally:
            update_ms = (time.perf_counter() - update_started) * 1000.0
            _log_pd_d_model_output_timing(
                tracked,
                model_output_received_at=model_output_received_at,
                scheduler_update_ms=update_ms,
                log=log,
            )

    setattr(scheduler_cls, _SCHEDULER_DIAGNOSTICS_ORIGINAL_ATTR, original)
    setattr(scheduler_cls, "schedule", _wrapped_schedule)
    if original_add_request is not None:
        setattr(
            scheduler_cls,
            _SCHEDULER_ADD_REQUEST_ORIGINAL_ATTR,
            original_add_request,
        )
        setattr(scheduler_cls, "add_request", _wrapped_add_request)
    if original_update_from_output is not None:
        setattr(
            scheduler_cls,
            _SCHEDULER_UPDATE_OUTPUT_ORIGINAL_ATTR,
            original_update_from_output,
        )
        setattr(
            scheduler_cls,
            "update_from_output",
            _wrapped_update_from_output,
        )
    else:
        log.warning("Jiuwen KVBM D model-output timing hook unavailable: Scheduler.update_from_output is missing")
    setattr(scheduler_cls, _SCHEDULER_DIAGNOSTICS_PATCH_ATTR, True)
    log.info(
        "Jiuwen KVBM scheduler cache-hit diagnostics hook installed on %s; d_model_output_timing=%s",
        scheduler_cls,
        original_update_from_output is not None,
    )
    return True


def _install_runtime_scheduler_schedule_hook(
    scheduler_cls: type[Any],
    log: logging.Logger,
    *,
    base_scheduler_cls: type[Any],
) -> bool:
    """为重写 `schedule()` 的实际 Scheduler 类型安装 attach 处理挂点。

    参数:
        scheduler_cls: `Scheduler.add_request()` 观察到的运行时类型，例如
            vLLM `AsyncScheduler` 或 vLLM-Ascend 自定义 Scheduler。
        log: 用于输出安装状态和 attach 诊断的日志对象。
        base_scheduler_cls: 已安装基础 schedule hook 的 vLLM Scheduler 类型。

    返回:
        运行时类型自定义了 schedule 且成功安装时返回 True；继承的 schedule
        已由基类 hook 覆盖、缺少方法或已经安装时返回 False。
    """

    schedule_owner = next(
        (candidate for candidate in scheduler_cls.__mro__ if "schedule" in candidate.__dict__),
        None,
    )
    if schedule_owner is None or schedule_owner is base_scheduler_cls:
        return False
    if schedule_owner.__dict__.get(_RUNTIME_SCHEDULER_PATCH_ATTR, False):
        return False
    original = schedule_owner.__dict__["schedule"]

    def _wrapped_runtime_schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _run_pd_scheduler_hook(
            self,
            original,
            args,
            kwargs,
            log,
        )

    setattr(schedule_owner, _RUNTIME_SCHEDULER_ORIGINAL_ATTR, original)
    setattr(schedule_owner, "schedule", _wrapped_runtime_schedule)
    setattr(schedule_owner, _RUNTIME_SCHEDULER_PATCH_ATTR, True)
    log.info(
        "Jiuwen KVBM runtime scheduler attach hook installed on %s",
        schedule_owner,
    )
    return True


def _install_new_request_data_diagnostics_hook(
    new_request_data_cls: type[Any],
    log: logging.Logger,
) -> bool:
    """安装 NewRequestData.from_request() 请求诊断主挂点。

    参数:
        new_request_data_cls: 待 hook 的 vLLM NewRequestData 类。
        log: 用于输出安装状态和请求诊断的日志对象。

    返回:
        安装成功或已经安装时返回 True；缺少 from_request 方法时返回 False。
    """

    if getattr(new_request_data_cls, _NEW_REQUEST_DIAGNOSTICS_PATCH_ATTR, False):
        log.info("Jiuwen KVBM request-data cache-hit diagnostics hook already installed")
        return True
    original = getattr(new_request_data_cls, "from_request", None)
    if original is None:
        log.warning("Jiuwen KVBM request-data cache-hit diagnostics hook skipped: from_request is missing")
        return False

    def _wrapped_from_request(
        cls: type[Any],
        request: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        _record_pd_request_block_hashes(request)
        result = original(request, *args, **kwargs)
        _log_new_request_cache_hit_diagnostics(result, log)
        return result

    setattr(new_request_data_cls, _NEW_REQUEST_DIAGNOSTICS_ORIGINAL_ATTR, original)
    setattr(new_request_data_cls, "from_request", classmethod(_wrapped_from_request))
    setattr(new_request_data_cls, _NEW_REQUEST_DIAGNOSTICS_PATCH_ATTR, True)
    log.info(
        "Jiuwen KVBM request-data cache-hit diagnostics hook installed on %s",
        new_request_data_cls,
    )
    return True


def _pd_scheduler_has_pending_work() -> bool:
    """判断 Scheduler hook 是否存在必须同步处理的 PD 工作。

    返回:
        P source intent、D gate、plan、attach 或待确认的 fast-path记录任一
        存在时返回True；全部为空时返回False。

    说明:
        这里只进行 CPython 原子容器的只读布尔检查，避免 Decode 每个调度周期
        为四个空容器获取全局锁。控制线程刚提交任务时即使与本次检查交错，
        下一调度周期也会观察到任务，而控制接口会继续等待 completed 事件。
    """

    return bool(
        any(not intent.capture_complete for intent in _PD_SOURCE_EXPORT_INTENTS.values())
        or _PD_TARGET_GATES
        or _PD_TARGET_PLAN_TASKS
        or _PD_TARGET_ATTACH_TASKS
        or _PD_FAST_PATH_PENDING
    )


def _pd_scheduler_timing_sample_due(scheduler: Any) -> bool:
    """按固定调度周期对空闲快速路径进行低开销采样。

    参数:
        scheduler: 当前 vLLM Scheduler 实例，用于保存采样计数。

    返回:
        当前周期需要输出阶段耗时日志时返回 True，否则返回 False。
    """

    counter = int(getattr(scheduler, _PD_SCHEDULER_SAMPLE_COUNTER_ATTR, 0)) + 1
    setattr(scheduler, _PD_SCHEDULER_SAMPLE_COUNTER_ATTR, counter)
    interval = max(int(_PD_SCHEDULER_IDLE_SAMPLE_EVERY), 1)
    return counter == 1 or counter % interval == 0


def _log_pd_scheduler_hook_timing(
    log: logging.Logger,
    *,
    active_pd: bool,
    pd_hook_total_us: float,
    original_schedule_us: float,
    gate_scan_us: float,
    task_scan_us: float,
) -> None:
    """输出一次 Scheduler hook 分阶段采样耗时。

    参数:
        log: 接收阶段耗时日志的 logger。
        active_pd: 当前周期是否存在 gate 或 PD 任务。
        pd_hook_total_us: KVBM 包装函数的总墙钟耗时，包含原始调度。
        original_schedule_us: vLLM 原始 ``schedule()`` 的耗时。
        gate_scan_us: gate 过期检查、移除和恢复等待请求的总耗时。
        task_scan_us: plan 与 attach 任务提取和处理的总耗时。

    返回:
        无返回值。
    """

    log.info(
        "Jiuwen KVBM scheduler hook timing: active_pd=%s idle_fast_path=%s "
        "pd_hook_total_us=%.3f original_schedule_us=%.3f "
        "gate_scan_us=%.3f task_scan_us=%.3f",
        active_pd,
        not active_pd,
        pd_hook_total_us,
        original_schedule_us,
        gate_scan_us,
        task_scan_us,
    )


def _run_pd_scheduler_hook(
    scheduler: Any,
    original_schedule: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    log: logging.Logger,
) -> Any:
    """执行带空闲快速路径的 vLLM Scheduler 包装逻辑。

    参数:
        scheduler: 当前 vLLM Scheduler 实例。
        original_schedule: 安装 hook 前保存的原始 ``schedule()`` 方法。
        args: 调用原始方法的位置参数。
        kwargs: 调用原始方法的关键字参数。
        log: 输出请求诊断和采样耗时的 logger。

    返回:
        原始 ``schedule()`` 返回的 SchedulerOutput 或兼容对象。
    """

    _record_scheduler(scheduler)
    active_pd = _pd_scheduler_has_pending_work()
    sample_timing = _pd_scheduler_timing_sample_due(scheduler)
    total_started_ns = time.perf_counter_ns() if sample_timing else 0

    if not active_pd:
        original_started_ns = time.perf_counter_ns() if sample_timing else 0
        output = original_schedule(scheduler, *args, **kwargs)
        original_schedule_us = (time.perf_counter_ns() - original_started_ns) / 1000.0 if sample_timing else 0.0
        _log_scheduler_cache_hit_diagnostics(output, log)
        _observe_pd_fallback_schedule(scheduler, output, log)
        if sample_timing:
            _log_pd_scheduler_hook_timing(
                log,
                active_pd=False,
                pd_hook_total_us=(time.perf_counter_ns() - total_started_ns) / 1000.0,
                original_schedule_us=original_schedule_us,
                gate_scan_us=0.0,
                task_scan_us=0.0,
            )
        return output

    cycle_started_at = time.monotonic()
    previous_cycle_at = float(getattr(scheduler, _PD_SCHEDULER_LAST_CYCLE_AT_ATTR, 0.0) or 0.0)
    setattr(scheduler, _PD_SCHEDULER_LAST_CYCLE_AT_ATTR, cycle_started_at)
    scheduler_cycle_ms = max(0.0, (cycle_started_at - previous_cycle_at) * 1000.0) if previous_cycle_at > 0 else 0.0
    cycle_id = _next_pd_scheduler_cycle(scheduler)
    task_started_ns = time.perf_counter_ns() if sample_timing else 0
    _retry_scheduler_owned_target_allocations(scheduler, log)
    if _PD_TARGET_PLAN_TASKS:
        _process_pd_target_plan_tasks(scheduler, log)
    fast_path_records = _process_pd_target_attach_tasks(
        scheduler,
        log,
        cycle_id=cycle_id,
    )
    task_scan_us = (time.perf_counter_ns() - task_started_ns) / 1000.0 if sample_timing else 0.0

    gate_started_ns = time.perf_counter_ns() if sample_timing else 0
    gated_requests = _remove_gated_pd_requests_from_queues(scheduler)
    gate_scan_us = (time.perf_counter_ns() - gate_started_ns) / 1000.0 if sample_timing else 0.0
    original_started_ns = time.perf_counter_ns() if sample_timing else 0
    try:
        output = original_schedule(scheduler, *args, **kwargs)
    finally:
        original_schedule_us = (time.perf_counter_ns() - original_started_ns) / 1000.0 if sample_timing else 0.0
        restore_started_ns = time.perf_counter_ns() if sample_timing else 0
        _restore_gated_pd_requests(scheduler, gated_requests)
        if sample_timing:
            gate_scan_us += (time.perf_counter_ns() - restore_started_ns) / 1000.0
    _log_pd_fast_path_schedule_result(
        output,
        fast_path_records,
        log,
        cycle_id=cycle_id,
    )
    _update_pd_source_schedule_diagnostics(
        scheduler,
        output,
        log,
        cycle_id=cycle_id,
        scheduler_cycle_ms=scheduler_cycle_ms,
        original_schedule_ms=original_schedule_us / 1000.0,
        log_cycle=sample_timing,
    )
    _log_scheduler_cache_hit_diagnostics(output, log)
    _observe_pd_fallback_schedule(scheduler, output, log)
    if sample_timing:
        _log_pd_scheduler_hook_timing(
            log,
            active_pd=True,
            pd_hook_total_us=(time.perf_counter_ns() - total_started_ns) / 1000.0,
            original_schedule_us=original_schedule_us,
            gate_scan_us=gate_scan_us,
            task_scan_us=task_scan_us,
        )
    return output


def _record_scheduler(scheduler: Any) -> None:
    """记录当前进程观察到的 vLLM Scheduler 实例。

    参数:
        scheduler: vLLM Scheduler 或字段兼容对象。

    返回:
        无返回值；同一实例只记录一次。
    """

    if getattr(scheduler, _PD_SCHEDULER_RECORDED_ATTR, False):
        return
    with _PD_TARGET_LOCK:
        if all(existing is not scheduler for existing in _OBSERVED_SCHEDULERS):
            _OBSERVED_SCHEDULERS.append(scheduler)
        setattr(scheduler, _PD_SCHEDULER_RECORDED_ATTR, True)


def _remove_gated_pd_requests_from_queues(scheduler: Any) -> list[Any]:
    """在本轮调度前临时移除仍等待D import的请求。

    参数:
        scheduler: 当前执行 schedule() 的 vLLM Scheduler。

    返回:
        本轮移除的请求列表；这些请求会在schedule返回后重新排队，使
        EngineCore保持可唤醒并处理控制线程随后提交的plan/attach任务。
    """

    requests = getattr(scheduler, "requests", None)
    if not isinstance(requests, dict):
        return []
    expired_gates: list[_PDTargetRequestGate] = []
    with _PD_TARGET_LOCK:
        now = time.monotonic()
        expired_ids = [request_id for request_id, gate in _PD_TARGET_GATES.items() if gate.expires_at <= now]
        for request_id in expired_ids:
            expired_gate = _PD_TARGET_GATES.pop(request_id, None)
            if expired_gate is not None:
                expired_gates.append(expired_gate)
            logger.warning(
                "Jiuwen KVBM D request import gate expired; falling back to local prefill: request_id=%s",
                request_id,
            )
        gated: list[Any] = []
        for gate in _PD_TARGET_GATES.values():
            engine_request_id = gate.engine_request_id or gate.request_id
            if engine_request_id in requests:
                gated.append(requests[engine_request_id])
    for expired_gate in expired_gates:
        with _VLLM_BLOCK_POOL_LOCK:
            _release_pd_decode_admission_ticket_locked(
                expired_gate.handoff_id,
                ticket_id=expired_gate.admission_ticket_id,
            )
        _fail_pending_scheduler_import(
            expired_gate.handoff_id,
            error="D request import gate expired while waiting for Scheduler capacity",
        )
        _rollback_pd_target_preplan(
            expired_gate.preplan,
            reason="D request import gate expired",
        )
        _release_scheduler_owned_pd_allocation(
            expired_gate,
            reason="D request import gate expired",
        )
        if _pd_block_ownership_mode() is SchedulerKVOwnershipMode.LEGACY:
            _process_block_reservations().release_pending_delta(expired_gate.handoff_id)
    if not gated:
        return []
    for queue_name in ("waiting", "skipped_waiting"):
        request_queue = getattr(scheduler, queue_name, None)
        remove_requests = getattr(request_queue, "remove_requests", None)
        if callable(remove_requests):
            remove_requests(gated)
    return gated


def _restore_gated_pd_requests(scheduler: Any, requests: list[Any]) -> None:
    """把本轮临时移除且仍受门控的 D 请求放回等待队列。

    参数:
        scheduler: 当前 vLLM Scheduler。
        requests: `_remove_gated_pd_requests_from_queues` 返回的请求列表。

    返回:
        无返回值；已经 attach 或已经结束的请求不会重新排队。
    """

    waiting = getattr(scheduler, "waiting", None)
    prepend_request = getattr(waiting, "prepend_request", None)
    if not callable(prepend_request):
        return
    scheduler_requests = getattr(scheduler, "requests", {})
    with _PD_TARGET_LOCK:
        remaining_ids = {gate.engine_request_id or gate.request_id for gate in _PD_TARGET_GATES.values()}
    for request in reversed(requests):
        request_id = str(getattr(request, "request_id", "") or "")
        if request_id in remaining_ids and request_id in scheduler_requests:
            prepend_request(request)


def _pd_block_group_signature(
    block_groups: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> tuple[tuple[int, int, bool, int], ...]:
    """提取不依赖物理block ID的PD逻辑布局签名。

    参数:
        block_groups: P export或D预计划生成的block/group描述序列。

    返回:
        由逻辑位置、group、partial标记和有效token数组成的稳定元组。
    """

    return tuple(
        (
            int(item.get("logical_index", -1)),
            int(item.get("group_id", -1)),
            bool(item.get("request_scoped_partial", False)),
            int(item.get("valid_token_count", 0)),
        )
        for item in block_groups
    )


def _pd_delta_plan_result(
    *,
    request_id: str,
    target_packed_hashes: tuple[bytes | None, ...] | list[bytes | None],
    reservation: Any,
    reserve_ms: float,
) -> dict[str, Any]:
    """把delta reservation转换为控制面统一响应。

    参数:
        request_id: D scheduler中的真实请求ID。
        target_packed_hashes: 完整目标逻辑布局对应的packed hash。
        reservation: ``reserve_delta``返回的BlockReservation。
        reserve_ms: BlockPool pin和预留耗时，单位毫秒。

    返回:
        包含复用、传输位置、目标block布局和耗时的JSON安全字典。
    """

    snapshot = reservation.to_dict()
    return {
        "ok": True,
        "request_id": request_id,
        "reservation": snapshot,
        "total_blocks": len(target_packed_hashes),
        "reused_blocks": len(target_packed_hashes) - len(reservation.raw_blocks),
        "transfer_blocks": len(reservation.raw_blocks),
        "transfer_positions": list(reservation.transfer_positions),
        "planned_destination_blocks": snapshot["planned_block_ids"],
        "transfer_destination_blocks": snapshot["block_ids"],
        "reserve_ms": reserve_ms,
        "delta_reserve_ms": reserve_ms,
        "free_blocks_after_reserve": _process_block_reservations().stats().get("free_blocks"),
    }


def _scheduler_owned_pd_target_allocation(
    scheduler: Any,
    request: Any,
    *,
    handoff_id: str,
) -> _PDSchedulerTargetAllocation:
    """让vLLM Scheduler为D请求原生分配外部KV目标block。

    参数:
        scheduler: 当前D Worker的vLLM Scheduler实例。
        request: 已加入Scheduler、尚未开始本地计算的D请求。
        handoff_id: Router为当前PD事务生成的handoff ID。

    返回:
        包含请求完整block table、稀疏传输位置和token进度的Scheduler分配记录。

    异常:
        请求已经执行、缺少hash或Scheduler容量不足时抛出RuntimeError或
        BlockReservationCapacityError；成功分配后构造布局失败会自动归还请求block。
    """

    if int(getattr(request, "num_computed_tokens", 0)) != 0:
        raise RuntimeError("D request already started local computation before allocation")
    if request in list(getattr(scheduler, "running", ())):
        raise RuntimeError("D request is already running before scheduler allocation")
    started = time.perf_counter()
    block_size = int(getattr(scheduler, "block_size", 0) or 0)
    if block_size <= 0:
        raise RuntimeError("D scheduler block_size is not available")
    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    if kv_cache_manager is None:
        raise RuntimeError("D scheduler KVCacheManager is not available")
    group_count = int(getattr(kv_cache_manager, "num_kv_cache_groups", 0) or 0)
    if group_count <= 0:
        raise RuntimeError("D scheduler reports no KV cache groups")
    prompt_token_ids = _diagnostic_prompt_token_ids(request)
    if not prompt_token_ids:
        raise RuntimeError("D request prompt tokens are not available")
    request_hashes = tuple(_flatten_request_block_hashes(getattr(request, "block_hashes", None)))
    full_logical_blocks = len(prompt_token_ids) // block_size
    if len(request_hashes) < full_logical_blocks:
        raise RuntimeError("D request does not expose all complete prompt hashes")
    partial_valid_tokens = len(prompt_token_ids) % block_size
    include_partial = partial_valid_tokens > 0 and _pd_request_supports_token_fingerprint(request)
    target_packed_hashes: list[bytes | None] = []
    block_groups: list[dict[str, Any]] = []
    for logical_index in range(full_logical_blocks):
        request_hash = request_hashes[logical_index]
        if not isinstance(request_hash, bytes | bytearray):
            raise RuntimeError("D request block hash type is not compatible")
        for group_id in range(group_count):
            packed_hash = bytes(request_hash) + group_id.to_bytes(4, "big")
            target_packed_hashes.append(packed_hash)
            block_groups.append(
                {
                    "logical_index": logical_index,
                    "group_id": group_id,
                    "request_scoped_partial": False,
                    "block_hash": _encode_packed_block_hash(packed_hash),
                }
            )
    if include_partial:
        for group_id in range(group_count):
            target_packed_hashes.append(None)
            block_groups.append(
                {
                    "logical_index": full_logical_blocks,
                    "group_id": group_id,
                    "request_scoped_partial": True,
                    "valid_token_count": partial_valid_tokens,
                    "block_hash": None,
                }
            )
    if not block_groups:
        raise RuntimeError("D request has no transferable prompt blocks")

    available_tokens = full_logical_blocks * block_size
    if include_partial:
        available_tokens += partial_valid_tokens
    request_num_tokens = int(getattr(request, "num_tokens", len(prompt_token_ids)))
    available_tokens = min(available_tokens, request_num_tokens)
    computed_blocks, local_computed_tokens = kv_cache_manager.get_computed_blocks(request)
    local_computed_tokens = min(int(local_computed_tokens), available_tokens)
    external_computed_tokens = max(0, available_tokens - local_computed_tokens)
    allocation_new_tokens = 0 if external_computed_tokens > 0 else 1
    allocated = kv_cache_manager.allocate_slots(
        request,
        allocation_new_tokens,
        num_new_computed_tokens=local_computed_tokens,
        new_computed_blocks=computed_blocks,
        num_external_computed_tokens=external_computed_tokens,
        delay_cache_blocks=True,
    )
    if allocated is None:
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        available_blocks = int(getattr(block_pool, "get_num_free_blocks", lambda: 0)())
        required_blocks = max(
            1,
            len(block_groups) - (local_computed_tokens // block_size) * group_count,
        )
        raise BlockReservationCapacityError(
            required_blocks,
            available_blocks,
            len(block_groups) - required_blocks,
            owner_id=handoff_id,
        )

    try:
        grouped_block_ids = kv_cache_manager.get_block_ids(request.request_id)
        planned_destination_blocks: list[int] = []
        transfer_positions: list[int] = []
        transfer_destination_blocks: list[int] = []
        local_logical_blocks = local_computed_tokens // block_size
        for position, item in enumerate(block_groups):
            group_id = int(item["group_id"])
            logical_index = int(item["logical_index"])
            group_block_ids = grouped_block_ids[group_id]
            if logical_index >= len(group_block_ids):
                raise RuntimeError("vLLM Scheduler block table is shorter than the PD target layout")
            block_id = int(group_block_ids[logical_index])
            planned_destination_blocks.append(block_id)
            if logical_index >= local_logical_blocks:
                transfer_positions.append(position)
                transfer_destination_blocks.append(block_id)
    except Exception:
        kv_cache_manager.free(request)
        with _VLLM_BLOCK_POOL_LOCK:
            _release_pd_decode_admission_ticket_locked(handoff_id)
        raise

    ready_computed_tokens = min(available_tokens, max(0, request_num_tokens - 1))
    allocation = _PDSchedulerTargetAllocation(
        handoff_id=handoff_id,
        request_id=str(getattr(request, "request_id", "")),
        target_packed_hashes=tuple(target_packed_hashes),
        block_groups=tuple(block_groups),
        planned_destination_blocks=tuple(planned_destination_blocks),
        transfer_positions=tuple(transfer_positions),
        transfer_destination_blocks=tuple(transfer_destination_blocks),
        local_computed_tokens=local_computed_tokens,
        external_computed_tokens=external_computed_tokens,
        ready_computed_tokens=ready_computed_tokens,
        full_computed_tokens=full_logical_blocks * block_size,
        partial_valid_tokens=partial_valid_tokens if include_partial else 0,
        created_at=time.monotonic(),
        allocation_ms=(time.perf_counter() - started) * 1000.0,
        capacity_generation=int(
            getattr(
                getattr(kv_cache_manager, "block_pool", None),
                _BLOCK_POOL_CAPACITY_GENERATION_ATTR,
                -1,
            )
        ),
    )
    with _VLLM_BLOCK_POOL_LOCK:
        released_ticket = _release_pd_decode_admission_ticket_locked(handoff_id)
    if released_ticket:
        logger.info(
            "Jiuwen KVBM D admission ticket consumed after Scheduler allocation: handoff_id=%s request_id=%s",
            handoff_id,
            getattr(request, "request_id", ""),
        )
    return allocation


def _scheduler_owned_pd_plan_result(
    allocation: _PDSchedulerTargetAllocation,
) -> dict[str, Any]:
    """把Scheduler原生目标布局转换为现有PD传输协议响应。

    参数:
        allocation: vLLM Scheduler为D请求持有的目标block分配记录。

    返回:
        与legacy delta plan兼容的JSON字典，同时标记scheduler_owned=True。
    """

    reservation_id = f"scheduler:{allocation.handoff_id}"
    return {
        "ok": True,
        "request_id": allocation.request_id,
        "ownership_mode": SchedulerKVOwnershipMode.SCHEDULER.value,
        "scheduler_owned": True,
        "reservation": {
            "reservation_id": reservation_id,
            "block_ids": list(allocation.transfer_destination_blocks),
            "state": "request_owned",
            "owner_id": allocation.request_id,
        },
        "planned_destination_blocks": list(allocation.planned_destination_blocks),
        "transfer_positions": list(allocation.transfer_positions),
        "total_blocks": len(allocation.planned_destination_blocks),
        "reused_blocks": len(allocation.planned_destination_blocks) - len(allocation.transfer_positions),
        "transfer_blocks": len(allocation.transfer_positions),
        "reserve_ms": allocation.allocation_ms,
        "delta_reserve_ms": allocation.allocation_ms,
        "delta_preplanned": True,
        "delta_plan_task_queue_ms": 0.0,
        "delta_plan_execution_ms": 0.0,
        "delta_preplan_age_ms": max(
            0.0,
            (time.monotonic() - allocation.created_at) * 1000.0,
        ),
    }


def _release_scheduler_owned_pd_allocation(
    gate: _PDTargetRequestGate,
    *,
    reason: str,
) -> bool:
    """通过vLLM KVCacheManager归还未提交的D请求block。

    参数:
        gate: 持有scheduler_allocation的D请求门控。
        reason: 触发释放的诊断原因。

    返回:
        实际调用KVCacheManager.free()归还block时返回True；没有可释放布局或布局
        已提交时返回False。
    """

    allocation = gate.scheduler_allocation
    if allocation is None or allocation.committed or allocation.released:
        return False
    scheduler = next(
        (
            candidate
            for candidate in _OBSERVED_SCHEDULERS
            if allocation.request_id in getattr(candidate, "requests", {})
        ),
        None,
    )
    request = getattr(scheduler, "requests", {}).get(allocation.request_id) if scheduler is not None else None
    if scheduler is not None and request is not None:
        kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
        if kv_cache_manager is not None:
            try:
                block_ids = kv_cache_manager.get_block_ids(request.request_id)
                has_request_blocks = any(bool(group) for group in block_ids)
            except Exception:
                has_request_blocks = True
            if has_request_blocks:
                kv_cache_manager.free(request)
            request.num_computed_tokens = 0
            if hasattr(request, "num_external_computed_tokens"):
                request.num_external_computed_tokens = 0
    allocation.released = True
    logger.info(
        "Jiuwen KVBM released Scheduler-owned D request blocks: handoff_id=%s request_id=%s reason=%s",
        gate.handoff_id,
        allocation.request_id,
        reason,
    )
    return True


def _activate_scheduler_import_submission(handoff_id: str) -> bool:
    """在Scheduler目标block就绪后投递真实HIXL import任务。

    参数:
        handoff_id: 需要检查并激活的PD handoff ID。

    返回:
        本次成功从容量等待状态投递到import executor时返回True；事务不存在、
        已投递或Scheduler分配尚未就绪时返回False。
    """

    with _PD_IMPORT_FUTURES_LOCK:
        submission = _PD_SCHEDULER_IMPORT_SUBMISSIONS.get(handoff_id)
        if submission is None or submission.dispatched:
            return False
    with _PD_TARGET_LOCK:
        gate = _PD_TARGET_GATES.get(submission.request_id)
        allocation_ready = bool(
            gate is not None
            and gate.handoff_id == handoff_id
            and gate.scheduler_allocation is not None
            and not gate.scheduler_allocation.released
        )
    if not allocation_ready:
        return False
    with _PD_IMPORT_FUTURES_LOCK:
        current = _PD_SCHEDULER_IMPORT_SUBMISSIONS.get(handoff_id)
        if current is not submission or submission.dispatched:
            return False
        submission.dispatched = True
        dispatched_at = time.monotonic()
        import_payload = dict(submission.payload)
        import_payload["_import_dispatched_at"] = dispatched_at
        import_payload["_scheduler_capacity_wait_ms"] = max(
            0.0,
            (dispatched_at - submission.created_at) * 1000.0,
        )
        inner_future = _PD_IMPORT_EXECUTOR.submit(
            _run_pd_import_task,
            submission.control,
            submission.backend,
            import_payload,
        )

    def _copy_result(completed: Future[dict[str, Any]]) -> None:
        """把executor终态复制到对外Future，并移除容量等待记录。"""

        try:
            result = completed.result()
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        if not submission.result_future.done():
            submission.result_future.set_result(result)
        with _PD_IMPORT_FUTURES_LOCK:
            current_submission = _PD_SCHEDULER_IMPORT_SUBMISSIONS.get(handoff_id)
            if current_submission is submission:
                _PD_SCHEDULER_IMPORT_SUBMISSIONS.pop(handoff_id, None)

    inner_future.add_done_callback(_copy_result)
    logger.info(
        "Jiuwen KVBM Scheduler-owned PD import dispatched after target allocation: "
        "handoff_id=%s request_id=%s capacity_wait_ms=%.3f",
        handoff_id,
        submission.request_id,
        max(0.0, (time.monotonic() - submission.created_at) * 1000.0),
    )
    return True


def _fail_pending_scheduler_import(
    handoff_id: str,
    *,
    error: str,
) -> bool:
    """终止尚未进入HIXL executor的Scheduler容量等待事务。

    参数:
        handoff_id: 需要终止的PD handoff ID。
        error: 返回Router并写入控制事件的稳定错误文本。

    返回:
        找到且终止未投递事务时返回True；事务不存在或已经投递时返回False。
    """

    with _PD_IMPORT_FUTURES_LOCK:
        submission = _PD_SCHEDULER_IMPORT_SUBMISSIONS.get(handoff_id)
        if submission is None or submission.dispatched:
            return False
        _PD_SCHEDULER_IMPORT_SUBMISSIONS.pop(handoff_id, None)
    result = {
        "ok": False,
        "error": error,
        "handoff_id": handoff_id,
        "request_id": submission.request_id,
        "capacity_waiting": True,
    }
    if not submission.result_future.done():
        submission.result_future.set_result(result)
    _PD_CONTROL_EVENTS.publish(
        "IMPORT_FAILED",
        handoff_id=handoff_id,
        request_id=submission.request_id,
        result=result,
    )
    return True


def _scheduler_block_pool_capacity_generation(scheduler: Any) -> int:
    """读取Scheduler当前BlockPool的单调容量generation。"""

    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    block_pool = getattr(kv_cache_manager, "block_pool", None)
    return int(getattr(block_pool, _BLOCK_POOL_CAPACITY_GENERATION_ATTR, -1))


def _retry_scheduler_owned_target_allocations(
    scheduler: Any,
    log: logging.Logger,
) -> None:
    """在每个Scheduler周期重试容量不足的请求原生block分配。

    参数:
        scheduler: 当前D Worker的vLLM Scheduler实例。
        log: 用于输出分配恢复及异常诊断的日志对象。

    返回:
        无返回值；成功分配会保存到gate并激活对应异步import事务。
    """

    if _pd_block_ownership_mode() is not SchedulerKVOwnershipMode.SCHEDULER:
        return
    now = time.monotonic()
    with _PD_IMPORT_FUTURES_LOCK:
        expired_submissions = []
        for selected_submission in _PD_SCHEDULER_IMPORT_SUBMISSIONS.values():
            if (
                not selected_submission.dispatched
                and selected_submission.capacity_deadline_at > 0.0
                and (selected_submission.capacity_deadline_at <= now)
            ):
                expired_submissions.append(selected_submission)
    for submission in expired_submissions:
        if not _fail_pending_scheduler_import(
            submission.handoff_id,
            error="timed out waiting for vLLM Scheduler target blocks",
        ):
            continue
        release = _local_pd_import_release(
            {
                "request_id": submission.request_id,
                "handoff_id": submission.handoff_id,
            }
        )
        log.warning(
            "Jiuwen KVBM Scheduler-owned capacity wait expired; D request gate "
            "released for local fallback: handoff_id=%s request_id=%s "
            "capacity_wait_ms=%.3f released=%s",
            submission.handoff_id,
            submission.request_id,
            max(0.0, (now - submission.created_at) * 1000.0),
            release.get("released", False),
        )
    capacity_generation = _scheduler_block_pool_capacity_generation(scheduler)
    previous_generation = int(getattr(scheduler, _PD_SCHEDULER_CAPACITY_GENERATION_ATTR, -2))
    generation_changed = capacity_generation < 0 or capacity_generation != previous_generation
    if capacity_generation >= 0:
        setattr(
            scheduler,
            _PD_SCHEDULER_CAPACITY_GENERATION_ATTR,
            capacity_generation,
        )
    if not generation_changed:
        return
    with _PD_TARGET_LOCK:
        candidates = []
        for selected_gate in _PD_TARGET_GATES.values():
            request_pending = (
                selected_gate.request_seen
                and selected_gate.expires_at > now
                and (selected_gate.scheduler_allocation is None)
            )
            if request_pending and selected_gate.preplan_capacity_waiting:
                if capacity_generation < 0 or selected_gate.last_capacity_retry_generation != capacity_generation:
                    candidates.append(selected_gate)
    requests = getattr(scheduler, "requests", {})
    for gate in candidates:
        if capacity_generation >= 0:
            with _PD_TARGET_LOCK:
                current = _PD_TARGET_GATES.get(gate.request_id)
                if current is not gate:
                    continue
                gate.last_capacity_retry_generation = capacity_generation
        request_id = gate.engine_request_id or gate.request_id
        request = requests.get(request_id) if isinstance(requests, dict) else None
        if request is None:
            continue
        try:
            allocation = _scheduler_owned_pd_target_allocation(
                scheduler,
                request,
                handoff_id=gate.handoff_id,
            )
        except BlockReservationCapacityError as exc:
            with _PD_TARGET_LOCK:
                current = _PD_TARGET_GATES.get(gate.request_id)
                if current is gate:
                    gate.preplan_error = str(exc)
            continue
        except Exception as exc:
            with _PD_TARGET_LOCK:
                current = _PD_TARGET_GATES.get(gate.request_id)
                if current is gate:
                    gate.preplan_error = str(exc)
                    gate.preplan_capacity_waiting = False
            log.warning(
                "Jiuwen KVBM Scheduler-owned target allocation retry stopped: handoff_id=%s request_id=%s error=%s",
                gate.handoff_id,
                gate.request_id,
                exc,
            )
            continue
        with _PD_TARGET_LOCK:
            current = _PD_TARGET_GATES.get(gate.request_id)
            if current is not gate:
                getattr(scheduler, "kv_cache_manager").free(request)
                allocation.released = True
                continue
            gate.scheduler_allocation = allocation
            gate.preplan_error = ""
            gate.preplan_capacity_waiting = False
        log.info(
            "Jiuwen KVBM Scheduler-owned target allocation became ready: "
            "handoff_id=%s request_id=%s reused_blocks=%s transfer_blocks=%s "
            "allocation_ms=%.3f",
            gate.handoff_id,
            gate.request_id,
            len(allocation.planned_destination_blocks) - len(allocation.transfer_positions),
            len(allocation.transfer_positions),
            allocation.allocation_ms,
        )
        _PD_CONTROL_EVENTS.publish(
            "TARGET_BLOCKS_ALLOCATED",
            handoff_id=gate.handoff_id,
            request_id=gate.request_id,
            result={
                "scheduler_allocation_ready": True,
                "allocated_blocks": len(allocation.planned_destination_blocks),
                "transfer_blocks": len(allocation.transfer_positions),
                "capacity_generation": int(getattr(allocation, "capacity_generation", -1)),
            },
        )
        _activate_scheduler_import_submission(gate.handoff_id)


def _scheduler_preplan_pd_import(
    scheduler: Any,
    request: Any,
    *,
    reservation_ttl_ms: int = 120_000,
    owner_id: str = "",
) -> _PDTargetPreplan:
    """在D请求进入gate时提前pin命中block并预留缺失目标block。

    参数:
        scheduler: 当前D vLLM Scheduler，提供block_size和KV cache group数量。
        request: 刚加入Scheduler且尚未执行prefill的vLLM请求。
        reservation_ttl_ms: 提前预留的有效期，单位毫秒。
        owner_id: 当前 handoff ID；容量不足时用于登记唯一
            的同hash等待 leader。

    返回:
        可在P export到达后校验并直接采用的D目标预计划。

    异常:
        请求hash、block_size或cache group信息不完整时抛出RuntimeError；调用方
        会记录诊断并继续使用原有同步delta plan。
    """

    hash_started = time.perf_counter()
    block_size = int(getattr(scheduler, "block_size", 0) or 0)
    if block_size <= 0:
        raise RuntimeError("D scheduler block_size is not available for preplan")
    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    group_count = int(getattr(kv_cache_manager, "num_kv_cache_groups", 0) or 0)
    if group_count <= 0:
        raise RuntimeError("D scheduler reports no KV cache groups for preplan")
    prompt_token_ids = _diagnostic_prompt_token_ids(request)
    if not prompt_token_ids:
        raise RuntimeError("D request prompt tokens are not available for preplan")
    request_hashes = tuple(_flatten_request_block_hashes(getattr(request, "block_hashes", None)))
    full_logical_blocks = len(prompt_token_ids) // block_size
    if len(request_hashes) < full_logical_blocks:
        raise RuntimeError("D request does not expose all complete prompt hashes")
    partial_valid_tokens = len(prompt_token_ids) % block_size
    include_partial = partial_valid_tokens > 0 and _pd_request_supports_token_fingerprint(request)
    target_packed_hashes: list[bytes | None] = []
    block_groups: list[dict[str, Any]] = []
    for logical_index in range(full_logical_blocks):
        request_hash = request_hashes[logical_index]
        if not isinstance(request_hash, bytes | bytearray):
            raise RuntimeError("D request block hash type is not compatible with preplan")
        for group_id in range(group_count):
            packed_hash = bytes(request_hash) + group_id.to_bytes(4, "big")
            target_packed_hashes.append(packed_hash)
            block_groups.append(
                {
                    "logical_index": logical_index,
                    "group_id": group_id,
                    "request_scoped_partial": False,
                    "block_hash": _encode_packed_block_hash(packed_hash),
                }
            )
    if include_partial:
        for group_id in range(group_count):
            target_packed_hashes.append(None)
            block_groups.append(
                {
                    "logical_index": full_logical_blocks,
                    "group_id": group_id,
                    "request_scoped_partial": True,
                    "valid_token_count": partial_valid_tokens,
                    "block_hash": None,
                }
            )
    if not target_packed_hashes:
        raise RuntimeError("D request has no transferable prompt blocks for preplan")
    hash_prepare_ms = (time.perf_counter() - hash_started) * 1000.0
    reserve_started = time.perf_counter()
    reservation = _process_block_reservations().reserve_delta(
        target_packed_hashes,
        ttl_ms=reservation_ttl_ms,
        owner_id=owner_id,
    )
    reserve_ms = (time.perf_counter() - reserve_started) * 1000.0
    request_id = str(getattr(request, "request_id", "") or "")
    try:
        result = _pd_delta_plan_result(
            request_id=request_id,
            target_packed_hashes=target_packed_hashes,
            reservation=reservation,
            reserve_ms=reserve_ms,
        )
    except Exception:
        _process_block_reservations().rollback(reservation.reservation_id)
        raise
    return _PDTargetPreplan(
        target_packed_hashes=tuple(target_packed_hashes),
        block_groups=tuple(block_groups),
        result=result,
        created_at=time.monotonic(),
        hash_prepare_ms=hash_prepare_ms,
        reserve_ms=reserve_ms,
    )


def _rollback_pd_target_preplan(
    preplan: _PDTargetPreplan | None,
    *,
    reason: str,
) -> None:
    """回滚尚未被import事务采用的D目标预计划。

    参数:
        preplan: gate持有的可选目标预计划。
        reason: 触发回滚的诊断原因。

    返回:
        无返回值；预留已经终态时只记录DEBUG日志。
    """

    if preplan is None:
        return
    reservation = preplan.result.get("reservation") or {}
    reservation_id = str(reservation.get("reservation_id", ""))
    if not reservation_id:
        return
    try:
        _process_block_reservations().rollback(reservation_id)
        logger.info(
            "Jiuwen KVBM rolled back unused D delta preplan: reservation_id=%s reason=%s",
            reservation_id,
            reason,
        )
    except Exception as exc:
        logger.debug(
            "Jiuwen KVBM D delta preplan rollback skipped: reservation_id=%s reason=%s error=%s",
            reservation_id,
            reason,
            exc,
        )


def _process_pd_target_plan_tasks(scheduler: Any, log: logging.Logger) -> None:
    """在 scheduler 线程为 gated D 请求生成原子 delta import plan。

    参数:
        scheduler: 当前 vLLM Scheduler，提供请求语义和 KVCacheManager。
        log: 用于输出 plan 成功或失败诊断的日志对象。

    返回:
        无返回值；每个任务通过 completed 事件返回 reservation 结果。
    """

    with _PD_TARGET_LOCK:
        tasks = list(_PD_TARGET_PLAN_TASKS)
        _PD_TARGET_PLAN_TASKS.clear()
    for task in tasks:
        execution_started = time.perf_counter()
        task_queue_ms = max(
            0.0,
            (time.monotonic() - task.created_at) * 1000.0,
        )
        try:
            if task.ownership_mode is SchedulerKVOwnershipMode.SCHEDULER:
                task.result = _scheduler_plan_owned_pd_import(scheduler, task)
            else:
                task.result = _scheduler_plan_pd_import(scheduler, task)
            task.result["delta_plan_task_queue_ms"] = task_queue_ms
            task.result["delta_plan_execution_ms"] = (time.perf_counter() - execution_started) * 1000.0
            log.info(
                "Jiuwen KVBM D delta import planned: handoff_id=%s "
                "request_id=%s total_blocks=%s reused_blocks=%s "
                "transfer_blocks=%s transfer_positions=%s "
                "task_queue_ms=%.3f hash_validate_ms=%.3f "
                "reserve_ms=%.3f execution_ms=%.3f",
                task.handoff_id,
                task.request_id,
                task.result.get("total_blocks"),
                task.result.get("reused_blocks"),
                task.result.get("transfer_blocks"),
                task.result.get("transfer_positions"),
                task_queue_ms,
                float(task.result.get("delta_hash_validate_ms", 0.0)),
                float(task.result.get("delta_reserve_ms", 0.0)),
                float(task.result.get("delta_plan_execution_ms", 0.0)),
            )
        except BlockHashesInFlightError as exc:
            task.error = str(exc)
            task.inflight_reservation_ids = exc.reservation_ids
            log.info(
                "Jiuwen KVBM D delta import is waiting for in-flight hashes: "
                "handoff_id=%s request_id=%s hash_count=%s leader_count=%s",
                task.handoff_id,
                task.request_id,
                exc.hash_count,
                len(exc.reservation_ids),
            )
        except BlockReservationCapacityError as exc:
            task.error = str(exc)
            task.capacity_required_blocks = exc.requested
            log.info(
                "Jiuwen KVBM D delta import is waiting for BlockPool capacity: "
                "handoff_id=%s request_id=%s requested=%s available=%s reused=%s",
                task.handoff_id,
                task.request_id,
                exc.requested,
                exc.available,
                exc.reused,
            )
        except Exception as exc:
            task.error = str(exc)
            log.exception(
                "Jiuwen KVBM D delta import plan failed: handoff_id=%s request_id=%s",
                task.handoff_id,
                task.request_id,
            )
        finally:
            task.completed.set()


def _scheduler_plan_owned_pd_import(
    scheduler: Any,
    task: _PDTargetPlanTask,
) -> dict[str, Any]:
    """在Scheduler线程校验或补建D请求原生目标block布局。

    参数:
        scheduler: 当前vLLM Scheduler。
        task: 包含P端block语义和目标handoff ID的plan任务。

    返回:
        与legacy delta plan响应兼容、但目标block由请求自身持有的计划字典。
    """

    requests = getattr(scheduler, "requests", None)
    request = requests.get(task.request_id) if isinstance(requests, dict) else None
    if request is None:
        raise RuntimeError(f"D request is not present in scheduler: {task.request_id}")
    with _PD_TARGET_LOCK:
        gate = None
        for candidate_gate in _PD_TARGET_GATES.values():
            if candidate_gate.handoff_id != task.handoff_id:
                continue
            if (candidate_gate.engine_request_id or candidate_gate.request_id) == task.request_id:
                gate = candidate_gate
                break
    if gate is None:
        raise RuntimeError("D request import gate is not prepared")
    allocation = gate.scheduler_allocation
    if allocation is None:
        allocation = _scheduler_owned_pd_target_allocation(
            scheduler,
            request,
            handoff_id=task.handoff_id,
        )
        with _PD_TARGET_LOCK:
            current = _PD_TARGET_GATES.get(gate.request_id)
            if current is not gate:
                getattr(scheduler, "kv_cache_manager").free(request)
                allocation.released = True
                raise RuntimeError("D request gate changed during Scheduler allocation")
            gate.scheduler_allocation = allocation
    request_hashes = tuple(_flatten_request_block_hashes(getattr(request, "block_hashes", None)))
    target_packed_hashes = tuple(
        _validate_pd_target_request_hashes(
            request,
            request_hashes,
            task.block_groups,
            task.token_block_fingerprints,
            task.partial_token_fingerprint,
            task.partial_valid_tokens,
            semantic_plain_text=task.semantic_plain_text,
            block_size=int(getattr(scheduler, "block_size", 0) or 0),
        )
    )
    if target_packed_hashes != allocation.target_packed_hashes:
        raise RuntimeError("P export layout differs from Scheduler-owned D allocation")
    if _pd_block_group_signature(task.block_groups) != _pd_block_group_signature(allocation.block_groups):
        raise RuntimeError("P export block groups differ from Scheduler-owned D layout")
    result = _scheduler_owned_pd_plan_result(allocation)
    result["delta_hash_validate_ms"] = 0.0
    return result


def _scheduler_plan_pd_import(
    scheduler: Any,
    task: _PDTargetPlanTask,
) -> dict[str, Any]:
    """校验 D 请求语义，并在 BlockPool 临界区建立 delta reservation。

    参数:
        scheduler: 当前 vLLM Scheduler。
        task: 包含 P block 布局、token 指纹和 reservation TTL 的 plan 任务。

    返回:
        包含 reservation ID、完整目标布局、复用数量和稀疏传输位置的字典。
    """

    requests = getattr(scheduler, "requests", None)
    request = requests.get(task.request_id) if isinstance(requests, dict) else None
    if request is None:
        raise RuntimeError(f"D request is not present in scheduler: {task.request_id}")
    if int(getattr(request, "num_computed_tokens", 0)) != 0:
        raise RuntimeError("D request already started local computation before delta plan")
    if request in list(getattr(scheduler, "running", ())):
        raise RuntimeError("D request is already running before delta plan")

    request_hashes = tuple(_flatten_request_block_hashes(getattr(request, "block_hashes", None)))
    hash_validate_started = time.perf_counter()
    target_packed_hashes = _validate_pd_target_request_hashes(
        request,
        request_hashes,
        task.block_groups,
        task.token_block_fingerprints,
        task.partial_token_fingerprint,
        task.partial_valid_tokens,
        semantic_plain_text=task.semantic_plain_text,
        block_size=int(getattr(scheduler, "block_size", 0) or 0),
    )
    hash_validate_ms = (time.perf_counter() - hash_validate_started) * 1000.0
    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    group_count = int(getattr(kv_cache_manager, "num_kv_cache_groups", 0))
    if group_count <= 0:
        raise RuntimeError("D scheduler reports no KV cache groups")
    observed_group_ids = {int(item.get("group_id", -1)) for item in task.block_groups}
    if any(group_id < 0 or group_id >= group_count for group_id in observed_group_ids):
        raise RuntimeError("D delta plan contains an invalid cache group ID")
    if len(observed_group_ids) != group_count:
        raise RuntimeError("D delta plan does not provide every KV cache group")

    reserve_started = time.perf_counter()
    reservation_manager = _process_block_reservations()
    reservation = reservation_manager.reserve_delta(
        target_packed_hashes,
        ttl_ms=task.reservation_ttl_ms,
        owner_id=task.handoff_id,
    )
    reserve_ms = (time.perf_counter() - reserve_started) * 1000.0
    result = _pd_delta_plan_result(
        request_id=task.request_id,
        target_packed_hashes=target_packed_hashes,
        reservation=reservation,
        reserve_ms=reserve_ms,
    )
    result["delta_hash_validate_ms"] = hash_validate_ms
    return result


def _next_pd_scheduler_cycle(scheduler: Any) -> int:
    """递增并返回当前 scheduler 的 KVBM 调度周期编号。

    参数:
        scheduler: 当前 vLLM Scheduler 实例。

    返回:
        从 1 开始递增的进程内周期编号，仅用于诊断 attach 是否同周期调度。
    """

    cycle_id = int(getattr(scheduler, _PD_SCHEDULER_CYCLE_ATTR, 0)) + 1
    setattr(scheduler, _PD_SCHEDULER_CYCLE_ATTR, cycle_id)
    return cycle_id


def _request_received_to_scheduler_seen_timing(
    request: Any,
    *,
    scheduler_seen_monotonic: float,
) -> tuple[float, str]:
    """读取vLLM请求到达时间并转换为同进程耗时。

    参数:
        request: vLLM Request或字段兼容对象。
        scheduler_seen_monotonic: add_request入口采集的单调时钟时间戳。

    返回:
        ``(耗时毫秒, 来源)``。仅接受一小时内且不晚于add_request的单调
        时钟或墙钟时间；字段缺失、时钟域未知时返回``(0, "unavailable")``。
    """

    candidates: list[tuple[str, Any]] = []
    for attribute in ("arrival_time", "arrival_timestamp", "received_at"):
        candidates.append((attribute, getattr(request, attribute, None)))
    engine_request = getattr(request, "engine_core_request", None)
    if engine_request is not None:
        for attribute in ("arrival_time", "arrival_timestamp", "received_at"):
            candidates.append((f"engine_core_request.{attribute}", getattr(engine_request, attribute, None)))

    wall_now = time.time()
    for source, raw_value in candidates:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        monotonic_delta = scheduler_seen_monotonic - value
        if 0.0 <= monotonic_delta <= 3_600.0:
            return monotonic_delta * 1000.0, f"{source}:monotonic"
        wall_delta = wall_now - value
        if 0.0 <= wall_delta <= 3_600.0:
            return wall_delta * 1000.0, f"{source}:wall"
    return 0.0, "unavailable"


def _scheduler_queue_length(pending_objects: Any) -> int:
    """兼容读取vLLM scheduler队列长度。

    参数:
        queue: running列表、请求队列对象或None。

    返回:
        可读取时返回非负队列长度；未知结构返回0，诊断不得影响调度。
    """

    if pending_objects is None:
        return 0
    try:
        return max(0, len(pending_objects))
    except (TypeError, AttributeError):
        pass
    for attribute in ("request_queue", "queue", "_queue"):
        value = getattr(pending_objects, attribute, None)
        try:
            return max(0, len(value))
        except (TypeError, AttributeError):
            continue
    return 0


def _current_pd_scheduler_cycle() -> int:
    """读取当前进程已观察scheduler中的最大PD周期编号。

    返回:
        当前最大周期编号；尚未观察到scheduler时返回0。
    """

    return max(
        (int(getattr(scheduler, _PD_SCHEDULER_CYCLE_ATTR, 0) or 0) for scheduler in _OBSERVED_SCHEDULERS),
        default=0,
    )


def _promote_pd_request_for_fast_path(scheduler: Any, request_id: str) -> bool:
    """把刚完成 attach 的 D 请求移动到 waiting 队首。

    参数:
        scheduler: 当前 vLLM Scheduler。
        request_id: vLLM 使用的真实 engine request ID。

    返回:
        成功从等待队列移除并重新插入队首时返回 True；请求或队列 API
        不可用时返回 False。
    """

    requests = getattr(scheduler, "requests", None)
    request = requests.get(request_id) if isinstance(requests, dict) else None
    if request is None or request in list(getattr(scheduler, "running", ())):
        return False
    waiting = getattr(scheduler, "waiting", None)
    remove_waiting = getattr(waiting, "remove_requests", None)
    prepend_request = getattr(waiting, "prepend_request", None)
    if not callable(remove_waiting) or not callable(prepend_request):
        return False
    skipped_waiting = getattr(scheduler, "skipped_waiting", None)
    remove_skipped = getattr(skipped_waiting, "remove_requests", None)
    if callable(remove_skipped):
        remove_skipped([request])
    remove_waiting([request])
    prepend_request(request)
    return True


def _scheduled_request_ids(output: Any) -> set[str]:
    """从不同 vLLM SchedulerOutput 版本中提取本周期请求 ID。

    参数:
        output: 原始 Scheduler.schedule() 返回对象。

    返回:
        本周期获得 token budget 的请求 ID 集合。
    """

    scheduled: set[str] = set()
    token_counts = getattr(output, "num_scheduled_tokens", None)
    if isinstance(token_counts, dict):
        scheduled.update(str(request_id) for request_id in token_counts)
    for attribute in ("scheduled_new_reqs", "scheduled_cached_reqs"):
        scheduled.update(_request_ids_from_scheduler_data(getattr(output, attribute, None)))
    return scheduled


def _request_ids_from_scheduler_data(value: Any) -> set[str]:
    """兼容解析 vLLM SchedulerOutput 中的请求 ID 容器。

    参数:
        value: 单个 NewRequestData/CacheRequestData、对象列表，或 None。
            vLLM 0.18 的 CacheRequestData 是包含 ``req_ids`` 的聚合对象，
            本身不可迭代。

    返回:
        从当前值中解析出的请求 ID 集合；未知结构返回空集合。
    """

    if value is None:
        return set()
    for attribute in ("req_ids", "request_ids"):
        request_ids = getattr(value, attribute, None)
        if isinstance(request_ids, list | tuple | set):
            return {str(request_id) for request_id in request_ids if request_id}
    request_id = getattr(value, "req_id", None) or getattr(
        value,
        "request_id",
        None,
    )
    if request_id:
        return {str(request_id)}
    if isinstance(value, list | tuple | set):
        resolved: set[str] = set()
        for item in value:
            resolved.update(_request_ids_from_scheduler_data(item))
        return resolved
    return set()


def _track_pd_d_schedule_output(
    record: dict[str, Any],
    *,
    request_id: str,
    cycle_id: int,
    token_counts: dict[str, Any],
    scheduled_ids: set[str],
) -> None:
    """记录D请求首次调度结果，供模型输出回调计算执行阶段耗时。

    参数:
        record: attach阶段生成的请求诊断记录。
        request_id: 本轮Scheduler实际调度的engine request ID。
        cycle_id: 本轮KVBM调度周期编号。
        token_counts: SchedulerOutput中的逐请求token budget。
        scheduled_ids: 本轮所有获得token budget的请求ID。

    返回:
        无返回值。记录最多保留4096项，超限时清理最早记录。
    """

    timing = _PDTargetExecutionTiming(
        handoff_id=str(record.get("handoff_id", "")),
        request_id=request_id,
        attach_completed_at=float(record.get("attach_completed_at", 0.0) or 0.0),
        attach_cycle_id=int(record.get("attach_cycle_id", 0) or 0),
        running_request_count=int(record.get("running_request_count", 0) or 0),
        schedule_output_at=time.monotonic(),
        first_scheduled_cycle_id=cycle_id,
        scheduled_token_count=int(token_counts.get(request_id, 0) or 0),
        scheduled_batch_request_count=len(scheduled_ids),
        scheduled_batch_token_count=sum(int(value or 0) for value in token_counts.values()),
    )
    _PD_D_EXECUTION_TIMINGS[request_id] = timing
    if len(_PD_D_EXECUTION_TIMINGS) > 4096:
        oldest_request_id = next(iter(_PD_D_EXECUTION_TIMINGS))
        _PD_D_EXECUTION_TIMINGS.pop(oldest_request_id, None)


def _pd_d_execution_timings_for_output(
    scheduler_output: Any,
) -> list[_PDTargetExecutionTiming]:
    """取出当前模型输出对应的D首轮执行诊断记录。

    参数:
        scheduler_output: 传入Scheduler.update_from_output()的调度结果。

    返回:
        与本轮调度请求匹配的诊断记录；每条记录只消费一次。
    """

    records: list[_PDTargetExecutionTiming] = []
    for request_id in _scheduled_request_ids(scheduler_output):
        record = _PD_D_EXECUTION_TIMINGS.pop(request_id, None)
        if record is not None:
            records.append(record)
    return records


def _log_pd_d_model_output_timing(
    records: list[_PDTargetExecutionTiming],
    *,
    model_output_received_at: float,
    scheduler_update_ms: float,
    log: logging.Logger,
) -> None:
    """输出D请求从attach到首轮模型结果回到Scheduler的分阶段耗时。

    参数:
        records: 当前模型输出包含的D请求诊断记录。
        model_output_received_at: update_from_output入口的单调时钟时间戳。
        scheduler_update_ms: vLLM处理本轮模型输出的墙钟耗时。
        log: 接收诊断信息的日志对象。

    返回:
        无返回值。仅输出诊断日志，不修改vLLM请求状态。
    """

    for record in records:
        attach_to_schedule_output_ms = (
            max(
                0.0,
                (record.schedule_output_at - record.attach_completed_at) * 1000.0,
            )
            if record.attach_completed_at > 0
            else 0.0
        )
        schedule_output_to_model_output_ms = max(
            0.0,
            (model_output_received_at - record.schedule_output_at) * 1000.0,
        )
        attach_to_model_output_ms = (
            max(
                0.0,
                (model_output_received_at - record.attach_completed_at) * 1000.0,
            )
            if record.attach_completed_at > 0
            else 0.0
        )
        log.info(
            "Jiuwen KVBM D first model-output timing: handoff_id=%s "
            "request_id=%s attach_cycle_id=%s first_scheduled_cycle_id=%s "
            "attach_to_schedule_output_ms=%.3f "
            "schedule_output_to_model_output_ms=%.3f "
            "attach_to_model_output_ms=%.3f scheduler_update_ms=%.3f "
            "running_request_count=%s scheduled_token_count=%s "
            "scheduled_batch_request_count=%s scheduled_batch_token_count=%s",
            record.handoff_id,
            record.request_id,
            record.attach_cycle_id,
            record.first_scheduled_cycle_id,
            attach_to_schedule_output_ms,
            schedule_output_to_model_output_ms,
            attach_to_model_output_ms,
            scheduler_update_ms,
            record.running_request_count,
            record.scheduled_token_count,
            record.scheduled_batch_request_count,
            record.scheduled_batch_token_count,
        )


def _update_pd_source_schedule_diagnostics(
    scheduler: Any,
    output: Any,
    log: logging.Logger,
    *,
    cycle_id: int,
    scheduler_cycle_ms: float,
    original_schedule_ms: float,
    log_cycle: bool = True,
) -> None:
    """更新P请求首次获得token budget时的调度诊断。

    参数:
        scheduler: 当前P scheduler实例。
        output: 原始schedule()返回的SchedulerOutput。
        log: 输出周期诊断的logger。
        cycle_id: 当前PD调度周期编号。
        scheduler_cycle_ms: 当前周期距上一个活跃PD周期的墙钟间隔。
        original_schedule_ms: vLLM原始schedule()本轮执行耗时。
        log_cycle: 是否输出本轮汇总日志；首次调度事件始终输出。

    返回:
        无返回值。只更新仍存在的P source intent，不改变调度结果。
    """

    scheduled_ids = _scheduled_request_ids(output)
    raw_token_counts = getattr(output, "num_scheduled_tokens", None)
    token_counts = raw_token_counts if isinstance(raw_token_counts, dict) else {}
    running_count = _scheduler_queue_length(getattr(scheduler, "running", None))
    waiting_count = _scheduler_queue_length(getattr(scheduler, "waiting", None))
    unique_intents: list[_PDSourceExportIntent] = []
    observed: set[int] = set()
    for intent in _PD_SOURCE_EXPORT_INTENTS.values():
        identity = id(intent)
        if identity in observed:
            continue
        observed.add(identity)
        unique_intents.append(intent)

    if not unique_intents:
        return

    first_scheduled: list[_PDSourceExportIntent] = []
    with _VLLM_BLOCK_POOL_LOCK:
        for intent in unique_intents:
            if intent.first_scheduled_cycle_id > 0:
                continue
            aliases = set(_pd_request_id_aliases(intent.request_id))
            scheduled_request_id = next(
                (
                    request_id
                    for request_id in scheduled_ids
                    if aliases.intersection(_pd_request_id_aliases(request_id))
                ),
                None,
            )
            if scheduled_request_id is None:
                continue
            intent.first_scheduled_at = time.monotonic()
            intent.first_scheduled_cycle_id = cycle_id
            intent.scheduler_wait_cycles = max(
                0,
                cycle_id - intent.scheduler_seen_cycle_id - 1,
            )
            intent.first_scheduled_token_count = int(token_counts.get(scheduled_request_id, 0) or 0)
            intent.first_scheduled_running_request_count = running_count
            intent.first_scheduled_waiting_request_count = waiting_count
            intent.first_scheduled_cycle_ms = scheduler_cycle_ms
            first_scheduled.append(intent)

    if log_cycle:
        log.info(
            "Jiuwen KVBM P scheduler cycle diagnostics: cycle_id=%s "
            "scheduler_cycle_ms=%.3f original_schedule_ms=%.3f "
            "running_requests=%s waiting_requests=%s scheduled_tokens=%s "
            "pending_source_requests=%s",
            cycle_id,
            scheduler_cycle_ms,
            original_schedule_ms,
            running_count,
            waiting_count,
            sum(int(value or 0) for value in token_counts.values()),
            sum(not intent.capture_complete for intent in unique_intents),
        )
    for intent in first_scheduled:
        log.info(
            "Jiuwen KVBM P request first scheduled: handoff_id=%s request_id=%s "
            "scheduler_seen_cycle_id=%s first_scheduled_cycle_id=%s "
            "scheduler_wait_cycles=%s scheduled_tokens=%s "
            "scheduler_cycle_ms=%.3f running_requests=%s waiting_requests=%s",
            intent.handoff_id,
            intent.request_id,
            intent.scheduler_seen_cycle_id,
            intent.first_scheduled_cycle_id,
            intent.scheduler_wait_cycles,
            intent.first_scheduled_token_count,
            intent.first_scheduled_cycle_ms,
            intent.first_scheduled_running_request_count,
            intent.first_scheduled_waiting_request_count,
        )


def _log_pd_fast_path_schedule_result(
    output: Any,
    records: list[dict[str, Any]],
    log: logging.Logger,
    *,
    cycle_id: int,
) -> None:
    """记录 attach 请求是否在同一 scheduler 周期获得 token budget。

    参数:
        output: 原始 Scheduler.schedule() 返回对象。
        records: 本周期成功 attach 的 fast-path 诊断记录。
        log: 日志对象。
        cycle_id: 当前 scheduler 周期编号。

    返回:
        无返回值。
    """

    scheduled_ids = _scheduled_request_ids(output)
    raw_token_counts = getattr(output, "num_scheduled_tokens", None)
    token_counts = (
        {str(request_id): count for request_id, count in raw_token_counts.items()}
        if isinstance(raw_token_counts, dict)
        else {}
    )
    for record in records:
        _PD_FAST_PATH_PENDING[str(record["request_id"])] = {
            **record,
            "pending_cycles": 0,
        }
    if len(_PD_FAST_PATH_PENDING) > 4096:
        oldest_request_id = next(iter(_PD_FAST_PATH_PENDING))
        _PD_FAST_PATH_PENDING.pop(oldest_request_id, None)
    for record in records:
        request_id = str(record["request_id"])
        scheduled_same_cycle = request_id in scheduled_ids
        if scheduled_same_cycle:
            _track_pd_d_schedule_output(
                record,
                request_id=request_id,
                cycle_id=cycle_id,
                token_counts=token_counts,
                scheduled_ids=scheduled_ids,
            )
        log.info(
            "Jiuwen KVBM D attach fast-path diagnostics: handoff_id=%s "
            "request_id=%s attach_cycle_id=%s first_scheduled_cycle_id=%s "
            "scheduled_same_cycle=%s promoted_to_waiting_head=%s "
            "d_request_seen_to_attach_start_ms=%.3f attach_task_queue_ms=%.3f "
            "attach_execution_ms=%.3f running_request_count=%s "
            "scheduled_token_count=%s",
            record["handoff_id"],
            request_id,
            record["attach_cycle_id"],
            record["attach_cycle_id"] if scheduled_same_cycle else None,
            scheduled_same_cycle,
            record["promoted_to_waiting_head"],
            record["d_request_seen_to_attach_start_ms"],
            record["attach_task_queue_ms"],
            record["attach_execution_ms"],
            record["running_request_count"],
            token_counts.get(request_id, 0),
        )
        if scheduled_same_cycle:
            _PD_FAST_PATH_PENDING.pop(request_id, None)
    newly_attached_ids = {str(record["request_id"]) for record in records}
    for request_id in scheduled_ids - newly_attached_ids:
        record = _PD_FAST_PATH_PENDING.pop(request_id, None)
        if record is None:
            continue
        _track_pd_d_schedule_output(
            record,
            request_id=request_id,
            cycle_id=cycle_id,
            token_counts=token_counts,
            scheduled_ids=scheduled_ids,
        )
        log.info(
            "Jiuwen KVBM D attach fast-path scheduled later: handoff_id=%s "
            "request_id=%s attach_cycle_id=%s first_scheduled_cycle_id=%s "
            "scheduled_same_cycle=False scheduled_token_count=%s",
            record["handoff_id"],
            request_id,
            record["attach_cycle_id"],
            cycle_id,
            token_counts.get(request_id, 0),
        )
    for request_id, record in list(_PD_FAST_PATH_PENDING.items()):
        pending_cycles = int(record.get("pending_cycles", 0)) + 1
        if pending_cycles < _PD_FAST_PATH_PENDING_MAX_CYCLES:
            record["pending_cycles"] = pending_cycles
            continue
        _PD_FAST_PATH_PENDING.pop(request_id, None)
        log.warning(
            "Jiuwen KVBM discarded stale D attach fast-path diagnostics: "
            "handoff_id=%s request_id=%s attach_cycle_id=%s waited_cycles=%s",
            record.get("handoff_id"),
            request_id,
            record.get("attach_cycle_id"),
            pending_cycles,
        )


def _remove_attached_pd_target_gates(
    request_id: str,
    handoff_id: str,
) -> int:
    """清理成功 attach 请求对应的全部 Router/engine request gate 别名。

    参数:
        request_id: Scheduler attach 任务使用的真实 engine request ID。
        handoff_id: 当前 PD handoff ID，用于避免误删其他请求的 gate。

    返回:
        实际移除的 gate 条目数量。
    """

    aliases = set(_pd_request_id_aliases(request_id))
    removed_gates: list[_PDTargetRequestGate] = []
    with _PD_TARGET_LOCK:
        removable = []
        for selected_gate_key, matched_selected_gate in _PD_TARGET_GATES.items():
            alias_matches = (
                selected_gate_key in aliases
                or matched_selected_gate.request_id in aliases
                or (matched_selected_gate.engine_request_id or "") in aliases
            )
            if matched_selected_gate.handoff_id == handoff_id and alias_matches:
                removable.append(selected_gate_key)
        for gate_key in removable:
            removed = _PD_TARGET_GATES.pop(gate_key, None)
            if removed is not None:
                removed_gates.append(removed)
    for removed in removed_gates:
        with _VLLM_BLOCK_POOL_LOCK:
            _release_pd_decode_admission_ticket_locked(
                removed.handoff_id,
                ticket_id=removed.admission_ticket_id,
            )
        _rollback_pd_target_preplan(
            removed.preplan,
            reason="D request attached before preplan adoption",
        )
        if _pd_block_ownership_mode() is SchedulerKVOwnershipMode.LEGACY:
            _process_block_reservations().release_pending_delta(removed.handoff_id)
    return len(removable)


def _process_pd_target_attach_tasks(
    scheduler: Any,
    log: logging.Logger,
    *,
    cycle_id: int = 0,
) -> list[dict[str, Any]]:
    """在 scheduler 线程处理所有已完成 HIXL 的 D request attach 任务。

    参数:
        scheduler: 当前 vLLM Scheduler，持有 requests 和 KVCacheManager。
        log: 用于输出 attach 成功或失败诊断的日志对象。

        cycle_id: 当前 scheduler 周期编号。

    返回:
        本周期成功 attach 的 fast-path 诊断记录；每个任务仍通过 completed
        事件把结果返回给控制线程。
    """

    with _PD_TARGET_LOCK:
        tasks = list(_PD_TARGET_ATTACH_TASKS)
        _PD_TARGET_ATTACH_TASKS.clear()
    records: list[dict[str, Any]] = []
    for task in tasks:
        attach_started_at = time.monotonic()
        with _PD_TARGET_LOCK:
            gate = next(
                (
                    _PD_TARGET_GATES[alias]
                    for alias in _pd_request_id_aliases(task.request_id)
                    if alias in _PD_TARGET_GATES
                ),
                None,
            )
        request_seen_at = gate.request_seen_at if gate is not None else 0.0
        attach_start_unix = time.time()
        try:
            if task.ownership_mode is SchedulerKVOwnershipMode.SCHEDULER:
                task.result = _scheduler_commit_owned_pd_request(scheduler, task)
            else:
                task.result = _scheduler_attach_pd_request(scheduler, task)
            _remove_attached_pd_target_gates(task.request_id, task.handoff_id)
            attach_completed_at = time.monotonic()
            promoted = _promote_pd_request_for_fast_path(scheduler, task.request_id)
            scheduler_timing = {
                "attach_cycle_id": cycle_id,
                "d_request_seen_to_attach_start_ms": max(
                    0.0,
                    (attach_started_at - request_seen_at) * 1000.0,
                )
                if request_seen_at > 0
                else 0.0,
                "attach_task_queue_ms": max(
                    0.0,
                    (attach_started_at - task.created_at) * 1000.0,
                ),
                "attach_execution_ms": max(
                    0.0,
                    (attach_completed_at - attach_started_at) * 1000.0,
                ),
                "promoted_to_waiting_head": promoted,
                "running_request_count": len(getattr(scheduler, "running", ()) or ()),
            }
            task.result["scheduler_timing"] = scheduler_timing
            records.append(
                {
                    "handoff_id": task.handoff_id,
                    "request_id": task.request_id,
                    "attach_completed_at": attach_completed_at,
                    **scheduler_timing,
                }
            )
            log.info(
                "Jiuwen KVBM D request blocks attached: handoff_id=%s "
                "request_id=%s destination_blocks=%s computed_tokens=%s "
                "reused_published_blocks=%s scheduler_timing=%s",
                task.handoff_id,
                task.request_id,
                task.result.get("destination_blocks"),
                task.result.get("computed_tokens"),
                task.result.get("reused_published_blocks"),
                scheduler_timing,
            )
        except Exception as exc:
            task.error = str(exc)
            log.exception(
                "Jiuwen KVBM D request block attach failed: handoff_id=%s request_id=%s",
                task.handoff_id,
                task.request_id,
            )
        finally:
            log.info(
                "[KVBMPDWorkWindow] phase=attach handoff_id=%s request_id=%s start_unix_s=%.6f end_unix_s=%.6f ok=%s",
                task.handoff_id,
                task.request_id,
                attach_start_unix,
                time.time(),
                not bool(task.error),
            )
            task.completed.set()
    return records


def _scheduler_commit_owned_pd_request(
    scheduler: Any,
    task: _PDTargetAttachTask,
) -> dict[str, Any]:
    """在Scheduler线程提交HIXL已经写完的请求自有block。

    参数:
        scheduler: 当前vLLM Scheduler。
        task: 包含handoff、P端hash语义和scheduler reservation ID的attach任务。

    返回:
        包含请求block table、computed token进度和prefix发布范围的兼容响应。

    异常:
        请求、handoff、hash或目标布局发生变化时抛出RuntimeError；调用方会保留
        gate并由失败回滚路径通过KVCacheManager.free()归还请求block。
    """

    requests = getattr(scheduler, "requests", None)
    request = requests.get(task.request_id) if isinstance(requests, dict) else None
    if request is None:
        raise RuntimeError(f"D request is not present in scheduler: {task.request_id}")
    if int(getattr(request, "num_computed_tokens", 0)) != 0:
        raise RuntimeError("D request already started local computation before commit")
    if request in list(getattr(scheduler, "running", ())):
        raise RuntimeError("D request is already running before Scheduler commit")
    with _PD_TARGET_LOCK:
        gate = None
        for candidate_gate in _PD_TARGET_GATES.values():
            if candidate_gate.handoff_id != task.handoff_id:
                continue
            if (candidate_gate.engine_request_id or candidate_gate.request_id) == task.request_id:
                gate = candidate_gate
                break
    if gate is None or gate.scheduler_allocation is None:
        raise RuntimeError("Scheduler-owned D target allocation is not available")
    allocation = gate.scheduler_allocation
    expected_reservation_id = f"scheduler:{task.handoff_id}"
    if task.reservation_id != expected_reservation_id:
        raise RuntimeError("Scheduler-owned reservation ID does not match handoff")
    request_hashes = tuple(_flatten_request_block_hashes(getattr(request, "block_hashes", None)))
    target_packed_hashes = tuple(
        _validate_pd_target_request_hashes(
            request,
            request_hashes,
            task.block_groups,
            task.token_block_fingerprints,
            task.partial_token_fingerprint,
            task.partial_valid_tokens,
            semantic_plain_text=task.semantic_plain_text,
            block_size=int(getattr(scheduler, "block_size", 0) or 0),
        )
    )
    if target_packed_hashes != allocation.target_packed_hashes:
        raise RuntimeError("P export hash differs from Scheduler-owned D allocation")
    if _pd_block_group_signature(task.block_groups) != _pd_block_group_signature(allocation.block_groups):
        raise RuntimeError("P export block groups differ from Scheduler-owned D layout")

    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    if kv_cache_manager is None:
        raise RuntimeError("D scheduler KVCacheManager is not available")
    current_block_ids = kv_cache_manager.get_block_ids(task.request_id)
    current_flattened = tuple(
        int(current_block_ids[int(item["group_id"])][int(item["logical_index"])]) for item in allocation.block_groups
    )
    if current_flattened != allocation.planned_destination_blocks:
        raise RuntimeError("D request block table changed before HIXL commit")

    kv_cache_manager.cache_blocks(request, allocation.full_computed_tokens)
    request.num_computed_tokens = allocation.ready_computed_tokens
    if int(getattr(request, "num_cached_tokens", -1)) < 0:
        request.num_cached_tokens = allocation.ready_computed_tokens
    if hasattr(request, "num_external_computed_tokens"):
        request.num_external_computed_tokens = allocation.external_computed_tokens
    allocation.committed = True

    transfer_positions = set(allocation.transfer_positions)
    reused_flags = [
        index not in transfer_positions and block_hash is not None
        for index, block_hash in enumerate(allocation.target_packed_hashes)
    ]
    target_hash_descriptors = []
    for selected_block_hash, selected_item in zip(
        allocation.target_packed_hashes, allocation.block_groups, strict=True
    ):
        target_hash_descriptors.append(
            {
                "request_scoped_partial": True,
                "logical_index": int(selected_item["logical_index"]),
                "group_id": int(selected_item["group_id"]),
                "valid_token_count": int(selected_item.get("valid_token_count", 0)),
            }
            if selected_block_hash is None
            else _encode_packed_block_hash(selected_block_hash)
        )
    published_indexes = [
        index for index in allocation.transfer_positions if allocation.target_packed_hashes[index] is not None
    ]
    partial_indexes = [
        index for index in allocation.transfer_positions if allocation.target_packed_hashes[index] is None
    ]
    return {
        "ok": True,
        "request_id": task.request_id,
        "ownership_mode": SchedulerKVOwnershipMode.SCHEDULER.value,
        "scheduler_owned": True,
        "destination_blocks": list(allocation.planned_destination_blocks),
        "computed_tokens": allocation.ready_computed_tokens,
        "full_computed_tokens": allocation.full_computed_tokens,
        "partial_valid_tokens": allocation.partial_valid_tokens,
        "logical_block_count": len({int(item["logical_index"]) for item in allocation.block_groups}),
        "reused_published_blocks": reused_flags,
        "planned_destination_blocks": list(allocation.planned_destination_blocks),
        "transfer_positions": list(allocation.transfer_positions),
        "transferred_block_count": len(allocation.transfer_positions),
        "reused_block_count": sum(reused_flags),
        "block_hashes": target_hash_descriptors,
        "published_destination_blocks": [allocation.planned_destination_blocks[index] for index in published_indexes],
        "published_block_hashes": [target_hash_descriptors[index] for index in published_indexes],
        "request_scoped_destination_blocks": [
            allocation.planned_destination_blocks[index] for index in partial_indexes
        ],
        "reservation": {
            "reservation_id": expected_reservation_id,
            "state": "request_owned_committed",
            "block_ids": list(allocation.transfer_destination_blocks),
        },
    }


def _scheduler_attach_pd_request(
    scheduler: Any,
    task: _PDTargetAttachTask,
) -> dict[str, Any]:
    """在 scheduler 线程原子发布并挂接一个 D 请求的 KV blocks。

    参数:
        scheduler: 当前 vLLM Scheduler。
        task: 包含 reservation、hash 和 group 布局的 attach 任务。

    返回:
        包含目标 block ID、computed token 数和 reservation 快照的字典。

    异常:
        请求不存在、已经开始计算、hash 不匹配或 BlockPool 提交失败时抛出异常；
        reservation manager 会撤销所有可见性和引用计数变化。
    """

    requests = getattr(scheduler, "requests", None)
    request = requests.get(task.request_id) if isinstance(requests, dict) else None
    if request is None:
        raise RuntimeError(f"D request is not present in scheduler: {task.request_id}")
    if int(getattr(request, "num_computed_tokens", 0)) != 0:
        raise RuntimeError("D request already started local computation before attach")
    if request in list(getattr(scheduler, "running", ())):
        raise RuntimeError("D request is already running before attach")

    request_hashes = tuple(_flatten_request_block_hashes(getattr(request, "block_hashes", None)))
    target_packed_hashes = _validate_pd_target_request_hashes(
        request,
        request_hashes,
        task.block_groups,
        task.token_block_fingerprints,
        task.partial_token_fingerprint,
        task.partial_valid_tokens,
        semantic_plain_text=task.semantic_plain_text,
        block_size=int(getattr(scheduler, "block_size", 0) or 0),
    )
    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    coordinator = getattr(kv_cache_manager, "coordinator", None)
    if coordinator is None:
        raise RuntimeError("D scheduler KVCacheManager coordinator is not available")
    group_count = int(getattr(kv_cache_manager, "num_kv_cache_groups", 0))
    if group_count <= 0:
        raise RuntimeError("D scheduler reports no KV cache groups")

    reservation_manager = _process_block_reservations()
    reservation = reservation_manager.get(task.reservation_id)
    reservation_layout_count = (
        len(reservation.planned_blocks) if reservation.planned_blocks else len(reservation.raw_blocks)
    )
    if len(task.block_groups) != reservation_layout_count:
        raise RuntimeError("block_groups count does not match destination delta plan")
    observed_group_ids: set[int] = set()
    for item in task.block_groups:
        group_id = int(item.get("group_id", -1))
        if group_id < 0 or group_id >= group_count:
            raise RuntimeError(f"cache group id out of range: group_id={group_id}, groups={group_count}")
        observed_group_ids.add(group_id)
    if len(observed_group_ids) != group_count:
        raise RuntimeError("D import does not provide blocks for every KV cache group")

    full_logical_positions = {
        int(item.get("logical_index", -1)) for item in task.block_groups if not item.get("request_scoped_partial")
    }
    partial_groups = [item for item in task.block_groups if item.get("request_scoped_partial")]
    logical_block_count = len(full_logical_positions) + bool(partial_groups)
    block_size = int(getattr(scheduler, "block_size", 0))
    if block_size <= 0:
        raise RuntimeError("D scheduler block_size is not available")
    full_computed_tokens = len(full_logical_positions) * block_size
    partial_valid_tokens = task.partial_valid_tokens if partial_groups else 0
    available_computed_tokens = full_computed_tokens + partial_valid_tokens
    request_num_tokens = int(getattr(request, "num_tokens", available_computed_tokens))
    computed_tokens = min(available_computed_tokens, max(0, request_num_tokens - 1))
    previous_num_computed = int(getattr(request, "num_computed_tokens", 0))
    previous_num_cached = int(getattr(request, "num_cached_tokens", -1))
    attached = False

    def attach_callback(selected_blocks: tuple[Any, ...]) -> None:
        """把已发布 block 增加请求引用并写入 coordinator 请求表。"""

        nonlocal attached
        grouped_blocks: list[list[Any]] = [[] for _ in range(group_count)]
        for block, item in zip(selected_blocks, task.block_groups, strict=True):
            grouped_blocks[int(item["group_id"])].append(block)
        attached = True
        coordinator.allocate_new_computed_blocks(
            request_id=task.request_id,
            new_computed_blocks=tuple(grouped_blocks),
            num_local_computed_tokens=full_computed_tokens,
            num_external_computed_tokens=partial_valid_tokens,
        )
        request.num_computed_tokens = computed_tokens
        if previous_num_cached < 0:
            request.num_cached_tokens = computed_tokens

    def rollback_callback() -> None:
        """撤销请求侧 block table 和 token 进度。"""

        if attached:
            kv_cache_manager.free(request)
        request.num_computed_tokens = previous_num_computed
        request.num_cached_tokens = previous_num_cached

    planned_block_ids = tuple(int(block.block_id) for block in (reservation.planned_blocks or reservation.raw_blocks))
    reserved_block_ids = tuple(reservation.block_ids)
    transfer_positions = set(reservation.transfer_positions)
    if partial_groups:
        attached_reservation = reservation_manager.attach_request_scoped(
            task.reservation_id,
            block_hashes=target_packed_hashes,
            attach_callback=attach_callback,
            rollback_callback=rollback_callback,
        )
    else:
        attached_reservation = reservation_manager.attach_reusing_published(
            task.reservation_id,
            block_hashes=target_packed_hashes,
            attach_callback=attach_callback,
            rollback_callback=rollback_callback,
        )
    target_hash_descriptors = []
    for matched_selected_block_hash, matched_selected_item in zip(target_packed_hashes, task.block_groups, strict=True):
        target_hash_descriptors.append(
            {
                "request_scoped_partial": True,
                "logical_index": int(matched_selected_item.get("logical_index", -1)),
                "group_id": int(matched_selected_item.get("group_id", -1)),
                "valid_token_count": int(matched_selected_item.get("valid_token_count", 0)),
            }
            if matched_selected_block_hash is None
            else _encode_packed_block_hash(matched_selected_block_hash)
        )
    newly_published_indexes = []
    for matched_matched_selected_index, selected_value in enumerate(target_packed_hashes):
        if selected_value is not None and (
            not reservation.planned_blocks or matched_matched_selected_index in transfer_positions
        ):
            newly_published_indexes.append(matched_matched_selected_index)
    if reservation.planned_blocks:
        reused_flags = [
            index not in transfer_positions and value is not None for index, value in enumerate(target_packed_hashes)
        ]
    else:
        block_id_pairs = zip(attached_reservation.block_ids, reserved_block_ids, strict=True)
        reused_flags = [actual_id != reserved_id for actual_id, reserved_id in block_id_pairs]
    return {
        "ok": True,
        "request_id": task.request_id,
        "destination_blocks": list(attached_reservation.block_ids),
        "computed_tokens": computed_tokens,
        "full_computed_tokens": full_computed_tokens,
        "partial_valid_tokens": partial_valid_tokens,
        "logical_block_count": logical_block_count,
        "reused_published_blocks": reused_flags,
        "planned_destination_blocks": list(planned_block_ids),
        "transfer_positions": sorted(transfer_positions),
        "transferred_block_count": (len(transfer_positions) if reservation.planned_blocks else len(reserved_block_ids)),
        "reused_block_count": sum(reused_flags),
        "block_hashes": target_hash_descriptors,
        "published_destination_blocks": [
            int(attached_reservation.block_ids[index]) for index in newly_published_indexes
        ],
        "published_block_hashes": [target_hash_descriptors[index] for index in newly_published_indexes],
        "request_scoped_destination_blocks": [
            int(attached_reservation.block_ids[index])
            for index, value in enumerate(target_packed_hashes)
            if value is None
        ],
        "reservation": attached_reservation.to_dict(),
    }


def _validate_pd_target_request_hashes(
    request: Any,
    request_hashes: tuple[Any, ...],
    block_groups: tuple[dict[str, Any], ...],
    source_token_fingerprints: tuple[str, ...],
    source_partial_token_fingerprint: str = "",
    source_partial_valid_tokens: int = 0,
    *,
    semantic_plain_text: bool,
    block_size: int,
) -> list[bytes | None]:
    """确认 P export 的每个逻辑 hash 都属于目标 D 请求公共前缀。

    参数:
        request: D vLLM Request，用于计算目标 token block 指纹。
        request_hashes: D vLLM Request 自己计算的完整 prompt block hash。
        block_groups: P export 返回的每个物理 block 及逻辑位置描述。
        source_token_fingerprints: P 请求按逻辑 block 计算的稳定 token 指纹。
        source_partial_token_fingerprint: P 请求尾部 partial token span 指纹。
        source_partial_valid_tokens: P partial block 中有效 token 数。
        semantic_plain_text: P 请求是否不含额外 KV hash 语义。
        block_size: D Scheduler 的逻辑 block token 数。

    返回:
        返回按 block_groups 顺序构造的 D 本地 packed hash；partial block 对应
        None，表示只挂接当前请求且禁止发布。位置越界、token 不一致或无法
        安全确认语义时抛出 RuntimeError。
    """

    target_token_fingerprints = _token_block_fingerprints(
        _diagnostic_prompt_token_ids(request),
        block_size,
    )
    checked_positions: set[int] = set()
    target_packed_hashes: list[bytes | None] = []
    for item in block_groups:
        logical_index = int(item.get("logical_index", -1))
        group_id = int(item.get("group_id", -1))
        if group_id < 0:
            raise RuntimeError("P export block group is missing a valid group_id")
        if bool(item.get("request_scoped_partial", False)):
            target_tokens = _diagnostic_prompt_token_ids(request)
            expected_valid_tokens = len(target_tokens) % block_size
            expected_logical_index = len(target_tokens) // block_size
            target_partial_fingerprint = _partial_token_fingerprint(
                target_tokens,
                block_size,
            )
            if logical_index != expected_logical_index:
                raise RuntimeError(
                    "P partial block is outside D request prompt tail: "
                    f"position={logical_index}, expected={expected_logical_index}"
                )
            if (
                expected_valid_tokens <= 0
                or source_partial_valid_tokens != expected_valid_tokens
                or int(item.get("valid_token_count", 0)) != expected_valid_tokens
            ):
                raise RuntimeError(
                    "P partial block valid token count does not match D request: "
                    f"source={source_partial_valid_tokens}, "
                    f"target={expected_valid_tokens}"
                )
            fingerprint_usable = (
                semantic_plain_text
                and _pd_request_supports_token_fingerprint(request)
                and bool(source_partial_token_fingerprint)
            )
            if not fingerprint_usable or source_partial_token_fingerprint != target_partial_fingerprint:
                raise RuntimeError("P partial block token fingerprint does not match D request")
            target_packed_hashes.append(None)
            continue
        if logical_index < 0 or logical_index >= len(request_hashes):
            raise RuntimeError(f"P export logical block is outside D request: position={logical_index}")
        target_hash = request_hashes[logical_index]
        if not isinstance(target_hash, bytes | bytearray):
            raise RuntimeError("D request block hash type is not compatible with vLLM 0.18")
        target_packed_hashes.append(bytes(target_hash) + group_id.to_bytes(4, "big", signed=False))
        if logical_index in checked_positions:
            continue
        descriptor = item.get("block_hash")
        external_hash = descriptor.get("external_hash") if isinstance(descriptor, dict) else None
        if not external_hash:
            raise RuntimeError("P export block group is missing external_hash")
        hash_matches = bool(
            set(_block_hash_aliases(external_hash)).intersection(_block_hash_aliases(request_hashes[logical_index]))
        )
        token_matches = (
            semantic_plain_text
            and _pd_request_supports_token_fingerprint(request)
            and logical_index < len(source_token_fingerprints)
            and logical_index < len(target_token_fingerprints)
            and source_token_fingerprints[logical_index] == target_token_fingerprints[logical_index]
        )
        if not hash_matches and not token_matches:
            raise RuntimeError(
                "P export hash does not match D request at position "
                f"{logical_index}: source_hash_fingerprint="
                f"{_block_hash_fingerprint(external_hash)} "
                f"target_hash_fingerprint="
                f"{_block_hash_fingerprint(request_hashes[logical_index])} "
                f"source_token_fingerprint="
                f"{source_token_fingerprints[logical_index] if logical_index < len(source_token_fingerprints) else ''} "
                f"target_token_fingerprint="
                f"{target_token_fingerprints[logical_index] if logical_index < len(target_token_fingerprints) else ''}"
            )
        if not hash_matches:
            logger.info(
                "Jiuwen KVBM PD accepted token-equivalent block with different "
                "internal hashes: position=%d token_fingerprint=%s; publishing "
                "with D request hash",
                logical_index,
                target_token_fingerprints[logical_index],
            )
        checked_positions.add(logical_index)
    return target_packed_hashes


def _log_scheduler_cache_hit_diagnostics(output: Any, log: logging.Logger) -> None:
    """记录 SchedulerOutput 中新请求的真实 cache-hit token 统计。

    参数:
        output: vLLM Scheduler.schedule() 返回的 SchedulerOutput。
        log: 用于输出请求级诊断信息的日志对象。

    返回:
        无返回值；字段缺失或单条数据异常时跳过对应请求，不影响调度。
    """

    new_requests = _read_object_field(output, "scheduled_new_reqs", [])
    scheduled_tokens = _read_object_field(output, "num_scheduled_tokens", {})
    if not isinstance(new_requests, list | tuple):
        return
    for request in new_requests:
        request_id = _new_request_id(request)
        if isinstance(scheduled_tokens, dict):
            scheduled_prompt_tokens = _safe_nonnegative_int(scheduled_tokens.get(request_id, 0))
        else:
            scheduled_prompt_tokens = None
        _log_new_request_cache_hit_diagnostics(
            request,
            log,
            scheduled_prompt_tokens=scheduled_prompt_tokens,
            source="scheduler_output",
        )


def _log_new_request_cache_hit_diagnostics(
    request: Any,
    log: logging.Logger,
    *,
    scheduled_prompt_tokens: int | None = None,
    source: str = "new_request_data",
) -> None:
    """输出一条去重后的请求级 KV cache 命中诊断。

    参数:
        request: vLLM NewRequestData 或字段兼容对象。
        log: 用于输出请求诊断的日志对象。
        scheduled_prompt_tokens: 当前调度步安排计算的 token 数；主挂点无法取得
            时为空。
        source: 产生诊断的挂点名称，用于现场确认实际命中的兼容路径。

    返回:
        无返回值；同一请求已经输出过诊断时直接返回。
    """

    request_id = _new_request_id(request)
    if not request_id or not _mark_cache_hit_diagnostic_request(request_id):
        return
    prompt_token_ids = _read_object_field(request, "prompt_token_ids", [])
    cached_prompt_tokens = _safe_nonnegative_int(_read_object_field(request, "num_computed_tokens", 0))
    try:
        prompt_tokens = len(prompt_token_ids)
    except TypeError:
        prompt_tokens = 0
    cached_prompt_tokens = min(cached_prompt_tokens, prompt_tokens)
    computed_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    cache_hit_ratio = cached_prompt_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
    log.info(
        "Jiuwen KVBM request cache-hit diagnostics: %s",
        {
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "computed_prompt_tokens": computed_prompt_tokens,
            "scheduled_prompt_tokens": scheduled_prompt_tokens,
            "cache_hit_ratio": round(cache_hit_ratio, 6),
            "source": source,
        },
    )


def _new_request_id(request: Any) -> str:
    """读取 NewRequestData 的请求 ID。

    参数:
        request: vLLM NewRequestData 或字段兼容对象。

    返回:
        优先返回 req_id，兼容旧结构的 request_id；字段为空时返回空字符串。
    """

    return str(
        _read_object_field(
            request,
            "req_id",
            _read_object_field(request, "request_id", ""),
        )
        or ""
    )


def _mark_cache_hit_diagnostic_request(request_id: str) -> bool:
    """登记已经输出过 cache-hit 诊断的请求。

    参数:
        request_id: vLLM engine-core 请求 ID。

    返回:
        首次登记返回 True；已经登记过返回 False。
    """

    with _CACHE_HIT_DIAGNOSTIC_LOCK:
        if request_id in _CACHE_HIT_DIAGNOSTIC_REQUEST_IDS:
            return False
        if len(_CACHE_HIT_DIAGNOSTIC_REQUEST_IDS) >= _CACHE_HIT_DIAGNOSTIC_LIMIT:
            _CACHE_HIT_DIAGNOSTIC_REQUEST_IDS.clear()
        _CACHE_HIT_DIAGNOSTIC_REQUEST_IDS.add(request_id)
        return True


def _read_object_field(value: Any, name: str, default: Any) -> Any:
    """兼容字典和普通对象读取字段。

    参数:
        value: 待读取的字典或对象。
        name: 字段名称。
        default: 字段不存在时返回的默认值。

    返回:
        字段存在时返回字段值，否则返回 default。
    """

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_nonnegative_int(value: Any) -> int:
    """把任意值转换为非负整数。

    参数:
        value: 待转换的值。

    返回:
        转换成功时返回非负整数，转换失败时返回 0。
    """

    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _record_engine_core_physical_mapping(block_hash: Any, block_id: Any) -> None:
    """记录 engine-core 内的 block hash 到物理 block id 映射。

    参数:
        block_hash: vLLM 内部 block hash，通常是带 group id 的 packed bytes。
        block_id: vLLM 物理 KV block id。

    返回:
        无返回值。
    """

    try:
        parsed_block_id = int(block_id)
    except (TypeError, ValueError):
        return
    resolver = _ENGINE_CORE_RESOLVER
    if resolver is None:
        _PENDING_PHYSICAL_BLOCK_MAPPINGS.append((block_hash, parsed_block_id))
        return
    resolver.record_physical_block(block_hash, parsed_block_id)


def _record_kv_cache_manager(manager: Any) -> None:
    """记录已见过的 KVCacheManager 实例。

    参数:
        manager: vLLM KVCacheManager 实例。

    返回:
        无返回值。
    """

    if all(existing is not manager for existing in _OBSERVED_KV_CACHE_MANAGERS):
        _OBSERVED_KV_CACHE_MANAGERS.append(manager)
    block_pool = getattr(manager, "block_pool", None)
    if block_pool is None:
        coordinator = getattr(manager, "coordinator", None)
        block_pool = getattr(coordinator, "block_pool", None)
    _record_block_pool(block_pool)


def _record_block_pool(block_pool: Any) -> None:
    """记录 engine-core 中构造或使用过的 vLLM BlockPool。

    参数:
        block_pool: vLLM BlockPool 实例；为空时忽略。

    返回:
        无返回值。
    """

    if block_pool is None:
        return
    if all(existing is not block_pool for existing in _OBSERVED_BLOCK_POOLS):
        _OBSERVED_BLOCK_POOLS.append(block_pool)


def _latest_reservable_block_pool() -> Any | None:
    """返回最近观察到且支持安全 reservation API 的 BlockPool。

    返回:
        支持 get_num_free_blocks、get_new_blocks、free_blocks 的 BlockPool；
        尚未观察到兼容实例时返回 None。
    """

    required = ("get_num_free_blocks", "get_new_blocks", "free_blocks")
    for block_pool in reversed(_OBSERVED_BLOCK_POOLS):
        if all(hasattr(block_pool, name) for name in required):
            return block_pool
    return None


def _record_from_observed_kv_cache_managers() -> None:
    """从已见过的 KVCacheManager 同步映射。

    返回:
        无返回值。
    """

    for manager in list(_OBSERVED_KV_CACHE_MANAGERS):
        _record_from_kv_cache_manager(manager)


def _record_from_kv_cache_manager(manager: Any) -> int:
    """从 KVCacheManager.block_pool 同步 hash 到物理 block id 映射。

    参数:
        manager: vLLM KVCacheManager 实例。

    返回:
        本次同步的映射数量。
    """

    block_pool = getattr(manager, "block_pool", None)
    if block_pool is None:
        coordinator = getattr(manager, "coordinator", None)
        block_pool = getattr(coordinator, "block_pool", None)
    return _record_from_block_pool(block_pool)


def _record_from_block_pool(block_pool: Any) -> int:
    """从 BlockPool 快照同步 hash 到物理 block id 映射。

    参数:
        block_pool: vLLM BlockPool 实例或兼容对象。

    返回:
        本次同步的映射数量。
    """

    if block_pool is None:
        return 0
    cached_map = getattr(block_pool, "cached_block_hash_to_block", None)
    cache = getattr(cached_map, "_cache", None)
    if not isinstance(cache, dict):
        return 0
    recorded = 0
    for block_hash, blocks in cache.items():
        for block in _iter_kv_cache_blocks(blocks):
            block_id = getattr(block, "block_id", None)
            if block_id is None:
                continue
            _record_engine_core_physical_mapping(block_hash, block_id)
            block_specific_hash = getattr(block, "block_hash", None)
            if block_specific_hash is not None:
                _record_engine_core_physical_mapping(block_specific_hash, block_id)
            recorded += 1
    if recorded:
        logger.info("Jiuwen KVBM synchronized %d physical block mappings from BlockPool snapshot", recorded)
    return recorded


def _record_from_events_with_manager(events: Any, manager: Any) -> None:
    """根据事件 group_idx 从 manager 的 BlockPool 同步映射。

    参数:
        events: vLLM KV event 列表。
        manager: vLLM KVCacheManager 实例。

    返回:
        无返回值。
    """

    if not events:
        return
    resolver = _ENGINE_CORE_RESOLVER
    if resolver is None:
        return
    for event in events:
        normalized = _normalize_kv_event(event)
        block_hashes = normalized["raw_block_hashes"] or normalized["block_hashes"]
        if not block_hashes:
            continue
        if resolver.resolve_block_hashes(block_hashes, refresh_on_miss=False):
            continue
        targeted_started_ns = time.perf_counter_ns()
        if _record_event_hashes_from_kv_cache_manager(
            manager,
            block_hashes,
            group_idx=normalized["group_idx"],
        ) and resolver.resolve_block_hashes(
            block_hashes,
            refresh_on_miss=False,
        ):
            resolver.record_fallback_diagnostic(
                event_type=normalized["event_type"],
                group_idx=normalized["group_idx"],
                hash_count=len(block_hashes),
                targeted_miss_count=0,
                targeted_lookup_ms=(time.perf_counter_ns() - targeted_started_ns) / 1_000_000.0,
                fallback_reason="targeted_lookup_resolved",
            )
            continue
        targeted_lookup_ms = (time.perf_counter_ns() - targeted_started_ns) / 1_000_000.0
        # remove/evict 事件到达时 block 通常已从 BlockPool 删除；全量扫描
        # 既无法恢复映射，又会让扫描成本随缓存规模增长。
        if normalized["event_type"] != "store":
            resolver.record_fallback_diagnostic(
                event_type=normalized["event_type"],
                group_idx=normalized["group_idx"],
                hash_count=len(block_hashes),
                targeted_miss_count=len(block_hashes),
                targeted_lookup_ms=targeted_lookup_ms,
                fallback_reason="non_store_snapshot_skipped",
            )
            continue

        snapshot_started_ns = time.perf_counter_ns()
        _record_from_kv_cache_manager(manager)
        snapshot_ms = (time.perf_counter_ns() - snapshot_started_ns) / 1_000_000.0
        resolver.add_snapshot_timing_ms(snapshot_ms)
        resolved = bool(resolver.resolve_block_hashes(block_hashes, refresh_on_miss=False))
        resolver.record_fallback_diagnostic(
            event_type=normalized["event_type"],
            group_idx=normalized["group_idx"],
            hash_count=len(block_hashes),
            targeted_miss_count=len(block_hashes),
            targeted_lookup_ms=targeted_lookup_ms,
            fallback_reason=("store_snapshot_resolved" if resolved else "store_snapshot_unresolved"),
        )


def _record_event_hashes_from_kv_cache_manager(
    manager: Any,
    block_hashes: list[Any],
    *,
    group_idx: int,
) -> int:
    """根据事件 hash 定向读取 BlockPool 映射。

    参数:
        manager: vLLM KVCacheManager 实例。
        block_hashes: 当前 KV 事件携带的 block hash 列表。
        group_idx: KV cache group 编号，用于构造 vLLM packed hash。

    返回:
        成功定向记录的物理 block 数量。

    说明:
        vLLM 事件上报 external hash，BlockPool 通常使用
        ``external_hash + group_idx(4 bytes)`` 作为键。此函数只做
        按键查询，避免遍历随请求数增长的整个 BlockPool。
    """

    block_pool = getattr(manager, "block_pool", None)
    if block_pool is None:
        coordinator = getattr(manager, "coordinator", None)
        block_pool = getattr(coordinator, "block_pool", None)
    cached_map = getattr(block_pool, "cached_block_hash_to_block", None)
    cache = getattr(cached_map, "_cache", None)
    if not isinstance(cache, dict):
        return 0

    recorded = 0
    for block_hash in block_hashes:
        blocks = None
        matched_key = None
        for candidate in _packed_block_hash_candidates(block_hash, group_idx):
            try:
                blocks = cache.get(candidate)
            except (TypeError, ValueError):
                blocks = None
            if blocks is not None:
                matched_key = candidate
                break
        if blocks is None:
            continue
        for block in _iter_kv_cache_blocks(blocks):
            block_id = getattr(block, "block_id", None)
            if block_id is None:
                continue
            _record_engine_core_physical_mapping(block_hash, block_id)
            _record_engine_core_physical_mapping(matched_key, block_id)
            block_specific_hash = getattr(block, "block_hash", None)
            if block_specific_hash is not None:
                _record_engine_core_physical_mapping(block_specific_hash, block_id)
            recorded += 1
    return recorded


def _packed_block_hash_candidates(block_hash: Any, group_idx: int) -> list[Any]:
    """生成可用于 BlockPool 定向查询的 hash 键。

    参数:
        block_hash: KV 事件携带的 external 或 packed block hash。
        group_idx: KV cache group 编号。

    返回:
        按优先级排列的原始键与 packed bytes 候选列表。
    """

    candidates = [block_hash]
    if isinstance(block_hash, bytes | bytearray):
        raw = bytes(block_hash)
        parsed_group_idx = max(int(group_idx), 0)
        for byteorder in ("big", "little"):
            packed_suffix = parsed_group_idx.to_bytes(4, byteorder=byteorder)
            if len(raw) < 4 or raw[-4:] != packed_suffix:
                candidates.append(raw + packed_suffix)
    return list(dict.fromkeys(candidates))


def _iter_kv_cache_blocks(blocks: Any) -> list[Any]:
    """展开 BlockHashToBlockMap 中的 block 容器。

    参数:
        blocks: KVCacheBlock、dict[int, KVCacheBlock] 或兼容容器。

    返回:
        KVCacheBlock 对象列表。
    """

    if blocks is None:
        return []
    if isinstance(blocks, dict):
        return list(blocks.values())
    if isinstance(blocks, list | tuple | set):
        return list(blocks)
    return [blocks]


def _apply_engine_core_kv_events(event_batch: Any) -> None:
    """在 engine-core 内消费 vLLM KV event batch。

    参数:
        event_batch: vLLM EventBatch/KVEventBatch 对象，需包含 events 字段。

    返回:
        无返回值。
    """

    global _KV_EVENT_TIMING_SAMPLE_COUNT, _UNRESOLVED_EVENT_LOG_COUNT
    control = _ENGINE_CORE_DATAPLANE
    if control is None:
        return
    events = getattr(event_batch, "events", None)
    if not events:
        return
    resolver = _ENGINE_CORE_RESOLVER
    apply_started_ns = time.perf_counter_ns()
    try:
        produced = control.apply_kv_events(events)
    except Exception as exc:
        logger.warning("Jiuwen KVBM failed to consume engine-core KV events: %s", exc)
        return
    kv_event_apply_ms = (time.perf_counter_ns() - apply_started_ns) / 1_000_000.0
    runtime_diagnostics = control.consume_kv_event_runtime_diagnostics()
    resolve_ms = resolver.consume_resolve_timing_ms() if resolver is not None else 0.0
    runtime_update_ms = max(kv_event_apply_ms - resolve_ms, 0.0)
    block_pool_snapshot_ms = resolver.consume_snapshot_timing_ms() if resolver is not None else 0.0
    fallback_diagnostics = resolver.consume_fallback_diagnostics() if resolver is not None else []
    cache_full_mappings = resolver.consume_cache_full_mapping_count() if resolver is not None else 0
    coordinator_cache_mappings = resolver.consume_coordinator_cache_mapping_count() if resolver is not None else 0
    coordinator_cache_timing = resolver.consume_coordinator_cache_mapping_timing() if resolver is not None else {}
    if produced:
        watermark_started_ns = time.perf_counter_ns()
        watermark_events = _enforce_engine_core_watermarks(control)
        watermark_ms = (time.perf_counter_ns() - watermark_started_ns) / 1_000_000.0
        # 原生 vLLM store 事件负责 token 链，KVBM store relay 额外携带
        # resolver 得到的物理 block id，供 Router 自动编排跨 Worker 传输。
        relay_started_ns = time.perf_counter_ns()
        _enqueue_runtime_events(
            [*produced, *watermark_events],
            include_store=True,
        )
        event_relay_ms = (time.perf_counter_ns() - relay_started_ns) / 1_000_000.0
        _KV_EVENT_TIMING_SAMPLE_COUNT += 1
        sample_due = (
            _KV_EVENT_TIMING_SAMPLE_COUNT <= 8
            or _KV_EVENT_TIMING_SAMPLE_COUNT % _KV_EVENT_TIMING_SAMPLE_EVERY == 0
            or block_pool_snapshot_ms > 0.0
            or bool(watermark_events)
            or any(item["fallback_reason"] != "targeted_lookup_resolved" for item in fallback_diagnostics)
        )
        if sample_due:
            logger.info(
                "Jiuwen KVBM engine-core KV event timing: batch=%d events=%d produced=%d "
                "block_pool_snapshot_ms=%.3f kv_event_apply_ms=%.3f resolve_ms=%.3f "
                "runtime_update_ms=%.3f describe_blocks_ms=%.3f "
                "metadata_update_ms=%.3f runtime_event_build_ms=%.3f "
                "registered_blocks=%d unchanged_blocks=%d cache_full_mappings=%d "
                "coordinator_cache_mappings=%d coordinator_cache_mapping_ms=%.3f "
                "coordinator_cache_visited_blocks=%d coordinator_cache_skipped_blocks=%d "
                "watermark_ms=%.3f event_relay_ms=%.3f watermark_events=%d",
                _KV_EVENT_TIMING_SAMPLE_COUNT,
                len(events),
                len(produced),
                block_pool_snapshot_ms,
                kv_event_apply_ms,
                resolve_ms,
                runtime_update_ms,
                float(runtime_diagnostics.get("describe_blocks_ms", 0.0)),
                float(runtime_diagnostics.get("metadata_update_ms", 0.0)),
                float(runtime_diagnostics.get("runtime_event_build_ms", 0.0)),
                int(runtime_diagnostics.get("registered_blocks", 0)),
                int(runtime_diagnostics.get("unchanged_blocks", 0)),
                cache_full_mappings,
                coordinator_cache_mappings,
                float(coordinator_cache_timing.get("elapsed_ms", 0.0)),
                int(coordinator_cache_timing.get("visited_blocks", 0)),
                int(coordinator_cache_timing.get("skipped_blocks", 0)),
                watermark_ms,
                event_relay_ms,
                len(watermark_events),
            )
            if fallback_diagnostics:
                logger.info(
                    "Jiuwen KVBM engine-core KV event fallback diagnostics: %s",
                    fallback_diagnostics,
                )
        logger.debug(
            "Jiuwen KVBM consumed %d engine-core KV events, produced %d data-plane events, "
            "and produced %d watermark move events; stats=%s",
            len(events),
            len(produced),
            len(watermark_events),
            control.stats(),
        )
        return
    if _UNRESOLVED_EVENT_LOG_COUNT < 5:
        _UNRESOLVED_EVENT_LOG_COUNT += 1
        resolver_stats = _ENGINE_CORE_RESOLVER.diagnostics() if _ENGINE_CORE_RESOLVER is not None else {}
        logger.info(
            "Jiuwen KVBM consumed %d engine-core KV events but resolved no physical blocks; resolver=%s",
            len(events),
            resolver_stats,
        )


def _enforce_engine_core_watermarks(control: "VllmAscend018DataPlaneControl") -> list[Any]:
    """在真实 KV event 消费后执行一次 engine-core 水位策略。

    参数:
        control: engine-core 内绑定的 KVBM 数据面控制对象。

    返回:
        本次水位策略产生的 move 事件列表；策略失败时返回空列表。
    """

    config = control.config
    if config.npu_high_watermark_blocks <= 0 and config.ram_high_watermark_bytes <= 0:
        return []
    try:
        events = []
        if config.npu_high_watermark_blocks > 0:
            events.extend(control.runtime.offload_cold_npu_blocks_to_ram())
        if config.ram_high_watermark_bytes > 0:
            events.extend(control.runtime.offload_cold_ram_blocks_to_disk())
    except Exception as exc:
        logger.warning("Jiuwen KVBM engine-core watermark enforcement failed: %s", exc)
        return []
    logger.info(
        "Jiuwen KVBM engine-core watermark enforcement produced %d move events; stats=%s",
        len(events),
        control.stats(),
    )
    return events


def _enqueue_runtime_events(
    events: list[Any] | tuple[Any, ...],
    *,
    include_store: bool = False,
) -> None:
    """将 engine-core KVBM runtime 事件放入待上报 outbox。

    参数:
        events: WorkerKVBMRuntime 产生的事件列表。
        include_store: 是否同时上报 store 事件。常规 vLLM KV store 已由原生
            KV event 上报；跨 worker 拉取完成时设为 True，用于发布新的 NPU 位置。

    返回:
        无返回值。
    """

    accepted_types = {"move"}
    if include_store:
        accepted_types.add("store")
    relay_events = [
        _runtime_event_to_relay_event(event) for event in events if _runtime_event_type(event) in accepted_types
    ]
    relay_events = [event for event in relay_events if event is not None]
    if not relay_events:
        return
    with _ENGINE_CORE_RUNTIME_EVENT_OUTBOX_LOCK:
        _ENGINE_CORE_RUNTIME_EVENT_OUTBOX.extend(relay_events)
        overflow = len(_ENGINE_CORE_RUNTIME_EVENT_OUTBOX) - _ENGINE_CORE_RUNTIME_EVENT_OUTBOX_LIMIT
        if overflow > 0:
            del _ENGINE_CORE_RUNTIME_EVENT_OUTBOX[:overflow]
            logger.warning("Jiuwen KVBM dropped %d old runtime events from engine-core outbox", overflow)


def _drain_runtime_event_outbox() -> list[dict[str, Any]]:
    """取出并清空 engine-core KVBM runtime 待上报事件。

    参数:
        无参数。

    返回:
        可直接放入 worker kv_events relay message 的事件字典列表。
    """

    with _ENGINE_CORE_RUNTIME_EVENT_OUTBOX_LOCK:
        events = list(_ENGINE_CORE_RUNTIME_EVENT_OUTBOX)
        _ENGINE_CORE_RUNTIME_EVENT_OUTBOX.clear()
    return events


def _runtime_event_to_relay_event(event: Any) -> dict[str, Any] | None:
    """将 KVBMRuntimeEvent 转换为 router 可消费的 KV event 字典。

    参数:
        event: KVBMRuntimeEvent 实例或兼容对象。

    返回:
        形如 {"type": ..., "data": ...} 的事件字典；无法转换时返回 None。
    """

    if hasattr(event, "to_dict"):
        payload = event.to_dict()
    elif isinstance(event, dict):
        payload = event
    else:
        return None

    event_type = str(payload.get("event_type", ""))
    location = str(payload.get("location", ""))
    engine_specific = payload.get("engine_specific") or {}
    if event_type == "move":
        relay_type = "BlockMoved"
    elif event_type == "store":
        relay_type = "BlockStored"
    elif event_type in {"evict", "removed", "remove"}:
        relay_type = "BlockRemoved"
    else:
        relay_type = event_type

    relay_data = {
        "block_hashes": list(payload.get("block_hashes") or []),
        "block_size": _int_env(_BLOCK_SIZE_ENV, 16),
        "medium": location,
        "to": engine_specific.get("to") or location,
        "from": engine_specific.get("from"),
        "token_count": int(payload.get("token_count") or 0),
    }
    # 物理 block id 是 Router 自动编排跨 Worker 传输所需的源地址语义。
    # 保留 runtime 事件中的扩展字段，避免 relay 只上报 hash 后丢失物理映射。
    relay_data.update({str(key): value for key, value in engine_specific.items() if value is not None})
    return {
        "type": relay_type,
        "data": relay_data,
    }


def _runtime_event_type(event: Any) -> str:
    """读取 KVBM runtime 事件类型。

    参数:
        event: KVBMRuntimeEvent 实例或兼容 dict。

    返回:
        事件类型字符串；无法读取时返回空字符串。
    """

    if hasattr(event, "event_type"):
        return str(getattr(event, "event_type"))
    if isinstance(event, dict):
        return str(event.get("event_type", ""))
    return ""


def diagnostics_enabled_from_env() -> bool:
    """Return the shared diagnostic switch for version-specific adapters."""
    return _diagnostics_enabled_from_env()


def _diagnostics_enabled_from_env() -> bool:
    """从环境变量读取 engine-core/worker 子进程诊断开关。

    返回:
        诊断开关为 1/true/yes/on 时返回 True，否则返回 False。
    """

    return os.getenv(_DIAGNOSTICS_ENV, "0").lower() in {"1", "true", "yes", "on"}


def _bind_provider_to_model_runner(
    model_runner: Any,
    kv_caches: dict[str, Any],
    *,
    diagnostics: bool,
    log: logging.Logger,
) -> None:
    """把真实 kv_caches 包装成 provider 并绑定到 model runner。

    参数:
        model_runner: vLLM-Ascend engine-core 内的 NPUModelRunner 实例。
        kv_caches: initialize_kv_cache_tensors 返回的真实 KV cache 字典。
        diagnostics: 是否打印只读诊断信息。
        log: 日志对象。

    返回:
        无返回值。
    """

    global _ENGINE_CORE_DATAPLANE, _ENGINE_CORE_RESOLVER
    provider = VllmAscend018TensorProvider(kv_caches)
    _synchronize_block_size_from_provider(provider, log)
    llm_datadist_backend = _create_engine_core_llm_datadist_backend(provider, log)
    control = VllmAscend018DataPlaneControl(
        provider=provider,
        config=_runtime_config_from_env(),
        llm_datadist_backend=llm_datadist_backend,
    )
    resolver = VllmAscend018PhysicalBlockResolver(model_runner)
    control.set_physical_block_resolver(resolver)
    setattr(model_runner, _PROVIDER_ATTR, provider)
    setattr(model_runner, _DATAPLANE_ATTR, control)
    setattr(model_runner, _RESOLVER_ATTR, resolver)
    setattr(model_runner, _DIAGNOSTICS_ATTR, {})
    _ENGINE_CORE_DATAPLANE = control
    _ENGINE_CORE_RESOLVER = resolver
    for block_hash, block_id in _PENDING_PHYSICAL_BLOCK_MAPPINGS:
        resolver.record_physical_block(block_hash, block_id)
    _PENDING_PHYSICAL_BLOCK_MAPPINGS.clear()
    if _bool_env(_AUTO_REGISTER_ENV):
        registered = control.register_physical_blocks()
        log.info(
            "Jiuwen KVBM registered %d physical KV blocks in engine-core; stats=%s",
            registered,
            control.stats(),
        )
    if _bool_env(_AUTO_ENFORCE_ENV):
        events = control.enforce_watermarks()
        log.info(
            "Jiuwen KVBM initial watermark enforcement produced %d events; stats=%s",
            len(events),
            control.stats(),
        )
    _start_engine_core_prefetch_server(log)
    _start_engine_core_tcp_staging_server(provider, log)
    if not diagnostics:
        log.info("Jiuwen KVBM provider and dataplane bound in vLLM-Ascend engine-core")
        return

    try:
        info = provider.diagnostics(0)
    except Exception as exc:
        log.warning("Jiuwen KVBM engine-core diagnostics failed: %s", exc)
        return

    setattr(model_runner, _DIAGNOSTICS_ATTR, info)
    log.info("Jiuwen KVBM engine-core diagnostics: %s", info)
    log.info("Jiuwen KVBM physical block resolver diagnostics: %s", resolver.diagnostics())


def _synchronize_block_size_from_provider(
    provider: VllmAscend018TensorProvider,
    log: logging.Logger,
) -> int:
    """从真实 KV tensor shape 识别并同步 KVBM block token 数。

    参数:
        provider: 已绑定真实 vLLM-Ascend KV cache 的 tensor provider。
        log: 日志对象，用于报告自动识别结果和配置不一致。

    返回:
        自动识别的 block token 数。

    说明:
        环境变量中的配置仅作为 KV cache 初始化前的兼容值。真实 tensor 可用后，
        始终以 key cache shape 的第 1 维为准，避免 Worker 与 Router 人工配置漂移。
    """

    actual = provider.block_size_tokens()
    configured = _int_env(_BLOCK_SIZE_ENV, actual)
    os.environ[_BLOCK_SIZE_ENV] = str(actual)
    if configured != actual:
        log.warning(
            "Jiuwen KVBM corrected configured block_size=%d to tensor-derived block_size=%d",
            configured,
            actual,
        )
    else:
        log.info("Jiuwen KVBM inferred block_size=%d from KV tensor shape", actual)
    return actual


def _create_engine_core_llm_datadist_backend(
    provider: VllmAscend018TensorProvider,
    log: logging.Logger,
) -> LLMDataDistBackend | None:
    """按环境变量创建并初始化 engine-core 内 LLM-DataDist backend。

    参数:
        provider: 已绑定真实 kv_caches 的 vLLM-Ascend provider，用于推导 device_id。
        log: 日志对象。

    返回:
        初始化成功时返回 backend；未启用或初始化失败时返回 None。
    """

    global _ENGINE_CORE_LLM_DATADIST_CACHE_ID
    data_backend = os.getenv(_DATA_BACKEND_ENV, "local").lower()
    if data_backend not in {"llm_datadist", "datadist"}:
        return None
    try:
        config = _llm_datadist_config_from_env(provider)
        backend = create_llm_datadist_backend(
            config,
            mode=os.getenv(_LLM_DATADIST_MODE_ENV, "python"),
        )
        backend.initialize()
        handle = backend.register_cache(
            _llm_datadist_cache_desc_from_provider(provider),
            [_llm_datadist_local_cache_key(config)],
        )
    except Exception as exc:
        log.warning("Jiuwen KVBM LLM-DataDist backend initialization failed: %s", exc)
        return None
    log.info(
        "Jiuwen KVBM LLM-DataDist backend initialized in engine-core: cache_id=%s health=%s",
        handle.cache_id,
        backend.health(),
    )
    _ENGINE_CORE_LLM_DATADIST_CACHE_ID = handle.cache_id
    return backend


def _llm_datadist_cache_desc_from_provider(
    provider: VllmAscend018TensorProvider,
) -> LLMDataDistCacheDesc:
    """从 vLLM-Ascend provider 构造 LLM-DataDist blocks cache 描述。

    参数:
        provider: 已绑定真实 kv_caches 的 vLLM-Ascend provider。

    返回:
        包含 tensor shape、dtype、地址和层数的 LLMDataDistCacheDesc。
    """

    diagnostics = provider.diagnostics(0)
    return LLMDataDistCacheDesc(
        placement="device",
        data_type=provider.blocks_cache_dtype(),
        shape=provider.blocks_cache_shape(),
        num_tensors=int(diagnostics["layer_count"]) * 2,
        tensor_addresses=provider.tensor_base_addresses(),
        tensor_num_per_layer=2,
        layer_count=int(diagnostics["layer_count"]),
    )


def _llm_datadist_local_cache_key(config: LLMDataDistConfig) -> LLMDataDistCacheKey:
    """构造本 worker 注册 blocks cache 时使用的 key。

    参数:
        config: LLM-DataDist backend 配置。

    返回:
        指向本地 cluster/model 的 cache key。
    """

    return LLMDataDistCacheKey(
        key=f"cluster:{config.endpoint.cluster_id}:model:0",
        prompt_cluster_id=config.endpoint.cluster_id,
        prompt_batch_index=0,
    )


def _llm_datadist_config_from_env(provider: VllmAscend018TensorProvider) -> LLMDataDistConfig:
    """从环境变量构造 LLM-DataDist backend 配置。

    参数:
        provider: vLLM-Ascend KV provider，用于推导 NPU device_id。

    返回:
        LLMDataDistConfig 实例。
    """

    device_id = _infer_provider_device_id(provider)
    listen_port = _int_env(_LLM_DATADIST_LISTEN_PORT_ENV, 0)
    if listen_port <= 0:
        listen_port = _int_env(_LLM_DATADIST_LISTEN_PORT_BASE_ENV, 19000) + max(device_id, 0)
    role = _llm_datadist_role_from_env()
    endpoint = LLMDataDistEndpoint(
        cluster_id=_int_env(_LLM_DATADIST_CLUSTER_ID_ENV, 0),
        role=role,
        listen_ip=os.getenv(_LLM_DATADIST_LISTEN_IP_ENV, "127.0.0.1"),
        listen_port=listen_port,
        device_id=device_id,
    )
    return LLMDataDistConfig(
        enabled=True,
        endpoint=endpoint,
        transfer_backend=os.getenv(_LLM_DATADIST_TRANSFER_BACKEND_ENV, "hixl"),
        enable_cache_manager=_bool_env(_LLM_DATADIST_ENABLE_CACHE_MANAGER_ENV, True),
        enable_remote_cache_accessible=_bool_env(
            _LLM_DATADIST_ENABLE_REMOTE_CACHE_ACCESSIBLE_ENV,
            False,
        ),
    )


def _llm_datadist_role_from_env() -> LLMDataDistRole:
    """读取 LLM-DataDist 角色环境变量。

    返回:
        LLMDataDistRole 枚举值；未知值按 mix 处理。
    """

    raw = os.getenv(_LLM_DATADIST_ROLE_ENV, "mix").lower()
    if raw in {"prompt", "prefill"}:
        return LLMDataDistRole.PROMPT
    if raw in {"decoder", "decode"}:
        return LLMDataDistRole.DECODER
    return LLMDataDistRole.MIX


def _infer_provider_device_id(provider: VllmAscend018TensorProvider) -> int:
    """从 provider 或 diagnostics 推导 NPU device_id。

    参数:
        provider: vLLM-Ascend KV provider。

    返回:
        解析到的 device_id；无法解析时返回 0。
    """

    if provider.device_id is not None:
        return int(provider.device_id)
    try:
        diagnostics = provider.diagnostics(0)
    except Exception:
        return 0
    for key in ("key_device", "value_device"):
        value = str(diagnostics.get(key, ""))
        if ":" not in value:
            continue
        try:
            return int(value.rsplit(":", 1)[1])
        except ValueError:
            continue
    return 0


def _start_engine_core_prefetch_server(log: logging.Logger) -> None:
    """按环境变量启动 engine-core 本地预取控制服务。

    参数:
        log: 日志对象，用于输出启动或失败信息。

    返回:
        无返回值。
    """

    global _ENGINE_CORE_PREFETCH_SERVER, _ENGINE_CORE_PREFETCH_THREAD
    if _ENGINE_CORE_PREFETCH_SERVER is not None:
        return

    port = _int_env(_PREFETCH_PORT_ENV, 0)
    if port <= 0:
        return
    host = os.getenv(_PREFETCH_HOST_ENV, "127.0.0.1")
    try:
        server = _KVBMControlHTTPServer((host, port), _EngineCorePrefetchHandler)
    except Exception as exc:
        log.warning(
            "Jiuwen KVBM engine-core prefetch server failed to start on %s:%s: %s",
            host,
            port,
            exc,
        )
        return

    thread = threading.Thread(
        target=server.serve_forever,
        name="jiuwen-kvbm-prefetch-server",
        daemon=True,
    )
    thread.start()
    _ENGINE_CORE_PREFETCH_SERVER = server
    _ENGINE_CORE_PREFETCH_THREAD = thread
    log.info("Jiuwen KVBM engine-core prefetch server listening on %s:%s", host, port)


def _start_engine_core_tcp_staging_server(
    provider: VllmAscend018TensorProvider,
    log: logging.Logger,
) -> None:
    """按环境变量启动仅供隔离基准使用的 TCP staging 服务。

    参数:
        provider: 当前 engine-core 绑定的真实 KV tensor provider。
        log: 日志对象，用于输出启动或失败信息。

    返回:
        无返回值。基准未启用或端口小于等于 0 时直接返回。
    """

    global _ENGINE_CORE_TCP_STAGING_SERVER, _ENGINE_CORE_TCP_STAGING_THREAD
    if not _bool_env(_BENCHMARK_ENABLED_ENV):
        return
    if _ENGINE_CORE_TCP_STAGING_SERVER is not None:
        return
    port = _int_env(_BENCHMARK_TCP_PORT_ENV, 0)
    if port <= 0:
        log.warning("Jiuwen KVBM transfer benchmark enabled but TCP staging port is disabled")
        return
    host = os.getenv(_BENCHMARK_TCP_HOST_ENV, "127.0.0.1")
    try:
        server, thread = start_tcp_staging_server(
            provider,
            host,
            port,
            _ENGINE_CORE_LLM_DATADIST_LOCK,
            max_blocks=_int_env(_BENCHMARK_MAX_BLOCKS_ENV, 64),
        )
    except Exception as exc:
        log.warning(
            "Jiuwen KVBM TCP staging benchmark server failed to start on %s:%s: %s",
            host,
            port,
            exc,
        )
        return
    _ENGINE_CORE_TCP_STAGING_SERVER = server
    _ENGINE_CORE_TCP_STAGING_THREAD = thread
    log.info(
        "Jiuwen KVBM TCP staging benchmark server listening on %s:%s",
        host,
        port,
    )


def _process_block_reservations() -> VllmBlockPoolReservationManager:
    """返回当前进程持有的 BlockPool reservation manager。

    返回:
        使用当前进程已观察 BlockPool 和共用锁的 reservation manager。
    """

    global _PROCESS_BLOCK_RESERVATIONS
    if _PROCESS_BLOCK_RESERVATIONS is None:
        _PROCESS_BLOCK_RESERVATIONS = VllmBlockPoolReservationManager(
            _latest_reservable_block_pool,
            lock=_VLLM_BLOCK_POOL_LOCK,
        )
    return _PROCESS_BLOCK_RESERVATIONS


def _process_source_block_leases() -> VllmSourceBlockLeaseManager:
    """返回当前 scheduler 进程的源 block lease manager。

    返回:
        使用当前 BlockPool 和共用锁的 VllmSourceBlockLeaseManager。
    """

    global _PROCESS_SOURCE_BLOCK_LEASES
    if _PROCESS_SOURCE_BLOCK_LEASES is None:
        _PROCESS_SOURCE_BLOCK_LEASES = VllmSourceBlockLeaseManager(
            _latest_reservable_block_pool,
            lock=_VLLM_BLOCK_POOL_LOCK,
        )
    return _PROCESS_SOURCE_BLOCK_LEASES


def _process_transfer_transactions() -> TransferTransactionManager:
    """返回当前 NPU worker 进程的跨 Worker 传输事务管理器。

    返回:
        使用环境变量配置并发上限和终态保留时间的事务管理器。
    """

    global _PROCESS_TRANSFER_TRANSACTIONS
    if _PROCESS_TRANSFER_TRANSACTIONS is None:
        _PROCESS_TRANSFER_TRANSACTIONS = TransferTransactionManager(
            max_inflight=max(1, _int_env(_TRANSFER_MAX_INFLIGHT_ENV, 1)),
            completed_retention_seconds=max(
                1.0,
                float(os.getenv(_TRANSFER_RETENTION_SECONDS_ENV, "300")),
            ),
        )
    return _PROCESS_TRANSFER_TRANSACTIONS


def _process_pd_handoffs() -> PDHandoffManager:
    """返回当前 Worker 进程的请求级 Prefill/Decode 交接管理器。

    返回:
        使用环境变量配置容量和终态保留时间的 PDHandoffManager。
    """

    global _PROCESS_PD_HANDOFFS
    if _PROCESS_PD_HANDOFFS is None:
        _PROCESS_PD_HANDOFFS = PDHandoffManager(
            max_inflight=max(1, _int_env(_PD_HANDOFF_MAX_INFLIGHT_ENV, 1024)),
            completed_retention_seconds=max(
                1.0,
                float(os.getenv(_PD_HANDOFF_RETENTION_SECONDS_ENV, "300")),
            ),
        )
    return _PROCESS_PD_HANDOFFS


def _start_engine_core_reservation_server(log: logging.Logger) -> None:
    """在拥有 BlockPool 的 engine-core scheduler 进程启动内部 reservation 服务。

    参数:
        log: 日志对象，用于输出启动或失败信息。

    返回:
        无返回值。
    """

    global _ENGINE_CORE_RESERVATION_SERVER, _ENGINE_CORE_RESERVATION_THREAD
    if _ENGINE_CORE_RESERVATION_SERVER is not None:
        return
    port = _int_env(_RESERVATION_PORT_ENV, 0)
    if port <= 0:
        return
    try:
        server = _KVBMControlHTTPServer(
            ("127.0.0.1", port),
            _EngineCoreReservationHandler,
        )
    except Exception as exc:
        log.warning(
            "Jiuwen KVBM engine-core reservation server failed to start on 127.0.0.1:%s: %s",
            port,
            exc,
        )
        return
    thread = threading.Thread(
        target=server.serve_forever,
        name="jiuwen-kvbm-reservation-server",
        daemon=True,
    )
    thread.start()
    _ENGINE_CORE_RESERVATION_SERVER = server
    _ENGINE_CORE_RESERVATION_THREAD = thread
    _start_source_lease_sweeper(log)
    log.info("Jiuwen KVBM engine-core reservation server listening on 127.0.0.1:%s", port)


def _start_source_lease_sweeper(log: logging.Logger) -> None:
    """启动源 block lease 到期回收线程。

    参数:
        log: 用于输出到期回收诊断的日志对象。

    返回:
        无返回值；进程内只会启动一个 daemon 线程。
    """

    global _SOURCE_LEASE_SWEEPER_THREAD
    if _SOURCE_LEASE_SWEEPER_THREAD is not None:
        return

    def _sweep() -> None:
        while not _SOURCE_LEASE_SWEEPER_STOP.wait(1.0):
            try:
                expired = _process_source_block_leases().expire_due()
                if expired:
                    log.warning(
                        "Jiuwen KVBM expired and released %d abandoned source block leases",
                        expired,
                    )
                expired_reservations = _process_block_reservations().expire_due()
                if expired_reservations:
                    log.warning(
                        "Jiuwen KVBM expired and released %d abandoned destination block reservations",
                        expired_reservations,
                    )
                expired_export_intents = _expire_pd_source_export_intents()
                if expired_export_intents:
                    log.warning(
                        "Jiuwen KVBM expired %d abandoned P export capture intents",
                        expired_export_intents,
                    )
            except Exception:
                log.exception("Jiuwen KVBM block resource sweeper failed")

    thread = threading.Thread(
        target=_sweep,
        name="jiuwen-kvbm-source-lease-sweeper",
        daemon=True,
    )
    thread.start()
    _SOURCE_LEASE_SWEEPER_THREAD = thread


class _LegacyVllmSchedulerKVBridge(SchedulerKVBridge):
    """封装当前KVBM直接管理BlockPool reservation的兼容实现。"""

    @property
    def ownership_mode(self) -> SchedulerKVOwnershipMode:
        """返回legacy所有权模式。

        返回:
            SchedulerKVOwnershipMode.LEGACY。
        """

        return SchedulerKVOwnershipMode.LEGACY

    def prepare_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """登记P请求捕获意图。

        参数:
            payload: 包含handoff、request、prompt长度和捕获TTL的请求字典。

        返回:
            包含捕获意图创建状态的可序列化字典。
        """

        return _local_pd_export_prepare(payload)

    def export_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """读取并pin P源block。

        参数:
            payload: 包含request、block限制和等待预算的请求字典。

        返回:
            包含源block布局、hash和lease的可序列化字典。
        """

        return _local_pd_export(payload)

    def release_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """释放P捕获意图。

        参数:
            payload: 包含request_id和handoff_id的请求字典。

        返回:
            包含是否实际释放捕获意图的幂等结果。
        """

        return _release_pd_source_export_intent(
            str(payload.get("request_id", "")),
            str(payload.get("handoff_id", "")),
        )

    def prepare_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """建立D请求门控。

        参数:
            payload: 包含handoff、request和门控TTL的请求字典。

        返回:
            包含D请求门控创建或续租状态的字典。
        """

        return _local_pd_import_prepare(payload)

    def plan_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """使用legacy reservation生成D目标布局。

        参数:
            payload: 包含P端block语义、request和等待预算的请求字典。

        返回:
            包含复用block、传输block和额外reservation的计划字典。
        """

        return _local_pd_import_plan(payload)

    def complete_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """通过legacy attach/publish提交D目标block。

        参数:
            payload: 包含reservation、block语义和request的请求字典。

        返回:
            包含D请求attach和prefix cache发布状态的字典。
        """

        return _local_pd_import_attach(payload)

    def abort_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """回滚D reservation并解除请求门控。

        参数:
            payload: 包含request_id和可选handoff_id的请求字典。

        返回:
            包含门控释放与reservation回滚状态的字典。
        """

        return _local_pd_import_release(payload)

    def target_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        """读取legacy D门控与reservation状态。

        参数:
            payload: 包含可选request_id的查询字典。

        返回:
            指定D请求或全部等待事务的状态字典。
        """

        return _local_pd_import_status(payload)

    def diagnostics(self) -> dict[str, Any]:
        """返回legacy bridge及其额外预留行为诊断。

        返回:
            标明目标block仍由KVBM额外预留的诊断字典。
        """

        return {
            **super().diagnostics(),
            "scheduler_owned_destination_blocks": False,
            "uses_external_reservation": True,
        }


class _VllmSchedulerOwnedKVBridge(SchedulerKVBridge):
    """使用vLLM Scheduler请求block table承载D端外部KV的实现。"""

    @property
    def ownership_mode(self) -> SchedulerKVOwnershipMode:
        """返回scheduler所有权模式。

        返回:
            SchedulerKVOwnershipMode.SCHEDULER。
        """

        return SchedulerKVOwnershipMode.SCHEDULER

    def prepare_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """登记P请求源block捕获意图。

        参数:
            payload: 包含handoff、request和捕获TTL的请求字典。

        返回:
            包含捕获意图状态的可序列化字典。
        """

        return _local_pd_export_prepare(payload)

    def export_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """读取并pin P请求可传输的源block。

        参数:
            payload: 包含request、block限制和等待预算的请求字典。

        返回:
            包含源block布局、hash和lease的可序列化字典。
        """

        return _local_pd_export(payload)

    def release_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """释放P请求捕获意图。

        参数:
            payload: 包含request_id和handoff_id的请求字典。

        返回:
            包含幂等释放状态的可序列化字典。
        """

        return _release_pd_source_export_intent(
            str(payload.get("request_id", "")),
            str(payload.get("handoff_id", "")),
        )

    def prepare_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """在D Scheduler接收请求前建立门控。

        参数:
            payload: 包含handoff、request和门控TTL的请求字典。

        返回:
            包含D请求门控状态的可序列化字典。
        """

        return _local_pd_import_prepare(payload)

    def plan_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """取得D请求由Scheduler持有的目标block布局。

        参数:
            payload: 包含P端block语义、request和等待预算的请求字典。

        返回:
            包含本地复用、稀疏传输位置和请求自有目标block的计划字典。
        """

        return _local_scheduler_owned_pd_import_plan(payload)

    def complete_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """在HIXL完成后提交D请求自有block。

        参数:
            payload: 包含scheduler reservation ID、block语义和request的字典。

        返回:
            包含computed token进度和prefix发布结果的可序列化字典。
        """

        return _local_pd_import_attach(
            {
                **payload,
                "ownership_mode": SchedulerKVOwnershipMode.SCHEDULER.value,
            }
        )

    def abort_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """解除D门控并归还请求自有block。

        参数:
            payload: 包含request_id和可选handoff_id的请求字典。

        返回:
            包含门控与Scheduler block释放状态的可序列化字典。
        """

        return _local_pd_import_release(payload)

    def target_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        """读取D门控和Scheduler原生分配状态。

        参数:
            payload: 包含可选request_id的查询字典。

        返回:
            指定D请求或全部等待事务的状态字典。
        """

        return _local_pd_import_status(payload)

    def diagnostics(self) -> dict[str, Any]:
        """返回Scheduler-owned bridge诊断。

        返回:
            标明D目标block由vLLM请求持有且不使用额外reservation的字典。
        """

        return {
            **super().diagnostics(),
            "scheduler_owned_destination_blocks": True,
            "uses_external_reservation": False,
        }


def _scheduler_kv_bridge() -> SchedulerKVBridge:
    """返回当前进程唯一的Scheduler KV bridge。

    返回:
        根据PD block所有权配置创建并缓存的bridge实例。
    """

    global _SCHEDULER_KV_BRIDGE
    mode = _pd_block_ownership_mode()
    if _SCHEDULER_KV_BRIDGE is not None and _SCHEDULER_KV_BRIDGE.ownership_mode is mode:
        return _SCHEDULER_KV_BRIDGE
    _SCHEDULER_KV_BRIDGE = (
        _VllmSchedulerOwnedKVBridge() if mode is SchedulerKVOwnershipMode.SCHEDULER else _LegacyVllmSchedulerKVBridge()
    )
    logger.info(
        "Jiuwen KVBM Scheduler KV bridge initialized: %s",
        _SCHEDULER_KV_BRIDGE.diagnostics(),
    )
    return _SCHEDULER_KV_BRIDGE


def _handle_local_block_reservation_request(
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """在 engine-core scheduler 进程处理一次 BlockPool reservation 请求。

    参数:
        path: reservation 请求路径。
        payload: JSON 请求体。

    返回:
        可序列化操作结果。
    """

    if path == "/datadist/blocks/metadata":
        return _local_block_metadata(payload)
    if path == "/datadist/block-pool/capacity/local":
        return _local_block_pool_capacity(payload)
    if path == "/datadist/block-pool/admission-ticket/release/local":
        return _local_pd_decode_admission_ticket_release(payload)
    if path.startswith("/datadist/pd/"):
        bridge = _scheduler_kv_bridge()
        bridge_handlers = {
            "/datadist/pd/export/prepare/local": bridge.prepare_source,
            "/datadist/pd/export/local": bridge.export_source,
            "/datadist/pd/export/release/local": bridge.release_source,
            "/datadist/pd/import/prepare/local": bridge.prepare_target,
            "/datadist/pd/import/plan/local": bridge.plan_target,
            "/datadist/pd/import/attach/local": bridge.complete_target,
            "/datadist/pd/import/release/local": bridge.abort_target,
            "/datadist/pd/import/status/local": bridge.target_status,
        }
        bridge_handler = bridge_handlers.get(path)
        if bridge_handler is not None:
            return bridge_handler(payload)
    if path.startswith("/datadist/source-leases/"):
        return _handle_local_source_lease_request(path, payload)

    manager = _process_block_reservations()
    if path == "/datadist/reservations/status":
        return {"ok": True, "reservations": manager.stats()}
    if path == "/datadist/reservations/reserve":
        reservation = manager.reserve(
            int(payload.get("count", 0)),
            ttl_ms=int(payload.get("ttl_ms", 120_000)),
        )
        operation = "reserved"
    elif path == "/datadist/reservations/renew":
        reservation = manager.renew(
            str(payload.get("reservation_id", "")),
            ttl_ms=int(payload.get("ttl_ms", 120_000)),
        )
        operation = "renewed"
    elif path == "/datadist/reservations/publish":
        reservation_id = str(payload.get("reservation_id", ""))
        try:
            reservation = manager.publish(
                reservation_id,
                block_hashes=[_decode_packed_block_hash(value) for value in payload.get("block_hashes", [])],
            )
        except BlockAlreadyPublishedError as exc:
            reservation = manager.rollback(reservation_id)
            logger.info(
                "Jiuwen KVBM skipped duplicate destination publish and rolled back "
                "reservation: reservation_id=%s existing_block_ids=%s",
                reservation_id,
                list(exc.existing_block_ids),
            )
            return {
                "ok": True,
                "skipped": True,
                "reason": "target_already_published",
                "already_published": True,
                "existing_block_ids": list(exc.existing_block_ids),
                "reservation": reservation.to_dict(),
            }
        operation = "published"
    elif path == "/datadist/reservations/commit":
        reservation = manager.commit(
            str(payload.get("reservation_id", "")),
            block_hashes=[str(value) for value in payload.get("block_hashes", [])],
        )
        operation = "committed"
    elif path == "/datadist/reservations/rollback":
        reservation = manager.rollback(str(payload.get("reservation_id", "")))
        operation = "rolled back"
    elif path == "/datadist/reservations/release":
        reservation = manager.release(str(payload.get("reservation_id", "")))
        operation = "released"
    else:
        return {"ok": False, "error": f"unsupported reservation path: {path}"}
    logger.info(
        "Jiuwen KVBM %s vLLM destination blocks: reservation_id=%s block_ids=%s",
        operation,
        reservation.reservation_id,
        reservation.block_ids,
    )
    return {"ok": True, "reservation": reservation.to_dict()}


def _handle_local_source_lease_request(
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """在 scheduler 进程处理源 block lease 请求。

    参数:
        path: source lease status、acquire、renew 或 release 路径。
        payload: 对应操作的 JSON 请求体。

    返回:
        包含 ok 和 lease/stats 的可序列化响应。
    """

    manager = _process_source_block_leases()
    if path == "/datadist/source-leases/status":
        return {"ok": True, "source_leases": manager.stats()}
    if path == "/datadist/source-leases/acquire":
        lease = manager.acquire(
            _required_int_list(payload, "block_ids"),
            ttl_ms=int(payload.get("ttl_ms", 30_000)),
        )
        operation = "acquired"
    elif path == "/datadist/source-leases/renew":
        lease = manager.renew(
            str(payload.get("lease_id", "")),
            ttl_ms=int(payload.get("ttl_ms", 30_000)),
        )
        operation = "renewed"
    elif path == "/datadist/source-leases/release":
        lease = manager.release(str(payload.get("lease_id", "")))
        operation = "released"
    else:
        return {"ok": False, "error": f"unsupported source lease path: {path}"}

    snapshot = lease.to_dict()
    snapshot["block_hashes"] = [
        (_encode_packed_block_hash(block_hash) if block_hash is not None else None) for block_hash in lease.block_hashes
    ]
    logger.info(
        "Jiuwen KVBM %s source block lease: lease_id=%s block_ids=%s state=%s",
        operation,
        lease.lease_id,
        lease.block_ids,
        lease.state.value,
    )
    return {"ok": True, "lease": snapshot}


def _local_pd_export(payload: dict[str, Any]) -> dict[str, Any]:
    """在 scheduler 进程按 request hash 原子定位并 pin P 端物理 block。

    参数:
        payload: 包含 request_id、可选 block_limit 和 ttl_ms 的请求体。
            block_limit 按逻辑 token block 计数，缺省时导出全部完整 prompt block。

    返回:
        成功时返回 ACTIVE source lease、按传输顺序排列的物理 block、完整
        packed hash 及逻辑 block/group 映射；请求尚未完成发布时返回明确错误。
    """

    request_id = str(payload.get("request_id", "")).strip()
    if not request_id:
        raise ValueError("request_id is required")
    raw_limit = payload.get("block_limit")
    block_limit = int(raw_limit) if raw_limit is not None else None
    if block_limit is not None and block_limit <= 0:
        raise ValueError("block_limit must be positive")

    wait_timeout_ms = max(0, int(payload.get("wait_timeout_ms", 0)))
    wait_event: threading.Event | None = None
    wait_started = time.monotonic()
    with _VLLM_BLOCK_POOL_LOCK:
        pending = next(
            (
                _PD_SOURCE_EXPORT_INTENTS[alias]
                for alias in _pd_request_id_aliases(request_id)
                if alias in _PD_SOURCE_EXPORT_INTENTS
            ),
            None,
        )
        if pending is not None and not pending.capture_complete:
            wait_event = pending.capture_ready
    if wait_event is not None and wait_timeout_ms > 0:
        wait_event.wait(wait_timeout_ms / 1000.0)
    capture_wait_ms = (time.monotonic() - wait_started) * 1000.0

    with _VLLM_BLOCK_POOL_LOCK:
        captured = next(
            (
                _PD_SOURCE_EXPORT_INTENTS[alias]
                for alias in _pd_request_id_aliases(request_id)
                if alias in _PD_SOURCE_EXPORT_INTENTS
            ),
            None,
        )
        if captured is not None and captured.error:
            return {
                "ok": False,
                "error": "P request-scoped block capture failed: " + captured.error,
                "request_id": request_id,
                "handoff_id": captured.handoff_id,
            }
        if captured is not None and not captured.capture_complete:
            return {
                "ok": False,
                "pending": True,
                "retryable": True,
                "error": "P request blocks have not reached the release capture hook",
                "request_id": request_id,
                "handoff_id": captured.handoff_id,
                "capture_wait_ms": capture_wait_ms,
            }
        if captured is not None and captured.source_lease_id:
            dispatch_to_scheduler_seen_ms = (
                max(0.0, (captured.scheduler_seen_at - captured.created_at) * 1000.0)
                if captured.scheduler_seen_at > 0
                else 0.0
            )
            scheduler_seen_to_kv_ready_ms = (
                max(
                    0.0,
                    (captured.kv_ready_at - captured.scheduler_seen_at) * 1000.0,
                )
                if captured.scheduler_seen_at > 0 and captured.kv_ready_at > 0
                else 0.0
            )
            kv_ready_to_export_ready_ms = (
                max(
                    0.0,
                    (captured.capture_ready_at - captured.kv_ready_at) * 1000.0,
                )
                if captured.kv_ready_at > 0 and captured.capture_ready_at > 0
                else 0.0
            )
            scheduler_seen_to_export_ready_ms = (
                max(
                    0.0,
                    (captured.capture_ready_at - captured.scheduler_seen_at) * 1000.0,
                )
                if captured.scheduler_seen_at > 0 and captured.capture_ready_at > 0
                else 0.0
            )
            lease = _process_source_block_leases().renew(
                captured.source_lease_id,
                ttl_ms=int(payload.get("ttl_ms", captured.ttl_ms)),
            )
            selected_indexes = []
            for matched_matched_matched_selected_index, matched_matched_selected_item in enumerate(
                captured.block_groups
            ):
                if block_limit is None or int(matched_matched_selected_item.get("logical_index", -1)) < block_limit:
                    selected_indexes.append(matched_matched_matched_selected_index)
            source_blocks = [captured.source_blocks[index] for index in selected_indexes]
            block_groups = [captured.block_groups[index] for index in selected_indexes]
            block_hashes = [captured.block_hashes[index] for index in selected_indexes]
            logical_positions = {int(item.get("logical_index", -1)) for item in block_groups}
            partial_valid_tokens = max(
                (int(item.get("valid_token_count", 0)) for item in block_groups if item.get("request_scoped_partial")),
                default=0,
            )
            logger.info(
                "Jiuwen KVBM exported captured P request blocks: request_id=%s "
                "logical_blocks=%d physical_blocks=%d partial_valid_tokens=%d "
                "lease_id=%s",
                request_id,
                len(logical_positions),
                len(source_blocks),
                partial_valid_tokens,
                lease.lease_id,
            )
            return {
                "ok": True,
                "request_id": request_id,
                "logical_block_count": len(logical_positions),
                "full_logical_block_count": len(
                    {
                        int(item.get("logical_index", -1))
                        for item in block_groups
                        if not item.get("request_scoped_partial")
                    }
                ),
                "physical_block_count": len(source_blocks),
                "source_blocks": source_blocks,
                "block_hashes": block_hashes,
                "block_groups": block_groups,
                "token_block_fingerprints": list(captured.token_block_fingerprints),
                "partial_token_fingerprint": captured.partial_token_fingerprint,
                "partial_valid_tokens": partial_valid_tokens,
                "request_scoped_partial": partial_valid_tokens > 0,
                "semantic_plain_text": captured.semantic_plain_text,
                "capture_wait_ms": capture_wait_ms,
                "p_dispatch_to_scheduler_seen_ms": dispatch_to_scheduler_seen_ms,
                "p_worker_request_received_to_scheduler_seen_ms": (
                    captured.worker_request_received_to_scheduler_seen_ms
                ),
                "p_worker_request_received_timing_source": (captured.worker_request_received_timing_source),
                "p_scheduler_queue_cycles": captured.scheduler_wait_cycles,
                "p_scheduler_seen_cycle_id": captured.scheduler_seen_cycle_id,
                "p_first_scheduled_cycle_id": captured.first_scheduled_cycle_id,
                "p_kv_ready_cycle_id": captured.kv_ready_cycle_id,
                "p_scheduler_compute_cycles": max(
                    0,
                    captured.kv_ready_cycle_id - captured.first_scheduled_cycle_id + 1,
                )
                if captured.first_scheduled_cycle_id > 0
                else 0,
                "p_first_scheduled_token_count": (captured.first_scheduled_token_count),
                "p_scheduler_running_request_count": (captured.scheduler_running_request_count),
                "p_scheduler_waiting_request_count": (captured.scheduler_waiting_request_count),
                "p_first_scheduled_running_request_count": (captured.first_scheduled_running_request_count),
                "p_first_scheduled_waiting_request_count": (captured.first_scheduled_waiting_request_count),
                "p_first_scheduled_cycle_ms": captured.first_scheduled_cycle_ms,
                "p_scheduler_seen_to_kv_ready_ms": scheduler_seen_to_kv_ready_ms,
                "p_kv_ready_to_export_ready_ms": kv_ready_to_export_ready_ms,
                "p_scheduler_seen_to_export_ready_ms": (scheduler_seen_to_export_ready_ms),
                "p_preemption_count": captured.preemption_count,
                "p_last_preempted_computed_tokens": (captured.last_preempted_computed_tokens),
                "lease": lease.to_dict(),
            }
        request_hashes = next(
            (
                _PD_REQUEST_BLOCK_HASHES[alias]
                for alias in _pd_request_id_aliases(request_id)
                if alias in _PD_REQUEST_BLOCK_HASHES
            ),
            None,
        )
        if not request_hashes:
            known_request_ids = list(_PD_REQUEST_BLOCK_HASHES)[-8:]
            return {
                "ok": False,
                "error": "P request block hashes are not available",
                "request_id": request_id,
                "request_id_fingerprint": hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16],
                "known_request_count": len(_PD_REQUEST_BLOCK_HASHES),
                "known_request_fingerprints": [
                    hashlib.sha256(value.encode("utf-8")).hexdigest()[:16] for value in known_request_ids
                ],
            }
        token_fingerprints = next(
            (
                _PD_REQUEST_TOKEN_BLOCK_FINGERPRINTS[alias]
                for alias in _pd_request_id_aliases(request_id)
                if alias in _PD_REQUEST_TOKEN_BLOCK_FINGERPRINTS
            ),
            (),
        )
        selected_hashes = request_hashes[:block_limit]
        block_pool = _latest_reservable_block_pool()
        if block_pool is None:
            return {"ok": False, "error": "vLLM BlockPool is not ready"}
        resolved, missing_positions = _resolve_cached_request_blocks(
            block_pool,
            selected_hashes,
        )
        if missing_positions:
            return {
                "ok": False,
                "error": "P request blocks are not fully published in prefix cache",
                "request_id": request_id,
                "requested_logical_blocks": len(selected_hashes),
                "resolved_logical_blocks": len(selected_hashes) - len(missing_positions),
                "missing_logical_positions": missing_positions[:16],
            }

        source_blocks = [item["block_id"] for item in resolved]
        if len(set(source_blocks)) != len(source_blocks):
            return {
                "ok": False,
                "error": "P request resolves to duplicate physical block ids",
                "request_id": request_id,
            }
        lease = _process_source_block_leases().acquire(
            source_blocks,
            ttl_ms=int(payload.get("ttl_ms", 30_000)),
        )

    lease_snapshot = lease.to_dict()
    lease_snapshot["block_hashes"] = [_encode_packed_block_hash(block_hash) for block_hash in lease.block_hashes]
    logger.info(
        "Jiuwen KVBM exported and pinned P request blocks: request_id=%s "
        "logical_blocks=%d physical_blocks=%d lease_id=%s",
        request_id,
        len(selected_hashes),
        len(source_blocks),
        lease.lease_id,
    )
    return {
        "ok": True,
        "request_id": request_id,
        "logical_block_count": len(selected_hashes),
        "physical_block_count": len(source_blocks),
        "source_blocks": source_blocks,
        "block_hashes": lease_snapshot["block_hashes"],
        "block_groups": resolved,
        "token_block_fingerprints": list(token_fingerprints[:len(selected_hashes)]),
        "semantic_plain_text": len(token_fingerprints) >= len(selected_hashes),
        "capture_wait_ms": capture_wait_ms,
        "p_dispatch_to_scheduler_seen_ms": 0.0,
        "p_worker_request_received_to_scheduler_seen_ms": 0.0,
        "p_worker_request_received_timing_source": "unavailable",
        "p_scheduler_queue_cycles": 0,
        "p_scheduler_seen_cycle_id": 0,
        "p_first_scheduled_cycle_id": 0,
        "p_kv_ready_cycle_id": 0,
        "p_scheduler_compute_cycles": 0,
        "p_first_scheduled_token_count": 0,
        "p_scheduler_running_request_count": 0,
        "p_scheduler_waiting_request_count": 0,
        "p_first_scheduled_running_request_count": 0,
        "p_first_scheduled_waiting_request_count": 0,
        "p_first_scheduled_cycle_ms": 0.0,
        "p_scheduler_seen_to_kv_ready_ms": 0.0,
        "p_kv_ready_to_export_ready_ms": 0.0,
        "p_scheduler_seen_to_export_ready_ms": 0.0,
        "lease": lease_snapshot,
    }


def _resolve_cached_request_blocks(
    block_pool: Any,
    request_hashes: tuple[Any, ...],
) -> tuple[list[dict[str, Any]], list[int]]:
    """按请求顺序从 BlockPool prefix-cache 索引解析物理 block。

    参数:
        block_pool: vLLM BlockPool 实例。
        request_hashes: 按 prompt 顺序排列且不含 cache group ID 的逻辑 hash。

    返回:
        ``(resolved, missing_positions)``。resolved 按逻辑位置、group ID 排序，
        每项包含 logical_index、group_id、block_id 和完整 block_hash 描述。
    """

    cached_map = getattr(block_pool, "cached_block_hash_to_block", None)
    raw_cache = getattr(cached_map, "_cache", cached_map)
    if not isinstance(raw_cache, dict):
        raise ValueError("vLLM BlockPool does not expose prefix-cache hash index")

    indexed: dict[str, list[tuple[Any, Any]]] = {}
    for packed_hash, blocks in raw_cache.items():
        for alias in _block_hash_aliases(packed_hash):
            indexed.setdefault(alias, []).extend((packed_hash, block) for block in _iter_kv_cache_blocks(blocks))

    resolved: list[dict[str, Any]] = []
    missing_positions: list[int] = []
    for logical_index, request_hash in enumerate(request_hashes):
        candidates: dict[int, tuple[Any, Any]] = {}
        for alias in _block_hash_aliases(request_hash):
            for packed_hash, block in indexed.get(alias, []):
                block_id = getattr(block, "block_id", None)
                block_specific_hash = getattr(block, "block_hash", packed_hash)
                if block_id is None or block_specific_hash is None:
                    continue
                if isinstance(packed_hash, bytes | bytearray) and (
                    not isinstance(block_specific_hash, bytes | bytearray)
                    or bytes(block_specific_hash) != bytes(packed_hash)
                ):
                    # Ignore a stale index entry; never export a repurposed block.
                    continue
                if bool(getattr(block, "is_null", False)):
                    continue
                candidates[int(block_id)] = (block_specific_hash, block)
        if not candidates:
            missing_positions.append(logical_index)
            continue
        # A cache hash may map to multiple physical replicas in the same group.
        # Export one replica per group, not one transfer entry per physical copy.
        by_group: dict[int, bytes] = {}
        ordered = sorted(
            candidates.items(),
            key=lambda item: (
                _packed_hash_group_id(item[1][0]),
                0 if int(getattr(item[1][1], "ref_cnt", 0)) > 0 else 1,
                item[0],
            ),
        )
        for block_id, (packed_hash, _block) in ordered:
            descriptor = _encode_packed_block_hash(packed_hash)
            group_id = descriptor["group_id"]
            previous = by_group.get(group_id)
            if previous is not None:
                if previous != bytes(packed_hash):
                    raise RuntimeError(
                        "P request hash ambiguously resolves to different packed hashes "
                        f"in one cache group: logical_index={logical_index}, group_id={group_id}"
                    )
                continue
            by_group[group_id] = bytes(packed_hash)
            resolved.append(
                {
                    "logical_index": logical_index,
                    "group_id": descriptor["group_id"],
                    "block_id": block_id,
                    "block_hash": descriptor,
                }
            )
    return resolved, missing_positions


def _packed_hash_group_id(block_hash: Any) -> int:
    """读取 vLLM packed block hash 的 cache group ID。

    参数:
        block_hash: 带 4 字节 group ID 后缀的 vLLM block hash。

    返回:
        可解析时返回非负 group ID；格式不兼容时返回最大 32 位整数用于排序。
    """

    if isinstance(block_hash, bytes | bytearray) and len(block_hash) > 4:
        return int.from_bytes(bytes(block_hash)[-4:], "big", signed=False)
    return (1 << 32) - 1


def _local_pd_import_prepare(payload: dict[str, Any]) -> dict[str, Any]:
    """在 D scheduler 注册请求门控，阻止目标请求提前执行本地 prefill。

    参数:
        payload: 包含 handoff_id、request_id 和可选 ttl_ms 的请求体。

    返回:
        包含 created、request_seen 和 gate 快照的响应；相同 handoff 重试幂等。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    request_id = str(payload.get("request_id", "")).strip()
    if not handoff_id or not request_id:
        raise ValueError("handoff_id and request_id are required")
    ttl_ms = int(payload.get("ttl_ms", 30_000))
    admission_ticket_id = str(payload.get("admission_ticket_id", "")).strip()
    require_existing = bool(payload.get("require_existing", False))
    if ttl_ms < 1_000 or ttl_ms > 600_000:
        raise ValueError("D request import gate ttl_ms must be between 1000 and 600000")
    if admission_ticket_id:
        with _VLLM_BLOCK_POOL_LOCK:
            _expire_pd_decode_admission_tickets_locked()
            admission_ticket = _PD_DECODE_ADMISSION_TICKETS.get(handoff_id)
            if admission_ticket is None or admission_ticket.ticket_id != admission_ticket_id:
                return {
                    "ok": False,
                    "error": "D decode admission ticket is missing or expired",
                    "request_id": request_id,
                }
    with _PD_TARGET_LOCK:
        existing = _PD_TARGET_GATES.get(request_id)
        if existing is not None:
            if existing.handoff_id != handoff_id:
                raise ValueError(f"D request {request_id} is already bound to handoff {existing.handoff_id}")
            gate = existing
            if admission_ticket_id:
                if gate.admission_ticket_id not in {"", admission_ticket_id}:
                    raise ValueError("admission_ticket_id does not match D request gate")
                gate.admission_ticket_id = admission_ticket_id
            gate.expires_at = max(
                gate.expires_at,
                time.monotonic() + ttl_ms / 1000.0,
            )
            created = False
        else:
            if require_existing:
                return {
                    "ok": False,
                    "error": "D request import gate disappeared before queued import started",
                    "request_id": request_id,
                }
            request_seen = any(request_id in getattr(scheduler, "requests", {}) for scheduler in _OBSERVED_SCHEDULERS)
            if request_seen:
                raise RuntimeError("D request already exists; prepare import before dispatching it")
            gate = _PDTargetRequestGate(
                handoff_id=handoff_id,
                request_id=request_id,
                created_at=time.monotonic(),
                expires_at=time.monotonic() + ttl_ms / 1000.0,
                admission_ticket_id=admission_ticket_id,
            )
            _PD_TARGET_GATES[request_id] = gate
            created = True
    return {
        "ok": True,
        "created": created,
        "gate": _pd_target_gate_snapshot(gate),
    }


def _local_scheduler_owned_pd_import_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """等待Scheduler为D请求分配并校验原生目标block布局。

    参数:
        payload: 包含handoff_id、request_id、P端block_groups、token指纹和
            可选timeout_ms的请求体。

    返回:
        成功时返回scheduler_owned计划；容量暂不可用时在截止时间内按Scheduler
        周期重试，超时或语义不匹配时返回ok=False。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    request_id = str(payload.get("request_id", "")).strip()
    if not handoff_id or not request_id:
        raise ValueError("handoff_id and request_id are required")
    block_groups = payload.get("block_groups")
    if not isinstance(block_groups, list | tuple) or not block_groups:
        raise ValueError("block_groups must be a non-empty list")
    if not all(isinstance(item, dict) for item in block_groups):
        raise ValueError("every block_groups item must be an object")
    timeout_seconds = max(0.1, int(payload.get("timeout_ms", 10_000)) / 1000.0)
    deadline = time.monotonic() + timeout_seconds
    while True:
        with _PD_TARGET_LOCK:
            gate = _PD_TARGET_GATES.get(request_id)
            if gate is None or gate.handoff_id != handoff_id:
                return {
                    "ok": False,
                    "error": "D request import gate is not prepared",
                    "request_id": request_id,
                }
            if not gate.request_seen:
                return {
                    "ok": False,
                    "error": "D request has not reached scheduler yet",
                    "request_id": request_id,
                }
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return {
                    "ok": False,
                    "error": "timed out waiting for vLLM Scheduler target blocks",
                    "request_id": request_id,
                }
            task = _PDTargetPlanTask(
                request_id=gate.engine_request_id or request_id,
                handoff_id=handoff_id,
                block_groups=tuple(dict(item) for item in block_groups),
                token_block_fingerprints=tuple(str(value) for value in payload.get("token_block_fingerprints", [])),
                partial_token_fingerprint=str(payload.get("partial_token_fingerprint", "")),
                partial_valid_tokens=int(payload.get("partial_valid_tokens", 0)),
                semantic_plain_text=bool(payload.get("semantic_plain_text", False)),
                reservation_ttl_ms=0,
                timeout_seconds=remaining_seconds,
                ownership_mode=SchedulerKVOwnershipMode.SCHEDULER,
            )
            _PD_TARGET_PLAN_TASKS.append(task)
        if not task.completed.wait(remaining_seconds):
            with _PD_TARGET_LOCK:
                if task in _PD_TARGET_PLAN_TASKS:
                    _PD_TARGET_PLAN_TASKS.remove(task)
                    return {
                        "ok": False,
                        "error": "timed out before vLLM Scheduler allocated target blocks",
                        "request_id": request_id,
                    }
            task.completed.wait()
        if task.result is not None:
            return task.result
        if task.capacity_required_blocks <= 0:
            return {"ok": False, "error": task.error, "request_id": request_id}
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def _local_pd_import_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """提交 D delta import plan，并等待 scheduler 原子 pin/预留完成。

    参数:
        payload: 包含 handoff_id、request_id、block_groups、token 指纹、
            reservation_ttl_ms 和可选 timeout_ms 的请求体。

    返回:
        成功时返回完整布局、复用数量、稀疏传输位置和 reservation 快照；
        gate 尚未观察请求、超时或语义校验失败时返回 ok=False。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    request_id = str(payload.get("request_id", "")).strip()
    if not handoff_id or not request_id:
        raise ValueError("handoff_id and request_id are required")
    block_groups = payload.get("block_groups")
    if not isinstance(block_groups, list | tuple) or not block_groups:
        raise ValueError("block_groups must be a non-empty list")
    if not all(isinstance(item, dict) for item in block_groups):
        raise ValueError("every block_groups item must be an object")
    timeout_seconds = max(0.1, int(payload.get("timeout_ms", 10_000)) / 1000.0)
    reservation_ttl_ms = int(payload.get("reservation_ttl_ms", 120_000))
    with _PD_TARGET_LOCK:
        gate = _PD_TARGET_GATES.get(request_id)
        if gate is None or gate.handoff_id != handoff_id:
            return {
                "ok": False,
                "error": "D request import gate is not prepared",
                "request_id": request_id,
            }
        if not gate.request_seen:
            return {
                "ok": False,
                "error": "D request has not reached scheduler yet",
                "request_id": request_id,
            }
        engine_request_id = gate.engine_request_id or request_id
        preplan = gate.preplan

    if preplan is not None:
        scheduler = next(
            (
                candidate
                for candidate in _OBSERVED_SCHEDULERS
                if engine_request_id in getattr(candidate, "requests", {})
            ),
            None,
        )
        request = getattr(scheduler, "requests", {}).get(engine_request_id) if scheduler is not None else None
        if scheduler is not None and request is not None:
            preplan_started = time.perf_counter()
            hash_validate_started = time.perf_counter()
            try:
                request_hashes = tuple(_flatten_request_block_hashes(getattr(request, "block_hashes", None)))
                token_fingerprints = payload.get("token_block_fingerprints", [])
                target_packed_hashes = tuple(
                    _validate_pd_target_request_hashes(
                        request,
                        request_hashes,
                        tuple(dict(item) for item in block_groups),
                        tuple(str(value) for value in token_fingerprints),
                        str(payload.get("partial_token_fingerprint", "")),
                        int(payload.get("partial_valid_tokens", 0)),
                        semantic_plain_text=bool(payload.get("semantic_plain_text", False)),
                        block_size=int(getattr(scheduler, "block_size", 0) or 0),
                    )
                )
            except Exception as exc:
                with _PD_TARGET_LOCK:
                    if gate.preplan is preplan:
                        gate.preplan = None
                _rollback_pd_target_preplan(
                    preplan,
                    reason=f"P/D preplan validation failed: {exc}",
                )
                return {"ok": False, "error": str(exc), "request_id": request_id}
            hash_validate_ms = (time.perf_counter() - hash_validate_started) * 1000.0
            layout_matches = target_packed_hashes == preplan.target_packed_hashes and _pd_block_group_signature(
                block_groups
            ) == _pd_block_group_signature(preplan.block_groups)
            if layout_matches:
                reservation_id = str(
                    (preplan.result.get("reservation") or {}).get(
                        "reservation_id",
                        "",
                    )
                )
                try:
                    reservation = _process_block_reservations().renew(
                        reservation_id,
                        ttl_ms=reservation_ttl_ms,
                    )
                except Exception as exc:
                    with _PD_TARGET_LOCK:
                        if gate.preplan is preplan:
                            gate.preplan = None
                    _rollback_pd_target_preplan(
                        preplan,
                        reason=f"D early preplan renewal failed: {exc}",
                    )
                    preplan = None
                else:
                    with _PD_TARGET_LOCK:
                        current_gate = _PD_TARGET_GATES.get(request_id)
                        adopted = current_gate is gate and gate.preplan is preplan
                        if adopted:
                            gate.preplan = None
                    if not adopted:
                        _rollback_pd_target_preplan(
                            preplan,
                            reason="D request gate changed during preplan adoption",
                        )
                        return {
                            "ok": False,
                            "error": "D request import gate changed during preplan adoption",
                            "request_id": request_id,
                        }
                    result = dict(preplan.result)
                    result["request_id"] = engine_request_id
                    result["reservation"] = reservation.to_dict()
                    result["delta_preplanned"] = True
                    result["delta_plan_task_queue_ms"] = 0.0
                    result["delta_hash_validate_ms"] = hash_validate_ms
                    result["delta_reserve_ms"] = preplan.reserve_ms
                    result["delta_plan_execution_ms"] = (time.perf_counter() - preplan_started) * 1000.0
                    result["delta_preplan_age_ms"] = max(
                        0.0,
                        (time.monotonic() - preplan.created_at) * 1000.0,
                    )
                    return result
            with _PD_TARGET_LOCK:
                if gate.preplan is preplan:
                    gate.preplan = None
            _rollback_pd_target_preplan(
                preplan,
                reason="P export layout differs from D early preplan",
            )
            preplan = None

    if preplan is not None:
        with _PD_TARGET_LOCK:
            if gate.preplan is preplan:
                gate.preplan = None
        _rollback_pd_target_preplan(
            preplan,
            reason="D scheduler request is unavailable while adopting preplan",
        )

    deadline = time.monotonic() + timeout_seconds
    inflight_wait_started: float | None = None
    inflight_retry_count = 0
    capacity_wait_started: float | None = None
    capacity_retry_count = 0
    while True:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            _process_block_reservations().release_pending_delta(handoff_id)
            return {
                "ok": False,
                "error": "timed out waiting for in-flight D delta import blocks",
                "request_id": request_id,
            }
        with _PD_TARGET_LOCK:
            gate = _PD_TARGET_GATES.get(request_id)
            if gate is None or gate.handoff_id != handoff_id:
                _process_block_reservations().release_pending_delta(handoff_id)
                return {
                    "ok": False,
                    "error": "D request import gate is not prepared",
                    "request_id": request_id,
                }
            task = _PDTargetPlanTask(
                request_id=engine_request_id,
                handoff_id=handoff_id,
                block_groups=tuple(dict(item) for item in block_groups),
                token_block_fingerprints=tuple(str(value) for value in payload.get("token_block_fingerprints", [])),
                partial_token_fingerprint=str(payload.get("partial_token_fingerprint", "")),
                partial_valid_tokens=int(payload.get("partial_valid_tokens", 0)),
                semantic_plain_text=bool(payload.get("semantic_plain_text", False)),
                reservation_ttl_ms=reservation_ttl_ms,
                timeout_seconds=remaining_seconds,
            )
            _PD_TARGET_PLAN_TASKS.append(task)
        if not task.completed.wait(remaining_seconds):
            with _PD_TARGET_LOCK:
                if task in _PD_TARGET_PLAN_TASKS:
                    _PD_TARGET_PLAN_TASKS.remove(task)
                    _process_block_reservations().release_pending_delta(handoff_id)
                    return {
                        "ok": False,
                        "error": "timed out before D scheduler started delta import plan",
                        "request_id": request_id,
                    }
            task.completed.wait()
        if task.result is not None:
            result = task.result
            result["delta_inflight_retry_count"] = inflight_retry_count
            result["delta_inflight_wait_ms"] = (
                max(0.0, (time.monotonic() - inflight_wait_started) * 1000.0)
                if inflight_wait_started is not None
                else 0.0
            )
            result["delta_capacity_retry_count"] = capacity_retry_count
            result["delta_capacity_wait_ms"] = (
                max(0.0, (time.monotonic() - capacity_wait_started) * 1000.0)
                if capacity_wait_started is not None
                else 0.0
            )
            if inflight_retry_count:
                logger.info(
                    "Jiuwen KVBM D delta import reused in-flight publication: "
                    "handoff_id=%s request_id=%s retries=%s wait_ms=%.3f",
                    handoff_id,
                    request_id,
                    inflight_retry_count,
                    result["delta_inflight_wait_ms"],
                )
            if capacity_retry_count:
                logger.info(
                    "Jiuwen KVBM D delta import admitted after capacity wait: "
                    "handoff_id=%s request_id=%s retries=%s wait_ms=%.3f",
                    handoff_id,
                    request_id,
                    capacity_retry_count,
                    result["delta_capacity_wait_ms"],
                )
            return result
        if task.capacity_required_blocks > 0:
            if capacity_wait_started is None:
                capacity_wait_started = time.monotonic()
            capacity_retry_count += 1
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds > 0 and _process_block_reservations().wait_for_free_blocks(
                task.capacity_required_blocks,
                timeout_seconds=remaining_seconds,
            ):
                continue
            _process_block_reservations().release_pending_delta(handoff_id)
            return {
                "ok": False,
                "error": "timed out waiting for D BlockPool capacity",
                "request_id": request_id,
            }
        if not task.inflight_reservation_ids:
            _process_block_reservations().release_pending_delta(handoff_id)
            return {"ok": False, "error": task.error, "request_id": request_id}
        if inflight_wait_started is None:
            inflight_wait_started = time.monotonic()
        inflight_retry_count += 1
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0 or not _process_block_reservations().wait_for_inflight_reservations(
            task.inflight_reservation_ids,
            timeout_seconds=max(0.0, remaining_seconds),
        ):
            _process_block_reservations().release_pending_delta(handoff_id)
            return {
                "ok": False,
                "error": "timed out waiting for in-flight D delta import blocks",
                "request_id": request_id,
            }


def _local_pd_import_attach(payload: dict[str, Any]) -> dict[str, Any]:
    """提交一个 D request attach 任务并等待 scheduler 线程原子执行。

    参数:
        payload: 包含 handoff_id、request_id、reservation_id、block_hashes、
            block_groups 和可选 timeout_ms 的请求体。

    返回:
        scheduler attach 成功结果；等待超时或提交失败时返回 ok=False。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    request_id = str(payload.get("request_id", "")).strip()
    reservation_id = str(payload.get("reservation_id", "")).strip()
    if not handoff_id or not request_id or not reservation_id:
        raise ValueError("handoff_id, request_id and reservation_id are required")
    block_hashes = payload.get("block_hashes")
    block_groups = payload.get("block_groups")
    if not isinstance(block_hashes, list | tuple) or not block_hashes:
        raise ValueError("block_hashes must be a non-empty list")
    if not isinstance(block_groups, list | tuple) or not block_groups:
        raise ValueError("block_groups must be a non-empty list")
    if not all(isinstance(item, dict) for item in block_groups):
        raise ValueError("every block_groups item must be an object")
    timeout_seconds = max(0.1, int(payload.get("timeout_ms", 10_000)) / 1000.0)
    with _PD_TARGET_LOCK:
        gate = _PD_TARGET_GATES.get(request_id)
        if gate is None or gate.handoff_id != handoff_id:
            return {
                "ok": False,
                "error": "D request import gate is not prepared",
                "request_id": request_id,
            }
        if not gate.request_seen:
            return {
                "ok": False,
                "error": "D request has not reached scheduler yet",
                "request_id": request_id,
            }
        task = _PDTargetAttachTask(
            request_id=gate.engine_request_id or request_id,
            handoff_id=handoff_id,
            reservation_id=reservation_id,
            block_hashes=tuple(block_hashes),
            block_groups=tuple(dict(item) for item in block_groups),
            token_block_fingerprints=tuple(str(value) for value in payload.get("token_block_fingerprints", [])),
            partial_token_fingerprint=str(payload.get("partial_token_fingerprint", "")),
            partial_valid_tokens=int(payload.get("partial_valid_tokens", 0)),
            semantic_plain_text=bool(payload.get("semantic_plain_text", False)),
            ownership_mode=SchedulerKVOwnershipMode(
                str(
                    payload.get(
                        "ownership_mode",
                        _pd_block_ownership_mode().value,
                    )
                ).lower()
            ),
            timeout_seconds=timeout_seconds,
        )
        _PD_TARGET_ATTACH_TASKS.append(task)
    if not task.completed.wait(timeout_seconds):
        with _PD_TARGET_LOCK:
            if task in _PD_TARGET_ATTACH_TASKS:
                _PD_TARGET_ATTACH_TASKS.remove(task)
                return {
                    "ok": False,
                    "error": "timed out before D scheduler started request attach",
                    "request_id": request_id,
                }
        # scheduler 已经取走任务时，必须等待原子 attach 得出终态，不能让控制
        # 线程并发 rollback 一个可能已经提交给请求的 reservation。
        task.completed.wait()
    if task.error is not None:
        return {"ok": False, "error": task.error, "request_id": request_id}
    if not (task.result is not None):
        raise RuntimeError("Required invariant failed: task.result is not None")
    return task.result


def _local_pd_import_release(payload: dict[str, Any]) -> dict[str, Any]:
    """解除 D request import gate，使失败请求可以退化为本地 prefill。

    参数:
        payload: 包含 request_id 和可选 handoff_id 的请求体。

    返回:
        包含 released 和 gate 快照的幂等响应。
    """

    request_id = str(payload.get("request_id", "")).strip()
    handoff_id = str(payload.get("handoff_id", "")).strip()
    if not request_id:
        raise ValueError("request_id is required")
    with _PD_TARGET_LOCK:
        gate = _PD_TARGET_GATES.get(request_id)
        if gate is None:
            with _VLLM_BLOCK_POOL_LOCK:
                ticket_released = (
                    _release_pd_decode_admission_ticket_locked(
                        handoff_id,
                        ticket_id=str(payload.get("admission_ticket_id", "")).strip(),
                    )
                    if handoff_id
                    else False
                )
            return {
                "ok": True,
                "released": ticket_released,
                "request_id": request_id,
            }
        if handoff_id and gate.handoff_id != handoff_id:
            raise ValueError("handoff_id does not match D request import gate")
        _PD_TARGET_GATES.pop(request_id, None)
    with _VLLM_BLOCK_POOL_LOCK:
        ticket_released = _release_pd_decode_admission_ticket_locked(
            gate.handoff_id,
            ticket_id=gate.admission_ticket_id,
        )
    _fail_pending_scheduler_import(
        gate.handoff_id,
        error="D request import gate was released before Scheduler allocation",
    )
    _rollback_pd_target_preplan(
        gate.preplan,
        reason="D request import gate released",
    )
    scheduler_blocks_released = _release_scheduler_owned_pd_allocation(
        gate,
        reason="D request import gate released",
    )
    if _pd_block_ownership_mode() is SchedulerKVOwnershipMode.LEGACY:
        _process_block_reservations().release_pending_delta(gate.handoff_id)
    if _pd_prompt_diagnostics_enabled() and gate.engine_request_id:
        with _PD_TARGET_LOCK:
            if len(_PD_FALLBACK_OBSERVATIONS) >= 2048:
                _PD_FALLBACK_OBSERVATIONS.pop(next(iter(_PD_FALLBACK_OBSERVATIONS)))
            _PD_FALLBACK_OBSERVATIONS[gate.engine_request_id] = {
                "handoff_id": gate.handoff_id,
                "started": time.monotonic(),
                "scheduled": False,
            }
        logger.info(
            "[PDFallbackDiagnostics] handoff_id=%s request_id=%s engine_request_id=%s "
            "phase=gate_released scheduler_blocks_released=%s ticket_released=%s",
            gate.handoff_id,
            gate.request_id,
            gate.engine_request_id,
            scheduler_blocks_released,
            ticket_released,
        )
    return {
        "ok": True,
        "released": True,
        "scheduler_blocks_released": scheduler_blocks_released,
        "admission_ticket_released": ticket_released,
        "gate": _pd_target_gate_snapshot(gate),
    }


def _local_pd_import_status(payload: dict[str, Any]) -> dict[str, Any]:
    """读取 D scheduler 请求门控状态。

    参数:
        payload: 可选包含 request_id 的请求体。

    返回:
        指定请求的 gate，或当前全部 gate 的容量诊断。
    """

    request_id = str(payload.get("request_id", "")).strip()
    block_pool = _latest_reservable_block_pool()
    free_blocks = int(block_pool.get_num_free_blocks()) if block_pool is not None else None
    with _PD_TARGET_LOCK:
        if request_id:
            gate = _PD_TARGET_GATES.get(request_id)
            return {
                "ok": True,
                "gate": _pd_target_gate_snapshot(gate) if gate is not None else None,
                "ownership_mode": _pd_block_ownership_mode().value,
                "free_blocks": free_blocks,
            }
        return {
            "ok": True,
            "gates": [_pd_target_gate_snapshot(gate) for gate in _PD_TARGET_GATES.values()],
            "pending_attach_tasks": len(_PD_TARGET_ATTACH_TASKS),
            "pending_plan_tasks": len(_PD_TARGET_PLAN_TASKS),
            "ownership_mode": _pd_block_ownership_mode().value,
            "free_blocks": free_blocks,
        }


def _local_block_pool_capacity(payload: dict[str, Any]) -> dict[str, Any]:
    """原子读取BlockPool容量并确认当前请求可复用的连续前缀。"""
    block_pool = _latest_reservable_block_pool()
    if block_pool is None:
        return {"ok": False, "error": "vLLM BlockPool is not ready"}
    raw_hashes = payload.get("block_hashes") or []
    if not isinstance(raw_hashes, list | tuple):
        raise ValueError("block_hashes must be a list")
    prompt_blocks = max(int(payload.get("prompt_blocks", 0) or 0), 0)
    decode_blocks = max(int(payload.get("decode_blocks", 0) or 0), 0)
    margin_blocks = max(int(payload.get("margin_blocks", 0) or 0), 0)
    router_pending_reserved_blocks = max(
        int(payload.get("pending_reserved_blocks", 0) or 0),
        0,
    )
    handoff_id = str(payload.get("handoff_id", "")).strip()
    request_id = str(payload.get("request_id", "")).strip()
    ticket_ttl_ms = min(
        max(int(payload.get("ticket_ttl_ms", 120_000) or 120_000), 1_000),
        600_000,
    )
    with _VLLM_BLOCK_POOL_LOCK:
        _expire_pd_decode_admission_tickets_locked()
        free_blocks_before_ticket = int(block_pool.get_num_free_blocks())
        capacity_generation = int(getattr(block_pool, _BLOCK_POOL_CAPACITY_GENERATION_ATTR, -1))
        confirmed_reusable_blocks = 0
        if raw_hashes:
            _, missing_positions = _resolve_cached_request_blocks(
                block_pool,
                tuple(raw_hashes),
            )
            confirmed_reusable_blocks = min(missing_positions, default=len(raw_hashes))
        confirmed_reusable_blocks = min(
            confirmed_reusable_blocks,
            prompt_blocks,
        )
        required_new_blocks = (
            max(
                prompt_blocks - confirmed_reusable_blocks,
                0,
            )
            + decode_blocks
        )
        outstanding_ticket_reserved_blocks = sum(
            ticket.required_new_blocks
            for ticket in _PD_DECODE_ADMISSION_TICKETS.values()
            if ticket.block_pool is block_pool and ticket.handoff_id != handoff_id
        )
        committed_reserved_blocks = max(
            outstanding_ticket_reserved_blocks,
            router_pending_reserved_blocks,
        )
        ticket = None
        if handoff_id and request_id and confirmed_reusable_blocks > 0:
            ticket = _acquire_pd_decode_admission_ticket_locked(
                block_pool,
                handoff_id=handoff_id,
                request_id=request_id,
                request_hashes=tuple(str(value) for value in raw_hashes),
                reusable_logical_blocks=confirmed_reusable_blocks,
                required_new_blocks=required_new_blocks,
                ttl_ms=ticket_ttl_ms,
            )
        free_blocks = int(block_pool.get_num_free_blocks())
        ticket_capacity_admitted = bool(
            ticket is not None and committed_reserved_blocks + required_new_blocks + margin_blocks <= free_blocks
        )
        if ticket is not None and not ticket_capacity_admitted:
            _release_pd_decode_admission_ticket_locked(
                handoff_id,
                ticket_id=ticket.ticket_id,
            )
            ticket = None
            free_blocks = int(block_pool.get_num_free_blocks())
        capacity_generation = int(getattr(block_pool, _BLOCK_POOL_CAPACITY_GENERATION_ATTR, -1))
    return {
        "ok": True,
        "free_blocks": free_blocks,
        "free_blocks_before_ticket": free_blocks_before_ticket,
        "capacity_generation": capacity_generation,
        "confirmed_reusable_blocks": confirmed_reusable_blocks,
        "required_new_blocks": required_new_blocks,
        "margin_blocks": margin_blocks,
        "outstanding_ticket_reserved_blocks": outstanding_ticket_reserved_blocks,
        "router_pending_reserved_blocks": router_pending_reserved_blocks,
        "committed_reserved_blocks": committed_reserved_blocks,
        "ticket_capacity_admitted": ticket_capacity_admitted,
        "queried_block_hashes": len(raw_hashes),
        "admission_ticket_id": ticket.ticket_id if ticket is not None else "",
        "admission_ticket_pinned_blocks": (len(ticket.raw_blocks) if ticket is not None else 0),
    }


def _cached_block_objects_by_id(block_pool: Any) -> dict[int, Any]:
    """返回当前prefix-cache索引中的物理block对象。"""

    cached_map = getattr(block_pool, "cached_block_hash_to_block", None)
    raw_cache = getattr(cached_map, "_cache", cached_map)
    if not isinstance(raw_cache, dict):
        return {}
    result: dict[int, Any] = {}
    for blocks in raw_cache.values():
        for block in _iter_kv_cache_blocks(blocks):
            block_id = getattr(block, "block_id", None)
            if block_id is not None:
                result[int(block_id)] = block
    return result


def _release_pd_decode_admission_ticket_locked(
    handoff_id: str,
    *,
    ticket_id: str = "",
) -> bool:
    """释放一个尚未被Scheduler请求接管的D前缀ticket。"""

    ticket = _PD_DECODE_ADMISSION_TICKETS.get(handoff_id)
    if ticket is None or (ticket_id and ticket.ticket_id != ticket_id):
        return False
    _PD_DECODE_ADMISSION_TICKETS.pop(handoff_id, None)
    if ticket.expiry_timer is not None:
        ticket.expiry_timer.cancel()
    if ticket.raw_blocks:
        ticket.block_pool.free_blocks(list(reversed(ticket.raw_blocks)))
    return True


def _expire_pd_decode_admission_tickets_locked() -> None:
    """惰性回收没有进入D Scheduler的过期准入ticket。"""

    now = time.monotonic()
    for handoff_id, ticket in tuple(_PD_DECODE_ADMISSION_TICKETS.items()):
        if ticket.expires_at <= now:
            _release_pd_decode_admission_ticket_locked(handoff_id)


def _acquire_pd_decode_admission_ticket_locked(
    block_pool: Any,
    *,
    handoff_id: str,
    request_id: str,
    request_hashes: tuple[str, ...],
    reusable_logical_blocks: int,
    required_new_blocks: int,
    ttl_ms: int,
) -> _PDDecodeAdmissionTicket:
    """原子固定容量确认命中的连续前缀，并返回请求级ticket。"""

    existing = _PD_DECODE_ADMISSION_TICKETS.get(handoff_id)
    existing_matches = (
        existing is not None and existing.request_id == request_id and existing.request_hashes == request_hashes
    )
    if existing_matches and existing.block_pool is block_pool:
        existing.required_new_blocks = required_new_blocks
        existing.expires_at = max(
            existing.expires_at,
            time.monotonic() + ttl_ms / 1000.0,
        )
        if existing.expiry_timer is not None:
            existing.expiry_timer.cancel()
        remaining_seconds = max(existing.expires_at - time.monotonic(), 0.001)
        timer = threading.Timer(
            remaining_seconds,
            _expire_pd_decode_admission_ticket,
            args=(handoff_id, existing.ticket_id),
        )
        timer.daemon = True
        existing.expiry_timer = timer
        timer.start()
        return existing
    if existing is not None:
        _release_pd_decode_admission_ticket_locked(handoff_id)

    resolved, missing_positions = _resolve_cached_request_blocks(
        block_pool,
        tuple(request_hashes),
    )
    confirmed = min(missing_positions, default=len(request_hashes))
    confirmed = min(confirmed, reusable_logical_blocks)
    block_by_id = _cached_block_objects_by_id(block_pool)
    prefix_blocks = []
    for resolved_block in resolved:
        if int(resolved_block["logical_index"]) >= confirmed:
            continue
        block = block_by_id.get(int(resolved_block["block_id"]))
        if block is not None:
            prefix_blocks.append(block)
    raw_blocks = tuple(prefix_blocks)
    if not raw_blocks:
        raise RuntimeError("confirmed D prefix blocks disappeared before ticket pin")
    touch = getattr(block_pool, "touch", None)
    if not callable(touch):
        raise RuntimeError("vLLM BlockPool admission ticket requires touch()")
    previous_ref_counts = [int(getattr(block, "ref_cnt", 0)) for block in raw_blocks]
    try:
        touch(list(raw_blocks))
    except Exception:
        acquired = [
            block
            for block, previous in zip(raw_blocks, previous_ref_counts, strict=True)
            if int(getattr(block, "ref_cnt", 0)) > previous
        ]
        if acquired:
            block_pool.free_blocks(list(reversed(acquired)))
        raise
    now = time.monotonic()
    ticket = _PDDecodeAdmissionTicket(
        ticket_id=uuid.uuid4().hex,
        handoff_id=handoff_id,
        request_id=request_id,
        request_hashes=request_hashes,
        raw_blocks=raw_blocks,
        block_pool=block_pool,
        required_new_blocks=required_new_blocks,
        created_at=now,
        expires_at=now + ttl_ms / 1000.0,
    )
    _PD_DECODE_ADMISSION_TICKETS[handoff_id] = ticket
    timer = threading.Timer(
        ttl_ms / 1000.0,
        _expire_pd_decode_admission_ticket,
        args=(handoff_id, ticket.ticket_id),
    )
    timer.daemon = True
    ticket.expiry_timer = timer
    timer.start()
    return ticket


def _expire_pd_decode_admission_ticket(handoff_id: str, ticket_id: str) -> None:
    """由后台timer回收单个过期D前缀ticket。"""

    with _VLLM_BLOCK_POOL_LOCK:
        ticket = _PD_DECODE_ADMISSION_TICKETS.get(handoff_id)
        if ticket is not None and ticket.ticket_id == ticket_id and ticket.expires_at <= time.monotonic():
            _release_pd_decode_admission_ticket_locked(
                handoff_id,
                ticket_id=ticket_id,
            )


def _local_pd_decode_admission_ticket_release(payload: dict[str, Any]) -> dict[str, Any]:
    """显式释放候选淘汰、取消或超时请求持有的D前缀ticket。"""

    handoff_id = str(payload.get("handoff_id", "")).strip()
    if not handoff_id:
        raise ValueError("handoff_id is required")
    with _VLLM_BLOCK_POOL_LOCK:
        released = _release_pd_decode_admission_ticket_locked(
            handoff_id,
            ticket_id=str(payload.get("ticket_id", "")).strip(),
        )
    return {"ok": True, "released": released, "handoff_id": handoff_id}


def _pd_target_gate_snapshot(gate: _PDTargetRequestGate) -> dict[str, Any]:
    """把 D request gate 转换为 JSON 安全快照。

    参数:
        gate: 待转换的门控记录。

    返回:
        包含 handoff_id、request_id、request_seen、age_seconds 和剩余 TTL 的字典。
    """

    return {
        "handoff_id": gate.handoff_id,
        "request_id": gate.request_id,
        "engine_request_id": gate.engine_request_id or None,
        "request_seen": gate.request_seen,
        "preplan_ready": gate.preplan is not None,
        "preplan_error": gate.preplan_error or None,
        "preplan_capacity_waiting": gate.preplan_capacity_waiting,
        "last_capacity_retry_generation": gate.last_capacity_retry_generation,
        "ownership_mode": _pd_block_ownership_mode().value,
        "scheduler_allocation_ready": gate.scheduler_allocation is not None,
        "scheduler_allocation": (
            {
                "request_id": gate.scheduler_allocation.request_id,
                "total_blocks": len(gate.scheduler_allocation.planned_destination_blocks),
                "transfer_blocks": len(gate.scheduler_allocation.transfer_positions),
                "local_computed_tokens": gate.scheduler_allocation.local_computed_tokens,
                "external_computed_tokens": (gate.scheduler_allocation.external_computed_tokens),
                "committed": gate.scheduler_allocation.committed,
                "released": gate.scheduler_allocation.released,
                "capacity_generation": int(getattr(gate.scheduler_allocation, "capacity_generation", -1)),
            }
            if gate.scheduler_allocation is not None
            else None
        ),
        "preplan_age_ms": (
            max(0.0, (time.monotonic() - gate.preplan.created_at) * 1000.0) if gate.preplan is not None else None
        ),
        "request_seen_age_ms": (
            max(0.0, (time.monotonic() - gate.request_seen_at) * 1000.0) if gate.request_seen_at > 0 else None
        ),
        "age_seconds": max(0.0, time.monotonic() - gate.created_at),
        "ttl_remaining_ms": max(
            0,
            int((gate.expires_at - time.monotonic()) * 1000),
        ),
    }


def _local_block_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """读取 scheduler BlockPool 中指定物理 block 的精确 hash。

    参数:
        payload: 包含非空 block_ids 列表的请求体。allow_missing=True 时，
            已释放或尚未发布的 block 会以 block_hash=None 返回。

    返回:
        blocks 列表；每项包含 block_id、packed_hex、external_hash 和 group_id。
    """

    block_pool = _latest_reservable_block_pool()
    if block_pool is None:
        return {"ok": False, "error": "vLLM BlockPool is not ready"}
    blocks = getattr(block_pool, "blocks", None)
    if not isinstance(blocks, list | tuple):
        return {"ok": False, "error": "vLLM BlockPool does not expose blocks"}

    result = []
    allow_missing = bool(payload.get("allow_missing", False))
    for block_id in _required_int_list(payload, "block_ids"):
        if block_id < 0 or block_id >= len(blocks):
            if allow_missing:
                result.append({"block_id": block_id, "block_hash": None})
                continue
            raise ValueError(f"block_id out of range: block_id={block_id}, block_count={len(blocks)}")
        block = blocks[block_id]
        block_hash = getattr(block, "block_hash", None)
        if block_hash is None:
            if allow_missing:
                result.append({"block_id": block_id, "block_hash": None})
                continue
            raise ValueError(f"vLLM block {block_id} is not published in prefix cache")
        result.append(
            {
                "block_id": block_id,
                "block_hash": _encode_packed_block_hash(block_hash),
            }
        )
    return {"ok": True, "blocks": result}


def _encode_packed_block_hash(block_hash: Any) -> dict[str, Any]:
    """把 vLLM 内部完整 block hash 编码为 JSON 安全描述。

    参数:
        block_hash: vLLM BlockHashWithGroupId，0.18 版本应为 bytes。

    返回:
        包含 packed_hex、external_hash 和 group_id 的描述字典。
    """

    if not isinstance(block_hash, bytes | bytearray):
        raise ValueError(f"unsupported vLLM packed block hash type: {type(block_hash).__name__}")
    packed = bytes(block_hash)
    if len(packed) <= 4:
        raise ValueError(f"invalid packed block hash length: {len(packed)}")
    external = packed[:-4]
    return {
        "packed_hex": packed.hex(),
        "external_hash": _external_block_hash_string(external),
        "group_id": int.from_bytes(packed[-4:], "big", signed=False),
    }


def _decode_packed_block_hash(value: Any) -> bytes:
    """从事务请求恢复 vLLM 内部完整 block hash。

    参数:
        value: `/datadist/blocks/metadata` 返回的 block_hash 描述，或带
            `packed_hex:` 前缀的字符串。

    返回:
        可直接写入 BlockPool 的 BlockHashWithGroupId bytes。
    """

    packed_hex = value.get("packed_hex") if isinstance(value, dict) else None
    if packed_hex is None and isinstance(value, str) and value.startswith("packed_hex:"):
        packed_hex = value.removeprefix("packed_hex:")
    if not isinstance(packed_hex, str) or not packed_hex:
        raise ValueError("block_hash must come from /datadist/blocks/metadata and contain packed_hex")
    try:
        packed = bytes.fromhex(packed_hex)
    except ValueError as exc:
        raise ValueError("block_hash packed_hex is not valid hexadecimal") from exc
    if len(packed) <= 4:
        raise ValueError(f"invalid packed block hash length: {len(packed)}")
    return packed


def _external_block_hash_string(block_hash: bytes) -> str:
    """生成与 vLLM KV event 一致的外部 hash 字符串。

    参数:
        block_hash: 不含 group ID 的原始 BlockHash bytes。

    返回:
        当前 vLLM 配置下 KV event 使用的 bytes 或 int 的字符串形式。
    """

    try:
        from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

        return str(maybe_convert_block_hash(block_hash))
    except Exception:
        return str(block_hash)


def _forward_block_reservation_request(
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """把 NPU worker 控制请求转发到 engine-core scheduler reservation 服务。

    参数:
        path: reservation 请求路径。
        payload: JSON 请求体。

    返回:
        scheduler reservation 服务返回的 JSON 字典。
    """

    port = _int_env(_RESERVATION_PORT_ENV, 0)
    if port <= 0:
        return {"ok": False, "error": "engine-core reservation port is not configured"}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout_seconds = _block_reservation_proxy_timeout_seconds(payload)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"reservation proxy HTTP {exc.code}: {body}"}
    except Exception as exc:
        result = {"ok": False, "error": f"engine-core reservation proxy failed: {exc}"}
        if path == "/datadist/block-pool/capacity/local":
            result.update(error_type=type(exc).__name__, failure_stage="engine_core_proxy")
        return result


def _block_reservation_proxy_timeout_seconds(payload: dict[str, Any]) -> float:
    """计算 NPU worker 到 scheduler 内部控制接口的 HTTP 超时。

    参数:
        payload: 内部请求体；`timeout_ms` 表示 scheduler 操作自身的预算。

    返回:
        HTTP 秒级超时，至少 10 秒，并比内部操作多保留 1 秒响应余量。
    """

    raw_timeout_ms = int(payload.get("timeout_ms", 9_000))
    return max(10.0, (max(raw_timeout_ms, 1) + 1_000) / 1000.0)


def _dispatch_block_pool_request(
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """把 BlockPool 请求发送到实际持有 scheduler BlockPool 的进程。

    参数:
        path: BlockPool 控制路径，包括 metadata、reserve、publish 和 rollback。
        payload: 对应控制路径的 JSON 请求体。

    返回:
        本进程可直接访问 BlockPool 时返回本地处理结果，否则返回 scheduler
        内部 HTTP 服务的代理结果。
    """

    control = _ENGINE_CORE_DATAPLANE
    reservations = getattr(control, "block_reservations", None)
    pool_ready = False
    if reservations is not None:
        try:
            pool_ready = bool(reservations.stats()["pool_ready"])
        except Exception:
            logger.debug(
                "Jiuwen KVBM failed to inspect local BlockPool; forwarding path=%s",
                path,
                exc_info=True,
            )
    if pool_ready:
        if path == "/datadist/block-pool/capacity/local":
            started = time.perf_counter()
            response = _handle_local_block_reservation_request(path, payload)
            response.setdefault("_control_timing", {})["engine_core_execution_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
            return response
        return _handle_local_block_reservation_request(path, payload)
    return _forward_block_reservation_request(path, payload)


class _EngineCoreReservationHandler(BaseHTTPRequestHandler):
    """engine-core scheduler 进程内部 BlockPool reservation HTTP handler。"""

    # BaseHTTPRequestHandler dispatches by this exact protocol method name.
    # pylint: disable=invalid-name
    def do_POST(self) -> None:
        """处理内部 reservation POST 请求并返回 JSON。"""

        request_started = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8") or "{}")
            request_read_ms = (time.perf_counter() - request_started) * 1000.0
            execution_started = time.perf_counter()
            response = _handle_local_block_reservation_request(self.path, payload)
            execution_ms = (time.perf_counter() - execution_started) * 1000.0
            if self.path in {
                "/datadist/pd/export/local",
                "/datadist/block-pool/capacity/local",
            }:
                response.setdefault("_control_timing", {}).update(
                    {
                        "engine_core_request_read_ms": request_read_ms,
                        "engine_core_execution_ms": execution_ms,
                    }
                )
            status = 200 if response.get("ok") else 503
        except Exception as exc:
            logger.exception("Jiuwen KVBM engine-core reservation request failed")
            response = {"ok": False, "error": str(exc)}
            status = 500
        encoded = json.dumps(response, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    # pylint: enable=invalid-name

    def log_message(self, message_format: str, *args: Any) -> None:
        """把内部访问日志转发到模块 debug logger。"""

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Jiuwen KVBM reservation control: %s", message_format % args)


def _handle_engine_core_prefetch_request(payload: dict[str, Any]) -> dict[str, Any]:
    """处理一次 engine-core 预取请求。

    参数:
        payload: JSON 请求体，支持 block_hashes、request_id、deadline_ms 和
            deadline_epoch_ms 字段。

    返回:
        可序列化响应字典，包含 prefetched_count、requested_count、stats 等字段。
    """

    control = _ENGINE_CORE_DATAPLANE
    if control is None:
        return {
            "ok": False,
            "error": "jiuwen kvbm dataplane is not ready",
            "prefetched_count": 0,
            "requested_count": 0,
        }

    raw_hashes = payload.get("block_hashes") or []
    block_hashes = [str(block_hash) for block_hash in raw_hashes if block_hash is not None]
    result = control.prefetch_block_hashes_for_request(
        block_hashes,
        request_id=str(payload.get("request_id") or ""),
        deadline_monotonic=_prefetch_deadline_from_payload(payload),
    )
    events = result.pop("events")
    _enqueue_runtime_events(events)
    return {
        "ok": True,
        "request_id": payload.get("request_id"),
        "requested_count": len(block_hashes),
        "stats": control.stats(),
        **result,
    }


def _prefetch_deadline_from_payload(payload: dict[str, Any]) -> float:
    """将 Router 传入的预取截止时间转换为本进程单调时钟。

    参数:
        payload: 预取请求体。deadline_ms 表示最大预算，deadline_epoch_ms
            表示 Router 计算的绝对 Unix 毫秒截止时间。

    返回:
        当前进程可安全比较的 monotonic 截止时间。绝对时间和相对预算同时存在
        时取更早者，避免网络转发时间重新获得一份完整预算。
    """

    now_monotonic = time.monotonic()
    raw_budget_ms = payload.get("deadline_ms")
    budget_ms = max(0, int(100 if raw_budget_ms is None else raw_budget_ms))
    candidates = [now_monotonic + budget_ms / 1000.0]
    raw_epoch_ms = payload.get("deadline_epoch_ms")
    if raw_epoch_ms is not None:
        remaining_ms = max(0.0, float(raw_epoch_ms) - time.time() * 1000.0)
        candidates.append(now_monotonic + remaining_ms / 1000.0)
    return min(candidates)


def _run_pd_import_task(
    control: "VllmAscend018DataPlaneControl",
    backend: LLMDataDistBackend,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """在后台执行一次D Worker KV import并发布完成事件。

    参数:
        control: 当前engine-core数据面控制对象。
        backend: 已初始化的LLM-DataDist传输后端。
        payload: Router提交的PD import请求体。

    返回:
        与同步/datadist/pd/import接口一致的结果字典。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    request_id = str(payload.get("request_id", "")).strip()
    submitted_at = float(payload.pop("_import_submitted_at", 0.0) or 0.0)
    dispatched_at = float(payload.pop("_import_dispatched_at", 0.0) or 0.0)
    scheduler_capacity_wait_ms = float(payload.pop("_scheduler_capacity_wait_ms", 0.0) or 0.0)
    queue_started_at = dispatched_at or submitted_at
    executor_queue_ms = max(0.0, (time.monotonic() - queue_started_at) * 1000.0) if queue_started_at > 0 else 0.0
    try:
        result = _pd_import_response(control, backend, payload)
    except Exception as exc:
        logger.exception(
            "Jiuwen KVBM asynchronous PD import failed: handoff_id=%s request_id=%s",
            handoff_id,
            request_id,
        )
        result = {"ok": False, "error": str(exc)}
    phase_ms = result.setdefault("phase_ms", {})
    if isinstance(phase_ms, dict):
        phase_ms["import_executor_queue_ms"] = executor_queue_ms
        phase_ms["scheduler_capacity_wait_ms"] = scheduler_capacity_wait_ms
    if executor_queue_ms >= 10.0:
        logger.warning(
            "Jiuwen KVBM PD import executor queue delay: handoff_id=%s request_id=%s queue_ms=%.3f",
            handoff_id,
            request_id,
            executor_queue_ms,
        )
    _PD_CONTROL_EVENTS.publish(
        "IMPORT_COMPLETE" if result.get("ok") else "IMPORT_FAILED",
        handoff_id=handoff_id,
        request_id=request_id,
        result=result,
    )
    return result


def _submit_pd_import_response(
    control: "VllmAscend018DataPlaneControl",
    backend: LLMDataDistBackend,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """幂等提交一次后台PD import事务。

    参数:
        control: 当前engine-core数据面控制对象。
        backend: 已初始化的LLM-DataDist传输后端。
        payload: 必须包含handoff_id和request_id的import请求体。

    返回:
        包含accepted和already_submitted字段的快速确认响应。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    request_id = str(payload.get("request_id", "")).strip()
    if not handoff_id:
        raise ValueError("handoff_id is required")
    if not request_id:
        raise ValueError("request_id is required")
    with _PD_IMPORT_FUTURES_LOCK:
        existing = _PD_IMPORT_FUTURES.get(handoff_id)
        if existing is not None:
            return {
                "ok": True,
                "accepted": True,
                "already_submitted": True,
                "completed": existing.done(),
                "handoff_id": handoff_id,
            }
    capacity_wait_timeout_ms = int(
        payload.get(
            "capacity_wait_timeout_ms",
            payload.get("attach_timeout_ms", 10_000),
        )
    )
    gate_ttl_ms = max(
        120_000,
        capacity_wait_timeout_ms + int(payload.get("attach_timeout_ms", 10_000)) + 30_000,
    )
    gate_renew = _dispatch_block_pool_request(
        "/datadist/pd/import/prepare/local",
        {
            "handoff_id": handoff_id,
            "request_id": request_id,
            "ttl_ms": gate_ttl_ms,
            "require_existing": True,
        },
    )
    gate = gate_renew.get("gate") or {}
    if not gate_renew.get("ok") or not gate.get("request_seen"):
        return {
            "ok": False,
            "accepted": False,
            "error": (gate_renew.get("error") or "D request gate disappeared before PD import submission"),
            "handoff_id": handoff_id,
        }
    queued_payload = dict(payload)
    queued_payload["gate_ttl_ms"] = gate_ttl_ms
    queued_payload["_import_submitted_at"] = time.monotonic()
    if _pd_block_ownership_mode() is SchedulerKVOwnershipMode.SCHEDULER:
        result_future: Future[dict[str, Any]] = Future()
        submission_created_at = time.monotonic()
        submission = _PDSchedulerImportSubmission(
            handoff_id=handoff_id,
            request_id=request_id,
            control=control,
            backend=backend,
            payload=queued_payload,
            result_future=result_future,
            created_at=submission_created_at,
            capacity_deadline_at=(submission_created_at + max(capacity_wait_timeout_ms, 1) / 1000.0),
        )
        with _PD_IMPORT_FUTURES_LOCK:
            existing = _PD_IMPORT_FUTURES.get(handoff_id)
            if existing is not None:
                return {
                    "ok": True,
                    "accepted": True,
                    "already_submitted": True,
                    "completed": existing.done(),
                    "handoff_id": handoff_id,
                }
            _PD_IMPORT_FUTURES[handoff_id] = result_future
            _PD_SCHEDULER_IMPORT_SUBMISSIONS[handoff_id] = submission
        dispatched = _activate_scheduler_import_submission(handoff_id)
        return {
            "ok": True,
            "accepted": True,
            "already_submitted": False,
            "completed": False,
            "waiting_for_scheduler_capacity": not dispatched,
            "handoff_id": handoff_id,
        }
    with _PD_IMPORT_FUTURES_LOCK:
        existing = _PD_IMPORT_FUTURES.get(handoff_id)
        if existing is not None:
            return {
                "ok": True,
                "accepted": True,
                "already_submitted": True,
                "completed": existing.done(),
                "handoff_id": handoff_id,
            }
        if len(_PD_IMPORT_FUTURES) >= _PD_CONTROL_EVENT_HISTORY_LIMIT:
            completed = [key for key, future in _PD_IMPORT_FUTURES.items() if future.done()]
            for key in completed[:max(1, len(completed) // 2)]:
                _PD_IMPORT_FUTURES.pop(key, None)
        future = _PD_IMPORT_EXECUTOR.submit(
            _run_pd_import_task,
            control,
            backend,
            queued_payload,
        )
        _PD_IMPORT_FUTURES[handoff_id] = future
    return {
        "ok": True,
        "accepted": True,
        "already_submitted": False,
        "completed": False,
        "handoff_id": handoff_id,
    }


def _pd_import_result_response(payload: dict[str, Any]) -> dict[str, Any]:
    """读取后台PD import结果，供事件流断开后的单次兜底查询使用。

    参数:
        payload: 包含handoff_id和可选wait_timeout_ms的请求体。

    返回:
        import已完成时返回原始结果；尚未提交或等待超时时返回ok=False。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    if not handoff_id:
        raise ValueError("handoff_id is required")
    with _PD_IMPORT_FUTURES_LOCK:
        future = _PD_IMPORT_FUTURES.get(handoff_id)
    if future is None:
        return {"ok": False, "error": "PD import has not been submitted"}
    timeout_ms = max(0, int(payload.get("wait_timeout_ms", 0)))
    try:
        return future.result(timeout=timeout_ms / 1000.0)
    except FutureTimeoutError:
        return {"ok": False, "pending": True, "error": "PD import is still running"}


def _handle_engine_core_datadist_request(
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """处理 engine-core 内 LLM-DataDist 建链、断链和 block pull 请求。

    参数:
        path: HTTP 请求路径，支持 /datadist/status、/datadist/link、
            /datadist/unlink、/datadist/pull、/datadist/transfers/pull，
            /datadist/pd/handoffs/*，以及 BlockPool metadata/reservation 控制路径。
        payload: JSON 请求体。

    返回:
        可序列化响应字典；操作成功时 ok 为 True。
    """

    control = _ENGINE_CORE_DATAPLANE
    if control is None:
        return {"ok": False, "error": "jiuwen kvbm dataplane is not ready"}
    if path == "/datadist/digest":
        block_ids = _required_int_list(payload, "block_ids")
        include_layers = bool(payload.get("include_layers", False))
        with _ENGINE_CORE_LLM_DATADIST_LOCK:
            blocks = [control.provider.block_digest(block_id, include_layers=include_layers) for block_id in block_ids]
        return {"ok": True, "blocks": blocks}
    if (
        path == "/datadist/blocks/metadata"
        or path.startswith("/datadist/reservations/")
        or path.startswith("/datadist/source-leases/")
    ):
        return _dispatch_block_pool_request(path, payload)
    if path == "/datadist/transfers/status":
        return _transfer_status_response(payload)
    if path == "/datadist/transfers/cancel":
        return _transfer_cancel_response(payload)
    if path == "/datadist/pd/handoffs/begin":
        return _pd_handoff_begin_response(payload)
    if path == "/datadist/pd/handoffs/advance":
        return _pd_handoff_advance_response(payload)
    if path == "/datadist/pd/handoffs/status":
        return _pd_handoff_status_response(payload)
    if path == "/datadist/pd/handoffs/rollback":
        return _pd_handoff_rollback_response(payload)
    if path == "/datadist/pd/export":
        return _pd_export_response(control, payload)
    if path == "/datadist/pd/export/release":
        return _pd_export_release_response(payload)
    if path == "/datadist/pd/import/prepare":
        return _dispatch_block_pool_request(
            "/datadist/pd/import/prepare/local",
            payload,
        )
    if path == "/datadist/pd/import/status":
        return _dispatch_block_pool_request(
            "/datadist/pd/import/status/local",
            payload,
        )
    if path == "/datadist/pd/import/release":
        return _dispatch_block_pool_request(
            "/datadist/pd/import/release/local",
            payload,
        )
    if path == "/datadist/block-pool/capacity":
        proxy_started = time.perf_counter()
        block_pool_status = _dispatch_block_pool_request(
            "/datadist/block-pool/capacity/local",
            payload,
        )
        proxy_ms = (time.perf_counter() - proxy_started) * 1000.0
        control_timing = dict(block_pool_status.get("_control_timing") or {})
        control_timing["engine_core_proxy_ms"] = proxy_ms
        if "engine_core_execution_ms" in control_timing:
            control_timing["engine_core_queue_and_transport_ms"] = max(
                proxy_ms - float(control_timing["engine_core_execution_ms"]), 0.0
            )
        return {
            "ok": bool(block_pool_status.get("ok")),
            "control_timing": control_timing,
            "error_type": block_pool_status.get("error_type"),
            "failure_stage": block_pool_status.get("failure_stage"),
            "block_pool": {
                "ready": block_pool_status.get("free_blocks") is not None,
                "free_blocks": block_pool_status.get("free_blocks"),
                "capacity_generation": block_pool_status.get("capacity_generation"),
                "block_size": _int_env(_BLOCK_SIZE_ENV, 16),
                "ownership_mode": _pd_block_ownership_mode().value,
                "confirmed_reusable_blocks": block_pool_status.get("confirmed_reusable_blocks", 0),
                "required_new_blocks": block_pool_status.get("required_new_blocks", 0),
                "margin_blocks": block_pool_status.get("margin_blocks", 0),
                "outstanding_ticket_reserved_blocks": block_pool_status.get("outstanding_ticket_reserved_blocks", 0),
                "router_pending_reserved_blocks": block_pool_status.get("router_pending_reserved_blocks", 0),
                "committed_reserved_blocks": block_pool_status.get("committed_reserved_blocks", 0),
                "ticket_capacity_admitted": block_pool_status.get("ticket_capacity_admitted", False),
                "queried_block_hashes": block_pool_status.get("queried_block_hashes", 0),
                "admission_ticket_id": block_pool_status.get("admission_ticket_id", ""),
                "admission_ticket_pinned_blocks": block_pool_status.get("admission_ticket_pinned_blocks", 0),
            },
            "error": block_pool_status.get("error"),
        }
    if path == "/datadist/block-pool/admission-ticket/release":
        return _dispatch_block_pool_request(
            "/datadist/block-pool/admission-ticket/release/local",
            payload,
        )
    backend = getattr(control, "llm_datadist_backend", None)
    if path == "/datadist/benchmark/pull":
        return _benchmark_pull_response(control, backend, payload)
    if backend is None:
        return {"ok": False, "error": "jiuwen kvbm llm_datadist backend is not ready"}
    if path == "/datadist/pd/import/submit":
        return _submit_pd_import_response(control, backend, payload)
    if path == "/datadist/pd/import/result":
        return _pd_import_result_response(payload)
    if path == "/datadist/pd/import":
        return _pd_import_response(control, backend, payload)
    if path == "/datadist/status":
        block_pool_status = _dispatch_block_pool_request(
            "/datadist/pd/import/status/local",
            {},
        )
        return {
            "ok": True,
            "cache_id": _ENGINE_CORE_LLM_DATADIST_CACHE_ID,
            "health": backend.health(),
            "transfers": _process_transfer_transactions().stats(),
            "pd_handoffs": _process_pd_handoffs().stats(),
            "scheduler_kv_bridge": _scheduler_kv_bridge().diagnostics(),
            "block_pool": {
                "ready": block_pool_status.get("free_blocks") is not None,
                "free_blocks": block_pool_status.get("free_blocks"),
                "block_size": _int_env(_BLOCK_SIZE_ENV, 16),
                "ownership_mode": block_pool_status.get("ownership_mode"),
            },
            "benchmark": {
                "enabled": _bool_env(_BENCHMARK_ENABLED_ENV),
                "tcp_host": os.getenv(_BENCHMARK_TCP_HOST_ENV, "127.0.0.1"),
                "tcp_port": _int_env(_BENCHMARK_TCP_PORT_ENV, 0),
                "tcp_server_ready": _ENGINE_CORE_TCP_STAGING_SERVER is not None,
                "max_blocks": _int_env(_BENCHMARK_MAX_BLOCKS_ENV, 64),
            },
        }

    timeout_ms = int(payload.get("timeout_ms", 3000))
    if path == "/datadist/transfers/pull":
        return _pull_blocks_and_publish_idempotent(control, backend, payload)
    with _ENGINE_CORE_LLM_DATADIST_LOCK:
        if path == "/datadist/link":
            cluster = _datadist_cluster_from_payload(payload)
            health = backend.health()
            linked_cluster_ids = {int(cluster_id) for cluster_id in health.get("linked_cluster_ids", [])}
            if cluster.remote_cluster_id in linked_cluster_ids:
                return {
                    "ok": True,
                    "remote_cluster_id": cluster.remote_cluster_id,
                    "already_linked": True,
                }
            backend.link_clusters([cluster], timeout_ms=timeout_ms)
            return {"ok": True, "remote_cluster_id": cluster.remote_cluster_id}
        if path == "/datadist/unlink":
            cluster = _datadist_cluster_from_payload(payload)
            backend.unlink_clusters(
                [cluster],
                timeout_ms=timeout_ms,
                force=bool(payload.get("force", False)),
            )
            return {"ok": True, "remote_cluster_id": cluster.remote_cluster_id}
        if path == "/datadist/pull":
            remote_cluster_id = int(payload["remote_cluster_id"])
            cache_id = _ENGINE_CORE_LLM_DATADIST_CACHE_ID
            if cache_id is None:
                raise ValueError("local LLM-DataDist cache is not registered")
            source_blocks = _required_int_list(payload, "source_blocks")
            destination_blocks = _required_int_list(payload, "destination_blocks")
            if len(source_blocks) != len(destination_blocks):
                raise ValueError("source_blocks and destination_blocks must have equal length")
            raw_layer_range = payload.get("layer_range")
            layer_range = None
            if raw_layer_range is not None:
                if not isinstance(raw_layer_range, list | tuple) or len(raw_layer_range) != 2:
                    raise ValueError("layer_range must contain [start, stop]")
                layer_range = (int(raw_layer_range[0]), int(raw_layer_range[1]))
            backend.pull_blocks(
                cache_id,
                LLMDataDistBlockMapping(
                    cache_key=LLMDataDistCacheKey(
                        key=str(payload.get("cache_key", remote_cluster_id)),
                        prompt_cluster_id=remote_cluster_id,
                        prompt_batch_index=int(payload.get("model_id", 0)),
                    ),
                    prompt_blocks=tuple(source_blocks),
                    decoder_blocks=tuple(destination_blocks),
                    layer_range=layer_range,
                ),
            )
            logger.info(
                "Jiuwen KVBM LLM-DataDist pulled %d blocks from cluster=%s into cache_id=%s",
                len(source_blocks),
                remote_cluster_id,
                cache_id,
            )
            return {
                "ok": True,
                "cache_id": cache_id,
                "remote_cluster_id": remote_cluster_id,
                "pulled_count": len(source_blocks),
            }
    return {"ok": False, "error": "not found"}


def _benchmark_pull_response(
    control: "VllmAscend018DataPlaneControl",
    backend: LLMDataDistBackend | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """执行隔离的 HIXL 或 TCP staging KV block 拉取基准。

    参数:
        control: 当前 engine-core 数据面控制对象。
        backend: 当前 LLM-DataDist backend；HIXL 基准必须提供。
        payload: 基准请求体，包含 backend、source_blocks、
            destination_blocks、warmup 和 iterations。TCP staging 还需提供
            source_host/source_port，HIXL 还需提供 remote_cluster_id。

    返回:
        包含统一延迟、有效带宽和路径语义的基准结果字典。

    说明:
        本入口会覆盖 destination_blocks，仅用于无推理流量的隔离环境。它不会
        reserve、publish 或修改 vLLM BlockPool 所有权。
    """

    if not _bool_env(_BENCHMARK_ENABLED_ENV):
        return {"ok": False, "error": "jiuwen kvbm transfer benchmark is disabled"}
    source_blocks = _required_int_list(payload, "source_blocks")
    destination_blocks = _required_int_list(payload, "destination_blocks")
    if len(source_blocks) != len(destination_blocks):
        raise ValueError("source_blocks and destination_blocks must have equal length")
    max_blocks = _int_env(_BENCHMARK_MAX_BLOCKS_ENV, 64)
    if len(source_blocks) > max_blocks:
        raise ValueError(f"block count exceeds benchmark max_blocks={max_blocks}")
    if payload.get("layer_range") is not None:
        raise ValueError("benchmark currently requires full-block transfers")

    warmup = int(payload.get("warmup", 3))
    iterations = int(payload.get("iterations", 20))
    transferred_bytes = sum(int(control.provider.describe_block(block_id).nbytes) for block_id in destination_blocks)
    benchmark_backend = str(payload.get("backend", "hixl")).strip().lower()

    if benchmark_backend == "hixl":
        if backend is None:
            return {
                "ok": False,
                "error": "jiuwen kvbm llm_datadist backend is not ready",
            }
        cache_id = _ENGINE_CORE_LLM_DATADIST_CACHE_ID
        if cache_id is None:
            raise ValueError("local LLM-DataDist cache is not registered")
        remote_cluster_id = int(payload["remote_cluster_id"])
        mapping = LLMDataDistBlockMapping(
            cache_key=LLMDataDistCacheKey(
                key=str(payload.get("cache_key", remote_cluster_id)),
                prompt_cluster_id=remote_cluster_id,
                prompt_batch_index=int(payload.get("model_id", 0)),
            ),
            prompt_blocks=tuple(source_blocks),
            decoder_blocks=tuple(destination_blocks),
        )

        def operation() -> dict[str, float]:
            """执行一次同步 LLM-DataDist/HIXL block 拉取并返回耗时。"""

            started = time.perf_counter()
            with _ENGINE_CORE_LLM_DATADIST_LOCK:
                backend.pull_blocks(cache_id, mapping)
            return {"elapsed_ms": (time.perf_counter() - started) * 1000.0}

        semantics = "NPU->HIXL fabric->NPU through LLM-DataDist pull_blocks"
        result = run_transfer_benchmark(
            operation,
            transferred_bytes=transferred_bytes,
            warmup=warmup,
            iterations=iterations,
        )
    elif benchmark_backend == "tcp_staging":
        source_host = str(payload["source_host"])
        source_port = int(payload["source_port"])
        timeout_seconds = max(0.001, float(payload.get("timeout_ms", 30000)) / 1000.0)
        semantics = "source NPU->CPU RAM->TCP->CPU RAM->destination NPU"
        with TcpStagingClient(
            source_host,
            source_port,
            control.provider,
            _ENGINE_CORE_LLM_DATADIST_LOCK,
            timeout_seconds=timeout_seconds,
        ) as client:
            result = run_transfer_benchmark(
                lambda: client.pull_blocks(
                    source_blocks,
                    destination_blocks,
                    expected_nbytes=transferred_bytes,
                ),
                transferred_bytes=transferred_bytes,
                warmup=warmup,
                iterations=iterations,
            )
    else:
        raise ValueError("backend must be 'hixl' or 'tcp_staging'")

    logger.info(
        "Jiuwen KVBM transfer benchmark completed backend=%s blocks=%d "
        "bytes=%d latency_p50_ms=%.3f effective_p50_gbps=%.3f",
        benchmark_backend,
        len(source_blocks),
        transferred_bytes,
        result["latency_ms"]["p50"],
        result["effective_gbps"]["p50"],
    )
    return {
        "ok": True,
        "backend": benchmark_backend,
        "semantics": semantics,
        "block_count": len(source_blocks),
        **result,
    }


def _transfer_status_response(payload: dict[str, Any]) -> dict[str, Any]:
    """返回单个传输事务或整体事务统计。

    参数:
        payload: 可选包含 transfer_id 的请求体。

    返回:
        指定 transfer_id 时返回事务快照，否则返回状态计数和并发容量。
    """

    manager = _process_transfer_transactions()
    transfer_id = str(payload.get("transfer_id", "")).strip()
    if not transfer_id:
        return {"ok": True, "transfers": manager.stats()}
    try:
        transaction = manager.get(transfer_id)
    except TransferTransactionError as exc:
        return {"ok": False, "error": str(exc), "transfer_id": transfer_id}
    return {"ok": True, "transfer": transaction.to_dict()}


def _transfer_cancel_response(payload: dict[str, Any]) -> dict[str, Any]:
    """请求协作式取消一次跨 Worker 传输。

    参数:
        payload: 必须包含 transfer_id 的请求体。

    返回:
        包含 accepted 和最新事务快照的响应。
    """

    transfer_id = str(payload.get("transfer_id", "")).strip()
    if not transfer_id:
        return {"ok": False, "error": "transfer_id is required"}
    try:
        transaction, accepted = _process_transfer_transactions().request_cancel(transfer_id)
    except TransferTransactionError as exc:
        return {"ok": False, "error": str(exc), "transfer_id": transfer_id}
    return {
        "ok": True,
        "accepted": accepted,
        "transfer": transaction.to_dict(include_result=False),
    }


def _pd_handoff_begin_response(payload: dict[str, Any]) -> dict[str, Any]:
    """创建请求级 PD handoff 记录，或幂等返回已有记录。

    参数:
        payload: 包含 request_id、源/目标 Worker、模型、prompt token 数和可选
            local_role 的请求体。local_role=source 时会在 P scheduler 预注册
            请求释放前的 partial block 捕获意图。

    返回:
        包含 created 和 handoff 快照的可序列化响应。
    """

    semantic_fields = {
        "request_id": str(payload.get("request_id", "")),
        "source_worker_id": str(payload.get("source_worker_id", "")),
        "target_worker_id": str(payload.get("target_worker_id", "")),
        "model_id": str(payload.get("model_id", "")),
        "prompt_token_count": int(payload.get("prompt_token_count", 0)),
    }
    request_fingerprint = str(payload.get("request_fingerprint", "")).strip()
    if not request_fingerprint:
        request_fingerprint = hashlib.sha256(
            json.dumps(
                semantic_fields,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    handoff, created = _process_pd_handoffs().begin(
        handoff_id=(str(payload["handoff_id"]).strip() if payload.get("handoff_id") is not None else None),
        request_fingerprint=request_fingerprint,
        deadline_epoch_ms=(int(payload["deadline_epoch_ms"]) if payload.get("deadline_epoch_ms") is not None else None),
        metadata=metadata,
        **semantic_fields,
    )
    local_role = str(payload.get("local_role", "")).strip().lower()
    if local_role not in {"", "source", "target"}:
        raise ValueError("local_role must be source, target or empty")
    source_prepare: dict[str, Any] | None = None
    if local_role == "source":
        source_prepare = _dispatch_block_pool_request(
            "/datadist/pd/export/prepare/local",
            {
                "handoff_id": handoff.handoff_id,
                "request_id": handoff.request_id,
                "prompt_token_count": handoff.prompt_token_count,
                "ttl_ms": int(payload.get("source_capture_ttl_ms", 120_000)),
            },
        )
        if not source_prepare.get("ok"):
            return {
                "ok": False,
                "error": f"failed to prepare P request-scoped export capture: {source_prepare.get('error')}",
                "created": created,
                "handoff": handoff.to_dict(),
                "source_prepare": source_prepare,
            }
    logger.info(
        "Jiuwen KVBM PD handoff begun: handoff_id=%s request_id=%s source=%s target=%s created=%s",
        handoff.handoff_id,
        handoff.request_id,
        handoff.source_worker_id,
        handoff.target_worker_id,
        created,
    )
    return {
        "ok": True,
        "created": created,
        "handoff": handoff.to_dict(),
        "source_prepare": source_prepare,
    }


def _pd_handoff_advance_response(payload: dict[str, Any]) -> dict[str, Any]:
    """执行一次受状态迁移表约束的 PD handoff 阶段更新。

    参数:
        payload: 包含 handoff_id、state 和可选资源字段的请求体。

    返回:
        包含更新后 handoff 快照的可序列化响应。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    if not handoff_id:
        raise ValueError("handoff_id is required")
    try:
        state = PDHandoffState(str(payload.get("state", "")))
        expected_state = (
            PDHandoffState(str(payload["expected_state"])) if payload.get("expected_state") is not None else None
        )
    except ValueError as exc:
        raise ValueError("state or expected_state is not a valid PD handoff state") from exc
    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    handoff = _process_pd_handoffs().advance(
        handoff_id,
        state,
        expected_state=expected_state,
        block_hashes=_optional_list(payload, "block_hashes", str),
        source_blocks=_optional_list(payload, "source_blocks", int),
        destination_blocks=_optional_list(payload, "destination_blocks", int),
        source_lease_id=_optional_string(payload, "source_lease_id"),
        reservation_id=_optional_string(payload, "reservation_id"),
        transfer_id=_optional_string(payload, "transfer_id"),
        error=_optional_string(payload, "error"),
        metadata=metadata,
    )
    logger.info(
        "Jiuwen KVBM PD handoff advanced: handoff_id=%s request_id=%s state=%s",
        handoff.handoff_id,
        handoff.request_id,
        handoff.state.value,
    )
    return {"ok": True, "handoff": handoff.to_dict()}


def _pd_handoff_status_response(payload: dict[str, Any]) -> dict[str, Any]:
    """读取单个 PD handoff 或返回进程级 handoff 统计。

    参数:
        payload: 可选包含 handoff_id 的请求体。

    返回:
        指定 ID 时返回 handoff 快照，否则返回状态和容量统计。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    if not handoff_id:
        return {"ok": True, "pd_handoffs": _process_pd_handoffs().stats()}
    return {
        "ok": True,
        "handoff": _process_pd_handoffs().get(handoff_id).to_dict(),
    }


def _pd_handoff_rollback_response(payload: dict[str, Any]) -> dict[str, Any]:
    """请求 PD handoff 进入 ROLLING_BACK，等待资源清理器完成回滚。

    参数:
        payload: 包含 handoff_id 和可选 reason 的请求体。

    返回:
        包含 accepted 和最新 handoff 快照的响应。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    if not handoff_id:
        raise ValueError("handoff_id is required")
    handoff, accepted = _process_pd_handoffs().request_rollback(
        handoff_id,
        reason=str(payload.get("reason", "PD handoff rollback requested")),
    )
    return {"ok": True, "accepted": accepted, "handoff": handoff.to_dict()}


def _pd_export_response(
    control: "VllmAscend018DataPlaneControl",
    payload: dict[str, Any],
) -> dict[str, Any]:
    """导出并 pin P Worker 当前 handoff 对应的真实 vLLM block。

    参数:
        control: 当前 NPU worker 数据面控制对象，用于返回本地 DataDist 信息。
        payload: 包含 handoff_id，可选 request_id、block_limit 和 ttl_ms 的请求体。

    返回:
        包含 SOURCE_PINNED handoff、源物理 block、packed hash、lease 及本地
        LLM-DataDist cache 信息的响应。重复调用已完成的 export 时幂等返回。
    """

    response_started = time.perf_counter()
    handoff_id = str(payload.get("handoff_id", "")).strip()
    if not handoff_id:
        raise ValueError("handoff_id is required")
    manager = _process_pd_handoffs()
    handoff = manager.get(handoff_id)
    request_id = str(payload.get("request_id", handoff.request_id)).strip()
    if request_id != handoff.request_id:
        raise ValueError("request_id does not match handoff")
    if handoff.state == PDHandoffState.SOURCE_PINNED:
        return {
            "ok": True,
            "already_exported": True,
            "handoff": handoff.to_dict(),
            "export": dict(handoff.metadata.get("source_export", {})),
        }
    if handoff.state not in {
        PDHandoffState.PREFILL_RUNNING,
        PDHandoffState.PREFILL_READY,
    }:
        return {
            "ok": False,
            "error": f"handoff state {handoff.state.value} cannot export P blocks",
            "handoff": handoff.to_dict(),
        }

    local_payload = {
        "request_id": request_id,
        "ttl_ms": int(payload.get("ttl_ms", 30_000)),
    }
    if payload.get("block_limit") is not None:
        local_payload["block_limit"] = int(payload["block_limit"])
    if payload.get("wait_timeout_ms") is not None:
        local_payload["wait_timeout_ms"] = int(payload["wait_timeout_ms"])
    engine_core_started = time.perf_counter()
    exported = _dispatch_block_pool_request(
        "/datadist/pd/export/local",
        local_payload,
    )
    engine_core_proxy_ms = (time.perf_counter() - engine_core_started) * 1000.0
    if not exported.get("ok"):
        return exported

    engine_core_control_timing = exported.get("_control_timing", {})
    if not isinstance(engine_core_control_timing, dict):
        engine_core_control_timing = {}
    engine_core_execution_ms = float(
        engine_core_control_timing.get(
            "engine_core_execution_ms",
            engine_core_proxy_ms,
        )
    )
    engine_core_queue_ms = max(
        0.0,
        engine_core_proxy_ms - engine_core_execution_ms,
    )
    response_build_started = time.perf_counter()

    lease = exported["lease"]
    external_hashes = []
    for matched_selected_value in exported["block_hashes"]:
        if (
            isinstance(matched_selected_value, dict)
            and (not matched_selected_value.get("request_scoped_partial"))
            and matched_selected_value.get("external_hash")
        ):
            external_hashes.append(str(matched_selected_value["external_hash"]))
    protocol_block_keys = _pd_protocol_block_keys(
        list(exported["block_hashes"]),
        partial_token_fingerprint=str(exported.get("partial_token_fingerprint", "")),
    )
    backend = getattr(control, "llm_datadist_backend", None)
    health = backend.health() if backend is not None else None
    export_metadata = {
        "request_id": request_id,
        "source_blocks": list(exported["source_blocks"]),
        "block_hashes": list(exported["block_hashes"]),
        "block_groups": list(exported["block_groups"]),
        "token_block_fingerprints": list(exported.get("token_block_fingerprints", [])),
        "semantic_plain_text": bool(exported.get("semantic_plain_text", False)),
        "request_scoped_partial": bool(exported.get("request_scoped_partial", False)),
        "partial_valid_tokens": int(exported.get("partial_valid_tokens", 0)),
        "partial_token_fingerprint": str(exported.get("partial_token_fingerprint", "")),
        "source_lease_id": str(lease["lease_id"]),
        "logical_block_count": int(exported["logical_block_count"]),
        "physical_block_count": int(exported["physical_block_count"]),
        "capture_wait_ms": float(exported.get("capture_wait_ms", 0.0)),
        "p_dispatch_to_scheduler_seen_ms": float(exported.get("p_dispatch_to_scheduler_seen_ms", 0.0)),
        "p_worker_request_received_to_scheduler_seen_ms": float(
            exported.get("p_worker_request_received_to_scheduler_seen_ms", 0.0)
        ),
        "p_worker_request_received_timing_source": str(
            exported.get("p_worker_request_received_timing_source", "unavailable")
        ),
        "p_scheduler_queue_cycles": int(exported.get("p_scheduler_queue_cycles", 0)),
        "p_scheduler_seen_cycle_id": int(exported.get("p_scheduler_seen_cycle_id", 0)),
        "p_first_scheduled_cycle_id": int(exported.get("p_first_scheduled_cycle_id", 0)),
        "p_kv_ready_cycle_id": int(exported.get("p_kv_ready_cycle_id", 0)),
        "p_scheduler_compute_cycles": int(exported.get("p_scheduler_compute_cycles", 0)),
        "p_first_scheduled_token_count": int(exported.get("p_first_scheduled_token_count", 0)),
        "p_scheduler_running_request_count": int(exported.get("p_scheduler_running_request_count", 0)),
        "p_scheduler_waiting_request_count": int(exported.get("p_scheduler_waiting_request_count", 0)),
        "p_first_scheduled_running_request_count": int(exported.get("p_first_scheduled_running_request_count", 0)),
        "p_first_scheduled_waiting_request_count": int(exported.get("p_first_scheduled_waiting_request_count", 0)),
        "p_first_scheduled_cycle_ms": float(exported.get("p_first_scheduled_cycle_ms", 0.0)),
        "p_scheduler_seen_to_kv_ready_ms": float(exported.get("p_scheduler_seen_to_kv_ready_ms", 0.0)),
        "p_kv_ready_to_export_ready_ms": float(exported.get("p_kv_ready_to_export_ready_ms", 0.0)),
        "p_scheduler_seen_to_export_ready_ms": float(exported.get("p_scheduler_seen_to_export_ready_ms", 0.0)),
        "cache_id": _ENGINE_CORE_LLM_DATADIST_CACHE_ID,
        "datadist_health": health,
    }
    export_metadata["control_timing"] = {
        "handler_request_read_ms": 0.0,
        "handler_execution_ms": 0.0,
        "engine_core_proxy_ms": engine_core_proxy_ms,
        "engine_core_queue_ms": engine_core_queue_ms,
        "engine_core_execution_ms": engine_core_execution_ms,
        "engine_core_request_read_ms": float(engine_core_control_timing.get("engine_core_request_read_ms", 0.0)),
        "response_build_ms": (time.perf_counter() - response_build_started) * 1000.0,
    }
    export_metadata["p_export_ready_to_http_response_ms"] = max(
        0.0,
        (time.perf_counter() - response_started) * 1000.0 - export_metadata["capture_wait_ms"],
    )
    try:
        if handoff.state == PDHandoffState.PREFILL_RUNNING:
            handoff = manager.advance(
                handoff_id,
                PDHandoffState.PREFILL_READY,
                expected_state=PDHandoffState.PREFILL_RUNNING,
                block_hashes=protocol_block_keys,
                source_blocks=list(exported["source_blocks"]),
                metadata={"source_export": export_metadata},
            )
        handoff = manager.advance(
            handoff_id,
            PDHandoffState.SOURCE_PINNED,
            expected_state=PDHandoffState.PREFILL_READY,
            block_hashes=protocol_block_keys,
            source_blocks=list(exported["source_blocks"]),
            source_lease_id=str(lease["lease_id"]),
            metadata={"source_export": export_metadata},
        )
    except Exception:
        _dispatch_block_pool_request(
            "/datadist/source-leases/release",
            {"lease_id": str(lease["lease_id"])},
        )
        raise

    logger.info(
        "Jiuwen KVBM P Worker export completed: handoff_id=%s request_id=%s "
        "logical_blocks=%s physical_blocks=%s lease_id=%s",
        handoff_id,
        request_id,
        exported["logical_block_count"],
        exported["physical_block_count"],
        lease["lease_id"],
    )
    return {
        "ok": True,
        "already_exported": False,
        "handoff": handoff.to_dict(),
        "export": export_metadata,
    }


def _pd_protocol_block_keys(
    block_hashes: list[Any] | tuple[Any, ...],
    *,
    partial_token_fingerprint: str,
) -> list[str]:
    """构造与物理 block 对齐的 PD 状态机资源键。

    参数:
        block_hashes: P export 返回的完整/partial block 描述。
        partial_token_fingerprint: 请求尾部 partial token span 指纹。

    返回:
        完整 block 使用 external_hash；partial block 使用仅限当前 handoff 的
        稳定键。返回长度始终与物理 block 列表一致，不用于 prefix-cache 发布。
    """

    keys: list[str] = []
    for item in block_hashes:
        if not isinstance(item, dict):
            raise ValueError("PD block descriptor must be an object")
        if item.get("request_scoped_partial"):
            if not partial_token_fingerprint:
                raise ValueError("partial block is missing token fingerprint")
            keys.append(
                "request-scoped-partial:"
                f"{int(item.get('logical_index', -1))}:"
                f"{int(item.get('group_id', -1))}:"
                f"{int(item.get('valid_token_count', 0))}:"
                f"{partial_token_fingerprint}"
            )
        else:
            external_hash = str(item.get("external_hash", ""))
            if not external_hash:
                raise ValueError("complete block is missing external_hash")
            keys.append(external_hash)
    return keys


def _pd_export_release_response(payload: dict[str, Any]) -> dict[str, Any]:
    """释放 P Worker export 持有的 source lease，并结束或回滚 handoff。

    参数:
        payload: 包含 handoff_id 和可选 reason 的请求体。DECODE_ACCEPTED 状态
            表示正常完成，其余非终态按回滚处理。

    返回:
        包含 source lease 释放结果和终态 handoff 快照的响应；重复调用幂等。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    if not handoff_id:
        raise ValueError("handoff_id is required")
    manager = _process_pd_handoffs()
    handoff = manager.get(handoff_id)
    if handoff.state in {
        PDHandoffState.SOURCE_RELEASED,
        PDHandoffState.ROLLED_BACK,
    }:
        return {
            "ok": True,
            "already_released": True,
            "handoff": handoff.to_dict(),
        }
    if not handoff.source_lease_id:
        capture_release = _dispatch_block_pool_request(
            "/datadist/pd/export/release/local",
            {"handoff_id": handoff_id, "request_id": handoff.request_id},
        )
        return {
            "ok": False,
            "error": "handoff does not own a source lease",
            "capture_release": capture_release,
            "handoff": handoff.to_dict(),
        }

    normal_completion = handoff.state == PDHandoffState.DECODE_ACCEPTED
    if not normal_completion and handoff.state != PDHandoffState.ROLLING_BACK:
        handoff, _accepted = manager.request_rollback(
            handoff_id,
            reason=str(payload.get("reason", "P Worker export released before decode ACK")),
        )
    released = _dispatch_block_pool_request(
        "/datadist/source-leases/release",
        {"lease_id": handoff.source_lease_id},
    )
    if not released.get("ok"):
        return released
    capture_release = _dispatch_block_pool_request(
        "/datadist/pd/export/release/local",
        {"handoff_id": handoff_id, "request_id": handoff.request_id},
    )
    if not capture_release.get("ok"):
        logger.warning(
            "Jiuwen KVBM source lease released but export capture cleanup failed: handoff_id=%s response=%s",
            handoff_id,
            capture_release,
        )
    final_state = PDHandoffState.SOURCE_RELEASED if normal_completion else PDHandoffState.ROLLED_BACK
    handoff = manager.advance(
        handoff_id,
        final_state,
        expected_state=(PDHandoffState.DECODE_ACCEPTED if normal_completion else PDHandoffState.ROLLING_BACK),
    )
    logger.info(
        "Jiuwen KVBM P Worker export released: handoff_id=%s request_id=%s state=%s",
        handoff_id,
        handoff.request_id,
        handoff.state.value,
    )
    return {
        "ok": True,
        "already_released": False,
        "source_lease": released["lease"],
        "capture_release": capture_release,
        "handoff": handoff.to_dict(),
    }


def _pd_import_response(
    control: "VllmAscend018DataPlaneControl",
    backend: LLMDataDistBackend,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """执行 D Worker reserve、HIXL pull 和 request attach 原子事务。

    参数:
        control: 当前 D Worker 数据面控制对象。
        backend: 已初始化并注册本地 KV cache 的 LLM-DataDist backend。
        payload: 包含 handoff_id、source_control_url、remote_cluster_id 和
            source_export 的请求体。source_export 应直接使用 P `/pd/export` 返回值。

    返回:
        成功时返回 decode_accepted/source_released handoff、目标 blocks 和阶段
        耗时；任一阶段失败时返回回滚结果，并解除 request gate 允许本地重算。
    """

    handoff_id = str(payload.get("handoff_id", "")).strip()
    source_control_url = str(payload.get("source_control_url", "")).strip()
    if not handoff_id or not source_control_url:
        raise ValueError("handoff_id and source_control_url are required")
    source_export = payload.get("source_export")
    if not isinstance(source_export, dict):
        raise ValueError("source_export must be the object returned by P /pd/export")
    request_id = str(payload.get("request_id", source_export.get("request_id", ""))).strip()
    source_blocks = [int(value) for value in source_export.get("source_blocks", [])]
    block_hashes = source_export.get("block_hashes")
    block_groups = source_export.get("block_groups")
    source_lease_id = str(source_export.get("source_lease_id", "")).strip()
    if not request_id or not source_blocks or not source_lease_id:
        raise ValueError("source_export is missing request_id, source_blocks or lease")
    if not isinstance(block_hashes, list | tuple) or len(block_hashes) != len(source_blocks):
        raise ValueError("source_export block_hashes must match source_blocks")
    if not isinstance(block_groups, list | tuple) or len(block_groups) != len(source_blocks):
        raise ValueError("source_export block_groups must match source_blocks")
    if not all(isinstance(item, dict) for item in block_groups):
        raise ValueError("source_export block_groups items must be objects")

    cache_id = _ENGINE_CORE_LLM_DATADIST_CACHE_ID
    if cache_id is None:
        raise ValueError("local LLM-DataDist cache is not registered")
    handoffs = _process_pd_handoffs()
    handoff = handoffs.get(handoff_id)
    if request_id != handoff.request_id:
        raise ValueError("source_export request_id does not match handoff")
    if not all(isinstance(item, dict) for item in block_hashes):
        raise ValueError("source_export block_hashes items must be objects")
    external_hashes = [
        str(item.get("external_hash", "")) for item in block_hashes if not item.get("request_scoped_partial")
    ]
    if any(not value for value in external_hashes):
        raise ValueError("source_export contains invalid complete block hashes")
    partial_descriptors = [item for item in block_hashes if item.get("request_scoped_partial")]
    if len(partial_descriptors) > len({int(item.get("group_id", -1)) for item in partial_descriptors}):
        raise ValueError("source_export contains duplicate partial cache groups")
    protocol_block_keys = _pd_protocol_block_keys(
        list(block_hashes),
        partial_token_fingerprint=str(source_export.get("partial_token_fingerprint", "")),
    )
    _synchronize_pd_handoff_source(
        handoffs,
        handoff_id,
        external_hashes=protocol_block_keys,
        source_blocks=source_blocks,
        source_lease_id=source_lease_id,
        source_export=source_export,
    )
    total_started = time.perf_counter()
    phase_ms: dict[str, float] = {}
    gate_renew_started = time.perf_counter()
    gate_renew = _dispatch_block_pool_request(
        "/datadist/pd/import/prepare/local",
        {
            "handoff_id": handoff_id,
            "request_id": request_id,
            "ttl_ms": int(payload.get("gate_ttl_ms", 120_000)),
            "require_existing": True,
        },
    )
    phase_ms["gate_renew"] = (time.perf_counter() - gate_renew_started) * 1000.0
    if not gate_renew.get("ok"):
        return gate_renew

    reservation_id = ""
    destination_blocks: list[int] = []
    planned_destination_blocks: list[int] = []
    transfer_source_blocks: list[int] = []
    transfer_positions: list[int] = []
    transfer_id = str(payload.get("transfer_id", f"pd-{handoff_id}"))
    attach_committed = False
    scheduler_owned = False
    source_lease = _AdoptedRemoteSourceLeaseGuard(
        source_control_url,
        source_lease_id,
        ttl_ms=int(payload.get("source_lease_ttl_ms", 30_000)),
    )
    reservation_guard: _TargetBlockReservationGuard | None = None
    try:
        source_lease.start()

        plan_started = time.perf_counter()
        reserve_response = _dispatch_block_pool_request(
            "/datadist/pd/import/plan/local",
            {
                "handoff_id": handoff_id,
                "request_id": request_id,
                "block_groups": list(block_groups),
                "token_block_fingerprints": list(source_export.get("token_block_fingerprints", [])),
                "partial_token_fingerprint": str(source_export.get("partial_token_fingerprint", "")),
                "partial_valid_tokens": int(source_export.get("partial_valid_tokens", 0)),
                "semantic_plain_text": bool(source_export.get("semantic_plain_text", False)),
                "reservation_ttl_ms": int(payload.get("reservation_ttl_ms", 120_000)),
                "timeout_ms": int(
                    payload.get(
                        "capacity_wait_timeout_ms",
                        payload.get("attach_timeout_ms", 10_000),
                    )
                ),
            },
        )
        phase_ms["delta_plan"] = (time.perf_counter() - plan_started) * 1000.0
        if not reserve_response.get("ok"):
            raise RuntimeError(f"failed to plan D delta import: {reserve_response.get('error')}")
        phase_ms["reserve"] = float(reserve_response.get("reserve_ms", 0.0))
        for timing_name in (
            "delta_plan_task_queue_ms",
            "delta_hash_validate_ms",
            "delta_reserve_ms",
            "delta_plan_execution_ms",
            "delta_preplan_age_ms",
            "delta_inflight_wait_ms",
            "delta_inflight_retry_count",
            "delta_capacity_wait_ms",
            "delta_capacity_retry_count",
            "import_executor_queue_ms",
        ):
            phase_ms[timing_name] = float(reserve_response.get(timing_name, 0.0))
        phase_ms["delta_preplanned"] = float(bool(reserve_response.get("delta_preplanned", False)))
        reservation = reserve_response["reservation"]
        scheduler_owned = bool(reserve_response.get("scheduler_owned", False))
        reservation_id = str(reservation["reservation_id"])
        destination_blocks = [int(value) for value in reservation["block_ids"]]
        planned_destination_blocks = [int(value) for value in reserve_response["planned_destination_blocks"]]
        transfer_positions = [int(value) for value in reserve_response["transfer_positions"]]
        transfer_source_blocks = [source_blocks[index] for index in transfer_positions]
        if len(transfer_source_blocks) != len(destination_blocks):
            raise RuntimeError("delta import source and destination mappings are not aligned")
        handoffs.advance(
            handoff_id,
            PDHandoffState.TARGET_RESERVED,
            expected_state=PDHandoffState.SOURCE_PINNED,
            destination_blocks=planned_destination_blocks,
            reservation_id=reservation_id,
        )

        if not scheduler_owned:
            reservation_guard = _TargetBlockReservationGuard(
                reservation_id,
                ttl_ms=int(payload.get("reservation_ttl_ms", 120_000)),
            )
            reservation_guard.start()
        handoffs.advance(
            handoff_id,
            PDHandoffState.TRANSFERRING,
            expected_state=PDHandoffState.TARGET_RESERVED,
            transfer_id=transfer_id,
        )
        hixl_started = time.perf_counter()
        pull_start_unix = time.time()
        if transfer_source_blocks:
            with _ENGINE_CORE_LLM_DATADIST_LOCK:
                backend.pull_blocks(
                    cache_id,
                    LLMDataDistBlockMapping(
                        cache_key=LLMDataDistCacheKey(
                            key=str(payload.get("cache_key", payload["remote_cluster_id"])),
                            prompt_cluster_id=int(payload["remote_cluster_id"]),
                            prompt_batch_index=int(payload.get("model_id", 0)),
                        ),
                        prompt_blocks=tuple(transfer_source_blocks),
                        decoder_blocks=tuple(destination_blocks),
                        layer_range=_optional_layer_range(payload),
                    ),
                )
        phase_ms["hixl_pull"] = (time.perf_counter() - hixl_started) * 1000.0
        logger.info(
            "[KVBMPDWorkWindow] phase=pull handoff_id=%s start_unix_s=%.6f end_unix_s=%.6f ok=True",
            handoff_id,
            pull_start_unix,
            time.time(),
        )

        if reservation_guard is not None:
            reservation_guard.prepare_publish()
        attach_started = time.perf_counter()
        attach_response = _dispatch_block_pool_request(
            "/datadist/pd/import/attach/local",
            {
                "handoff_id": handoff_id,
                "request_id": request_id,
                "reservation_id": reservation_id,
                "ownership_mode": (
                    SchedulerKVOwnershipMode.SCHEDULER.value
                    if scheduler_owned
                    else SchedulerKVOwnershipMode.LEGACY.value
                ),
                "block_hashes": list(block_hashes),
                "block_groups": list(block_groups),
                "token_block_fingerprints": list(source_export.get("token_block_fingerprints", [])),
                "partial_token_fingerprint": str(source_export.get("partial_token_fingerprint", "")),
                "partial_valid_tokens": int(source_export.get("partial_valid_tokens", 0)),
                "semantic_plain_text": bool(source_export.get("semantic_plain_text", False)),
                "timeout_ms": int(payload.get("attach_timeout_ms", 10_000)),
            },
        )
        phase_ms["request_attach"] = (time.perf_counter() - attach_started) * 1000.0
        if not attach_response.get("ok"):
            raise RuntimeError(f"failed to attach D request blocks: {attach_response.get('error')}")
        scheduler_timing = dict(attach_response.get("scheduler_timing") or {})
        for timing_name in (
            "d_request_seen_to_attach_start_ms",
            "attach_task_queue_ms",
            "attach_execution_ms",
        ):
            if timing_name in scheduler_timing:
                phase_ms[timing_name] = float(scheduler_timing[timing_name])
        attach_committed = True
        pulled_destination_blocks = list(destination_blocks)
        destination_blocks = [int(value) for value in attach_response["destination_blocks"]]
        if reservation_guard is not None:
            reservation_guard.close()
        block_pool = _latest_reservable_block_pool()
        with _VLLM_BLOCK_POOL_LOCK:
            free_blocks_after_attach = int(block_pool.get_num_free_blocks()) if block_pool is not None else None
            capacity_generation_after_attach = (
                int(
                    getattr(
                        block_pool,
                        _BLOCK_POOL_CAPACITY_GENERATION_ATTR,
                        -1,
                    )
                )
                if block_pool is not None
                else None
            )
        handoff = handoffs.advance(
            handoff_id,
            PDHandoffState.TARGET_ATTACHED,
            expected_state=PDHandoffState.TRANSFERRING,
            metadata={
                "pulled_destination_blocks": pulled_destination_blocks,
                "planned_destination_blocks": planned_destination_blocks,
                "transfer_positions": transfer_positions,
                "attached_destination_blocks": list(destination_blocks),
                "reused_published_blocks": list(attach_response.get("reused_published_blocks", [])),
            },
        )
        handoff = handoffs.advance(
            handoff_id,
            PDHandoffState.DECODE_ACCEPTED,
            expected_state=PDHandoffState.TARGET_ATTACHED,
        )

        attached_hashes = attach_response.get("published_block_hashes") or []
        published_destination_blocks = [int(value) for value in attach_response.get("published_destination_blocks", [])]
        attached_external_hashes = tuple(
            str(value.get("external_hash", "")) for value in attached_hashes if isinstance(value, dict)
        )
        metadata_task = _PublishedMetadataTask(
            transfer_id=transfer_id,
            control=control,
            block_ids=tuple(published_destination_blocks),
            packed_hashes=tuple(_decode_packed_block_hash(value).hex() for value in attached_hashes),
            external_hashes=(
                attached_external_hashes
                if len(attached_external_hashes) == len(published_destination_blocks)
                else tuple(external_hashes)
            ),
        )
        if published_destination_blocks:
            try:
                metadata_submit = _async_metadata_publisher().submit(metadata_task)
            except Exception as exc:
                metadata_submit = {"queued": False, "error": str(exc)}
                logger.warning(
                    "Jiuwen KVBM D request attached but metadata enqueue failed: handoff_id=%s error=%s",
                    handoff_id,
                    exc,
                )
        else:
            metadata_submit = {
                "queued": False,
                "skipped": True,
                "reason": "no_new_prefix_blocks",
            }
        reconcile_started = time.perf_counter()
        _start_pd_import_reconciliation(
            handoffs=handoffs,
            source_control_url=source_control_url,
            handoff_id=handoff_id,
            destination_blocks=planned_destination_blocks,
            reservation_id=reservation_id,
            transfer_id=transfer_id,
        )
        phase_ms["reconcile_enqueue"] = (time.perf_counter() - reconcile_started) * 1000.0
        phase_ms["pd_import_total"] = (time.perf_counter() - total_started) * 1000.0
        logger.info(
            "Jiuwen KVBM D Worker import completed: handoff_id=%s request_id=%s "
            "total_blocks=%d reused_blocks=%d transferred_blocks=%d "
            "transfer_positions=%s source_blocks=%s destination_blocks=%s computed_tokens=%s "
            "partial_valid_tokens=%s request_scoped_destination_blocks=%s "
            "phase_ms=%s",
            handoff_id,
            request_id,
            len(source_blocks),
            len(source_blocks) - len(transfer_source_blocks),
            len(transfer_source_blocks),
            transfer_positions,
            transfer_source_blocks,
            destination_blocks,
            attach_response.get("computed_tokens"),
            attach_response.get("partial_valid_tokens", 0),
            attach_response.get("request_scoped_destination_blocks", []),
            phase_ms,
        )
        return {
            "ok": True,
            "handoff": handoff.to_dict(),
            "source_blocks": source_blocks,
            "transfer_source_blocks": transfer_source_blocks,
            "transfer_destination_blocks": pulled_destination_blocks,
            "transfer_positions": transfer_positions,
            "reused_block_count": len(source_blocks) - len(transfer_source_blocks),
            "transferred_block_count": len(transfer_source_blocks),
            "destination_blocks": destination_blocks,
            "attach": attach_response,
            "source_release": {"queued": True},
            "remote_reconciliation": {"queued": True},
            "metadata_publish": metadata_submit,
            "scheduler_timing": scheduler_timing,
            "phase_ms": phase_ms,
            "free_blocks": free_blocks_after_attach,
            "capacity_generation": capacity_generation_after_attach,
        }
    except Exception as exc:
        if reservation_guard is not None:
            reservation_guard.close()
        if attach_committed:
            source_lease.close()
            logger.exception(
                "Jiuwen KVBM D request attach committed but protocol reconciliation "
                "failed; preserving request-owned blocks: handoff_id=%s request_id=%s",
                handoff_id,
                request_id,
            )
            return {
                "ok": False,
                "attached": True,
                "requires_reconciliation": True,
                "error": str(exc),
                "handoff": handoffs.get(handoff_id).to_dict(),
                "source_blocks": source_blocks,
                "destination_blocks": destination_blocks,
                "rollback": {"ok": True, "skipped": True, "reason": "attach_committed"},
                "phase_ms": phase_ms,
            }
        rollback = (
            {
                "ok": True,
                "skipped": True,
                "reason": "scheduler_owned_request_blocks",
            }
            if scheduler_owned
            else (_rollback_block_reservation(reservation_id) if reservation_id else {"ok": True, "skipped": True})
        )
        gate_release = _dispatch_block_pool_request(
            "/datadist/pd/import/release/local",
            {"handoff_id": handoff_id, "request_id": request_id},
        )
        try:
            handoff, _accepted = handoffs.request_rollback(
                handoff_id,
                reason=str(exc),
            )
            if handoff.state == PDHandoffState.ROLLING_BACK:
                handoff = handoffs.advance(
                    handoff_id,
                    PDHandoffState.ROLLED_BACK,
                    expected_state=PDHandoffState.ROLLING_BACK,
                )
        except Exception:
            handoff = handoffs.get(handoff_id)
        remote_rollback = _post_remote_control(
            _normalize_control_url(source_control_url),
            "/datadist/pd/handoffs/rollback",
            {"handoff_id": handoff_id, "reason": str(exc)},
        )
        source_release = _post_remote_control(
            _normalize_control_url(source_control_url),
            "/datadist/pd/export/release",
            {"handoff_id": handoff_id, "reason": str(exc)},
        )
        phase_ms["pd_import_total"] = (time.perf_counter() - total_started) * 1000.0
        logger.warning(
            "Jiuwen KVBM D Worker import rolled back: handoff_id=%s request_id=%s error=%s rollback=%s gate_release=%s",
            handoff_id,
            request_id,
            exc,
            rollback,
            gate_release,
        )
        return {
            "ok": False,
            "error": str(exc),
            "handoff": handoff.to_dict(),
            "source_blocks": source_blocks,
            "destination_blocks": destination_blocks,
            "rollback": rollback,
            "gate_release": gate_release,
            "remote_rollback": remote_rollback,
            "source_release": source_release,
            "phase_ms": phase_ms,
        }
    finally:
        source_lease.close()


def _synchronize_pd_handoff_source(
    manager: PDHandoffManager,
    handoff_id: str,
    *,
    external_hashes: list[str],
    source_blocks: list[int],
    source_lease_id: str,
    source_export: dict[str, Any],
) -> None:
    """把 P export 资源幂等同步到 D Worker 的本地 handoff。

    参数:
        manager: D Worker 的 PDHandoffManager。
        handoff_id: 待同步 handoff ID。
        external_hashes: P export 返回的逻辑 block hash。
        source_blocks: P Worker 物理 block ID。
        source_lease_id: P Worker source lease ID。
        source_export: 完整 P export 元数据。

    返回:
        handoff 到达 SOURCE_PINNED 时无返回值；资源冲突时抛出异常。
    """

    handoff = manager.get(handoff_id)
    if handoff.state == PDHandoffState.PREFILL_RUNNING:
        handoff = manager.advance(
            handoff_id,
            PDHandoffState.PREFILL_READY,
            expected_state=PDHandoffState.PREFILL_RUNNING,
            block_hashes=external_hashes,
            source_blocks=source_blocks,
            metadata={"source_export": source_export},
        )
    if handoff.state == PDHandoffState.PREFILL_READY:
        manager.advance(
            handoff_id,
            PDHandoffState.SOURCE_PINNED,
            expected_state=PDHandoffState.PREFILL_READY,
            block_hashes=external_hashes,
            source_blocks=source_blocks,
            source_lease_id=source_lease_id,
            metadata={"source_export": source_export},
        )
    elif handoff.state == PDHandoffState.SOURCE_PINNED:
        manager.advance(
            handoff_id,
            PDHandoffState.SOURCE_PINNED,
            block_hashes=external_hashes,
            source_blocks=source_blocks,
            source_lease_id=source_lease_id,
            metadata={"source_export": source_export},
        )
    else:
        raise RuntimeError(f"D handoff state {handoff.state.value} cannot start import")


def _advance_remote_pd_handoff(
    source_control_url: str,
    handoff_id: str,
    state: PDHandoffState,
    **resources: Any,
) -> None:
    """推进 P Worker 上的镜像 handoff 状态。

    参数:
        source_control_url: P Worker KVBM 控制服务根地址。
        handoff_id: 需要推进的 handoff ID。
        state: 目标 PD handoff 状态。
        resources: 当前状态新增的 destination_blocks、reservation_id 或 transfer_id。

    返回:
        P Worker 接受状态更新时无返回值；远端失败时抛出 RuntimeError。
    """

    response = _post_remote_control(
        _normalize_control_url(source_control_url),
        "/datadist/pd/handoffs/advance",
        {"handoff_id": handoff_id, "state": state.value, **resources},
    )
    if not response.get("ok"):
        raise RuntimeError(f"failed to advance P handoff to {state.value}: {response.get('error')}")


def _try_advance_remote_pd_handoff(
    source_control_url: str,
    handoff_id: str,
    state: PDHandoffState,
    **resources: Any,
) -> dict[str, Any]:
    """在 D request attach 提交后尽力同步 P handoff，不反向回滚 D blocks。

    参数:
        source_control_url: P Worker KVBM 控制服务根地址。
        handoff_id: 需要同步的 handoff ID。
        state: 目标 PD handoff 状态。
        resources: 状态对应的可选资源字段。

    返回:
        P Worker 原始响应；网络或状态错误转换为 ok=False 诊断字典。
    """

    try:
        _advance_remote_pd_handoff(
            source_control_url,
            handoff_id,
            state,
            **resources,
        )
        return {"ok": True, "state": state.value}
    except Exception as exc:
        logger.warning(
            "Jiuwen KVBM D attach committed but P handoff sync failed: handoff_id=%s state=%s error=%s",
            handoff_id,
            state.value,
            exc,
        )
        return {"ok": False, "state": state.value, "error": str(exc)}


def _start_pd_import_reconciliation(
    *,
    handoffs: PDHandoffManager,
    source_control_url: str,
    handoff_id: str,
    destination_blocks: list[int],
    reservation_id: str,
    transfer_id: str,
) -> None:
    """在 request attach 提交后异步同步 P 状态并释放 source lease。

    参数:
        handoffs: D Worker 本地 PD handoff 管理器。
        source_control_url: P Worker KVBM 控制服务地址。
        handoff_id: 当前请求的 PD handoff ID。
        destination_blocks: delta plan 合并本地复用与 HIXL 写入后的完整 D 端布局。
        reservation_id: D BlockPool reservation ID。
        transfer_id: 本次 HIXL 传输事务 ID。

    返回:
        无返回值。函数只负责启动后台 reconciliation；任一远端步骤失败时由
        source lease TTL 最终释放 P block，不反向回滚已提交给 D 请求的 block。
    """

    def reconcile() -> None:
        started = time.perf_counter()
        steps = (
            (
                PDHandoffState.TARGET_RESERVED,
                {
                    "destination_blocks": destination_blocks,
                    "reservation_id": reservation_id,
                },
            ),
            (PDHandoffState.TRANSFERRING, {"transfer_id": transfer_id}),
            (PDHandoffState.TARGET_ATTACHED, {}),
            (PDHandoffState.DECODE_ACCEPTED, {}),
        )
        try:
            for state, resources in steps:
                _advance_remote_pd_handoff(
                    source_control_url,
                    handoff_id,
                    state,
                    **resources,
                )
            source_release = _post_remote_control(
                _normalize_control_url(source_control_url),
                "/datadist/pd/export/release",
                {"handoff_id": handoff_id},
            )
            if not source_release.get("ok"):
                raise RuntimeError(f"P source release failed: {source_release.get('error', source_release)}")
            handoffs.advance(
                handoff_id,
                PDHandoffState.SOURCE_RELEASED,
                expected_state=PDHandoffState.DECODE_ACCEPTED,
            )
            logger.info(
                "Jiuwen KVBM async PD reconciliation completed: handoff_id=%s phase_ms=%.3f",
                handoff_id,
                (time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            logger.warning(
                "Jiuwen KVBM async PD reconciliation failed; source lease TTL "
                "will release P blocks: handoff_id=%s error=%s",
                handoff_id,
                exc,
            )

    _PD_RECONCILE_EXECUTOR.submit(reconcile)


def _optional_list(
    payload: dict[str, Any],
    name: str,
    converter: Callable[[Any], Any],
) -> list[Any] | None:
    """读取可选列表并转换每个元素。

    参数:
        payload: 原始请求体。
        name: 待读取字段名。
        converter: 单个列表元素的转换函数。

    返回:
        字段缺失时返回 None，否则返回转换后的列表。
    """

    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a list")
    return [converter(item) for item in value]


def _optional_string(payload: dict[str, Any], name: str) -> str | None:
    """读取可选非空字符串。

    参数:
        payload: 原始请求体。
        name: 待读取字段名。

    返回:
        字段缺失时返回 None，否则返回去除首尾空白后的字符串。
    """

    value = payload.get(name)
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _transfer_request_fingerprint(payload: dict[str, Any]) -> str:
    """计算跨 Worker 传输请求的稳定语义指纹。

    参数:
        payload: `/datadist/transfers/pull` 请求体。

    返回:
        排除 transfer_id 和可调超时参数后的 SHA-256 十六进制指纹。

    说明:
        重试时允许调整 lease/reservation TTL，但不允许同一 transfer_id 改变
        源端、block 列表、模型、cache key 或 layer 范围。
    """

    ignored = {
        "transfer_id",
        "timeout_ms",
        "source_lease_ttl_ms",
        "reservation_ttl_ms",
        "deadline_epoch_ms",
    }
    semantic_payload = {str(key): value for key, value in payload.items() if str(key) not in ignored}
    encoded = json.dumps(
        semantic_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pull_blocks_and_publish_idempotent(
    control: "VllmAscend018DataPlaneControl",
    backend: LLMDataDistBackend,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """以 transfer_id 幂等执行跨 Worker block 拉取事务。

    参数:
        control: 当前 engine-core worker 的 KVBM 数据面控制对象。
        backend: 已初始化并注册本地 KV cache 的 LLM-DataDist backend。
        payload: 传输请求体；可包含调用方生成的 transfer_id。

    返回:
        首次调用同步返回最终结果；重复调用返回已有终态结果或进行中
        快照。
    """

    source_blocks = _required_int_list(payload, "source_blocks")
    manager = _process_transfer_transactions()
    try:
        transaction, created = manager.begin(
            transfer_id=(str(payload["transfer_id"]).strip() if payload.get("transfer_id") is not None else None),
            request_fingerprint=_transfer_request_fingerprint(payload),
            source_blocks=source_blocks,
        )
    except (TransferTransactionError, TransferCapacityError) as exc:
        return {"ok": False, "error": str(exc)}

    if not created:
        if transaction.terminal and transaction.result is not None:
            response = dict(transaction.result)
            response["idempotent_replay"] = True
        else:
            response = {
                "ok": True,
                "accepted": True,
                "completed": False,
                "idempotent_replay": True,
            }
        response["transfer"] = transaction.to_dict(include_result=False)
        return response

    transfer_id = transaction.transfer_id
    try:
        manager.raise_if_cancelled(transfer_id)
        with _ENGINE_CORE_LLM_DATADIST_LOCK:
            result = _pull_blocks_and_publish(
                control,
                backend,
                payload,
                transfer_id=transfer_id,
            )
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "source_blocks": source_blocks}

    if result.get("ok"):
        final_state = TransferTransactionState.COMPLETED
    elif result.get("cancelled"):
        final_state = TransferTransactionState.CANCELLED
    elif result.get("rollback", {}).get("ok"):
        final_state = TransferTransactionState.ROLLED_BACK
    else:
        final_state = TransferTransactionState.FAILED
    transaction = manager.finish(
        transfer_id,
        final_state,
        result=result,
        error=None if result.get("ok") else str(result.get("error", "transfer failed")),
    )
    result["transfer"] = transaction.to_dict(include_result=False)
    return result


def _pull_blocks_and_publish(
    control: "VllmAscend018DataPlaneControl",
    backend: LLMDataDistBackend,
    payload: dict[str, Any],
    *,
    transfer_id: str,
) -> dict[str, Any]:
    """持有远端源 lease，并执行目标 block 拉取和原子发布事务。

    参数:
        control: 当前 engine-core worker 的 KVBM 数据面控制对象。
        backend: 已初始化并注册本地 KV cache 的 LLM-DataDist backend。
        payload: 事务请求体。source_blocks 是源端物理 block ID；
            source_control_url 是源 worker 的 KVBM 控制服务地址。
        transfer_id: 当前幂等传输事务 ID。

    返回:
        成功时返回源 lease、源/目标 block 和发布数量；失败时返回 ok=False。
        无论成功失败都会释放源 lease，网络异常时由源端 TTL 回收兜底。
    """

    cache_id = _ENGINE_CORE_LLM_DATADIST_CACHE_ID
    if cache_id is None:
        raise ValueError("local LLM-DataDist cache is not registered")
    total_started = time.perf_counter()
    phase_ms: dict[str, float] = {}
    source_blocks = _required_int_list(payload, "source_blocks")
    source_control_url = str(payload.get("source_control_url", "")).strip()
    if not source_control_url:
        raise ValueError("source_control_url is required for concurrency-safe source block pinning")
    source_lease = _RemoteSourceLeaseGuard(
        source_control_url,
        source_blocks,
        ttl_ms=int(payload.get("source_lease_ttl_ms", 30_000)),
    )
    try:
        lease_acquire_started = time.perf_counter()
        _process_transfer_transactions().raise_if_cancelled(transfer_id)
        lease_snapshot = source_lease.acquire()
        phase_ms["lease_acquire"] = (time.perf_counter() - lease_acquire_started) * 1000.0
    except Exception as exc:
        phase_ms["lease_acquire"] = (time.perf_counter() - lease_acquire_started) * 1000.0
        phase_ms["worker_total"] = (time.perf_counter() - total_started) * 1000.0
        return {
            "ok": False,
            "error": f"failed to acquire source block lease: {exc}",
            "source_blocks": source_blocks,
            "cancelled": isinstance(exc, TransferCancelledError),
            "phase_ms": phase_ms,
        }

    result: dict[str, Any]
    try:
        _process_transfer_transactions().update(
            transfer_id,
            TransferTransactionState.LEASED,
            source_lease_id=str(lease_snapshot.get("lease_id", "")),
        )
        _process_transfer_transactions().raise_if_cancelled(transfer_id)
        _raise_if_transfer_deadline_expired(payload)
        hash_descriptors = lease_snapshot.get("block_hashes")
        if not isinstance(hash_descriptors, list | tuple) or len(hash_descriptors) != len(source_blocks):
            raise RuntimeError("source lease returned an invalid block_hashes list")
        _validate_expected_source_hashes(payload, hash_descriptors)
        _validate_expected_external_source_hashes(payload, hash_descriptors)
        result = _pull_leased_blocks_and_publish(
            control,
            backend,
            payload,
            source_blocks=source_blocks,
            hash_descriptors=list(hash_descriptors),
            source_lease=source_lease,
            transfer_id=transfer_id,
        )
    except Exception as exc:
        result = {
            "ok": False,
            "error": str(exc),
            "source_blocks": source_blocks,
            "cancelled": isinstance(exc, TransferCancelledError),
        }
    phase_ms.update(result.get("phase_ms") or {})
    lease_release_started = time.perf_counter()
    release_response = source_lease.release()
    phase_ms["lease_release"] = (time.perf_counter() - lease_release_started) * 1000.0
    phase_ms["worker_total"] = (time.perf_counter() - total_started) * 1000.0
    result["phase_ms"] = phase_ms
    if not release_response.get("ok"):
        logger.warning(
            "Jiuwen KVBM failed to release source block lease immediately; "
            "source TTL cleanup will retry safety release: lease_id=%s response=%s",
            lease_snapshot.get("lease_id"),
            release_response,
        )
    result["source_lease"] = {
        "lease_id": lease_snapshot.get("lease_id"),
        "release": release_response,
    }
    return result


@dataclass(frozen=True)
class _PublishedMetadataTask:
    """一次已经完成 vLLM publish、等待注册 KVBM 元数据的批次。"""

    transfer_id: str
    control: Any
    block_ids: tuple[int, ...]
    packed_hashes: tuple[str, ...]
    external_hashes: tuple[str, ...]

    @property
    def keys(self) -> tuple[tuple[int, str], ...]:
        """返回用于队列内去重的物理 block 与完整 hash 组合。"""

        return tuple(zip(self.block_ids, self.packed_hashes, strict=True))


class _AsyncPublishedMetadataPublisher:
    """在请求关键路径外校验并发布跨 Worker block 的 KVBM 元数据。"""

    def __init__(self, *, max_pending_batches: int) -> None:
        """创建单线程有界发布器。

        参数:
            max_pending_batches: 最多允许排队的元数据批次数量。

        返回:
            无返回值。
        """

        self._queue: queue.Queue[_PublishedMetadataTask | None] = queue.Queue(maxsize=max_pending_batches)
        self._pending_keys: set[tuple[int, str]] = set()
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="jiuwen-kvbm-metadata-publisher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, task: _PublishedMetadataTask) -> dict[str, Any]:
        """提交一个元数据批次，并过滤仍在队列中的重复 block。

        参数:
            task: 已经完成 vLLM BlockPool publish 的 block 元数据批次。

        返回:
            包含 queued、duplicate_count、queued_count 和 reason 的提交结果。
        """

        with self._lock:
            if self._closed:
                return {"queued": False, "reason": "publisher_closed"}
            selected_indexes = [index for index, key in enumerate(task.keys) if key not in self._pending_keys]
            duplicate_count = len(task.block_ids) - len(selected_indexes)
            if not selected_indexes:
                return {
                    "queued": True,
                    "duplicate_count": duplicate_count,
                    "queued_count": 0,
                    "reason": "already_pending",
                }
            if self._queue.full():
                return {
                    "queued": False,
                    "duplicate_count": duplicate_count,
                    "queued_count": 0,
                    "reason": "queue_full",
                }
            selected = _PublishedMetadataTask(
                transfer_id=task.transfer_id,
                control=task.control,
                block_ids=tuple(task.block_ids[index] for index in selected_indexes),
                packed_hashes=tuple(task.packed_hashes[index] for index in selected_indexes),
                external_hashes=tuple(task.external_hashes[index] for index in selected_indexes),
            )
            self._pending_keys.update(selected.keys)
            self._queue.put_nowait(selected)
            return {
                "queued": True,
                "duplicate_count": duplicate_count,
                "queued_count": len(selected.block_ids),
                "reason": "queued",
            }

    def wait_for_idle(self, timeout_seconds: float) -> bool:
        """等待当前队列完成，主要用于测试和进程优雅退出。

        参数:
            timeout_seconds: 最大等待时间，单位秒。

        返回:
            队列在超时前清空返回 True，否则返回 False。
        """

        deadline = time.monotonic() + max(0.0, timeout_seconds)
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
        return True

    def close(self, timeout_seconds: float = 2.0) -> None:
        """停止接收新任务，并尽量排空已经提交的批次。

        参数:
            timeout_seconds: 等待队列排空及线程退出的最长时间，单位秒。

        返回:
            无返回值。
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.wait_for_idle(timeout_seconds)
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return
        self._thread.join(timeout=max(0.0, timeout_seconds))

    def _run(self) -> None:
        """持续消费元数据任务，失败时记录诊断并继续处理后续批次。"""

        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                for attempt in range(1, 4):
                    try:
                        _publish_published_block_metadata(task)
                        break
                    except Exception:
                        if attempt == 3:
                            logger.exception(
                                "Jiuwen KVBM async metadata publish failed after retries: transfer_id=%s block_ids=%s",
                                task.transfer_id,
                                task.block_ids,
                            )
                            break
                        logger.warning(
                            "Jiuwen KVBM async metadata publish attempt failed; retrying: transfer_id=%s attempt=%d",
                            task.transfer_id,
                            attempt,
                            exc_info=True,
                        )
                        time.sleep(0.05 * attempt)
            finally:
                if task is not None:
                    with self._lock:
                        self._pending_keys.difference_update(task.keys)
                self._queue.task_done()


def _async_metadata_publisher() -> _AsyncPublishedMetadataPublisher:
    """返回进程级异步元数据发布器，并在首次使用时启动线程。"""

    global _ASYNC_METADATA_PUBLISHER
    with _ASYNC_METADATA_PUBLISHER_LOCK:
        if _ASYNC_METADATA_PUBLISHER is None:
            _ASYNC_METADATA_PUBLISHER = _AsyncPublishedMetadataPublisher(
                max_pending_batches=_ASYNC_METADATA_QUEUE_LIMIT
            )
        return _ASYNC_METADATA_PUBLISHER


def _wait_for_async_metadata_publish(timeout_seconds: float = 5.0) -> bool:
    """等待异步元数据队列清空。

    参数:
        timeout_seconds: 最大等待时间，单位秒。

    返回:
        发布器尚未创建或队列已清空返回 True，等待超时返回 False。
    """

    publisher = _ASYNC_METADATA_PUBLISHER
    return publisher is None or publisher.wait_for_idle(timeout_seconds)


def _shutdown_async_metadata_publisher() -> None:
    """在解释器退出时尽量排空并关闭进程级元数据发布器。

    返回:
        无返回值。
    """

    publisher = _ASYNC_METADATA_PUBLISHER
    if publisher is not None:
        publisher.close()


atexit.register(_shutdown_async_metadata_publisher)


def _shutdown_pd_reconcile_executor() -> None:
    """停止 PD 后台状态收敛线程池。

    返回:
        无返回值。进程退出时不等待网络收尾任务，未释放的 source lease 由
        P Worker TTL 回收。
    """

    _PD_RECONCILE_EXECUTOR.shutdown(wait=False, cancel_futures=False)
    _PD_IMPORT_EXECUTOR.shutdown(wait=False, cancel_futures=False)


atexit.register(_shutdown_pd_reconcile_executor)


def _publish_published_block_metadata(task: _PublishedMetadataTask) -> None:
    """校验 BlockPool 所有权后注册 KVBM 元数据并发送 location event。

    参数:
        task: 已完成 vLLM publish 的目标 block 元数据批次。

    返回:
        无返回值；校验或注册失败时抛出异常，由发布线程记录。
    """

    started = time.perf_counter()
    validation_started = time.perf_counter()
    response = _dispatch_block_pool_request(
        "/datadist/blocks/metadata",
        {"block_ids": list(task.block_ids), "allow_missing": True},
    )
    if not response.get("ok"):
        raise RuntimeError(f"failed to validate published block ownership: {response.get('error')}")
    actual_hashes = {
        int(item["block_id"]): str(item["block_hash"]["packed_hex"])
        for item in response.get("blocks", [])
        if isinstance(item, dict) and isinstance(item.get("block_hash"), dict)
    }
    valid_indexes = []
    for selected_index, (selected_block_id, selected_packed_hash) in enumerate(
        zip(task.block_ids, task.packed_hashes, strict=True)
    ):
        if actual_hashes.get(selected_block_id) == selected_packed_hash:
            valid_indexes.append(selected_index)
    validation_ms = (time.perf_counter() - validation_started) * 1000.0
    stale_count = len(task.block_ids) - len(valid_indexes)
    if not valid_indexes:
        logger.info(
            "Jiuwen KVBM skipped stale async metadata batch: transfer_id=%s requested=%d ownership_validate_ms=%.3f",
            task.transfer_id,
            len(task.block_ids),
            validation_ms,
        )
        return

    candidate_block_ids = [task.block_ids[index] for index in valid_indexes]
    candidate_external_hashes = [task.external_hashes[index] for index in valid_indexes]
    candidate_packed_hashes = [task.packed_hashes[index] for index in valid_indexes]
    pin_started = time.perf_counter()
    pin_response = _dispatch_block_pool_request(
        "/datadist/source-leases/acquire",
        {"block_ids": candidate_block_ids, "ttl_ms": 30_000},
    )
    if not pin_response.get("ok"):
        raise RuntimeError(f"failed to pin published blocks for metadata: {pin_response.get('error')}")
    lease = pin_response.get("lease") or {}
    lease_id = str(lease.get("lease_id", ""))
    leased_descriptors = lease.get("block_hashes") or []
    leased_hashes = [
        str(descriptor.get("packed_hex", "")) if isinstance(descriptor, dict) else ""
        for descriptor in leased_descriptors
    ]
    stable_indexes = []
    for matched_selected_index, (selected_expected, selected_actual) in enumerate(
        zip(candidate_packed_hashes, leased_hashes, strict=False)
    ):
        if selected_expected == selected_actual:
            stable_indexes.append(matched_selected_index)
    stale_count += len(candidate_block_ids) - len(stable_indexes)
    block_ids = [candidate_block_ids[index] for index in stable_indexes]
    external_hashes = [candidate_external_hashes[index] for index in stable_indexes]
    pin_ms = (time.perf_counter() - pin_started) * 1000.0
    register_ms = 0.0
    event_ms = 0.0
    try:
        if not block_ids:
            logger.info(
                "Jiuwen KVBM skipped metadata batch changed before pin: transfer_id=%s requested=%d",
                task.transfer_id,
                len(task.block_ids),
            )
            return
        register_started = time.perf_counter()
        store_events = _register_published_metadata_with_fallback(
            task.control,
            block_ids,
            external_hashes,
        )
        register_ms = (time.perf_counter() - register_started) * 1000.0
        event_started = time.perf_counter()
        _enqueue_runtime_events(store_events, include_store=True)
        event_ms = (time.perf_counter() - event_started) * 1000.0
    finally:
        if lease_id:
            release_response = _dispatch_block_pool_request(
                "/datadist/source-leases/release",
                {"lease_id": lease_id},
            )
            if not release_response.get("ok"):
                logger.warning(
                    "Jiuwen KVBM failed to release async metadata ownership lease; "
                    "TTL will release it: transfer_id=%s lease_id=%s response=%s",
                    task.transfer_id,
                    lease_id,
                    release_response,
                )
    logger.info(
        "Jiuwen KVBM async metadata publish completed: transfer_id=%s requested=%d published=%d stale=%d phase_ms=%s",
        task.transfer_id,
        len(task.block_ids),
        len(block_ids),
        stale_count,
        {
            "ownership_validate": validation_ms,
            "ownership_pin": pin_ms,
            "metadata_register": register_ms,
            "metadata_event_enqueue": event_ms,
            "metadata_publish_async": (time.perf_counter() - started) * 1000.0,
        },
    )


def _register_published_metadata_with_fallback(
    control: Any,
    block_ids: list[int],
    external_hashes: list[str],
) -> list[Any]:
    """优先批量注册元数据，失败时逐 block 退化并保留 relay 事件。

    参数:
        control: 当前 engine-core 的 KVBM 数据面控制对象。
        block_ids: 已通过 BlockPool 所有权校验的物理 block ID。
        external_hashes: 与 block_ids 对齐的 Router 可见 block hash。

    返回:
        可放入 engine-core runtime event outbox 的 store 事件列表。
    """

    try:
        return [
            control.register_published_physical_blocks(
                block_ids,
                external_hashes,
                token_count=control.config.block_size,
                active=False,
            )
        ]
    except Exception as batch_exc:
        logger.warning(
            "Jiuwen KVBM batch metadata registration failed after vLLM publish; "
            "falling back to per-block registration: %s",
            batch_exc,
        )
    store_events = []
    for block_id, block_hash in zip(block_ids, external_hashes, strict=True):
        try:
            store_events.append(
                control.register_physical_block(
                    block_id,
                    block_hash,
                    token_count=control.config.block_size,
                    active=False,
                )
            )
        except Exception as exc:
            logger.warning(
                "Jiuwen KVBM published vLLM block %s but failed to register local "
                "management metadata; using relay-only NPU location event: %s",
                block_id,
                exc,
            )
            store_events.append(
                {
                    "event_type": "store",
                    "location": "npu",
                    "block_hashes": [block_hash],
                    "token_count": control.config.block_size,
                    "engine_specific": {
                        "to": "npu",
                        "physical_block_id": block_id,
                    },
                }
            )
    return store_events


def _pull_leased_blocks_and_publish(
    control: "VllmAscend018DataPlaneControl",
    backend: LLMDataDistBackend,
    payload: dict[str, Any],
    *,
    source_blocks: list[int],
    hash_descriptors: list[Any],
    source_lease: "_RemoteSourceLeaseGuard",
    transfer_id: str,
) -> dict[str, Any]:
    """把已 pin 的远端源 block 拉取并发布到本地 prefix cache。

    参数:
        control: 当前 engine-core worker 的 KVBM 数据面控制对象。
        backend: 已初始化的 LLM-DataDist backend。
        payload: 原始事务请求体。
        source_blocks: 已由远端 lease 保护的源物理 block ID。
        hash_descriptors: lease 获取时原子读取的完整 block hash 描述。
        source_lease: 正在续租的远端源 block lease guard。
        transfer_id: 当前幂等传输事务 ID。

    返回:
        成功时返回发布结果；拉取、续租或发布失败时返回回滚诊断。
    """

    cache_id = _ENGINE_CORE_LLM_DATADIST_CACHE_ID
    if cache_id is None:
        raise ValueError("local LLM-DataDist cache is not registered")
    total_started = time.perf_counter()
    phase_ms: dict[str, float] = {}
    _raise_if_transfer_deadline_expired(payload)
    packed_hashes = [_decode_packed_block_hash(value) for value in hash_descriptors]
    external_hashes = [_external_block_hash_string(block_hash[:-4]) for block_hash in packed_hashes]

    reserve_started = time.perf_counter()
    reserve_response = _dispatch_block_pool_request(
        "/datadist/reservations/reserve",
        {
            "count": len(source_blocks),
            "ttl_ms": int(payload.get("reservation_ttl_ms", 120_000)),
        },
    )
    phase_ms["reserve"] = (time.perf_counter() - reserve_started) * 1000.0
    if not reserve_response.get("ok"):
        phase_ms["worker_dataplane_total"] = (time.perf_counter() - total_started) * 1000.0
        reserve_response["phase_ms"] = phase_ms
        return reserve_response
    reservation = reserve_response["reservation"]
    reservation_id = str(reservation["reservation_id"])
    destination_blocks = [int(value) for value in reservation["block_ids"]]
    transactions = _process_transfer_transactions()
    transactions.update(
        transfer_id,
        TransferTransactionState.RESERVED,
        reservation_id=reservation_id,
        destination_blocks=destination_blocks,
    )
    remote_cluster_id = int(payload["remote_cluster_id"])
    publish_response: dict[str, Any] | None = None
    reservation_guard: _TargetBlockReservationGuard | None = None
    try:
        reservation_guard = _TargetBlockReservationGuard(
            reservation_id,
            ttl_ms=int(payload.get("reservation_ttl_ms", 120_000)),
        )
        reservation_guard.start()
        transactions.raise_if_cancelled(transfer_id)
        transactions.update(transfer_id, TransferTransactionState.TRANSFERRING)
        hixl_pull_started = time.perf_counter()
        backend.pull_blocks(
            cache_id,
            LLMDataDistBlockMapping(
                cache_key=LLMDataDistCacheKey(
                    key=str(payload.get("cache_key", remote_cluster_id)),
                    prompt_cluster_id=remote_cluster_id,
                    prompt_batch_index=int(payload.get("model_id", 0)),
                ),
                prompt_blocks=tuple(source_blocks),
                decoder_blocks=tuple(destination_blocks),
                layer_range=_optional_layer_range(payload),
            ),
        )
        phase_ms["hixl_pull"] = (time.perf_counter() - hixl_pull_started) * 1000.0
        _raise_if_transfer_deadline_expired(payload)
        transactions.raise_if_cancelled(transfer_id)
        lease_renew_started = time.perf_counter()
        source_lease.renew_now()
        phase_ms["lease_renew"] = (time.perf_counter() - lease_renew_started) * 1000.0
        transactions.raise_if_cancelled(transfer_id)
        reservation_renew_started = time.perf_counter()
        reservation_guard.prepare_publish()
        phase_ms["reservation_renew"] = (time.perf_counter() - reservation_renew_started) * 1000.0
        _raise_if_transfer_deadline_expired(payload)
        transactions.update(transfer_id, TransferTransactionState.PUBLISHING)
        publish_started = time.perf_counter()
        publish_response = _dispatch_block_pool_request(
            "/datadist/reservations/publish",
            {
                "reservation_id": reservation_id,
                "block_hashes": list(hash_descriptors),
            },
        )
        phase_ms["publish"] = (time.perf_counter() - publish_started) * 1000.0
        if not publish_response.get("ok"):
            raise RuntimeError(f"failed to publish destination blocks: {publish_response.get('error')}")
        reservation_guard.close()
        if publish_response.get("already_published"):
            logger.info(
                "Jiuwen KVBM discarded redundant HIXL pull because destination hash "
                "was already published: source_blocks=%s reserved_blocks=%s "
                "existing_block_ids=%s",
                source_blocks,
                destination_blocks,
                publish_response.get("existing_block_ids", []),
            )
            phase_ms["worker_dataplane_total"] = (time.perf_counter() - total_started) * 1000.0
            return {
                "ok": True,
                "skipped": True,
                "reason": "target_already_published",
                "already_published": True,
                "cache_id": cache_id,
                "remote_cluster_id": remote_cluster_id,
                "source_blocks": source_blocks,
                "destination_blocks": [],
                "discarded_destination_blocks": destination_blocks,
                "existing_block_ids": publish_response.get(
                    "existing_block_ids",
                    [],
                ),
                "published_count": 0,
                "reservation": publish_response["reservation"],
                "phase_ms": phase_ms,
            }
    except Exception as exc:
        if reservation_guard is not None:
            reservation_guard.close()
        try:
            transactions.update(transfer_id, TransferTransactionState.ROLLING_BACK)
        except TransferTransactionError:
            logger.debug(
                "Jiuwen KVBM transfer transaction could not enter rolling_back: transfer_id=%s",
                transfer_id,
                exc_info=True,
            )
        rollback_started = time.perf_counter()
        rollback_response = _rollback_block_reservation(reservation_id)
        phase_ms["rollback"] = (time.perf_counter() - rollback_started) * 1000.0
        phase_ms["worker_dataplane_total"] = (time.perf_counter() - total_started) * 1000.0
        logger.warning(
            "Jiuwen KVBM LLM-DataDist pull-and-publish failed; reservation_id=%s rollback=%s error=%s",
            reservation_id,
            rollback_response,
            exc,
        )
        return {
            "ok": False,
            "error": str(exc),
            "reservation_id": reservation_id,
            "source_blocks": source_blocks,
            "destination_blocks": destination_blocks,
            "publish": publish_response,
            "rollback": rollback_response,
            "cancelled": isinstance(exc, TransferCancelledError),
            "phase_ms": phase_ms,
        }

    metadata_started = time.perf_counter()
    metadata_task = _PublishedMetadataTask(
        transfer_id=transfer_id,
        control=control,
        block_ids=tuple(destination_blocks),
        packed_hashes=tuple(block_hash.hex() for block_hash in packed_hashes),
        external_hashes=tuple(external_hashes),
    )
    metadata_submit = _async_metadata_publisher().submit(metadata_task)
    if not metadata_submit.get("queued"):
        # 有界队列满或发布器关闭时同步退化，避免成功 publish 的 block 永久
        # 缺失 KVBM location 元数据。该路径只在持续过载或退出阶段触发。
        logger.warning(
            "Jiuwen KVBM async metadata queue unavailable; publishing inline: transfer_id=%s reason=%s",
            transfer_id,
            metadata_submit.get("reason"),
        )
        _publish_published_block_metadata(metadata_task)
    phase_ms["metadata_register"] = 0.0
    phase_ms["metadata_event_enqueue"] = 0.0
    phase_ms["metadata_publish"] = (time.perf_counter() - metadata_started) * 1000.0
    phase_ms["worker_dataplane_total"] = (time.perf_counter() - total_started) * 1000.0
    logger.info(
        "Jiuwen KVBM LLM-DataDist pulled and published %d blocks from cluster=%s "
        "into cache_id=%s destination_blocks=%s phase_ms=%s",
        len(source_blocks),
        remote_cluster_id,
        cache_id,
        destination_blocks,
        phase_ms,
    )
    return {
        "ok": True,
        "cache_id": cache_id,
        "remote_cluster_id": remote_cluster_id,
        "source_blocks": source_blocks,
        "destination_blocks": destination_blocks,
        "published_count": len(destination_blocks),
        "metadata_publish": {
            "mode": "async" if metadata_submit.get("queued") else "inline_fallback",
            **metadata_submit,
        },
        "reservation": publish_response["reservation"],
        "phase_ms": phase_ms,
    }


def _raise_if_transfer_deadline_expired(payload: dict[str, Any]) -> None:
    """在目标 BlockPool publish 前阻止迟到的跨 Worker 传输。

    参数:
        payload: Router 事务请求体；可包含毫秒时间戳 deadline_epoch_ms。

    返回:
        未配置或尚未到期时无返回值；过期时抛出 TransferCancelledError，
        上层事务会回滚目标 reservation。
    """

    raw_deadline = payload.get("deadline_epoch_ms")
    if raw_deadline is None:
        return
    try:
        deadline_epoch_ms = int(raw_deadline)
    except (TypeError, ValueError) as exc:
        raise ValueError("deadline_epoch_ms must be an integer") from exc
    if int(time.time() * 1000) >= deadline_epoch_ms:
        raise TransferCancelledError("cross-worker transfer deadline expired")


def _validate_expected_source_hashes(
    payload: dict[str, Any],
    actual_hashes: list[Any] | tuple[Any, ...],
) -> None:
    """校验调用方可选的预期源 block hash。

    参数:
        payload: 可能包含 block_hashes 的事务请求体。
        actual_hashes: source lease 原子读取到的真实完整 hash。

    返回:
        校验成功或调用方未提供预期值时无返回值；不一致时抛出 ValueError。
    """

    expected_hashes = payload.get("block_hashes")
    if expected_hashes is None:
        return
    if not isinstance(expected_hashes, list | tuple) or len(expected_hashes) != len(actual_hashes):
        raise ValueError("block_hashes must match source_blocks count")
    expected = [_decode_packed_block_hash(value) for value in expected_hashes]
    actual = [_decode_packed_block_hash(value) for value in actual_hashes]
    if expected != actual:
        raise ValueError("source block hashes changed before lease acquisition")


def _validate_expected_external_source_hashes(
    payload: dict[str, Any],
    actual_hashes: list[Any] | tuple[Any, ...],
) -> None:
    """校验 Router 观察到的外部 hash 与源 lease 原子读取结果一致。

    参数:
        payload: 可能包含 external_block_hashes 的事务请求体。
        actual_hashes: source lease 原子读取到的 vLLM 完整 block hash 描述。

    返回:
        校验成功或调用方未提供预期值时无返回值；不一致时抛出 ValueError。
    """

    expected_hashes = payload.get("external_block_hashes")
    if expected_hashes is None:
        return
    if not isinstance(expected_hashes, list | tuple) or len(expected_hashes) != len(actual_hashes):
        raise ValueError("external_block_hashes must match source_blocks count")
    actual = [_external_block_hash_string(_decode_packed_block_hash(value)[:-4]) for value in actual_hashes]
    if [str(value) for value in expected_hashes] != actual:
        raise ValueError("source external block hashes changed before lease acquisition")


class _RemoteSourceLeaseGuard:
    """目标 worker 持有并自动续租的远端源 block lease。"""

    def __init__(
        self,
        source_control_url: str,
        block_ids: list[int],
        *,
        ttl_ms: int,
    ) -> None:
        """创建远端 lease guard。

        参数:
            source_control_url: 源 worker 的 KVBM HTTP 控制服务根地址。
            block_ids: 需要 pin 的源端物理 block ID。
            ttl_ms: 每次 acquire/renew 使用的租期，单位毫秒。

        返回:
            无返回值。
        """

        self.source_control_url = _normalize_control_url(source_control_url)
        self.block_ids = list(block_ids)
        self.ttl_ms = int(ttl_ms)
        if self.ttl_ms < 30_000 or self.ttl_ms > 600_000:
            raise ValueError("source_lease_ttl_ms must be between 30000 and 600000")
        self.lease_id: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lost_error: str | None = None
        self._request_lock = threading.Lock()

    def acquire(self) -> dict[str, Any]:
        """远程获取 lease 并启动续租心跳。

        返回:
            源端 acquire 接口返回的 lease 快照。
        """

        response = _post_remote_control(
            self.source_control_url,
            "/datadist/source-leases/acquire",
            {"block_ids": self.block_ids, "ttl_ms": self.ttl_ms},
        )
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "source lease acquire failed")))
        lease = response.get("lease")
        if not isinstance(lease, dict) or not lease.get("lease_id"):
            raise RuntimeError("source lease acquire returned no lease_id")
        self.lease_id = str(lease["lease_id"])
        self._thread = threading.Thread(
            target=self._heartbeat,
            name=f"jiuwen-kvbm-source-lease-{self.lease_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        return lease

    def ensure_valid(self) -> None:
        """确认续租期间未观察到 lease 丢失。

        返回:
            lease 仍有效时无返回值；续租失败时抛出 RuntimeError。
        """

        if self._lost_error is not None:
            raise RuntimeError(f"source block lease lost during transfer: {self._lost_error}")

    def renew_now(self) -> None:
        """在目标 publish 前同步续租并确认 lease 仍然有效。

        返回:
            续租成功时无返回值；源端报告 lease 丢失时抛出 RuntimeError。
        """

        response = self._renew()
        if not response.get("ok"):
            self._lost_error = str(response.get("error", "source lease renew failed"))
            raise RuntimeError(f"source block lease lost during transfer: {self._lost_error}")
        self._lost_error = None

    def release(self) -> dict[str, Any]:
        """停止心跳并尽力释放远端 lease。

        返回:
            源端 release 响应；网络失败时返回 ok=False 诊断。
        """

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.lease_id is None:
            return {"ok": False, "error": "source lease was not acquired"}
        with self._request_lock:
            return _post_remote_control(
                self.source_control_url,
                "/datadist/source-leases/release",
                {"lease_id": self.lease_id},
            )

    def _heartbeat(self) -> None:
        """按 TTL 的三分之一周期续租，失败时阻止目标 publish。"""

        interval_seconds = max(0.25, self.ttl_ms / 3000.0)
        while not self._stop.wait(interval_seconds):
            response = self._renew()
            if not response.get("ok"):
                self._lost_error = str(response.get("error", "source lease renew failed"))
                return

    def _renew(self) -> dict[str, Any]:
        """串行发送一次远端续租请求。"""

        with self._request_lock:
            return _post_remote_control(
                self.source_control_url,
                "/datadist/source-leases/renew",
                {"lease_id": self.lease_id, "ttl_ms": self.ttl_ms},
            )


class _AdoptedRemoteSourceLeaseGuard:
    """续租 P `/pd/export` 已获取的 source lease，但不负责最终释放。"""

    def __init__(
        self,
        source_control_url: str,
        lease_id: str,
        *,
        ttl_ms: int,
    ) -> None:
        """创建已存在 source lease 的续租 guard。

        参数:
            source_control_url: P Worker KVBM 控制服务根地址。
            lease_id: P `/pd/export` 返回的 source_lease_id。
            ttl_ms: 每次续租后的有效期，单位毫秒。

        返回:
            无返回值。
        """

        self.source_control_url = _normalize_control_url(source_control_url)
        self.lease_id = str(lease_id).strip()
        self.ttl_ms = int(ttl_ms)
        if not self.lease_id:
            raise ValueError("source_lease_id must not be empty")
        if self.ttl_ms < 30_000 or self.ttl_ms > 600_000:
            raise ValueError("source_lease_ttl_ms must be between 30000 and 600000")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._request_lock = threading.Lock()
        self._lost_error: str | None = None

    def start(self) -> None:
        """启动后台续租线程。

        返回:
            无返回值；重复调用不会创建第二个线程。
        """

        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._heartbeat,
            name=f"jiuwen-kvbm-pd-source-lease-{self.lease_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def renew_now(self) -> None:
        """同步续租并确认 P source lease 仍有效。

        返回:
            续租成功时无返回值；lease 丢失或网络失败时抛出 RuntimeError。
        """

        response = self._renew()
        if not response.get("ok"):
            self._lost_error = str(response.get("error", "source lease renew failed"))
            raise RuntimeError(f"P source lease lost: {self._lost_error}")
        self._lost_error = None

    def close(self) -> None:
        """停止后台续租，但保留 P source lease 等待 decode ACK 释放。

        返回:
            无返回值；该操作可重复调用。
        """

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _heartbeat(self) -> None:
        """按 TTL 三分之一周期续租，失败后停止。"""

        interval_seconds = max(0.25, self.ttl_ms / 3000.0)
        while not self._stop.wait(interval_seconds):
            response = self._renew()
            if not response.get("ok"):
                self._lost_error = str(response.get("error", "source lease renew failed"))
                return

    def _renew(self) -> dict[str, Any]:
        """串行调用 P source lease renew 接口并返回响应。"""

        with self._request_lock:
            return _post_remote_control(
                self.source_control_url,
                "/datadist/source-leases/renew",
                {"lease_id": self.lease_id, "ttl_ms": self.ttl_ms},
            )


class _TargetBlockReservationGuard:
    """目标 NPU worker 持有并自动续租的 scheduler block reservation。"""

    def __init__(self, reservation_id: str, *, ttl_ms: int) -> None:
        """创建目标 reservation guard。

        参数:
            reservation_id: scheduler reserve 返回的 reservation ID。
            ttl_ms: 每次续租使用的有效期，单位毫秒。

        返回:
            无返回值。
        """

        self.reservation_id = str(reservation_id)
        self.ttl_ms = int(ttl_ms)
        if self.ttl_ms < 10_000 or self.ttl_ms > 1_800_000:
            raise ValueError("reservation_ttl_ms must be between 10000 and 1800000")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lost_error: str | None = None
        self._request_lock = threading.Lock()

    def start(self) -> None:
        """启动目标 reservation 续租心跳。

        返回:
            无返回值。
        """

        self._thread = threading.Thread(
            target=self._heartbeat,
            name=f"jiuwen-kvbm-reservation-{self.reservation_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def prepare_publish(self) -> None:
        """停止后台心跳，并在 publish 前完成最后一次同步续租。

        返回:
            reservation 仍有效时无返回值。

        异常:
            RuntimeError: scheduler 已回收或无法续租 reservation。
        """

        self.close()
        response = self._renew()
        if not response.get("ok"):
            self._lost_error = str(response.get("error", "destination reservation renew failed"))
            raise RuntimeError(f"destination reservation lost during transfer: {self._lost_error}")
        self._lost_error = None

    def close(self) -> None:
        """停止 reservation 心跳。

        返回:
            无返回值；该操作可重复调用。
        """

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _heartbeat(self) -> None:
        """按 TTL 的三分之一周期续租，失败后记录错误并停止。

        返回:
            无返回值。
        """

        interval_seconds = max(0.25, self.ttl_ms / 3000.0)
        while not self._stop.wait(interval_seconds):
            response = self._renew()
            if not response.get("ok"):
                self._lost_error = str(response.get("error", "destination reservation renew failed"))
                return

    def _renew(self) -> dict[str, Any]:
        """串行发送一次 scheduler reservation 续租请求。

        返回:
            scheduler reservation 服务返回的响应字典。
        """

        with self._request_lock:
            return _dispatch_block_pool_request(
                "/datadist/reservations/renew",
                {
                    "reservation_id": self.reservation_id,
                    "ttl_ms": self.ttl_ms,
                },
            )


def _normalize_control_url(value: str) -> str:
    """校验并规范化 worker KVBM 控制服务 URL。

    参数:
        value: 调用方提供的 source_control_url。

    返回:
        去除末尾斜杠的 HTTP(S) URL。
    """

    normalized = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_control_url must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("source_control_url must not contain query or fragment")
    return normalized


def _post_remote_control(
    base_url: str,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """向远端 worker KVBM 控制服务发送 JSON POST。

    参数:
        base_url: 已规范化的远端控制服务根地址。
        path: 以斜杠开头的控制 API 路径。
        payload: JSON 请求体。

    返回:
        远端 JSON 响应；HTTP 或网络错误转换为 ok=False 字典。
    """

    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": f"source control HTTP {exc.code}: {body}",
            }
    except Exception as exc:
        return {"ok": False, "error": f"source control request failed: {exc}"}


def _rollback_block_reservation(reservation_id: str) -> dict[str, Any]:
    """尽力回滚一次尚未发布的目标 block reservation。

    参数:
        reservation_id: reserve 返回的 reservation 唯一 ID。

    返回:
        rollback 控制结果；回滚请求本身抛错时转换为 ok=False 诊断字典。
    """

    try:
        return _dispatch_block_pool_request(
            "/datadist/reservations/rollback",
            {"reservation_id": reservation_id},
        )
    except Exception as exc:
        return {"ok": False, "error": f"reservation rollback failed: {exc}"}


def _optional_layer_range(payload: dict[str, Any]) -> tuple[int, int] | None:
    """读取可选的 LLM-DataDist layer_range。

    参数:
        payload: 可能包含 `[start, stop]` 格式 layer_range 的请求体。

    返回:
        未提供时返回 None；提供时返回两个整数构成的半开区间。
    """

    raw_layer_range = payload.get("layer_range")
    if raw_layer_range is None:
        return None
    if not isinstance(raw_layer_range, list | tuple) or len(raw_layer_range) != 2:
        raise ValueError("layer_range must contain [start, stop]")
    return int(raw_layer_range[0]), int(raw_layer_range[1])


def _datadist_cluster_from_payload(payload: dict[str, Any]) -> LLMDataDistCluster:
    """从 HTTP JSON 请求构造远端 cluster 描述。

    参数:
        payload: 包含 remote_cluster_id、remote_ip、remote_port、local_ip 和
            local_port 的请求字典。

    返回:
        校验完成的 LLMDataDistCluster。
    """

    required = ("remote_cluster_id", "remote_ip", "remote_port", "local_ip", "local_port")
    missing = [name for name in required if payload.get(name) is None]
    if missing:
        raise ValueError(f"missing LLM-DataDist cluster fields: {missing}")
    return LLMDataDistCluster(
        remote_cluster_id=int(payload["remote_cluster_id"]),
        remote_ip=str(payload["remote_ip"]),
        remote_port=int(payload["remote_port"]),
        local_ip=str(payload["local_ip"]),
        local_port=int(payload["local_port"]),
    )


def _required_int_list(payload: dict[str, Any], name: str) -> list[int]:
    """读取非空整数列表请求字段。

    参数:
        payload: HTTP JSON 请求字典。
        name: 字段名称。

    返回:
        转换后的整数列表。
    """

    value = payload.get(name)
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return [int(item) for item in value]


class _EngineCorePrefetchHandler(BaseHTTPRequestHandler):
    """engine-core 本地 KVBM 预取 HTTP handler。"""

    protocol_version = "HTTP/1.1"

    # BaseHTTPRequestHandler dispatches by this exact protocol method name.
    # pylint: disable=invalid-name
    def do_GET(self) -> None:
        """建立Router到Worker的PD控制事件长连接。

        参数:
            无显式参数；handoff_id和after_sequence来自URL查询参数。

        返回:
            无返回值；持续写出NDJSON事件，客户端断开后注销订阅。
        """

        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/datadist/pd/events":
            self._write_json({"ok": False, "error": "not found"}, status=404)
            return
        query = urllib.parse.parse_qs(parsed.query)
        handoff_id = str(query.get("handoff_id", [""])[0]).strip()
        if not handoff_id:
            self._write_json(
                {"ok": False, "error": "handoff_id is required"},
                status=400,
            )
            return
        try:
            after_sequence = max(
                0,
                int(query.get("after_sequence", ["0"])[0]),
            )
        except (TypeError, ValueError):
            self._write_json(
                {"ok": False, "error": "after_sequence must be an integer"},
                status=400,
            )
            return
        subscriber, history = _PD_CONTROL_EVENTS.subscribe(
            handoff_id=handoff_id,
            after_sequence=after_sequence,
        )
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.flush()
            for event in history:
                self._write_pd_event(event)
            while True:
                try:
                    event = subscriber.get(timeout=10.0)
                except queue.Empty:
                    event = {
                        "sequence": 0,
                        "event_type": "HEARTBEAT",
                        "handoff_id": handoff_id,
                        "request_id": "",
                        "created_at_ms": int(time.time() * 1000),
                    }
                if event.get("handoff_id") == handoff_id:
                    self._write_pd_event(event)
        except ConnectionError:
            logger.debug(
                "Jiuwen KVBM PD event stream disconnected: handoff_id=%s",
                handoff_id,
            )
        finally:
            _PD_CONTROL_EVENTS.unsubscribe(
                handoff_id=handoff_id,
                subscriber=subscriber,
            )

    # pylint: enable=invalid-name

    def _write_pd_event(self, event: dict[str, Any]) -> None:
        """向当前长连接写入一个NDJSON事件。

        参数:
            event: 已包含sequence、event_type和handoff_id的事件字典。

        返回:
            无返回值；写入失败时由调用方处理连接异常。
        """

        payload = json.dumps(event, default=str).encode("utf-8") + b"\n"
        self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    # BaseHTTPRequestHandler dispatches by this exact protocol method name.
    # pylint: disable=invalid-name
    def do_POST(self) -> None:
        """处理 POST /prefetch 请求。

        参数:
            无显式参数；请求信息来自 BaseHTTPRequestHandler 实例。

        返回:
            无返回值；通过 HTTP response 写回 JSON。
        """

        handler_started = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
        except Exception as exc:
            self._write_json(
                {"ok": False, "error": f"failed to read request body: {exc}"},
                status=400,
            )
            return
        if self.path == "/events/drain":
            self._write_json(
                {"ok": True, "events": _drain_runtime_event_outbox()},
                status=200,
            )
            return
        datadist_paths = {
            "/datadist/status",
            "/datadist/digest",
            "/datadist/blocks/metadata",
            "/datadist/reservations/status",
            "/datadist/reservations/reserve",
            "/datadist/reservations/renew",
            "/datadist/reservations/publish",
            "/datadist/reservations/commit",
            "/datadist/reservations/rollback",
            "/datadist/reservations/release",
            "/datadist/source-leases/status",
            "/datadist/source-leases/acquire",
            "/datadist/source-leases/renew",
            "/datadist/source-leases/release",
            "/datadist/link",
            "/datadist/unlink",
            "/datadist/pull",
            "/datadist/transfers/pull",
            "/datadist/transfers/status",
            "/datadist/transfers/cancel",
            "/datadist/pd/handoffs/begin",
            "/datadist/pd/handoffs/advance",
            "/datadist/pd/handoffs/status",
            "/datadist/pd/handoffs/rollback",
            "/datadist/pd/export",
            "/datadist/pd/export/release",
            "/datadist/pd/import/prepare",
            "/datadist/pd/import",
            "/datadist/pd/import/submit",
            "/datadist/pd/import/result",
            "/datadist/pd/import/status",
            "/datadist/pd/import/release",
            "/datadist/block-pool/capacity",
            "/datadist/benchmark/pull",
        }
        if self.path != "/prefetch" and self.path not in datadist_paths:
            self._write_json({"ok": False, "error": "not found"}, status=404)
            return
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
            handler_request_read_ms = (time.perf_counter() - handler_started) * 1000.0
            handler_execution_started = time.perf_counter()
            if self.path in datadist_paths:
                response = _handle_engine_core_datadist_request(self.path, payload)
            else:
                response = _handle_engine_core_prefetch_request(payload)
            handler_execution_ms = (time.perf_counter() - handler_execution_started) * 1000.0
            if self.path == "/datadist/block-pool/capacity":
                timing = response.setdefault("control_timing", {})
                timing.update(
                    handler_request_read_ms=handler_request_read_ms,
                    handler_execution_ms=handler_execution_ms,
                )
                logger.info(
                    "[KVBMCapacityControlTiming] request=%s handoff_id=%s ok=%s control_timing=%s",
                    payload.get("request_id", ""),
                    payload.get("handoff_id", ""),
                    bool(response.get("ok")),
                    timing,
                )
            if self.path == "/datadist/pd/export":
                export = response.get("export")
                if isinstance(export, dict):
                    control_timing = export.setdefault("control_timing", {})
                    control_timing.update(
                        {
                            "handler_request_read_ms": handler_request_read_ms,
                            "handler_execution_ms": handler_execution_ms,
                        }
                    )
                    logger.info(
                        "[KVBMPDExportControlTiming] handoff_id=%s request_id=%s "
                        "handler_request_read_ms=%.3f handler_execution_ms=%.3f "
                        "engine_core_proxy_ms=%.3f engine_core_queue_ms=%.3f "
                        "engine_core_execution_ms=%.3f response_build_ms=%.3f",
                        payload.get("handoff_id"),
                        payload.get("request_id"),
                        handler_request_read_ms,
                        handler_execution_ms,
                        float(control_timing.get("engine_core_proxy_ms", 0.0)),
                        float(control_timing.get("engine_core_queue_ms", 0.0)),
                        float(control_timing.get("engine_core_execution_ms", 0.0)),
                        float(control_timing.get("response_build_ms", 0.0)),
                    )
            status = 200 if response.get("ok") else 503
            if response.get("pending") and response.get("retryable"):
                status = 202
            self._write_json(response, status=status)
        except ConnectionError:
            logger.debug("Jiuwen KVBM engine-core prefetch response dropped because client disconnected")
        except Exception as exc:
            logger.exception("Jiuwen KVBM engine-core prefetch request failed")
            try:
                self._write_json({"ok": False, "error": str(exc)}, status=500)
            except ConnectionError:
                logger.debug("Jiuwen KVBM engine-core prefetch error response dropped because client disconnected")

    # pylint: enable=invalid-name

    def log_message(self, message_format: str, *args: Any) -> None:
        """将 BaseHTTPRequestHandler 访问日志转到模块 logger。

        参数:
            message_format: 原始日志格式字符串。
            args: 原始日志格式化参数。

        返回:
            无返回值。
        """

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Jiuwen KVBM prefetch control: %s", message_format % args)

    def _write_json(self, payload: dict[str, Any], *, status: int) -> None:
        """写出 JSON 响应。

        参数:
            payload: 响应体字典。
            status: HTTP 状态码。

        返回:
            无返回值。
        """

        body = json.dumps(payload, default=str).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            logger.debug("Jiuwen KVBM prefetch response write skipped because client disconnected")


@dataclass
class VllmAscend018DataPlaneControl:
    """绑定在 vLLM-Ascend model runner 上的 KVBM 数据面控制对象。

    属性:
        provider: 已绑定真实 kv_caches 的 vLLM-Ascend 0.18 provider。
        config: worker KVBM runtime 配置。
        llm_datadist_backend: 可选 LLM-DataDist backend。
    """

    provider: VllmAscend018TensorProvider
    config: WorkerKVBMConfig
    llm_datadist_backend: LLMDataDistBackend | None = None

    def __post_init__(self) -> None:
        """初始化 WorkerKVBMRuntime。

        返回:
            无返回值。
        """

        self.physical_block_resolver: VllmAscend018PhysicalBlockResolver | None = None
        self.block_reservations = VllmBlockPoolReservationManager(
            _latest_reservable_block_pool,
            lock=_VLLM_BLOCK_POOL_LOCK,
        )
        self.runtime = WorkerKVBMRuntime(
            self.config,
            accessor=VllmAscendKVAccessor(self.provider),
            llm_datadist_backend=self.llm_datadist_backend,
        )
        self._prefetch_state_lock = threading.RLock()
        self._prefetch_inflight: dict[str, str] = {}
        self._kv_event_runtime_diagnostics: dict[str, float | int] = {}

    def set_physical_block_resolver(self, resolver: "VllmAscend018PhysicalBlockResolver | None") -> None:
        """设置逻辑 block 到物理 block 的解析器。

        参数:
            resolver: 绑定在 engine-core 对象树上的物理 block 解析器；传入 None
                表示禁用自动补齐。

        返回:
            无返回值。
        """

        self.physical_block_resolver = resolver

    def register_physical_block(
        self,
        block_id: int,
        block_hash: str | None = None,
        *,
        token_count: int | None = None,
        active: bool = False,
    ) -> Any:
        """按物理 block_id 注册真实 NPU KV block。

        参数:
            block_id: vLLM-Ascend 物理 KV block 编号。
            block_hash: 可选稳定 hash；为空时使用 synthetic hash。
            token_count: 该 block 的 token 数；为空时使用配置 block_size。
            active: 是否标记为正在使用。

        返回:
            WorkerKVBMRuntime 产生的 store 事件。
        """

        return self.runtime.register_block(
            block_id,
            block_hash or self._physical_hash(block_id),
            token_count=token_count or self.config.block_size,
            active=active,
        )

    def register_published_physical_blocks(
        self,
        block_ids: list[int] | tuple[int, ...],
        block_hashes: list[str] | tuple[str, ...],
        *,
        token_count: int | None = None,
        active: bool = False,
    ) -> Any:
        """批量注册刚发布到 vLLM BlockPool 的真实 NPU KV block。

        参数:
            block_ids: 目标 Worker 内按前缀顺序排列的物理 block ID。
            block_hashes: 与 block_ids 一一对应的稳定 block hash。
            token_count: 每个 block 覆盖的 token 数；为空时使用配置 block_size。
            active: 是否将全部 block 标记为正在使用。

        返回:
            WorkerKVBMRuntime 产生的一条聚合 store 事件。

        异常:
            ValueError: 两个列表为空或长度不一致时抛出。
        """

        if not block_ids or len(block_ids) != len(block_hashes):
            raise ValueError("block_ids and block_hashes must be non-empty and aligned")
        return self.runtime.register_blocks(
            list(zip(block_ids, block_hashes, strict=True)),
            token_count=token_count or self.config.block_size,
            active=active,
        )

    def register_physical_blocks(self, *, limit: int | None = None) -> int:
        """批量注册从 0 开始的真实物理 KV block。

        参数:
            limit: 最多注册的 block 数；为空时注册 provider 诊断出的全部 block。

        返回:
            实际注册的 block 数量。
        """

        diagnostics = self.provider.diagnostics(0)
        num_blocks = int(diagnostics["num_blocks"])
        count = num_blocks if limit is None else min(limit, num_blocks)
        for block_id in range(count):
            if self._is_registered(block_id):
                continue
            self.register_physical_block(block_id)
        return count

    def offload_block_to_ram(self, block_id: int, block_hash: str | None = None) -> Any:
        """复制指定真实 NPU block 到 RAM 后端。

        参数:
            block_id: vLLM-Ascend 物理 KV block 编号。
            block_hash: 可选稳定 hash；未注册时用于注册。

        返回:
            WorkerKVBMRuntime 产生的 move 事件。
        """

        self._ensure_registered(block_id, block_hash)
        block = self.runtime.manager.get_metadata(block_id)
        return self.runtime.offload_to_ram(block)

    def offload_block_to_disk(self, block_id: int, block_hash: str | None = None) -> Any:
        """复制指定真实 block 到磁盘后端。

        参数:
            block_id: vLLM-Ascend 物理 KV block 编号。
            block_hash: 可选稳定 hash；未注册时用于注册。

        返回:
            WorkerKVBMRuntime 产生的 move 事件。
        """

        self._ensure_registered(block_id, block_hash)
        block = self.runtime.manager.get_metadata(block_id)
        return self.runtime.offload_to_disk(block)

    def prefetch_block(self, block_hash: str) -> Any | None:
        """将指定 hash 的 block 从 RAM/磁盘预取回 NPU。

        参数:
            block_hash: 已注册 block 的 hash。

        返回:
            命中时返回 move/hit 事件；未命中时返回 None。
        """

        return self.runtime.prefetch_block(block_hash)

    def prefetch_block_hashes(self, block_hashes: list[str] | tuple[str, ...]) -> list[Any]:
        """批量将指定 hash 的 block 从 RAM/磁盘预取回 NPU。

        参数:
            block_hashes: 需要预取的 Block hash 列表。

        返回:
            命中并成功预取的 move/hit 事件列表；未命中的 hash 不产生事件。
        """

        result = self.prefetch_block_hashes_for_request(
            block_hashes,
            request_id="legacy-prefetch",
            deadline_monotonic=float("inf"),
        )
        return result["events"]

    def prefetch_block_hashes_for_request(
        self,
        block_hashes: list[str] | tuple[str, ...],
        *,
        request_id: str,
        deadline_monotonic: float,
    ) -> dict[str, Any]:
        """按请求级 deadline 执行可去重、可回滚的批量预取。

        参数:
            block_hashes: 需要从 RAM/DISK 预取的逻辑 Block hash。
            request_id: Router 请求 ID，用作 in-flight 所有者和日志关联。
            deadline_monotonic: 本进程 monotonic 截止时间；每个 block 写入前和
                最终提交前都会检查。

        返回:
            包含 events、prefetched_count、already_ready_count、
            deduplicated_count、expired_count、rolled_back_count 和
            missing_count 的结果字典。events 仅供进程内 location event relay。

        说明:
            同一 hash 同时只允许一个请求执行真实搬运。RAM/DISK 源副本在最终
            检查通过前保持不变；超时、目标物理 block 被复用或相同 hash 已由
            推理请求发布时，预取结果不会更新为 NPU location。
        """

        owner = request_id or (f"anonymous-{threading.get_ident()}-{time.monotonic_ns()}")
        unique_hashes = list(dict.fromkeys(str(value) for value in block_hashes))
        claimed: list[str] = []
        deduplicated_count = len(block_hashes) - len(unique_hashes)
        with self._prefetch_state_lock:
            for block_hash in unique_hashes:
                if block_hash in self._prefetch_inflight:
                    deduplicated_count += 1
                    continue
                self._prefetch_inflight[block_hash] = owner
                claimed.append(block_hash)

        events: list[Any] = []
        prefetched_count = 0
        already_ready_count = 0
        expired_count = 0
        rolled_back_count = 0
        missing_count = 0
        outcomes: dict[str, str] = {}

        def _make_commit_check(
            expected_hash: str,
            rejection: dict[str, str],
        ) -> Callable[[Any], bool]:
            """为单个 hash 构造 deadline 与重复发布最终检查。"""

            def _should_commit(block: Any) -> bool:
                if time.monotonic() >= deadline_monotonic:
                    rejection["value"] = "deadline"
                    return False
                if self.runtime.manager.has_npu_block(
                    expected_hash,
                    exclude_block_id=int(block.block_id),
                ):
                    rejection["value"] = "already_published"
                    return False
                return True

            return _should_commit

        try:
            for block_hash in claimed:
                if time.monotonic() >= deadline_monotonic:
                    expired_count += 1
                    outcomes[block_hash] = "expired_before_copy"
                    continue
                if self.runtime.manager.has_npu_block(block_hash):
                    already_ready_count += 1
                    outcomes[block_hash] = "already_ready"
                    continue

                rejection_reason = {"value": ""}
                should_commit = _make_commit_check(block_hash, rejection_reason)

                try:
                    event = self.runtime.prefetch_block(
                        block_hash,
                        should_commit=should_commit,
                    )
                except PrefetchCommitRejectedError as exc:
                    rolled_back_count += 1
                    reason = rejection_reason["value"] or "target_reused"
                    if reason == "deadline":
                        expired_count += 1
                    outcomes[block_hash] = f"rolled_back:{reason}"
                    logger.info(
                        "Jiuwen KVBM prefetch rolled back request_id=%s block_hash=%s reason=%s error=%s",
                        owner,
                        block_hash,
                        reason,
                        exc,
                    )
                    continue

                if event is None:
                    missing_count += 1
                    outcomes[block_hash] = "missing"
                    continue
                events.append(event)
                if _runtime_event_type(event) == "hit":
                    already_ready_count += 1
                    outcomes[block_hash] = "already_ready"
                else:
                    prefetched_count += 1
                    outcomes[block_hash] = "prefetched"
        finally:
            with self._prefetch_state_lock:
                for block_hash in claimed:
                    if self._prefetch_inflight.get(block_hash) == owner:
                        self._prefetch_inflight.pop(block_hash, None)

        source_counts: dict[str, int] = {}
        for event in events:
            source = str(event.engine_specific.get("from", event.location))
            source_counts[source] = source_counts.get(source, 0) + 1
        logger.info(
            "Jiuwen KVBM prefetch request finished request_id=%s "
            "prefetched=%d/%d already_ready=%d deduplicated=%d expired=%d "
            "rolled_back=%d missing=%d sources=%s stats=%s",
            owner,
            prefetched_count,
            len(block_hashes),
            already_ready_count,
            deduplicated_count,
            expired_count,
            rolled_back_count,
            missing_count,
            source_counts,
            self.stats(),
        )
        return {
            "events": events,
            "prefetched_count": prefetched_count,
            "ready_count": prefetched_count + already_ready_count,
            "already_ready_count": already_ready_count,
            "deduplicated_count": deduplicated_count,
            "expired_count": expired_count,
            "rolled_back_count": rolled_back_count,
            "missing_count": missing_count,
            "outcomes": outcomes,
        }

    def apply_kv_events(self, events: list[Any] | tuple[Any, ...]) -> list[Any]:
        """批量应用 vLLM/vLLM-Ascend KV 事件。

        参数:
            events: vLLM KV event 对象或 relay dict 列表。

        返回:
            KVBM runtime 产生的事件列表。
        """

        self._kv_event_runtime_diagnostics = {
            "describe_blocks_ms": 0.0,
            "metadata_update_ms": 0.0,
            "runtime_event_build_ms": 0.0,
            "registered_blocks": 0,
            "unchanged_blocks": 0,
        }
        results = []
        for event in events:
            result = self.apply_kv_event(event)
            if result is None:
                continue
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        return results

    def consume_kv_event_runtime_diagnostics(self) -> dict[str, float | int]:
        """取出并清空当前 KV 事件批次的 runtime 更新诊断。

        返回:
            包含 block 描述、元数据更新、事件构造耗时以及
            实际注册数、幂等命中数的字典。
        """

        diagnostics = dict(self._kv_event_runtime_diagnostics)
        self._kv_event_runtime_diagnostics = {}
        return diagnostics

    def apply_kv_event(self, event: Any) -> Any | list[Any] | None:
        """应用单个 vLLM/vLLM-Ascend KV 事件。

        参数:
            event: vLLM KV event 对象，或形如 {"type": ..., "data": ...} 的
                relay dict。

        返回:
            事件成功映射时返回 KVBM runtime 事件；无法识别、缺少物理 block id、
            或事件类型暂不支持时返回 None。

        说明:
            真实 NPU tensor 读写必须依赖物理 block id。只有 block_hash 而没有
            block_id/block_ids 的事件，只能用于 router 管理面，不能在 worker
            数据面定位真实 KV cache slice。
        """

        resolve_started_ns = time.perf_counter_ns()
        normalized = _normalize_kv_event(event)
        if not normalized["block_ids"] and self.physical_block_resolver is not None:
            enriched = self.physical_block_resolver.enrich_event(
                event,
                refresh_on_miss=False,
            )
            normalized = _normalize_kv_event(enriched)
            if not normalized["block_ids"] and normalized["event_type"] == "store":
                enriched = self.physical_block_resolver.enrich_event(event)
                normalized = _normalize_kv_event(enriched)
            if normalized["block_ids"]:
                logger.debug(
                    "Jiuwen KVBM resolved KV event physical block ids: type=%s block_ids=%s block_hashes=%s",
                    normalized["event_type"],
                    normalized["block_ids"],
                    normalized["block_hashes"],
                )
        if self.physical_block_resolver is not None:
            self.physical_block_resolver.add_resolve_timing_ms(
                (time.perf_counter_ns() - resolve_started_ns) / 1_000_000.0
            )
        event_type = normalized["event_type"]
        block_ids = normalized["block_ids"]
        block_hashes = normalized["block_hashes"]
        if not block_ids:
            logger.debug("Jiuwen KVBM ignored KV event without physical block ids: %s", normalized)
            return None

        if event_type == "store":
            block_specs: list[tuple[int, str]] = []
            token_counts: list[int] = []
            block_tokens_by_index: list[list[int]] = []
            for index, block_id in enumerate(block_ids):
                block_hash = _block_hash_for_index(block_hashes, index, block_id)
                block_tokens = _store_event_tokens_for_index(
                    normalized["token_ids"],
                    index,
                    len(block_ids),
                    self.config.block_size,
                )
                block_specs.append((block_id, block_hash))
                token_counts.append(len(block_tokens) or self.config.block_size)
                block_tokens_by_index.append(block_tokens)

            registration_diagnostics: dict[str, Any] = {}
            self.runtime.register_blocks(
                block_specs,
                token_counts=token_counts,
                active=False,
                diagnostics=registration_diagnostics,
            )
            build_started_ns = time.perf_counter_ns()
            results = []
            for index, (block_id, block_hash) in enumerate(block_specs):
                block_tokens = block_tokens_by_index[index]
                runtime_event = KVBMRuntimeEvent(
                    event_type="store",
                    block_hashes=[block_hash],
                    location="npu",
                    token_count=token_counts[index],
                    engine_specific={
                        "worker_id": self.config.worker_id,
                        "block_size": self.config.block_size,
                        "to": "npu",
                        "medium": "npu",
                        "physical_block_id": block_id,
                    },
                )
                if block_tokens:
                    runtime_event.engine_specific["token_ids"] = block_tokens
                parent_hash = normalized["parent_block_hash"]
                if index > 0 and index - 1 < len(block_hashes):
                    parent_hash = block_hashes[index - 1]
                if parent_hash:
                    runtime_event.engine_specific["parent_block_hash"] = parent_hash
                results.append(runtime_event)
            registration_diagnostics["runtime_event_build_ms"] = (
                time.perf_counter_ns() - build_started_ns
            ) / 1_000_000.0
            for key, value in registration_diagnostics.items():
                self._kv_event_runtime_diagnostics[key] = self._kv_event_runtime_diagnostics.get(key, 0) + value
            return results
        if event_type in {"removed", "remove", "evict", "cleared", "clear"}:
            results = []
            for block_id in block_ids:
                if self._is_registered(block_id):
                    results.append(self.runtime.manager.evict_block(block_id))
            return results
        if event_type in {"move", "transfer"}:
            target = str(normalized["target"]).lower()
            results = []
            for index, block_id in enumerate(block_ids):
                block_hash = _block_hash_for_index(block_hashes, index, block_id)
                if target in {"cpu", "host", "ram", "host_ram"}:
                    results.append(self.offload_block_to_ram(block_id, block_hash))
                elif target in {"disk", "ssd", "nvme", "file", "filesystem", "remote"}:
                    results.append(self.offload_block_to_disk(block_id, block_hash))
                elif target in {"gpu", "npu", "device"}:
                    event = self.prefetch_block(block_hash)
                    if event is not None:
                        results.append(event)
            return results
        if event_type == "hit":
            for block_hash in block_hashes:
                self.runtime.manager.match([block_hash])
            return None
        return None

    def enforce_watermarks(self) -> list[Any]:
        """执行一次 NPU/RAM 水位策略。

        返回:
            本次策略执行产生的事件列表。
        """

        return self.runtime.enforce_watermarks()

    def stats(self) -> dict[str, Any]:
        """返回数据面 runtime 统计。

        返回:
            WorkerKVBMRuntime 统计字典。
        """

        return self.runtime.stats()

    def _ensure_registered(self, block_id: int, block_hash: str | None = None) -> None:
        """确保指定物理 block 已注册。

        参数:
            block_id: vLLM-Ascend 物理 KV block 编号。
            block_hash: 可选稳定 hash。

        返回:
            无返回值。
        """

        if not self._is_registered(block_id):
            self.register_physical_block(block_id, block_hash)

    def _is_registered(self, block_id: int) -> bool:
        """判断 block 是否已注册。

        参数:
            block_id: vLLM-Ascend 物理 KV block 编号。

        返回:
            已注册返回 True，否则返回 False。
        """

        try:
            self.runtime.manager.get_metadata(block_id)
            return True
        except KeyError:
            return False

    @staticmethod
    def _physical_hash(block_id: int) -> str:
        """生成物理 block 的 synthetic hash。

        参数:
            block_id: vLLM-Ascend 物理 KV block 编号。

        返回:
            synthetic hash 字符串。
        """

        return f"physical:{block_id}"


def _normalize_kv_event(event: Any) -> dict[str, Any]:
    """将 vLLM KV event 对象或 dict 标准化为数据面可消费的字典。

    参数:
        event: vLLM KV event 对象，或 relay dict。

    返回:
        包含 event_type、block_ids、block_hashes、target、token_count、
        token_ids 和 parent_block_hash 的字典。
    """

    raw = _event_to_mapping(event)
    event_type = str(raw.get("type") or raw.get("event_type") or raw.get("name") or type(event).__name__)
    data = raw.get("data")
    if isinstance(data, dict):
        raw = {**raw, **data}
    return {
        "event_type": _normalize_event_type(event_type),
        "block_ids": _extract_int_list(raw, ("block_ids", "block_id", "physical_block_ids", "physical_block_id")),
        "block_hashes": [str(value) for value in _extract_list(raw, ("block_hashes", "block_hash", "hashes", "hash"))],
        "raw_block_hashes": _extract_list(raw, ("block_hashes", "block_hash", "hashes", "hash")),
        "target": raw.get("to") or raw.get("destination") or raw.get("medium") or raw.get("location") or "",
        "token_count": _int_or_zero(raw.get("token_count") or raw.get("block_size")),
        "token_ids": _extract_int_list(raw, ("token_ids", "tokens")),
        "parent_block_hash": (str(raw.get("parent_block_hash")) if raw.get("parent_block_hash") is not None else ""),
        "group_idx": _int_or_zero(raw.get("group_idx") if raw.get("group_idx") is not None else raw.get("group_id")),
    }


def _store_event_tokens_for_index(
    token_ids: list[int],
    index: int,
    block_count: int,
    block_size: int,
) -> list[int]:
    """提取一个 store runtime 事件对应的 block token 片段。

    参数:
        token_ids: 原始 vLLM store 事件携带的 token 列表。
        index: 当前物理 block 在事件中的索引。
        block_count: 原始事件包含的物理 block 数。
        block_size: 从真实 KV tensor shape 自动识别的 block token 数。

    返回:
        当前 block 对应的 token ID 列表；原始事件没有 token 时返回空列表。
    """

    if not token_ids:
        return []
    if block_count <= 1:
        return list(token_ids)
    start = index * block_size
    stop = min(start + block_size, len(token_ids))
    if start >= len(token_ids):
        return []
    return list(token_ids[start:stop])


def _event_to_mapping(event: Any) -> dict[str, Any]:
    """将事件对象转换为浅层 dict。

    参数:
        event: vLLM KV event 对象或 dict。

    返回:
        事件字段字典。
    """

    if isinstance(event, dict):
        return dict(event)
    if hasattr(event, "__struct_fields__"):
        return {key: getattr(event, key) for key in event.__struct_fields__}
    if hasattr(event, "__dict__"):
        return dict(event.__dict__)
    return {}


def _normalize_event_type(event_type: str) -> str:
    """标准化 KV event 类型名。

    参数:
        event_type: vLLM 原始事件名。

    返回:
        KVBM 数据面使用的事件类型名。
    """

    mapping = {
        "BlockStored": "store",
        "BlockRemoved": "removed",
        "BlockEvicted": "evict",
        "BlockCleared": "cleared",
        "BlockHit": "hit",
        "BlockMoved": "move",
        "BlockTransferred": "move",
    }
    return mapping.get(event_type, event_type).lower()


def _extract_list(raw: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    """从多个候选字段中提取列表值。

    参数:
        raw: 事件字段字典。
        keys: 候选字段名。

    返回:
        提取到的列表；字段不存在时返回空列表。
    """

    for key in keys:
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        if isinstance(value, list | tuple):
            return list(value)
        return [value]
    return []


def _extract_int_list(raw: dict[str, Any], keys: tuple[str, ...]) -> list[int]:
    """从事件字段中提取整数列表。

    参数:
        raw: 事件字段字典。
        keys: 候选字段名。

    返回:
        可解析为 int 的字段值列表。
    """

    values = []
    for value in _extract_list(raw, keys):
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return values


def _int_or_zero(value: Any) -> int:
    """将值转换为 int，失败时返回 0。

    参数:
        value: 任意值。

    返回:
        整数值或 0。
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _block_hash_for_index(block_hashes: list[str], index: int, block_id: int) -> str:
    """按下标获取 block hash，不存在时生成 synthetic hash。

    参数:
        block_hashes: 事件携带的 block hash 列表。
        index: 当前 block 下标。
        block_id: 物理 block id。

    返回:
        block hash 字符串。
    """

    if index < len(block_hashes):
        return block_hashes[index]
    return f"physical:{block_id}"


@dataclass
class VllmAscend018PhysicalBlockResolver:
    """从 vLLM-Ascend engine-core 对象树中解析物理 block id。

    属性:
        root: 通常是 NPUModelRunner 实例，用作扫描 block allocator、KVCacheManager
            或 block table 的起点。
        max_depth: 对象树最大扫描深度。
        max_nodes: 单次扫描最多访问对象数量。
    """

    root: Any
    max_depth: int = 6
    max_nodes: int = 4096

    def __post_init__(self) -> None:
        """初始化解析器缓存。

        返回:
            无返回值。
        """

        self._observed_hash_to_block_ids: dict[str, set[int]] = {}
        self._hash_to_block_ids: dict[str, list[int]] = {}
        self._last_scan_nodes = 0
        self._last_scan_candidates: list[str] = []
        self._snapshot_timing_ms = 0.0
        self._resolve_timing_ms = 0.0
        self._fallback_diagnostics: list[dict[str, Any]] = []
        self._cache_full_mapping_count = 0
        self._pending_cache_full_mapping_count = 0
        self._coordinator_cache_mapping_count = 0
        self._pending_coordinator_cache_mapping_count = 0
        self._pending_coordinator_cache_mapping_ms = 0.0
        self._pending_coordinator_cache_visited_blocks = 0
        self._pending_coordinator_cache_skipped_blocks = 0

    def record_physical_block(self, block_hash: Any, block_id: int) -> None:
        """记录已知 hash 到物理 block id 的映射。

        参数:
            block_hash: vLLM 内部或外部 block hash。
            block_id: 物理 KV block id。

        返回:
            无返回值。
        """

        _add_hash_block_mapping(self._observed_hash_to_block_ids, block_hash, block_id)
        for alias in _block_hash_aliases(block_hash):
            block_ids = self._hash_to_block_ids.setdefault(alias, [])
            if block_id not in block_ids:
                block_ids.append(block_id)
                block_ids.sort()

    def enrich_event(
        self,
        event: Any,
        *,
        refresh_on_miss: bool = True,
    ) -> dict[str, Any]:
        """为 KV event 补齐 physical_block_ids。

        参数:
            event: vLLM KV event 对象，或 relay dict。

        返回:
            标准 relay dict。若无法解析物理 block id，则保留原字段并不补齐。
        """

        raw = _event_to_mapping(event)
        data = dict(raw.get("data") or {}) if isinstance(raw.get("data"), dict) else {}
        merged = {**raw, **data}
        normalized = _normalize_kv_event(event)
        if normalized["block_ids"]:
            return {"type": raw.get("type") or type(event).__name__, "data": merged}

        block_hashes = normalized["raw_block_hashes"] or normalized["block_hashes"]
        block_ids = self.resolve_block_hashes(
            block_hashes,
            refresh_on_miss=refresh_on_miss,
        )
        if not block_ids:
            return {"type": raw.get("type") or type(event).__name__, "data": merged}

        merged["physical_block_ids"] = block_ids
        merged.setdefault("block_hashes", block_hashes)
        return {"type": raw.get("type") or type(event).__name__, "data": merged}

    def resolve_block_hashes(
        self,
        block_hashes: list[Any],
        *,
        refresh_on_miss: bool = True,
    ) -> list[int]:
        """按 block hash 列表解析物理 block id。

        参数:
            block_hashes: vLLM KV event 中的 block hash 列表。

        返回:
            与 block_hashes 顺序对应的物理 block id 列表；任意 hash 未命中时返回空列表。
        """

        if not block_hashes:
            return []
        resolved = self._resolve_cached_block_hashes(block_hashes)
        if resolved or not refresh_on_miss:
            return resolved
        started_ns = time.perf_counter_ns()
        self.refresh()
        self._snapshot_timing_ms += (time.perf_counter_ns() - started_ns) / 1_000_000.0
        return self._resolve_cached_block_hashes(block_hashes)

    def _resolve_cached_block_hashes(self, block_hashes: list[Any]) -> list[int]:
        """仅使用已有增量映射解析 block hash。

        参数:
            block_hashes: 需要按顺序解析的 block hash 列表。

        返回:
            全部命中时返回物理 block id 列表；任一 hash 未命中时返回空列表。
        """

        resolved = []
        for block_hash in block_hashes:
            ids = self._lookup_block_hash(block_hash)
            if not ids:
                return []
            resolved.append(ids[0])
        return resolved

    def add_snapshot_timing_ms(self, elapsed_ms: float) -> None:
        """累加一次全量映射兜底扫描耗时。

        参数:
            elapsed_ms: 本次 BlockPool 快照扫描耗时，单位毫秒。

        返回:
            无返回值。
        """

        self._snapshot_timing_ms += max(float(elapsed_ms), 0.0)

    def add_resolve_timing_ms(self, elapsed_ms: float) -> None:
        """累加当前事件批次的物理 block 解析耗时。

        参数:
            elapsed_ms: 事件标准化与 hash 解析耗时，单位毫秒。

        返回:
            无返回值。
        """

        self._resolve_timing_ms += max(float(elapsed_ms), 0.0)

    def consume_resolve_timing_ms(self) -> float:
        """取出并清空当前事件批次的解析耗时。

        返回:
            返回累计解析耗时，单位毫秒。
        """

        elapsed_ms = self._resolve_timing_ms
        self._resolve_timing_ms = 0.0
        return elapsed_ms

    def record_fallback_diagnostic(
        self,
        *,
        event_type: str,
        group_idx: int,
        hash_count: int,
        targeted_miss_count: int,
        targeted_lookup_ms: float,
        fallback_reason: str,
    ) -> None:
        """记录一次事件定向查询或全量兜底结果。

        参数:
            event_type: 标准化后的 KV 事件类型。
            group_idx: KV cache group 编号。
            hash_count: 当前事件携带的 hash 数量。
            targeted_miss_count: 定向查询未命中的 hash 数量。
            targeted_lookup_ms: 定向查询耗时，单位毫秒。
            fallback_reason: 处理结果或触发兜底的原因。

        返回:
            无返回值。
        """

        self._fallback_diagnostics.append(
            {
                "event_type": event_type,
                "group_idx": group_idx,
                "hash_count": hash_count,
                "targeted_miss_count": targeted_miss_count,
                "targeted_lookup_ms": round(max(targeted_lookup_ms, 0.0), 3),
                "fallback_reason": fallback_reason,
            }
        )

    def note_cache_full_mappings(self, count: int) -> None:
        """累加从 cache_full_blocks 增量捕获的物理映射数。

        参数:
            count: 本次捕获的 block 映射数量。

        返回:
            无返回值。
        """

        self._cache_full_mapping_count += max(int(count), 0)
        self._pending_cache_full_mapping_count += max(int(count), 0)

    def consume_cache_full_mapping_count(self) -> int:
        """取出并清空当前事件批次前捕获的 full block 映射数。

        返回:
            返回自上次取出后新捕获的物理 block 映射数。
        """

        count = self._pending_cache_full_mapping_count
        self._pending_cache_full_mapping_count = 0
        return count

    def note_coordinator_cache_mapping_result(
        self,
        *,
        recorded: int,
        visited_blocks: int,
        skipped_blocks: int,
        elapsed_ms: float,
    ) -> None:
        """累加 coordinator.cache_blocks 增量映射结果。

        参数:
            recorded: 本次新记录的映射数量。
            visited_blocks: 本次实际遍历的新 block 数量。
            skipped_blocks: 请求级游标跳过的旧 block 数量。
            elapsed_ms: hook 本次总耗时，单位毫秒。

        返回:
            无返回值。
        """

        parsed_count = max(int(recorded), 0)
        self._coordinator_cache_mapping_count += parsed_count
        self._pending_coordinator_cache_mapping_count += parsed_count
        self._pending_coordinator_cache_mapping_ms += max(float(elapsed_ms), 0.0)
        self._pending_coordinator_cache_visited_blocks += max(int(visited_blocks), 0)
        self._pending_coordinator_cache_skipped_blocks += max(int(skipped_blocks), 0)

    def consume_coordinator_cache_mapping_count(self) -> int:
        """取出并清空本批事件前 coordinator 捕获的映射数。

        返回:
            自上次取出后新捕获的物理 block 映射数。
        """

        count = self._pending_coordinator_cache_mapping_count
        self._pending_coordinator_cache_mapping_count = 0
        return count

    def consume_coordinator_cache_mapping_timing(self) -> dict[str, float | int]:
        """取出并清空 coordinator 增量映射的耗时与 block 计数。

        返回:
            包含 elapsed_ms、visited_blocks 和 skipped_blocks 的字典。
        """

        result = {
            "elapsed_ms": self._pending_coordinator_cache_mapping_ms,
            "visited_blocks": self._pending_coordinator_cache_visited_blocks,
            "skipped_blocks": self._pending_coordinator_cache_skipped_blocks,
        }
        self._pending_coordinator_cache_mapping_ms = 0.0
        self._pending_coordinator_cache_visited_blocks = 0
        self._pending_coordinator_cache_skipped_blocks = 0
        return result

    def consume_fallback_diagnostics(self) -> list[dict[str, Any]]:
        """取出并清空当前事件批次的兜底诊断。

        返回:
            返回本批次诊断字典列表。
        """

        diagnostics = list(self._fallback_diagnostics)
        self._fallback_diagnostics.clear()
        return diagnostics

    def reset_snapshot_timing(self) -> None:
        """清空当前事件批次的兜底扫描耗时。

        返回:
            无返回值。
        """

        self._snapshot_timing_ms = 0.0

    def consume_snapshot_timing_ms(self) -> float:
        """取出并清空当前事件批次的兜底扫描耗时。

        返回:
            返回累计扫描耗时，单位毫秒。
        """

        elapsed_ms = self._snapshot_timing_ms
        self._snapshot_timing_ms = 0.0
        return elapsed_ms

    def _lookup_block_hash(self, block_hash: Any) -> list[int]:
        """按多种 hash 表示查找物理 block id。

        参数:
            block_hash: vLLM 事件里的 ExternalBlockHash，可能是 bytes、int 或 str。

        返回:
            命中的物理 block id 列表；未命中时返回空列表。
        """

        # 先查精确表示，避免多个 hash 在去掉 group id 后的
        # 兼容别名发生碰撞，导致批量事件解析到同一 block。
        for alias in [
            *_exact_block_hash_aliases(block_hash),
            *_block_hash_aliases(block_hash),
        ]:
            ids = self._hash_to_block_ids.get(alias)
            if ids:
                return ids
        return []

    def refresh(self) -> dict[str, list[int]]:
        """重新扫描 engine-core 对象树并刷新映射缓存。

        返回:
            block_hash 到物理 block id 列表的映射。
        """

        mapping: dict[str, set[int]] = {
            block_hash: set(block_ids) for block_hash, block_ids in self._observed_hash_to_block_ids.items()
        }
        candidates: list[str] = []
        visited: set[int] = set()
        pending_objects: list[tuple[Any, str, int]] = [(self.root, "root", 0)]
        self._last_scan_nodes = 0
        while pending_objects and self._last_scan_nodes < self.max_nodes:
            obj, path, depth = pending_objects.pop(0)
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)
            self._last_scan_nodes += 1
            self._collect_mapping(obj, path, mapping, candidates)
            if depth >= self.max_depth or _is_scalar(obj):
                continue
            for child_name, child in _iter_scan_children(obj):
                if not _is_relevant_scan_name(child_name):
                    continue
                pending_objects.append((child, f"{path}.{child_name}", depth + 1))

        self._hash_to_block_ids = {block_hash: sorted(block_ids) for block_hash, block_ids in mapping.items()}
        self._last_scan_candidates = candidates[:64]
        return self._hash_to_block_ids

    def diagnostics(self) -> dict[str, Any]:
        """返回解析器诊断信息。

        返回:
            包含扫描节点数、映射数量和候选路径的诊断字典。
        """

        self.refresh()
        return {
            "mapped_hashes": len(self._hash_to_block_ids),
            "observed_mapped_hashes": len(self._observed_hash_to_block_ids),
            "cache_full_mappings": self._cache_full_mapping_count,
            "coordinator_cache_mappings": self._coordinator_cache_mapping_count,
            "scan_nodes": self._last_scan_nodes,
            "sample_hashes": list(self._hash_to_block_ids.keys())[:8],
            "candidate_paths": self._last_scan_candidates[:16],
        }

    def _collect_mapping(
        self,
        obj: Any,
        path: str,
        mapping: dict[str, set[int]],
        candidates: list[str],
    ) -> None:
        """从当前对象中收集 hash 到物理 block 的映射。

        参数:
            obj: 当前扫描对象。
            path: 当前对象在扫描树中的路径。
            mapping: 待更新的映射集合。
            candidates: 记录可能相关的对象路径，便于日志诊断。

        返回:
            无返回值。
        """

        before = sum(len(ids) for ids in mapping.values())
        if isinstance(obj, dict):
            self._collect_from_dict(obj, mapping)
        elif isinstance(obj, list | tuple | set):
            for item in obj:
                self._collect_from_object(item, mapping)
        else:
            self._collect_from_object(obj, mapping)
        after = sum(len(ids) for ids in mapping.values())
        if after > before:
            candidates.append(path)

    @staticmethod
    def _collect_from_dict(value: dict[Any, Any], mapping: dict[str, set[int]]) -> None:
        """从 dict 中收集 hash 到物理 block 的映射。

        参数:
            value: 待检查 dict。
            mapping: 待更新的映射集合。

        返回:
            无返回值。
        """

        keys = {str(key) for key in value.keys()}
        if {"block_hash", "hash"} & keys and {"physical_block_id", "block_id"} & keys:
            block_hash = value.get("block_hash", value.get("hash"))
            block_id = value.get("physical_block_id", value.get("block_id"))
            _add_hash_block_mapping(mapping, block_hash, block_id)
        if {"_block_hash", "block_hash"} & keys and "block_id" in keys:
            block_hash = value.get("_block_hash", value.get("block_hash"))
            _add_hash_block_mapping(mapping, block_hash, value.get("block_id"))
        block_hashes = _extract_list(value, ("block_hashes", "hashes"))
        block_ids = _extract_int_list(value, ("physical_block_ids", "block_ids"))
        for index, block_hash in enumerate(block_hashes):
            if index < len(block_ids):
                _add_hash_block_mapping(mapping, block_hash, block_ids[index])
        for key, item in value.items():
            if _looks_like_block_hash(key) and not _is_scalar(item):
                for block_id in _extract_block_ids_from_value(item):
                    _add_hash_block_mapping(mapping, key, block_id)

    def _collect_from_object(self, obj: Any, mapping: dict[str, set[int]]) -> None:
        """从普通对象中收集 hash 到物理 block 的映射。

        参数:
            obj: 待检查对象。
            mapping: 待更新的映射集合。

        返回:
            无返回值。
        """

        if _is_scalar(obj):
            return
        if isinstance(obj, dict):
            self._collect_from_dict(obj, mapping)
            return
        raw = _object_public_attrs(obj)
        if not raw:
            return
        self._collect_from_dict(raw, mapping)


def get_bound_vllm_ascend_018_provider(model_runner: Any) -> Any | None:
    """读取 hook 绑定到 model runner 上的 provider。

    参数:
        model_runner: vLLM-Ascend engine-core 内的 NPUModelRunner 实例。

    返回:
        如果 hook 已绑定 provider，则返回该 provider；否则返回 None。
    """

    return getattr(model_runner, _PROVIDER_ATTR, None)


def get_bound_vllm_ascend_018_dataplane(model_runner: Any) -> Any | None:
    """读取 hook 绑定到 model runner 上的数据面控制对象。

    参数:
        model_runner: vLLM-Ascend engine-core 内的 NPUModelRunner 实例。

    返回:
        如果 hook 已绑定数据面控制对象，则返回该对象；否则返回 None。
    """

    return getattr(model_runner, _DATAPLANE_ATTR, None)


def get_bound_vllm_ascend_018_physical_block_resolver(model_runner: Any) -> Any | None:
    """读取 hook 绑定到 model runner 上的物理 block 解析器。

    参数:
        model_runner: vLLM-Ascend engine-core 内的 NPUModelRunner 实例。

    返回:
        如果 hook 已绑定解析器，则返回该解析器；否则返回 None。
    """

    return getattr(model_runner, _RESOLVER_ATTR, None)


def _iter_scan_children(obj: Any) -> list[tuple[str, Any]]:
    """枚举对象树扫描子节点。

    参数:
        obj: 当前扫描对象。

    返回:
        子节点名称和值列表。
    """

    children: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            children.append((str(key), value))
        return children
    if isinstance(obj, list | tuple | set):
        for index, value in enumerate(obj):
            children.append((str(index), value))
        return children
    for name in dir(obj):
        if name.startswith("_") and not _is_relevant_scan_name(name):
            continue
        if not _is_relevant_scan_name(name):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            logger.debug("Cannot inspect cache attribute %s", name, exc_info=True)
            continue
        if callable(value) or _is_scalar(value):
            continue
        children.append((name, value))
    return children


def _object_public_attrs(obj: Any) -> dict[str, Any]:
    """读取对象的公开属性快照。

    参数:
        obj: 待读取对象。

    返回:
        属性名到属性值的浅层字典。
    """

    if hasattr(obj, "__dict__"):
        return {key: value for key, value in vars(obj).items() if not callable(value)}
    fields = getattr(obj, "__struct_fields__", None)
    if fields:
        return {key: getattr(obj, key) for key in fields if hasattr(obj, key)}
    return {}


def _extract_block_ids_from_value(value: Any) -> list[int]:
    """从任意值中提取物理 block id。

    参数:
        value: 可能是 int、list、dict 或带 block_id 属性的对象。

    返回:
        解析出的物理 block id 列表。
    """

    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list | tuple | set):
        ids = []
        for item in value:
            ids.extend(_extract_block_ids_from_value(item))
        return ids
    if isinstance(value, dict):
        return _extract_int_list(value, ("physical_block_ids", "physical_block_id", "block_ids", "block_id"))
    raw = _object_public_attrs(value)
    if raw:
        return _extract_int_list(raw, ("physical_block_ids", "physical_block_id", "block_ids", "block_id"))
    return []


def _add_hash_block_mapping(mapping: dict[str, set[int]], block_hash: Any, block_id: Any) -> None:
    """向映射中追加一个 hash 到物理 block id 的关系。

    参数:
        mapping: 待更新映射。
        block_hash: 逻辑 block hash。
        block_id: 物理 block id。

    返回:
        无返回值。
    """

    if block_hash is None:
        return
    try:
        parsed_block_id = int(block_id)
    except (TypeError, ValueError):
        return
    for alias in _block_hash_aliases(block_hash):
        mapping.setdefault(alias, set()).add(parsed_block_id)


def _block_hash_aliases(block_hash: Any) -> list[str]:
    """生成 block hash 的多种可匹配表示。

    参数:
        block_hash: vLLM 内部或外部 KV block hash。

    返回:
        可用于映射查找的字符串别名列表。
    """

    if block_hash is None:
        return []
    aliases = {str(block_hash)}
    if isinstance(block_hash, bytes | bytearray):
        raw = bytes(block_hash)
        aliases.add(repr(raw))
        aliases.add(raw.hex())
        if len(raw) > 4:
            external = raw[:-4]
            aliases.add(str(external))
            aliases.add(repr(external))
            aliases.add(external.hex())
            aliases.add(str(int.from_bytes(external, byteorder="big") & ((1 << 64) - 1)))
    else:
        try:
            aliases.add(str(int(block_hash)))
        except (TypeError, ValueError):
            pass
    return [alias for alias in aliases if alias]


def _exact_block_hash_aliases(block_hash: Any) -> list[str]:
    """生成不删减 group id 的精确 hash 表示。

    参数:
        block_hash: vLLM 内部或外部 KV block hash。

    返回:
        按优先级排列的精确字符串别名列表。
    """

    if block_hash is None:
        return []
    aliases = [str(block_hash)]
    if isinstance(block_hash, bytes | bytearray):
        raw = bytes(block_hash)
        aliases.extend((repr(raw), raw.hex()))
    else:
        try:
            aliases.append(str(int(block_hash)))
        except (TypeError, ValueError):
            pass
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _is_relevant_scan_name(name: str) -> bool:
    """判断属性名是否值得扫描。

    参数:
        name: 属性或路径名。

    返回:
        值得扫描返回 True，否则返回 False。
    """

    lowered = name.lower()
    keywords = (
        "block",
        "cache",
        "kv",
        "manager",
        "allocator",
        "table",
        "scheduler",
        "runner",
        "request",
    )
    return any(keyword in lowered for keyword in keywords)


def _looks_like_block_hash(value: Any) -> bool:
    """判断字符串是否像 block hash。

    参数:
        value: 待判断字符串。

    返回:
        像 block hash 返回 True，否则返回 False。
    """

    if isinstance(value, bytes | bytearray):
        return len(value) >= 8
    if not isinstance(value, str):
        return False
    return len(value) >= 8 and not value.isdigit()


def _is_scalar(value: Any) -> bool:
    """判断值是否为无需继续扫描的标量。

    参数:
        value: 任意值。

    返回:
        是标量返回 True，否则返回 False。
    """

    return value is None or isinstance(value, str | bytes | int | float | bool)


def _runtime_config_from_env() -> WorkerKVBMConfig:
    """从环境变量构造 engine-core 内 WorkerKVBMRuntime 配置。

    返回:
        WorkerKVBMConfig 实例。
    """

    return WorkerKVBMConfig(
        worker_id=os.getenv(_WORKER_ID_ENV, ""),
        disk_root=Path(os.getenv(_DISK_ROOT_ENV, "/tmp/jiuwen-kvbm-worker")),
        npu_high_watermark_blocks=_int_env(_NPU_HIGH_WATERMARK_ENV, 0),
        ram_high_watermark_bytes=_int_env(_RAM_HIGH_WATERMARK_ENV, 0),
        block_size=_int_env(_BLOCK_SIZE_ENV, 16),
        clean_disk_on_start=_bool_env(_CLEAN_DISK_ON_START_ENV, True),
        data_backend=os.getenv(_DATA_BACKEND_ENV, "local"),
    )


def _int_env(name: str, default: int) -> int:
    """读取整数环境变量。

    参数:
        name: 环境变量名称。
        default: 解析失败或不存在时的默认值。

    返回:
        整数值。
    """

    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    """读取布尔环境变量。

    参数:
        name: 环境变量名称。
        default: 环境变量不存在时的默认值。

    返回:
        环境变量为 1/true/yes/on 时返回 True；为 0/false/no/off 时返回 False；
        不存在时返回 default。
    """

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def uninstall_vllm_ascend_018_engine_core_hook(model_runner_cls: type[Any]) -> bool:
    """卸载测试中安装的 vLLM-Ascend v0.18.0 hook。

    参数:
        model_runner_cls: 被 patch 的 model runner 类。

    返回:
        成功卸载返回 True；未安装时返回 False。

    说明:
        该函数主要用于单元测试，生产流程通常不需要卸载 hook。
    """

    original: Callable[..., Any] | None = getattr(model_runner_cls, _ORIGINAL_ATTR, None)
    if original is None:
        return False
    setattr(model_runner_cls, "initialize_kv_cache_tensors", original)
    timing_attr = "_jiuwen_kvbm_original_execute_model_timing"
    timing_original = getattr(model_runner_cls, timing_attr, None)
    if timing_original is not None:
        setattr(model_runner_cls, "execute_model", timing_original)
        delattr(model_runner_cls, timing_attr)
    setattr(model_runner_cls, _PATCH_ATTR, False)
    delattr(model_runner_cls, _ORIGINAL_ATTR)
    return True
