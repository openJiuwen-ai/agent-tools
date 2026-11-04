"""数据面入口 — 单进程运行 Python 控制面 + Rust 代理。

Python 控制面负责路由决策（预处理、KV 缓存匹配、调度、负载均衡），
Rust 代理负责高性能 HTTP 转发（连接池、流式传输、TTFT 计时）。

用法::

    python -m openjiuwentools.infer_router.data_plane.server --config config.yaml
    python -m openjiuwentools.infer_router.data_plane.server --config config.yaml --listen 0.0.0.0:8080
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass

from loguru import logger

from openjiuwentools.infer_router.config.config import create_settings, set_global_settings
from openjiuwentools.infer_router.kv_cache.event import CacheEvent, CacheLocation
from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.kv_cache.kv_event_generator import KVEventGenerator
from openjiuwentools.infer_router.kv_cache.kv_event_manager import KVEventManager
from openjiuwentools.infer_router.kvbm.transfer_orchestrator import (
    orchestrate_cross_worker_transfer,
)
from openjiuwentools.infer_router.monitoring.metrics import metrics
from openjiuwentools.infer_router.monitoring.performance_stats import performance_stats
from openjiuwentools.infer_router.preprocess.preprocessor import Preprocessor
from openjiuwentools.infer_router.prompt_diagnostics import (
    digest,
    enabled as prompt_diagnostics_enabled,
    prompt_observation,
)
from openjiuwentools.infer_router.routing.router import Router
from openjiuwentools.infer_router.schedule.scheduler import Scheduler
from openjiuwentools.infer_router.schemas.agent_hints import (
    ChatCompletionRequest,
    CompletionRequest,
)
from openjiuwentools.infer_router.worker.worker_manager import WorkerManager
from openjiuwentools.infer_router.worker.workload_manager import WorkloadManager


def _resolve_pd_prefill_max_inflight(cfg) -> int:
    """解析后端无关的 PD Prefill 并发上限。

    Args:
        cfg: Router 配置对象，包含通用配置及旧 KVBM 兼容配置。

    Returns:
        每个 P Worker 允许同时执行的 PD Prefill 请求数。返回 0 表示由
        Router 动态预算准入，Rust 数据面不再叠加固定 Semaphore 限制。

    """
    configured = getattr(cfg, "router_pd_prefill_max_inflight", None)
    if configured is None:
        configured = getattr(cfg, "kvbm_router_pd_prefill_max_inflight", 8)
    return max(int(configured), 0)


class NoWorkerCapacityError(RuntimeError):
    """Typed capacity rejection exposed to the Rust callback boundary."""

    status_code = 503


def _build_models_fn(worker_manager: WorkerManager):
    """构建 OpenAI 兼容的模型列表回调，供 Rust 数据面查询。"""

    def models_fn() -> str:
        models = sorted({worker.model for worker in worker_manager.get_all_workers().values() if worker.model})
        return json.dumps(
            {
                "object": "list",
                "data": [{"id": model, "object": "model", "owned_by": "jiuwen"} for model in models],
            },
            ensure_ascii=False,
        )

    return models_fn


def _normalize_max_tokens(
    body: dict,
    default_max_tokens: int | None,
    *,
    prefer_max_completion_tokens: bool = False,
) -> None:
    """仅在显式配置 Router 默认值时补充生成上限。

    默认配置为 ``None``，此时保持请求原样，让 vLLM 根据剩余上下文动态
    决定上限。Chat Completions 的 ``max_completion_tokens`` 优先级高于旧的
    ``max_tokens``，存在时也不应再注入后者。
    """
    if default_max_tokens is None:
        return
    if prefer_max_completion_tokens and body.get("max_completion_tokens") is not None:
        return
    if body.get("max_tokens") is None:
        body["max_tokens"] = default_max_tokens


def _build_route_fn(
    preprocessor: Preprocessor,
    kvcache_manager: KVCacheManager,
    scheduler: Scheduler,
    router: Router,
    worker_manager: WorkerManager,
    event_manager: KVEventManager | None = None,
    event_generator: KVEventGenerator | None = None,
    request_workers: dict | None = None,
    request_contexts: dict | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
):
    """构建路由回调函数，供 Rust 代理对每个请求调用。

    Rust 侧通过 spawn_blocking + GIL 调用此回调，执行完整的路由决策流程：
    解析请求 → 预处理/分词 → KV 缓存匹配 → 调度 → 选择 worker。
    若所有 worker 饱和，会在 router 侧排队等待直到超时。

    Args:
        preprocessor: 请求预处理器（分词、构建 RouteHint）
        kvcache_manager: KV 缓存管理器（前缀匹配）
        scheduler: 请求调度器（FCFS/优先级等策略）
        router: 路由器（负载均衡算法选择 worker）
        worker_manager: Worker 管理器（获取 worker 信息）
        event_manager: KV 事件管理器（可选）
        event_generator: KV 事件生成器（inner_event 模式，可选）
        request_workers: 请求 ID → worker ID 列表的映射，用于完成时释放
        request_contexts: 请求 ID → 路由上下文的映射，用于完成时上报训练数据
        event_loop: 后台 asyncio 事件循环，用于同步路由回调触发异步 worker 预取

    Returns:
        route_fn 回调函数，签名为 (body_json: str, path: str) -> dict

    """

    from openjiuwentools.infer_router.config.config import get_global_settings

    lock = threading.Lock()
    cfg = get_global_settings()
    router_managed_pd = bool(getattr(cfg, "kvbm_enabled", False) and getattr(cfg, "kvbm_router_pd_enabled", False))
    queue_timeout = cfg.router_queue_timeout
    poll_interval = cfg.router_queue_poll_interval
    default_max_tokens = cfg.default_max_tokens
    default_ignore_eos = cfg.default_ignore_eos
    _request_workers = request_workers if request_workers is not None else {}
    _request_contexts = request_contexts if request_contexts is not None else {}
    request_sequence = itertools.count(1)

    def route_fn(body_json: str, path: str) -> dict:
        route_started = time.perf_counter()
        request_parse_started = route_started
        body = json.loads(body_json)
        _normalize_max_tokens(
            body,
            default_max_tokens,
            prefer_max_completion_tokens=path == "/v1/chat/completions",
        )
        is_stream = body.get("stream", False)

        if path == "/v1/chat/completions":
            req = ChatCompletionRequest(**body)
        elif path == "/v1/completions":
            req = CompletionRequest(**body)
        else:
            raise ValueError(f"Unsupported path: {path}")

        if "ignore_eos" not in body and default_ignore_eos:
            body["ignore_eos"] = True

        agent_hints = None
        if req.jiuwenext and req.jiuwenext.agent_hints:
            agent_hints = req.jiuwenext.agent_hints

        request_id = f"chatcmpl-{uuid.uuid4()}" if "chat" in path else f"cmpl-{uuid.uuid4()}"
        request_parse_ms = (time.perf_counter() - request_parse_started) * 1000.0
        preprocess_started = time.perf_counter()
        if isinstance(req, ChatCompletionRequest):
            route_hint = preprocessor.process(req, agent_hints, request_id)
            if prompt_diagnostics_enabled():
                manager = preprocessor.tokenizer_manager
                observed_tokenizer = manager.get_cached_tokenizer(req.model)
                logger.info(
                    "[PDPromptDiagnostics] {}",
                    json.dumps(
                        prompt_observation(
                            body,
                            route_hint.token_ids,
                            stage="router",
                            request_id=request_id,
                            tokenizer=observed_tokenizer,
                            template_kwargs={
                                "add_generation_prompt": True,
                                **({"tools": req.tools} if req.tools else {}),
                            },
                            effective_messages_sha256=digest(manager.format_chat_messages(req.messages)),
                        ),
                        sort_keys=True,
                    ),
                )
            preprocessor_ms = (time.perf_counter() - preprocess_started) * 1000.0
            logger.info(
                "[REQUEST] {} model={}, messages={}, max_tokens={}, stream={}",
                request_id,
                req.model,
                len(req.messages),
                req.max_tokens,
                is_stream,
            )
        else:
            route_hint = preprocessor.process_completion(req, agent_hints, request_id)
            preprocessor_ms = (time.perf_counter() - preprocess_started) * 1000.0
            logger.info(
                "[REQUEST] {} model={}, prompt_len={}, max_tokens={}, stream={}",
                request_id,
                req.model,
                len(req.prompt) if isinstance(req.prompt, (str, list)) else 0,
                req.max_tokens,
                is_stream,
            )

        deadline = time.monotonic() + queue_timeout
        logged_queuing = False
        worker_selection_started = time.perf_counter()
        route_attempt_ms = 0.0
        capacity_wait_ms = 0.0
        capacity_turn_wait_ms = 0.0
        pd_pending_prefix_wait_ms = 0.0
        route_attempt_count = 0
        scheduled_route_hint = None
        overlap_scores = None
        last_admission_signature = None
        capacity_waiter_registered = False

        def remove_capacity_waiter() -> None:
            nonlocal capacity_waiter_registered
            if not capacity_waiter_registered:
                return
            remove_waiter = getattr(router, "remove_capacity_waiter", None)
            if callable(remove_waiter):
                remove_waiter(request_id, str(body.get("model", "")))
            capacity_waiter_registered = False

        while True:
            attempt_started = time.perf_counter()
            route_attempt_count += 1
            if router_managed_pd:
                pending_hint = scheduled_route_hint or route_hint
                wait_for_prefix = getattr(
                    router,
                    "wait_for_pd_prefix_affinity",
                    None,
                )
                refresh_prefix = getattr(
                    router,
                    "refresh_pd_prefix_affinity",
                    None,
                )
                remaining = deadline - time.monotonic()
                if callable(wait_for_prefix) and remaining > 0:
                    pending_started = time.perf_counter()
                    pending_match = wait_for_prefix(
                        pending_hint,
                        timeout=remaining,
                    )
                    pd_pending_prefix_wait_ms += (time.perf_counter() - pending_started) * 1000.0
                    if pending_match is not None and callable(refresh_prefix):
                        refresh_prefix(pending_hint, pending_match)
            get_generation = getattr(router, "get_capacity_generation", None)
            generation_value = get_generation() if callable(get_generation) else 0
            attempt_generation = generation_value if isinstance(generation_value, int) else 0
            overlap_timing: dict[str, float] = {}
            route_lock_wait_ms = 0.0
            snapshot_age_ms = 0.0
            snapshot_reused = scheduled_route_hint is not None

            # 同一请求在容量排队期间复用一份prefix快照。容量变化只决定
            # 是否重试准入，不改变请求Token，因此不需要重复扫描Radix索引。
            if scheduled_route_hint is None:
                route_lock_started = time.perf_counter()
                with lock:
                    route_lock_wait_ms = (time.perf_counter() - route_lock_started) * 1000.0
                    overlap_scores = None
                    snapshot_ready_at = time.perf_counter()
                    if route_hint.token_ids:
                        match_worker_ids = _get_prefix_match_worker_ids(
                            worker_manager=worker_manager,
                            model=route_hint.model,
                            include_decode_workers=bool(
                                getattr(
                                    cfg,
                                    "router_prefix_match_decode_workers",
                                    True,
                                )
                            ),
                        )
                        prefix_snapshots = {}
                        overlap_scores = kvcache_manager.find_matches(
                            token_ids=route_hint.token_ids,
                            model=route_hint.model,
                            include_workers=match_worker_ids,
                            timing=overlap_timing,
                            snapshot_output=prefix_snapshots,
                        )
                        route_hint.precomputed_prefix_snapshots = prefix_snapshots
                        logger.info(
                            "[KVMatchTiming] request={} phase=data_plane workers={} "
                            "total={:.2f}ms lock_wait={:.2f}ms compute={:.2f}ms "
                            "cursor_hits={} cursor_misses={}",
                            request_id,
                            int(overlap_timing.get("workers", 0)),
                            overlap_timing.get("total_ms", 0.0),
                            overlap_timing.get("lock_wait_ms", 0.0),
                            overlap_timing.get("compute_ms", 0.0),
                            int(overlap_timing.get("cursor_cache_hits", 0)),
                            int(overlap_timing.get("cursor_cache_misses", 0)),
                        )
                    if scheduled_route_hint is None:
                        scheduler.submit(route_hint, overlap_scores)
                        scheduled = scheduler.get_next_request()
                        if not scheduled:
                            raise RuntimeError("No request from scheduler")
                        scheduled_route_hint = scheduled.route_hint
                        scheduler.mark_task_completed()
                    snapshot_age_ms = (time.perf_counter() - snapshot_ready_at) * 1000.0

            model = str(body.get("model", ""))
            if not capacity_waiter_registered:
                enqueue_waiter = getattr(router, "enqueue_capacity_waiter", None)
                if callable(enqueue_waiter):
                    registered = enqueue_waiter(
                        request_id,
                        model,
                        only_if_contended=True,
                    )
                    if isinstance(registered, bool):
                        capacity_waiter_registered = registered

            can_attempt_route = True
            if capacity_waiter_registered:
                remaining = deadline - time.monotonic()
                wait_for_turn = getattr(router, "wait_for_capacity_turn", None)
                if remaining <= 0:
                    can_attempt_route = False
                elif callable(wait_for_turn):
                    turn_window_size = 1
                    if router_managed_pd:
                        healthy_decode_workers = worker_manager.get_all_healthy_decode_workers(model)
                        turn_window_size = max(len(healthy_decode_workers), 1)
                    turn_wait_started = time.perf_counter()
                    try:
                        can_attempt_route = bool(
                            wait_for_turn(
                                request_id,
                                model,
                                timeout=remaining,
                                window_size=turn_window_size,
                            )
                        )
                    except Exception:
                        remove_capacity_waiter()
                        raise
                    finally:
                        capacity_turn_wait_ms += (time.perf_counter() - turn_wait_started) * 1000.0

            routing_started = time.perf_counter()
            if can_attempt_route:
                try:
                    prefill_id, decode_id = router.route_to_workers(
                        scheduled_route_hint,
                        router_managed_pd=router_managed_pd,
                    )
                except Exception:
                    remove_capacity_waiter()
                    raise
            else:
                prefill_id, decode_id = None, None
            routing_ms = (time.perf_counter() - routing_started) * 1000

            attempt_elapsed_ms = (time.perf_counter() - attempt_started) * 1000.0
            route_attempt_ms += attempt_elapsed_ms
            logger.info(
                "[RouteDecisionTiming] request={} route_lock_wait={:.2f}ms "
                "kv_match={:.2f}ms snapshot_age={:.2f}ms routing={:.2f}ms "
                "total={:.2f}ms capacity_generation={} snapshot_reused={}",
                request_id,
                route_lock_wait_ms,
                overlap_timing.get("total_ms", 0.0),
                max(snapshot_age_ms, 0.0),
                routing_ms,
                attempt_elapsed_ms,
                attempt_generation,
                snapshot_reused,
            )

            if prefill_id is not None:
                remove_capacity_waiter()
                break

            try:
                admission = router.get_worker_admission_snapshot(
                    model,
                    route_hint,
                    router_managed_pd=router_managed_pd,
                )
            except Exception:
                remove_capacity_waiter()
                raise
            prefill_details = admission.get("prefill_details", {})
            decode_details = admission.get("decode_details", {})
            admission_signature = (
                admission["blocker"],
                admission["prefill_admitted"],
                admission["prefill_healthy"],
                admission["decode_admitted"],
                admission["decode_healthy"],
                tuple(sorted((worker_id, details.get("reason")) for worker_id, details in prefill_details.items())),
                tuple(sorted((worker_id, details.get("reason")) for worker_id, details in decode_details.items())),
            )
            if admission_signature != last_admission_signature:
                logger.info(
                    "[PDAdmissionWait] request={} attempt={} generation={} "
                    "blocker={} prefill={}/{}/{} decode={}/{}/{} "
                    "prefill_details={} decode_details={}",
                    route_hint.request_id or request_id,
                    route_attempt_count,
                    attempt_generation,
                    admission["blocker"],
                    admission["prefill_admitted"],
                    admission["prefill_healthy"],
                    admission["prefill_configured"],
                    admission["decode_admitted"],
                    admission["decode_healthy"],
                    admission["decode_configured"],
                    admission.get("prefill_details", {}),
                    admission.get("decode_details", {}),
                )
                last_admission_signature = admission_signature

            if time.monotonic() >= deadline:
                remove_capacity_waiter()
                failed_request_id = route_hint.request_id or request_id
                raise NoWorkerCapacityError(
                    f"No available workers with capacity for model {model} "
                    f"(request_id={failed_request_id}, "
                    f"queued {queue_timeout:.0f}s, "
                    f"blocker={admission['blocker']}, "
                    "prefill "
                    f"admitted={admission['prefill_admitted']}/"
                    f"healthy={admission['prefill_healthy']}/"
                    f"configured={admission['prefill_configured']}, "
                    "decode "
                    f"admitted={admission['decode_admitted']}/"
                    f"healthy={admission['decode_healthy']}/"
                    f"configured={admission['decode_configured']}, "
                    f"decode_details={admission.get('decode_details', {})})"
                )

            if not capacity_waiter_registered:
                enqueue_waiter = getattr(router, "enqueue_capacity_waiter", None)
                if callable(enqueue_waiter):
                    registered = enqueue_waiter(
                        request_id,
                        model,
                        only_if_contended=False,
                    )
                    if isinstance(registered, bool):
                        capacity_waiter_registered = registered

            if not logged_queuing:
                logger.info(
                    "[Queue] {} waiting for available worker (all at concurrency limit)",
                    request_id,
                )
                logged_queuing = True

            if router_managed_pd and capacity_waiter_registered:
                defer_waiter = getattr(router, "defer_capacity_waiter", None)
                if callable(defer_waiter):
                    defer_waiter(
                        request_id,
                        model,
                        attempt_generation,
                        blocked_worker_ids=tuple(
                            worker_id
                            for worker_id, details in decode_details.items()
                            if details.get("reason") != "decode_affinity_grace_wait"
                        ),
                    )

            remaining = deadline - time.monotonic()
            if remaining > 0:
                capacity_wait_started = time.perf_counter()
                wait_for_change = getattr(router, "wait_for_capacity_change", None)
                if callable(wait_for_change) and isinstance(generation_value, int):
                    wait_for_change(
                        attempt_generation,
                        timeout=remaining,
                        request_id=request_id,
                    )
                else:
                    router.wait_for_capacity(timeout=min(poll_interval, remaining))
                capacity_wait_ms += (time.perf_counter() - capacity_wait_started) * 1000.0

        worker_selection_ms = (time.perf_counter() - worker_selection_started) * 1000.0
        route_build_started = time.perf_counter()
        req_id = route_hint.request_id or request_id
        request_context = {
            "request_index": next(request_sequence),
            "route_hint": route_hint,
            "overlap_scores": overlap_scores,
            "decode_id": decode_id,
            "prefill_id": prefill_id,
            "prompt_tokens": len(route_hint.token_ids) if route_hint.token_ids else 0,
            "router_phase_ms": {},
        }
        if decode_id is not None:
            decode_worker = worker_manager.get_worker(decode_id)
            request_context["decode_selection_snapshot"] = {
                "selected_at_monotonic": time.monotonic(),
                "inflight": router.inflight.get(decode_id),
                "running": int(getattr(decode_worker, "running_requests", 0) or 0),
                "waiting": int(getattr(decode_worker, "queue_depth", 0) or 0),
            }
        _request_contexts[req_id] = request_context
        if decode_id is None:
            _request_workers[req_id] = [prefill_id]
            kvbm_timing = _prefetch_kvbm_blocks_for_worker(
                worker_id=prefill_id,
                route_hint=route_hint,
                request_id=req_id,
                cfg=cfg,
                kvcache_manager=kvcache_manager,
                worker_manager=worker_manager,
                event_loop=event_loop,
            )
            result = _build_combined_result(
                prefill_id,
                body,
                route_hint,
                is_stream,
                worker_manager,
                event_manager,
                event_generator,
                default_max_tokens=default_max_tokens,
            )
        else:
            decode_direct = _should_use_pd_decode_direct(
                decode_worker_id=decode_id,
                route_hint=route_hint,
                cfg=cfg,
                kvcache_manager=kvcache_manager,
            )
            admission_ticket_id = router.take_pd_decode_admission_ticket(
                req_id,
                decode_id,
            )
            if decode_direct["enabled"] and admission_ticket_id:
                worker_manager.release_kvbm_decode_admission_ticket(
                    decode_id,
                    handoff_id=f"pd-{req_id}",
                    ticket_id=admission_ticket_id,
                )
                admission_ticket_id = ""
            result = _build_disagg_result(
                prefill_id,
                decode_id,
                body,
                route_hint,
                is_stream,
                worker_manager,
                cfg,
                decode_direct=decode_direct,
                admission_ticket_id=admission_ticket_id,
            )
            if decode_direct["enabled"]:
                # route_to_workers 已为 P/D 同时占用并发额度。直达 D 不会请求 P，
                # 必须立即释放 P，完成回调只再释放实际承载请求的 D。
                router.release_worker(prefill_id, request_id=req_id)
                _request_workers[req_id] = [decode_id]
                kvbm_timing = {"phase_ms": {}}
            else:
                _request_workers[req_id] = [prefill_id, decode_id]
                pd_handoff_ready = bool(
                    result.get("pd_kvbm_enabled")
                    and result.get("pd_prefill_control_url")
                    and result.get("pd_decode_control_url")
                )
                if pd_handoff_ready:
                    # 专用 PD 事务会根据 P export 在 D 端执行 delta plan、HIXL
                    # pull 和 request attach。此处再做通用跨 Worker prefetch 会
                    # 重复搬运，并阻塞 Rust 提前派发 D 请求。
                    kvbm_timing = {"phase_ms": {}}
                    logger.info(
                        "KVBM generic prefetch skipped for dedicated PD handoff request={} prefill={} decode={}",
                        req_id,
                        prefill_id,
                        decode_id,
                    )
                else:
                    prefill_timing = _prefetch_kvbm_blocks_for_worker(
                        worker_id=prefill_id,
                        route_hint=route_hint,
                        request_id=req_id,
                        cfg=cfg,
                        kvcache_manager=kvcache_manager,
                        worker_manager=worker_manager,
                        event_loop=event_loop,
                    )
                    decode_timing = _prefetch_kvbm_blocks_for_worker(
                        worker_id=decode_id,
                        route_hint=route_hint,
                        request_id=req_id,
                        cfg=cfg,
                        kvcache_manager=kvcache_manager,
                        worker_manager=worker_manager,
                        event_loop=event_loop,
                    )
                    kvbm_timing = decode_timing if decode_timing.get("phase_ms") else prefill_timing

        route_finished = time.perf_counter()
        route_build_ms = (route_finished - route_build_started) * 1000.0
        router_route_ms = max((route_finished - route_started) * 1000.0, 0.0)
        measured_route_ms = request_parse_ms + preprocessor_ms + worker_selection_ms + route_build_ms
        route_unaccounted_ms = max(router_route_ms - measured_route_ms, 0.0)
        phase_ms = dict(kvbm_timing.get("phase_ms", {}))
        transfer_finished_at = kvbm_timing.get("transfer_finished_at")
        phase_ms["router_request_parse_ms"] = max(request_parse_ms, 0.0)
        phase_ms["router_preprocessor_ms"] = max(preprocessor_ms, 0.0)
        phase_ms["router_worker_selection_ms"] = max(worker_selection_ms, 0.0)
        phase_ms["router_route_attempt_ms"] = max(route_attempt_ms, 0.0)
        phase_ms["router_capacity_wait_ms"] = max(capacity_wait_ms, 0.0)
        phase_ms["router_capacity_turn_wait_ms"] = max(
            capacity_turn_wait_ms,
            0.0,
        )
        phase_ms["pd_pending_prefix_wait_ms"] = max(
            pd_pending_prefix_wait_ms,
            0.0,
        )
        phase_ms["router_route_build_ms"] = max(route_build_ms, 0.0)
        phase_ms["router_route_unaccounted_ms"] = route_unaccounted_ms
        phase_ms["router_route_attempt_count"] = route_attempt_count
        phase_ms["router_route_ms"] = router_route_ms
        phase_ms["router_post_transfer_ms"] = (
            max(
                (route_finished - transfer_finished_at) * 1000.0,
                0.0,
            )
            if transfer_finished_at is not None
            else 0.0
        )
        request_context["router_phase_ms"] = phase_ms
        logger.info(
            "[RouterRoutePhaseTiming] request={} parse={:.2f}ms "
            "preprocessor={:.2f}ms worker_selection={:.2f}ms "
            "route_attempt={:.2f}ms capacity_wait={:.2f}ms "
            "route_build={:.2f}ms unaccounted={:.2f}ms attempts={} "
            "total={:.2f}ms",
            req_id,
            request_parse_ms,
            preprocessor_ms,
            worker_selection_ms,
            route_attempt_ms,
            capacity_wait_ms,
            route_build_ms,
            route_unaccounted_ms,
            route_attempt_count,
            router_route_ms,
        )
        logger.info(
            "[KVBMRouterTiming] request={} stage=route target={} phase_ms={}",
            req_id,
            decode_id or prefill_id,
            phase_ms,
        )
        return result

    return route_fn


def _get_prefix_match_worker_ids(
    *,
    worker_manager: WorkerManager,
    model: str,
    include_decode_workers: bool,
) -> set[str]:
    """获取一次 Prefix-aware 匹配需要查询的 Worker 集合。

    Args:
        worker_manager: Worker 注册与健康状态管理器。
        model: 当前请求使用的模型名称。
        include_decode_workers: PD 模式下是否同时查询 D Worker。HIXL 的
            decode-direct、location scorer 等功能需要开启；Mooncake 对照测试可关闭，
            以恢复只查询 P Worker 的旧路由语义。

    Returns:
        需要参与前缀匹配的健康 Worker ID 集合。PD 模式至少包含 P Worker；
        非 PD 模式返回健康的 combined Worker。
    """
    prefill_workers = worker_manager.get_all_healthy_prefill_workers(model)
    if prefill_workers:
        workers = list(prefill_workers)
        if include_decode_workers:
            decode_workers = worker_manager.get_all_healthy_decode_workers(model)
            if isinstance(decode_workers, list | tuple):
                workers.extend(decode_workers)
        return {worker.worker_id for worker in workers}

    combined_workers = worker_manager.get_all_healthy_combined_workers(model)
    return {worker.worker_id for worker in combined_workers}


def _should_use_pd_decode_direct(
    *,
    decode_worker_id: str,
    route_hint,
    cfg,
    kvcache_manager: KVCacheManager,
) -> dict[str, object]:
    """判断 PD 请求是否可跳过 P/handoff 并直接发送到 D Worker。

    参数:
        decode_worker_id: Router 已选中的 D Worker ID。
        route_hint: 当前请求路由提示，包含完整 prompt token IDs。
        cfg: Router 配置，提供开关、最少 NPU token 和命中比例阈值。
        kvcache_manager: Router KVCacheManager，用于读取 D 的连续前缀快照。

    返回:
        包含 enabled、reason、npu_prefix_tokens、prompt_tokens、ratio 和阈值的
        诊断字典。只有 D 从第一个 block 开始连续位于 NPU 且同时满足两个阈值
        时 enabled=True。
    """

    token_ids = list(getattr(route_hint, "token_ids", None) or [])
    enabled_by_config = bool(
        getattr(cfg, "kvbm_enabled", False)
        and getattr(cfg, "kvbm_router_pd_enabled", False)
        and getattr(cfg, "kvbm_router_pd_decode_direct_enabled", False)
    )
    min_tokens = max(
        int(getattr(cfg, "kvbm_router_pd_decode_direct_min_npu_tokens", 1024)),
        0,
    )
    min_ratio = min(
        max(float(getattr(cfg, "kvbm_router_pd_decode_direct_min_ratio", 0.8)), 0.0),
        1.0,
    )
    npu_prefix_tokens = 0
    if enabled_by_config and token_ids:
        snapshots = getattr(route_hint, "precomputed_prefix_snapshots", None) or {}
        snapshot = snapshots.get(decode_worker_id)
        if snapshot is None:
            snapshot = kvcache_manager.get_worker_prefix_snapshot(
                decode_worker_id,
                token_ids,
            )
        for block in snapshot.blocks:
            alias = block.select({CacheLocation.NPU})
            if alias is None:
                break
            remaining = max(0, len(token_ids) - npu_prefix_tokens)
            npu_prefix_tokens += min(max(int(alias.token_count), 0), remaining)
    ratio = npu_prefix_tokens / len(token_ids) if token_ids else 0.0
    direct_enabled = enabled_by_config and npu_prefix_tokens >= min_tokens and ratio >= min_ratio
    if not enabled_by_config:
        reason = "disabled"
    elif not token_ids:
        reason = "missing_prompt_tokens"
    elif npu_prefix_tokens < min_tokens:
        reason = "below_min_npu_tokens"
    elif ratio < min_ratio:
        reason = "below_min_ratio"
    else:
        reason = "target_npu_prefix_sufficient"
    decision = {
        "enabled": direct_enabled,
        "reason": reason,
        "npu_prefix_tokens": npu_prefix_tokens,
        "prompt_tokens": len(token_ids),
        "ratio": round(ratio, 6),
        "min_npu_tokens": min_tokens,
        "min_ratio": min_ratio,
    }
    logger.info(
        "[KVBMPDDecodeDirectDecision] decode={} decision={}",
        decode_worker_id,
        decision,
    )
    return decision


def _prefetch_kvbm_blocks_for_worker(
    *,
    worker_id: str,
    route_hint,
    request_id: str,
    cfg,
    kvcache_manager: KVCacheManager,
    worker_manager: WorkerManager,
    event_loop: asyncio.AbstractEventLoop | None,
) -> dict[str, object]:
    """在 data plane 同步路由回调中触发 worker KVBM 预取。

    参数:
        worker_id: 目标 worker ID。
        route_hint: 预处理得到的路由提示，包含 token_ids。
        request_id: 当前请求 ID，用于 worker 侧日志串联。
        cfg: router 全局配置对象。
        kvcache_manager: KV 缓存管理器，用于读取命中的 block_hashes。
        worker_manager: Worker 管理器，用于调用 worker 预取控制接口。
        event_loop: 后台 asyncio 事件循环；为空或未运行时跳过预取。

    返回:
        返回 Router 传输阶段耗时及传输完成时间；预取失败只记录日志，不影响
        主请求路由结果。
    """

    if not getattr(route_hint, "token_ids", None) or event_loop is None or not event_loop.is_running():
        return {"phase_ms": {}}

    timing: dict[str, object] = {"phase_ms": {}}
    target_prefix_snapshot = None
    request_snapshots = getattr(route_hint, "precomputed_prefix_snapshots", None) or {}
    target_prefix_snapshot = request_snapshots.get(worker_id)

    if getattr(cfg, "kvbm_router_transfer_enabled", False):
        transfer_submitted_at = time.perf_counter()
        transfer_future = asyncio.run_coroutine_threadsafe(
            orchestrate_cross_worker_transfer(
                target_worker_id=worker_id,
                token_ids=route_hint.token_ids,
                request_id=request_id,
                config=cfg,
                kvcache_manager=kvcache_manager,
                worker_manager=worker_manager,
                submitted_at=transfer_submitted_at,
            ),
            event_loop,
        )
        transfer_wait_ms = (
            max(int(getattr(cfg, "kvbm_router_link_timeout_ms", 3000)), 1)
            + max(int(getattr(cfg, "kvbm_router_transfer_timeout_ms", 3000)), 1)
            + 1000
        )
        try:
            transfer_result = transfer_future.result(timeout=transfer_wait_ms / 1000.0)
            target_prefix_snapshot = transfer_result.pop(
                "_target_prefix_snapshot",
                None,
            )
            timing["phase_ms"] = dict(transfer_result.get("phase_ms", {}))
            timing["transfer_finished_at"] = time.perf_counter()
            logger.info(
                "KVBM data-plane cross-worker transfer request={} source={} target={} result={}",
                request_id,
                transfer_result.get("source_worker_id", ""),
                worker_id,
                transfer_result,
            )
        except Exception as exc:
            timing["transfer_finished_at"] = time.perf_counter()
            transfer_future.cancel()
            logger.warning(
                "KVBM data-plane cross-worker transfer failed request={} target={} error={}; falling back to recompute",
                request_id,
                worker_id,
                exc,
            )

    if not getattr(cfg, "kvbm_router_prefetch_enabled", True):
        return timing

    if target_prefix_snapshot is None:
        target_prefix_snapshot = kvcache_manager.get_worker_prefix_snapshot(
            worker_id,
            route_hint.token_ids,
        )
    limit = int(getattr(cfg, "kvbm_router_prefetch_max_blocks", 64))
    location_counts = kvcache_manager.get_matched_block_location_counts(
        worker_id,
        route_hint.token_ids,
        snapshot=target_prefix_snapshot,
    )
    cpu_remote_matches = location_counts.get(CacheLocation.CPU, 0) + location_counts.get(CacheLocation.REMOTE, 0)
    npu_matches = location_counts.get(CacheLocation.NPU, 0)
    block_hashes = kvcache_manager.get_matched_block_hashes(
        worker_id,
        route_hint.token_ids,
        limit=limit,
        locations={CacheLocation.CPU, CacheLocation.REMOTE},
        snapshot=target_prefix_snapshot,
    )
    logger.info(
        "KVBM data-plane prefetch candidates worker={} cpu_remote_matches={} npu_matches={} requested_blocks={}",
        worker_id,
        cpu_remote_matches,
        npu_matches,
        len(block_hashes),
    )
    if not block_hashes:
        logger.debug("KVBM data-plane prefetch skipped: worker={} matched_blocks=0", worker_id)
        return timing

    deadline_ms = int(getattr(cfg, "kvbm_router_prefetch_timeout_ms", 100))
    future = asyncio.run_coroutine_threadsafe(
        worker_manager.prefetch_kv_blocks(
            worker_id,
            block_hashes,
            request_id=request_id,
            deadline_ms=deadline_ms,
        ),
        event_loop,
    )
    if not getattr(cfg, "kvbm_router_prefetch_wait_for_result", False):
        future.add_done_callback(
            lambda done: logger.debug(
                "KVBM data-plane prefetch task finished: worker={} result={}",
                worker_id,
                _prefetch_future_result(done),
            )
        )
        logger.info(
            "KVBM data-plane prefetch dispatched worker={} requested_blocks={}",
            worker_id,
            len(block_hashes),
        )
        return timing

    try:
        result = future.result(timeout=(max(deadline_ms, 1) + 200) / 1000.0)
        logger.info(
            "KVBM data-plane prefetch worker={} requested_blocks={} result={}",
            worker_id,
            len(block_hashes),
            result,
        )
    except Exception as exc:
        future.cancel()
        logger.debug(
            "KVBM data-plane prefetch skipped: worker={} blocks={} error={}",
            worker_id,
            len(block_hashes),
            exc,
        )
    return timing


def _prefetch_future_result(future) -> object:
    """读取 data plane 后台预取 future 结果，避免异常未被消费。

    参数:
        future: asyncio.run_coroutine_threadsafe 返回的 Future。

    返回:
        Future 返回值；取消时返回字符串标记；异常时返回异常字符串。
    """

    if future.cancelled():
        return "cancelled"
    try:
        return future.result()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _build_combined_result(
    worker_id, body, route_hint, is_stream, worker_manager, event_manager, event_generator, default_max_tokens=None
):
    """构建组合模式（prefill + decode 同节点）的路由结果。

    生成 KV 缓存事件（如启用），清理请求体中的 jiuwenext 字段，
    返回包含目标 URL、请求体、流式标志等信息的 dict 供 Rust 代理转发。

    """
    worker = worker_manager.get_worker(worker_id)
    if not worker:
        raise ValueError(f"Worker {worker_id} not found")

    if event_generator and route_hint.token_ids:
        events = event_generator.generate_events(worker_id, route_hint.token_ids)
        for event in events:
            event_manager.process_event(event)

    if event_generator:
        max_tokens = body.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = default_max_tokens or route_hint.estimated_output_tokens
        jiuwenext = body.get("jiuwenext", {})
        hints = jiuwenext.get("agent_hints", {}) if jiuwenext else {}
        estimated = hints.get("estimated_output_tokens", max_tokens)
        event_manager.process_event(
            CacheEvent(
                event_type="decode_start",
                token_count=len(route_hint.token_ids) if route_hint.token_ids else 0,
                worker_id=worker_id,
                engine_specific={"worker_id": worker_id, "osl": estimated},
            )
        )

    sanitized = {k: v for k, v in body.items() if k != "jiuwenext"}

    return {
        "target_url": worker.url,
        "worker_id": worker_id,
        "headers": {},
        "body_json": json.dumps(sanitized, ensure_ascii=False),
        "request_id": route_hint.request_id or f"req-{uuid.uuid4()}",
        "stream": is_stream,
    }


def _pd_engine_request_id(forwarded_request_id: str, body: dict) -> str:
    """计算 Worker 最终传给 vLLM engine-core 的 PD 请求 ID。

    参数:
        forwarded_request_id: Router 通过 `X-Request-Id` 发送的原始请求 ID。
        body: OpenAI 请求体，可能包含 cache_sharing 和 cache_salt。

    返回:
        与 Worker `_make_engine_request_id` 完全一致的 engine request ID；未启用
        cache sharing 时原样返回。
    """

    cache_salt = body.get("cache_salt")
    if body.get("cache_sharing") and cache_salt:
        return json.dumps(
            {
                "request_id": forwarded_request_id,
                "sharing_cache_salt": cache_salt,
            }
        )
    return forwarded_request_id


def _build_disagg_result(
    prefill_id,
    decode_id,
    body,
    route_hint,
    is_stream,
    worker_manager,
    cfg,
    *,
    decode_direct: dict[str, object] | None = None,
    admission_ticket_id: str = "",
):
    """构建分离模式（prefill 和 decode 在不同节点）的路由结果。

    设置 KV 传输所需的 header（prefill/decode 地址、dp_rank），
    Rust 代理先发 prefill 请求（max_tokens=1, stream=False）等待完成，
    再发 decode 请求（原始 max_tokens/stream）并将 decode 响应返回客户端。

    参数:
        prefill_id: Router 选中的 P Worker ID。
        decode_id: Router 选中的 D Worker ID。
        body: 原始 OpenAI 请求体。
        route_hint: 预处理后的请求提示，包含 request ID 和 token IDs。
        is_stream: 是否按流式响应转发。
        worker_manager: Worker 注册信息管理器。
        cfg: Router 全局配置。
        decode_direct: 可选 D 直达决策及诊断字段。

    返回:
        Rust data-plane 所需的推理地址、请求体及 KVBM PD 控制元数据。

    """
    prefill_worker = worker_manager.get_worker(prefill_id)
    decode_worker = worker_manager.get_worker(decode_id)
    if not prefill_worker or not decode_worker:
        raise ValueError(f"Workers not found: prefill={prefill_id}, decode={decode_id}")

    prefill_kv_addr = getattr(prefill_worker, "kv_addr", "")
    decode_kv_addr = getattr(decode_worker, "kv_addr", "")
    dp_rank = getattr(prefill_worker, "dp_rank", 0)

    router_request_id = route_hint.request_id or f"req-{uuid.uuid4()}"
    request_id_inner = f"___prefill_addr_{prefill_kv_addr}___decode_addr_{decode_kv_addr}_{uuid.uuid4().hex}"
    pd_request_id = _pd_engine_request_id(request_id_inner, body)
    headers = {
        "X-Request-Id": request_id_inner,
        "X-Jiuwen-Request-Id": router_request_id,
    }
    kv_target = os.environ.get("KV_TARGET")
    headers["X-KV-Target"] = kv_target if kv_target else decode_worker.url
    headers["X-data-parallel-rank"] = str(dp_rank)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    sanitized = {k: v for k, v in body.items() if k != "jiuwenext"}

    prefill_body = copy.deepcopy(sanitized)
    prefill_body["max_tokens"] = 1
    prefill_body["stream"] = False
    if "max_completion_tokens" in prefill_body:
        prefill_body["max_completion_tokens"] = 1
    if "stream_options" in prefill_body:
        del prefill_body["stream_options"]

    decode_body = copy.deepcopy(sanitized)
    direct_decision = dict(decode_direct or {})
    decode_direct_enabled = bool(direct_decision.get("enabled", False))
    if decode_direct_enabled:
        headers["X-Request-Id"] = router_request_id
        headers.pop("X-KV-Target", None)
        pd_request_id = _pd_engine_request_id(router_request_id, body)
    pd_enabled = bool(getattr(cfg, "kvbm_enabled", False) and getattr(cfg, "kvbm_router_pd_enabled", False))
    if not pd_enabled and not decode_direct_enabled:
        # Native KVConnector/Mooncake uses the P response as the transfer
        # descriptor for the later D request. HIXL owns a separate control path.
        prefill_body["min_tokens"] = 1
        prefill_body["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
    prefill_control_url = worker_manager.worker_control_url(prefill_worker)
    decode_control_url = worker_manager.worker_control_url(decode_worker)
    logger.info(
        "[KVBMPDRouteDecision] request={} enabled={} decode_direct={} prefill={} "
        "decode={} prefill_control_url={} prefill_control_source={} "
        "decode_control_url={} decode_control_source={} prompt_tokens={} "
        "timeout_ms={} export_timeout_ms={} direct_decision={}",
        router_request_id,
        pd_enabled,
        decode_direct_enabled,
        prefill_id,
        decode_id,
        prefill_control_url,
        "configured" if prefill_worker.kvbm_control_url.strip() else "derived_port_offset",
        decode_control_url,
        "configured" if decode_worker.kvbm_control_url.strip() else "derived_port_offset",
        len(route_hint.token_ids or []),
        max(int(getattr(cfg, "kvbm_router_pd_timeout_ms", 10000)), 1),
        max(int(getattr(cfg, "kvbm_router_pd_export_timeout_ms", 120000)), 1),
        direct_decision,
    )

    return {
        "target_url": prefill_worker.url,
        "worker_id": prefill_id,
        "headers": headers,
        "body_json": json.dumps(prefill_body, ensure_ascii=False),
        "request_id": router_request_id,
        "stream": is_stream,
        "disagg": True,
        "decode_target_url": decode_worker.url,
        "decode_worker_id": decode_id,
        "decode_body_json": json.dumps(decode_body, ensure_ascii=False),
        "pd_kvbm_enabled": pd_enabled,
        "pd_decode_direct": decode_direct_enabled,
        "pd_handoff_id": f"pd-{router_request_id}",
        "pd_decode_admission_ticket_id": admission_ticket_id,
        "pd_request_id": pd_request_id,
        "pd_prefill_control_url": prefill_control_url,
        "pd_decode_control_url": decode_control_url,
        "pd_prompt_token_count": len(route_hint.token_ids or []),
        "pd_timeout_ms": max(int(getattr(cfg, "kvbm_router_pd_timeout_ms", 10000)), 1),
        "pd_export_timeout_ms": max(
            int(getattr(cfg, "kvbm_router_pd_export_timeout_ms", 120000)),
            1,
        ),
    }


@dataclass(frozen=True)
class _CompletionEvent:
    """Named completion data; field order preserves the Rust positional callback ABI."""

    request_id: str
    worker_id: str
    ttft_s: float
    total_s: float
    completion_tokens: int = 0
    prompt_tokens: int = 0
    worker_dispatch_to_first_byte_s: float = 0.0
    router_to_early_d_dispatch_s: float = 0.0
    d_request_seen_to_attach_start_s: float = 0.0
    d_attach_complete_to_first_byte_s: float = 0.0
    pd_diagnostics_json: str = "{}"


def _build_complete_fn(
    router: Router,
    worker_manager: WorkerManager,
    request_workers: dict,
    request_contexts: dict,
    event_loop: asyncio.AbstractEventLoop | None = None,
):
    """构建请求完成回调，供 Rust 代理在请求结束时调用。

    释放该请求占用的所有 worker 的 inflight 计数，通过 Condition 唤醒排队线程，
    并上报延迟预测训练数据（若配置了 latency-predict-scorer）。

    """

    def handle_completion(event: _CompletionEvent) -> None:
        """释放请求资源并记录端到端 Router/Worker 耗时。

        参数:
            event: 请求标识、token 计数及 Rust 数据面上报的各阶段秒数和诊断 JSON。

        返回:
            无返回值。
        """

        request_id = event.request_id
        worker_id = event.worker_id
        ttft_s = event.ttft_s
        total_s = event.total_s
        completion_tokens = event.completion_tokens
        prompt_tokens = event.prompt_tokens
        worker_dispatch_to_first_byte_s = event.worker_dispatch_to_first_byte_s
        router_to_early_d_dispatch_s = event.router_to_early_d_dispatch_s
        d_request_seen_to_attach_start_s = event.d_request_seen_to_attach_start_s
        d_attach_complete_to_first_byte_s = event.d_attach_complete_to_first_byte_s
        pd_diagnostics_json = event.pd_diagnostics_json

        workers = request_workers.pop(request_id, None)
        if workers:
            for wid in workers:
                router.release_worker(wid, request_id=request_id)
        elif worker_id:
            router.release_worker(worker_id, request_id=request_id)

        ctx = request_contexts.pop(request_id, None)
        if prompt_tokens <= 0:
            prompt_tokens = ctx.get("prompt_tokens", 0) if ctx else 0
        total_tokens = prompt_tokens + completion_tokens
        overlap_scores = ctx.get("overlap_scores") if ctx else None
        router_phase_ms = dict(ctx.get("router_phase_ms", {})) if ctx else {}
        client_ttft_ms = max(ttft_s * 1000.0, 0.0)
        router_phase_ms["worker_dispatch_to_first_byte_ms"] = max(
            worker_dispatch_to_first_byte_s * 1000.0,
            0.0,
        )
        router_phase_ms["router_to_early_d_dispatch_ms"] = max(
            router_to_early_d_dispatch_s * 1000.0,
            0.0,
        )
        router_phase_ms["d_request_seen_to_attach_start_ms"] = max(
            d_request_seen_to_attach_start_s * 1000.0,
            0.0,
        )
        router_phase_ms["d_attach_complete_to_first_byte_ms"] = max(
            d_attach_complete_to_first_byte_s * 1000.0,
            0.0,
        )
        router_phase_ms["client_ttft_ms"] = client_ttft_ms
        try:
            pd_diagnostics = json.loads(pd_diagnostics_json or "{}")
        except (TypeError, ValueError):
            pd_diagnostics = {}
        if not isinstance(pd_diagnostics, dict):
            pd_diagnostics = {}
        completion_intervals = max(completion_tokens - 1, 0)
        tpot_ms = max(total_s - ttft_s, 0.0) * 1000.0 / completion_intervals if completion_intervals > 0 else 0.0
        if worker_id and completion_intervals > 0 and total_s > ttft_s:
            router.record_decode_tpot(worker_id, tpot_ms)
        if pd_diagnostics:
            logger.info(
                "[KVBMPDRequestSummary] request_index={} request={} target={} "
                "free_blocks={} p_request_ms={:.3f} p_http_dispatch_ms={:.3f} "
                "p_prepare_to_http_dispatch_ms={:.3f} "
                "p_dispatch_to_scheduler_seen_ms={:.3f} "
                "p_worker_request_received_to_scheduler_seen_ms={:.3f} "
                "p_worker_request_received_timing_source={} "
                "p_scheduler_queue_cycles={} p_scheduler_seen_cycle_id={} "
                "p_first_scheduled_cycle_id={} p_kv_ready_cycle_id={} "
                "p_scheduler_compute_cycles={} p_first_scheduled_tokens={} "
                "p_scheduler_running_requests={} "
                "p_scheduler_waiting_requests={} "
                "p_first_scheduled_running_requests={} "
                "p_first_scheduled_waiting_requests={} "
                "p_first_scheduled_cycle_ms={:.3f} "
                "p_scheduler_seen_to_kv_ready_ms={:.3f} "
                "p_kv_ready_to_export_ready_ms={:.3f} "
                "p_export_ready_to_http_response_ms={:.3f} "
                "p_response_network_ms={:.3f} "
                "p_prefill_admission_wait_ms={:.3f} delta_plan_ms={:.3f} "
                "delta_preplanned={} delta_plan_task_queue_ms={:.3f} "
                "delta_hash_validate_ms={:.3f} delta_reserve_ms={:.3f} "
                "delta_plan_execution_ms={:.3f} delta_preplan_age_ms={:.3f} "
                "delta_inflight_wait_ms={:.3f} delta_inflight_retries={} "
                "delta_capacity_wait_ms={:.3f} delta_capacity_retries={} "
                "import_executor_queue_ms={:.3f} "
                "reserve_ms={:.3f} hixl_pull_ms={:.3f} "
                "attach_task_queue_ms={:.3f} attach_execution_ms={:.3f} "
                "d_request_seen_to_attach_start_ms={:.3f} "
                "worker_dispatch_to_first_byte_ms={:.3f} ttft_ms={:.3f} "
                "total_ms={:.3f} completion_tokens={} tpot_ms={:.3f} "
                "transferred_blocks={} reused_blocks={}",
                int(ctx.get("request_index", 0)) if ctx else 0,
                request_id,
                worker_id,
                pd_diagnostics.get("free_blocks"),
                float(pd_diagnostics.get("p_request_ms", 0.0)),
                float(pd_diagnostics.get("p_http_dispatch_ms", 0.0)),
                float(pd_diagnostics.get("p_prepare_to_http_dispatch_ms", 0.0)),
                float(pd_diagnostics.get("p_dispatch_to_scheduler_seen_ms", 0.0)),
                float(
                    pd_diagnostics.get(
                        "p_worker_request_received_to_scheduler_seen_ms",
                        0.0,
                    )
                ),
                str(
                    pd_diagnostics.get(
                        "p_worker_request_received_timing_source",
                        "unavailable",
                    )
                ),
                int(pd_diagnostics.get("p_scheduler_queue_cycles", 0)),
                int(pd_diagnostics.get("p_scheduler_seen_cycle_id", 0)),
                int(pd_diagnostics.get("p_first_scheduled_cycle_id", 0)),
                int(pd_diagnostics.get("p_kv_ready_cycle_id", 0)),
                int(pd_diagnostics.get("p_scheduler_compute_cycles", 0)),
                int(pd_diagnostics.get("p_first_scheduled_token_count", 0)),
                int(pd_diagnostics.get("p_scheduler_running_request_count", 0)),
                int(pd_diagnostics.get("p_scheduler_waiting_request_count", 0)),
                int(
                    pd_diagnostics.get(
                        "p_first_scheduled_running_request_count",
                        0,
                    )
                ),
                int(
                    pd_diagnostics.get(
                        "p_first_scheduled_waiting_request_count",
                        0,
                    )
                ),
                float(pd_diagnostics.get("p_first_scheduled_cycle_ms", 0.0)),
                float(pd_diagnostics.get("p_scheduler_seen_to_kv_ready_ms", 0.0)),
                float(pd_diagnostics.get("p_kv_ready_to_export_ready_ms", 0.0)),
                float(pd_diagnostics.get("p_export_ready_to_http_response_ms", 0.0)),
                float(pd_diagnostics.get("p_response_network_ms", 0.0)),
                float(pd_diagnostics.get("p_prefill_admission_wait_ms", 0.0)),
                float(pd_diagnostics.get("delta_plan_ms", 0.0)),
                bool(pd_diagnostics.get("delta_preplanned", False)),
                float(pd_diagnostics.get("delta_plan_task_queue_ms", 0.0)),
                float(pd_diagnostics.get("delta_hash_validate_ms", 0.0)),
                float(pd_diagnostics.get("delta_reserve_ms", 0.0)),
                float(pd_diagnostics.get("delta_plan_execution_ms", 0.0)),
                float(pd_diagnostics.get("delta_preplan_age_ms", 0.0)),
                float(pd_diagnostics.get("delta_inflight_wait_ms", 0.0)),
                int(pd_diagnostics.get("delta_inflight_retry_count", 0)),
                float(pd_diagnostics.get("delta_capacity_wait_ms", 0.0)),
                int(pd_diagnostics.get("delta_capacity_retry_count", 0)),
                float(pd_diagnostics.get("import_executor_queue_ms", 0.0)),
                float(pd_diagnostics.get("reserve_ms", 0.0)),
                float(pd_diagnostics.get("hixl_pull_ms", 0.0)),
                float(pd_diagnostics.get("attach_task_queue_ms", 0.0)),
                float(pd_diagnostics.get("attach_execution_ms", 0.0)),
                d_request_seen_to_attach_start_s * 1000.0,
                worker_dispatch_to_first_byte_s * 1000.0,
                ttft_s * 1000.0,
                total_s * 1000.0,
                completion_tokens,
                tpot_ms,
                int(pd_diagnostics.get("transferred_blocks", 0)),
                int(pd_diagnostics.get("reused_blocks", 0)),
            )
        logger.info(
            "[Complete] {} workers={} ttft={:.1f}ms total={:.1f}ms tokens={}/{}/{} overlap={}",
            request_id,
            workers or [worker_id],
            ttft_s * 1000,
            total_s * 1000,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            overlap_scores,
        )
        logger.info(
            "[KVBMRouterTiming] request={} stage=complete target={} phase_ms={}",
            request_id,
            worker_id,
            router_phase_ms,
        )
        if ctx:
            _report_latency_training_data(
                router=router,
                worker_manager=worker_manager,
                prefill_id=ctx["prefill_id"],
                decode_id=ctx["decode_id"],
                route_hint=ctx["route_hint"],
                overlap_scores=ctx["overlap_scores"],
                ttft_s=ttft_s,
                total_s=total_s,
                completion_tokens=completion_tokens,
                event_loop=event_loop,
            )

    def complete_fn(*args, **kwargs) -> None:
        """Adapt the existing Rust callback ABI to a validated named completion event."""
        handle_completion(_CompletionEvent(*args, **kwargs))

    return complete_fn


def _build_pd_prefill_complete_fn(router: Router, request_workers: dict):
    """构建 P 阶段完成回调，供 Rust 数据面提前归还 Prefill 预算。

    参数:
        router: Router 实例，负责释放 inflight 与动态 token/block 预算。
        request_workers: 请求 ID 到实际 Worker 列表的共享映射。

    返回:
        可由 Rust 调用的 ``prefill_complete_fn(request_id, worker_id)``。
    """

    def prefill_complete_fn(request_id: str, worker_id: str) -> None:
        """在 P export-ready 或原生 Connector 的 P 响应完成时释放 P 预算。"""
        workers = request_workers.get(request_id)
        if not workers or worker_id not in workers:
            return
        workers.remove(worker_id)
        router.release_pd_prefill(worker_id, request_id)
        logger.debug(
            "[PDPrefillAdmission] request={} worker={} phase budget released",
            request_id,
            worker_id,
        )

    return prefill_complete_fn


def _build_pd_dispatch_snapshot_fn(
    router: Router,
    worker_manager: WorkerManager,
    request_contexts: dict,
):
    """构建原生Connector在D派发前读取最新负载的诊断回调。"""

    def dispatch_snapshot_fn(request_id: str, worker_id: str) -> str:
        context = request_contexts.get(request_id) or {}
        selected = context.get("decode_selection_snapshot") or {}
        selected_at = float(selected.get("selected_at_monotonic") or 0.0)
        selection_to_dispatch_ms = max((time.monotonic() - selected_at) * 1000.0, 0.0) if selected_at > 0.0 else 0.0
        worker = worker_manager.get_worker(worker_id)
        return json.dumps(
            {
                "selection_to_dispatch_ms": selection_to_dispatch_ms,
                "selection_inflight": int(selected.get("inflight") or 0),
                "selection_running": int(selected.get("running") or 0),
                "selection_waiting": int(selected.get("waiting") or 0),
                "dispatch_inflight": router.inflight.get(worker_id),
                "dispatch_running": int(getattr(worker, "running_requests", 0) or 0),
                "dispatch_waiting": int(getattr(worker, "queue_depth", 0) or 0),
            },
            separators=(",", ":"),
        )

    return dispatch_snapshot_fn


def _build_pd_decode_budget_fn(
    router: Router,
    request_workers: dict,
    request_contexts: dict,
    kvcache_manager: KVCacheManager,
):
    """构建D分配/attach/首字节预算校正及Router前缀发布回调。

    参数:
        router: Router 实例，负责收缩请求的 D block 预算。
        request_workers: 请求 ID 到实际 Worker 列表的共享映射。
        request_contexts: 请求 ID 到预处理上下文的共享映射，用于读取 prompt
            token IDs。
        kvcache_manager: Router KV 缓存管理器，用于立即发布 D Worker 的 NPU
            前缀位置。

    返回:
        可由 Rust 调用的 ``decode_budget_fn(request_id, worker_id, phase,
        metadata_json)``。
    """

    def decode_budget_fn(
        request_id: str,
        worker_id: str,
        phase: str,
        metadata_json: str = "",
    ) -> None:
        """确认D预算物化，并在attach成功后发布完整NPU前缀。

        参数:
            request_id: Router 对外请求 ID。
            worker_id: 完成 attach 或开始输出的 D Worker ID。
            phase: 回调阶段，当前为 ``allocated``、``fallback``、
                ``attached`` 或 ``first_byte``。
            metadata_json: allocated阶段携带目标block数量；attach阶段携带
                P Worker、完整block hashes、D物理block IDs和block size JSON；
                其他阶段为空对象。

        返回:
            无返回值。元数据缺失时仍执行预算校正，等待普通 KV event 收敛索引。
        """
        workers = request_workers.get(request_id)
        if not workers or worker_id not in workers:
            return
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except (TypeError, ValueError):
            metadata = {}
        raw_generation = metadata.get("capacity_generation")
        capacity_generation = int(raw_generation) if raw_generation is not None else None
        if phase == "allocated":
            if capacity_generation is None:
                router.mark_pd_decode_blocks_allocated(worker_id, request_id)
            else:
                router.mark_pd_decode_blocks_allocated(
                    worker_id,
                    request_id,
                    capacity_generation,
                )
        elif phase == "fallback":
            router.restore_pd_decode_budget_after_fallback(worker_id, request_id)
            mark_failed = getattr(router, "mark_pd_prefix_failed", None)
            if callable(mark_failed):
                mark_failed(request_id, worker_id)
            return
        elif phase == "attached":
            # 兼容尚未发布allocated事件的旧Worker；attach仍可启动容量确认。
            if capacity_generation is None:
                router.reconcile_pd_decode_budget(worker_id, request_id)
            else:
                router.reconcile_pd_decode_budget(
                    worker_id,
                    request_id,
                    capacity_generation,
                )
        else:
            # first_byte不是block attach证明；fallback后尤其不能据此收缩预算。
            return
        if phase == "attached":
            published = 0
            try:
                if metadata_json:
                    block_hashes = list(metadata.get("block_hashes") or [])
                    destination_block_ids = list(metadata.get("destination_block_ids") or [])
                    context = request_contexts.get(request_id) or {}
                    route_hint = context.get("route_hint")
                    token_ids = list(getattr(route_hint, "token_ids", None) or [])
                    block_size = metadata.get("block_size")
                    if not block_size and block_hashes:
                        partial_valid_tokens = int(metadata.get("partial_valid_tokens") or 0)
                        complete_token_count = len(token_ids) - partial_valid_tokens
                        if complete_token_count > 0:
                            block_size = complete_token_count // len(block_hashes)
                    published = kvcache_manager.record_pd_import(
                        worker_id,
                        block_hashes,
                        destination_block_ids,
                        token_ids,
                        block_size=block_size,
                    )
                    logger.info(
                        "KVBM PD D prefix published to Router index: request={} "
                        "source={} target={} requested_blocks={} published_blocks={} "
                        "location=npu",
                        request_id,
                        metadata.get("source_worker_id", ""),
                        worker_id,
                        len(block_hashes),
                        published,
                    )
            except Exception as exc:
                logger.warning(
                    "KVBM PD D prefix Router index publish failed: request={} target={} error={}",
                    request_id,
                    worker_id,
                    exc,
                )
            mark_ready = getattr(router, "mark_pd_prefix_ready", None)
            if callable(mark_ready):
                mark_ready(request_id, worker_id, published)
        logger.debug(
            "[PDDecodeAdmission] request={} worker={} phase={} attach observed",
            request_id,
            worker_id,
            phase,
        )

    return decode_budget_fn


def _report_latency_training_data(
    *,
    router: Router,
    worker_manager: WorkerManager,
    prefill_id: str,
    decode_id: str | None,
    route_hint,
    overlap_scores: dict | None,
    ttft_s: float,
    total_s: float,
    completion_tokens: int = 0,
    event_loop: asyncio.AbstractEventLoop | None = None,
):
    """若当前算法配置了 latency-predict-scorer，异步上报训练样本到 training_server。"""
    from openjiuwentools.infer_router.routing.scorer import PluginScorerAlgorithm

    algo = router.get_algorithm_for_phase("decode") if decode_id is not None else router.current_algorithm
    if not isinstance(algo, PluginScorerAlgorithm):
        return

    reporter = algo.get_latency_predict_scorer()
    if reporter is None:
        return

    target_worker_id = prefill_id if decode_id is None else decode_id
    target_worker = worker_manager.get_worker(target_worker_id)
    if not target_worker:
        logger.warning("[LatencyTraining] Worker {} not found, skip reporting", target_worker_id)
        return

    actual_ttft_ms = ttft_s * 1000.0
    decode_s = total_s - ttft_s
    actual_tpot_ms = 0.0
    if completion_tokens > 0 and decode_s > 0:
        actual_tpot_ms = (decode_s * 1000.0) / completion_tokens

    logger.info(
        "[LatencyTraining] Reporting: worker={}, ttft={:.1f}ms, tpot={:.2f}ms, tokens={}",
        target_worker_id,
        actual_ttft_ms,
        actual_tpot_ms,
        completion_tokens,
    )
    reporter.report_training_data_fire_and_forget(
        loop=event_loop,
        worker=target_worker,
        route_hint=route_hint,
        overlap_scores=overlap_scores or {},
        actual_ttft_ms=actual_ttft_ms,
        actual_tpot_ms=actual_tpot_ms,
        completion_tokens=completion_tokens,
    )


async def _init_components(cfg):
    """初始化所有路由组件（与 api/server.py 的 create_app 相同，但不创建 FastAPI 应用）。

    组件初始化顺序：预处理器 → KV 缓存 → Worker 管理 → 调度器 → 路由器 → 事件订阅。
    启动 worker 发现、健康检查、metrics 服务器和性能统计。

    """
    set_global_settings(cfg)

    preprocessor = Preprocessor()
    event_manager = KVEventManager()

    event_generator = None
    if cfg.kv_event_mode == "inner_event":
        event_generator = KVEventGenerator()

    kvcache_manager = KVCacheManager(
        event_manager=event_manager,
        event_generator=event_generator,
        enable_radix_tree=cfg.kv_cache_enable_radix_tree,
    )
    worker_manager = WorkerManager(kv_cache_manager=kvcache_manager)
    workload_manager = WorkloadManager()
    worker_manager.set_workload_manager(workload_manager)
    worker_manager.register_event_handlers()

    scheduler = Scheduler(kvcache_manager)
    router = Router(kvcache_manager, worker_manager)

    event_subscriber = None
    if cfg.kv_event_mode == "worker_event":
        from openjiuwentools.infer_router.kv_cache.kv_event_handler import KVEventHandler
        from openjiuwentools.infer_router.worker.metrics_handler import MetricsHandler
        from openjiuwentools.infer_router.worker.worker_event_subscriber import WorkerEventSubscriber

        event_subscriber = WorkerEventSubscriber()
        event_subscriber.register_handler("kv_events", KVEventHandler(event_manager))
        event_subscriber.register_handler(
            "kv_metrics",
            MetricsHandler(worker_manager, on_metrics_updated=router.notify_metrics_updated),
        )
        worker_manager.on_worker_discovered(
            lambda w: event_subscriber.subscribe(w.worker_id, w.publisher_endpoint) if w.publisher_endpoint else None
        )

    await worker_manager.start()
    await _preload_router_tokenizers(
        preprocessor=preprocessor,
        worker_manager=worker_manager,
        enabled=bool(cfg.tokenizer_load_from_file),
    )
    metrics.start_metrics_server()
    await performance_stats.start()

    if event_subscriber is not None:
        for wid, worker in worker_manager.get_all_workers().items():
            if worker.publisher_endpoint:
                event_subscriber.subscribe(wid, worker.publisher_endpoint)

    return {
        "preprocessor": preprocessor,
        "kvcache_manager": kvcache_manager,
        "scheduler": scheduler,
        "router": router,
        "worker_manager": worker_manager,
        "event_manager": event_manager,
        "event_generator": event_generator,
        "event_subscriber": event_subscriber,
    }


async def _preload_router_tokenizers(
    *,
    preprocessor: Preprocessor,
    worker_manager: WorkerManager,
    enabled: bool,
) -> dict[str, bool]:
    """在 Rust data-plane 开始监听前预加载 Worker 模型 tokenizer。

    Args:
        preprocessor: Router 请求预处理器，持有 TokenizerManager。
        worker_manager: Worker 管理器，用于读取已发现 Worker 或发现源。
        enabled: 是否启用了本地 tokenizer 加载。

    Returns:
        模型名称到预加载结果的映射。未启用或尚未发现模型时返回空字典。
    """

    if not enabled:
        logger.info("Tokenizer preload skipped because tokenizer_load_from_file=false")
        return {}

    workers = list(worker_manager.get_all_workers().values())
    if not workers:
        discovery = getattr(worker_manager, "discovery", None)
        discover = getattr(discovery, "discover", None)
        if callable(discover):
            try:
                workers = list(await discover())
            except Exception as exc:
                logger.warning("Tokenizer preload worker discovery failed: {}", exc)

    models = sorted({str(getattr(worker, "model", "")) for worker in workers if str(getattr(worker, "model", ""))})
    if not models:
        logger.warning(
            "Tokenizer preload skipped because no Worker models were discovered; "
            "the first request may load its tokenizer lazily"
        )
        return {}

    started = time.perf_counter()
    results = preprocessor.preload_tokenizers(models)
    logger.info(
        "[TokenizerPreloadSummary] models={} ready={} failed={} total={:.2f}ms",
        models,
        sum(1 for ready in results.values() if ready),
        sum(1 for ready in results.values() if not ready),
        (time.perf_counter() - started) * 1000.0,
    )
    return results


def main():
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )

    parser = argparse.ArgumentParser(description="Jiuwen Router (Rust data plane + Python control plane)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--listen", type=str, default="0.0.0.0:8080", help="Data plane listen address")
    args = parser.parse_args()

    if args.config:
        os.environ.setdefault("CONFIG_PATH", args.config)

    cfg = create_settings(args.config)

    # Initialize components and keep the asyncio event loop running in a background
    # thread so worker discovery, health checks, and event subscribers stay alive.
    loop = asyncio.new_event_loop()
    components = loop.run_until_complete(_init_components(cfg))

    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    logger.info("asyncio event loop running in background thread (worker discovery, health checks, events)")

    request_workers = {}
    request_contexts = {}
    route_fn = _build_route_fn(
        preprocessor=components["preprocessor"],
        kvcache_manager=components["kvcache_manager"],
        scheduler=components["scheduler"],
        router=components["router"],
        worker_manager=components["worker_manager"],
        event_manager=components["event_manager"],
        event_generator=components["event_generator"],
        request_workers=request_workers,
        request_contexts=request_contexts,
        event_loop=loop,
    )
    complete_fn = _build_complete_fn(
        router=components["router"],
        worker_manager=components["worker_manager"],
        request_workers=request_workers,
        request_contexts=request_contexts,
        event_loop=loop,
    )
    pd_prefill_complete_fn = _build_pd_prefill_complete_fn(components["router"], request_workers)
    pd_dispatch_snapshot_fn = _build_pd_dispatch_snapshot_fn(
        components["router"],
        components["worker_manager"],
        request_contexts,
    )
    pd_decode_budget_fn = _build_pd_decode_budget_fn(
        components["router"],
        request_workers,
        request_contexts,
        components["kvcache_manager"],
    )

    logger.info(
        "Admission control: mode={}, max_concurrency_per_worker={}, headroom={}, "
        "saturation_ratio={}, queue_timeout={}s",
        cfg.admission_mode,
        cfg.max_concurrency_per_worker,
        cfg.concurrency_headroom,
        cfg.admission_saturation_ratio,
        cfg.router_queue_timeout,
    )
    logger.info("Starting Rust data plane proxy on {}", args.listen)

    from openjiuwentools.infer_router.data_plane import start_proxy

    pd_prefill_max_inflight = _resolve_pd_prefill_max_inflight(cfg)
    logger.info(
        "PD Prefill admission: mode={}, max_inflight_per_prefill_worker={}, "
        "block_budget_enabled={}, block_headroom_ratio={:.2f}",
        cfg.router_pd_prefill_admission_mode,
        pd_prefill_max_inflight,
        cfg.router_pd_prefill_block_admission_enabled,
        cfg.router_pd_prefill_block_headroom_ratio,
    )
    logger.info(
        "PD Decode admission: mode={}, max_inflight_per_decode_worker={}, "
        "block_budget_enabled={}, block_headroom_ratio={:.2f}, "
        "initial_window_blocks={}, capacity_metrics_ttl_seconds={:.1f}, "
        "capacity_confirmation_enabled={}, capacity_confirmation_timeout_ms={}, "
        "stale_capacity_confirmation_timeout_ms={}",
        cfg.router_pd_decode_admission_mode,
        cfg.router_pd_decode_max_inflight,
        cfg.router_pd_decode_block_admission_enabled,
        cfg.router_pd_decode_block_headroom_ratio,
        cfg.router_pd_decode_initial_window_blocks,
        cfg.router_pd_block_pool_metrics_ttl_seconds,
        cfg.router_pd_decode_capacity_confirmation_enabled,
        cfg.router_pd_decode_capacity_confirmation_timeout_ms,
        cfg.router_pd_decode_stale_capacity_confirmation_timeout_ms,
    )
    start_proxy(
        listen_addr=args.listen,
        route_fn=route_fn,
        complete_fn=complete_fn,
        models_fn=_build_models_fn(components["worker_manager"]),
        pd_prefill_complete_fn=pd_prefill_complete_fn,
        pd_dispatch_snapshot_fn=pd_dispatch_snapshot_fn,
        pd_decode_budget_fn=pd_decode_budget_fn,
        pd_prefill_max_inflight=pd_prefill_max_inflight,
    )


if __name__ == "__main__":
    main()
