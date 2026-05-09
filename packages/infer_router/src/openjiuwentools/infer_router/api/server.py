import asyncio
import copy
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from openjiuwentools.infer_router.api.auth import api_key_auth_middleware
from openjiuwentools.infer_router.config.config import (
    Settings,
    create_settings,
    set_global_settings,
)
from openjiuwentools.infer_router.fault_tolerance.fault_tolerance import (
    CircuitBreaker,
)
from openjiuwentools.infer_router.kv_cache.event import CacheEvent
from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.kv_cache.kv_event_generator import KVEventGenerator
from openjiuwentools.infer_router.kv_cache.kv_event_manager import KVEventManager
from openjiuwentools.infer_router.monitoring.metrics import metrics
from openjiuwentools.infer_router.monitoring.performance_stats import (
    RecordRequestParams,
    performance_stats,
)
from openjiuwentools.infer_router.preprocess.preprocessor import Preprocessor
from openjiuwentools.infer_router.routing.router import Router
from openjiuwentools.infer_router.schedule.scheduler import Scheduler
from openjiuwentools.infer_router.schemas.agent_hints import ChatCompletionRequest
from openjiuwentools.infer_router.worker.worker_manager import WorkerManager
from openjiuwentools.infer_router.worker.workload_manager import WorkloadManager


@dataclass
class DisaggWorkerParams:
    """非组合型worker处理参数"""

    prefill_id: str
    decode_id: str
    chat_request: "ChatCompletionRequest"
    route_hint: Any
    max_tokens: int
    prefill_start_time: float


# 全局组件变量（将在 create_app 函数中初始化）
preprocessor: Preprocessor | None = None
event_manager: KVEventManager | None = None
event_generator: KVEventGenerator | None = None
kvcache_manager: KVCacheManager | None = None
worker_manager: WorkerManager | None = None
workload_manager: WorkloadManager | None = None
scheduler: Scheduler | None = None
router: Router | None = None
circuit_breaker: CircuitBreaker | None = None
settings: Settings | None = None


def _build_kv_headers(prefill_kv_addr: str, decode_kv_addr: str, decode_url: str, dp_rank: int = 0) -> dict[str, str]:
    """Construct the headers expected by vLLM's P2P disagg connector."""
    request_id = f"___prefill_addr_{prefill_kv_addr}___decode_addr_{decode_kv_addr}_{uuid.uuid4().hex}"
    headers: dict[str, str] = {"X-Request-Id": request_id}
    kv_target = os.environ.get("KV_TARGET")
    if kv_target:
        headers["X-KV-Target"] = kv_target
    else:
        headers["X-KV-Target"] = decode_url
    headers["X-data-parallel-rank"] = str(dp_rank)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _parse_agent_hints(chat_request: ChatCompletionRequest) -> tuple:
    """解析Agent Hints和Cache Control"""
    agent_hints = None
    cache_control = None

    if chat_request.jiuwenext:
        if chat_request.jiuwenext.agent_hints:
            agent_hints = chat_request.jiuwenext.agent_hints
            logger.debug(
                f"[STEP 2] Agent Hints found: priority={agent_hints.priority}, "
                f"estimated_output_tokens={agent_hints.estimated_output_tokens}, "
                f"next_turn_prefill={agent_hints.next_turn_prefill}, "
                f"prefix_id={agent_hints.prefix_id}, "
                f"total_requests={agent_hints.total_requests}, "
                f"iat={agent_hints.iat}"
            )

            metrics.record_agent_hints_request()
            metrics.record_priority(agent_hints.priority or 0)
            metrics.record_estimated_output_tokens(agent_hints.estimated_output_tokens or 128)

        if chat_request.jiuwenext.cache_control:
            cache_control = chat_request.jiuwenext.cache_control
            logger.debug(f"[STEP 2] Cache Control found: type={cache_control.type}, ttl={cache_control.ttl}")
    else:
        logger.debug("[STEP 2] No Agent Hints provided, using defaults")

    return agent_hints, cache_control


async def _handle_combined_worker(
    worker_id: str,
    chat_request: ChatCompletionRequest,
    route_hint,
    max_tokens: int,
    prefill_start_time: float,
):
    """处理组合型worker的请求"""
    combined_worker = worker_manager.get_worker(worker_id)
    if not combined_worker:
        raise ValueError(f"Combined worker {worker_id} not found")

    logger.debug(f"[STEP 9] Selected combined worker: {combined_worker.worker_id}")

    estimated_tokens = max_tokens
    if chat_request.jiuwenext and chat_request.jiuwenext.agent_hints:
        estimated_tokens = chat_request.jiuwenext.agent_hints.estimated_output_tokens or max_tokens

    if event_generator:
        decode_start_event = CacheEvent(
            event_type="decode_start",
            token_count=len(route_hint.token_ids) if route_hint.token_ids else 0,
            worker_id=combined_worker.worker_id,
            engine_specific={
                "worker_id": combined_worker.worker_id,
                "osl": estimated_tokens,
            },
        )
        event_manager.process_event(decode_start_event)

    if event_generator and route_hint.token_ids:
        events = event_generator.generate_events(
            combined_worker.worker_id,
            route_hint.token_ids,
        )
        for event in events:
            event_manager.process_event(event)

    request_data = chat_request.model_dump(exclude_none=True, exclude={"jiuwenext"})
    logger.debug(f"[STEP 10] Forwarding request to {combined_worker.worker_id} with max_tokens={max_tokens}")
    prefill_end_time = time.time()
    response_data = await worker_manager.forward_request(
        worker_id=combined_worker.worker_id,
        endpoint="/v1/chat/completions",
        method="POST",
        data=request_data,
    )
    decode_end_time = time.time()
    logger.debug(f"[STEP 11] Received response from {combined_worker.worker_id}")

    # decode_end事件将在响应返回后生成（在register_routes中处理）

    return response_data, prefill_start_time, prefill_end_time, decode_end_time


async def _handle_disagg_workers(params: DisaggWorkerParams):
    """处理非组合型worker的请求（Mooncake模式）"""
    prefill_id = params.prefill_id
    decode_id = params.decode_id
    chat_request = params.chat_request
    route_hint = params.route_hint
    max_tokens = params.max_tokens
    prefill_start_time = params.prefill_start_time

    prefill_worker = worker_manager.get_worker(prefill_id)
    decode_worker = worker_manager.get_worker(decode_id)

    if not prefill_worker:
        raise ValueError(f"Prefill worker {prefill_id} not found")
    if not decode_worker:
        raise ValueError(f"Decode worker {decode_id} not found")

    logger.debug(f"[STEP 9] Selected prefill worker: {prefill_worker.worker_id}")
    logger.debug(f"[STEP 9] Selected decode worker: {decode_worker.worker_id}")

    prefill_kv_addr = getattr(prefill_worker, "kv_addr", "")
    decode_kv_addr = getattr(decode_worker, "kv_addr", "")
    dp_rank = getattr(prefill_worker, "dp_rank", 0)
    logger.debug(f"[STEP 9] Prefill KV addr: {prefill_kv_addr}, Decode KV addr: {decode_kv_addr}, DP rank: {dp_rank}")

    kv_headers = _build_kv_headers(prefill_kv_addr, decode_kv_addr, decode_worker.url, dp_rank)
    logger.debug(f"[STEP 9] Built KV headers: {kv_headers}")

    req_data_to_prefill = copy.deepcopy(chat_request.model_dump(exclude_none=True, exclude={"jiuwenext"}))
    req_data_to_decode = copy.deepcopy(chat_request.model_dump(exclude_none=True, exclude={"jiuwenext"}))

    req_data_to_prefill["stream"] = False
    req_data_to_prefill["max_tokens"] = 1
    if "max_completion_tokens" in req_data_to_prefill:
        req_data_to_prefill["max_completion_tokens"] = 1
    if "stream_options" in req_data_to_prefill:
        del req_data_to_prefill["stream_options"]

    logger.debug("[STEP 11] KV Transfer type: MOONCAKE")
    logger.debug(f"[STEP 11] Forwarding prefill request to {prefill_worker.worker_id} with max_tokens=1")

    if event_generator:
        prefill_start_event = CacheEvent(
            event_type="prefill_start",
            token_count=len(route_hint.token_ids) if route_hint.token_ids else 0,
            worker_id=prefill_worker.worker_id,
            engine_specific={"worker_id": prefill_worker.worker_id},
        )
        event_manager.process_event(prefill_start_event)

    try:
        await worker_manager.forward_request(
            worker_id=prefill_worker.worker_id,
            endpoint="/v1/chat/completions",
            method="POST",
            data=req_data_to_prefill,
            headers=kv_headers,
        )
        prefill_end_time = time.time()
        logger.debug(f"[STEP 12] Prefill request completed for {prefill_worker.worker_id}")
    except Exception as e:
        logger.error(f"[STEP 12] Error sending prefill request: {e}")
        # 生成 prefill_end 事件（即使失败）
        if event_generator:
            prefill_end_event = CacheEvent(
                event_type="prefill_end",
                token_count=len(route_hint.token_ids) if route_hint.token_ids else 0,
                worker_id=prefill_worker.worker_id,
                engine_specific={"worker_id": prefill_worker.worker_id},
            )
            event_manager.process_event(prefill_end_event)
        raise

    # 生成 prefill_end 事件
    if event_generator:
        prefill_end_event = CacheEvent(
            event_type="prefill_end",
            token_count=len(route_hint.token_ids) if route_hint.token_ids else 0,
            worker_id=prefill_worker.worker_id,
            engine_specific={"worker_id": prefill_worker.worker_id},
        )
        event_manager.process_event(prefill_end_event)

    logger.debug(f"[STEP 13] Forwarding decode request to {decode_worker.worker_id} with max_tokens={max_tokens}")

    # 计算 estimated_tokens（供 decode_start 和 decode_end 共用）
    estimated_tokens = max_tokens
    if chat_request.jiuwenext and chat_request.jiuwenext.agent_hints:
        estimated_tokens = chat_request.jiuwenext.agent_hints.estimated_output_tokens or max_tokens

    # 生成 decode_start 事件
    if event_generator:
        decode_start_event = CacheEvent(
            event_type="decode_start",
            token_count=len(route_hint.token_ids) if route_hint.token_ids else 0,
            worker_id=decode_worker.worker_id,
            engine_specific={
                "worker_id": decode_worker.worker_id,
                "osl": estimated_tokens,
            },
        )
        event_manager.process_event(decode_start_event)

    if event_generator and route_hint.token_ids:
        # 为prefill worker生成事件
        prefill_events = event_generator.generate_events(
            prefill_worker.worker_id,
            route_hint.token_ids,
        )
        for event in prefill_events:
            event_manager.process_event(event)

        # 为decode worker生成事件
        decode_events = event_generator.generate_events(
            decode_worker.worker_id,
            route_hint.token_ids,
        )
        for event in decode_events:
            event_manager.process_event(event)

    if chat_request.stream:
        response = await worker_manager.forward_request_stream(
            worker_id=decode_worker.worker_id,
            endpoint="/v1/chat/completions",
            method="POST",
            data=req_data_to_decode,
            headers=kv_headers,
        )
        response_data = StreamingResponse(response, media_type="text/event-stream")

        # 对于流式响应，decode_end 事件需要在流式响应结束时生成
        if event_generator:
            decode_end_event = CacheEvent(
                event_type="decode_end",
                token_count=len(route_hint.token_ids) if route_hint.token_ids else 0,
                token_ids=[],
                worker_id=decode_worker.worker_id,
                engine_specific={
                    "worker_id": decode_worker.worker_id,
                    "osl": estimated_tokens,
                    "completion_tokens": estimated_tokens,
                },
            )
            event_manager.process_event(decode_end_event)
        decode_end_time = time.time()
    else:
        response_data = await worker_manager.forward_request(
            worker_id=decode_worker.worker_id,
            endpoint="/v1/chat/completions",
            method="POST",
            data=req_data_to_decode,
            headers=kv_headers,
        )
        decode_end_time = time.time()
        logger.debug(f"[STEP 14] Received decode response from {decode_worker.worker_id}")

        # 非流式响应的decode_end事件将在响应返回后生成（在register_routes中处理）

    return response_data, prefill_start_time, prefill_end_time, decode_end_time


def register_routes(app: FastAPI):
    """注册所有路由到应用

    Args:
        app: FastAPI 应用实例
    """

    @app.post("/v1/chat/completions")
    async def chat_completions(chat_request: ChatCompletionRequest):
        """聊天完成API（Mooncake模式）

        处理流程：
        1. 预处理请求，转换为路由提示
        2. 提交请求到调度队列
        3. 调度器按策略从队列取出请求
        4. 路由决策层选择最佳prefill和decode worker
        5. 如果是组合型worker，直接发送请求（不修改max_tokens）
        6. 如果是非组合型worker（Mooncake模式）：
            a. 发送prefill请求（max_tokens=1）到prefill节点
            b. 等待prefill完成后，发送decode请求（使用原始max_tokens）到decode节点
            c. KV传输通过headers实现（X-Request-Id, X-KV-Target）
        7. 返回响应给用户
        """
        start_time = time.time()
        endpoint = "/v1/chat/completions"
        method = "POST"
        try:
            request_id = f"chatcmpl-{uuid.uuid4()}"
            logger.info(
                f"[REQUEST] Received request {request_id}: model={chat_request.model},"
                f"messages_count={len(chat_request.messages)}, max_tokens={chat_request.max_tokens}"
            )

            agent_hints, cache_control = _parse_agent_hints(chat_request)

            route_hint = preprocessor.process(chat_request, agent_hints, request_id)
            logger.debug(f"[STEP 3] Preprocessed route_hint: {route_hint}")

            overlap_scores = None
            if route_hint.token_ids:
                overlap_scores = kvcache_manager.find_matches(
                    token_ids=route_hint.token_ids,
                    model=route_hint.model,
                )
                logger.debug(f"[STEP 4] KV cache overlap scores: {overlap_scores}")
            else:
                logger.debug("[STEP 4] No token_ids for KV cache matching")

            logger.debug("[STEP 5] Submitting request to scheduler")
            scheduler.submit(route_hint, overlap_scores)

            from openjiuwentools.infer_router.fault_tolerance import retry

            @retry(max_attempts=3, delay=0.1, circuit_breaker=circuit_breaker)
            def dispatch_request():
                logger.debug("[STEP 6] Getting next request from scheduler")
                scheduled = scheduler.get_next_request()
                if not scheduled:
                    raise RuntimeError("No request available from scheduler")
                logger.debug("[STEP 7] Routing request to worker pair")
                prefill_id, decode_id = router.route_to_workers(scheduled.route_hint)
                return scheduled, prefill_id, decode_id

            scheduled_request, prefill_id, decode_id = dispatch_request()
            router_dispatch_end_time = time.time()
            logger.debug(f"[STEP 8] Request {request_id} dispatched to prefill={prefill_id}, decode={decode_id}")

            max_tokens = chat_request.max_tokens or 128
            prefill_start_time = time.time()

            if decode_id is None:
                (
                    response_data,
                    _,
                    prefill_end_time,
                    decode_end_time,
                ) = await _handle_combined_worker(prefill_id, chat_request, route_hint, max_tokens, prefill_start_time)
            else:
                (
                    response_data,
                    _,
                    prefill_end_time,
                    decode_end_time,
                ) = await _handle_disagg_workers(
                    prefill_id,
                    decode_id,
                    chat_request,
                    route_hint,
                    max_tokens,
                    prefill_start_time,
                )

            scheduler.mark_task_completed()
            logger.debug("[STEP 15] Marked task as completed")

            end_time = time.time()
            duration = end_time - start_time
            metrics.record_request(endpoint, method, 200)
            metrics.record_request_duration(endpoint, method, duration)

            router_dispatch_duration = router_dispatch_end_time - start_time
            prefill_duration = prefill_end_time - prefill_start_time
            decode_duration = decode_end_time - prefill_end_time
            response_return_duration = end_time - decode_end_time

            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            osl = agent_hints.estimated_output_tokens if agent_hints and agent_hints.estimated_output_tokens else 0

            if isinstance(response_data, dict) and "usage" in response_data:
                usage = response_data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                # 在响应返回后生成decode_end事件，以便包含真实的completion_tokens
                if event_generator:
                    if decode_id is None:
                        # 组合型worker
                        decode_end_event = CacheEvent(
                            event_type="decode_end",
                            token_count=len(route_hint.token_ids) if route_hint.token_ids else 0,
                            token_ids=[],
                            worker_id=prefill_id,
                            engine_specific={
                                "worker_id": prefill_id,
                                "osl": osl,
                                "completion_tokens": completion_tokens,
                            },
                        )
                        event_manager.process_event(decode_end_event)
                    else:
                        # 分离型worker - 更新decode worker的decode_end事件
                        decode_end_event = CacheEvent(
                            event_type="decode_end",
                            token_count=len(route_hint.token_ids) if route_hint.token_ids else 0,
                            token_ids=[],
                            worker_id=decode_id,
                            engine_specific={
                                "worker_id": decode_id,
                                "osl": osl,
                                "completion_tokens": completion_tokens,
                            },
                        )
                        event_manager.process_event(decode_end_event)

            request_params = RecordRequestParams(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                router_dispatch_duration=router_dispatch_duration,
                prefill_duration=prefill_duration,
                decode_duration=decode_duration,
                response_return_duration=response_return_duration,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                osl=osl,
            )
            performance_stats.record_request(request_params)

            logger.info(f"[REQUEST] Request {request_id} completed in {duration:.3f}s")

            return JSONResponse(response_data)

        except (httpx.TimeoutException, asyncio.TimeoutError) as e:
            logger.error(f"Request timeout: {e}")

            scheduler.mark_task_failed()

            end_time = time.time()
            duration = end_time - start_time
            metrics.record_request(endpoint, method, 504)
            metrics.record_request_duration(endpoint, method, duration)

            return JSONResponse(
                status_code=504,
                content={"error": {"message": "Request timeout", "type": "gateway_timeout"}},
            )
        except HTTPException as e:
            logger.error(f"HTTP exception: {e.status_code} - {e.detail}")

            scheduler.mark_task_failed()

            end_time = time.time()
            duration = end_time - start_time
            status_code = e.status_code
            metrics.record_request(endpoint, method, status_code)
            metrics.record_request_duration(endpoint, method, duration)

            return JSONResponse(
                status_code=status_code,
                content={"error": {"message": e.detail, "type": "http_error"}},
            )
        except Exception as e:
            logger.error(f"Error processing request: {e}")

            scheduler.mark_task_failed()

            end_time = time.time()
            duration = end_time - start_time
            metrics.record_request(endpoint, method, 500)
            metrics.record_request_duration(endpoint, method, duration)

            # 检查异常消息中是否包含timeout关键词
            error_message = str(e)
            if "timeout" in error_message.lower():
                return JSONResponse(
                    status_code=504,
                    content={
                        "error": {
                            "message": "Request timeout",
                            "type": "gateway_timeout",
                        }
                    },
                )

            return JSONResponse(
                status_code=500,
                content={"error": {"message": error_message, "type": "internal_server_error"}},
            )

    @app.get("/health")
    async def health_check():
        """健康检查API"""
        # 获取工作器状态
        workers = worker_manager.get_all_workers()
        worker_statuses = worker_manager.get_all_worker_statuses()

        # 获取缓存统计
        cache_stats = kvcache_manager.get_cache_stats()

        queue_stats = scheduler.get_queue_stats()
        scheduler_stats = scheduler.get_stats()

        # 获取路由统计
        try:
            # 从metrics对象直接获取routed_requests_total的数据
            routed_total = 0
            worker_route_stats = {}

            if hasattr(metrics, "routed_requests_total"):
                # 使用collect()方法获取所有Sample
                counter = metrics.routed_requests_total

                for metric in counter.collect():
                    for sample in metric.samples:
                        # 只处理_total样本，忽略_created样本
                        if sample.name.endswith("_total"):
                            # 从labels字典中获取worker_id和model
                            worker_id = sample.labels.get("worker_id", "unknown")
                            model = sample.labels.get("model", "unknown")
                            count_value = sample.value

                            # 确保是有效的计数值
                            if isinstance(count_value, (int, float)) and count_value > 0:
                                routed_total += count_value

                                if worker_id not in worker_route_stats:
                                    worker_route_stats[worker_id] = {
                                        "total": 0,
                                        "by_model": {},
                                    }

                                worker_route_stats[worker_id]["total"] += count_value
                                worker_route_stats[worker_id]["by_model"][model] = count_value

            route_stats = {"total": routed_total, "by_worker": worker_route_stats}
        except Exception as e:
            logger.warning(f"Failed to collect route stats: {e}")
            route_stats = {"total": 0, "by_worker": {}, "error": str(e)}

        return {
            "status": "healthy",
            "components": {
                "preprocessor": "ok",
                "kvcache_manager": "ok",
                "scheduler": "ok",
                "router": "ok",
                "worker_manager": "ok",
                "performance_optimizer": "ok",
            },
            "workers": {
                "total": len(workers),
                "healthy": len([s for s in worker_statuses.values() if s.is_healthy]),
                "details": {
                    worker_id: {
                        "is_healthy": status.is_healthy,
                        "last_health_check": status.last_health_check,
                        "response_time": status.response_time,
                    }
                    for worker_id, status in worker_statuses.items()
                },
            },
            "cache": cache_stats,
            "queue": queue_stats,
            "scheduler": scheduler_stats,
            "routing": route_stats,
        }

    @app.get("/metrics")
    async def get_metrics():
        """Prometheus指标接口"""
        from prometheus_client import generate_latest

        return generate_latest()


def create_app(cfg: Settings) -> FastAPI:
    """创建 FastAPI 应用并初始化所有组件

    Args:
        cfg: 配置对象

    Returns:
        FastAPI: 配置好的应用实例
    """
    global \
        preprocessor, \
        event_manager, \
        event_generator, \
        kvcache_manager, \
        worker_manager, \
        workload_manager, \
        scheduler, \
        router, \
        circuit_breaker, \
        settings

    settings = cfg

    # 初始化组件
    preprocessor = Preprocessor()
    event_manager = KVEventManager()

    # 根据配置决定是否创建 KVEventGenerator
    if settings.kv_event_mode == "inner_event":
        event_generator = KVEventGenerator()
        logger.info(f"KVEventGenerator initialized for mode: {settings.kv_event_mode}")
    else:
        logger.info(f"Using worker event mode: {settings.kv_event_mode}")

    kvcache_manager = KVCacheManager(
        event_manager=event_manager,
        event_generator=event_generator,
        enable_radix_tree=settings.kv_cache_enable_radix_tree,
    )
    worker_manager = WorkerManager(kv_cache_manager=kvcache_manager)

    # 初始化 WorkloadManager（仅在 inner_event 模式下启用）
    workload_manager = WorkloadManager()
    worker_manager.set_workload_manager(workload_manager)

    # 注册事件处理器
    worker_manager.register_event_handlers()

    scheduler = Scheduler(kvcache_manager)
    router = Router(kvcache_manager, worker_manager)

    # 初始化容错组件
    circuit_breaker = CircuitBreaker()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""

        import sys

        loguru_level = settings.log_level.upper()
        logger.remove()

        # 检测是否输出到终端
        is_tty = sys.stdout.isatty()

        def sink_to_file(message):
            """输出到文件，不使用颜色"""
            record = message.record
            exception = record["exception"]

            log_line = (
                f"{record['time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
                f"{record['level'].name:8} | "
                f"{record['name']}:{record['function']}:{record['line']} - "
                f"{record['message']}"
            )

            if exception:
                log_line += f"\n{exception}"

            log_dir = "/home/xrx/agent-tools/packages/infer_router/logs"
            os.makedirs(log_dir, exist_ok=True)
            with open(f"{log_dir}/router.log", "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

        # 终端输出：根据是否为 tty 决定是否启用颜色
        if is_tty:
            logger.add(
                sink=sys.stdout,
                level=loguru_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                colorize=True,
                diagnose=False,
            )
        else:
            # 重定向到文件时，不使用颜色
            logger.add(
                sink=sys.stdout,
                level=loguru_level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:8} | {name}:{function}:{line} - {message}",
                colorize=False,
                diagnose=False,
            )

        # 文件输出
        logger.add(
            sink=sink_to_file,
            level=loguru_level,
            format="{message}",
            colorize=False,
            diagnose=False,
        )

        logger.info("Starting Jiuwen Agent Router...")

        await worker_manager.start()
        metrics.start_metrics_server()
        await performance_stats.start()

        logger.info("Jiuwen Agent Router started successfully")

        yield

        logger.info("Shutting down Jiuwen Agent Router...")
        await worker_manager.stop()
        await performance_stats.stop()
        logger.info("Jiuwen Agent Router shutdown completed")

    # 创建FastAPI应用
    app = FastAPI(title="Jiuwen Agent Router", version="1.0.0", lifespan=lifespan)

    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 添加认证中间件
    app.middleware("http")(api_key_auth_middleware)

    # 注册路由
    register_routes(app)

    return app


_app_instance: FastAPI | None = None


def get_app() -> FastAPI:
    """获取或创建 FastAPI 应用实例

    用于测试目的。如果还没有创建应用实例，则使用默认配置创建一个。

    Returns:
        FastAPI: 应用实例

    """
    global _app_instance
    if _app_instance is None:
        cfg = create_settings()
        set_global_settings(cfg)
        _app_instance = create_app(cfg)
    return _app_instance


def main():
    """主入口点"""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Jiuwen Agent Router")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (overrides config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Log level (overrides config)",
    )

    args = parser.parse_args()

    # 先加载配置并创建应用
    cfg = create_settings(args.config)
    set_global_settings(cfg)
    app = create_app(cfg)

    host = args.host or settings.host
    port = args.port or settings.port
    log_level = args.log_level or settings.log_level

    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
