"""vLLM Worker 进程入口，负责引擎初始化、消息发布、服务注册和优雅关闭。

启动流程::

    main() → worker()
      1. 构建 vLLM AsyncLLM 引擎
      2. 启动 HTTP/TCP 请求服务
      3. 注册到 etcd（可选）
      4. 启动 WorkerPublisher 上报 KV 事件和负载指标
      5. 安装信号处理器，等待退出
"""

import asyncio
import json
import logging
import signal
from dataclasses import dataclass

import uvloop

from openjiuwentools.infer_router.vllm.args import parse_args

logger = logging.getLogger("jiuwen.worker")

_DEFAULT_GRACE_PERIOD_SECS = 5.0


async def kv_event_listener(endpoint: str, topic: str = "") -> None:
    """本地 KV 事件监听器，订阅 vLLM ZMQ PUB 并将事件记录到日志（调试用）。"""
    import msgspec
    import zmq
    import zmq.asyncio
    from vllm.distributed.kv_events import KVEventBatch

    ctx = zmq.asyncio.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, topic.encode())

    decoder = msgspec.msgpack.Decoder(KVEventBatch)
    logger.info("KV event listener started, subscribing to %s", endpoint)

    try:
        while True:
            topic_bytes, seq_bytes, payload = await sub.recv_multipart()
            seq = int.from_bytes(seq_bytes, "big")
            batch: KVEventBatch = decoder.decode(payload)
            for event in batch.events:
                logger.info("KV event seq=%d ts=%s type=%s event=%s",
                            seq, batch.ts, type(event).__name__, event)
    finally:
        sub.close()
        ctx.term()


def _build_engine(args):
    """构建 vLLM AsyncLLM 引擎，配置 KV 事件、StatLogger 和推理参数。"""
    from vllm.config.kv_events import KVEventsConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    kv_cfg_dict = json.loads(args.kv_events_config) if args.kv_events_config else None
    kv_cfg = KVEventsConfig(**kv_cfg_dict) if kv_cfg_dict else None

    kv_transfer_cfg = json.loads(args.kv_transfer_config) if args.kv_transfer_config else None

    extra_kwargs = {}
    if getattr(args, "enable_sleep_mode", False):
        extra_kwargs["enable_sleep_mode"] = True
    if kv_transfer_cfg is not None:
        extra_kwargs["kv_transfer_config"] = kv_transfer_cfg

    engine_args = AsyncEngineArgs(
        model=args.model,
        served_model_name=[args.served_model_name or args.model],
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
        enable_prefix_caching=args.enable_prefix_caching,
        kv_events_config=kv_cfg,
        **extra_kwargs,
    )

    from openjiuwentools.infer_router.vllm.prometheus_logger import WorkerBackendMetrics, StatLoggerFactory

    model_name = args.served_model_name or args.model
    component_gauges = WorkerBackendMetrics(model_name=model_name, component_name="vllm-worker")
    stat_logger_factory = StatLoggerFactory(component_gauges=component_gauges)

    engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=[stat_logger_factory])
    return engine, kv_cfg_dict, stat_logger_factory


def _normalize_zmq_endpoint(endpoint: str) -> str:
    for wildcard in ("tcp://*:", "tcp://:", "tcp://0.0.0.0:"):
        if endpoint.startswith(wildcard):
            return "tcp://127.0.0.1:" + endpoint[len(wildcard):]
    return endpoint


async def _start_worker_publisher(
    kv_cfg: dict | None,
    relay_endpoint: str | None = None,
    worker_id: str = "",
    worker_mode: str = "aggregated",
    metrics_interval: float = 5.0,
) -> tuple:
    """启动 WorkerPublisher 及其子发布器，返回 (worker_pub, kv_pub, metrics_pub)。

    - 无 relay_endpoint 时退化为本地 KV 事件日志监听（调试模式）
    - decode 节点不上报 KV 事件，但仍上报负载指标
    - prefill/aggregated 节点同时上报 KV 事件和负载指标
    """
    from openjiuwentools.infer_router.vllm.worker_publisher import KvEventPublisher, MetricsPublisher, WorkerPublisher

    if not relay_endpoint:
        # 无 relay_endpoint 时仅本地监听 KV 事件日志
        if kv_cfg and kv_cfg.get("enable_kv_cache_events") and worker_mode != "decode":
            vllm_pub = kv_cfg.get("endpoint", "tcp://*:5557")
            sub_endpoint = _normalize_zmq_endpoint(vllm_pub)
            topic = kv_cfg.get("topic", "")
            task = asyncio.create_task(kv_event_listener(sub_endpoint, topic))

            def _on_listener_done(t):
                if t.cancelled():
                    return
                exc = t.exception()
                if exc:
                    logger.error("KV event listener crashed: %s", exc, exc_info=exc)

            task.add_done_callback(_on_listener_done)
        return None, None, None

    # 启动 WorkerPublisher（ZMQ PUB socket）
    worker_pub = WorkerPublisher(relay_endpoint)
    await worker_pub.start()

    # 启动 KvEventPublisher（仅 prefill/aggregated 节点）
    kv_pub = None
    if worker_mode != "decode" and kv_cfg and kv_cfg.get("enable_kv_cache_events"):
        vllm_pub = kv_cfg.get("endpoint", "tcp://*:5557")
        sub_endpoint = _normalize_zmq_endpoint(vllm_pub)
        topic = kv_cfg.get("topic", "")
        kv_pub = KvEventPublisher(
            worker_publisher=worker_pub,
            sub_endpoint=sub_endpoint,
            topic=topic,
            worker_id=worker_id,
        )
        kv_pub.start()

    # 启动 MetricsPublisher（所有节点都上报负载）
    metrics_pub = MetricsPublisher(
        worker_publisher=worker_pub,
        worker_id=worker_id,
        interval=metrics_interval,
    )
    metrics_pub.start()

    return worker_pub, kv_pub, metrics_pub


@dataclass
class ShutdownContext:
    """优雅关闭所需的上下文"""

    server_close: object  # callable
    engine: object
    shutdown_event: asyncio.Event
    registry: object = None
    worker_publisher: object = None
    kv_publisher: object = None
    metrics_publisher: object = None


async def _graceful_shutdown(ctx: ShutdownContext, grace_period_s: float | None = None):
    """优雅关闭：注销服务 → 停止发布器 → 等待宽限期 → 关闭服务和引擎。"""
    if ctx.shutdown_event.is_set():
        return
    ctx.shutdown_event.set()

    if grace_period_s is None:
        grace_period_s = _DEFAULT_GRACE_PERIOD_SECS

    if ctx.registry:
        await ctx.registry.deregister()

    if ctx.metrics_publisher:
        await ctx.metrics_publisher.stop()
    if ctx.kv_publisher:
        await ctx.kv_publisher.stop()
    if ctx.worker_publisher:
        await ctx.worker_publisher.stop()

    logger.info("Shutdown signal received, grace period %.2fs for in-flight requests", grace_period_s)
    if grace_period_s > 0:
        await asyncio.sleep(grace_period_s)

    logger.info("Grace period ended, stopping server")
    ctx.server_close()

    if hasattr(ctx.engine, "shutdown"):
        ctx.engine.shutdown()
    logger.info("Shutdown complete")


def _install_signal_handlers(loop, ctx: ShutdownContext):
    """注册 SIGTERM/SIGINT 信号处理器，触发优雅关闭流程。"""
    shutdown_task = None

    def _on_done(task):
        nonlocal shutdown_task
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Graceful shutdown failed")
        finally:
            if shutdown_task is task:
                shutdown_task = None

    def _handler():
        nonlocal shutdown_task
        if shutdown_task is not None and not shutdown_task.done():
            logger.debug("Shutdown already in progress, ignoring duplicate signal")
            return
        shutdown_task = asyncio.create_task(
            _graceful_shutdown(ctx)
        )
        shutdown_task.add_done_callback(_on_done)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handler)
    logger.info("Signal handlers installed for graceful shutdown (grace=%.1fs)", _DEFAULT_GRACE_PERIOD_SECS)


async def worker(args=None) -> None:
    """Worker 主协程：初始化引擎、启动服务、注册发现、上报指标，阻塞直到退出。"""
    if args is None:
        args = parse_args()

    from openjiuwentools.infer_router.vllm.transport import serve_http, serve_tcp

    engine, kv_cfg, stat_logger_factory = _build_engine(args)

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    if args.request_plane == "tcp":
        close_fn, serve_coro = await serve_tcp(args, engine, shutdown_event)
    else:
        close_fn, serve_coro = await serve_http(args, engine, shutdown_event)

    registry = None
    if args.etcd_endpoints:
        from openjiuwentools.infer_router.vllm.registry import EtcdRegistry

        model_name = args.served_model_name or args.model
        registry = EtcdRegistry(
            etcd_endpoints=args.etcd_endpoints,
            service_name=args.service_name or model_name,
            host=args.host,
            port=args.port,
            ttl=args.registry_ttl,
            metadata={
                "model": model_name,
                "request_plane": args.request_plane,
                "worker_mode": args.worker_mode,
            },
        )
        await registry.register()

    worker_id = registry.worker_id if registry else ""
    worker_publisher, kv_publisher, metrics_publisher = await _start_worker_publisher(
        kv_cfg,
        relay_endpoint=args.kv_relay_endpoint,
        worker_id=worker_id,
        worker_mode=args.worker_mode,
        metrics_interval=args.metrics_interval,
    )

    if metrics_publisher:
        metrics_publisher.set_metrics_source(stat_logger_factory)

    shutdown_ctx = ShutdownContext(
        server_close=close_fn,
        engine=engine,
        shutdown_event=shutdown_event,
        registry=registry,
        worker_publisher=worker_publisher,
        kv_publisher=kv_publisher,
        metrics_publisher=metrics_publisher,
    )
    _install_signal_handlers(loop, shutdown_ctx)
    await serve_coro
    logger.info("Worker exited")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    uvloop.run(worker())


if __name__ == "__main__":
    main()
