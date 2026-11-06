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
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass

import uvloop

from openjiuwentools.infer_router.vllm.args import parse_args

logger = logging.getLogger("jiuwen.worker")

_DEFAULT_GRACE_PERIOD_SECS = 5.0


def _configure_prefix_caching_hash_seed(args) -> str:
    """为 vLLM prefix-cache hash 链设置跨 worker 一致的初始种子。

    vLLM 0.18 在 ``PYTHONHASHSEED`` 未设置时使用随机字节初始化首个
    block 的 parent hash。该环境变量必须在导入 vLLM 和创建 engine-core
    子进程之前设置，才能让不同 worker 对相同输入生成相同 hash 链。

    参数:
        args: worker 启动参数，包含 ``prefix_caching_hash_seed`` 整数配置。

    返回:
        写入 ``PYTHONHASHSEED`` 的十进制字符串。

    异常:
        ValueError: 配置值不在 Python 支持的 0 到 4294967295 范围内。
    """

    seed = int(getattr(args, "prefix_caching_hash_seed", 0))
    if not 0 <= seed <= 4294967295:
        raise ValueError("prefix_caching_hash_seed must be between 0 and 4294967295")
    configured_seed = str(seed)
    previous_seed = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = configured_seed
    if previous_seed is not None and previous_seed != configured_seed:
        logger.warning(
            "Overriding PYTHONHASHSEED for deterministic cross-worker prefix hashes: previous=%s configured=%s",
            previous_seed,
            configured_seed,
        )
    return configured_seed


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
                logger.info("KV event seq=%d ts=%s type=%s event=%s", seq, batch.ts, type(event).__name__, event)
    finally:
        sub.close()
        ctx.term()


def _build_engine(args):
    """构建 vLLM AsyncLLM 引擎，配置 KV 事件、StatLogger 和推理参数。

    参数:
        args: worker 启动参数，包含模型、前缀缓存、KV 事件及传输配置。

    返回:
        三元组，依次为 AsyncLLM 引擎、原始 KV 事件配置字典和指标日志工厂。
    """
    prefix_caching_hash_seed = _configure_prefix_caching_hash_seed(args)

    from vllm.config.kv_events import KVEventsConfig
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    kv_cfg_dict = json.loads(args.kv_events_config) if args.kv_events_config else None
    kv_cfg = KVEventsConfig(**kv_cfg_dict) if kv_cfg_dict else None

    kv_transfer_dict = json.loads(args.kv_transfer_config) if args.kv_transfer_config else None
    kv_transfer_cfg = KVTransferConfig(**kv_transfer_dict) if kv_transfer_dict else None

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
        prefix_caching_hash_algo=args.prefix_caching_hash_algo,
        kv_events_config=kv_cfg,
        **extra_kwargs,
    )

    logger.info(
        "Building vLLM engine with prefix caching enabled=%s hash_algo=%s hash_seed=%s",
        args.enable_prefix_caching,
        args.prefix_caching_hash_algo,
        prefix_caching_hash_seed,
    )

    from openjiuwentools.infer_router.vllm.prometheus_logger import WorkerBackendMetrics, StatLoggerFactory

    model_name = args.served_model_name or args.model
    component_gauges = WorkerBackendMetrics(model_name=model_name, component_name="vllm-worker")
    stat_logger_factory = StatLoggerFactory(component_gauges=component_gauges)

    engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=[stat_logger_factory])
    return engine, kv_cfg_dict, stat_logger_factory


def _install_kvbm_engine_core_hook(args) -> None:
    """在创建 vLLM engine 前安装 KVBM engine-core hook。

    参数:
        args: worker argparse Namespace。

    返回:
        无返回值。
    """

    if not getattr(args, "kvbm_worker_enabled", False):
        return
    from openjiuwentools.infer_router.vllm.kvbm_provider import (
        resolve_kvbm_worker_provider,
    )

    resolved = resolve_kvbm_worker_provider(getattr(args, "kvbm_worker_provider", "auto"))
    provider = resolved.name
    args.kvbm_worker_provider = provider
    logger.info(
        "Worker KVBM provider resolved: requested=%s provider=%s vllm_ascend_version=%s vllm_version=%s",
        resolved.requested,
        provider,
        resolved.vllm_ascend_version,
        resolved.vllm_version,
    )
    if not resolved.spec.engine_core_hook:
        return
    prefetch_port = getattr(args, "kvbm_prefetch_port", None)
    if prefetch_port is None:
        prefetch_port = int(getattr(args, "port", 8001)) + 10000
    try:
        from jiuwen_kvbm import integrations as kvbm_integrations

        install_multiprocess_hooks = getattr(
            kvbm_integrations,
            resolved.spec.multiprocess_installer,
        )

        installed = install_multiprocess_hooks(
            diagnostics=bool(getattr(args, "kvbm_worker_diagnostics", False)),
            worker_id=str(getattr(args, "worker_id", "")),
            disk_root=str(getattr(args, "kvbm_worker_disk_root", "/tmp/jiuwen-kvbm-worker")),
            clean_disk_on_start=bool(getattr(args, "kvbm_clean_disk_on_start", True)),
            npu_high_watermark_blocks=int(getattr(args, "kvbm_npu_high_watermark_blocks", 0)),
            ram_high_watermark_bytes=int(getattr(args, "kvbm_ram_high_watermark_bytes", 0)),
            block_size=int(getattr(args, "kvbm_block_size", 16)),
            auto_register_physical_blocks=bool(getattr(args, "kvbm_auto_register_physical_blocks", False)),
            auto_enforce_watermarks_on_init=bool(getattr(args, "kvbm_auto_enforce_watermarks_on_init", False)),
            prefetch_host=str(getattr(args, "kvbm_prefetch_host", "127.0.0.1")),
            prefetch_port=int(prefetch_port),
            reservation_port=int(getattr(args, "kvbm_reservation_port", 0)),
            transfer_max_inflight=int(getattr(args, "kvbm_transfer_max_inflight", 1)),
            transfer_retention_seconds=float(getattr(args, "kvbm_transfer_retention_seconds", 300.0)),
            benchmark_enabled=bool(getattr(args, "kvbm_benchmark_enabled", False)),
            benchmark_tcp_host=str(getattr(args, "kvbm_benchmark_tcp_host", "127.0.0.1")),
            benchmark_tcp_port=int(getattr(args, "kvbm_benchmark_tcp_port", 0)),
            benchmark_max_blocks=int(getattr(args, "kvbm_benchmark_max_blocks", 64)),
            data_backend=str(getattr(args, "kvbm_data_backend", "local")),
            llm_datadist_mode=str(getattr(args, "kvbm_llm_datadist_mode", "python")),
            llm_datadist_role=str(getattr(args, "kvbm_llm_datadist_role", "mix")),
            llm_datadist_cluster_id=int(getattr(args, "kvbm_llm_datadist_cluster_id", 0)),
            llm_datadist_listen_ip=str(getattr(args, "kvbm_llm_datadist_listen_ip", "127.0.0.1")),
            llm_datadist_listen_port=int(getattr(args, "kvbm_llm_datadist_listen_port", 0)),
            llm_datadist_listen_port_base=int(getattr(args, "kvbm_llm_datadist_listen_port_base", 19000)),
            llm_datadist_transfer_backend=str(getattr(args, "kvbm_llm_datadist_transfer_backend", "hixl")),
            llm_datadist_enable_cache_manager=bool(getattr(args, "kvbm_llm_datadist_enable_cache_manager", True)),
            llm_datadist_enable_remote_cache_accessible=bool(
                getattr(args, "kvbm_llm_datadist_enable_remote_cache_accessible", False)
            ),
            pd_block_ownership=str(getattr(args, "kvbm_pd_block_ownership", "scheduler")),
        )
        if not installed:
            raise RuntimeError(f"KVBM multiprocess hook was not installed for provider {provider}")
        logger.info("Worker KVBM multiprocess hook enabled for provider %s", provider)
    except Exception as exc:
        raise RuntimeError(f"Worker KVBM engine-core hook setup failed for provider {provider}: {exc}") from exc


def _install_mooncake_diagnostics_hook(args) -> None:
    """Install native Mooncake timing hooks on decode workers only."""

    worker_mode = str(getattr(args, "worker_mode", "")).strip().lower()
    if worker_mode != "decode":
        return
    raw_config = getattr(args, "kv_transfer_config", None)
    if not raw_config:
        logger.info(
            "[MooncakeDiagnosticsSetup] enabled=False reason=missing_kv_transfer_config worker_mode=%s",
            worker_mode,
        )
        return
    try:
        if isinstance(raw_config, str):
            config = json.loads(raw_config)
        elif isinstance(raw_config, Mapping):
            config = dict(raw_config)
        elif callable(getattr(raw_config, "model_dump", None)):
            config = raw_config.model_dump()
        else:
            config = {
                "kv_connector": getattr(raw_config, "kv_connector", ""),
                "kv_role": getattr(raw_config, "kv_role", ""),
            }
    except (TypeError, ValueError) as exc:
        logger.warning(
            "[MooncakeDiagnosticsSetup] enabled=False "
            "reason=invalid_kv_transfer_config worker_mode=%s "
            "config_type=%s error=%s",
            worker_mode,
            type(raw_config).__name__,
            exc,
        )
        return
    if not isinstance(config, dict):
        logger.info(
            "[MooncakeDiagnosticsSetup] enabled=False reason=unsupported_config_type worker_mode=%s config_type=%s",
            worker_mode,
            type(raw_config).__name__,
        )
        return
    connector = str(config.get("kv_connector", "")).strip().lower()
    role = str(config.get("kv_role", "")).strip().lower()
    if "mooncake" not in connector:
        logger.info(
            "[MooncakeDiagnosticsSetup] enabled=False "
            "reason=non_mooncake_connector worker_mode=%s connector=%s role=%s",
            worker_mode,
            connector or "unavailable",
            role or "unavailable",
        )
        return
    if getattr(args, "kvbm_worker_enabled", False):
        logger.info(
            "[MooncakeDiagnosticsSetup] enabled=False reason=kvbm_owns_engine_core worker_mode=%s connector=%s role=%s",
            worker_mode,
            connector,
            role or "unavailable",
        )
        return
    try:
        from openjiuwentools.infer_router.vllm.mooncake_diagnostics import (
            install_mooncake_engine_core_diagnostics,
        )

        installed = install_mooncake_engine_core_diagnostics(log=logger)
        logger.info(
            "[MooncakeDiagnosticsSetup] enabled=%s reason=%s worker_mode=%s connector=%s role=%s config_type=%s",
            installed,
            "installed" if installed else "engine_core_hook_unavailable",
            worker_mode,
            connector,
            role or "unavailable",
            type(raw_config).__name__,
        )
    except Exception as exc:
        logger.warning(
            "[MooncakeDiagnosticsSetup] enabled=False reason=installation_failed "
            "worker_mode=%s connector=%s role=%s error=%s",
            worker_mode,
            connector,
            role or "unavailable",
            exc,
        )


def _normalize_zmq_endpoint(endpoint: str) -> str:
    for wildcard in ("tcp://*:", "tcp://:", "tcp://0.0.0.0:"):
        if endpoint.startswith(wildcard):
            return "tcp://127.0.0.1:" + endpoint[len(wildcard):]
    return endpoint


def _router_worker_type(worker_mode: str) -> str:
    """将 Worker 运行模式转换为 Router 使用的 Worker 类型。

    参数:
        worker_mode: Worker YAML 中的运行模式，支持 aggregated、prefill、decode。

    返回:
        Router `WorkerType` 对应的 combined、prefill 或 decode 字符串；未知模式
        按 combined 兼容处理。
    """

    return {
        "aggregated": "combined",
        "prefill": "prefill",
        "decode": "decode",
    }.get(str(worker_mode).strip().lower(), "combined")


async def _start_worker_publisher(
    kv_cfg: dict | None,
    relay_endpoint: str | None = None,
    worker_id: str = "",
    worker_mode: str = "aggregated",
    metrics_interval: float = 5.0,
) -> tuple:
    """启动 WorkerPublisher 及其子发布器，返回 (worker_pub, kv_pub, metrics_pub)。

    - 无 relay_endpoint 时退化为本地 KV 事件日志监听（调试模式）
    - 配置 enable_kv_cache_events 的 prefill/decode/aggregated 节点均上报 KV 事件
    - 所有节点均上报负载指标
    """
    from openjiuwentools.infer_router.vllm.worker_publisher import KvEventPublisher, MetricsPublisher, WorkerPublisher

    if not relay_endpoint:
        # 无 relay_endpoint 时仅本地监听 KV 事件日志
        if kv_cfg and kv_cfg.get("enable_kv_cache_events"):
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

    # 启动 KvEventPublisher。D 节点也必须转发删除、淘汰和清空事件，
    # 否则 Router 中由 PD attach 直接发布的 NPU 前缀会长期变成陈旧索引。
    kv_pub = None
    if kv_cfg and kv_cfg.get("enable_kv_cache_events"):
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


def _kvbm_prefetch_port(args) -> int:
    """计算 worker engine-core KVBM 控制端口。

    参数:
        args: worker argparse Namespace。

    返回:
        engine-core KVBM 控制端口；小于等于 0 表示禁用。
    """

    prefetch_port = getattr(args, "kvbm_prefetch_port", None)
    if prefetch_port is None:
        prefetch_port = int(getattr(args, "port", 8001)) + 10000
    return int(prefetch_port)


def _should_start_kvbm_engine_core_event_relay(args, worker_publisher: object | None) -> bool:
    """判断是否需要启动 engine-core KVBM 事件中继。

    参数:
        args: worker argparse Namespace。
        worker_publisher: 已启动的 WorkerPublisher；为空时无法向 router 上报。

    返回:
        满足 worker KVBM、vLLM-Ascend provider、publisher 可用时返回 True。
    """

    if worker_publisher is None:
        return False
    if not getattr(args, "kvbm_worker_enabled", False):
        return False
    from openjiuwentools.infer_router.vllm.kvbm_provider import (
        get_kvbm_provider_spec,
    )

    spec = get_kvbm_provider_spec(getattr(args, "kvbm_worker_provider", "auto"))
    return bool(spec and spec.event_relay)


def _start_kvbm_engine_core_event_relay(args, worker_publisher: object | None, worker_id: str):
    """启动 engine-core KVBM runtime 事件轮询中继任务。

    参数:
        args: worker argparse Namespace。
        worker_publisher: WorkerPublisher，用于向 router 发布 kv_events。
        worker_id: 当前 worker ID。

    返回:
        已启动的 asyncio.Task；不满足启动条件时返回 None。
    """

    if not _should_start_kvbm_engine_core_event_relay(args, worker_publisher):
        return None
    port = _kvbm_prefetch_port(args)
    if port <= 0:
        return None
    host = str(getattr(args, "kvbm_prefetch_host", "127.0.0.1"))
    task = asyncio.create_task(
        _run_kvbm_engine_core_event_relay(
            worker_publisher=worker_publisher,
            worker_id=worker_id,
            host=host,
            port=port,
        )
    )

    def _on_done(done):
        if done.cancelled():
            return
        exc = done.exception()
        if exc:
            logger.error("KVBM engine-core event relay crashed: %s", exc, exc_info=exc)

    task.add_done_callback(_on_done)
    logger.info("KVBM engine-core event relay started: http://%s:%s/events/drain", host, port)
    return task


async def _run_kvbm_engine_core_event_relay(
    *,
    worker_publisher: object,
    worker_id: str,
    host: str,
    port: int,
) -> None:
    """轮询 engine-core KVBM event outbox 并向 router 发布 kv_events。

    参数:
        worker_publisher: WorkerPublisher 实例。
        worker_id: 当前 worker ID。
        host: engine-core KVBM 控制服务 host。
        port: engine-core KVBM 控制服务端口。

    返回:
        无返回值；任务会一直运行直到被取消。
    """

    import httpx
    from openjiuwentools.infer_router.vllm.worker_publisher import TOPIC_KV_EVENTS

    seq = 0
    url = f"http://{host}:{port}/events/drain"
    capacity_task = asyncio.create_task(
        _run_kvbm_engine_core_capacity_relay(
            worker_publisher=worker_publisher,
            worker_id=worker_id,
            host=host,
            port=port,
        )
    )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1.5, connect=0.1)) as client:
            while True:
                try:
                    response = await client.post(url, json={})
                    if response.status_code == 200:
                        payload = response.json()
                        events = payload.get("events") or []
                        if events:
                            seq += 1
                            relay_msg = {
                                "worker_id": worker_id,
                                "seq": seq,
                                "ts": "",
                                "events": events,
                            }
                            await worker_publisher.publish(TOPIC_KV_EVENTS, relay_msg)
                            logger.info(
                                "KVBM engine-core event relay published %d location events",
                                len(events),
                            )
                except Exception as exc:
                    logger.debug("KVBM engine-core event relay poll skipped: %s", exc)
                await asyncio.sleep(0.5)
    finally:
        capacity_task.cancel()
        try:
            await capacity_task
        except asyncio.CancelledError:
            pass


async def _run_kvbm_engine_core_capacity_relay(
    *,
    worker_publisher: object,
    worker_id: str,
    host: str,
    port: int,
) -> None:
    """独立轮询engine-core BlockPool容量并向Router发布。

    参数:
        worker_publisher: WorkerPublisher实例，用于发布kv_metrics。
        worker_id: 当前Worker ID。
        host: engine-core KVBM控制服务host。
        port: engine-core KVBM控制服务端口。

    返回:
        无返回值；任务持续运行直到被取消。
    """

    import httpx
    from openjiuwentools.infer_router.vllm.worker_publisher import TOPIC_KV_METRICS

    url = f"http://{host}:{port}/datadist/block-pool/capacity"
    last_warning_at = 0.0
    relay_announced = False
    async with httpx.AsyncClient(timeout=httpx.Timeout(1.5, connect=0.1)) as client:
        while True:
            loop = asyncio.get_running_loop()
            iteration_started = loop.time()
            try:
                response = await client.post(url, json={})
                if response.status_code == 200:
                    block_pool = response.json().get("block_pool") or {}
                    free_blocks = block_pool.get("free_blocks")
                    block_size = block_pool.get("block_size")
                    capacity_generation = block_pool.get("capacity_generation")
                    if free_blocks is not None:
                        metrics_payload = {
                            "worker_id": worker_id,
                            "kvbm_free_blocks": max(int(free_blocks), 0),
                        }
                        if block_size is not None and int(block_size) > 0:
                            metrics_payload["kvbm_block_size"] = int(block_size)
                        if capacity_generation is not None:
                            metrics_payload["kvbm_capacity_generation"] = int(capacity_generation)
                        await worker_publisher.publish(
                            TOPIC_KV_METRICS,
                            metrics_payload,
                        )
                        if not relay_announced:
                            logger.info(
                                "KVBM engine-core BlockPool capacity relay ready: "
                                "worker=%s free_blocks=%s block_size=%s",
                                worker_id,
                                free_blocks,
                                block_size,
                            )
                            relay_announced = True
                    elif iteration_started - last_warning_at >= 10.0:
                        logger.warning(
                            "KVBM engine-core BlockPool capacity unavailable: worker=%s response=%s",
                            worker_id,
                            response.text[:512],
                        )
                        last_warning_at = iteration_started
                elif iteration_started - last_warning_at >= 10.0:
                    logger.warning(
                        "KVBM engine-core BlockPool capacity query failed: worker=%s status=%s response=%s",
                        worker_id,
                        response.status_code,
                        response.text[:512],
                    )
                    last_warning_at = iteration_started
            except Exception as exc:
                if iteration_started - last_warning_at >= 10.0:
                    logger.warning(
                        "KVBM engine-core BlockPool capacity query skipped: worker=%s error=%s",
                        worker_id,
                        exc,
                    )
                    last_warning_at = iteration_started
            elapsed = loop.time() - iteration_started
            await asyncio.sleep(max(1.0 - elapsed, 0.05))


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
    kvbm_adapter: object = None
    kvbm_event_relay_task: asyncio.Task | None = None


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
    if ctx.kvbm_adapter:
        await ctx.kvbm_adapter.stop()
    if ctx.kvbm_event_relay_task:
        ctx.kvbm_event_relay_task.cancel()
        try:
            await ctx.kvbm_event_relay_task
        except asyncio.CancelledError:
            pass
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
        shutdown_task = asyncio.create_task(_graceful_shutdown(ctx))
        shutdown_task.add_done_callback(_on_done)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handler)
    logger.info("Signal handlers installed for graceful shutdown (grace=%.1fs)", _DEFAULT_GRACE_PERIOD_SECS)


async def worker(args=None) -> None:
    """Worker 主协程：初始化引擎、启动服务、注册发现、上报指标，阻塞直到退出。"""
    if args is None:
        args = parse_args()

    # 必须在 transport、KVBM hook 或 vLLM 的任何模块导入前设置，确保 spawn
    # 出来的 P/D engine-core 对首个 prefix block 使用同一个 hash 链种子。
    _configure_prefix_caching_hash_seed(args)
    from openjiuwentools.infer_router.vllm.transport import serve_http, serve_tcp

    _install_kvbm_engine_core_hook(args)
    _install_mooncake_diagnostics_hook(args)
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
            worker_type=_router_worker_type(args.worker_mode),
            metadata={
                "model": model_name,
                "request_plane": args.request_plane,
                "worker_mode": args.worker_mode,
                "kvbm_control_url": str(getattr(args, "kvbm_control_url", "")).strip(),
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

    kvbm_event_relay_task = _start_kvbm_engine_core_event_relay(
        args,
        worker_publisher,
        worker_id,
    )

    kvbm_adapter = None
    try:
        from openjiuwentools.infer_router.vllm.kvbm_worker_adapter import (
            VllmWorkerKVBMAdapter,
            VllmWorkerKVBMConfig,
        )

        kvbm_adapter = VllmWorkerKVBMAdapter(
            config=VllmWorkerKVBMConfig.from_args(args, worker_id),
            engine=engine,
            worker_publisher=worker_publisher,
        )
        await kvbm_adapter.start()
        if kv_publisher and hasattr(kv_publisher, "set_local_event_sink"):
            kv_publisher.set_local_event_sink(kvbm_adapter.apply_kv_events)
        if metrics_publisher:
            metrics_publisher.set_extra_metrics_source(kvbm_adapter)
    except Exception as exc:
        logger.warning("Worker KVBM adapter setup skipped: %s", exc)

    shutdown_ctx = ShutdownContext(
        server_close=close_fn,
        engine=engine,
        shutdown_event=shutdown_event,
        registry=registry,
        worker_publisher=worker_publisher,
        kv_publisher=kv_publisher,
        metrics_publisher=metrics_publisher,
        kvbm_adapter=kvbm_adapter,
        kvbm_event_relay_task=kvbm_event_relay_task,
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
