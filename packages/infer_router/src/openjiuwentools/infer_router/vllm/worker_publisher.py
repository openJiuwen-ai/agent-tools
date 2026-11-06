"""Worker 端 ZMQ 消息发布模块，负责将 KV cache 事件和负载指标推送给 router。

包含三个核心组件：
- WorkerPublisher: 管理 ZMQ PUB socket，提供统一的消息发布接口
- KvEventPublisher: 订阅 vLLM 的 KV cache 事件并通过 WorkerPublisher 转发
- MetricsPublisher: 周期性采集并上报 worker 负载指标

数据流::

    vLLM PUB (msgpack) → [SUB] KvEventPublisher ─┐
                                                  ├→ WorkerPublisher [PUB] → router
                          MetricsPublisher ───────┘
"""

import asyncio
import logging
from typing import Any, Callable

import msgspec

logger = logging.getLogger("jiuwen.kv_publisher")

TOPIC_KV_EVENTS = "kv_events"
TOPIC_KV_METRICS = "kv_metrics"


class WorkerPublisher:
    """ZMQ PUB socket 管理器，提供统一的消息发布接口。

    - 所有消息类型共享同一个 PUB socket（同一端口），通过 topic 前缀区分
    - 支持两种 topic: kv_events（KV cache 事件）、kv_metrics（负载指标）
    - 使用 msgspec msgpack 编码，保证高效序列化
    """

    def __init__(self, pub_endpoint: str) -> None:
        self.pub_endpoint = pub_endpoint
        self._ctx = None
        self._pub_sock = None
        self._encoder = msgspec.msgpack.Encoder()

    async def start(self) -> None:
        import zmq
        import zmq.asyncio

        self._ctx = zmq.asyncio.Context()
        self._pub_sock = self._ctx.socket(zmq.PUB)
        self._pub_sock.bind(self.pub_endpoint)
        logger.info("WorkerPublisher started: %s", self.pub_endpoint)

    async def stop(self) -> None:
        if self._pub_sock:
            self._pub_sock.close()
        if self._ctx:
            self._ctx.term()
        self._pub_sock = None
        self._ctx = None
        logger.info("WorkerPublisher stopped")

    async def publish(self, topic: str, data: dict) -> None:
        if not self._pub_sock:
            raise RuntimeError("WorkerPublisher not started")
        payload = self._encoder.encode(data)
        await self._pub_sock.send_multipart([topic.encode(), payload])


class KvEventPublisher:
    """vLLM KV cache 事件中继器。

    - 订阅 vLLM 内部的 ZMQ PUB 端点，接收 KVEventBatch 消息
    - 将事件解码后转为可序列化的 dict，以 kv_events topic 通过 WorkerPublisher 转发
    - 每个实例对应一个 vLLM 引擎的 KV 事件流
    """

    def __init__(
        self,
        worker_publisher: WorkerPublisher,
        sub_endpoint: str,
        topic: str = "",
        worker_id: str = "",
        local_event_sink: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        self.worker_publisher = worker_publisher
        self.sub_endpoint = sub_endpoint
        self.topic = topic
        self.worker_id = worker_id
        self.local_event_sink = local_event_sink
        self._task: asyncio.Task | None = None

    def set_local_event_sink(self, sink: Callable[[list[Any]], Any] | None) -> None:
        """设置本地 KV 事件消费入口。

        参数:
            sink: 接收 vLLM KV event 对象列表的回调；传入 None 表示取消消费。

        返回:
            无返回值。
        """

        self.local_event_sink = sink

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        logger.info(
            "KvEventPublisher started: sub=%s worker=%s",
            self.sub_endpoint, self.worker_id,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        logger.info("KvEventPublisher stopped")

    async def _run(self) -> None:
        import zmq
        import zmq.asyncio
        from vllm.distributed.kv_events import KVEventBatch

        ctx = zmq.asyncio.Context()
        sub_sock = ctx.socket(zmq.SUB)
        sub_sock.connect(self.sub_endpoint)
        sub_sock.setsockopt(zmq.SUBSCRIBE, self.topic.encode())

        decoder = msgspec.msgpack.Decoder(KVEventBatch)
        logger.info(
            "KV event relay running: SUB %s → PUB (topic=%s)",
            self.sub_endpoint, TOPIC_KV_EVENTS,
        )

        try:
            while True:
                _topic_bytes, seq_bytes, payload = await sub_sock.recv_multipart()
                seq = int.from_bytes(seq_bytes, "big")
                batch: KVEventBatch = decoder.decode(payload)

                for event in batch.events:
                    logger.info(
                        "KV event seq=%d ts=%s type=%s event=%s",
                        seq, batch.ts, type(event).__name__, event,
                    )

                await self._apply_local_event_sink(batch.events)
                relay_msg = {
                    "worker_id": self.worker_id,
                    "seq": seq,
                    "ts": str(batch.ts),
                    "events": [
                        {
                            "type": type(e).__name__,
                            "data": _event_to_dict(e),
                        }
                        for e in batch.events
                    ],
                }
                await self.worker_publisher.publish(TOPIC_KV_EVENTS, relay_msg)
        finally:
            sub_sock.close()
            ctx.term()

    async def _apply_local_event_sink(self, events: list[Any]) -> None:
        """把 vLLM KV 事件同步投递给本地消费者。

        参数:
            events: 当前 KVEventBatch 中的事件对象列表。

        返回:
            无返回值。
        """

        if self.local_event_sink is None:
            return
        try:
            result = self.local_event_sink(events)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.warning("Local KV event sink failed: %s", exc)


class MetricsPublisher:
    """Worker 负载指标发布器。

    - 按固定间隔采集 worker 运行时指标（如 KV cache 使用率、请求数等）
    - 通过 WorkerPublisher 以 kv_metrics topic 推送给 router
    - 指标来源通过 set_metrics_source() 注入（如 WorkerStatLogger）
    """

    def __init__(
        self,
        worker_publisher: WorkerPublisher,
        worker_id: str = "",
        interval: float = 5.0,
    ) -> None:
        self.worker_publisher = worker_publisher
        self.worker_id = worker_id
        self.interval = interval
        self._task: asyncio.Task | None = None
        self._metrics_source = None
        self._extra_metrics_sources = []

    def set_metrics_source(self, source) -> None:
        """设置指标采集源（如 WorkerStatLogger），需实现 get_metrics() 方法。"""
        self._metrics_source = source

    def set_extra_metrics_source(self, source) -> None:
        """追加额外指标采集源。

        参数:
            source: 实现 get_metrics() 方法的指标源。

        返回:
            无返回值。
        """
        self._extra_metrics_sources.append(source)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        logger.info(
            "MetricsPublisher started: worker=%s interval=%.1fs",
            self.worker_id, self.interval,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        logger.info("MetricsPublisher stopped")

    async def _run(self) -> None:
        while True:
            try:
                metrics_data = self._collect_metrics()
                await self.worker_publisher.publish(TOPIC_KV_METRICS, metrics_data)
                logger.debug("Published metrics for worker %s", self.worker_id)
            except Exception as e:
                logger.error("Failed to publish metrics: %s", e)
            await asyncio.sleep(self.interval)

    def _collect_metrics(self) -> dict:
        """采集当前负载指标，合并 metrics_source 提供的运行时数据。"""
        data = {
            "worker_id": self.worker_id,
            "type": "worker_metrics",
        }
        if self._metrics_source and hasattr(self._metrics_source, "get_metrics"):
            data.update(self._metrics_source.get_metrics())
        for source in self._extra_metrics_sources:
            if hasattr(source, "get_metrics"):
                data.update(source.get_metrics())
        return data


def _event_to_dict(event) -> dict:
    """将 vLLM KV event 对象转为可序列化的 dict。
    兼容 msgspec.Struct（使用 __struct_fields__）和普通对象（使用 __dict__）。
    """
    if hasattr(event, "__struct_fields__"):
        payload = {
            key: _make_serializable(getattr(event, key))
            for key in event.__struct_fields__
        }
    elif hasattr(event, "__dict__"):
        payload = {
            key: _make_serializable(value)
            for key, value in event.__dict__.items()
        }
    else:
        return {"repr": repr(event)}
    _add_inferred_block_size(payload)
    return payload


def _add_inferred_block_size(payload: dict[str, Any]) -> None:
    """为缺少 block_size 的 vLLM store 事件补充可验证的尺寸。

    参数:
        payload: 已转换为可序列化字典的 vLLM KV event。

    返回:
        无返回值；只有 token_ids 能被 block_hashes 等分时才原地补充 block_size。

    说明:
        真实 tensor shape 仍是 KVBM engine-core 的最终事实来源。本推导用于缩短
        原生 token 链事件与 KVBM 真实尺寸 relay 之间的纠正窗口。
    """

    if payload.get("block_size") is not None:
        return
    token_ids = payload.get("token_ids")
    block_hashes = payload.get("block_hashes")
    if not isinstance(token_ids, list | tuple) or not token_ids:
        return
    if not isinstance(block_hashes, list | tuple) or not block_hashes:
        return
    block_count = len(block_hashes)
    if len(token_ids) % block_count != 0:
        return
    inferred = len(token_ids) // block_count
    if inferred > 0:
        payload["block_size"] = inferred


def _make_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "__struct_fields__"):
        return {k: _make_serializable(getattr(obj, k)) for k in obj.__struct_fields__}
    if hasattr(obj, "__dict__"):
        return {k: _make_serializable(v) for k, v in obj.__dict__.items()}
    return str(obj)
