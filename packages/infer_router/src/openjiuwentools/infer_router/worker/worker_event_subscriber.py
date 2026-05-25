"""通用 ZMQ 事件订阅器，管理与 worker 的 ZMQ SUB 连接并按 topic 分发消息。

每个 worker 一个 ZMQ SUB 连接，通过 topic 前缀匹配将消息路由给注册的 handler。
订阅器本身不包含任何业务逻辑。

用法::

    subscriber = WorkerEventSubscriber()
    subscriber.register_handler("kv_events", kv_event_handler)
    subscriber.register_handler("kv_metrics", metrics_handler)
    subscriber.subscribe("worker-1", "tcp://127.0.0.1:9010")
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from loguru import logger

TopicHandler = Callable[[str, dict], None]


class WorkerEventSubscriber:
    """通用 worker 事件订阅器。

    - 每个 worker 一个 ZMQ SUB 连接
    - 按 topic 分发到注册的 handler
    - handler 签名: (worker_id: str, msg: dict) -> None
    """

    def __init__(self) -> None:
        self._handlers: dict[str, TopicHandler] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def register_handler(self, topic: str, handler: TopicHandler) -> None:
        self._handlers[topic] = handler

    def subscribe(self, worker_id: str, endpoint: str) -> None:
        if worker_id in self._tasks:
            logger.warning(f"Already subscribed to worker {worker_id}, skipping")
            return
        topics = list(self._handlers.keys())
        if not topics:
            logger.warning("No handlers registered, skipping subscription")
            return
        task = asyncio.create_task(self._subscribe_loop(worker_id, endpoint, topics))
        self._tasks[worker_id] = task
        logger.info(
            f"WorkerEventSubscriber: subscribed to {worker_id} at {endpoint}, "
            f"topics={topics}"
        )

    def unsubscribe(self, worker_id: str) -> None:
        task = self._tasks.pop(worker_id, None)
        if task:
            task.cancel()
            logger.info(f"WorkerEventSubscriber: unsubscribed from {worker_id}")

    async def stop(self) -> None:
        for worker_id in list(self._tasks):
            self.unsubscribe(worker_id)
        logger.info("WorkerEventSubscriber stopped")

    async def _subscribe_loop(
        self, worker_id: str, endpoint: str, topics: list[str]
    ) -> None:
        import msgspec
        import zmq
        import zmq.asyncio

        ctx = zmq.asyncio.Context()
        sub_sock = ctx.socket(zmq.SUB)
        sub_sock.connect(endpoint)

        for topic in topics:
            sub_sock.setsockopt(zmq.SUBSCRIBE, topic.encode())

        decoder = msgspec.msgpack.Decoder()
        logger.info(f"WorkerEventSubscriber: listening on {endpoint} for {worker_id}")

        try:
            while True:
                parts = await sub_sock.recv_multipart()
                if len(parts) < 2:
                    continue
                topic_str = parts[0].decode("utf-8", errors="replace")
                try:
                    msg = decoder.decode(parts[-1])
                except Exception as e:
                    logger.error(f"Failed to decode message from {worker_id}: {e}")
                    continue

                handler = self._handlers.get(topic_str)
                if handler:
                    try:
                        handler(worker_id, msg)
                    except Exception as e:
                        logger.error(
                            f"Handler error for topic={topic_str} worker={worker_id}: {e}"
                        )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WorkerEventSubscriber: loop error for {worker_id}: {e}")
        finally:
            sub_sock.close()
            ctx.term()
