from __future__ import annotations

import time
from collections.abc import Callable

from loguru import logger

from openjiuwentools.infer_router.config.config import settings
from openjiuwentools.infer_router.kv_cache.event import CacheEvent
from openjiuwentools.infer_router.monitoring.metrics import metrics


class KVEventManager:
    """KV 事件管理器（完全独立）"""

    def __init__(self):
        """初始化事件管理器"""
        self.events: list[CacheEvent] = []
        self.event_buffer_size = settings.kv_event_buffer_size
        self.event_stats = {
            "store": 0,
            "removed": 0,
            "Cleared": 0,
            "evict": 0,
            "hit": 0,
            "total": 0,
        }
        self.event_handlers: dict[str, list[Callable]] = {}
        logger.info("KVEventManager initialized")

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """注册事件处理器

        Args:
            event_type: 事件类型
            handler: 事件处理器函数

        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Registered handler for event type: {event_type}")

    def process_event(self, event: CacheEvent) -> bool:
        """处理事件

        Args:
            event: 缓存事件

        Returns:
            是否处理成功

        """
        start_time = time.time()
        try:
            # 记录事件
            self.events.append(event)
            if len(self.events) > self.event_buffer_size:
                self.events.pop(0)

            # 更新统计
            self.event_stats[event.event_type] = self.event_stats.get(event.event_type, 0) + 1
            self.event_stats["total"] += 1

            # 记录监控指标
            worker_id = event.engine_specific.get("worker_id", "unknown")
            metrics.record_event(event.event_type, worker_id)
            metrics.record_event_latency(event.event_type, time.time() - start_time)

            # 调用所有对应的处理器
            if event.event_type in self.event_handlers:
                success_count = 0
                for handler in self.event_handlers[event.event_type]:
                    try:
                        handler(event)
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Error in event handler for {event.event_type}: {e}")
                logger.info(
                    f"[KV_EVENT] {event.event_type}: block_hashes={event.block_hashes}, "
                    f"worker={event.engine_specific.get('worker_id', '?')}, "
                    f"handlers={success_count}/{len(self.event_handlers[event.event_type])}"
                )
                return success_count > 0
            else:
                logger.warning(f"No handler for event type: {event.event_type}")
                return False

        except Exception as e:
            logger.error(f"Error processing event: {e}")
            return False

    def get_event_stats(self) -> dict:
        """获取事件统计信息

        Returns:
            事件统计字典

        """
        return self.event_stats.copy()

    def get_recent_events(self, count: int = 10) -> list[CacheEvent]:
        """获取最近的事件

        Args:
            count: 事件数量

        Returns:
            最近的事件列表

        """
        return self.events[-count:]

    def clear_events(self) -> None:
        """清空事件缓冲区"""
        self.events.clear()
        logger.debug("Event buffer cleared")
