"""KV cache 事件处理器，作为 WorkerEventSubscriber 的 topic handler 注册。

将 worker 上报的 vLLM KV cache 事件转换为内部 CacheEvent 并更新缓存状态。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from openjiuwentools.infer_router.kv_cache.event import CacheEvent

if TYPE_CHECKING:
    from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
    from openjiuwentools.infer_router.kv_cache.kv_event_manager import KVEventManager

_TYPE_MAPPING = {
    "BlockStored": "store",
    "BlockRemoved": "evict",
    "BlockCleared": "Cleared",
}


class KVEventHandler:
    """处理 kv_events topic 的消息，将 vLLM 事件转为 CacheEvent。"""

    def __init__(
        self,
        event_manager: KVEventManager,
        kvcache_manager: KVCacheManager,
    ) -> None:
        self.event_manager = event_manager
        self.kvcache_manager = kvcache_manager

    def __call__(self, worker_id: str, msg: dict) -> None:
        from openjiuwentools.infer_router.kv_cache.kv_cache import CacheStateUpdate

        events = msg.get("events", [])
        logger.debug(f"[KV_EVENT] Received {len(events)} events from {worker_id}")
        for event_data in events:
            event_type_name = event_data.get("type", "")
            data = event_data.get("data", {})
            cache_event = self._convert(worker_id, event_type_name, data)
            if cache_event:
                self.event_manager.process_event(cache_event)
                self.kvcache_manager.update_cache_state(
                    CacheStateUpdate(
                        worker_id=worker_id,
                        event_type=cache_event.event_type,
                        token_ids=cache_event.token_ids,
                        token_count=cache_event.token_count,
                    )
                )

    @staticmethod
    def _convert(
        worker_id: str, event_type_name: str, data: dict
    ) -> CacheEvent | None:
        event_type = _TYPE_MAPPING.get(event_type_name)
        if not event_type:
            logger.debug(f"Ignoring unknown KV event type: {event_type_name}")
            return None

        block_hashes = [str(h) for h in data.get("block_hashes", [])]
        token_ids = data.get("token_ids", [])
        block_size = data.get("block_size", 16)

        return CacheEvent(
            event_type=event_type,
            block_hashes=block_hashes,
            token_ids=token_ids,
            worker_id=worker_id,
            token_count=len(token_ids) if token_ids else block_size,
            engine_specific={
                "worker_id": worker_id,
                "parent_block_hash": str(data.get("parent_block_hash", "")) if data.get("parent_block_hash") else None,
                "block_size": block_size,
                "medium": data.get("medium", "GPU"),
            },
        )
