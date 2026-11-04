"""KV cache 事件处理器，作为 WorkerEventSubscriber 的 topic handler 注册。

将 worker 上报的 vLLM KV cache 事件转换为内部 CacheEvent 并更新缓存状态。
"""

from __future__ import annotations

from loguru import logger

from openjiuwentools.infer_router.kv_cache.event import CacheEvent, cache_location_from_medium
from openjiuwentools.infer_router.kv_cache.kv_event_manager import KVEventManager

_TYPE_MAPPING = {
    "BlockStored": "store",
    "BlockRemoved": "removed",
    "BlockEvicted": "evict",
    "BlockCleared": "cleared",
    "BlockHit": "hit",
    "BlockMoved": "move",
    "BlockTransferred": "move",
}


class KVEventHandler:
    """处理 kv_events topic 的消息，将 vLLM 事件转为 CacheEvent。"""

    def __init__(
        self,
        event_manager: KVEventManager,
    ) -> None:
        self.event_manager = event_manager

    def __call__(self, worker_id: str, msg: dict) -> None:
        events = msg.get("events", [])
        logger.debug(f"[KV_EVENT] Received {len(events)} events from {worker_id}")
        for event_data in events:
            event_type_name = event_data.get("type", "")
            data = event_data.get("data", {})
            cache_event = self._convert(worker_id, event_type_name, data)
            if cache_event:
                # KVCacheManager已经向同一个event_manager注册了状态处理器；
                # process_event会完成索引更新，无需再次update_cache_state。
                self.event_manager.process_event(cache_event)

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
        medium = data.get("medium") or data.get("location") or data.get("to") or "GPU"
        physical_block_ids = data.get("physical_block_ids")
        if physical_block_ids is None and data.get("physical_block_id") is not None:
            physical_block_ids = [data["physical_block_id"]]

        return CacheEvent(
            event_type=event_type,
            block_hashes=block_hashes,
            location=cache_location_from_medium(medium),
            token_ids=token_ids,
            worker_id=worker_id,
            token_count=len(token_ids) if token_ids else block_size,
            engine_specific={
                "worker_id": worker_id,
                "parent_block_hash": str(data.get("parent_block_hash", "")) if data.get("parent_block_hash") else None,
                "block_size": block_size,
                "medium": medium,
                "from": data.get("from") or data.get("source"),
                "to": data.get("to") or data.get("destination") or medium,
                "physical_block_ids": physical_block_ids or [],
            },
        )
