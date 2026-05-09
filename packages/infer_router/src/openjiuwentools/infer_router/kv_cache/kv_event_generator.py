from loguru import logger

from openjiuwentools.infer_router.kv_cache.event import CacheEvent
from openjiuwentools.infer_router.kv_cache.worker_token_manager import (
    WorkerTokenManager,
)


class KVEventGenerator:
    """KV 事件生成器（仅实现二使用）"""

    def __init__(self, enable_radix_tree: bool = False):
        """初始化事件生成器

        Args:
            enable_radix_tree: 是否启用 Radix Tree，禁用时不生成事件
        """
        self.worker_token_manager = WorkerTokenManager()
        self.enable_radix_tree = enable_radix_tree
        logger.info(f"KVEventGenerator initialized (enable_radix_tree={enable_radix_tree})")

    def generate_events(self, worker_id: str, token_ids: list[int]) -> list[CacheEvent]:
        """生成事件

        Args:
            worker_id: 工作器 ID
            token_ids: token ID 列表

        Returns:
            CacheEvent 对象列表

        """
        # 如果 Radix Tree 未启用，不生成任何事件
        if not self.enable_radix_tree:
            logger.debug(f"Radix Tree is disabled, skipping event generation for worker {worker_id}")
            return []

        if not token_ids:
            logger.debug(f"No tokens to process for worker {worker_id}")
            return []

        dict_events = self.worker_token_manager.add_tokens(worker_id, token_ids)
        events = []

        for event in dict_events:
            cache_event = CacheEvent(
                event_type=event["event_type"],
                token_ids=event.get("token_ids", []),
                worker_id=worker_id,
                engine_specific={"worker_id": worker_id},
            )
            events.append(cache_event)

        logger.debug(f"Generated {len(events)} events for worker {worker_id}")
        return events

    def clear_worker_events(self, worker_id: str) -> list[CacheEvent]:
        """清空工作器的事件

        Args:
            worker_id: 工作器 ID

        Returns:
            事件列表

        """
        dict_events = self.worker_token_manager.clear_worker_tokens(worker_id)
        events = []

        for event in dict_events:
            cache_event = CacheEvent(
                event_type=event["event_type"],
                token_ids=event.get("token_ids", []),
                worker_id=worker_id,
                engine_specific={"worker_id": worker_id},
            )
            events.append(cache_event)

        logger.debug(f"Generated {len(events)} clear events for worker {worker_id}")
        return events

    def remove_worker(self, worker_id: str) -> None:
        """移除工作器

        Args:
            worker_id: 工作器 ID

        """
        self.worker_token_manager.remove_worker(worker_id)
        logger.debug(f"Removed worker {worker_id} from event generator")

    def get_worker_token_count(self, worker_id: str) -> int:
        """获取工作器的 token 数量

        Args:
            worker_id: 工作器 ID

        Returns:
            token 数量

        """
        return self.worker_token_manager.get_worker_token_count(worker_id)

    def get_all_workers(self) -> list[str]:
        """获取所有工作器 ID

        Returns:
            工作器 ID 列表

        """
        return self.worker_token_manager.get_all_workers()

    def get_stats(self) -> dict[str, any]:
        """获取统计信息

        Returns:
            统计信息

        """
        stats = self.worker_token_manager.get_stats()
        logger.debug(f"Event generator stats: {stats}")
        return stats
