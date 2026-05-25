import time

from loguru import logger

from openjiuwentools.infer_router.config.config import settings


class WorkerTokenManager:
    """工作器 Token 管理器（实现二）"""

    def __init__(self):
        """初始化工作器 Token 管理器"""
        self.worker_tokens: dict[
            str, list[tuple[int, float]]
        ] = {}  # worker_id -> [(token_id, timestamp)]
        self.token_capacity = settings.worker_token_capacity
        self.block_size = settings.kv_cache_block_size
        logger.info(f"WorkerTokenManager initialized with capacity: {self.token_capacity}")

    def add_tokens(self, worker_id: str, token_ids: list[int]) -> list[dict]:
        """添加 token 到工作器

        Args:
            worker_id: 工作器 ID
            token_ids: token ID 列表

        Returns:
            生成的事件列表

        """
        events = []

        if worker_id not in self.worker_tokens:
            self.worker_tokens[worker_id] = []
            logger.debug(f"Created token list for worker: {worker_id}")

        # 添加所有 token（不立即生成事件）
        for token_id in token_ids:
            self.worker_tokens[worker_id].append((token_id, time.time()))
            logger.debug(f"Added token {token_id} to worker {worker_id}")

        # 收集需要 evict 的 token（合并成一个事件）
        evicted_tokens = []
        while len(self.worker_tokens[worker_id]) > self.token_capacity:
            removed_token, _ = self.worker_tokens[worker_id].pop(0)
            evicted_tokens.append(removed_token)
            logger.debug(
                f"Removed token {removed_token} from worker {worker_id} (capacity exceeded)"
            )

        # 生成合并后的事件
        events = []

        # 合并的 store 事件
        events.append(
            {
                "event_type": "store",
                "token_count": len(token_ids),
                "worker_id": worker_id,
                "token_ids": token_ids,
            }
        )

        # 合并的 evict 事件（如果有需要 evict 的 token）
        if evicted_tokens:
            events.append(
                {
                    "event_type": "evict",
                    "token_count": len(evicted_tokens),
                    "worker_id": worker_id,
                    "token_ids": evicted_tokens,
                }
            )

        logger.debug(
            f"Added {len(token_ids)} tokens to worker {worker_id}, generated {len(events)} events"
        )
        return events

    def clear_worker_tokens(self, worker_id: str) -> list[dict]:
        """清空工作器的 token

        Args:
            worker_id: 工作器 ID

        Returns:
            生成的事件列表

        """
        if worker_id in self.worker_tokens:
            tokens = [tid for tid, _ in self.worker_tokens[worker_id]]

            events = [
                {
                    "event_type": "Cleared",
                    "token_count": len(tokens),
                    "worker_id": worker_id,
                    "token_ids": tokens,
                }
            ]
            token_count = len(tokens)
            del self.worker_tokens[worker_id]
            logger.debug(f"Cleared {token_count} tokens from worker {worker_id}")
            return events
        return []

    def get_worker_token_count(self, worker_id: str) -> int:
        """获取工作器的 token 数量

        Args:
            worker_id: 工作器 ID

        Returns:
            token 数量

        """
        if worker_id in self.worker_tokens:
            return len(self.worker_tokens[worker_id])
        return 0

    def get_worker_tokens(self, worker_id: str) -> list[int]:
        """获取工作器的 token 列表

        Args:
            worker_id: 工作器 ID

        Returns:
            token ID 列表

        """
        if worker_id in self.worker_tokens:
            return [tid for tid, _ in self.worker_tokens[worker_id]]
        return []

    def has_token(self, worker_id: str, token_id: int) -> bool:
        """检查工作器是否包含指定 token

        Args:
            worker_id: 工作器 ID
            token_id: token ID

        Returns:
            是否包含

        """
        if worker_id in self.worker_tokens:
            return any(tid == token_id for tid, _ in self.worker_tokens[worker_id])
        return False

    def remove_worker(self, worker_id: str) -> None:
        """移除工作器

        Args:
            worker_id: 工作器 ID

        """
        if worker_id in self.worker_tokens:
            del self.worker_tokens[worker_id]
            logger.debug(f"Removed worker {worker_id} from token manager")

    def get_all_workers(self) -> list[str]:
        """获取所有工作器 ID

        Returns:
            工作器 ID 列表

        """
        return list(self.worker_tokens.keys())

    def get_stats(self) -> dict[str, any]:
        """获取统计信息

        Returns:
            统计信息

        """
        stats = {
            "total_workers": len(self.worker_tokens),
            "total_tokens": sum(len(tokens) for tokens in self.worker_tokens.values()),
            "capacity": self.token_capacity,
            "block_size": self.block_size,
            "workers": {},
        }

        for worker_id, tokens in self.worker_tokens.items():
            stats["workers"][worker_id] = {
                "token_count": len(tokens),
                "block_count": len(tokens) // self.block_size,
                "capacity_usage": len(tokens) / self.token_capacity
                if self.token_capacity > 0
                else 0,
            }

        return stats
