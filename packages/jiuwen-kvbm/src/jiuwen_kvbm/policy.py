from __future__ import annotations

from collections.abc import Iterable

from jiuwen_kvbm.types import BlockState, KVBlockMetadata


class LRUEvictionPolicy:
    """按最近最少访问策略选择可淘汰 Block。"""

    @staticmethod
    def select(blocks: Iterable[KVBlockMetadata], limit: int) -> list[KVBlockMetadata]:
        """从未被引用的缓存 Block 中选择最旧的若干个。

        参数:
            blocks: 候选 Block 元数据集合。
            limit: 最多返回的 Block 数量。

        返回:
            按 last_accessed 从旧到新排序后的可淘汰 Block 列表。
        """
        candidates = []
        for selected_block in blocks:
            if selected_block.ref_count == 0 and selected_block.state in {
                BlockState.CACHED_NPU,
                BlockState.CACHED_RAM,
                BlockState.CACHED_DISK,
            }:
                candidates.append(selected_block)
        return sorted(candidates, key=lambda block: block.last_accessed)[:limit]
