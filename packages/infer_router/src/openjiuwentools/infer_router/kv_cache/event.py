import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CacheLocation(Enum):
    """缓存位置"""

    NPU = "npu"
    CPU = "cpu"
    REMOTE = "remote"


@dataclass
class CacheEvent:
    """缓存事件"""

    event_type: str  # store | evict | hit | transfer | removed | Cleared
    block_hashes: list[str] = field(default_factory=list)
    location: CacheLocation = CacheLocation.NPU
    timestamp: float = field(default_factory=time.time)
    engine_specific: dict[str, Any] = field(default_factory=dict)
    token_ids: list[int] = field(default_factory=list)
    worker_id: str = ""
    token_count: int = 0

    @property
    def effective_token_count(self) -> int:
        """有效 token 数量（优先使用 token_count，如果为 0 则从 token_ids 计算）"""
        if self.token_count > 0:
            return self.token_count
        return len(self.token_ids)
