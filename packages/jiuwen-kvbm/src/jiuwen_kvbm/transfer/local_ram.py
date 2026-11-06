from __future__ import annotations

from typing import Any
import threading

from jiuwen_kvbm.types import KVBlockMetadata
from jiuwen_kvbm.util import estimate_payload_size


class LocalRamBackend:
    """本地 RAM 存储后端。

    当前实现使用进程内字典保存载荷，后续可替换为 pinned host memory 或
    昇腾侧推荐的主机内存管理方式。

    参数:
        max_bytes: RAM 后端允许占用的最大字节数；为 None 时不做容量限制。
    """

    def __init__(self, max_bytes: int | None = None):
        self.max_bytes = max_bytes
        self._payloads: dict[str, Any] = {}
        self._sizes: dict[str, int] = {}
        self._lock = threading.RLock()

    @property
    def bytes_used(self) -> int:
        """返回当前 RAM 后端保存的载荷总大小。

        返回:
            当前已保存载荷的总字节数。
        """
        with self._lock:
            return sum(self._sizes.values())

    def put(self, metadata: KVBlockMetadata, payload: Any) -> str:
        """将 Block 载荷保存到本地 RAM。

        参数:
            metadata: 待保存 Block 的元数据。
            payload: 待保存的 Block 载荷。

        返回:
            RAM 后端内部引用字符串。

        异常:
            MemoryError: 保存后会超过 max_bytes 容量限制。
        """
        with self._lock:
            size = estimate_payload_size(payload)
            ref = _ref(metadata)
            if self.max_bytes is not None and self.bytes_used - self._sizes.get(ref, 0) + size > self.max_bytes:
                raise MemoryError("LocalRamBackend capacity exceeded")
            self._payloads[ref] = payload
            self._sizes[ref] = size
            return ref

    def get(self, metadata: KVBlockMetadata) -> Any:
        """从本地 RAM 读取 Block 载荷。

        参数:
            metadata: 包含 RAM 后端 store_ref 的 Block 元数据。

        返回:
            保存于 RAM 中的 Block 载荷。

        异常:
            KeyError: metadata 缺少 store_ref 或引用不存在。
        """
        with self._lock:
            if metadata.store_ref is None:
                raise KeyError("metadata has no store_ref")
            return self._payloads[metadata.store_ref]

    def delete(self, metadata: KVBlockMetadata) -> None:
        """从本地 RAM 删除 Block 载荷。

        参数:
            metadata: 包含 RAM 后端 store_ref 的 Block 元数据。

        返回:
            无返回值。
        """
        with self._lock:
            if metadata.store_ref is None:
                return
            self._payloads.pop(metadata.store_ref, None)
            self._sizes.pop(metadata.store_ref, None)

    def contains(self, metadata: KVBlockMetadata) -> bool:
        """判断 Block 载荷是否存在于本地 RAM。

        参数:
            metadata: 包含 RAM 后端 store_ref 的 Block 元数据。

        返回:
            如果载荷仍存在于 RAM 后端，返回 True；否则返回 False。
        """
        with self._lock:
            return metadata.store_ref is not None and metadata.store_ref in self._payloads


def _ref(metadata: KVBlockMetadata) -> str:
    """生成 RAM 后端内部引用。

    参数:
        metadata: 需要生成引用的 Block 元数据。

    返回:
        由 block_hash 和 block_id 组成的 RAM 后端引用。
    """
    return f"{metadata.block_hash}:{metadata.block_id}"
