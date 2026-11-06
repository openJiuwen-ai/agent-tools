from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from openjiuwentools.infer_router.kv_cache.event import CacheEvent, CacheLocation

try:
    from jiuwen_kvbm import BlockLocation, BlockState, KVBMManager, MemoryRegion
    from jiuwen_kvbm.accessor import FakeKVAccessor
    from jiuwen_kvbm.transfer import LocalDiskBackend, LocalRamBackend
except ImportError:  # pragma: no cover - 由未安装 jiuwen-kvbm 的部署环境触发
    BlockLocation = None
    BlockState = None
    KVBMManager = None
    MemoryRegion = None
    FakeKVAccessor = None
    LocalDiskBackend = None
    LocalRamBackend = None


@dataclass(frozen=True)
class RouterKVBMConfig:
    """Router 侧 KVBM 适配器配置。

    属性:
        disk_root: 本地磁盘后端根目录。
        ram_max_bytes: RAM 后端容量上限；为 None 时不限制。
        disk_max_bytes: 磁盘后端容量上限；为 None 时不限制。
    """

    disk_root: Path
    ram_max_bytes: int | None = None
    disk_max_bytes: int | None = None


class RouterKVBMAdapter:
    """将 router KV 事件同步到独立 jiuwen-kvbm 管理面。

    参数:
        config: Router KVBM 适配器配置。
        manager: 可选外部 KVBMManager，主要用于单测或自定义运行时接入。

    说明:
        当前适配器只做管理面同步。由于 router 侧没有真实 NPU KV tensor 载荷，
        默认使用 FakeKVAccessor 保存轻量占位 payload，后续 worker/runtime adapter
        可以传入真实 accessor 和 manager。
    """

    def __init__(self, config: RouterKVBMConfig, manager: Any | None = None):
        if KVBMManager is None:
            raise RuntimeError("jiuwen-kvbm is not installed")
        self.config = config
        self._hash_to_block_id: dict[tuple[str, str], int] = {}
        self._block_id_to_key: dict[int, tuple[str, str]] = {}
        self._next_block_id = 1

        if manager is None:
            self.accessor = FakeKVAccessor({})
            self.manager = KVBMManager(
                accessor=self.accessor,
                ram_backend=LocalRamBackend(max_bytes=config.ram_max_bytes),
                disk_backend=LocalDiskBackend(config.disk_root, max_bytes=config.disk_max_bytes),
            )
        else:
            self.accessor = None
            self.manager = manager

    @classmethod
    def from_settings(cls, settings: Any) -> "RouterKVBMAdapter":
        """从 router settings 创建适配器。

        参数:
            settings: router 全局配置对象。

        返回:
            已初始化的 RouterKVBMAdapter。
        """

        disk_root = Path(getattr(settings, "kvbm_disk_root", "/tmp/jiuwen-kvbm-router"))
        return cls(
            RouterKVBMConfig(
                disk_root=disk_root,
                ram_max_bytes=getattr(settings, "kvbm_ram_max_bytes", None),
                disk_max_bytes=getattr(settings, "kvbm_disk_max_bytes", None),
            )
        )

    def register_handlers(self, event_manager: Any) -> None:
        """向 KVEventManager 注册 KVBM 事件处理器。

        参数:
            event_manager: router 侧 KVEventManager。

        返回:
            无返回值。
        """

        event_manager.register_handler("store", self.handle_event)
        event_manager.register_handler("removed", self.handle_event)
        event_manager.register_handler("remove", self.handle_event)
        event_manager.register_handler("evict", self.handle_event)
        event_manager.register_handler("cleared", self.handle_event)
        event_manager.register_handler("clear", self.handle_event)
        event_manager.register_handler("hit", self.handle_event)
        event_manager.register_handler("move", self.handle_event)
        event_manager.register_handler("transfer", self.handle_event)

    def handle_event(self, event: CacheEvent) -> None:
        """处理单个 router KV 事件。

        参数:
            event: router 内部标准化后的 CacheEvent。

        返回:
            无返回值。
        """

        event_type = event.event_type
        if event_type == "store":
            self._handle_store(event)
        elif event_type in {"removed", "remove", "evict"}:
            self._handle_removed(event)
        elif event_type in {"cleared", "clear"}:
            self._handle_cleared(event)
        elif event_type == "hit":
            self._handle_hit(event)
        elif event_type in {"move", "transfer"}:
            self._handle_move(event)

    def get_stats(self) -> dict[str, Any]:
        """返回 KVBM 统计信息。

        返回:
            可序列化的 KVBM 状态统计字典。
        """

        stats = self.manager.stats()
        return {
            "total_blocks": stats.total_blocks,
            "active_blocks": stats.active_blocks,
            "cached_npu_blocks": stats.cached_npu_blocks,
            "ram_blocks": stats.ram_blocks,
            "disk_blocks": stats.disk_blocks,
            "evicted_blocks": stats.evicted_blocks,
            "failed_blocks": stats.failed_blocks,
            "ram_bytes": stats.ram_bytes,
            "disk_bytes": stats.disk_bytes,
        }

    def _handle_store(self, event: CacheEvent) -> None:
        worker_id = _worker_id(event)
        if not worker_id:
            return
        block_size = int(event.engine_specific.get("block_size", 16))
        for index, block_hash in enumerate(event.block_hashes):
            block_id = self._ensure_block_id(worker_id, block_hash)
            token_count = _token_count_for_block(event, index, block_size)
            if self.accessor is not None:
                self.accessor.write_block(block_id, _payload_for_event(event, block_hash, token_count))
            self.manager.register_npu_block(
                block_id,
                block_hash,
                session_id=_session_id(event),
                token_count=token_count,
                active=False,
            )

    def _handle_removed(self, event: CacheEvent) -> None:
        for block_hash in event.block_hashes:
            block_id = self._lookup_block_id(event, block_hash)
            if block_id is None:
                continue
            metadata = self.manager.get_metadata(block_id)
            if metadata.state != BlockState.EVICTED:
                self.manager.evict_block(block_id)
            key = self._block_id_to_key.pop(block_id, None)
            if key is not None:
                self._hash_to_block_id.pop(key, None)

    def _handle_cleared(self, event: CacheEvent) -> None:
        worker_id = _worker_id(event)
        if not worker_id:
            return
        for (cached_worker, block_hash), block_id in list(self._hash_to_block_id.items()):
            if cached_worker != worker_id:
                continue
            metadata = self.manager.get_metadata(block_id)
            if metadata.state != BlockState.EVICTED:
                self.manager.evict_block(block_id)
            self._hash_to_block_id.pop((cached_worker, block_hash), None)
            self._block_id_to_key.pop(block_id, None)

    def _handle_hit(self, event: CacheEvent) -> None:
        for block_hash in event.block_hashes:
            block_id = self._lookup_block_id(event, block_hash)
            if block_id is not None:
                self.manager.get_metadata(block_id).touch()

    def _handle_move(self, event: CacheEvent) -> None:
        for block_hash in event.block_hashes:
            block_id = self._lookup_block_id(event, block_hash)
            if block_id is None:
                continue
            target = _target_location(event)
            try:
                if target == BlockLocation.HOST_RAM:
                    self.manager.offload_to_ram(block_id)
                elif target == BlockLocation.DISK:
                    self.manager.offload_to_disk(block_id)
                elif target == BlockLocation.NPU:
                    self.manager.prefetch_to_npu(block_id)
            except Exception as exc:
                logger.warning("KVBM move failed for block %s: %s", block_hash, exc)

    def _ensure_block_id(self, worker_id: str, block_hash: str) -> int:
        key = (worker_id, block_hash)
        block_id = self._hash_to_block_id.get(key)
        if block_id is not None:
            return block_id
        block_id = self._next_block_id
        self._next_block_id += 1
        self._hash_to_block_id[key] = block_id
        self._block_id_to_key[block_id] = key
        return block_id

    def _lookup_block_id(self, event: CacheEvent, block_hash: str) -> int | None:
        worker_id = _worker_id(event)
        if not worker_id:
            return None
        return self._hash_to_block_id.get((worker_id, block_hash))


def _worker_id(event: CacheEvent) -> str:
    """从 CacheEvent 中提取 worker_id。

    参数:
        event: router 内部缓存事件。

    返回:
        worker_id 字符串；不存在时返回空字符串。
    """

    return event.worker_id or event.engine_specific.get("worker_id", "")


def _session_id(event: CacheEvent) -> str | None:
    """从 CacheEvent 中提取 session_id。

    参数:
        event: router 内部缓存事件。

    返回:
        session_id 字符串；不存在时返回 None。
    """

    session_id = event.engine_specific.get("session_id") or event.engine_specific.get("cache_salt")
    return str(session_id) if session_id else None


def _token_count_for_block(event: CacheEvent, index: int, block_size: int) -> int:
    """计算事件中单个 Block 覆盖的 token 数量。

    参数:
        event: router 内部缓存事件。
        index: 当前 Block 在 event.block_hashes 中的下标。
        block_size: 默认 Block 大小。

    返回:
        当前 Block 覆盖的 token 数量。
    """

    if index == len(event.block_hashes) - 1 and event.token_ids:
        remainder = len(event.token_ids) % block_size
        return remainder or block_size
    return event.token_count or block_size


def _payload_for_event(event: CacheEvent, block_hash: str, token_count: int) -> bytes:
    """为 router 管理面生成轻量占位载荷。

    参数:
        event: router 内部缓存事件。
        block_hash: 当前 Block hash。
        token_count: 当前 Block 覆盖的 token 数量。

    返回:
        可被 FakeKVAccessor 存储的 bytes 载荷。
    """

    worker_id = _worker_id(event)
    return f"{worker_id}:{block_hash}:{token_count}".encode()


def _target_location(event: CacheEvent) -> Any:
    """将 router CacheLocation 转换为 jiuwen-kvbm BlockLocation。

    参数:
        event: router 内部缓存事件。

    返回:
        对应的 BlockLocation。
    """

    if event.location == CacheLocation.CPU:
        return BlockLocation.HOST_RAM
    if event.location == CacheLocation.REMOTE:
        medium = str(event.engine_specific.get("to") or event.engine_specific.get("medium") or "").lower()
        return BlockLocation.DISK if medium in {"disk", "ssd", "nvme", "file", "filesystem"} else BlockLocation.REMOTE
    return BlockLocation.NPU
