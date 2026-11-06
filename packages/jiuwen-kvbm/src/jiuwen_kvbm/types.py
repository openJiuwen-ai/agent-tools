from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockLocation(str, Enum):
    """KV Block 当前所在的存储层级。

    枚举值:
        NPU: Block 当前驻留在昇腾 NPU 运行时管理的 KV cache 中。
        HOST_RAM: Block 当前驻留在主机侧 RAM 中。
        DISK: Block 当前驻留在本地磁盘中。
        REMOTE: Block 当前不在本地管理的快速层级中，可表示远端或已淘汰。
    """

    NPU = "npu"
    HOST_RAM = "host_ram"
    DISK = "disk"
    REMOTE = "remote"


class BlockState(str, Enum):
    """KV Block 生命周期状态。

    枚举值:
        ACTIVE: Block 正在被请求使用。
        CACHED_NPU: Block 在 NPU KV cache 中，可被复用。
        OFFLOADING: Block 正在从高层级卸载到低层级。
        CACHED_RAM: Block 已缓存在主机 RAM 中。
        CACHED_DISK: Block 已缓存在本地磁盘中。
        PREFETCHING: Block 正在被预取回高层级。
        EVICTED: Block 的本地载荷已被淘汰。
        FAILED: Block 最近一次状态迁移失败。
    """

    ACTIVE = "active"
    CACHED_NPU = "cached_npu"
    OFFLOADING = "offloading"
    CACHED_RAM = "cached_ram"
    CACHED_DISK = "cached_disk"
    PREFETCHING = "prefetching"
    EVICTED = "evicted"
    FAILED = "failed"


@dataclass(frozen=True)
class KVBlockHandle:
    """对外暴露的轻量 Block 引用。

    属性:
        block_id: 推理运行时中的 Block 编号。
        block_hash: Block 内容或 token 前缀对应的稳定 hash。
    """

    block_id: int
    block_hash: str


@dataclass
class MemoryRegion:
    """中性的内存区域描述，用于隔离 NPU/HIXL 等底层实现细节。

    属性:
        device_type: 内存所属设备类型，例如 npu、host、disk、remote。
        nbytes: 内存区域大小，单位为字节。
        device_id: 设备编号；主机内存或磁盘场景可为空。
        address: 底层内存地址；Python 对象或文件场景可为空。
        layout: KV 数据布局描述，例如按层、按 head 或连续 block 的布局名。
        backend_handle: 运行时或传输后端持有的原生句柄。
    """

    device_type: str
    nbytes: int
    device_id: int | None = None
    address: int | None = None
    layout: str | None = None
    backend_handle: Any = None


@dataclass
class KVBlockMetadata:
    """KVBM 内部维护的 Block 元数据。

    属性:
        block_id: 推理运行时中的 Block 编号。
        block_hash: Block 内容或 token 前缀对应的稳定 hash。
        token_count: 该 Block 覆盖的 token 数量。
        location: Block 当前所在的存储层级。
        state: Block 当前生命周期状态。
        session_ids: 当前引用该 Block 的 session 集合。
        region: Block 在当前运行时中的中性内存描述。
        size_bytes: Block 载荷大小，单位为字节。
        created_at: 元数据创建时间戳。
        last_accessed: 最近访问时间戳，用于 LRU 等策略。
        ref_count: 当前引用计数。
        store_ref: RAM/Disk 等后端返回的本地存储引用。
        error: 最近一次失败迁移的错误信息。
    """

    block_id: int
    block_hash: str
    token_count: int = 0
    location: BlockLocation = BlockLocation.NPU
    state: BlockState = BlockState.ACTIVE
    session_ids: set[str] = field(default_factory=set)
    region: MemoryRegion | None = None
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ref_count: int = 0
    store_ref: str | None = None
    error: str | None = None

    @property
    def handle(self) -> KVBlockHandle:
        """生成不暴露内部状态的 Block 引用。

        返回:
            仅包含 block_id 和 block_hash 的轻量引用。
        """
        return KVBlockHandle(block_id=self.block_id, block_hash=self.block_hash)

    def touch(self) -> None:
        """刷新最近访问时间，用于 LRU 等淘汰策略。

        返回:
            无返回值。
        """
        self.last_accessed = time.time()


@dataclass(frozen=True)
class KVBMStats:
    """KVBM 当前状态的统计快照。

    属性:
        total_blocks: 已注册 Block 总数。
        active_blocks: 处于 ACTIVE 状态的 Block 数。
        cached_npu_blocks: 处于 CACHED_NPU 状态的 Block 数。
        ram_blocks: 处于 CACHED_RAM 状态的 Block 数。
        disk_blocks: 处于 CACHED_DISK 状态的 Block 数。
        evicted_blocks: 处于 EVICTED 状态的 Block 数。
        failed_blocks: 处于 FAILED 状态的 Block 数。
        ram_bytes: RAM 后端当前占用字节数。
        disk_bytes: Disk 后端当前占用字节数。
    """

    total_blocks: int
    active_blocks: int
    cached_npu_blocks: int
    ram_blocks: int
    disk_blocks: int
    evicted_blocks: int
    failed_blocks: int
    ram_bytes: int
    disk_bytes: int
