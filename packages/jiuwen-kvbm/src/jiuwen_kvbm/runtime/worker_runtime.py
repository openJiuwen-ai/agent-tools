from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jiuwen_kvbm.accessor.base import KVAccessor
from jiuwen_kvbm.manager import KVBMManager
from jiuwen_kvbm.transfer import LLMDataDistBackend, LocalDiskBackend, LocalRamBackend
from jiuwen_kvbm.types import BlockLocation, BlockState, KVBlockMetadata

logger = logging.getLogger("jiuwen.kvbm.runtime")


@dataclass(frozen=True)
class WorkerKVBMConfig:
    """worker 侧 KVBM runtime 配置。

    属性:
        worker_id: worker 唯一标识，用于生成上报事件。
        disk_root: 本地磁盘后端目录。
        npu_high_watermark_blocks: NPU 缓存 block 数超过该值时触发 NPU->RAM。
        ram_high_watermark_bytes: RAM 占用超过该值时触发 RAM->DISK。
        ram_max_bytes: RAM 后端最大容量；为 None 时不限制。
        disk_max_bytes: 磁盘后端最大容量；为 None 时不限制。
        block_size: 默认 block token 数。
        clean_disk_on_start: 是否在 runtime 启动时清理旧磁盘 payload。
        data_backend: worker 数据面 backend 名称，例如 local、llm_datadist。
    """

    worker_id: str
    disk_root: Path
    npu_high_watermark_blocks: int = 0
    ram_high_watermark_bytes: int = 0
    ram_max_bytes: int | None = None
    disk_max_bytes: int | None = None
    block_size: int = 16
    clean_disk_on_start: bool = True
    data_backend: str = "local"


@dataclass(frozen=True)
class KVBMRuntimeEvent:
    """worker KVBM runtime 产生的轻量事件。

    属性:
        event_type: 事件类型，例如 move、store、evict。
        block_hashes: 事件涉及的 Block hash 列表。
        location: 事件完成后的目标层级。
        token_count: 事件涉及的 token 数量。
        engine_specific: 补充字段，供 router 转换为 CacheEvent。
    """

    event_type: str
    block_hashes: list[str]
    location: str
    token_count: int
    engine_specific: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典。

        返回:
            可被 worker 上报链路进一步转换的事件字典。
        """

        return {
            "event_type": self.event_type,
            "block_hashes": self.block_hashes,
            "location": self.location,
            "token_count": self.token_count,
            "engine_specific": dict(self.engine_specific),
        }


class WorkerKVBMRuntime:
    """worker 侧真实数据面 runtime。

    参数:
        config: worker 侧 KVBM 配置。
        accessor: worker 运行时 KV Block 访问器。
        manager: 可选外部 KVBMManager，主要用于定制后端或单测。
        llm_datadist_backend: 可选 LLM-DataDist backend，后续用于远端 KV
            cache/block 语义传输。

    说明:
        当前实现仍为 Python 数据面，能完成真实 payload 在 fake NPU/RAM/Disk
        之间的搬运。接入昇腾时，需要替换 accessor 为真实 NPU KVAccessor。
    """

    def __init__(
        self,
        config: WorkerKVBMConfig,
        accessor: KVAccessor,
        manager: KVBMManager | None = None,
        llm_datadist_backend: LLMDataDistBackend | None = None,
    ):
        self.config = config
        self.accessor = accessor
        self.llm_datadist_backend = llm_datadist_backend
        if manager is not None:
            self.manager = manager
        else:
            disk_backend = LocalDiskBackend(config.disk_root, max_bytes=config.disk_max_bytes)
            if config.clean_disk_on_start:
                deleted = disk_backend.clear_managed_payloads()
                if deleted:
                    logger.info(
                        "Worker KVBM cleaned %d old disk payloads from %s",
                        deleted,
                        config.disk_root,
                    )
            self.manager = KVBMManager(
                accessor=accessor,
                ram_backend=LocalRamBackend(max_bytes=config.ram_max_bytes),
                disk_backend=disk_backend,
            )

    def register_block(
        self,
        block_id: int,
        block_hash: str,
        *,
        token_count: int | None = None,
        session_id: str | None = None,
        active: bool = False,
    ) -> KVBMRuntimeEvent:
        """注册 worker runtime 中的新 KV Block。

        参数:
            block_id: worker runtime 中的 Block 编号。
            block_hash: Block 内容或 token 前缀对应的稳定 hash。
            token_count: 该 Block 覆盖的 token 数；为 None 时使用配置中的 block_size。
            session_id: 可选 session 标识。
            active: 是否标记为正在使用。

        返回:
            可上报给 router 的 store 事件。
        """

        effective_token_count = token_count or self.config.block_size
        self.manager.register_npu_block(
            block_id,
            block_hash,
            session_id=session_id,
            token_count=effective_token_count,
            active=active,
        )
        return self._event(
            "store",
            block_hash,
            "npu",
            effective_token_count,
            physical_block_id=block_id,
        )

    def register_blocks(
        self,
        blocks: list[tuple[int, str]] | tuple[tuple[int, str], ...],
        *,
        token_count: int | None = None,
        token_counts: list[int] | tuple[int, ...] | None = None,
        session_id: str | None = None,
        active: bool = False,
        diagnostics: dict[str, Any] | None = None,
    ) -> KVBMRuntimeEvent:
        """批量注册 NPU KV Block，并生成一条聚合 store 事件。

        参数:
            blocks: 按前缀顺序排列的 ``(block_id, block_hash)`` 序列。
            token_count: 每个 block 覆盖的 token 数；为 None 时使用配置 block_size。
            token_counts: 可选的逐 block token 数，用于保留最后一个
                partial block 的有效 token 数。
            session_id: 可选 session 标识，应用到全部 block。
            active: 是否将全部 block 标记为正在使用。
            diagnostics: 可选诊断字典，写入批量注册阶段耗时。

        返回:
            同时携带全部 block hash 和物理 ID 的聚合 store 事件。

        异常:
            ValueError: blocks 为空时抛出。
        """

        block_specs = [(int(block_id), str(block_hash)) for block_id, block_hash in blocks]
        if not block_specs:
            raise ValueError("blocks must not be empty")
        effective_token_count = token_count or self.config.block_size
        self.manager.register_npu_blocks(
            block_specs,
            session_id=session_id,
            token_count=effective_token_count,
            token_counts=token_counts,
            active=active,
            diagnostics=diagnostics,
        )
        block_ids = [block_id for block_id, _block_hash in block_specs]
        block_hashes = [block_hash for _block_id, block_hash in block_specs]
        return KVBMRuntimeEvent(
            event_type="store",
            block_hashes=block_hashes,
            location="npu",
            token_count=effective_token_count,
            engine_specific={
                "worker_id": self.config.worker_id,
                "block_size": self.config.block_size,
                "to": "npu",
                "medium": "npu",
                "physical_block_ids": block_ids,
            },
        )

    def enforce_watermarks(self) -> list[KVBMRuntimeEvent]:
        """执行 NPU/RAM 水位策略。

        返回:
            策略执行期间产生的 move 事件列表。
        """

        events: list[KVBMRuntimeEvent] = []
        events.extend(self.offload_cold_npu_blocks_to_ram())
        events.extend(self.offload_cold_ram_blocks_to_disk())
        return events

    def offload_cold_npu_blocks_to_ram(self, max_blocks: int | None = None) -> list[KVBMRuntimeEvent]:
        """将冷 NPU Block 卸载到 RAM。

        参数:
            max_blocks: 最多卸载的 Block 数；为 None 时根据 npu_high_watermark_blocks 计算。

        返回:
            每个成功卸载 Block 对应的 move 事件列表。
        """

        npu_blocks = self.manager.list_blocks(
            location=BlockLocation.NPU,
            state=BlockState.CACHED_NPU,
        )
        if max_blocks is None:
            overflow = len(npu_blocks) - self.config.npu_high_watermark_blocks
            max_blocks = max(0, overflow)
        return [self._offload_to_ram(block) for block in npu_blocks[:max_blocks]]

    def offload_cold_ram_blocks_to_disk(self, max_blocks: int | None = None) -> list[KVBMRuntimeEvent]:
        """将冷 RAM Block 卸载到磁盘。

        参数:
            max_blocks: 最多卸载的 Block 数；为 None 时根据 ram_high_watermark_bytes 计算。

        返回:
            每个成功卸载 Block 对应的 move 事件列表。
        """

        ram_blocks = self.manager.list_blocks(
            location=BlockLocation.HOST_RAM,
            state=BlockState.CACHED_RAM,
        )
        if max_blocks is None:
            max_blocks = 0
            if self.config.ram_high_watermark_bytes > 0:
                ram_bytes = self.manager.stats().ram_bytes
                for block in ram_blocks:
                    if ram_bytes <= self.config.ram_high_watermark_bytes:
                        break
                    max_blocks += 1
                    ram_bytes -= block.size_bytes
        return [self._offload_to_disk(block) for block in ram_blocks[:max_blocks]]

    def prefetch_block(
        self,
        block_hash: str,
        *,
        should_commit: Callable[[KVBlockMetadata], bool] | None = None,
    ) -> KVBMRuntimeEvent | None:
        """将指定 hash 的 Block 预取回 NPU。

        参数:
            block_hash: 需要预取的 Block hash。
            should_commit: 可选最终提交检查；返回 False 时保留 RAM/DISK
                源副本并拒绝将预取结果发布为 NPU 缓存。

        返回:
            命中并成功预取时返回 move 事件；未命中时返回 None。
        """

        matches = self.manager.match([block_hash])
        if not matches:
            return None
        block = matches[0]
        if block.location == BlockLocation.NPU:
            return self._event(
                "hit",
                block.block_hash,
                "npu",
                block.token_count,
                physical_block_id=block.block_id,
            )
        source = block.location.value
        self.manager.prefetch_to_npu(
            block.block_id,
            should_commit=should_commit,
        )
        return self._event(
            "move",
            block.block_hash,
            "npu",
            block.token_count,
            source=source,
            physical_block_id=block.block_id,
        )

    def prefetch_blocks(self, block_hashes: list[str] | tuple[str, ...]) -> list[KVBMRuntimeEvent]:
        """批量将指定 hash 的 Block 预取回 NPU。

        参数:
            block_hashes: 需要预取的 Block hash 列表。

        返回:
            命中并成功预取的事件列表；未命中的 hash 不产生事件。
        """

        events = []
        for block_hash in block_hashes:
            event = self.prefetch_block(block_hash)
            if event is not None:
                events.append(event)
        return events

    def stats(self) -> dict[str, Any]:
        """返回 runtime 统计信息。

        返回:
            包含 KVBMManager 统计和 worker_id 的字典。
        """

        stats = self.manager.stats()
        return {
            "worker_id": self.config.worker_id,
            "data_backend": self.config.data_backend,
            "block_size": self.config.block_size,
            "total_blocks": stats.total_blocks,
            "active_blocks": stats.active_blocks,
            "cached_npu_blocks": stats.cached_npu_blocks,
            "ram_blocks": stats.ram_blocks,
            "disk_blocks": stats.disk_blocks,
            "ram_bytes": stats.ram_bytes,
            "disk_bytes": stats.disk_bytes,
            "failed_blocks": stats.failed_blocks,
            "llm_datadist": (self.llm_datadist_backend.health() if self.llm_datadist_backend is not None else None),
        }

    def offload_to_ram(self, block: KVBlockMetadata) -> KVBMRuntimeEvent:
        """Move a registered block to RAM through the runtime lifecycle."""
        return self._offload_to_ram(block)

    def offload_to_disk(self, block: KVBlockMetadata) -> KVBMRuntimeEvent:
        """Move a registered block to disk through the runtime lifecycle."""
        return self._offload_to_disk(block)

    def _offload_to_ram(self, block: KVBlockMetadata) -> KVBMRuntimeEvent:
        """卸载单个 Block 到 RAM。

        参数:
            block: 需要卸载的 Block 元数据。

        返回:
            对应的 move 事件。
        """

        self.manager.offload_to_ram(block.block_id)
        return self._event(
            "move",
            block.block_hash,
            "ram",
            block.token_count,
            source="npu",
            physical_block_id=block.block_id,
        )

    def _offload_to_disk(self, block: KVBlockMetadata) -> KVBMRuntimeEvent:
        """卸载单个 Block 到磁盘。

        参数:
            block: 需要卸载的 Block 元数据。

        返回:
            对应的 move 事件。
        """

        self.manager.offload_to_disk(block.block_id)
        return self._event(
            "move",
            block.block_hash,
            "disk",
            block.token_count,
            source="ram",
            physical_block_id=block.block_id,
        )

    def _event(
        self,
        event_type: str,
        block_hash: str,
        location: str,
        token_count: int,
        *,
        source: str | None = None,
        physical_block_id: int | None = None,
    ) -> KVBMRuntimeEvent:
        """构造 runtime 事件。

        参数:
            event_type: 事件类型。
            block_hash: 事件涉及的 Block hash。
            location: 事件完成后的目标层级。
            token_count: 事件涉及的 token 数量。
            source: 可选源层级。
            physical_block_id: Worker 内部物理 block 编号；未知时为 None。

        返回:
            KVBMRuntimeEvent 实例。
        """

        engine_specific = {
            "worker_id": self.config.worker_id,
            "block_size": self.config.block_size,
            "to": location,
            "medium": location,
        }
        if source is not None:
            engine_specific["from"] = source
        if physical_block_id is not None:
            engine_specific["physical_block_id"] = int(physical_block_id)
        return KVBMRuntimeEvent(
            event_type=event_type,
            block_hashes=[block_hash],
            location=location,
            token_count=token_count,
            engine_specific=engine_specific,
        )
