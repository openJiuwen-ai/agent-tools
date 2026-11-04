from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from jiuwen_kvbm.accessor.base import KVAccessor
from jiuwen_kvbm.policy import LRUEvictionPolicy
from jiuwen_kvbm.transfer.base import TransferBackend
from jiuwen_kvbm.types import (
    BlockLocation,
    BlockState,
    KVBlockHandle,
    KVBlockMetadata,
    KVBMStats,
)
from jiuwen_kvbm.util import estimate_payload_size


class PrefetchCommitRejectedError(RuntimeError):
    """预取数据在最终提交前因超时或重复缓存而被拒绝。"""


class KVBMManager:
    """KV Block Manager 核心管理面。

    该类只维护 Block 元数据、生命周期状态和分层存储策略，不直接依赖具体推理
    框架。真实 KV tensor 的读写由 accessor 注入，数据搬运由 transfer backend
    注入，方便后续接入昇腾 NPU/HIXL。

    参数:
        accessor: 推理运行时 KV Block 访问器，负责从 NPU runtime 读写 Block。
        ram_backend: 可选 RAM 存储后端，用于 NPU 到主机内存的卸载。
        disk_backend: 可选磁盘存储后端，用于 NPU/RAM 到本地磁盘的卸载。
        eviction_policy: 可选淘汰策略；未指定时使用 LRU 策略。
    """

    def __init__(
        self,
        accessor: KVAccessor,
        ram_backend: TransferBackend | None = None,
        disk_backend: TransferBackend | None = None,
        eviction_policy: LRUEvictionPolicy | None = None,
    ):
        self.accessor = accessor
        self.ram_backend = ram_backend
        self.disk_backend = disk_backend
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        self._by_id: dict[int, KVBlockMetadata] = {}
        self._hash_to_ids: dict[str, set[int]] = {}
        self._metadata_lock = threading.RLock()

    def register_npu_block(
        self,
        block_id: int,
        block_hash: str,
        *,
        session_id: str | None = None,
        token_count: int = 0,
        active: bool = True,
    ) -> KVBlockHandle:
        """注册一个当前驻留在 NPU 侧的 KV Block。

        参数:
            block_id: 推理运行时中的 Block 编号。
            block_hash: Block 内容或 token 前缀对应的稳定 hash。
            session_id: 可选 session 标识；传入时会记录该 session 对 Block 的引用。
            token_count: 该 Block 覆盖的 token 数量。
            active: 是否将 Block 标记为正在被请求使用。

        返回:
            已注册 Block 的轻量引用。

        异常:
            KeyError: accessor 无法描述该 block_id。
        """
        return self.register_npu_blocks(
            [(block_id, block_hash)],
            session_id=session_id,
            token_count=token_count,
            active=active,
        )[0]

    def register_npu_blocks(
        self,
        blocks: Iterable[tuple[int, str]],
        *,
        session_id: str | None = None,
        token_count: int = 0,
        token_counts: Iterable[int] | None = None,
        active: bool = True,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[KVBlockHandle]:
        """在一次元数据锁内批量注册当前驻留于 NPU 的 KV Block。

        参数:
            blocks: ``(block_id, block_hash)`` 二元组序列。
            session_id: 可选 session 标识；传入时应用到所有 block。
            token_count: 每个 block 覆盖的 token 数量。
            token_counts: 可选的逐 block token 数量；传入时覆盖
                token_count，且数量必须与 blocks 一致。
            active: 是否将所有 block 标记为正在被请求使用。
            diagnostics: 可选诊断字典；传入时写入描述、
                元数据更新耗时以及幂等命中数。

        返回:
            与输入顺序一致的轻量 Block handle 列表；输入为空时返回空列表。

        异常:
            KeyError: accessor 无法描述任意一个 block_id；索引更新前即失败。
        """

        block_specs = [(int(block_id), str(block_hash)) for block_id, block_hash in blocks]
        per_block_token_counts = (
            [int(value) for value in token_counts]
            if token_counts is not None
            else [int(token_count)] * len(block_specs)
        )
        if len(per_block_token_counts) != len(block_specs):
            raise ValueError("token_counts must have the same length as blocks")
        desired_state = BlockState.ACTIVE if active else BlockState.CACHED_NPU
        unchanged: dict[int, KVBlockMetadata] = {}
        metadata_items: list[KVBlockMetadata] = []
        metadata_indices: list[int] = []
        describe_blocks_ms = 0.0
        with self._metadata_lock:
            describe_started_ns = time.perf_counter_ns()
            for index, (block_id, block_hash) in enumerate(block_specs):
                previous = self._by_id.get(block_id)
                if previous is not None:
                    if (
                        previous.block_hash == block_hash
                        and previous.location == BlockLocation.NPU
                        and previous.state == desired_state
                    ):
                        token_count_matches = previous.token_count == per_block_token_counts[index]
                        session_matches = not session_id or session_id in previous.session_ids
                        if token_count_matches and session_matches:
                            unchanged[index] = previous
                            continue
                region = self.accessor.describe_block(block_id)
                metadata = KVBlockMetadata(
                    block_id=block_id,
                    block_hash=block_hash,
                    token_count=per_block_token_counts[index],
                    location=BlockLocation.NPU,
                    state=desired_state,
                    region=region,
                    size_bytes=region.nbytes,
                    ref_count=1 if active else 0,
                )
                if session_id:
                    metadata.session_ids.add(session_id)
                metadata_items.append(metadata)
                metadata_indices.append(index)
            describe_blocks_ms = (time.perf_counter_ns() - describe_started_ns) / 1_000_000.0

            metadata_started_ns = time.perf_counter_ns()
            for metadata in metadata_items:
                self._replace_npu_metadata_locked(metadata)
            metadata_update_ms = (time.perf_counter_ns() - metadata_started_ns) / 1_000_000.0
        metadata_by_index = dict(zip(metadata_indices, metadata_items, strict=True))
        handles = [
            (unchanged[index] if index in unchanged else metadata_by_index[index]).handle
            for index in range(len(block_specs))
        ]
        if diagnostics is not None:
            diagnostics.update(
                {
                    "describe_blocks_ms": describe_blocks_ms,
                    "metadata_update_ms": metadata_update_ms,
                    "registered_blocks": len(metadata_items),
                    "unchanged_blocks": len(unchanged),
                }
            )
        return handles

    def _replace_npu_metadata_locked(self, metadata: KVBlockMetadata) -> None:
        """在已持有 metadata lock 时替换单个 NPU block 元数据。

        参数:
            metadata: 已完成 region 描述的新 NPU block 元数据。

        返回:
            无返回值。
        """

        block_id = metadata.block_id
        block_hash = metadata.block_hash
        previous = self._by_id.get(block_id)
        if previous is not None and previous.location in {
            BlockLocation.HOST_RAM,
            BlockLocation.DISK,
        }:
            # vLLM 已把同一 physical block 发布为新的 NPU 内容，旧 backing
            # payload 不再具备可恢复的所有权，必须在替换元数据前删除。
            self._delete_backend_payload(previous)
        if previous is not None and previous.block_hash != block_hash:
            previous_ids = self._hash_to_ids.get(previous.block_hash)
            if previous_ids is not None:
                previous_ids.discard(block_id)
                if not previous_ids:
                    self._hash_to_ids.pop(previous.block_hash, None)
        self._by_id[block_id] = metadata
        self._hash_to_ids.setdefault(block_hash, set()).add(block_id)

    def get_metadata(self, block_id: int) -> KVBlockMetadata:
        """获取指定 Block 的内部元数据。

        参数:
            block_id: 推理运行时中的 Block 编号。

        返回:
            KVBM 内部维护的 Block 元数据对象。

        异常:
            KeyError: block_id 尚未注册。
        """
        with self._metadata_lock:
            return self._by_id[block_id]

    def match(self, block_hashes: Iterable[str]) -> list[KVBlockMetadata]:
        """按 hash 顺序查找连续命中的 KV Block。

        参数:
            block_hashes: 按 token 前缀顺序排列的 Block hash 序列。

        返回:
            从序列开头开始连续命中的 Block 元数据列表；遇到首个 miss 后停止。
        """
        with self._metadata_lock:
            matches: list[KVBlockMetadata] = []
            for block_hash in block_hashes:
                block = self._match_one(block_hash)
                if block is None:
                    break
                block.touch()
                matches.append(block)
            return matches

    def release(self, session_id: str, block_hashes: Iterable[str]) -> int:
        """释放指定 session 对一组连续 Block 的引用。

        参数:
            session_id: 需要释放引用的 session 标识。
            block_hashes: 按 token 前缀顺序排列的 Block hash 序列。

        返回:
            实际释放 session 引用的 Block 数量。
        """
        with self._metadata_lock:
            released = 0
            for block_hash in block_hashes:
                block = self._match_one(block_hash)
                if block is None:
                    break
                if session_id in block.session_ids:
                    block.session_ids.remove(session_id)
                    released += 1
                if not block.session_ids:
                    block.ref_count = 0
                    if block.location == BlockLocation.NPU and block.state == BlockState.ACTIVE:
                        block.state = BlockState.CACHED_NPU
            return released

    def offload_to_ram(self, block_id: int) -> KVBlockHandle:
        """将 NPU 驻留 Block 卸载到本地 RAM 后端。

        参数:
            block_id: 需要卸载的 Block 编号。

        返回:
            卸载成功后的 Block 轻量引用。

        异常:
            RuntimeError: 未配置 ram_backend。
            KeyError: block_id 尚未注册，或 accessor 无法读取该 Block。
            ValueError: Block 当前不在 NPU 层级。
            MemoryError: RAM 后端容量不足。
        """
        with self._metadata_lock:
            if self.ram_backend is None:
                raise RuntimeError("ram_backend is not configured")
            block = self._by_id[block_id]
            if block.state in {BlockState.PREFETCHING, BlockState.OFFLOADING}:
                raise ValueError(f"block {block_id} has an active migration")
            if block.location != BlockLocation.NPU:
                raise ValueError(f"block {block_id} is not NPU resident")

            block.state = BlockState.OFFLOADING
            try:
                payload = self.accessor.read_block(block_id)
                block.store_ref = self.ram_backend.put(block, payload)
                self._delete_runtime_payload(block_id)
                block.size_bytes = estimate_payload_size(payload)
                block.location = BlockLocation.HOST_RAM
                block.state = BlockState.CACHED_RAM
                block.touch()
                return block.handle
            except Exception as exc:
                block.state = BlockState.FAILED
                block.error = str(exc)
                raise

    def offload_to_disk(self, block_id: int) -> KVBlockHandle:
        """将 NPU/RAM 驻留 Block 卸载到本地磁盘后端。

        参数:
            block_id: 需要卸载的 Block 编号。

        返回:
            卸载成功后的 Block 轻量引用。

        异常:
            RuntimeError: 未配置 disk_backend，或从 RAM 卸载但未配置 ram_backend。
            KeyError: block_id 尚未注册，或后端引用不存在。
            ValueError: Block 当前层级不支持卸载到磁盘。
            OSError: 磁盘后端容量不足或文件写入失败。
        """
        with self._metadata_lock:
            if self.disk_backend is None:
                raise RuntimeError("disk_backend is not configured")
            block = self._by_id[block_id]

            if block.state in {BlockState.PREFETCHING, BlockState.OFFLOADING}:
                raise ValueError(f"block {block_id} has an active migration")
            block.state = BlockState.OFFLOADING
            try:
                if block.location == BlockLocation.HOST_RAM:
                    if self.ram_backend is None:
                        raise RuntimeError("ram_backend is not configured")
                    payload = self.ram_backend.get(block)
                elif block.location == BlockLocation.NPU:
                    payload = self.accessor.read_block(block_id)
                else:
                    raise ValueError(f"block {block_id} cannot be offloaded from {block.location}")

                new_ref = self.disk_backend.put(block, payload)
                if block.location == BlockLocation.HOST_RAM:
                    self.ram_backend.delete(block)
                else:
                    self._delete_runtime_payload(block_id)
                block.store_ref = new_ref
                block.size_bytes = estimate_payload_size(payload)
                block.location = BlockLocation.DISK
                block.state = BlockState.CACHED_DISK
                block.touch()
                return block.handle
            except Exception as exc:
                block.state = BlockState.FAILED
                block.error = str(exc)
                raise

    def prefetch_to_npu(
        self,
        block_id: int,
        *,
        should_commit: Callable[[KVBlockMetadata], bool] | None = None,
    ) -> KVBlockHandle:
        """将 RAM/DISK 中的 Block 预取回 NPU 运行时。

        参数:
            block_id: 需要预取回 NPU 的 Block 编号。
            should_commit: 可选最终提交检查。在载荷写入后、删除 RAM/DISK
                源副本前调用；返回 False 时保留源副本并回滚元数据。

        返回:
            预取成功后的 Block 轻量引用。

        异常:
            RuntimeError: 当前层级需要的后端未配置。
            KeyError: block_id 尚未注册，或后端引用不存在。
            ValueError: Block 当前层级不可预取。
            PrefetchCommitRejectedError: 最终提交检查拒绝本次预取。
        """

        with self._metadata_lock:
            block = self._by_id[block_id]
            source_location = block.location
            source_state = block.state
            source_store_ref = block.store_ref
            if source_location not in {BlockLocation.HOST_RAM, BlockLocation.DISK}:
                raise ValueError(f"block {block_id} cannot be prefetched from {source_location}")
            block.state = BlockState.PREFETCHING

        try:
            payload = self._read_backend_payload(block)
            if should_commit is not None and not should_commit(block):
                raise PrefetchCommitRejectedError(f"prefetch commit rejected before NPU write: block_id={block_id}")
            self.accessor.write_block(block_id, payload)
            with self._metadata_lock:
                current = self._by_id.get(block_id)
                if current is not block:
                    raise PrefetchCommitRejectedError(f"prefetch target block was reused: block_id={block_id}")
                if should_commit is not None and not should_commit(block):
                    raise PrefetchCommitRejectedError(f"prefetch commit rejected after NPU write: block_id={block_id}")
                self._delete_backend_payload(block)
                block.location = BlockLocation.NPU
                block.state = BlockState.CACHED_NPU
                block.store_ref = None
                block.error = None
                block.touch()
                return block.handle
        except PrefetchCommitRejectedError:
            with self._metadata_lock:
                if self._by_id.get(block_id) is block:
                    try:
                        self._delete_runtime_payload(block_id)
                    except Exception:
                        # 回滚的核心语义是保留源副本和禁止发布；运行时清理失败
                        # 由上层诊断处理，不能阻止元数据恢复。
                        pass
                    block.location = source_location
                    block.state = source_state
                    block.store_ref = source_store_ref
                    block.error = None
                    block.touch()
            raise
        except Exception as exc:
            with self._metadata_lock:
                if self._by_id.get(block_id) is block:
                    block.location = source_location
                    block.state = source_state
                    block.store_ref = source_store_ref
                    block.error = str(exc)
            raise

    def has_npu_block(
        self,
        block_hash: str,
        *,
        exclude_block_id: int | None = None,
    ) -> bool:
        """判断指定 hash 是否已经存在另一个可用 NPU Block。

        参数:
            block_hash: 需要检查的逻辑 Block hash。
            exclude_block_id: 可选排除的物理 Block ID，通常是当前预取目标。

        返回:
            存在 ACTIVE 或 CACHED_NPU 副本时返回 True，否则返回 False。
        """

        with self._metadata_lock:
            for block_id in self._hash_to_ids.get(block_hash, set()):
                if exclude_block_id is not None and block_id == exclude_block_id:
                    continue
                block = self._by_id.get(block_id)
                if block is None or block.location != BlockLocation.NPU:
                    continue
                if block.state in {BlockState.ACTIVE, BlockState.CACHED_NPU}:
                    return True
            return False

    def evict(self, limit: int = 1) -> list[KVBlockHandle]:
        """按淘汰策略删除缓存载荷并标记为 EVICTED。

        参数:
            limit: 最多淘汰的 Block 数量。

        返回:
            实际被淘汰的 Block 轻量引用列表。
        """
        evicted: list[KVBlockHandle] = []
        for block in self.eviction_policy.select(self._by_id.values(), limit):
            self._delete_all_payloads(block)
            block.location = BlockLocation.REMOTE
            block.state = BlockState.EVICTED
            block.store_ref = None
            block.touch()
            evicted.append(block.handle)
        return evicted

    def list_blocks(
        self,
        *,
        location: BlockLocation | None = None,
        state: BlockState | None = None,
        include_failed: bool = False,
    ) -> list[KVBlockMetadata]:
        """按条件列出 KVBM 已注册 Block。

        参数:
            location: 可选存储层级过滤条件。
            state: 可选生命周期状态过滤条件。
            include_failed: 是否包含 FAILED 状态的 Block。

        返回:
            满足条件的 Block 元数据列表，按最近访问时间从旧到新排序。
        """

        blocks = list(self._by_id.values())
        if location is not None:
            blocks = [block for block in blocks if block.location == location]
        if state is not None:
            blocks = [block for block in blocks if block.state == state]
        if not include_failed:
            blocks = [block for block in blocks if block.state != BlockState.FAILED]
        return sorted(blocks, key=lambda block: block.last_accessed)

    def evict_block(self, block_id: int) -> KVBlockHandle:
        """淘汰指定 Block。

        参数:
            block_id: 需要淘汰的 Block 编号。

        返回:
            被淘汰 Block 的轻量引用。

        异常:
            KeyError: block_id 尚未注册。
        """

        block = self._by_id[block_id]
        self._delete_all_payloads(block)
        block.location = BlockLocation.REMOTE
        block.state = BlockState.EVICTED
        block.store_ref = None
        block.ref_count = 0
        block.touch()
        return block.handle

    def stats(self) -> KVBMStats:
        """返回当前 KVBM 状态统计。

        返回:
            包含各状态 Block 数量和 RAM/Disk 占用的统计快照。
        """
        blocks = list(self._by_id.values())
        return KVBMStats(
            total_blocks=len(blocks),
            active_blocks=sum(block.state == BlockState.ACTIVE for block in blocks),
            cached_npu_blocks=sum(block.state == BlockState.CACHED_NPU for block in blocks),
            ram_blocks=sum(block.state == BlockState.CACHED_RAM for block in blocks),
            disk_blocks=sum(block.state == BlockState.CACHED_DISK for block in blocks),
            evicted_blocks=sum(block.state == BlockState.EVICTED for block in blocks),
            failed_blocks=sum(block.state == BlockState.FAILED for block in blocks),
            ram_bytes=self.ram_backend.bytes_used if self.ram_backend is not None else 0,
            disk_bytes=self.disk_backend.bytes_used if self.disk_backend is not None else 0,
        )

    def _match_one(self, block_hash: str) -> KVBlockMetadata | None:
        """查找单个 hash 对应的最旧 Block 元数据。

        参数:
            block_hash: 需要查找的 Block hash。

        返回:
            命中时返回最近访问时间最旧的 Block 元数据；未命中时返回 None。
        """
        with self._metadata_lock:
            ids = self._hash_to_ids.get(block_hash, set())
            if not ids:
                return None
            candidates = [
                self._by_id[block_id]
                for block_id in ids
                if self._by_id[block_id].state not in {BlockState.EVICTED, BlockState.FAILED}
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda block: block.last_accessed)

    def _load_payload(self, block: KVBlockMetadata) -> Any:
        """从当前所在层级加载载荷，并清理原后端引用。

        参数:
            block: 需要加载载荷的 Block 元数据。

        返回:
            从 RAM、Disk 或 NPU 读取出的 Block 载荷。

        异常:
            RuntimeError: 当前层级需要的后端未配置。
            KeyError: 后端引用不存在。
            ValueError: Block 当前层级不可加载。
        """
        if block.location == BlockLocation.HOST_RAM:
            if self.ram_backend is None:
                raise RuntimeError("ram_backend is not configured")
            payload = self.ram_backend.get(block)
            self.ram_backend.delete(block)
            return payload
        if block.location == BlockLocation.DISK:
            if self.disk_backend is None:
                raise RuntimeError("disk_backend is not configured")
            payload = self.disk_backend.get(block)
            self.disk_backend.delete(block)
            return payload
        if block.location == BlockLocation.NPU:
            return self.accessor.read_block(block.block_id)
        raise ValueError(f"block {block.block_id} is not prefetchable from {block.location}")

    def _read_backend_payload(self, block: KVBlockMetadata) -> Any:
        """读取 RAM/DISK payload，但在预取提交前保留源副本。

        参数:
            block: 当前位于 RAM 或 Disk 的 Block 元数据。

        返回:
            对应后端中的 payload。

        异常:
            RuntimeError: 当前层级所需后端未配置。
            KeyError: 后端引用不存在。
            ValueError: Block 当前不位于 RAM/DISK。
        """

        if block.location == BlockLocation.HOST_RAM:
            if self.ram_backend is None:
                raise RuntimeError("ram_backend is not configured")
            return self.ram_backend.get(block)
        if block.location == BlockLocation.DISK:
            if self.disk_backend is None:
                raise RuntimeError("disk_backend is not configured")
            return self.disk_backend.get(block)
        raise ValueError(f"block {block.block_id} is not prefetchable from {block.location}")

    def _delete_backend_payload(self, block: KVBlockMetadata) -> None:
        """删除 Block 在当前后端中的载荷。

        参数:
            block: 需要删除载荷的 Block 元数据。

        返回:
            无返回值。
        """

        if block.location == BlockLocation.HOST_RAM and self.ram_backend is not None:
            self.ram_backend.delete(block)
        elif block.location == BlockLocation.DISK and self.disk_backend is not None:
            self.disk_backend.delete(block)

    def _delete_runtime_payload(self, block_id: int) -> None:
        """从推理运行时释放 Block 载荷。

        参数:
            block_id: 需要释放的 Block 编号。

        返回:
            无返回值。
        """

        delete_block = getattr(self.accessor, "delete_block", None)
        if callable(delete_block):
            delete_block(block_id)

    def _delete_all_payloads(self, block: KVBlockMetadata) -> None:
        """删除 Block 在运行时或后端中的载荷。

        参数:
            block: 需要删除载荷的 Block 元数据。

        返回:
            无返回值。
        """

        if block.location == BlockLocation.NPU:
            self._delete_runtime_payload(block.block_id)
        else:
            self._delete_backend_payload(block)
