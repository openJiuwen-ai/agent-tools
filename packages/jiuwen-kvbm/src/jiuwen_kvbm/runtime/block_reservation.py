from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


def _iter_cached_blocks(value: Any) -> list[Any]:
    """把 vLLM hash 索引中的单 block 或容器统一转换为列表。"""

    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, list | tuple | set):
        return list(value)
    return [value]


class BlockReservationError(RuntimeError):
    """vLLM 物理 block 预留错误。"""


class _IndexRollbackError(BlockReservationError):
    def __init__(self, unsafe_ids: set[int]):
        self.unsafe_ids = unsafe_ids
        super().__init__(f"cache index rollback failed for blocks {sorted(unsafe_ids)}")


class BlockHashesInFlightError(BlockReservationError):
    """相同逻辑 block hash 正由其他 reservation 传输。

    参数:
        hash_count: 与当前 delta plan 冲突的完整 block hash 数量。
        reservation_ids: 正持有这些 hash 的 reservation ID。
    """

    def __init__(self, hash_count: int, reservation_ids: list[str]) -> None:
        """保存冲突摘要并构造稳定错误信息。

        参数:
            hash_count: 冲突的完整 block hash 数量。
            reservation_ids: 去重后的 leader reservation ID。

        返回:
            无返回值。
        """

        self.hash_count = int(hash_count)
        self.reservation_ids = tuple(dict.fromkeys(str(value) for value in reservation_ids))
        super().__init__(
            "delta import block hashes are already in flight: "
            f"hashes={self.hash_count}, reservations={list(self.reservation_ids)}"
        )


class BlockReservationCapacityError(BlockReservationError):
    """delta import 当前没有足够的空闲物理 block。

    参数:
        requested: 本次仍需预留的 block 数量。
        available: BlockPool 当前空闲 block 数量。
        reused: 本次已复用的本地 block 数量。
        owner_id: 已登记等待所有权的 delta plan ID。
    """

    def __init__(
        self,
        requested: int,
        available: int,
        reused: int,
        *,
        owner_id: str = "",
    ) -> None:
        """保存容量摘要并构造兼容现有日志的错误文本。

        参数:
            requested: 需要预留的 block 数量。
            available: 当前可用 block 数量。
            reused: 已复用 block 数量。
            owner_id: 容量等待 leader ID；空字符串表示未登记。

        返回:
            无返回值。
        """

        self.requested = int(requested)
        self.available = int(available)
        self.reused = int(reused)
        self.owner_id = str(owner_id)
        super().__init__(
            "insufficient free vLLM blocks for delta import: "
            f"requested={self.requested}, available={self.available}, "
            f"reused={self.reused}"
        )


class BlockAlreadyPublishedError(BlockReservationError):
    """目标 BlockPool 已经发布相同完整 hash。

    参数:
        existing_block_ids: 当前已绑定该 hash 的物理 block ID。
    """

    def __init__(self, existing_block_ids: list[int]) -> None:
        """保存已存在的物理 block，并构造稳定错误信息。

        参数:
            existing_block_ids: 已发布相同 hash 的物理 block ID。

        返回:
            无返回值。
        """

        self.existing_block_ids = tuple(int(value) for value in existing_block_ids)
        super().__init__(
            "block hash was already published before reservation commit: "
            f"existing_block_ids={list(self.existing_block_ids)}"
        )


class BlockReservationState(str, Enum):
    """物理 block reservation 生命周期状态。"""

    RESERVED = "reserved"
    COMMITTED = "committed"
    PUBLISHED = "published"
    ATTACHED = "attached"
    ROLLED_BACK = "rolled_back"
    RELEASED = "released"
    EXPIRED = "expired"
    CLEANUP_FAILED = "cleanup_failed"


@dataclass
class BlockReservation:
    """一次 vLLM BlockPool 物理 block 预留记录。

    属性:
        reservation_id: reservation 唯一 ID。
        block_ids: 已由 vLLM BlockPool 分配并增加引用计数的物理 block ID。
        state: 当前 reservation 状态。
        created_at: 创建时的单调时钟时间戳。
        expires_at: reservation 自动回收的单调时钟时间戳。
        finished_at: reservation 进入终态的单调时钟时间戳。
        block_hashes: commit 或 publish 时绑定的逻辑 block hash。
        raw_blocks: vLLM 返回的 KVCacheBlock 对象，仅供提交或回滚生命周期使用。
        planned_blocks: delta reservation 的完整逻辑 block 布局；本地命中位置保存
            已 pin block，缺失位置保存本次预留 block。
        transfer_positions: raw_blocks 在完整逻辑布局中的位置。
        reused_raw_blocks: delta plan 提前 pin 的本地 prefix-cache block。
        block_pool: 分配这些 block 的 vLLM BlockPool 实例。
    """

    reservation_id: str
    block_ids: tuple[int, ...]
    state: BlockReservationState
    created_at: float
    expires_at: float
    finished_at: float | None = None
    block_hashes: tuple[Any, ...] = ()
    raw_blocks: tuple[Any, ...] = field(default=(), repr=False)
    planned_blocks: tuple[Any, ...] = field(default=(), repr=False)
    transfer_positions: tuple[int, ...] = ()
    reused_raw_blocks: tuple[Any, ...] = field(default=(), repr=False)
    block_pool: Any = field(default=None, repr=False)
    cleanup_error: str | None = None
    release_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """返回不包含 vLLM 原始对象的可序列化快照。

        返回:
            reservation_id、block_ids、state、block_hashes 和 age_seconds。
        """

        now = time.monotonic()
        return {
            "reservation_id": self.reservation_id,
            "block_ids": list(self.block_ids),
            "state": self.state.value,
            "cleanup_error": self.cleanup_error,
            "block_hashes": [_json_safe_block_hash(value) for value in self.block_hashes],
            "planned_block_ids": [int(block.block_id) for block in self.planned_blocks],
            "transfer_positions": list(self.transfer_positions),
            "reused_block_ids": [int(block.block_id) for block in self.reused_raw_blocks],
            "age_seconds": max(0.0, now - self.created_at),
            "ttl_remaining_ms": (
                max(0, int((self.expires_at - now) * 1000))
                if self.state
                in {
                    BlockReservationState.RESERVED,
                    BlockReservationState.COMMITTED,
                }
                else 0
            ),
        }


class VllmBlockPoolReservationManager:
    """通过 vLLM BlockPool 官方分配/释放路径管理目标物理 block。

    参数:
        block_pool_provider: 动态返回当前 engine-core BlockPool 的函数。
        lock: 与 vLLM BlockPool hook 共用的可重入锁，用于串行化 scheduler 与
            KVBM 控制线程的分配和释放操作。
        min_ttl_ms: 允许的最短 reservation TTL。
        max_ttl_ms: 允许的最长 reservation TTL。
        completed_retention_seconds: 终态诊断记录保留时长。
        max_completed_records: 最多保留的终态诊断记录数量。
    """

    def __init__(
        self,
        block_pool_provider: Callable[[], Any | None],
        *,
        lock: threading.RLock | None = None,
        min_ttl_ms: int = 10_000,
        max_ttl_ms: int = 1_800_000,
        completed_retention_seconds: float = 300.0,
        max_completed_records: int = 10_000,
    ) -> None:
        self.block_pool_provider = block_pool_provider
        self.lock = lock or threading.RLock()
        self.min_ttl_ms = int(min_ttl_ms)
        self.max_ttl_ms = int(max_ttl_ms)
        self.completed_retention_seconds = float(completed_retention_seconds)
        self.max_completed_records = int(max_completed_records)
        self.reservations: dict[str, BlockReservation] = {}
        self._pending_delta_hashes: dict[str, tuple[Any, ...]] = {}
        self._state_changed = threading.Condition(self.lock)

    def reserve(self, count: int, *, ttl_ms: int = 120_000) -> BlockReservation:
        """从 vLLM free block queue 安全预留指定数量的物理 block。

        参数:
            count: 需要预留的物理 block 数量，必须大于 0。
            ttl_ms: 未完成事务最多持有目标 block 的时间，单位毫秒。

        返回:
            状态为 RESERVED 的 BlockReservation。
        """

        if count <= 0:
            raise BlockReservationError("reservation count must be positive")
        effective_ttl_ms = self._validate_ttl(ttl_ms)
        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            self._prune_completed_locked(now)
            block_pool = self._require_block_pool()
            available = int(block_pool.get_num_free_blocks())
            if count > available:
                raise BlockReservationError(f"insufficient free vLLM blocks: requested={count}, available={available}")
            raw_blocks = tuple(block_pool.get_new_blocks(count))
            if len(raw_blocks) != count:
                if raw_blocks:
                    block_pool.free_blocks(list(raw_blocks))
                raise BlockReservationError(
                    f"vLLM returned unexpected block count: requested={count}, actual={len(raw_blocks)}"
                )
            block_ids = tuple(int(block.block_id) for block in raw_blocks)
            reservation = BlockReservation(
                reservation_id=uuid.uuid4().hex,
                block_ids=block_ids,
                state=BlockReservationState.RESERVED,
                created_at=now,
                expires_at=now + effective_ttl_ms / 1000.0,
                raw_blocks=raw_blocks,
                block_pool=block_pool,
            )
            self.reservations[reservation.reservation_id] = reservation
            return reservation

    def reserve_delta(
        self,
        block_hashes: list[Any | None] | tuple[Any | None, ...],
        *,
        ttl_ms: int = 120_000,
        owner_id: str = "",
    ) -> BlockReservation:
        """原子 pin 本地命中 block，并只为缺失位置预留传输目标。

        参数:
            block_hashes: 完整逻辑布局对应的 packed hash；request-scoped partial
                block 使用 None，因此始终需要预留和传输。
            ttl_ms: delta plan 最多持有本地 pin 和目标 reservation 的时间。
            owner_id: 可选的请求级所有者 ID；容量不足时用于保留
                同hash的唯一等待 leader。

        返回:
            RESERVED 状态的 reservation。``block_ids``/``raw_blocks`` 只包含需要
            HIXL 传输的位置；``planned_blocks`` 与输入完整布局一一对应。

        说明:
            本方法在同一个 BlockPool 临界区内完成查询、pin 和缺失块分配，避免
            本地命中 block 在 HIXL 传输期间被淘汰或复用。
        """

        hashes = tuple(block_hashes)
        if not hashes:
            raise BlockReservationError("delta reservation block_hashes must not be empty")
        effective_ttl_ms = self._validate_ttl(ttl_ms)
        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            self._prune_completed_locked(now)
            block_pool = self._require_block_pool()
            touch = getattr(block_pool, "touch", None)
            if not callable(touch):
                raise BlockReservationError("vLLM BlockPool delta reservation requires touch()")

            planned_blocks: list[Any | None] = [None] * len(hashes)
            reused_blocks: list[Any] = []
            transfer_positions: list[int] = []
            for position, block_hash in enumerate(hashes):
                existing_blocks = self._existing_cached_blocks(block_pool, block_hash) if block_hash is not None else []
                if existing_blocks:
                    selected = existing_blocks[0]
                    planned_blocks[position] = selected
                    reused_blocks.append(selected)
                else:
                    transfer_positions.append(position)

            inflight_reservations: list[str] = []
            inflight_hash_count = 0
            normalized_owner_id = str(owner_id).strip()
            active_hash_owners = self._active_transfer_hash_owners_locked()
            for position in transfer_positions:
                block_hash = hashes[position]
                if block_hash is None:
                    continue
                conflict_owner_id = active_hash_owners.get(block_hash)
                if conflict_owner_id is not None and conflict_owner_id != f"pending:{normalized_owner_id}":
                    inflight_hash_count += 1
                    inflight_reservations.append(conflict_owner_id)
            if inflight_reservations:
                raise BlockHashesInFlightError(
                    inflight_hash_count,
                    inflight_reservations,
                )

            reserved_blocks: tuple[Any, ...] = ()
            reused_ref_counts = [int(getattr(block, "ref_cnt", 0)) for block in reused_blocks]
            try:
                if reused_blocks:
                    touch(reused_blocks)
                transfer_count = len(transfer_positions)
                available = int(block_pool.get_num_free_blocks())
                if transfer_count > available:
                    if normalized_owner_id:
                        self._pending_delta_hashes[normalized_owner_id] = tuple(
                            hashes[position] for position in transfer_positions if hashes[position] is not None
                        )
                        self._state_changed.notify_all()
                    raise BlockReservationCapacityError(
                        transfer_count,
                        available,
                        len(reused_blocks),
                        owner_id=normalized_owner_id,
                    )
                reserved_blocks = tuple(block_pool.get_new_blocks(transfer_count)) if transfer_count else ()
                if len(reserved_blocks) != transfer_count:
                    raise BlockReservationError(
                        "vLLM returned unexpected delta block count: "
                        f"requested={transfer_count}, actual={len(reserved_blocks)}"
                    )
                for position, block in zip(
                    transfer_positions,
                    reserved_blocks,
                    strict=True,
                ):
                    planned_blocks[position] = block
            except Exception:
                if reserved_blocks:
                    block_pool.free_blocks(list(reserved_blocks))
                pinned_reused_blocks = []
                for matched_selected_block, selected_previous_ref_count in zip(
                    reused_blocks, reused_ref_counts, strict=True
                ):
                    if int(getattr(matched_selected_block, "ref_cnt", 0)) > selected_previous_ref_count:
                        pinned_reused_blocks.append(matched_selected_block)
                if pinned_reused_blocks:
                    block_pool.free_blocks(list(reversed(pinned_reused_blocks)))
                raise

            if any(block is None for block in planned_blocks):
                if reserved_blocks:
                    block_pool.free_blocks(list(reserved_blocks))
                if reused_blocks:
                    block_pool.free_blocks(list(reversed(reused_blocks)))
                raise BlockReservationError("delta reservation produced incomplete layout")

            reservation = BlockReservation(
                reservation_id=uuid.uuid4().hex,
                block_ids=tuple(int(block.block_id) for block in reserved_blocks),
                state=BlockReservationState.RESERVED,
                created_at=now,
                expires_at=now + effective_ttl_ms / 1000.0,
                block_hashes=hashes,
                raw_blocks=reserved_blocks,
                planned_blocks=tuple(planned_blocks),
                transfer_positions=tuple(transfer_positions),
                reused_raw_blocks=tuple(reused_blocks),
                block_pool=block_pool,
            )
            if normalized_owner_id:
                self._pending_delta_hashes.pop(normalized_owner_id, None)
                self._state_changed.notify_all()
            self.reservations[reservation.reservation_id] = reservation
            return reservation

    def release_pending_delta(self, owner_id: str) -> bool:
        """释放容量等待 leader 持有的逻辑 hash 声明。

        参数:
            owner_id: ``reserve_delta`` 使用的请求级所有者 ID。

        返回:
            确实删除了等待声明时返回 True，否则返回 False。
        """

        normalized_owner_id = str(owner_id).strip()
        if not normalized_owner_id:
            return False
        with self._state_changed:
            removed = self._pending_delta_hashes.pop(normalized_owner_id, None)
            if removed is not None:
                self._state_changed.notify_all()
            return removed is not None

    def wait_for_free_blocks(self, minimum: int, *, timeout_seconds: float) -> bool:
        """等待 BlockPool 达到指定空闲 block 数量。

        参数:
            minimum: delta leader 下次预留所需的最小 block 数。
            timeout_seconds: 最长等待时间，单位秒。

        返回:
            容量达标时返回 True；超时时返回 False。
        """

        required = max(0, int(minimum))

        def available() -> bool:
            block_pool = self.block_pool_provider()
            return block_pool is not None and int(block_pool.get_num_free_blocks()) >= required

        with self._state_changed:
            return self._state_changed.wait_for(
                available,
                timeout=max(0.0, float(timeout_seconds)),
            )

    def notify_capacity_changed(self) -> None:
        """通知容量等待者 vLLM BlockPool 已发生变化。

        返回:
            无返回值。
        """

        with self._state_changed:
            self._state_changed.notify_all()

    def commit(
        self,
        reservation_id: str,
        *,
        block_hashes: list[str] | tuple[str, ...] = (),
    ) -> BlockReservation:
        """确认一次成功传输对应的 block reservation。

        参数:
            reservation_id: reserve 返回的 reservation ID。
            block_hashes: 与 block_ids 一一对应的逻辑 block hash；为空时只提交
                物理 block 所有权。

        返回:
            状态为 COMMITTED 的 BlockReservation。

        说明:
            commit 不会调用 free_blocks，因此物理 block 继续由该 reservation
            持有；本阶段也不注入 vLLM prefix cache。
        """

        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            reservation = self._require_state(
                reservation_id,
                BlockReservationState.RESERVED,
            )
            hashes = tuple(str(value) for value in block_hashes)
            if hashes and len(hashes) != len(reservation.block_ids):
                raise BlockReservationError(
                    "block_hashes count must match reserved block count: "
                    f"hashes={len(hashes)}, blocks={len(reservation.block_ids)}"
                )
            reservation.block_hashes = hashes
            reservation.state = BlockReservationState.COMMITTED
            return reservation

    def rollback(self, reservation_id: str) -> BlockReservation:
        """回滚尚未提交的 reservation，并把 block 归还 vLLM。

        参数:
            reservation_id: reserve 返回的 reservation ID。

        返回:
            状态为 ROLLED_BACK 的 BlockReservation。
        """

        return self._free(
            reservation_id,
            expected_state=BlockReservationState.RESERVED,
            final_state=BlockReservationState.ROLLED_BACK,
        )

    def renew(self, reservation_id: str, *, ttl_ms: int) -> BlockReservation:
        """延长仍在使用的目标 block reservation。

        参数:
            reservation_id: reserve 返回的 reservation ID。
            ttl_ms: 从当前时刻重新计算的有效期，单位毫秒。

        返回:
            更新到期时间后的 RESERVED 或 COMMITTED reservation。
        """

        effective_ttl_ms = self._validate_ttl(ttl_ms)
        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            reservation = self.get(reservation_id)
            if reservation.state not in {
                BlockReservationState.RESERVED,
                BlockReservationState.COMMITTED,
            }:
                raise BlockReservationError(
                    f"reservation {reservation_id} state is {reservation.state.value}, expected reserved or committed"
                )
            reservation.expires_at = now + effective_ttl_ms / 1000.0
            return reservation

    def publish(
        self,
        reservation_id: str,
        *,
        block_hashes: list[Any] | tuple[Any, ...],
    ) -> BlockReservation:
        """把传输完成的 block 发布到 vLLM prefix cache。

        参数:
            reservation_id: reserve 返回的 reservation ID。
            block_hashes: 与目标物理 block 一一对应的 vLLM 内部完整 hash，
                必须包含 KV cache group ID。

        返回:
            状态为 PUBLISHED 的 BlockReservation。发布后的 block 引用计数降为
            0，并进入 vLLM free queue，后续可被 prefix cache 命中或按 LRU 淘汰。

        异常:
            任一 block 校验或索引写入失败时撤销已发布索引，归还全部预留 block，
            并抛出 BlockReservationError。
        """

        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            reservation = self._require_state(
                reservation_id,
                BlockReservationState.RESERVED,
            )
            hashes = tuple(block_hashes)
            self._validate_publish(reservation, hashes)
            inserted: list[tuple[Any, Any]] = []
            try:
                for block_hash, block in zip(hashes, reservation.raw_blocks):
                    inserted.append((block_hash, block))
                    self._publish_block(reservation.block_pool, block_hash, block)
                self._return_reserved_blocks(reservation, include_reused=False)
            except Exception as exc:
                self._cleanup_failed_operation(reservation, inserted)
                reservation.state = BlockReservationState.ROLLED_BACK
                reservation.finished_at = time.monotonic()
                self._state_changed.notify_all()
                raise BlockReservationError(f"failed to publish vLLM prefix cache blocks: {exc}") from exc

            reservation.block_hashes = hashes
            reservation.state = BlockReservationState.PUBLISHED
            reservation.finished_at = time.monotonic()
            self._state_changed.notify_all()
            self._prune_completed_locked(reservation.finished_at)
            return reservation

    def attach(
        self,
        reservation_id: str,
        *,
        block_hashes: list[Any] | tuple[Any, ...],
        attach_callback: Callable[[tuple[Any, ...]], None],
        rollback_callback: Callable[[], None],
    ) -> BlockReservation:
        """原子发布目标 block，并把 reservation 所有权转交给当前请求。

        参数:
            reservation_id: reserve 返回的 reservation ID。
            block_hashes: 与目标物理 block 一一对应的完整 vLLM packed hash。
            attach_callback: hash 已写入且 reservation 引用仍有效时执行的请求
                block-table 挂接函数，参数为 reservation 原始 block 元组。
            rollback_callback: attach_callback 失败后撤销请求侧部分挂接的函数。

        返回:
            状态为 ATTACHED 的 BlockReservation。成功后 reservation 引用被释放，
            block 仅由目标请求持有；失败时 hash 索引、请求引用和 reservation
            引用全部回滚。
        """

        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            reservation = self._require_state(
                reservation_id,
                BlockReservationState.RESERVED,
            )
            hashes = tuple(block_hashes)
            self._validate_publish(reservation, hashes)
            inserted: list[tuple[Any, Any]] = []
            try:
                for block_hash, block in zip(hashes, reservation.raw_blocks):
                    inserted.append((block_hash, block))
                    self._publish_block(reservation.block_pool, block_hash, block)
                attach_callback(reservation.raw_blocks)
                self._return_reserved_blocks(reservation, include_reused=False)
            except Exception as exc:
                self._cleanup_failed_operation(reservation, inserted, rollback_callback)
                reservation.state = BlockReservationState.ROLLED_BACK
                reservation.finished_at = time.monotonic()
                self._state_changed.notify_all()
                raise BlockReservationError(f"failed to attach vLLM request blocks: {exc}") from exc

            reservation.block_hashes = hashes
            reservation.state = BlockReservationState.ATTACHED
            reservation.finished_at = time.monotonic()
            self._state_changed.notify_all()
            self._prune_completed_locked(reservation.finished_at)
            return reservation

    def attach_reusing_published(
        self,
        reservation_id: str,
        *,
        block_hashes: list[Any] | tuple[Any, ...],
        attach_callback: Callable[[tuple[Any, ...]], None],
        rollback_callback: Callable[[], None],
    ) -> BlockReservation:
        """发布 reservation，并在 hash 已存在时复用现有物理 block。

        参数:
            reservation_id: reserve 返回的 reservation ID。
            block_hashes: 与已完成传输的 reservation block 一一对应的完整
                vLLM packed hash。
            attach_callback: 接收最终选中的物理 block；每个位置可能是本次
                reservation block，也可能是并发请求已发布的同 hash block。
            rollback_callback: attach_callback 失败后撤销请求侧引用和 block
                table 的函数。

        返回:
            状态为 ATTACHED 的 reservation。``block_ids`` 表示请求最终实际
            持有的物理 block；未被采用的重复 reservation block 已归还池中。

        说明:
            本方法在 BlockPool 共用锁内完成“查重、发布、请求挂接”，避免检查
            hash 后再次出现并发发布窗口。已有 block 的 KV 内容由完整 packed
            hash 保证等价，不会使用刚拉取但未发布的重复副本。
        """

        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            reservation = self._require_state(
                reservation_id,
                BlockReservationState.RESERVED,
            )
            hashes = tuple(block_hashes)
            self._validate_publish(
                reservation,
                hashes,
                reject_existing=False,
            )
            selected_blocks: list[Any] = []
            inserted: list[tuple[Any, Any]] = []
            try:
                for block_hash, reserved_block in zip(
                    hashes,
                    reservation.raw_blocks,
                    strict=True,
                ):
                    existing_blocks = self._existing_cached_blocks(
                        reservation.block_pool,
                        block_hash,
                    )
                    if existing_blocks:
                        selected_blocks.append(existing_blocks[0])
                        continue
                    inserted.append((block_hash, reserved_block))
                    self._publish_block(
                        reservation.block_pool,
                        block_hash,
                        reserved_block,
                    )
                    selected_blocks.append(reserved_block)
                attach_callback(tuple(selected_blocks))
                self._return_reserved_blocks(reservation, include_reused=False)
            except Exception as exc:
                self._cleanup_failed_operation(reservation, inserted, rollback_callback)
                reservation.state = BlockReservationState.ROLLED_BACK
                reservation.finished_at = time.monotonic()
                self._state_changed.notify_all()
                raise BlockReservationError(f"failed to attach vLLM request blocks: {exc}") from exc

            reservation.block_ids = tuple(int(block.block_id) for block in selected_blocks)
            reservation.block_hashes = hashes
            reservation.state = BlockReservationState.ATTACHED
            reservation.finished_at = time.monotonic()
            self._state_changed.notify_all()
            self._prune_completed_locked(reservation.finished_at)
            return reservation

    def attach_request_scoped(
        self,
        reservation_id: str,
        *,
        block_hashes: list[Any | None] | tuple[Any | None, ...],
        attach_callback: Callable[[tuple[Any, ...]], None],
        rollback_callback: Callable[[], None],
    ) -> BlockReservation:
        """把完整 block 发布到 prefix cache，并把 partial block 仅挂接给请求。

        参数:
            reservation_id: reserve 返回的 reservation ID。
            block_hashes: 与 reservation block 一一对应的 packed hash。完整 block
                提供 hash；请求尾部 partial block 使用 None，禁止进入通用缓存。
            attach_callback: 接收最终选中 block 的请求 block-table 挂接函数。
            rollback_callback: 挂接失败时撤销请求引用和 token 进度的函数。

        返回:
            状态为 ATTACHED 的 reservation。完整 hash 已存在时复用现有 block；
            None 对应的 partial block 始终使用本次 HIXL 写入的 reservation block。

        异常:
            数量不匹配、发布或请求挂接失败时回滚全部新索引和 reservation 引用。
        """

        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            reservation = self._require_state(
                reservation_id,
                BlockReservationState.RESERVED,
            )
            hashes = tuple(block_hashes)
            delta_planned = bool(reservation.planned_blocks)
            expected_count = len(reservation.planned_blocks) if delta_planned else len(reservation.raw_blocks)
            if len(hashes) != expected_count:
                raise BlockReservationError(
                    f"block_hashes count must match planned block count: hashes={len(hashes)}, blocks={expected_count}"
                )
            if delta_planned and hashes != reservation.block_hashes:
                raise BlockReservationError("delta attach block_hashes do not match reservation plan")
            selected_blocks: list[Any] = list(reservation.planned_blocks) if delta_planned else []
            inserted: list[tuple[Any, Any]] = []
            try:
                if delta_planned:
                    for position, reserved_block in zip(
                        reservation.transfer_positions,
                        reservation.raw_blocks,
                        strict=True,
                    ):
                        block_hash = hashes[position]
                        if block_hash is None:
                            continue
                        inserted.append((block_hash, reserved_block))
                        self._publish_block(
                            reservation.block_pool,
                            block_hash,
                            reserved_block,
                        )
                else:
                    for block_hash, reserved_block in zip(
                        hashes,
                        reservation.raw_blocks,
                        strict=True,
                    ):
                        if block_hash is None:
                            selected_blocks.append(reserved_block)
                            continue
                        existing_blocks = self._existing_cached_blocks(
                            reservation.block_pool,
                            block_hash,
                        )
                        if existing_blocks:
                            selected_blocks.append(existing_blocks[0])
                            continue
                        inserted.append((block_hash, reserved_block))
                        self._publish_block(
                            reservation.block_pool,
                            block_hash,
                            reserved_block,
                        )
                        selected_blocks.append(reserved_block)
                attach_callback(tuple(selected_blocks))
                self._return_reserved_blocks(reservation)
            except Exception as exc:
                self._cleanup_failed_operation(reservation, inserted, rollback_callback)
                reservation.state = BlockReservationState.ROLLED_BACK
                reservation.finished_at = time.monotonic()
                self._state_changed.notify_all()
                raise BlockReservationError(f"failed to attach request-scoped vLLM blocks: {exc}") from exc

            reservation.block_ids = tuple(int(block.block_id) for block in selected_blocks)
            reservation.block_hashes = hashes
            reservation.state = BlockReservationState.ATTACHED
            reservation.finished_at = time.monotonic()
            self._state_changed.notify_all()
            self._prune_completed_locked(reservation.finished_at)
            return reservation

    def wait_for_inflight_reservations(
        self,
        reservation_ids: list[str] | tuple[str, ...],
        *,
        timeout_seconds: float,
    ) -> bool:
        """等待指定 leader reservation 离开传输中状态。

        参数:
            reservation_ids: ``BlockHashesInFlightError`` 返回的 leader ID。
            timeout_seconds: 最长等待时间，单位秒。

        返回:
            所有 leader 已 publish、attach、rollback 或释放时返回 True；
            超时时返回 False。

        说明:
            该方法仅阻塞控制线程，不会阻塞 vLLM Scheduler 线程。
        """

        pending_ids = frozenset(str(value) for value in reservation_ids)
        if not pending_ids:
            return True

        def completed() -> bool:
            for reservation_id in pending_ids:
                if reservation_id.startswith("pending:"):
                    owner_id = reservation_id.removeprefix("pending:")
                    if owner_id in self._pending_delta_hashes:
                        return False
                    continue
                reservation = self.reservations.get(reservation_id)
                if reservation is not None and reservation.state in {
                    BlockReservationState.RESERVED,
                    BlockReservationState.COMMITTED,
                }:
                    return False
            return True

        with self._state_changed:
            return self._state_changed.wait_for(
                completed,
                timeout=max(0.0, float(timeout_seconds)),
            )

    def release(self, reservation_id: str) -> BlockReservation:
        """释放已提交但尚未被 prefix cache 消费的 reservation。

        参数:
            reservation_id: commit 后的 reservation ID。

        返回:
            状态为 RELEASED 的 BlockReservation。
        """

        return self._free(
            reservation_id,
            expected_state=BlockReservationState.COMMITTED,
            final_state=BlockReservationState.RELEASED,
        )

    def get(self, reservation_id: str) -> BlockReservation:
        """读取 reservation。

        参数:
            reservation_id: reservation 唯一 ID。

        返回:
            对应 BlockReservation。
        """

        reservation = self.reservations.get(reservation_id)
        if reservation is None:
            raise BlockReservationError(f"unknown reservation_id: {reservation_id}")
        return reservation

    def expire_due(self) -> int:
        """释放所有已到期且仍持有物理 block 的 reservation。

        返回:
            本次转为 EXPIRED 状态的 reservation 数量。
        """

        with self.lock:
            now = time.monotonic()
            expired = self._expire_due_locked(now)
            self._prune_completed_locked(now)
            return expired

    def stats(self) -> dict[str, Any]:
        """返回 reservation 和当前 BlockPool 空闲统计。

        返回:
            包含 pool_ready、free_blocks 和各状态计数的字典。
        """

        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            self._prune_completed_locked(now)
            block_pool = self.block_pool_provider()
            counts = {state.value: 0 for state in BlockReservationState}
            for reservation in self.reservations.values():
                counts[reservation.state.value] += 1
            active_expirations = []
            for selected_reservation in self.reservations.values():
                if selected_reservation.state in {BlockReservationState.RESERVED, BlockReservationState.COMMITTED}:
                    active_expirations.append(selected_reservation.expires_at)
            return {
                "pool_ready": block_pool is not None,
                "free_blocks": (int(block_pool.get_num_free_blocks()) if block_pool is not None else None),
                "states": counts,
                "next_expiry_ms": (max(0, int((min(active_expirations) - now) * 1000)) if active_expirations else None),
            }

    def _free(
        self,
        reservation_id: str,
        *,
        expected_state: BlockReservationState,
        final_state: BlockReservationState,
    ) -> BlockReservation:
        """通过 reservation 原始 BlockPool 释放 block 并更新状态。"""

        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            reservation = self._require_state(reservation_id, expected_state)
            self._cleanup_failed_operation(reservation, [])
            reservation.state = final_state
            reservation.finished_at = time.monotonic()
            self._state_changed.notify_all()
            self._prune_completed_locked(reservation.finished_at)
            return reservation

    def _expire_due_locked(self, now: float) -> int:
        """在持锁状态下回收所有已超过 TTL 的目标 block。

        参数:
            now: 当前单调时钟时间戳。

        返回:
            本次转为 EXPIRED 的 reservation 数量。
        """

        expired = 0
        for reservation in self.reservations.values():
            if (
                reservation.state
                in {
                    BlockReservationState.RESERVED,
                    BlockReservationState.COMMITTED,
                }
                and reservation.expires_at <= now
            ):
                try:
                    self._cleanup_failed_operation(reservation, [])
                except BlockReservationError:
                    logger.exception("Reservation expiry failed; continuing other records")
                    continue
                reservation.state = BlockReservationState.EXPIRED
                reservation.finished_at = now
                expired += 1
        if expired:
            self._state_changed.notify_all()
        return expired

    def _active_transfer_hash_owners_locked(self) -> dict[Any, str]:
        """返回尚未发布的完整 block hash 与 leader reservation 映射。

        返回:
            key 为完整 packed hash，value 为首个持有者的 reservation ID。
            request-scoped partial block 的 hash 为 None，不参与合并。
        """

        owners: dict[Any, str] = {}
        for reservation in self.reservations.values():
            if reservation.state not in {
                BlockReservationState.RESERVED,
                BlockReservationState.COMMITTED,
            }:
                continue
            for position in reservation.transfer_positions:
                if position >= len(reservation.block_hashes):
                    continue
                block_hash = reservation.block_hashes[position]
                if block_hash is not None:
                    owners.setdefault(block_hash, reservation.reservation_id)
        for owner_id, block_hashes in self._pending_delta_hashes.items():
            for block_hash in block_hashes:
                owners.setdefault(block_hash, f"pending:{owner_id}")
        return owners

    def _prune_completed_locked(self, now: float) -> None:
        """删除过期或超出数量上限的终态 reservation 诊断记录。

        参数:
            now: 当前单调时钟时间戳。

        返回:
            无返回值。
        """

        completed = [reservation for reservation in self.reservations.values() if reservation.finished_at is not None]
        for reservation in completed:
            if not (reservation.finished_at is not None):
                raise RuntimeError("Required invariant failed: reservation.finished_at is not None")
            if now - reservation.finished_at >= self.completed_retention_seconds:
                self.reservations.pop(reservation.reservation_id, None)

        completed = sorted(
            (reservation for reservation in self.reservations.values() if reservation.finished_at is not None),
            key=lambda reservation: reservation.finished_at or 0.0,
        )
        overflow = len(completed) - self.max_completed_records
        for reservation in completed[: max(0, overflow)]:
            self.reservations.pop(reservation.reservation_id, None)

    def _validate_ttl(self, ttl_ms: int) -> int:
        """校验并返回 reservation TTL。

        参数:
            ttl_ms: 调用方请求的 reservation TTL，单位毫秒。

        返回:
            校验后的 TTL 毫秒值。
        """

        value = int(ttl_ms)
        if value < self.min_ttl_ms or value > self.max_ttl_ms:
            raise BlockReservationError(f"reservation ttl_ms must be between {self.min_ttl_ms} and {self.max_ttl_ms}")
        return value

    def _validate_publish(
        self,
        reservation: BlockReservation,
        block_hashes: tuple[Any, ...],
        *,
        reject_existing: bool = True,
    ) -> None:
        """在修改 BlockPool 前校验 publish 的全部输入。

        参数:
            reservation: 待发布的目标 block reservation。
            block_hashes: 与 reservation block 对齐的完整 packed hash。
            reject_existing: 已存在相同 hash 时是否抛出
                BlockAlreadyPublishedError。

        返回:
            校验成功时无返回值；输入或 BlockPool 状态不合法时抛出异常。
        """

        if len(block_hashes) != len(reservation.raw_blocks):
            raise BlockReservationError(
                "block_hashes count must match reserved block count: "
                f"hashes={len(block_hashes)}, blocks={len(reservation.raw_blocks)}"
            )
        if any(value is None for value in block_hashes):
            raise BlockReservationError("published block hashes must not contain None")
        block_pool = reservation.block_pool
        if not bool(getattr(block_pool, "enable_caching", True)):
            raise BlockReservationError("vLLM prefix caching is disabled")
        cached_map = getattr(block_pool, "cached_block_hash_to_block", None)
        if not callable(getattr(block_pool, "_insert_block_hash", None)) and not callable(
            getattr(cached_map, "insert", None)
        ):
            raise BlockReservationError(
                "vLLM BlockPool exposes neither _insert_block_hash nor cached_block_hash_to_block.insert"
            )
        for block in reservation.raw_blocks:
            if getattr(block, "block_hash", None) is not None:
                raise BlockReservationError(f"reserved block {getattr(block, 'block_id', '?')} already has a hash")
            if int(getattr(block, "ref_cnt", 0)) != 1:
                raise BlockReservationError(
                    f"reserved block {getattr(block, 'block_id', '?')} has unexpected "
                    f"ref_cnt={getattr(block, 'ref_cnt', None)}"
                )
        if reject_existing:
            for block_hash in block_hashes:
                existing_ids = self._existing_cached_block_ids(block_pool, block_hash)
                if existing_ids:
                    raise BlockAlreadyPublishedError(existing_ids)

    @staticmethod
    def _existing_cached_blocks(block_pool: Any, block_hash: Any) -> list[Any]:
        """查询同一完整 hash 已发布的物理 block 对象。

        参数:
            block_pool: 当前 vLLM BlockPool。
            block_hash: 包含 KV cache group ID 的完整内部 hash。

        返回:
            按物理 block ID 排序的已发布 block；未命中时返回空列表。
        """

        cached_map = getattr(block_pool, "cached_block_hash_to_block", None)
        for attribute in ("_cache", "values"):
            cache = getattr(cached_map, attribute, None)
            if not isinstance(cache, dict):
                continue
            blocks = cache.get(block_hash)
            if blocks is not None:
                return sorted(
                    _iter_cached_blocks(blocks),
                    key=lambda block: int(block.block_id),
                )

        blocks = getattr(block_pool, "blocks", None)
        if isinstance(blocks, list | tuple):
            return sorted(
                (block for block in blocks if getattr(block, "block_hash", None) == block_hash),
                key=lambda block: int(block.block_id),
            )
        return []

    @staticmethod
    def _existing_cached_block_ids(block_pool: Any, block_hash: Any) -> list[int]:
        """查询同一完整 hash 当前已经发布的物理 block。

        参数:
            block_pool: 当前 vLLM BlockPool。
            block_hash: 包含 KV cache group ID 的完整内部 hash。

        返回:
            已发布相同 hash 的物理 block ID；未命中时返回空列表。

        说明:
            优先读取 BlockHashToBlockMap 的内部快照；若当前 vLLM 版本未暴露
            该结构，则退化为扫描 BlockPool.blocks。调用方已经持有与 vLLM
            mutation hook 共用的锁，因此查询和随后 publish 之间没有窗口。
        """

        selected_block_ids = []
        for selected_block in VllmBlockPoolReservationManager._existing_cached_blocks(block_pool, block_hash):
            selected_block_ids.append(int(selected_block.block_id))
        return selected_block_ids

    @staticmethod
    def _publish_block(block_pool: Any, block_hash: Any, block: Any) -> None:
        """使用 vLLM 自有 hash 插入路径发布单个 block。"""

        insert_method = getattr(block_pool, "_insert_block_hash", None)
        if callable(insert_method):
            insert_method(block_hash, block)
            if getattr(block, "block_hash", None) is None:
                block.block_hash = block_hash
            return

        block.block_hash = block_hash
        block_pool.cached_block_hash_to_block.insert(block_hash, block)

    def _cleanup_failed_operation(self, reservation, inserted, rollback_callback=None) -> None:
        errors = []
        unsafe_ids = set()
        if rollback_callback is not None:
            try:
                rollback_callback()
            except Exception as exc:
                errors.append(f"request rollback: {exc}")
        try:
            self._undo_published_blocks(reservation.block_pool, inserted)
        except _IndexRollbackError as exc:
            unsafe_ids = exc.unsafe_ids
            errors.append(str(exc))
        except Exception as exc:
            unsafe_ids = {int(block.block_id) for _, block in inserted}
            errors.append(f"index rollback: {exc}")
        try:
            self._return_reserved_blocks(reservation, excluded_ids=unsafe_ids)
        except Exception as exc:
            errors.append(f"block release: {exc}")
        if errors:
            reservation.state = BlockReservationState.CLEANUP_FAILED
            reservation.cleanup_error = "; ".join(errors)
            self._state_changed.notify_all()
            logger.error("Reservation cleanup failed; retaining ownership record: %s", reservation.reservation_id)
            raise BlockReservationError(reservation.cleanup_error)

    @staticmethod
    def _undo_published_blocks(
        block_pool: Any,
        inserted: list[tuple[Any, Any]],
    ) -> None:
        """撤销一次失败 publish 已写入的 hash 索引。"""

        cached_map = getattr(block_pool, "cached_block_hash_to_block", None)
        pop = getattr(cached_map, "pop", None)
        unsafe_ids = set()
        for block_hash, block in reversed(inserted):
            try:
                if not callable(pop):
                    raise BlockReservationError("BlockPool index does not support rollback")
                pop(block_hash, int(block.block_id))
                if getattr(block, "block_hash", None) == block_hash:
                    reset_hash = getattr(block, "reset_hash", None)
                    if callable(reset_hash):
                        reset_hash()
                    elif hasattr(block, "_block_hash"):
                        # Older blocks need the compatibility reset during rollback.
                        block._block_hash = None  # pylint: disable=protected-access
                    else:
                        block.block_hash = None
            except Exception:
                unsafe_ids.add(int(block.block_id))
                logger.exception("Block index cleanup failed; block remains pinned: %s", block.block_id)
        if unsafe_ids:
            raise _IndexRollbackError(unsafe_ids)

    @staticmethod
    def _return_reserved_blocks(
        reservation: BlockReservation, *, excluded_ids: set[int] | None = None, include_reused: bool = True
    ) -> None:
        """仅在 block 仍被 reservation/delta pin 引用时归还物理 block。"""

        if reservation.release_error:
            raise BlockReservationError("partial block release requires pool recovery; refusing blind retry")
        owned = (*reservation.raw_blocks, *(reservation.reused_raw_blocks if include_reused else ()))
        blocks = [
            block
            for block in owned
            if int(getattr(block, "ref_cnt", 0)) > 0 and int(block.block_id) not in (excluded_ids or set())
        ]
        if blocks:
            try:
                reservation.block_pool.free_blocks(blocks)
            except Exception as exc:
                reservation.release_error = str(exc) or type(exc).__name__
                raise

    def _require_block_pool(self) -> Any:
        """返回支持 reservation API 的 BlockPool，否则抛出明确异常。"""

        block_pool = self.block_pool_provider()
        required = ("get_num_free_blocks", "get_new_blocks", "free_blocks")
        missing = [name for name in required if block_pool is None or not hasattr(block_pool, name)]
        if missing:
            raise BlockReservationError(f"vLLM BlockPool reservation API is not ready; missing={missing}")
        return block_pool

    def _require_state(
        self,
        reservation_id: str,
        expected_state: BlockReservationState,
    ) -> BlockReservation:
        """读取 reservation 并校验其当前状态。"""

        reservation = self.get(reservation_id)
        if reservation.state != expected_state:
            raise BlockReservationError(
                f"reservation {reservation_id} state is {reservation.state.value}, expected {expected_state.value}"
            )
        return reservation


def _json_safe_block_hash(value: Any) -> Any:
    """把内部 block hash 转为可安全输出到 JSON 的诊断值。

    参数:
        value: vLLM 内部 block hash。

    返回:
        bytes 使用十六进制对象表示，其余标量尽量原样返回。
    """

    if isinstance(value, bytes | bytearray):
        return {"packed_hex": bytes(value).hex()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)
