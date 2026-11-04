from __future__ import annotations

import threading
import time
import uuid
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SourceBlockLeaseError(RuntimeError):
    """vLLM 源 block lease 操作错误。"""


class SourceBlockLeaseState(str, Enum):
    """源 block lease 生命周期状态。"""

    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"
    RELEASE_FAILED = "release_failed"


@dataclass
class SourceBlockLease:
    """一次源端 vLLM block 引用租约。

    属性:
        lease_id: lease 唯一 ID。
        block_ids: 被保护的源端物理 block ID。
        block_hashes: 获取 lease 时校验并记录的完整 vLLM block hash。
        state: 当前 lease 状态。
        created_at: 创建时的单调时钟时间戳。
        expires_at: lease 到期的单调时钟时间戳。
        finished_at: release 或 expire 完成时的单调时钟时间戳。
        raw_blocks: 被 vLLM `BlockPool.touch()` 增加引用的原始 block 对象。
        block_pool: 持有这些 block 的 vLLM BlockPool。
    """

    lease_id: str
    block_ids: tuple[int, ...]
    block_hashes: tuple[Any, ...]
    state: SourceBlockLeaseState
    created_at: float
    expires_at: float
    finished_at: float | None = None
    raw_blocks: tuple[Any, ...] = field(default=(), repr=False)
    block_pool: Any = field(default=None, repr=False)
    cleanup_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """返回不包含 vLLM 原始对象的 lease 快照。

        返回:
            包含 lease ID、物理 block ID、状态、租期和 hash 的可序列化字典。
        """

        now = time.monotonic()
        return {
            "lease_id": self.lease_id,
            "block_ids": list(self.block_ids),
            "block_hashes": [_json_safe_hash(value) for value in self.block_hashes],
            "state": self.state.value,
            "cleanup_error": self.cleanup_error,
            "age_seconds": max(0.0, now - self.created_at),
            "ttl_remaining_ms": max(0, int((self.expires_at - now) * 1000)),
        }


class VllmSourceBlockLeaseManager:
    """通过 vLLM BlockPool 引用计数保护跨 worker 传输的源 block。

    参数:
        block_pool_provider: 动态返回当前 engine-core BlockPool 的函数。
        lock: 与 vLLM scheduler 共用的可重入锁。
        min_ttl_ms: 允许的最短 lease 时长。
        max_ttl_ms: 允许的最长 lease 时长。
        completed_retention_seconds: 已完成 lease 诊断记录的保留时间。
        max_completed_records: 最多保留的已完成 lease 诊断记录数。
    """

    def __init__(
        self,
        block_pool_provider: Callable[[], Any | None],
        *,
        lock: threading.RLock | None = None,
        min_ttl_ms: int = 1_000,
        max_ttl_ms: int = 600_000,
        completed_retention_seconds: float = 300.0,
        max_completed_records: int = 10_000,
    ) -> None:
        self.block_pool_provider = block_pool_provider
        self.lock = lock or threading.RLock()
        self.min_ttl_ms = int(min_ttl_ms)
        self.max_ttl_ms = int(max_ttl_ms)
        self.completed_retention_seconds = float(completed_retention_seconds)
        self.max_completed_records = int(max_completed_records)
        self.leases: dict[str, SourceBlockLease] = {}

    def acquire(
        self,
        block_ids: list[int] | tuple[int, ...],
        *,
        ttl_ms: int,
        require_published: bool = True,
    ) -> SourceBlockLease:
        """校验并 pin 一组源端物理 block。

        参数:
            block_ids: 需要保护的源端物理 block ID，必须非空且不能重复。
            ttl_ms: lease 有效期，单位毫秒。
            require_published: 是否要求每个 block 已写入 prefix-cache hash。
                普通跨请求复用必须为 True；PD 请求尾部 partial block 尚未形成
                完整 hash，只有在请求释放前捕获时才允许传 False。

        返回:
            状态为 ACTIVE 的 SourceBlockLease，其中 hash 与 pin 操作位于同一
            BlockPool 临界区，不存在 metadata 与引用增加之间的竞态窗口。
        """

        ids = tuple(int(value) for value in block_ids)
        if not ids:
            raise SourceBlockLeaseError("source lease block_ids must not be empty")
        if len(set(ids)) != len(ids):
            raise SourceBlockLeaseError("source lease block_ids must not contain duplicates")
        effective_ttl_ms = self._validate_ttl(ttl_ms)

        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            self._prune_completed_locked(now)
            block_pool = self._require_block_pool()
            blocks = getattr(block_pool, "blocks", None)
            if not isinstance(blocks, list | tuple):
                raise SourceBlockLeaseError("vLLM BlockPool does not expose blocks")

            raw_blocks = []
            block_hashes = []
            for block_id in ids:
                if block_id < 0 or block_id >= len(blocks):
                    raise SourceBlockLeaseError(
                        f"source block_id out of range: block_id={block_id}, block_count={len(blocks)}"
                    )
                block = blocks[block_id]
                block_hash = getattr(block, "block_hash", None)
                if require_published and block_hash is None:
                    raise SourceBlockLeaseError(f"source block {block_id} is not published in prefix cache")
                raw_blocks.append(block)
                block_hashes.append(block_hash)

            previous_ref_counts = [int(getattr(block, "ref_cnt", 0)) for block in raw_blocks]
            try:
                block_pool.touch(raw_blocks)
            except Exception as exc:
                incremented = [
                    block
                    for block, previous in zip(raw_blocks, previous_ref_counts)
                    if int(getattr(block, "ref_cnt", 0)) > previous
                ]
                if incremented:
                    block_pool.free_blocks(reversed(incremented))
                raise SourceBlockLeaseError(f"failed to pin vLLM source blocks: {exc}") from exc

            now = time.monotonic()
            lease = SourceBlockLease(
                lease_id=uuid.uuid4().hex,
                block_ids=ids,
                block_hashes=tuple(block_hashes),
                state=SourceBlockLeaseState.ACTIVE,
                created_at=now,
                expires_at=now + effective_ttl_ms / 1000.0,
                raw_blocks=tuple(raw_blocks),
                block_pool=block_pool,
            )
            self.leases[lease.lease_id] = lease
            return lease

    def renew(self, lease_id: str, *, ttl_ms: int) -> SourceBlockLease:
        """延长一个仍然有效的源 block lease。

        参数:
            lease_id: acquire 返回的 lease ID。
            ttl_ms: 从当前时刻重新计算的有效期，单位毫秒。

        返回:
            更新到期时间后的 ACTIVE lease。
        """

        effective_ttl_ms = self._validate_ttl(ttl_ms)
        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            lease = self._require_lease(lease_id)
            if lease.state != SourceBlockLeaseState.ACTIVE:
                raise SourceBlockLeaseError(f"source lease {lease_id} is {lease.state.value}, expected active")
            lease.expires_at = now + effective_ttl_ms / 1000.0
            return lease

    def release(self, lease_id: str) -> SourceBlockLease:
        """释放源 block lease 持有的 vLLM 引用。

        参数:
            lease_id: acquire 返回的 lease ID。

        返回:
            状态为 RELEASED 的 lease；已释放或已过期时幂等返回原记录。
        """

        with self.lock:
            self._expire_due_locked(time.monotonic())
            lease = self._require_lease(lease_id)
            if lease.state == SourceBlockLeaseState.RELEASE_FAILED:
                raise SourceBlockLeaseError(f"source lease cleanup requires pool recovery: {lease_id}")
            if lease.state != SourceBlockLeaseState.ACTIVE:
                return lease
            try:
                lease.block_pool.free_blocks(reversed(lease.raw_blocks))
            except Exception as exc:
                lease.state = SourceBlockLeaseState.RELEASE_FAILED
                lease.cleanup_error = str(exc)
                logger.exception("Source lease release failed; retaining ownership record: %s", lease_id)
                raise SourceBlockLeaseError(f"source lease release failed: {lease_id}") from exc
            lease.state = SourceBlockLeaseState.RELEASED
            lease.finished_at = time.monotonic()
            self._prune_completed_locked(lease.finished_at)
            return lease

    def expire_due(self) -> int:
        """释放所有已到期 lease。

        返回:
            本次从 ACTIVE 转为 EXPIRED 的 lease 数量。
        """

        with self.lock:
            now = time.monotonic()
            expired = self._expire_due_locked(now)
            self._prune_completed_locked(now)
            return expired

    def stats(self) -> dict[str, Any]:
        """返回源 block lease 和 BlockPool 状态。

        返回:
            包含 pool_ready、各状态数量和最近到期时间的诊断字典。
        """

        with self.lock:
            now = time.monotonic()
            self._expire_due_locked(now)
            self._prune_completed_locked(now)
            block_pool = self.block_pool_provider()
            counts = {state.value: 0 for state in SourceBlockLeaseState}
            for lease in self.leases.values():
                counts[lease.state.value] += 1
            active_expirations = [
                lease.expires_at for lease in self.leases.values() if lease.state == SourceBlockLeaseState.ACTIVE
            ]
            return {
                "pool_ready": block_pool is not None,
                "states": counts,
                "next_expiry_ms": (max(0, int((min(active_expirations) - now) * 1000)) if active_expirations else None),
            }

    def _expire_due_locked(self, now: float) -> int:
        """在持锁状态下释放所有到期 lease。"""

        expired = 0
        for lease in self.leases.values():
            if lease.state == SourceBlockLeaseState.ACTIVE and lease.expires_at <= now:
                try:
                    lease.block_pool.free_blocks(reversed(lease.raw_blocks))
                except Exception as exc:
                    lease.state = SourceBlockLeaseState.RELEASE_FAILED
                    lease.cleanup_error = str(exc)
                    logger.exception("Source lease expiry failed; continuing other leases: %s", lease.lease_id)
                    continue
                lease.state = SourceBlockLeaseState.EXPIRED
                lease.finished_at = now
                expired += 1
        return expired

    def _prune_completed_locked(self, now: float) -> None:
        """删除过期或超出数量上限的已完成 lease 诊断记录。"""

        completed = [lease for lease in self.leases.values() if lease.finished_at is not None]
        for lease in completed:
            if not (lease.finished_at is not None):
                raise RuntimeError("Required invariant failed: lease.finished_at is not None")
            if now - lease.finished_at >= self.completed_retention_seconds:
                self.leases.pop(lease.lease_id, None)

        completed = sorted(
            (lease for lease in self.leases.values() if lease.finished_at is not None),
            key=lambda lease: lease.finished_at or 0.0,
        )
        overflow = len(completed) - self.max_completed_records
        for lease in completed[: max(0, overflow)]:
            self.leases.pop(lease.lease_id, None)

    def _validate_ttl(self, ttl_ms: int) -> int:
        """校验并返回 lease TTL 毫秒值。"""

        value = int(ttl_ms)
        if value < self.min_ttl_ms or value > self.max_ttl_ms:
            raise SourceBlockLeaseError(f"source lease ttl_ms must be between {self.min_ttl_ms} and {self.max_ttl_ms}")
        return value

    def _require_block_pool(self) -> Any:
        """返回支持 source lease 的 vLLM BlockPool。"""

        block_pool = self.block_pool_provider()
        required = ("touch", "free_blocks")
        missing = [name for name in required if block_pool is None or not callable(getattr(block_pool, name, None))]
        if missing:
            raise SourceBlockLeaseError(f"vLLM BlockPool source lease API is not ready; missing={missing}")
        return block_pool

    def _require_lease(self, lease_id: str) -> SourceBlockLease:
        """读取 lease，不存在时抛出明确错误。"""

        lease = self.leases.get(lease_id)
        if lease is None:
            raise SourceBlockLeaseError(f"unknown source lease_id: {lease_id}")
        return lease


def _json_safe_hash(value: Any) -> Any:
    """把内部 block hash 转为可安全输出到 JSON 的值。

    参数:
        value: vLLM 内部完整 block hash。

    返回:
        bytes 使用 packed_hex 对象表示，其余值使用 repr。
    """

    if isinstance(value, bytes | bytearray):
        return {"packed_hex": bytes(value).hex()}
    return repr(value)
