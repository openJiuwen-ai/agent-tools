from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TransferTransactionError(RuntimeError):
    """跨 Worker KV block 传输事务错误。"""


class TransferCapacityError(TransferTransactionError):
    """跨 Worker 传输并发容量已耗尽。"""


class TransferCancelledError(TransferTransactionError):
    """跨 Worker 传输已被调用方取消。"""


class TransferTransactionState(str, Enum):
    """跨 Worker KV block 传输事务状态。"""

    CREATED = "created"
    LEASED = "leased"
    RESERVED = "reserved"
    TRANSFERRING = "transferring"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"
    FAILED = "failed"


_TERMINAL_STATES = {
    TransferTransactionState.COMPLETED,
    TransferTransactionState.ROLLED_BACK,
    TransferTransactionState.CANCELLED,
    TransferTransactionState.FAILED,
}

_CANCELLABLE_STATES = {
    TransferTransactionState.CREATED,
    TransferTransactionState.LEASED,
    TransferTransactionState.RESERVED,
    TransferTransactionState.TRANSFERRING,
}


@dataclass
class TransferTransaction:
    """一次跨 Worker KV block 传输事务。

    属性:
        transfer_id: 调用方提供或 KVBM 生成的幂等事务 ID。
        request_fingerprint: 不包含超时参数的请求指纹，用于拒绝 ID 冲突。
        source_blocks: 远端源物理 block ID。
        state: 当前事务状态。
        created_at: 创建事务时的单调时钟时间戳。
        updated_at: 最近一次状态变化的单调时钟时间戳。
        source_lease_id: 当前事务持有的源 block lease ID。
        reservation_id: 当前事务持有的目标 block reservation ID。
        destination_blocks: 目标端预留的物理 block ID。
        cancel_requested: 是否已经收到协作式取消请求。
        error: 终态错误信息。
        result: 终态操作结果，不包含事务自身快照。
    """

    transfer_id: str
    request_fingerprint: str
    source_blocks: tuple[int, ...]
    state: TransferTransactionState
    created_at: float
    updated_at: float
    source_lease_id: str | None = None
    reservation_id: str | None = None
    destination_blocks: tuple[int, ...] = ()
    cancel_requested: bool = False
    error: str | None = None
    result: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def terminal(self) -> bool:
        """判断事务是否已经进入终态。

        返回:
            事务处于完成、回滚、取消或失败状态时返回 True。
        """

        return self.state in _TERMINAL_STATES

    def to_dict(self, *, include_result: bool = True) -> dict[str, Any]:
        """返回可序列化事务快照。

        参数:
            include_result: 是否在终态快照中包含操作结果。

        返回:
            包含事务状态、资源 ID、耗时和可选结果的字典。
        """

        now = time.monotonic()
        snapshot: dict[str, Any] = {
            "transfer_id": self.transfer_id,
            "state": self.state.value,
            "terminal": self.terminal,
            "source_blocks": list(self.source_blocks),
            "destination_blocks": list(self.destination_blocks),
            "source_lease_id": self.source_lease_id,
            "reservation_id": self.reservation_id,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "age_seconds": max(0.0, now - self.created_at),
            "idle_seconds": max(0.0, now - self.updated_at),
        }
        if include_result and self.result is not None:
            snapshot["result"] = dict(self.result)
        return snapshot


class TransferTransactionManager:
    """管理跨 Worker KV block 传输的幂等状态和并发容量。

    参数:
        max_inflight: 允许同时处于非终态的最大事务数。
        completed_retention_seconds: 终态事务保留时间，便于 HTTP 超时后查询。
        max_completed_records: 最多保留的终态事务数量。
    """

    def __init__(
        self,
        *,
        max_inflight: int = 1,
        completed_retention_seconds: float = 300.0,
        max_completed_records: int = 10_000,
    ) -> None:
        """初始化传输事务管理器。

        参数:
            max_inflight: 允许同时处于非终态的最大事务数。
            completed_retention_seconds: 终态事务保留时间。
            max_completed_records: 最多保留的终态事务数量。

        返回:
            无返回值。
        """

        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        self.max_inflight = int(max_inflight)
        self.completed_retention_seconds = float(completed_retention_seconds)
        self.max_completed_records = int(max_completed_records)
        self.lock = threading.RLock()
        self.transactions: dict[str, TransferTransaction] = {}

    def begin(
        self,
        *,
        transfer_id: str | None,
        request_fingerprint: str,
        source_blocks: list[int] | tuple[int, ...],
    ) -> tuple[TransferTransaction, bool]:
        """创建事务，或返回同一幂等请求的已有事务。

        参数:
            transfer_id: 可选调用方幂等 ID；为空时自动生成。
            request_fingerprint: 请求语义指纹。
            source_blocks: 远端源物理 block ID。

        返回:
            二元组 ``(transaction, created)``；created=False 表示幂等重放。

        异常:
            TransferTransactionError: 同一 transfer_id 对应了不同请求。
            TransferCapacityError: 当前非终态事务数已达到并发上限。
        """

        normalized_id = str(transfer_id or uuid.uuid4().hex).strip()
        if not normalized_id:
            raise TransferTransactionError("transfer_id must not be empty")
        if len(normalized_id) > 128:
            raise TransferTransactionError("transfer_id must not exceed 128 characters")

        with self.lock:
            now = time.monotonic()
            self._prune_completed_locked(now)
            existing = self.transactions.get(normalized_id)
            if existing is not None:
                if existing.request_fingerprint != request_fingerprint:
                    raise TransferTransactionError(
                        f"transfer_id {normalized_id} is already bound to another request"
                    )
                return existing, False

            inflight = sum(
                1 for transaction in self.transactions.values() if not transaction.terminal
            )
            if inflight >= self.max_inflight:
                raise TransferCapacityError(
                    "cross-worker transfer capacity exhausted: "
                    f"inflight={inflight}, max_inflight={self.max_inflight}"
                )
            transaction = TransferTransaction(
                transfer_id=normalized_id,
                request_fingerprint=request_fingerprint,
                source_blocks=tuple(int(value) for value in source_blocks),
                state=TransferTransactionState.CREATED,
                created_at=now,
                updated_at=now,
            )
            self.transactions[normalized_id] = transaction
            return transaction, True

    def update(
        self,
        transfer_id: str,
        state: TransferTransactionState,
        *,
        source_lease_id: str | None = None,
        reservation_id: str | None = None,
        destination_blocks: list[int] | tuple[int, ...] | None = None,
    ) -> TransferTransaction:
        """更新事务阶段及其持有的资源。

        参数:
            transfer_id: 待更新事务 ID。
            state: 新事务状态。
            source_lease_id: 可选源 block lease ID。
            reservation_id: 可选目标 reservation ID。
            destination_blocks: 可选目标物理 block ID。

        返回:
            更新后的事务对象。
        """

        with self.lock:
            transaction = self._require(transfer_id)
            if transaction.terminal:
                raise TransferTransactionError(
                    f"transfer {transfer_id} is already {transaction.state.value}"
                )
            if (
                state == TransferTransactionState.PUBLISHING
                and transaction.cancel_requested
            ):
                raise TransferCancelledError(f"transfer {transfer_id} was cancelled")
            transaction.state = state
            transaction.updated_at = time.monotonic()
            if source_lease_id is not None:
                transaction.source_lease_id = str(source_lease_id)
            if reservation_id is not None:
                transaction.reservation_id = str(reservation_id)
            if destination_blocks is not None:
                transaction.destination_blocks = tuple(
                    int(value) for value in destination_blocks
                )
            return transaction

    def request_cancel(self, transfer_id: str) -> tuple[TransferTransaction, bool]:
        """请求协作式取消事务。

        参数:
            transfer_id: 待取消事务 ID。

        返回:
            二元组 ``(transaction, accepted)``。进入 publish 后不能安全取消，
            此时 accepted=False；终态事务也会幂等返回 False。
        """

        with self.lock:
            transaction = self._require(transfer_id)
            if transaction.state not in _CANCELLABLE_STATES:
                return transaction, False
            transaction.cancel_requested = True
            transaction.updated_at = time.monotonic()
            return transaction, True

    def raise_if_cancelled(self, transfer_id: str) -> None:
        """在安全检查点发现取消请求时抛出异常。

        参数:
            transfer_id: 待检查事务 ID。

        返回:
            未取消时无返回值。

        异常:
            TransferCancelledError: 事务已经收到取消请求。
        """

        with self.lock:
            transaction = self._require(transfer_id)
            if transaction.cancel_requested:
                raise TransferCancelledError(f"transfer {transfer_id} was cancelled")

    def finish(
        self,
        transfer_id: str,
        state: TransferTransactionState,
        *,
        result: dict[str, Any],
        error: str | None = None,
    ) -> TransferTransaction:
        """将事务设置为终态并保存可查询结果。

        参数:
            transfer_id: 待结束事务 ID。
            state: 完成、回滚、取消或失败终态。
            result: HTTP 超时后仍可查询的操作结果。
            error: 可选错误信息。

        返回:
            已进入终态的事务对象。
        """

        if state not in _TERMINAL_STATES:
            raise TransferTransactionError(f"{state.value} is not a terminal state")
        with self.lock:
            transaction = self._require(transfer_id)
            if transaction.terminal:
                if transaction.state == state and transaction.result == result and transaction.error == error:
                    return transaction
                raise TransferTransactionError(f"cannot overwrite terminal transfer {transfer_id}")
            transaction.state = state
            transaction.error = error
            transaction.result = dict(result)
            transaction.updated_at = time.monotonic()
            self._prune_completed_locked(transaction.updated_at)
            return transaction

    def get(self, transfer_id: str) -> TransferTransaction:
        """读取事务。

        参数:
            transfer_id: 待读取事务 ID。

        返回:
            对应事务对象。
        """

        with self.lock:
            self._prune_completed_locked(time.monotonic())
            return self._require(transfer_id)

    def stats(self) -> dict[str, Any]:
        """返回事务状态和并发容量统计。

        返回:
            包含各状态数量、当前并发数和上限的字典。
        """

        with self.lock:
            self._prune_completed_locked(time.monotonic())
            states = {state.value: 0 for state in TransferTransactionState}
            for transaction in self.transactions.values():
                states[transaction.state.value] += 1
            inflight = sum(
                count
                for state, count in states.items()
                if TransferTransactionState(state) not in _TERMINAL_STATES
            )
            return {
                "max_inflight": self.max_inflight,
                "inflight": inflight,
                "states": states,
                "retained": len(self.transactions),
            }

    def _require(self, transfer_id: str) -> TransferTransaction:
        """返回指定事务。

        参数:
            transfer_id: 待读取事务 ID。

        返回:
            对应事务对象。

        异常:
            TransferTransactionError: transfer_id 不存在。
        """

        transaction = self.transactions.get(str(transfer_id))
        if transaction is None:
            raise TransferTransactionError(f"unknown transfer_id: {transfer_id}")
        return transaction

    def _prune_completed_locked(self, now: float) -> None:
        """在持锁状态下清理过期或超量的终态事务记录。

        参数:
            now: 当前单调时钟时间戳。

        返回:
            无返回值。
        """

        completed = [
            transaction
            for transaction in self.transactions.values()
            if transaction.terminal
        ]
        for transaction in completed:
            if now - transaction.updated_at >= self.completed_retention_seconds:
                self.transactions.pop(transaction.transfer_id, None)

        completed = sorted(
            (
                transaction
                for transaction in self.transactions.values()
                if transaction.terminal
            ),
            key=lambda transaction: transaction.updated_at,
        )
        overflow = len(completed) - self.max_completed_records
        for transaction in completed[: max(0, overflow)]:
            self.transactions.pop(transaction.transfer_id, None)
