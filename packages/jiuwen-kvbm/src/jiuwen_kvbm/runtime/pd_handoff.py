from __future__ import annotations

import threading
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PDHandoffError(RuntimeError):
    """Prefill/Decode 请求交接协议错误。"""


class PDHandoffCapacityError(PDHandoffError):
    """Prefill/Decode 请求交接并发容量已耗尽。"""


class PDHandoffState(str, Enum):
    """一次 Prefill/Decode 请求交接的生命周期状态。"""

    PREFILL_RUNNING = "prefill_running"
    PREFILL_READY = "prefill_ready"
    SOURCE_PINNED = "source_pinned"
    TARGET_RESERVED = "target_reserved"
    TRANSFERRING = "transferring"
    TARGET_ATTACHED = "target_attached"
    DECODE_ACCEPTED = "decode_accepted"
    SOURCE_RELEASED = "source_released"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


_TERMINAL_STATES = {
    PDHandoffState.SOURCE_RELEASED,
    PDHandoffState.ROLLED_BACK,
    PDHandoffState.FAILED,
}

_ALLOWED_TRANSITIONS = {
    PDHandoffState.PREFILL_RUNNING: {
        PDHandoffState.PREFILL_READY,
        PDHandoffState.ROLLING_BACK,
        PDHandoffState.FAILED,
    },
    PDHandoffState.PREFILL_READY: {
        PDHandoffState.SOURCE_PINNED,
        PDHandoffState.ROLLING_BACK,
        PDHandoffState.FAILED,
    },
    PDHandoffState.SOURCE_PINNED: {
        PDHandoffState.TARGET_RESERVED,
        PDHandoffState.ROLLING_BACK,
        PDHandoffState.FAILED,
    },
    PDHandoffState.TARGET_RESERVED: {
        PDHandoffState.TRANSFERRING,
        PDHandoffState.ROLLING_BACK,
        PDHandoffState.FAILED,
    },
    PDHandoffState.TRANSFERRING: {
        PDHandoffState.TARGET_ATTACHED,
        PDHandoffState.ROLLING_BACK,
        PDHandoffState.FAILED,
    },
    PDHandoffState.TARGET_ATTACHED: {
        PDHandoffState.DECODE_ACCEPTED,
        PDHandoffState.ROLLING_BACK,
        PDHandoffState.FAILED,
    },
    PDHandoffState.DECODE_ACCEPTED: {
        PDHandoffState.SOURCE_RELEASED,
        PDHandoffState.ROLLING_BACK,
        PDHandoffState.FAILED,
    },
    PDHandoffState.ROLLING_BACK: {
        PDHandoffState.ROLLED_BACK,
        PDHandoffState.FAILED,
    },
}


@dataclass
class PDHandoff:
    """一次跨 Prefill/Decode Worker 的请求级 KV 交接记录。

    属性:
        handoff_id: 调用方提供或 KVBM 生成的幂等交接 ID。
        request_id: 推理请求 ID，用于串联 Router、P Worker 和 D Worker。
        request_fingerprint: 排除超时参数后的请求语义指纹。
        source_worker_id: 执行 prefill 并持有源 KV 的 Worker ID。
        target_worker_id: 接管 decode 请求的 Worker ID。
        model_id: 两端必须一致的模型标识。
        prompt_token_count: prefill 完成时的 prompt token 数。
        state: 当前交接状态。
        created_at: 创建记录时的单调时钟时间戳。
        updated_at: 最近一次状态变化的单调时钟时间戳。
        deadline_epoch_ms: 可选的交接绝对截止时间。
        block_hashes: 按前缀顺序排列的逻辑 KV block hash。
        source_blocks: P Worker 的源物理 block ID。
        destination_blocks: D Worker 的目标物理 block ID。
        source_lease_id: P Worker 的源 block lease ID。
        reservation_id: D Worker 的目标 BlockPool reservation ID。
        transfer_id: 复用现有 HIXL pull 事务的 transfer ID。
        error: 失败或回滚原因。
        metadata: 协议扩展字段，不参与核心状态判断。
        history: 状态迁移历史，用于跨进程联调诊断。
    """

    handoff_id: str
    request_id: str
    request_fingerprint: str
    source_worker_id: str
    target_worker_id: str
    model_id: str
    prompt_token_count: int
    state: PDHandoffState
    created_at: float
    updated_at: float
    deadline_epoch_ms: int | None = None
    block_hashes: tuple[str, ...] = ()
    source_blocks: tuple[int, ...] = ()
    destination_blocks: tuple[int, ...] = ()
    source_lease_id: str | None = None
    reservation_id: str | None = None
    transfer_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def terminal(self) -> bool:
        """判断交接是否已经进入成功、回滚或失败终态。

        返回:
            处于 SOURCE_RELEASED、ROLLED_BACK 或 FAILED 时返回 True。
        """

        return self.state in _TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        """返回可通过 HTTP 返回的交接快照。

        返回:
            包含状态、资源、耗时、扩展字段和迁移历史的字典。
        """

        now = time.monotonic()
        return {
            "handoff_id": self.handoff_id,
            "request_id": self.request_id,
            "source_worker_id": self.source_worker_id,
            "target_worker_id": self.target_worker_id,
            "model_id": self.model_id,
            "prompt_token_count": self.prompt_token_count,
            "state": self.state.value,
            "terminal": self.terminal,
            "deadline_epoch_ms": self.deadline_epoch_ms,
            "block_hashes": list(self.block_hashes),
            "source_blocks": list(self.source_blocks),
            "destination_blocks": list(self.destination_blocks),
            "source_lease_id": self.source_lease_id,
            "reservation_id": self.reservation_id,
            "transfer_id": self.transfer_id,
            "error": self.error,
            "metadata": dict(self.metadata),
            "history": [dict(item) for item in self.history],
            "age_seconds": max(0.0, now - self.created_at),
            "idle_seconds": max(0.0, now - self.updated_at),
        }


class PDHandoffManager:
    """管理请求级 Prefill/Decode 交接状态、幂等和并发容量。"""

    def __init__(
        self,
        *,
        max_inflight: int = 1024,
        completed_retention_seconds: float = 300.0,
        max_completed_records: int = 10_000,
    ) -> None:
        """初始化交接管理器。

        参数:
            max_inflight: 允许同时处于非终态的最大 handoff 数。
            completed_retention_seconds: 终态记录的保留时间，单位秒。
            max_completed_records: 最多保留的终态记录数量。

        返回:
            无返回值。
        """

        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        self.max_inflight = int(max_inflight)
        self.completed_retention_seconds = float(completed_retention_seconds)
        self.max_completed_records = int(max_completed_records)
        self.lock = threading.RLock()
        self.handoffs: dict[str, PDHandoff] = {}

    def begin(
        self,
        *,
        handoff_id: str | None,
        request_id: str,
        request_fingerprint: str,
        source_worker_id: str,
        target_worker_id: str,
        model_id: str,
        prompt_token_count: int,
        deadline_epoch_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[PDHandoff, bool]:
        """创建 PREFILL_RUNNING 记录，或返回同一语义请求的已有记录。

        参数:
            handoff_id: 可选幂等 ID；为空时自动生成。
            request_id: 推理请求 ID，不能为空。
            request_fingerprint: 请求语义指纹，用于拒绝 ID 冲突。
            source_worker_id: Prefill Worker ID。
            target_worker_id: Decode Worker ID。
            model_id: 两端共同使用的模型 ID。
            prompt_token_count: 当前 prompt token 数。
            deadline_epoch_ms: 可选的绝对截止时间，单位毫秒。
            metadata: 可选协议扩展字段。

        返回:
            ``(handoff, created)``；created=False 表示幂等重放。
        """

        normalized_id = str(handoff_id or uuid.uuid4().hex).strip()
        normalized_request_id = str(request_id).strip()
        normalized_fingerprint = str(request_fingerprint).strip()
        if not normalized_id or len(normalized_id) > 128:
            raise PDHandoffError("handoff_id must contain 1 to 128 characters")
        if not normalized_request_id:
            raise PDHandoffError("request_id must not be empty")
        if not normalized_fingerprint:
            raise PDHandoffError("request_fingerprint must not be empty")
        normalized_source_worker_id = str(source_worker_id).strip()
        normalized_target_worker_id = str(target_worker_id).strip()
        normalized_model_id = str(model_id).strip()
        if not normalized_source_worker_id or not normalized_target_worker_id:
            raise PDHandoffError("source_worker_id and target_worker_id are required")
        if normalized_source_worker_id == normalized_target_worker_id:
            raise PDHandoffError("source_worker_id and target_worker_id must differ")
        if not normalized_model_id:
            raise PDHandoffError("model_id must not be empty")
        if int(prompt_token_count) < 0:
            raise PDHandoffError("prompt_token_count must not be negative")

        with self.lock:
            now = time.monotonic()
            self._check_rollback_deadlines_locked()
            self._prune_completed_locked(now)
            existing = self.handoffs.get(normalized_id)
            if existing is not None:
                identity = (
                    normalized_request_id,
                    normalized_source_worker_id,
                    normalized_target_worker_id,
                    normalized_model_id,
                    int(prompt_token_count),
                )
                existing_identity = (
                    existing.request_id,
                    existing.source_worker_id,
                    existing.target_worker_id,
                    existing.model_id,
                    existing.prompt_token_count,
                )
                if existing.request_fingerprint != normalized_fingerprint or existing_identity != identity:
                    raise PDHandoffError(f"handoff_id {normalized_id} is already bound to another request")
                return existing, False
            inflight = sum(not handoff.terminal for handoff in self.handoffs.values())
            if inflight >= self.max_inflight:
                raise PDHandoffCapacityError(
                    f"PD handoff capacity exhausted: inflight={inflight}, max_inflight={self.max_inflight}"
                )
            handoff = PDHandoff(
                handoff_id=normalized_id,
                request_id=normalized_request_id,
                request_fingerprint=normalized_fingerprint,
                source_worker_id=normalized_source_worker_id,
                target_worker_id=normalized_target_worker_id,
                model_id=normalized_model_id,
                prompt_token_count=int(prompt_token_count),
                state=PDHandoffState.PREFILL_RUNNING,
                created_at=now,
                updated_at=now,
                deadline_epoch_ms=(int(deadline_epoch_ms) if deadline_epoch_ms is not None else None),
                metadata=dict(metadata or {}),
                history=[self._history_item(PDHandoffState.PREFILL_RUNNING, now)],
            )
            self.handoffs[normalized_id] = handoff
            return handoff, True

    def advance(
        self,
        handoff_id: str,
        state: PDHandoffState,
        *,
        expected_state: PDHandoffState | None = None,
        block_hashes: list[str] | tuple[str, ...] | None = None,
        source_blocks: list[int] | tuple[int, ...] | None = None,
        destination_blocks: list[int] | tuple[int, ...] | None = None,
        source_lease_id: str | None = None,
        reservation_id: str | None = None,
        transfer_id: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PDHandoff:
        """校验状态迁移并原子更新 handoff 资源快照。

        参数:
            handoff_id: 待更新的 handoff ID。
            state: 目标状态。
            expected_state: 可选乐观并发前置状态。
            block_hashes: 可选逻辑 block hash 列表。
            source_blocks: 可选 P Worker 物理 block ID。
            destination_blocks: 可选 D Worker 物理 block ID。
            source_lease_id: 可选源 block lease ID。
            reservation_id: 可选目标 reservation ID。
            transfer_id: 可选 HIXL pull 事务 ID。
            error: 可选失败或回滚原因。
            metadata: 需要合并的协议扩展字段。

        返回:
            更新后的 handoff；重复提交当前状态时幂等返回。
        """

        with self.lock:
            handoff = self._require(handoff_id)
            if (
                expected_state is not None
                and handoff.state != expected_state
                and handoff.state != state
            ):
                raise PDHandoffError(
                    f"handoff {handoff_id} state is {handoff.state.value}, "
                    f"expected {expected_state.value}"
                )
            if handoff.state != state:
                allowed = _ALLOWED_TRANSITIONS.get(handoff.state, set())
                if state not in allowed:
                    raise PDHandoffError(
                        f"invalid PD handoff transition: {handoff.state.value} -> "
                        f"{state.value}"
                    )
            previous_resources = (
                handoff.block_hashes,
                handoff.source_blocks,
                handoff.destination_blocks,
                handoff.source_lease_id,
                handoff.reservation_id,
                handoff.transfer_id,
                handoff.error,
                dict(handoff.metadata),
            )
            try:
                self._merge_resources(
                    handoff,
                    block_hashes=block_hashes,
                    source_blocks=source_blocks,
                    destination_blocks=destination_blocks,
                    source_lease_id=source_lease_id,
                    reservation_id=reservation_id,
                    transfer_id=transfer_id,
                )
                if metadata:
                    handoff.metadata.update(metadata)
                if error is not None:
                    handoff.error = str(error)
                self._validate_state_requirements(handoff, state)
            except Exception:
                (
                    handoff.block_hashes,
                    handoff.source_blocks,
                    handoff.destination_blocks,
                    handoff.source_lease_id,
                    handoff.reservation_id,
                    handoff.transfer_id,
                    handoff.error,
                    handoff.metadata,
                ) = previous_resources
                raise
            if handoff.state != state:
                handoff.state = state
                handoff.updated_at = time.monotonic()
                handoff.history.append(self._history_item(state, handoff.updated_at))
            self._prune_completed_locked(time.monotonic())
            return handoff

    def request_rollback(self, handoff_id: str, *, reason: str) -> tuple[PDHandoff, bool]:
        """请求资源清理并把非终态 handoff 迁移到 ROLLING_BACK。

        参数:
            handoff_id: 待回滚的 handoff ID。
            reason: 触发回滚的错误或取消原因。

        返回:
            ``(handoff, accepted)``；终态或已回滚时 accepted=False。
        """

        with self.lock:
            handoff = self._require(handoff_id)
            if handoff.terminal or handoff.state == PDHandoffState.ROLLING_BACK:
                return handoff, False
            if handoff.deadline_epoch_ms is None:
                handoff.deadline_epoch_ms = int(time.time() * 1000) + 60_000
            return (
                self.advance(
                    handoff_id,
                    PDHandoffState.ROLLING_BACK,
                    error=reason,
                ),
                True,
            )

    def get(self, handoff_id: str) -> PDHandoff:
        """读取指定 handoff。

        参数:
            handoff_id: 待读取的 handoff ID。

        返回:
            对应的请求级交接记录。
        """

        with self.lock:
            self._check_rollback_deadlines_locked()
            self._prune_completed_locked(time.monotonic())
            return self._require(handoff_id)

    def stats(self) -> dict[str, Any]:
        """返回各状态数量、当前并发数和容量。

        返回:
            可序列化的 handoff 管理器统计字典。
        """

        with self.lock:
            self._check_rollback_deadlines_locked()
            self._prune_completed_locked(time.monotonic())
            states = {state.value: 0 for state in PDHandoffState}
            for handoff in self.handoffs.values():
                states[handoff.state.value] += 1
            return {
                "max_inflight": self.max_inflight,
                "inflight": sum(not handoff.terminal for handoff in self.handoffs.values()),
                "states": states,
                "retained": len(self.handoffs),
                "cleanup_required": [
                    h.handoff_id
                    for h in self.handoffs.values()
                    if not h.terminal and h.metadata.get("cleanup_required")
                ],
            }

    def _check_rollback_deadlines_locked(self) -> None:
        now_ms = int(time.time() * 1000)
        for handoff in self.handoffs.values():
            if handoff.state != PDHandoffState.ROLLING_BACK:
                continue
            if handoff.deadline_epoch_ms is None or handoff.deadline_epoch_ms > now_ms:
                continue
            # Never free a capacity slot while physical ownership is unresolved.
            owns_resources = any(
                (
                    handoff.source_lease_id,
                    handoff.reservation_id,
                    handoff.transfer_id,
                    handoff.source_blocks,
                    handoff.destination_blocks,
                )
            )
            if owns_resources:
                if not handoff.metadata.get("cleanup_required"):
                    logger.error("Expired PD rollback requires resource recovery: %s", handoff.handoff_id)
                handoff.metadata["cleanup_required"] = True
                continue
            handoff.state = PDHandoffState.FAILED
            handoff.error = "rollback deadline expired before resource acquisition"
            handoff.updated_at = time.monotonic()
            handoff.history.append(self._history_item(handoff.state, handoff.updated_at))

    @staticmethod
    def _validate_state_requirements(
        handoff: PDHandoff,
        state: PDHandoffState,
    ) -> None:
        """确认资源快照足以进入指定 PD handoff 阶段。

        参数:
            handoff: 已合并本次资源字段的 handoff。
            state: 调用方请求进入的目标状态。

        返回:
            资源完整时无返回值，否则抛出 PDHandoffError。
        """

        requirements = {
            PDHandoffState.PREFILL_READY: (
                (handoff.block_hashes, "block_hashes"),
                (handoff.source_blocks, "source_blocks"),
            ),
            PDHandoffState.SOURCE_PINNED: (
                (handoff.source_lease_id, "source_lease_id"),
            ),
            PDHandoffState.TARGET_RESERVED: (
                (handoff.reservation_id, "reservation_id"),
            ),
            PDHandoffState.TRANSFERRING: (
                (handoff.transfer_id, "transfer_id"),
                (handoff.destination_blocks, "destination_blocks"),
            ),
            PDHandoffState.TARGET_ATTACHED: (
                (handoff.destination_blocks, "destination_blocks"),
            ),
        }
        missing = [name for value, name in requirements.get(state, ()) if not value]
        if missing:
            raise PDHandoffError(
                f"handoff {handoff.handoff_id} cannot enter {state.value}; "
                f"missing={missing}"
            )

    @staticmethod
    def _merge_resources(
        handoff: PDHandoff,
        *,
        block_hashes: list[str] | tuple[str, ...] | None,
        source_blocks: list[int] | tuple[int, ...] | None,
        destination_blocks: list[int] | tuple[int, ...] | None,
        source_lease_id: str | None,
        reservation_id: str | None,
        transfer_id: str | None,
    ) -> None:
        """合并资源字段，并拒绝幂等 ID 下的资源替换。

        参数:
            handoff: 待更新的 handoff。
            block_hashes: 可选逻辑 block hash。
            source_blocks: 可选源物理 block ID。
            destination_blocks: 可选目标物理 block ID。
            source_lease_id: 可选源 lease ID。
            reservation_id: 可选目标 reservation ID。
            transfer_id: 可选 HIXL transfer ID。

        返回:
            无返回值；字段冲突或 block 列表不对齐时抛出 PDHandoffError。
        """

        incoming_hashes = (
            tuple(str(value) for value in block_hashes)
            if block_hashes is not None
            else None
        )
        incoming_source = (
            tuple(int(value) for value in source_blocks)
            if source_blocks is not None
            else None
        )
        incoming_destination = (
            tuple(int(value) for value in destination_blocks)
            if destination_blocks is not None
            else None
        )
        if incoming_hashes is not None and incoming_source is not None:
            if len(incoming_hashes) != len(incoming_source):
                raise PDHandoffError("block_hashes and source_blocks must be aligned")
        if incoming_hashes is not None and incoming_destination is not None:
            if len(incoming_hashes) != len(incoming_destination):
                raise PDHandoffError(
                    "block_hashes and destination_blocks must be aligned"
                )
        for field_name, incoming in (
            ("block_hashes", incoming_hashes),
            ("source_blocks", incoming_source),
            ("destination_blocks", incoming_destination),
            ("source_lease_id", str(source_lease_id) if source_lease_id else None),
            ("reservation_id", str(reservation_id) if reservation_id else None),
            ("transfer_id", str(transfer_id) if transfer_id else None),
        ):
            if incoming is None:
                continue
            current = getattr(handoff, field_name)
            if current not in (None, ()) and current != incoming:
                raise PDHandoffError(
                    f"handoff {handoff.handoff_id} already has different {field_name}"
                )
            setattr(handoff, field_name, incoming)

    @staticmethod
    def _history_item(state: PDHandoffState, timestamp: float) -> dict[str, Any]:
        """构造一条状态迁移历史。

        参数:
            state: 本次进入的状态。
            timestamp: 状态变化的单调时钟时间戳。

        返回:
            包含状态和相对进程时间戳的字典。
        """

        return {"state": state.value, "timestamp": timestamp}

    def _require(self, handoff_id: str) -> PDHandoff:
        """返回指定 handoff，不存在时抛出协议错误。

        参数:
            handoff_id: 待读取的 handoff ID。

        返回:
            对应 handoff。
        """

        handoff = self.handoffs.get(str(handoff_id))
        if handoff is None:
            raise PDHandoffError(f"unknown handoff_id: {handoff_id}")
        return handoff

    def _prune_completed_locked(self, now: float) -> None:
        """清理过期或超过数量上限的终态 handoff。

        参数:
            now: 当前单调时钟时间戳。

        返回:
            无返回值；调用方必须持有 manager lock。
        """

        completed = [handoff for handoff in self.handoffs.values() if handoff.terminal]
        for handoff in completed:
            if now - handoff.updated_at >= self.completed_retention_seconds:
                self.handoffs.pop(handoff.handoff_id, None)
        completed = sorted(
            (handoff for handoff in self.handoffs.values() if handoff.terminal),
            key=lambda handoff: handoff.updated_at,
        )
        overflow = len(completed) - self.max_completed_records
        for handoff in completed[: max(0, overflow)]:
            self.handoffs.pop(handoff.handoff_id, None)
