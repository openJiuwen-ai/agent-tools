from __future__ import annotations

import pytest

from jiuwen_kvbm.runtime import (
    TransferCancelledError,
    TransferCapacityError,
    TransferTransactionError,
    TransferTransactionManager,
    TransferTransactionState,
)


def test_transfer_id_is_idempotent_and_rejects_request_conflict() -> None:
    """相同请求可以重放，但同一 transfer_id 不能绑定不同请求。"""

    manager = TransferTransactionManager(max_inflight=2)
    transaction, created = manager.begin(
        transfer_id="transfer-1",
        request_fingerprint="fingerprint-a",
        source_blocks=[1, 2],
    )
    replay, replay_created = manager.begin(
        transfer_id="transfer-1",
        request_fingerprint="fingerprint-a",
        source_blocks=[1, 2],
    )

    assert created is True
    assert replay_created is False
    assert replay is transaction

    with pytest.raises(TransferTransactionError, match="another request"):
        manager.begin(
            transfer_id="transfer-1",
            request_fingerprint="fingerprint-b",
            source_blocks=[3],
        )


def test_finish_cannot_overwrite_a_different_terminal_state():
    manager = TransferTransactionManager()
    transaction, _ = manager.begin(transfer_id="done", request_fingerprint="x", source_blocks=[1])
    manager.finish(transaction.transfer_id, TransferTransactionState.ROLLED_BACK, result={"ok": False})
    replay = manager.finish(transaction.transfer_id, TransferTransactionState.ROLLED_BACK, result={"ok": False})
    assert replay is transaction
    with pytest.raises(TransferTransactionError, match="terminal"):
        manager.finish(transaction.transfer_id, TransferTransactionState.COMPLETED, result={"ok": True})


def test_transfer_manager_enforces_inflight_capacity_and_releases_terminal_slot() -> None:
    """达到并发上限时拒绝新事务，已有事务进入终态后恢复容量。"""

    manager = TransferTransactionManager(max_inflight=1)
    first, _ = manager.begin(
        transfer_id="transfer-1",
        request_fingerprint="fingerprint-a",
        source_blocks=[1],
    )

    with pytest.raises(TransferCapacityError, match="capacity exhausted"):
        manager.begin(
            transfer_id="transfer-2",
            request_fingerprint="fingerprint-b",
            source_blocks=[2],
        )

    manager.finish(
        first.transfer_id,
        TransferTransactionState.COMPLETED,
        result={"ok": True},
    )
    second, created = manager.begin(
        transfer_id="transfer-2",
        request_fingerprint="fingerprint-b",
        source_blocks=[2],
    )

    assert created is True
    assert second.state == TransferTransactionState.CREATED
    assert manager.stats()["inflight"] == 1


def test_transfer_cancel_is_cooperative_and_rejected_after_publish_starts() -> None:
    """取消标记应在安全检查点生效，进入 publish 后不再接受取消。"""

    manager = TransferTransactionManager(max_inflight=2)
    transaction, _ = manager.begin(
        transfer_id="transfer-cancel",
        request_fingerprint="fingerprint-cancel",
        source_blocks=[1],
    )
    transaction, accepted = manager.request_cancel(transaction.transfer_id)

    assert accepted is True
    assert transaction.cancel_requested is True
    with pytest.raises(TransferCancelledError, match="was cancelled"):
        manager.raise_if_cancelled(transaction.transfer_id)
    with pytest.raises(TransferCancelledError, match="was cancelled"):
        manager.update(
            transaction.transfer_id,
            TransferTransactionState.PUBLISHING,
        )

    publishing, _ = manager.begin(
        transfer_id="transfer-publishing",
        request_fingerprint="fingerprint-publishing",
        source_blocks=[2],
    )
    manager.update(publishing.transfer_id, TransferTransactionState.PUBLISHING)
    publishing, accepted = manager.request_cancel(publishing.transfer_id)

    assert accepted is False
    assert publishing.cancel_requested is False
