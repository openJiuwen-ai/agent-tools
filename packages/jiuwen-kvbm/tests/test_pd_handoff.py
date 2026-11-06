from __future__ import annotations

import pytest

from jiuwen_kvbm.runtime import (
    PDHandoffCapacityError,
    PDHandoffError,
    PDHandoffManager,
    PDHandoffState,
)


def _begin(manager: PDHandoffManager, handoff_id: str = "handoff-1"):
    """创建测试 handoff 并返回记录。

    参数:
        manager: 待测试的 PD handoff 管理器。
        handoff_id: 本次测试使用的幂等 ID。

    返回:
        新建或幂等返回的 PDHandoff。
    """

    handoff, _created = manager.begin(
        handoff_id=handoff_id,
        request_id="request-1",
        request_fingerprint="fingerprint-1",
        source_worker_id="prefill-0",
        target_worker_id="decode-0",
        model_id="model-a",
        prompt_token_count=1024,
    )
    return handoff


def test_pd_handoff_runs_complete_state_sequence() -> None:
    """PD handoff 应按 P 就绪、传输、D 接管和源释放顺序进入成功终态。"""

    manager = PDHandoffManager(max_inflight=2)
    handoff = _begin(manager)
    transitions = (
        (PDHandoffState.PREFILL_READY, {"block_hashes": ["h0"], "source_blocks": [1]}),
        (PDHandoffState.SOURCE_PINNED, {"source_lease_id": "lease-1"}),
        (PDHandoffState.TARGET_RESERVED, {"reservation_id": "reservation-1"}),
        (
            PDHandoffState.TRANSFERRING,
            {"transfer_id": "transfer-1", "destination_blocks": [11]},
        ),
        (PDHandoffState.TARGET_ATTACHED, {}),
        (PDHandoffState.DECODE_ACCEPTED, {}),
        (PDHandoffState.SOURCE_RELEASED, {}),
    )

    for state, resources in transitions:
        handoff = manager.advance(handoff.handoff_id, state, **resources)

    snapshot = handoff.to_dict()
    assert snapshot["state"] == "source_released"
    assert snapshot["terminal"] is True
    assert snapshot["source_blocks"] == [1]
    assert snapshot["destination_blocks"] == [11]
    assert [item["state"] for item in snapshot["history"]] == [
        "prefill_running",
        "prefill_ready",
        "source_pinned",
        "target_reserved",
        "transferring",
        "target_attached",
        "decode_accepted",
        "source_released",
    ]


@pytest.mark.parametrize("owns_resources", [False, True])
def test_expired_rollback_never_hides_unreleased_resources(owns_resources):
    manager = PDHandoffManager(max_inflight=1)
    handoff = _begin(manager)
    manager.request_rollback(handoff.handoff_id, reason="test")
    handoff.deadline_epoch_ms = 1
    if owns_resources:
        handoff.source_lease_id = "pending-cleanup"
    stats = manager.stats()
    if owns_resources:
        assert handoff.state == PDHandoffState.ROLLING_BACK
        assert stats["inflight"] == 1
        assert stats["cleanup_required"] == [handoff.handoff_id]
    else:
        assert handoff.state == PDHandoffState.FAILED
        assert stats["inflight"] == 0


def test_pd_handoff_is_idempotent_and_rejects_invalid_transition() -> None:
    """相同语义可幂等重放，ID 冲突和越级迁移必须被拒绝。"""

    manager = PDHandoffManager(max_inflight=2)
    first = _begin(manager)
    replay, created = manager.begin(
        handoff_id="handoff-1",
        request_id="request-1",
        request_fingerprint="fingerprint-1",
        source_worker_id="prefill-0",
        target_worker_id="decode-0",
        model_id="model-a",
        prompt_token_count=1024,
    )

    assert replay is first
    assert created is False
    with pytest.raises(PDHandoffError, match="another request"):
        manager.begin(
            handoff_id="handoff-1",
            request_id="request-2",
            request_fingerprint="different",
            source_worker_id="prefill-0",
            target_worker_id="decode-0",
            model_id="model-a",
            prompt_token_count=1024,
        )
    with pytest.raises(PDHandoffError, match="invalid PD handoff transition"):
        manager.advance("handoff-1", PDHandoffState.TRANSFERRING)


def test_pd_handoff_checks_expected_state_and_resource_identity() -> None:
    """并发调用方必须使用正确前置状态，已绑定资源不能被另一个值替换。"""

    manager = PDHandoffManager()
    handoff = _begin(manager)
    manager.advance(
        handoff.handoff_id,
        PDHandoffState.PREFILL_READY,
        expected_state=PDHandoffState.PREFILL_RUNNING,
        block_hashes=["h0", "h1"],
        source_blocks=[1, 2],
    )
    replay = manager.advance(
        handoff.handoff_id,
        PDHandoffState.PREFILL_READY,
        expected_state=PDHandoffState.PREFILL_RUNNING,
        block_hashes=["h0", "h1"],
        source_blocks=[1, 2],
    )

    assert replay.state == PDHandoffState.PREFILL_READY
    with pytest.raises(PDHandoffError, match="expected prefill_running"):
        manager.advance(
            handoff.handoff_id,
            PDHandoffState.SOURCE_PINNED,
            expected_state=PDHandoffState.PREFILL_RUNNING,
        )
    with pytest.raises(PDHandoffError, match="different source_blocks"):
        manager.advance(
            handoff.handoff_id,
            PDHandoffState.PREFILL_READY,
            source_blocks=[3, 4],
        )


def test_pd_handoff_rollback_is_cooperative_and_idempotent() -> None:
    """回滚请求先进入 ROLLING_BACK，资源清理完成后才能进入 ROLLED_BACK。"""

    manager = PDHandoffManager()
    handoff = _begin(manager)

    rolling_back, accepted = manager.request_rollback(
        handoff.handoff_id,
        reason="decode worker unavailable",
    )
    duplicate, duplicate_accepted = manager.request_rollback(
        handoff.handoff_id,
        reason="duplicate",
    )
    rolled_back = manager.advance(
        handoff.handoff_id,
        PDHandoffState.ROLLED_BACK,
        expected_state=PDHandoffState.ROLLING_BACK,
    )

    assert accepted is True
    assert duplicate_accepted is False
    assert duplicate is rolling_back
    assert rolled_back.terminal is True
    assert rolled_back.error == "decode worker unavailable"


def test_pd_handoff_rejects_stage_without_required_resources_atomically() -> None:
    """阶段所需资源缺失时不能迁移，也不能留下部分写入的资源字段。"""

    manager = PDHandoffManager()
    handoff = _begin(manager)

    with pytest.raises(PDHandoffError, match="missing=.*source_blocks"):
        manager.advance(
            handoff.handoff_id,
            PDHandoffState.PREFILL_READY,
            block_hashes=["hash-1"],
        )

    snapshot = manager.get(handoff.handoff_id).to_dict()
    assert snapshot["state"] == "prefill_running"
    assert snapshot["block_hashes"] == []
    assert snapshot["source_blocks"] == []


def test_pd_handoff_capacity_is_released_by_terminal_state() -> None:
    """达到容量时拒绝新 handoff，已有 handoff 失败后恢复容量。"""

    manager = PDHandoffManager(max_inflight=1)
    first = _begin(manager)
    with pytest.raises(PDHandoffCapacityError, match="capacity exhausted"):
        manager.begin(
            handoff_id="handoff-2",
            request_id="request-2",
            request_fingerprint="fingerprint-2",
            source_worker_id="prefill-0",
            target_worker_id="decode-0",
            model_id="model-a",
            prompt_token_count=1024,
        )

    manager.advance(first.handoff_id, PDHandoffState.FAILED, error="prefill failed")
    second, created = manager.begin(
        handoff_id="handoff-2",
        request_id="request-2",
        request_fingerprint="fingerprint-2",
        source_worker_id="prefill-0",
        target_worker_id="decode-0",
        model_id="model-a",
        prompt_token_count=1024,
    )

    assert created is True
    assert second.state == PDHandoffState.PREFILL_RUNNING
    assert manager.stats()["inflight"] == 1
