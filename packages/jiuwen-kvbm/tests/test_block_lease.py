# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
from __future__ import annotations

import time

import pytest

from jiuwen_kvbm.integrations import vllm_ascend_018 as integration
from jiuwen_kvbm.runtime import (
    SourceBlockLeaseError,
    SourceBlockLeaseState,
    VllmSourceBlockLeaseManager,
)


class FakeSourceBlock:
    """测试用源端 vLLM block。"""

    def __init__(
        self,
        block_id: int,
        block_hash: bytes | None,
        *,
        ref_cnt: int = 0,
    ) -> None:
        self.block_id = block_id
        self.block_hash = block_hash
        self.ref_cnt = ref_cnt


class FakeSourceBlockPool:
    """实现 touch/free_blocks 引用语义的测试 BlockPool。"""

    def __init__(self, blocks: list[FakeSourceBlock]) -> None:
        self.blocks = blocks
        self.free = [block for block in blocks if block.ref_cnt == 0]

    def touch(self, blocks: list[FakeSourceBlock]) -> None:
        """增加 block 引用，并从 free queue 移除零引用 block。"""

        for block in blocks:
            if block.ref_cnt == 0:
                self.free.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, blocks: list[FakeSourceBlock]) -> None:
        """减少 block 引用，并把零引用 block 放回 free queue。"""

        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self.free.append(block)


def test_source_lease_pins_cached_blocks_and_releases_idempotently() -> None:
    """lease 应从 free queue pin 源 block，并允许重复 release。"""

    block = FakeSourceBlock(0, b"a" * 36)
    pool = FakeSourceBlockPool([block])
    manager = VllmSourceBlockLeaseManager(lambda: pool)

    lease = manager.acquire([0], ttl_ms=5_000)

    assert lease.state == SourceBlockLeaseState.ACTIVE
    assert lease.block_hashes == (b"a" * 36,)
    assert block.ref_cnt == 1
    assert pool.free == []

    released = manager.release(lease.lease_id)
    released_again = manager.release(lease.lease_id)

    assert released.state == SourceBlockLeaseState.RELEASED
    assert released_again.state == SourceBlockLeaseState.RELEASED
    assert block.ref_cnt == 0
    assert pool.free == [block]


def test_failed_lease_release_is_not_reported_as_success(monkeypatch):
    block = FakeSourceBlock(0, b"a" * 36)
    pool = FakeSourceBlockPool([block])
    manager = VllmSourceBlockLeaseManager(lambda: pool)
    lease = manager.acquire([0], ttl_ms=5000)
    calls = []

    def fail_free(blocks):
        calls.append(list(blocks))
        raise RuntimeError("injected free failure")

    monkeypatch.setattr(pool, "free_blocks", fail_free)
    with pytest.raises(SourceBlockLeaseError, match="release failed"):
        manager.release(lease.lease_id)
    assert lease.state == SourceBlockLeaseState.RELEASE_FAILED
    assert lease.finished_at is None
    assert block.ref_cnt == 1
    with pytest.raises(SourceBlockLeaseError, match="pool recovery"):
        manager.release(lease.lease_id)
    assert len(calls) == 1


def test_failed_expiry_does_not_block_other_leases(monkeypatch):
    blocks = [FakeSourceBlock(i, b"a" * 36) for i in range(2)]
    pool = FakeSourceBlockPool(blocks)
    manager = VllmSourceBlockLeaseManager(lambda: pool)
    first = manager.acquire([0], ttl_ms=5000)
    second = manager.acquire([1], ttl_ms=5000)
    original = pool.free_blocks

    def free_selected(items):
        items = list(items)
        if items[0].block_id == 0:
            raise RuntimeError("injected expiry failure")
        original(items)

    monkeypatch.setattr(pool, "free_blocks", free_selected)
    first.expires_at = second.expires_at = 0
    assert manager.expire_due() == 1
    assert first.state == SourceBlockLeaseState.RELEASE_FAILED
    assert second.state == SourceBlockLeaseState.EXPIRED
    assert blocks[1].ref_cnt == 0


def test_source_lease_preserves_existing_request_reference() -> None:
    """源 block 已被请求引用时，release 只能移除 lease 自己增加的引用。"""

    block = FakeSourceBlock(0, b"a" * 36, ref_cnt=1)
    pool = FakeSourceBlockPool([block])
    manager = VllmSourceBlockLeaseManager(lambda: pool)

    lease = manager.acquire([0], ttl_ms=5_000)
    manager.release(lease.lease_id)

    assert block.ref_cnt == 1
    assert pool.free == []


def test_source_lease_expiration_releases_abandoned_reference() -> None:
    """调用方未 release 时，过期回收应归还源 block 引用。"""

    block = FakeSourceBlock(0, b"a" * 36)
    pool = FakeSourceBlockPool([block])
    manager = VllmSourceBlockLeaseManager(
        lambda: pool,
        min_ttl_ms=1,
        max_ttl_ms=10_000,
    )
    lease = manager.acquire([0], ttl_ms=1)

    time.sleep(0.01)
    expired_count = manager.expire_due()

    assert expired_count == 1
    assert lease.state == SourceBlockLeaseState.EXPIRED
    assert block.ref_cnt == 0
    assert pool.free == [block]


def test_source_lease_rejects_unpublished_and_duplicate_blocks() -> None:
    """lease 应拒绝未发布 hash 和重复物理 block ID。"""

    block = FakeSourceBlock(0, None)
    pool = FakeSourceBlockPool([block])
    manager = VllmSourceBlockLeaseManager(lambda: pool)

    with pytest.raises(SourceBlockLeaseError, match="duplicates"):
        manager.acquire([0, 0], ttl_ms=5_000)
    with pytest.raises(SourceBlockLeaseError, match="not published"):
        manager.acquire([0], ttl_ms=5_000)

    assert block.ref_cnt == 0
    assert pool.free == [block]


def test_source_lease_can_pin_request_scoped_unpublished_block() -> None:
    """PD 请求尾部 partial block 应可在释放前以请求级 lease 保护。"""

    block = FakeSourceBlock(0, None, ref_cnt=1)
    pool = FakeSourceBlockPool([block])
    manager = VllmSourceBlockLeaseManager(lambda: pool)

    lease = manager.acquire(
        [0],
        ttl_ms=5_000,
        require_published=False,
    )

    assert lease.block_hashes == (None,)
    assert block.ref_cnt == 2
    manager.release(lease.lease_id)
    assert block.ref_cnt == 1


def test_source_lease_control_api_returns_atomic_hash_and_releases(
    monkeypatch,
) -> None:
    """scheduler 控制 API 应返回 pin 时读取的完整 hash 并支持 release。"""

    packed_hash = b"a" * 32 + (3).to_bytes(4, "big")
    block = FakeSourceBlock(0, packed_hash)
    pool = FakeSourceBlockPool([block])
    manager = VllmSourceBlockLeaseManager(lambda: pool)
    monkeypatch.setattr(integration, "_PROCESS_SOURCE_BLOCK_LEASES", manager)

    acquired = integration._handle_local_source_lease_request(
        "/datadist/source-leases/acquire",
        {"block_ids": [0], "ttl_ms": 5_000},
    )
    lease = acquired["lease"]
    released = integration._handle_local_source_lease_request(
        "/datadist/source-leases/release",
        {"lease_id": lease["lease_id"]},
    )

    assert acquired["ok"] is True
    assert lease["block_hashes"][0]["packed_hex"] == packed_hash.hex()
    assert lease["block_hashes"][0]["group_id"] == 3
    assert released["lease"]["state"] == "released"
    assert block.ref_cnt == 0


def test_source_lease_control_api_renews_request_scoped_partial_block(
    monkeypatch,
) -> None:
    """partial block 续租响应应保留空 hash，不能按完整 packed hash 编码。"""

    block = FakeSourceBlock(0, None, ref_cnt=1)
    pool = FakeSourceBlockPool([block])
    manager = VllmSourceBlockLeaseManager(lambda: pool)
    lease = manager.acquire(
        [0],
        ttl_ms=5_000,
        require_published=False,
    )
    monkeypatch.setattr(integration, "_PROCESS_SOURCE_BLOCK_LEASES", manager)

    renewed = integration._handle_local_source_lease_request(
        "/datadist/source-leases/renew",
        {"lease_id": lease.lease_id, "ttl_ms": 10_000},
    )
    released = integration._handle_local_source_lease_request(
        "/datadist/source-leases/release",
        {"lease_id": lease.lease_id},
    )

    assert renewed["ok"] is True
    assert renewed["lease"]["block_hashes"] == [None]
    assert released["lease"]["block_hashes"] == [None]
    assert released["lease"]["state"] == "released"
    assert block.ref_cnt == 1
