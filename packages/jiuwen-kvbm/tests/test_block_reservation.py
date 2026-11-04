from __future__ import annotations

import threading

import pytest

from jiuwen_kvbm.runtime import (
    BlockAlreadyPublishedError,
    BlockHashesInFlightError,
    BlockReservationCapacityError,
    BlockReservationError,
    BlockReservationState,
    VllmBlockPoolReservationManager,
)


class FakeBlock:
    """测试用 vLLM KVCacheBlock。"""

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0
        self._block_hash: bytes | None = None

    @property
    def block_hash(self) -> bytes | None:
        """返回测试 block 当前发布的完整 hash。"""

        return self._block_hash

    @block_hash.setter
    def block_hash(self, value: bytes) -> None:
        """设置一次完整 hash，模拟 vLLM 只允许从空值发布。"""

        assert self._block_hash is None
        self._block_hash = value

    def reset_hash(self) -> None:
        """清除失败 publish 写入的 hash。"""

        self._block_hash = None


class FakeHashMap:
    """测试用 vLLM BlockHashToBlockMap。"""

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.values: dict[bytes, dict[int, FakeBlock]] = {}
        self.insert_count = 0
        self.fail_after = fail_after

    def insert(self, block_hash: bytes, block: FakeBlock) -> None:
        """插入 hash 到物理 block 的映射。"""

        self.insert_count += 1
        if self.fail_after is not None and self.insert_count > self.fail_after:
            raise RuntimeError("injected hash map failure")
        self.values.setdefault(block_hash, {})[block.block_id] = block

    def pop(self, block_hash: bytes, block_id: int) -> FakeBlock | None:
        """删除指定 hash 和物理 block ID 的映射。"""

        blocks = self.values.get(block_hash)
        if not blocks:
            return None
        block = blocks.pop(block_id, None)
        if not blocks:
            self.values.pop(block_hash, None)
        return block


class FakeBlockPool:
    """实现 vLLM get_new_blocks/free_blocks 语义的测试 block pool。"""

    def __init__(
        self,
        count: int,
        *,
        enable_caching: bool = True,
        fail_hash_insert_after: int | None = None,
    ) -> None:
        self.blocks = [FakeBlock(block_id) for block_id in range(count)]
        self.free = list(self.blocks)
        self.enable_caching = enable_caching
        self.cached_block_hash_to_block = FakeHashMap(
            fail_after=fail_hash_insert_after
        )

    def get_num_free_blocks(self) -> int:
        """返回空闲 block 数量。"""

        return len(self.free)

    def get_new_blocks(self, count: int) -> list[FakeBlock]:
        """取出 block 并增加引用计数。"""

        blocks = self.free[:count]
        del self.free[:count]
        for block in blocks:
            block.ref_cnt += 1
        return blocks

    def touch(self, blocks: list[FakeBlock]) -> None:
        """pin 已发布 block，并在零引用时从 free queue 移除。"""

        for block in blocks:
            if block.ref_cnt == 0:
                self.free.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, blocks: list[FakeBlock]) -> None:
        """减少引用计数并把 block 放回空闲列表。"""

        for block in blocks:
            block.ref_cnt -= 1
            assert block.ref_cnt >= 0
            if block.ref_cnt == 0:
                self.free.append(block)


def test_reservation_rollback_returns_blocks_to_vllm_pool() -> None:
    """rollback 应通过 vLLM free_blocks 归还所有预留 block。"""

    pool = FakeBlockPool(4)
    manager = VllmBlockPoolReservationManager(lambda: pool, lock=threading.RLock())

    reservation = manager.reserve(2)

    assert reservation.block_ids == (0, 1)
    assert reservation.state == BlockReservationState.RESERVED
    assert pool.get_num_free_blocks() == 2
    assert [block.ref_cnt for block in reservation.raw_blocks] == [1, 1]

    rolled_back = manager.rollback(reservation.reservation_id)

    assert rolled_back.state == BlockReservationState.ROLLED_BACK
    assert pool.get_num_free_blocks() == 4


def test_index_cleanup_failure_quarantines_only_unsafe_blocks(monkeypatch):
    pool = FakeBlockPool(2, fail_hash_insert_after=1)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    reservation = manager.reserve(2)
    original = pool.cached_block_hash_to_block.pop

    def pop_selected(block_hash, block_id):
        if block_id == 0:
            raise RuntimeError("injected pop failure")
        return original(block_hash, block_id)

    monkeypatch.setattr(pool.cached_block_hash_to_block, "pop", pop_selected)
    with pytest.raises(BlockReservationError, match="rollback failed"):
        manager.publish(reservation.reservation_id, block_hashes=[b"a", b"b"])
    assert reservation.state == BlockReservationState.CLEANUP_FAILED
    assert reservation.finished_at is None
    assert pool.blocks[0].ref_cnt == 1
    assert pool.blocks[1].ref_cnt == 0
    assert pool.get_num_free_blocks() == 1


def test_request_rollback_failure_still_cleans_safe_reservation_blocks():
    pool = FakeBlockPool(1)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    reservation = manager.reserve(1)

    def fail_attach(_blocks):
        raise RuntimeError("injected attach failure")

    def fail_rollback():
        raise RuntimeError("injected callback cleanup failure")

    with pytest.raises(BlockReservationError, match="request rollback"):
        manager.attach(
            reservation.reservation_id,
            block_hashes=[b"a"],
            attach_callback=fail_attach,
            rollback_callback=fail_rollback,
        )
    assert pool.get_num_free_blocks() == 1
    assert pool.cached_block_hash_to_block.values == {}
    assert reservation.state == BlockReservationState.CLEANUP_FAILED


def test_partial_free_failure_is_never_blindly_retried(monkeypatch):
    pool = FakeBlockPool(1)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    reservation = manager.reserve(1)
    calls = []

    def partial_free(blocks):
        calls.append(list(blocks))
        blocks[0].ref_cnt -= 1
        raise RuntimeError("injected partial release")

    monkeypatch.setattr(pool, "free_blocks", partial_free)
    with pytest.raises(BlockReservationError):
        manager.publish(reservation.reservation_id, block_hashes=[b"a"])
    assert len(calls) == 1
    assert reservation.state == BlockReservationState.CLEANUP_FAILED
    assert pool.blocks[0].ref_cnt == 0


def test_reservation_commit_holds_blocks_until_explicit_release() -> None:
    """commit 后 block 应继续被引用，release 才归还 vLLM。"""

    pool = FakeBlockPool(3)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    reservation = manager.reserve(2)

    committed = manager.commit(
        reservation.reservation_id,
        block_hashes=["hash-a", "hash-b"],
    )

    assert committed.state == BlockReservationState.COMMITTED
    assert committed.block_hashes == ("hash-a", "hash-b")
    assert pool.get_num_free_blocks() == 1

    released = manager.release(reservation.reservation_id)

    assert released.state == BlockReservationState.RELEASED
    assert pool.get_num_free_blocks() == 3


def test_reservation_publish_exposes_blocks_to_prefix_cache() -> None:
    """publish 应写入精确 hash 索引并把零引用 block 放回 free queue。"""

    pool = FakeBlockPool(3)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    reservation = manager.reserve(2)
    hashes = [
        b"a" * 32 + (0).to_bytes(4, "big"),
        b"b" * 32 + (0).to_bytes(4, "big"),
    ]

    published = manager.publish(reservation.reservation_id, block_hashes=hashes)

    assert published.state == BlockReservationState.PUBLISHED
    assert published.block_hashes == tuple(hashes)
    assert published.to_dict()["block_hashes"][0] == {"packed_hex": hashes[0].hex()}
    assert pool.get_num_free_blocks() == 3
    assert [block.ref_cnt for block in reservation.raw_blocks] == [0, 0]
    assert pool.cached_block_hash_to_block.values[hashes[0]][0] is pool.blocks[0]
    assert pool.cached_block_hash_to_block.values[hashes[1]][1] is pool.blocks[1]


def test_reservation_attach_transfers_reference_to_request_atomically() -> None:
    """attach 成功后 reservation 引用应转移给请求，且 hash 已经可见。"""

    pool = FakeBlockPool(2)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    reservation = manager.reserve(2)
    hashes = [b"a" * 36, b"b" * 36]
    request_blocks: list[FakeBlock] = []

    def attach_callback(blocks: tuple[FakeBlock, ...]) -> None:
        request_blocks.extend(blocks)
        for block in blocks:
            block.ref_cnt += 1

    def rollback_callback() -> None:
        for block in request_blocks:
            block.ref_cnt -= 1
        request_blocks.clear()

    attached = manager.attach(
        reservation.reservation_id,
        block_hashes=hashes,
        attach_callback=attach_callback,
        rollback_callback=rollback_callback,
    )

    assert attached.state == BlockReservationState.ATTACHED
    assert request_blocks == list(reservation.raw_blocks)
    assert [block.ref_cnt for block in request_blocks] == [1, 1]
    assert pool.get_num_free_blocks() == 0
    assert pool.cached_block_hash_to_block.values[hashes[0]][0] is pool.blocks[0]


def test_reservation_attach_reuses_concurrently_published_block() -> None:
    """attach 前 hash 已发布时应复用现有 block，并归还重复传输的目标块。"""

    pool = FakeBlockPool(4)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    existing_hash = b"a" * 36
    new_hash = b"b" * 36
    existing = pool.get_new_blocks(1)[0]
    existing.block_hash = existing_hash
    pool.cached_block_hash_to_block.insert(existing_hash, existing)
    pool.free_blocks([existing])
    reservation = manager.reserve(2)
    reserved_ids = reservation.block_ids
    request_blocks: list[FakeBlock] = []

    def attach_callback(blocks: tuple[FakeBlock, ...]) -> None:
        request_blocks.extend(blocks)
        for block in blocks:
            if block in pool.free:
                pool.free.remove(block)
            block.ref_cnt += 1

    def rollback_callback() -> None:
        for block in request_blocks:
            pool.free_blocks([block])
        request_blocks.clear()

    attached = manager.attach_reusing_published(
        reservation.reservation_id,
        block_hashes=[existing_hash, new_hash],
        attach_callback=attach_callback,
        rollback_callback=rollback_callback,
    )

    assert attached.state == BlockReservationState.ATTACHED
    assert attached.block_ids == (existing.block_id, reserved_ids[1])
    assert request_blocks == [existing, reservation.raw_blocks[1]]
    assert reservation.raw_blocks[0].ref_cnt == 0
    assert reservation.raw_blocks[0] in pool.free
    assert request_blocks[0].ref_cnt == 1
    assert request_blocks[1].ref_cnt == 1
    assert pool.cached_block_hash_to_block.values[new_hash][reserved_ids[1]] is (
        reservation.raw_blocks[1]
    )


def test_reservation_attach_keeps_partial_block_request_scoped() -> None:
    """混合 attach 应发布完整 block，但不能把 partial block 写入 hash 索引。"""

    pool = FakeBlockPool(3)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    reservation = manager.reserve(2)
    full_hash = b"a" * 36
    request_blocks: list[FakeBlock] = []

    def attach_callback(blocks: tuple[FakeBlock, ...]) -> None:
        request_blocks.extend(blocks)
        for block in blocks:
            block.ref_cnt += 1

    def rollback_callback() -> None:
        for block in request_blocks:
            block.ref_cnt -= 1
        request_blocks.clear()

    attached = manager.attach_request_scoped(
        reservation.reservation_id,
        block_hashes=[full_hash, None],
        attach_callback=attach_callback,
        rollback_callback=rollback_callback,
    )

    assert attached.state == BlockReservationState.ATTACHED
    assert attached.block_hashes == (full_hash, None)
    assert request_blocks == list(reservation.raw_blocks)
    assert pool.cached_block_hash_to_block.values[full_hash][0] is pool.blocks[0]
    assert pool.blocks[1].block_hash is None
    assert [block.ref_cnt for block in request_blocks] == [1, 1]


def test_delta_reservation_transfers_only_missing_and_partial_blocks() -> None:
    """delta plan 应 pin 本地命中，并只预留缺失完整块和 partial block。"""

    pool = FakeBlockPool(5)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    cached_hashes = [b"a" * 36, b"b" * 36]
    cached_blocks = pool.get_new_blocks(2)
    for block_hash, block in zip(cached_hashes, cached_blocks, strict=True):
        block.block_hash = block_hash
        pool.cached_block_hash_to_block.insert(block_hash, block)
    pool.free_blocks(cached_blocks)

    reservation = manager.reserve_delta([*cached_hashes, None])

    assert reservation.transfer_positions == (2,)
    assert len(reservation.raw_blocks) == 1
    assert reservation.block_ids == (2,)
    assert [block.block_id for block in reservation.planned_blocks] == [0, 1, 2]
    assert [block.ref_cnt for block in cached_blocks] == [1, 1]
    assert all(block not in pool.free for block in cached_blocks)

    request_blocks: list[FakeBlock] = []

    def attach_callback(blocks: tuple[FakeBlock, ...]) -> None:
        request_blocks.extend(blocks)
        pool.touch(list(blocks))

    def rollback_callback() -> None:
        pool.free_blocks(list(reversed(request_blocks)))
        request_blocks.clear()

    attached = manager.attach_request_scoped(
        reservation.reservation_id,
        block_hashes=[*cached_hashes, None],
        attach_callback=attach_callback,
        rollback_callback=rollback_callback,
    )

    assert attached.block_ids == (0, 1, 2)
    assert [block.ref_cnt for block in request_blocks] == [1, 1, 1]
    assert pool.blocks[2].block_hash is None


def test_delta_reservation_coalesces_inflight_complete_hashes() -> None:
    """并发delta plan遇到相同完整hash时应等待leader，不重复占用block。"""

    pool = FakeBlockPool(6)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    hashes = [b"a" * 36, b"b" * 36, None]
    leader = manager.reserve_delta(hashes)

    with pytest.raises(BlockHashesInFlightError) as error:
        manager.reserve_delta(hashes)

    assert error.value.hash_count == 2
    assert error.value.reservation_ids == (leader.reservation_id,)
    assert pool.get_num_free_blocks() == 3
    assert manager.wait_for_inflight_reservations(
        error.value.reservation_ids,
        timeout_seconds=0.001,
    ) is False

    manager.rollback(leader.reservation_id)

    assert manager.wait_for_inflight_reservations(
        error.value.reservation_ids,
        timeout_seconds=0.1,
    ) is True
    follower = manager.reserve_delta(hashes)
    assert follower.transfer_positions == (0, 1, 2)


def test_delta_capacity_waiter_owns_hashes_until_capacity_is_available() -> None:
    """容量不足时只允许一个同hash leader等待，follower不重复预留。"""

    pool = FakeBlockPool(5)
    occupied = pool.get_new_blocks(3)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    hashes = [b"a" * 36, b"b" * 36, None]

    with pytest.raises(BlockReservationCapacityError) as capacity_error:
        manager.reserve_delta(hashes, owner_id="leader")
    assert capacity_error.value.requested == 3
    assert capacity_error.value.available == 2

    with pytest.raises(BlockHashesInFlightError) as inflight_error:
        manager.reserve_delta(hashes, owner_id="follower")
    assert inflight_error.value.reservation_ids == ("pending:leader",)

    pool.free_blocks([occupied.pop()])
    manager.notify_capacity_changed()
    assert manager.wait_for_free_blocks(3, timeout_seconds=0.1) is True
    reservation = manager.reserve_delta(hashes, owner_id="leader")

    assert reservation.transfer_positions == (0, 1, 2)
    assert manager.release_pending_delta("leader") is False


def test_delta_reservation_touch_failure_releases_only_acquired_pins() -> None:
    """touch 中途失败时不能释放尚未由本次 plan 持有的其他请求引用。"""

    class PartiallyFailingTouchPool(FakeBlockPool):
        def touch(self, blocks: list[FakeBlock]) -> None:
            super().touch(blocks[:1])
            raise RuntimeError("injected partial touch failure")

    pool = PartiallyFailingTouchPool(3)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    cached_hashes = [b"a" * 36, b"b" * 36]
    cached_blocks = pool.get_new_blocks(2)
    for block_hash, block in zip(cached_hashes, cached_blocks, strict=True):
        block.block_hash = block_hash
        pool.cached_block_hash_to_block.insert(block_hash, block)
    pool.free_blocks(cached_blocks)

    with pytest.raises(RuntimeError, match="partial touch failure"):
        manager.reserve_delta(cached_hashes)

    assert [block.ref_cnt for block in cached_blocks] == [0, 0]
    assert all(block in pool.free for block in cached_blocks)


def test_delta_reservation_rollback_releases_reused_pins_and_missing_blocks() -> None:
    """delta rollback 应同时释放本地命中 pin 和尚未 attach 的目标 block。"""

    pool = FakeBlockPool(3)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    cached_hash = b"a" * 36
    cached = pool.get_new_blocks(1)[0]
    cached.block_hash = cached_hash
    pool.cached_block_hash_to_block.insert(cached_hash, cached)
    pool.free_blocks([cached])

    reservation = manager.reserve_delta([cached_hash, None])
    manager.rollback(reservation.reservation_id)

    assert reservation.state == BlockReservationState.ROLLED_BACK
    assert cached.ref_cnt == 0
    assert pool.get_num_free_blocks() == 3


def test_reservation_attach_rolls_back_request_and_hash_on_failure() -> None:
    """request attach 失败时应同时撤销请求引用、hash 索引和 reservation。"""

    pool = FakeBlockPool(2)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    reservation = manager.reserve(2)
    request_blocks: list[FakeBlock] = []

    def attach_callback(blocks: tuple[FakeBlock, ...]) -> None:
        request_blocks.extend(blocks)
        for block in blocks:
            block.ref_cnt += 1
        raise RuntimeError("injected request attach failure")

    def rollback_callback() -> None:
        for block in request_blocks:
            block.ref_cnt -= 1
        request_blocks.clear()

    with pytest.raises(BlockReservationError, match="failed to attach"):
        manager.attach(
            reservation.reservation_id,
            block_hashes=[b"a" * 36, b"b" * 36],
            attach_callback=attach_callback,
            rollback_callback=rollback_callback,
        )

    assert reservation.state == BlockReservationState.ROLLED_BACK
    assert request_blocks == []
    assert pool.get_num_free_blocks() == 2
    assert pool.cached_block_hash_to_block.values == {}
    assert [block.block_hash for block in reservation.raw_blocks] == [None, None]


def test_reservation_publish_rolls_back_partial_hash_insertion() -> None:
    """任一 hash 插入失败时应撤销索引并归还全部预留 block。"""

    pool = FakeBlockPool(3, fail_hash_insert_after=1)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    reservation = manager.reserve(2)
    hashes = [b"a" * 36, b"b" * 36]

    with pytest.raises(BlockReservationError, match="failed to publish"):
        manager.publish(reservation.reservation_id, block_hashes=hashes)

    assert reservation.state == BlockReservationState.ROLLED_BACK
    assert pool.get_num_free_blocks() == 3
    assert pool.cached_block_hash_to_block.values == {}
    assert [block.block_hash for block in reservation.raw_blocks] == [None, None]


def test_reservation_publish_rejects_hash_published_by_concurrent_request() -> None:
    """publish 前若相同 packed hash 已存在，应拒绝并允许调用方 rollback。"""

    pool = FakeBlockPool(3)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    existing = pool.get_new_blocks(1)[0]
    packed_hash = b"a" * 32 + (0).to_bytes(4, "big")
    existing.block_hash = packed_hash
    pool.cached_block_hash_to_block.insert(packed_hash, existing)
    pool.free_blocks([existing])
    reservation = manager.reserve(1)

    with pytest.raises(BlockAlreadyPublishedError, match="already published") as exc_info:
        manager.publish(
            reservation.reservation_id,
            block_hashes=[packed_hash],
        )

    assert exc_info.value.existing_block_ids == (existing.block_id,)
    assert reservation.state == BlockReservationState.RESERVED
    manager.rollback(reservation.reservation_id)
    assert pool.get_num_free_blocks() == 3


def test_reservation_publish_prevalidates_hash_count_and_cache_mode() -> None:
    """publish 输入无效时不应修改 reservation 或 BlockPool。"""

    pool = FakeBlockPool(2, enable_caching=False)
    manager = VllmBlockPoolReservationManager(lambda: pool)
    reservation = manager.reserve(1)

    with pytest.raises(BlockReservationError, match="count must match"):
        manager.publish(reservation.reservation_id, block_hashes=[])
    with pytest.raises(BlockReservationError, match="prefix caching is disabled"):
        manager.publish(reservation.reservation_id, block_hashes=[b"a" * 36])

    assert reservation.state == BlockReservationState.RESERVED
    assert pool.get_num_free_blocks() == 1
    manager.rollback(reservation.reservation_id)


def test_reservation_rejects_insufficient_blocks_and_invalid_transitions() -> None:
    """reservation 应拒绝超量分配和重复回滚等非法状态迁移。"""

    pool = FakeBlockPool(1)
    manager = VllmBlockPoolReservationManager(lambda: pool)

    with pytest.raises(BlockReservationError, match="insufficient free"):
        manager.reserve(2)

    reservation = manager.reserve(1)
    manager.rollback(reservation.reservation_id)

    with pytest.raises(BlockReservationError, match="expected reserved"):
        manager.rollback(reservation.reservation_id)


def test_reservation_requires_supported_vllm_block_pool() -> None:
    """尚未观察到 vLLM BlockPool 时应返回明确错误，不猜测物理 block。"""

    manager = VllmBlockPoolReservationManager(lambda: None)

    with pytest.raises(BlockReservationError, match="reservation API is not ready"):
        manager.reserve(1)


def test_reservation_ttl_returns_abandoned_blocks_to_vllm_pool(monkeypatch) -> None:
    """目标 reservation 超时后应自动归还物理 block，并保留过期诊断状态。"""

    from jiuwen_kvbm.runtime import block_reservation as reservation_module

    now = [100.0]
    monkeypatch.setattr(reservation_module.time, "monotonic", lambda: now[0])
    pool = FakeBlockPool(2)
    manager = VllmBlockPoolReservationManager(
        lambda: pool,
        min_ttl_ms=1,
        max_ttl_ms=60_000,
    )
    reservation = manager.reserve(1, ttl_ms=10)

    assert pool.get_num_free_blocks() == 1
    now[0] += 0.011

    assert manager.expire_due() == 1
    assert reservation.state == BlockReservationState.EXPIRED
    assert pool.get_num_free_blocks() == 2
    assert manager.stats()["states"]["expired"] == 1


def test_reservation_renew_extends_target_block_ownership(monkeypatch) -> None:
    """活跃传输续租后，旧到期时间不应回收目标物理 block。"""

    from jiuwen_kvbm.runtime import block_reservation as reservation_module

    now = [200.0]
    monkeypatch.setattr(reservation_module.time, "monotonic", lambda: now[0])
    pool = FakeBlockPool(2)
    manager = VllmBlockPoolReservationManager(
        lambda: pool,
        min_ttl_ms=1,
        max_ttl_ms=60_000,
    )
    reservation = manager.reserve(1, ttl_ms=10)

    now[0] += 0.005
    manager.renew(reservation.reservation_id, ttl_ms=20)
    now[0] += 0.006

    assert manager.expire_due() == 0
    assert reservation.state == BlockReservationState.RESERVED
    assert pool.get_num_free_blocks() == 1
