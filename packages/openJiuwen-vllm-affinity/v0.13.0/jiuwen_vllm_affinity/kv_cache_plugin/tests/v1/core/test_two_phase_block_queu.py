from vllm.v1.core.kv_cache_utils import KVCacheBlock
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.two_phase_block_queue import (
    TwoPhaseBlockQueue,
)


def create_two_phase_queue(queue_len: int) -> TwoPhaseBlockQueue:
    blocks: list[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(queue_len)]
    return TwoPhaseBlockQueue(blocks)


def test_basic_fifo_free():
    block_queue = create_two_phase_queue(5)
    assert [b.block_id for b in block_queue.popleft_n(5)] == [0, 1, 2, 3, 4]
    assert block_queue.num_free_blocks == 0


def test_two_phase_queue():
    block_count = 6
    block_queue = create_two_phase_queue(block_count)
    blocks = block_queue.get_all_free_blocks()
    for idx in range(block_count // 2):
        blk = block_queue.popleft()
        assert blk.block_id == idx
        block_queue.append(blk)
    for idx in range(0, block_count, 2):
        block_queue.aging_block(blocks[idx])
        assert blocks[idx].block_id == idx
    # free [3,5], release [0,2,4], normal stale [1]
    block_ids = [3, 5, 0, 2, 4, 1]
    for expected in block_ids:
        blk = block_queue.popleft()
        assert blk.block_id == expected
    assert len(block_queue.get_all_free_blocks()) == 0


def test_free_preferred_over_release():
    """Never-allocated (free) blocks must be allocated before released blocks."""
    block_queue = create_two_phase_queue(20)
    block_queue.popleft()  # null block 0

    used = block_queue.popleft_n(5)  # 1-5
    for blk in used:
        block_queue.append(blk)
    for blk in used:
        block_queue.aging_block(blk)  # release zone = 1-5

    # 6-19 are still free (never allocated) and must come first.
    nxt = block_queue.popleft_n(3)
    assert [b.block_id for b in nxt] == [6, 7, 8]


def test_release_preferred_over_normal():
    """Released blocks must be reused before plain normal-stale blocks."""
    block_queue = create_two_phase_queue(20)
    block_queue.popleft()  # null block 0

    req1 = block_queue.popleft_n(5)  # 1-5
    for blk in req1:
        block_queue.append(blk)  # normal stale = 1-5

    a1 = block_queue.popleft_n(14)  # 6-19, drains the free zone
    for blk in a1:
        block_queue.append(blk)
    for blk in a1:
        block_queue.aging_block(blk)  # release zone = 6-19

    ids = [b.block_id for b in block_queue.popleft_n(5)]
    assert ids == [6, 7, 8, 9, 10], ids
    assert 1 not in ids


def test_user_scenario_release_reused_before_normal():
    """Request1 holds 1-205 (normal); a1.. release 206+; next reuses release, not 204."""
    block_queue = create_two_phase_queue(986)

    null_blk = block_queue.popleft()
    assert null_blk.block_id == 0

    req1 = block_queue.popleft_n(205)  # 1-205
    assert [b.block_id for b in req1] == list(range(1, 206))
    for blk in req1:
        block_queue.append(blk)  # normal stale 1-205

    a1 = block_queue.popleft_n(10)  # 206-215
    assert [b.block_id for b in a1] == list(range(206, 216))
    for blk in a1:
        block_queue.append(blk)
    for blk in a1:
        block_queue.aging_block(blk)  # release zone seeded with 206-215

    # Drain the remaining free zone 216-985 (770 blocks), releasing each batch.
    for _ in range(77):
        batch = block_queue.popleft_n(10)
        for blk in batch:
            block_queue.append(blk)
        for blk in batch:
            block_queue.aging_block(blk)

    # Free zone empty: next request must come from release LRU (206-215),
    # never from normal stale (204, etc.).
    next_batch = block_queue.popleft_n(10)
    ids = sorted(b.block_id for b in next_batch)
    assert ids == list(range(206, 216)), f"expected release 206-215, got {ids}"
    assert 204 not in ids


def test_reclaimed_flag_free_vs_reused():
    """Free pops are not reclaimed; release/normal pops are (for the 4-evict log)."""
    block_queue = create_two_phase_queue(10)

    fresh = block_queue.popleft()
    assert fresh.jiuwen_reclaimed is False

    used = block_queue.popleft_n(4)  # 1-4
    for blk in used:
        block_queue.append(blk)
    block_queue.aging_block(used[0])  # release block 1

    for blk in block_queue.popleft_n(5):  # free 5-9
        assert blk.jiuwen_reclaimed is False

    released = block_queue.popleft()
    assert released.block_id == 1
    assert released.jiuwen_reclaimed is True

    normal = block_queue.popleft()
    assert normal.jiuwen_reclaimed is True


def test_remove_keeps_released_tail_valid():
    block_queue = create_two_phase_queue(10)
    blocks = block_queue.get_all_free_blocks()
    block_queue.remove(blocks[9])  # remove current released_tail (last block)
    block_queue.remove(blocks[3])
    assert [block_queue.popleft().block_id for _ in range(3)] == [0, 1, 2]
    assert block_queue.num_free_blocks == 5


def test_aging_busy_block_returns_zero():
    block_queue = create_two_phase_queue(5)
    blocks = block_queue.get_all_free_blocks()
    blocks[2].ref_cnt = 1
    assert block_queue.aging_block(blocks[2]) == 0


def test_aging_out_of_queue_block_returns_zero():
    block_queue = create_two_phase_queue(5)
    popped = block_queue.popleft()
    assert block_queue.aging_block(popped) == 0


def test_full_drain_then_recycle():
    block_queue = create_two_phase_queue(8)
    all_blocks = block_queue.popleft_n(8)
    assert block_queue.num_free_blocks == 0
    for blk in all_blocks:
        block_queue.append(blk)
    for blk in all_blocks[:4]:
        block_queue.aging_block(blk)  # release 0-3, normal 4-7
    assert [block_queue.popleft().block_id for _ in range(8)] == [0, 1, 2, 3, 4, 5, 6, 7]
