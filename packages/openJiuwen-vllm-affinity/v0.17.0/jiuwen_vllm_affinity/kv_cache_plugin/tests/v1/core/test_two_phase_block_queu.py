import pytest
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.two_phase_block_queue import TwoPhaseBlockQueue


def create_two_phase_queue(queue_len: int) -> TwoPhaseBlockQueue:
    blocks: list[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(queue_len)]
    return TwoPhaseBlockQueue(blocks)


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
    # Released zone [0,2,4] (LRU->MRU), then normal zone [3,5,1].
    block_ids = [0, 2, 4, 3, 5, 1]
    for idx in range(block_count):
        blk = block_queue.popleft()
        assert blk.block_id == block_ids[idx]
    assert len(block_queue.get_all_free_blocks()) == 0


def test_released_zone_evicted_before_normal():
    block_queue = create_two_phase_queue(10)
    all_blocks = block_queue.get_all_free_blocks()

    # Simulate request1 occupying blocks 0-4 (normal zone, MRU side after append).
    for i in range(5):
        block_queue.popleft()
    for i in range(5):
        block_queue.append(all_blocks[i])

    # Simulate serial requests releasing blocks 6-8.
    for i in [6, 7, 8]:
        block_queue.aging_block(all_blocks[i])

    # Evict released blocks first (LRU among released), then normal blocks.
    popped = [block_queue.popleft().block_id for _ in range(6)]
    assert popped == [6, 7, 8, 5, 9, 0]
