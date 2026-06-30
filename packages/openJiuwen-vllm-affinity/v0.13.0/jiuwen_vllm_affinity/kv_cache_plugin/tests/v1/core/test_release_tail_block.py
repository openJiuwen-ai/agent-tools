import pytest
from vllm.v1.core.kv_cache_utils import KVCacheBlock

from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_session_manager import (
    KvCacheSessionManager,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.single_type_kv_cache_manager import (
    should_supplement_partial_tail,
)


def test_should_supplement_partial_tail():
    block_size = 128
    num_tokens = 1000  # partial tail: 1000 % 128 = 104
    partial_start = (1000 // 128) * 128  # 896

    assert should_supplement_partial_tail(0, num_tokens, block_size) is True
    assert should_supplement_partial_tail(partial_start, num_tokens, block_size) is True
    assert should_supplement_partial_tail(partial_start + 1, num_tokens, block_size) is False
    assert should_supplement_partial_tail(None, num_tokens, block_size) is True
    assert should_supplement_partial_tail(0, 128, block_size) is False
    assert should_supplement_partial_tail(0, 256, block_size) is False


def test_collect_orphan_blocks_partial_tail():
    pool_blocks = [KVCacheBlock(idx) for idx in range(4)]
    session_mgr = KvCacheSessionManager()
    session_mgr.reset_blocks(pool_blocks, "s1")

    orphans = session_mgr.collect_orphan_blocks(
        "s1", pool_blocks, exclude_block_ids={0, 1, 2}, only_idle=True
    )
    assert len(orphans) == 1
    assert orphans[0].block_id == 3


def test_collect_orphan_blocks_skips_busy_tail():
    pool_blocks = [KVCacheBlock(idx) for idx in range(4)]
    pool_blocks[3].ref_cnt = 1
    session_mgr = KvCacheSessionManager()
    session_mgr.reset_blocks(pool_blocks, "s1")

    orphans = session_mgr.collect_orphan_blocks(
        "s1", pool_blocks, exclude_block_ids={0, 1, 2}, only_idle=True
    )
    assert orphans == []


def test_collect_orphan_blocks_release_merges_tail():
    """Simulate hash-resolved full blocks + partial tail orphan release."""
    pool_blocks = [KVCacheBlock(idx) for idx in range(4)]
    session_mgr = KvCacheSessionManager()
    session_mgr.reset_blocks(pool_blocks, "s1")

    resolved = pool_blocks[:3]
    tail = session_mgr.collect_orphan_blocks(
        "s1", pool_blocks, exclude_block_ids={b.block_id for b in resolved}
    )
    blocks_to_release = resolved + tail
    released = session_mgr.release_blocks(blocks_to_release, "s1")
    assert len(released) == 4
    assert {b.block_id for b in released} == {0, 1, 2, 3}
