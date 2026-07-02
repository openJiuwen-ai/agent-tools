from vllm.v1.core.kv_cache_utils import KVCacheBlock

from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_session_manager import (
    KvCacheSessionManager,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.single_type_kv_cache_manager import (
    collect_consecutive_session_tail,
    collect_release_tail_orphans,
    expected_release_block_count,
    should_collect_orphan_tail,
)


def test_expected_release_block_count():
    block_size = 128
    assert expected_release_block_count(0, 1000, block_size) == 8
    assert expected_release_block_count(0, 1024, block_size) == 8
    assert expected_release_block_count(896, 1000, block_size) == 1
    assert expected_release_block_count(0, 128, block_size) == 1


def test_should_collect_orphan_tail():
    block_size = 128
    num_tokens = 1000
    assert should_collect_orphan_tail(0, num_tokens, block_size, 7, 7) is True
    assert should_collect_orphan_tail(0, num_tokens, block_size, 8, 8) is True
    assert should_collect_orphan_tail(0, 1024, block_size, 7, 7) is True
    assert should_collect_orphan_tail(0, 1024, block_size, 8, 8) is True
    assert should_collect_orphan_tail(896, num_tokens, block_size, 0, 1) is True
    assert should_collect_orphan_tail(None, num_tokens, block_size, 0, 0) is True


def test_collect_consecutive_session_tail():
    pool_blocks = [KVCacheBlock(idx) for idx in range(220)]
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks(pool_blocks[206:214], "a1")

    resolved = pool_blocks[206:213]
    tail = collect_consecutive_session_tail(
        session_mgr, "a1", pool_blocks, resolved
    )
    assert [b.block_id for b in tail] == [213]


def test_collect_consecutive_session_tail_skips_earlier_session_blocks():
    pool_blocks = [KVCacheBlock(idx) for idx in range(220)]
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks(pool_blocks[1:206], "shared")
    session_mgr.add_blocks(pool_blocks[206:214], "shared")

    resolved = pool_blocks[206:213]
    tail = collect_consecutive_session_tail(
        session_mgr, "shared", pool_blocks, resolved
    )
    assert [b.block_id for b in tail] == [213]


def test_collect_consecutive_session_tail_busy_fallback():
    pool_blocks = [KVCacheBlock(idx) for idx in range(220)]
    pool_blocks[213].ref_cnt = 1
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks(pool_blocks[206:214], "a1")

    resolved = pool_blocks[206:213]
    assert collect_consecutive_session_tail(
        session_mgr, "a1", pool_blocks, resolved, only_idle=True
    ) == []
    tail = collect_release_tail_orphans(
        session_mgr,
        "a1",
        pool_blocks,
        resolved,
        release_token_index=0,
        num_tokens=1024,
        block_size=128,
        hash_suffix_len=7,
    )
    assert [b.block_id for b in tail] == [213]


def test_collect_orphan_blocks_partial_tail():
    pool_blocks = [KVCacheBlock(idx) for idx in range(4)]
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks(pool_blocks, "s1")

    orphans = session_mgr.collect_orphan_blocks(
        "s1", pool_blocks, exclude_block_ids={0, 1, 2}, only_idle=True
    )
    assert len(orphans) == 1
    assert orphans[0].block_id == 3


def test_collect_orphan_blocks_release_merges_tail():
    pool_blocks = [KVCacheBlock(idx) for idx in range(4)]
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks(pool_blocks, "s1")

    resolved = pool_blocks[:3]
    tail = collect_release_tail_orphans(
        session_mgr,
        "s1",
        pool_blocks,
        resolved,
        release_token_index=0,
        num_tokens=512,
        block_size=128,
        hash_suffix_len=3,
    )
    blocks_to_release = resolved + tail
    released = session_mgr.release_blocks(blocks_to_release, "s1")
    assert len(released) == 4
    assert {b.block_id for b in released} == {0, 1, 2, 3}


def test_collect_release_tail_when_block_ids_descend():
    """Release-zone reuse can make the partial tail lower than max resolved id."""
    pool_blocks = [KVCacheBlock(idx) for idx in range(300)]
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks([pool_blocks[i] for i in range(280, 274, -1)], "s")

    resolved = [pool_blocks[i] for i in range(280, 275, -1)]
    tail = collect_release_tail_orphans(
        session_mgr,
        "s",
        pool_blocks,
        resolved,
        release_token_index=0,
        num_tokens=6 * 128 - 32,
        block_size=128,
        hash_suffix_len=5,
    )
    assert [b.block_id for b in tail] == [275]

    session_mgr.add_blocks([pool_blocks[267]], "s2")
    session_mgr.add_blocks([pool_blocks[i] for i in range(274, 267, -1)], "s2")
    resolved2 = [pool_blocks[i] for i in range(274, 267, -1)]
    tail2 = collect_release_tail_orphans(
        session_mgr,
        "s2",
        pool_blocks,
        resolved2,
        release_token_index=0,
        num_tokens=8 * 128 - 64,
        block_size=128,
        hash_suffix_len=7,
    )
    assert [b.block_id for b in tail2] == [267]
