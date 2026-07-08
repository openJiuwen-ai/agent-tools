from vllm.v1.core.kv_cache_utils import KVCacheBlock

from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_session_manager import (
    KvCacheSessionManager,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.single_type_kv_cache_manager import (
    bind_allocated_blocks,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.two_phase_block_queue import (
    jiuwen_is_reclaimed,
    jiuwen_mark_reclaimed_for_test,
)


def test_reclaimed_blocks_reset_stale_bindings():
    blocks = [KVCacheBlock(0), KVCacheBlock(1)]
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks(blocks, "old_a")
    session_mgr.add_blocks(blocks, "old_b")
    assert session_mgr.block_to_sessions[0] == ["old_a", "old_b"]
    assert session_mgr.block_to_sessions[1] == ["old_a", "old_b"]

    jiuwen_mark_reclaimed_for_test(blocks[0])
    jiuwen_mark_reclaimed_for_test(blocks[1])

    bind_allocated_blocks(session_mgr, blocks, "new_owner")

    assert session_mgr.block_to_sessions[0] == ["new_owner"]
    assert session_mgr.block_to_sessions[1] == ["new_owner"]
    assert not jiuwen_is_reclaimed(blocks[0])
    assert not jiuwen_is_reclaimed(blocks[1])


def test_fresh_blocks_use_add_semantics():
    blocks = [KVCacheBlock(0)]
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks(blocks, "existing")

    bind_allocated_blocks(session_mgr, blocks, "new_owner")

    assert session_mgr.block_to_sessions[0] == ["existing", "new_owner"]


def test_prefix_hit_style_binding_unchanged():
    blocks = [KVCacheBlock(0)]
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks(blocks, "shared_a")

    session_mgr.add_blocks(blocks, "shared_b")

    assert session_mgr.block_to_sessions[0] == ["shared_a", "shared_b"]


def test_reclaimed_blocks_clear_stale_bindings_without_session():
    blocks = [KVCacheBlock(0)]
    jiuwen_mark_reclaimed_for_test(blocks[0])
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks(blocks, "old")

    bind_allocated_blocks(session_mgr, blocks, None)

    assert session_mgr.block_to_sessions[0] == []
    assert not jiuwen_is_reclaimed(blocks[0])


def test_fresh_blocks_unbound_without_session():
    blocks = [KVCacheBlock(0)]
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks(blocks, "existing")

    bind_allocated_blocks(session_mgr, blocks, None)

    assert session_mgr.block_to_sessions[0] == ["existing"]
