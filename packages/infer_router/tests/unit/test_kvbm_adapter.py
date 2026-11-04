import pytest

pytest.importorskip("jiuwen_kvbm")

from openjiuwentools.infer_router.kv_cache.event import CacheEvent, CacheLocation
from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.kvbm.adapter import RouterKVBMAdapter, RouterKVBMConfig


def test_router_kvbm_adapter_tracks_store_move_and_remove(tmp_path):
    """router KVBM adapter 应将 KV 事件同步到 KVBMManager。"""
    adapter = RouterKVBMAdapter(RouterKVBMConfig(disk_root=tmp_path))
    event = CacheEvent(
        event_type="store",
        block_hashes=["block-1"],
        token_ids=list(range(16)),
        worker_id="worker-1",
        location=CacheLocation.NPU,
        engine_specific={"worker_id": "worker-1", "block_size": 16},
    )

    adapter.handle_event(event)
    assert adapter.get_stats()["cached_npu_blocks"] == 1

    adapter.handle_event(
        CacheEvent(
            event_type="move",
            block_hashes=["block-1"],
            worker_id="worker-1",
            location=CacheLocation.CPU,
            engine_specific={"worker_id": "worker-1", "block_size": 16, "to": "ram"},
        )
    )
    assert adapter.get_stats()["ram_blocks"] == 1

    adapter.handle_event(
        CacheEvent(
            event_type="move",
            block_hashes=["block-1"],
            worker_id="worker-1",
            location=CacheLocation.REMOTE,
            engine_specific={"worker_id": "worker-1", "block_size": 16, "to": "disk"},
        )
    )
    assert adapter.get_stats()["disk_blocks"] == 1

    adapter.handle_event(
        CacheEvent(
            event_type="removed",
            block_hashes=["block-1"],
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1", "block_size": 16},
        )
    )
    assert adapter.get_stats()["evicted_blocks"] == 1


def test_kv_cache_manager_exposes_kvbm_stats():
    """KVCacheManager 应通过事件处理链路暴露 kvbm_stats。"""
    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("worker-1", "vllm", "model-1")

    assert manager.kvbm_adapter is not None
    assert manager.process_event(
        CacheEvent(
            event_type="store",
            block_hashes=["block-1"],
            token_ids=list(range(manager.block_size)),
            worker_id="worker-1",
            location=CacheLocation.NPU,
            engine_specific={"worker_id": "worker-1", "block_size": manager.block_size},
        )
    )

    stats = manager.get_cache_stats()
    assert stats["kvbm_stats"]["total_blocks"] == 1
    assert stats["kvbm_stats"]["cached_npu_blocks"] == 1
