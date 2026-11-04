import logging

from openjiuwentools.infer_router.kv_cache.event import CacheEvent
from openjiuwentools.infer_router.kv_cache.event import CacheLocation
from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager


def run_test(name, test_func):
    """运行单个测试"""
    try:
        test_func()
        logging.info(f"✓ {name}")
        return True
    except AssertionError as e:
        logging.info(f"✗ {name}: {e}")
        return False
    except Exception as e:
        logging.info(f"✗ {name}: Unexpected error - {e}")
        logging.exception("Traceback:")
        return False


def test_initial_overlap_is_zero():
    """测试初始状态下overlap为0"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    test_tokens = [60, 124, 117, 115, 101, 114, 124, 62, 105, 32, 119, 97, 110, 116, 32, 116]

    workers = [
        "worker-1-prefill",
        "worker-2-decode",
        "worker-3-decode",
        "worker-4-prefill",
        "worker-5-decode",
        "worker-6-decode",
    ]

    for worker in workers:
        kv_cache_manager.register_worker(worker, "vllm", "Qwen3-8B")

    overlap_scores = kv_cache_manager.find_matches(token_ids=test_tokens, model="Qwen3-8B")

    for worker_id in workers:
        assert overlap_scores.get(worker_id, 0) == 0, f"初始状态下 {worker_id} 的overlap应为0"


def test_store_and_retrieve_block():
    """测试存储块后能正确检索"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    test_tokens = [60, 124, 117, 115, 101, 114, 124, 62, 105, 32, 119, 97, 110, 116, 32, 116]

    kv_cache_manager.register_worker("worker-2-decode", "vllm", "Qwen3-8B")

    block_size = kv_cache_manager.block_size
    assert len(test_tokens) >= block_size, "测试token数量不足"

    block_tokens = test_tokens[:block_size]
    block_hash = kv_cache_manager.scorer.compute_block_hash(block_tokens)

    event = CacheEvent(
        event_type="store",
        block_hashes=[block_hash],
        token_ids=block_tokens,
        worker_id="worker-2-decode",
        engine_specific={"worker_id": "worker-2-decode"},
    )

    kv_cache_manager.process_event(event)

    overlap_scores = kv_cache_manager.find_matches(token_ids=test_tokens, model="Qwen3-8B")

    assert overlap_scores.get("worker-2-decode", 0) == block_size, f"存储块后overlap应为{block_size}"


def test_get_matched_block_hashes_returns_event_hashes():
    """测试 KVBM 预取路径能取回 worker 上真实命中的 block hash。"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    kv_cache_manager.register_worker("worker-1", "vllm", "Qwen3-8B")

    block_size = kv_cache_manager.block_size
    token_ids = list(range(block_size * 2))
    block_hashes = ["vllm-hash-0", "vllm-hash-1"]
    event = CacheEvent(
        event_type="store",
        block_hashes=block_hashes,
        token_ids=token_ids,
        worker_id="worker-1",
        engine_specific={"worker_id": "worker-1", "block_size": block_size},
    )

    kv_cache_manager.process_event(event)

    assert kv_cache_manager.get_matched_block_hashes("worker-1", token_ids) == block_hashes
    assert kv_cache_manager.get_matched_block_hashes("worker-1", token_ids, limit=1) == ["vllm-hash-0"]


def test_find_matches_returns_reusable_snapshot_and_updates_active_nodes():
    """首次匹配应产出快照，移除 block 后增量活跃节点索引应立即失效。"""

    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("worker-1", "vllm", "Qwen3-8B")
    block_size = manager.block_size
    token_ids = list(range(block_size * 2))
    block_hashes = ["snapshot-hash-0", "snapshot-hash-1"]
    manager.process_event(
        CacheEvent(
            event_type="store",
            block_hashes=block_hashes,
            token_ids=token_ids,
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1", "block_size": block_size},
        )
    )
    snapshots = {}

    matches = manager.find_matches(
        token_ids,
        "Qwen3-8B",
        snapshot_output=snapshots,
    )

    state = manager.index.pod_states["worker-1"]
    assert matches == {"worker-1": block_size * 2}
    assert snapshots.get("worker-1").matched_tokens == block_size * 2
    assert len(state.active_node_ref_counts) == 2

    manager.process_event(
        CacheEvent(
            event_type="removed",
            block_hashes=[block_hashes[1]],
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1", "block_size": block_size},
        )
    )

    assert manager.find_matches(token_ids, "Qwen3-8B") == {"worker-1": block_size}
    assert len(state.active_node_ref_counts) == 1


def test_common_prefix_cursor_cache_reuses_path_and_keeps_metadata_live():
    """公共前缀应复用 Radix 游标，location 更新仍应从实时元数据读取。"""

    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("worker-1", "vllm", "Qwen3-8B")
    event_block_size = manager.block_size * 2
    common_tokens = list(range(event_block_size * 2))
    block_hashes = ["cursor-hash-0", "cursor-hash-1"]
    manager.process_event(
        CacheEvent(
            event_type="store",
            block_hashes=block_hashes,
            token_ids=common_tokens,
            worker_id="worker-1",
            location=CacheLocation.NPU,
            engine_specific={
                "worker_id": "worker-1",
                "block_size": event_block_size,
            },
        )
    )

    first_timing = {}
    first = manager.get_worker_prefix_snapshot(
        "worker-1",
        common_tokens,
        timing=first_timing,
    )
    assert first.matched_tokens == len(common_tokens)
    assert first_timing.get("cursor_cache_hits") == 0
    assert first_timing.get("cursor_cache_misses") == 2

    manager.process_event(
        CacheEvent(
            event_type="move",
            block_hashes=[block_hashes[0]],
            worker_id="worker-1",
            location=CacheLocation.CPU,
            engine_specific={
                "worker_id": "worker-1",
                "block_size": event_block_size,
            },
        )
    )
    second_timing = {}
    second = manager.get_worker_prefix_snapshot(
        "worker-1",
        common_tokens,
        timing=second_timing,
    )

    assert second_timing.get("cursor_cache_hits") == 2
    assert second_timing.get("cursor_cache_misses") == 0
    assert second.blocks[0].select().location == CacheLocation.CPU
    assert manager.index.pod_states["worker-1"].radix_block_size == event_block_size


def test_common_prefix_cursor_cache_is_invalidated_by_clear():
    """Worker clear 后必须提升查询代次，不能复用旧 RadixTree 游标。"""

    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("worker-1", "vllm", "Qwen3-8B")
    block_size = manager.block_size
    token_ids = list(range(block_size))
    manager.process_event(
        CacheEvent(
            event_type="store",
            block_hashes=["cursor-hash-0"],
            token_ids=token_ids,
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1", "block_size": block_size},
        )
    )
    manager.get_worker_prefix_snapshot("worker-1", token_ids)
    state = manager.index.pod_states["worker-1"]
    generation_before_clear = state.radix_query_generation

    manager.process_event(
        CacheEvent(
            event_type="cleared",
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1"},
        )
    )
    timing = {}
    snapshot = manager.get_worker_prefix_snapshot(
        "worker-1",
        token_ids,
        timing=timing,
    )

    assert state.radix_query_generation == generation_before_clear + 1
    assert snapshot.matched_tokens == 0
    assert timing.get("cursor_cache_hits") == 0
    assert timing.get("cursor_cache_misses") == 1


def test_get_matched_block_hashes_can_filter_by_location():
    """测试 KVBM prefetch 可只选择 RAM/DISK 层级的命中 block。"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    kv_cache_manager.register_worker("worker-1", "vllm", "Qwen3-8B")

    block_size = kv_cache_manager.block_size
    token_ids = list(range(block_size * 2))
    block_hashes = ["vllm-hash-0", "vllm-hash-1"]
    kv_cache_manager.process_event(
        CacheEvent(
            event_type="store",
            block_hashes=block_hashes,
            token_ids=token_ids,
            worker_id="worker-1",
            location=CacheLocation.NPU,
            engine_specific={"worker_id": "worker-1", "block_size": block_size},
        )
    )

    assert (
        kv_cache_manager.get_matched_block_hashes(
            "worker-1",
            token_ids,
            locations={CacheLocation.CPU, CacheLocation.REMOTE},
        )
        == []
    )

    kv_cache_manager.process_event(
        CacheEvent(
            event_type="move",
            block_hashes=["vllm-hash-0"],
            worker_id="worker-1",
            location=CacheLocation.CPU,
            engine_specific={"worker_id": "worker-1", "block_size": block_size},
        )
    )

    assert kv_cache_manager.get_matched_block_hashes(
        "worker-1",
        token_ids,
        locations={CacheLocation.CPU, CacheLocation.REMOTE},
    ) == ["vllm-hash-0"]
    assert kv_cache_manager.get_matched_block_location_counts("worker-1", token_ids) == {
        CacheLocation.CPU: 1,
        CacheLocation.NPU: 1,
    }
    assert kv_cache_manager.get_matched_block_location_token_counts("worker-1", token_ids) == {
        CacheLocation.CPU: block_size,
        CacheLocation.NPU: block_size,
    }


def test_same_request_twice_overlap():
    """测试同一个请求发送两次时，第二次应该有overlap"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    test_tokens = [60, 124, 117, 115, 101, 114, 124, 62, 105, 32, 119, 97, 110, 116, 32, 116]

    kv_cache_manager.register_worker("worker-2-decode", "vllm", "Qwen3-8B")

    overlap_scores1 = kv_cache_manager.find_matches(token_ids=test_tokens, model="Qwen3-8B")
    assert overlap_scores1.get("worker-2-decode", 0) == 0, "第一次请求时overlap应为0"

    block_size = kv_cache_manager.block_size
    block_tokens = test_tokens[:block_size]
    block_hash = kv_cache_manager.scorer.compute_block_hash(block_tokens)

    event = CacheEvent(
        event_type="store",
        block_hashes=[block_hash],
        token_ids=block_tokens,
        worker_id="worker-2-decode",
        engine_specific={"worker_id": "worker-2-decode"},
    )

    kv_cache_manager.process_event(event)

    overlap_scores2 = kv_cache_manager.find_matches(token_ids=test_tokens, model="Qwen3-8B")

    assert overlap_scores2.get("worker-2-decode", 0) == block_size, (
        f"第二次请求时worker-2-decode的overlap应为{block_size}"
    )


def test_event_generator_stores_blocks():
    """测试事件生成器能正确存储块"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    test_tokens = [60, 124, 117, 115, 101, 114, 62, 105, 32, 119, 97, 110, 116, 100, 99, 98]

    kv_cache_manager.register_worker("worker-5-decode", "vllm", "Qwen3-8B")

    if kv_cache_manager.event_generator:
        events = kv_cache_manager.event_generator.generate_events("worker-5-decode", test_tokens)

        assert len(events) > 0, "应该生成事件"

        for event in events:
            kv_cache_manager.event_manager.process_event(event)

        overlap_scores = kv_cache_manager.find_matches(token_ids=test_tokens, model="Qwen3-8B")

        logging.info(f"  生成的事件数量: {len(events)}")
        logging.info(
            f"  Worker-5-decode的缓存块数量: {len(kv_cache_manager.index.pod_states['worker-5-decode'].cached_blocks)}"
        )
        logging.info(f"  Overlap scores: {overlap_scores}")

        assert overlap_scores.get("worker-5-decode", 0) >= 1, "生成事件并处理后应该有overlap"
    else:
        logging.info("  跳过（event_generator未启用）")


def test_group_based_routing_with_overlap():
    """测试基于组的路由和overlap"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    test_tokens = [60, 124, 117, 115, 101, 114, 124, 62, 105, 32, 119, 97, 110, 116, 32, 116]

    workers = [
        "worker-1-prefill",
        "worker-2-decode",
        "worker-3-decode",
        "worker-4-prefill",
        "worker-5-decode",
        "worker-6-decode",
    ]

    for worker in workers:
        kv_cache_manager.register_worker(worker, "vllm", "Qwen3-8B")

    block_size = kv_cache_manager.block_size
    block_tokens = test_tokens[:block_size]
    block_hash = kv_cache_manager.scorer.compute_block_hash(block_tokens)

    event1 = CacheEvent(
        event_type="store",
        block_hashes=[block_hash],
        worker_id="worker-2-decode",
        engine_specific={"worker_id": "worker-2-decode"},
        token_ids=block_tokens,
    )

    event2 = CacheEvent(
        event_type="store",
        block_hashes=[block_hash],
        worker_id="worker-5-decode",
        engine_specific={"worker_id": "worker-5-decode"},
        token_ids=block_tokens,
    )

    kv_cache_manager.process_event(event1)
    kv_cache_manager.process_event(event2)

    overlap_scores = kv_cache_manager.find_matches(token_ids=test_tokens, model="Qwen3-8B")

    assert overlap_scores.get("worker-2-decode") == block_size, f"worker-2-decode应该有overlap {block_size}"
    assert overlap_scores.get("worker-5-decode") == block_size, f"worker-5-decode应该有overlap {block_size}"
    assert overlap_scores.get("worker-1-prefill") == 0, "worker-1-prefill不应该有overlap"


if __name__ == "__main__":
    logging.info("=== 运行 KV 缓存重叠检测测试 ===")

    tests = [
        ("初始状态下overlap为0", test_initial_overlap_is_zero),
        ("存储块后能正确检索", test_store_and_retrieve_block),
        ("同一个请求发送两次时，第二次应该有overlap", test_same_request_twice_overlap),
        ("事件生成器能正确存储块", test_event_generator_stores_blocks),
        ("基于组的路由和overlap", test_group_based_routing_with_overlap),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        logging.info(f"测试: {name}")
        if run_test(name, test_func):
            passed += 1
        else:
            failed += 1

    logging.info(f"=== 测试完成: {passed} 通过, {failed} 失败 ===")
