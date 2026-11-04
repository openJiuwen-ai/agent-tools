"""测试 KV 事件功能"""

from openjiuwentools.infer_router.kv_cache import CacheEvent, CacheLocation, KVCacheManager
from openjiuwentools.infer_router.kv_cache.kv_cache import RadixTree


def test_kv_event_manager():
    """测试 KV 事件管理器"""
    kv_cache_manager = KVCacheManager()
    event_manager = kv_cache_manager.event_manager

    # 测试处理 store 事件
    event = CacheEvent(
        event_type="store",
        block_hashes=["test_block"],
        token_ids=[1],
        engine_specific={"worker_id": "test_worker"},
    )
    assert event_manager.process_event(event)

    # 测试处理 removed 事件
    event = CacheEvent(
        event_type="removed",
        block_hashes=["test_block"],
        token_ids=[1],
        engine_specific={"worker_id": "test_worker"},
    )
    assert event_manager.process_event(event)

    # 测试处理 Cleared 事件
    event = CacheEvent(
        event_type="Cleared",
        block_hashes=["test_block"],
        token_ids=[1],
        engine_specific={"worker_id": "test_worker"},
    )
    assert event_manager.process_event(event)

    # 测试处理 evict 事件
    event = CacheEvent(
        event_type="evict",
        block_hashes=["test_block"],
        token_ids=[1],
        engine_specific={"worker_id": "test_worker"},
    )
    assert event_manager.process_event(event)

    # 测试处理 hit 事件
    event = CacheEvent(
        event_type="hit",
        block_hashes=["test_block"],
        token_ids=[1],
        engine_specific={"worker_id": "test_worker"},
    )
    assert event_manager.process_event(event)

    # 测试获取事件统计
    stats = event_manager.get_event_stats()
    assert stats["total"] > 0
    assert stats["cleared"] == 1
    assert "Cleared" not in stats

    # 测试获取最近事件
    recent_events = event_manager.get_recent_events(3)
    assert len(recent_events) <= 3

    # 测试清空事件
    event_manager.clear_events()
    assert len(event_manager.get_recent_events()) == 0


def test_kv_event_generator():
    """测试 KV 事件生成器"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)

    if kv_cache_manager.event_generator:
        # 注册工作器
        kv_cache_manager.register_worker("test_worker", "vllm", "test_model")

        # 测试生成事件（需要至少一个完整的block才能生成store事件）
        block_size = kv_cache_manager.block_size
        tokens = list(range(block_size))
        events = kv_cache_manager.generate_events("test_worker", tokens)
        assert len(events) > 0
        assert events[0].event_type == "store"
        assert events[0].block_hashes
        assert events[0].engine_specific["block_size"] == block_size

        # 测试获取事件生成器统计
        stats = kv_cache_manager.get_event_generator_stats()
        assert "total_workers" in stats
        assert "total_tokens" in stats

        # 测试清空事件
        clear_events = kv_cache_manager.clear_worker_events("test_worker")
        assert len(clear_events) > 0
        assert clear_events[0].event_type == "cleared"
        assert clear_events[0].block_hashes

        # 注销工作器
        kv_cache_manager.unregister_worker("test_worker")


def test_worker_token_manager():
    """测试工作器 Token 管理器"""
    from openjiuwentools.infer_router.kv_cache.worker_token_manager import (
        WorkerTokenManager,
    )

    manager = WorkerTokenManager()

    # 测试添加 token（每次添加都会生成store事件）
    block_size = manager.block_size
    tokens = list(range(block_size))
    events = manager.add_tokens("test_worker", tokens)
    assert len(events) == 1
    assert events[0]["event_type"] == "store"
    assert events[0]["token_count"] == block_size

    # 测试 token 数量
    assert manager.get_worker_token_count("test_worker") == block_size

    # 测试添加新 token（现在每次添加都会生成事件）
    events = manager.add_tokens("test_worker", [block_size])
    assert len(events) == 1  # 每次添加都会生成事件
    assert events[0]["event_type"] == "store"
    assert events[0]["token_count"] == 1

    # 添加更多 token
    more_tokens = list(range(block_size + 1, block_size * 2))
    events = manager.add_tokens("test_worker", more_tokens)
    assert len(events) == 1
    assert events[0]["event_type"] == "store"
    assert events[0]["token_count"] == len(more_tokens)
    assert manager.get_worker_token_count("test_worker") == block_size * 2  # 16 + 1 + 15 = 32

    # 测试获取 token 列表
    worker_tokens = manager.get_worker_tokens("test_worker")
    assert len(worker_tokens) == block_size * 2

    # 测试检查 token
    assert manager.has_token("test_worker", 0)
    assert not manager.has_token("test_worker", block_size * 2 + 1)

    # 测试清空 token
    events = manager.clear_worker_tokens("test_worker")
    assert len(events) == 1
    assert events[0]["event_type"] == "cleared"
    assert manager.get_worker_token_count("test_worker") == 0

    # 测试移除工作器
    manager.add_tokens("test_worker", [1, 2, 3])
    assert manager.get_worker_token_count("test_worker") == 3
    manager.remove_worker("test_worker")
    assert manager.get_worker_token_count("test_worker") == 0


def test_kv_cache_event_integration():
    """测试 KV 缓存事件集成"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)

    # 注册工作器
    kv_cache_manager.register_worker("test_worker", "vllm", "test_model")

    # 生成事件（需要至少一个完整的block才能生成store事件）
    block_size = kv_cache_manager.block_size
    tokens = list(range(block_size))
    events = kv_cache_manager.generate_events("test_worker", tokens)
    assert len(events) > 0

    # 获取缓存统计
    stats = kv_cache_manager.get_cache_stats()
    assert "event_stats" in stats
    assert "event_generator_stats" in stats

    # 清空工作器事件
    clear_events = kv_cache_manager.clear_worker_events("test_worker")
    assert len(clear_events) > 0

    # 注销工作器
    kv_cache_manager.unregister_worker("test_worker")


def test_store_updates_all_cache_indexes():
    """store 事件应同时更新 radix、block 元数据和 prefix 索引。"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    kv_cache_manager.register_worker("worker-1", "vllm", "test_model")
    tokens = list(range(kv_cache_manager.block_size))
    block_hash = kv_cache_manager.scorer.compute_block_hash(tokens)

    event = CacheEvent(
        event_type="store",
        block_hashes=[block_hash],
        token_ids=tokens,
        worker_id="worker-1",
        location=CacheLocation.NPU,
        engine_specific={"worker_id": "worker-1", "block_size": kv_cache_manager.block_size},
    )

    assert kv_cache_manager.process_event(event)

    pod_state = kv_cache_manager.index.pod_states["worker-1"]
    assert block_hash in pod_state.cached_blocks
    assert block_hash in pod_state.block_node_map
    assert "worker-1" in kv_cache_manager.index.prefix_index[block_hash]
    assert kv_cache_manager.find_matches(tokens, "test_model")["worker-1"] == len(tokens)


def test_removed_event_invalidates_matches_and_indexes():
    """removed/evict 事件应让已删除 block 不再参与 overlap。"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    kv_cache_manager.register_worker("worker-1", "vllm", "test_model")
    tokens = list(range(kv_cache_manager.block_size))
    block_hash = kv_cache_manager.scorer.compute_block_hash(tokens)

    kv_cache_manager.process_event(
        CacheEvent(
            event_type="store",
            block_hashes=[block_hash],
            token_ids=tokens,
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1", "block_size": kv_cache_manager.block_size},
        )
    )
    kv_cache_manager.process_event(
        CacheEvent(
            event_type="removed",
            block_hashes=[block_hash],
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1", "block_size": kv_cache_manager.block_size},
        )
    )

    pod_state = kv_cache_manager.index.pod_states["worker-1"]
    assert block_hash not in pod_state.cached_blocks
    assert block_hash not in pod_state.block_node_map
    assert block_hash not in kv_cache_manager.index.prefix_index
    assert kv_cache_manager.find_matches(tokens, "test_model")["worker-1"] == 0


def test_hit_updates_access_metadata():
    """hit 事件应更新 block 和 radix node 的访问信息。"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    kv_cache_manager.register_worker("worker-1", "vllm", "test_model")
    tokens = list(range(kv_cache_manager.block_size))
    block_hash = kv_cache_manager.scorer.compute_block_hash(tokens)
    kv_cache_manager.process_event(
        CacheEvent(
            event_type="store",
            block_hashes=[block_hash],
            token_ids=tokens,
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1", "block_size": kv_cache_manager.block_size},
        )
    )

    pod_state = kv_cache_manager.index.pod_states["worker-1"]
    before_ref = pod_state.cached_blocks[block_hash].ref_count
    before_access = pod_state.cached_blocks[block_hash].last_accessed
    kv_cache_manager.process_event(
        CacheEvent(
            event_type="hit",
            block_hashes=[block_hash],
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1", "block_size": kv_cache_manager.block_size},
        )
    )

    assert pod_state.cached_blocks[block_hash].ref_count == before_ref + 1
    assert pod_state.cached_blocks[block_hash].last_accessed >= before_access


def test_cleared_event_resets_worker_cache_state():
    """clear 事件应清空 block 元数据、映射和 radix tree。"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    kv_cache_manager.register_worker("worker-1", "vllm", "test_model")
    tokens = list(range(kv_cache_manager.block_size))
    block_hash = kv_cache_manager.scorer.compute_block_hash(tokens)
    kv_cache_manager.process_event(
        CacheEvent(
            event_type="store",
            block_hashes=[block_hash],
            token_ids=tokens,
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1", "block_size": kv_cache_manager.block_size},
        )
    )

    kv_cache_manager.process_event(
        CacheEvent(
            event_type="cleared",
            worker_id="worker-1",
            engine_specific={"worker_id": "worker-1"},
        )
    )

    pod_state = kv_cache_manager.index.pod_states["worker-1"]
    assert pod_state.cached_blocks == {}
    assert pod_state.block_node_map == {}
    assert pod_state.radix_tree_root.children == {}
    assert kv_cache_manager.find_matches(tokens, "test_model")["worker-1"] == 0


def test_move_event_updates_location_without_invalidating_match():
    """move/transfer 事件只更新 tier，不应删除可复用 cache。"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    kv_cache_manager.register_worker("worker-1", "vllm", "test_model")
    tokens = list(range(kv_cache_manager.block_size))
    block_hash = kv_cache_manager.scorer.compute_block_hash(tokens)
    kv_cache_manager.process_event(
        CacheEvent(
            event_type="store",
            block_hashes=[block_hash],
            token_ids=tokens,
            worker_id="worker-1",
            location=CacheLocation.NPU,
            engine_specific={"worker_id": "worker-1", "block_size": kv_cache_manager.block_size},
        )
    )

    kv_cache_manager.process_event(
        CacheEvent(
            event_type="move",
            block_hashes=[block_hash],
            worker_id="worker-1",
            location=CacheLocation.CPU,
            engine_specific={"worker_id": "worker-1", "block_size": kv_cache_manager.block_size},
        )
    )

    pod_state = kv_cache_manager.index.pod_states["worker-1"]
    assert pod_state.cached_blocks[block_hash].location == CacheLocation.CPU
    assert kv_cache_manager.find_matches(tokens, "test_model")["worker-1"] == len(tokens)


def test_vllm_block_removed_invalidates_prefix_by_hash():
    """BlockRemoved 未携带 token_ids 时，也应按照 block hash 失效路由前缀。"""
    from openjiuwentools.infer_router.kv_cache.kv_event_handler import KVEventHandler

    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    kv_cache_manager.register_worker("prefill-0", "vllm", "test_model")
    handler = KVEventHandler(kv_cache_manager.event_manager)

    tokens = list(range(kv_cache_manager.block_size))
    block_hash = "vllm-block-hash"
    handler(
        "prefill-0",
        {
            "events": [
                {
                    "type": "BlockStored",
                    "data": {
                        "block_hashes": [block_hash],
                        "token_ids": tokens,
                        "block_size": kv_cache_manager.block_size,
                    },
                }
            ]
        },
    )
    assert kv_cache_manager.find_matches(tokens, "test_model")["prefill-0"] == len(tokens)

    handler(
        "prefill-0",
        {
            "events": [
                {
                    "type": "BlockRemoved",
                    "data": {"block_hashes": [block_hash]},
                }
            ]
        },
    )
    assert kv_cache_manager.find_matches(tokens, "test_model")["prefill-0"] == 0


def test_prefix_matching_is_safe_during_block_event_updates():
    """路由读取KV索引时，BlockStored/BlockRemoved并发更新不应抛出异常。"""
    from concurrent.futures import ThreadPoolExecutor

    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    kv_cache_manager.register_worker("prefill-0", "vllm", "test_model")
    tokens = list(range(kv_cache_manager.block_size))
    block_hash = "concurrent-block-hash"

    store_event = CacheEvent(
        event_type="store",
        block_hashes=[block_hash],
        token_ids=tokens,
        engine_specific={
            "worker_id": "prefill-0",
            "block_size": kv_cache_manager.block_size,
        },
    )
    removed_event = CacheEvent(
        event_type="removed",
        block_hashes=[block_hash],
        engine_specific={
            "worker_id": "prefill-0",
            "block_size": kv_cache_manager.block_size,
        },
    )

    def update_cache_state():
        for _ in range(500):
            kv_cache_manager.process_event(store_event)
            kv_cache_manager.process_event(removed_event)

    def match_prefix():
        for _ in range(1000):
            matched = kv_cache_manager.find_matches(tokens, "test_model")["prefill-0"]
            assert matched in {0, len(tokens)}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(update_cache_state)]
        futures.extend(executor.submit(match_prefix) for _ in range(3))
        for future in futures:
            future.result()


def test_worker_event_handler_updates_cache_state_once():
    """worker事件只通过event_manager更新一次缓存索引。"""
    from unittest.mock import Mock

    from openjiuwentools.infer_router.kv_cache.kv_event_handler import KVEventHandler

    event_manager = Mock()
    handler = KVEventHandler(event_manager)

    handler(
        "prefill-0",
        {
            "events": [
                {
                    "type": "BlockStored",
                    "data": {
                        "block_hashes": ["block-0"],
                        "token_ids": list(range(16)),
                        "block_size": 16,
                    },
                }
            ]
        },
    )

    event_manager.process_event.assert_called_once()


def test_worker_event_handler_preserves_physical_block_ids():
    """Worker location 事件中的物理 block id 应完整进入 CacheEvent。"""
    from unittest.mock import Mock

    from openjiuwentools.infer_router.kv_cache.kv_event_handler import KVEventHandler

    event_manager = Mock()
    handler = KVEventHandler(event_manager)
    handler(
        "worker-0",
        {
            "events": [
                {
                    "type": "BlockStored",
                    "data": {
                        "block_hashes": ["hash-0", "hash-1"],
                        "physical_block_ids": [7, 8],
                        "block_size": 16,
                    },
                }
            ]
        },
    )

    event = event_manager.process_event.call_args.args[0]
    assert event.engine_specific["physical_block_ids"] == [7, 8]


def test_cross_worker_transfer_candidate_uses_npu_physical_blocks():
    """跨 Worker 候选应返回同模型源端的 NPU 物理 block 映射。"""
    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("source", "vllm", "model-a")
    manager.register_worker("target", "vllm", "model-a")
    token_ids = list(range(manager.block_size * 2))
    manager.process_event(
        CacheEvent(
            event_type="store",
            worker_id="source",
            block_hashes=["hash-0", "hash-1"],
            token_ids=token_ids,
            engine_specific={
                "worker_id": "source",
                "block_size": manager.block_size,
                "physical_block_ids": [17, 18],
            },
        )
    )

    candidate = manager.get_cross_worker_transfer_candidate(
        "target",
        token_ids,
        candidate_worker_ids=["source"],
        limit=1,
    )

    assert candidate is not None
    assert candidate.source_worker_id == "source"
    assert candidate.block_hashes == ("hash-0",)
    assert candidate.source_block_ids == (17,)
    assert candidate.source_matched_blocks == 2
    assert candidate.source_npu_blocks == 2
    assert candidate.physical_mapped_npu_blocks == 2
    assert candidate.target_matched_blocks == 0
    assert candidate.target_npu_blocks == 0
    assert candidate.first_unselected_reason == "transfer_limit_reached"


def test_cross_worker_transfer_candidate_reports_source_prefix_exhausted():
    """候选少于配置上限时应明确指出源端连续前缀已经耗尽。"""
    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("source", "vllm", "model-a")
    manager.register_worker("target", "vllm", "model-a")
    token_ids = list(range(manager.block_size))
    manager.process_event(
        CacheEvent(
            event_type="store",
            worker_id="source",
            block_hashes=["hash-0"],
            token_ids=token_ids,
            engine_specific={
                "worker_id": "source",
                "block_size": manager.block_size,
                "physical_block_ids": [17],
            },
        )
    )

    candidate = manager.get_cross_worker_transfer_candidate(
        "target",
        token_ids,
        candidate_worker_ids=["source"],
        limit=8,
    )

    assert candidate is not None
    assert candidate.block_hashes == ("hash-0",)
    assert candidate.source_matched_blocks == 1
    assert candidate.first_unselected_reason == "source_prefix_exhausted"


def test_record_cross_worker_transfer_updates_target_prefix_immediately():
    """传输发布后 Router 应立即让目标前缀可命中，避免下一请求重复传输。"""
    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("source", "vllm", "model-a")
    manager.register_worker("target", "vllm", "model-a")
    token_ids = list(range(manager.block_size * 2))
    manager.process_event(
        CacheEvent(
            event_type="store",
            worker_id="source",
            block_hashes=["hash-0", "hash-1"],
            token_ids=token_ids,
            engine_specific={
                "worker_id": "source",
                "block_size": manager.block_size,
                "physical_block_ids": [17, 18],
            },
        )
    )

    updated = manager.record_cross_worker_transfer(
        "source",
        "target",
        ["hash-0", "hash-1"],
        [70, 71],
    )

    assert updated == 2
    assert manager.get_matched_block_hashes(
        "target",
        token_ids,
        locations={CacheLocation.NPU},
    ) == ["hash-0", "hash-1"]
    target_state = manager.index.pod_states["target"]
    assert target_state.cached_blocks["hash-0"].physical_block_id == 70
    assert target_state.cached_blocks["hash-1"].physical_block_id == 71
    assert (
        manager.get_cross_worker_transfer_candidate(
            "target",
            token_ids,
            candidate_worker_ids=["source"],
        )
        is None
    )


def test_record_cross_worker_transfer_replays_long_prefix_once(monkeypatch):
    """批量发布长前缀时只应向目标 RadixTree 插入一次最长 token 路径。

    参数:
        monkeypatch: pytest 方法替换工具，用于统计 RadixTree.insert 调用。

    返回:
        无返回值；重复重放前缀或目标索引不完整时由断言报告失败。
    """

    manager = KVCacheManager(block_size=128, enable_radix_tree=True)
    manager.register_worker("source", "vllm", "model-a")
    manager.register_worker("target", "vllm", "model-a")
    block_count = 40
    token_ids = list(range(manager.block_size * block_count))
    block_hashes = [f"hash-{index}" for index in range(block_count)]
    manager.process_event(
        CacheEvent(
            event_type="store",
            worker_id="source",
            block_hashes=block_hashes,
            token_ids=token_ids,
            engine_specific={
                "worker_id": "source",
                "block_size": manager.block_size,
                "physical_block_ids": list(range(100, 100 + block_count)),
            },
        )
    )

    original_insert = RadixTree.insert
    inserted_lengths: list[int] = []

    def counted_insert(self, values, start_node=None):
        inserted_lengths.append(len(values))
        return original_insert(self, values, start_node=start_node)

    monkeypatch.setattr(RadixTree, "insert", counted_insert)
    timing: dict[str, float] = {}

    updated = manager.record_cross_worker_transfer(
        "source",
        "target",
        block_hashes,
        list(range(200, 200 + block_count)),
        timing=timing,
    )

    assert updated == block_count
    assert inserted_lengths == [len(token_ids)]
    assert timing["router_index_lock_wait_ms"] >= 0.0
    assert timing["router_index_update_ms"] >= 0.0
    assert manager.get_matched_block_hashes(
        "target",
        token_ids,
        locations={CacheLocation.NPU},
    ) == block_hashes


def test_record_pd_import_publishes_target_prefix_without_decode_store_event():
    """D attach成功后应不依赖D普通Store事件即可让Router命中完整前缀。"""
    manager = KVCacheManager(block_size=128, enable_radix_tree=True)
    manager.register_worker("decode", "vllm", "model-a")
    token_ids = list(range(128 * 2 + 28))

    published = manager.record_pd_import(
        "decode",
        ["hash-0", "hash-1"],
        [70, 71],
        token_ids,
        block_size=128,
    )

    assert published == 2
    assert manager.get_matched_block_hashes(
        "decode",
        token_ids,
        locations={CacheLocation.NPU},
    ) == ["hash-0", "hash-1"]
    state = manager.index.pod_states["decode"]
    assert state.cached_blocks["hash-0"].physical_block_id == 70
    assert state.cached_blocks["hash-1"].physical_block_id == 71
    ref_count_before_repeat = state.cached_blocks["hash-0"].ref_count

    # 重复回调只校正位置与物理ID，不增加引用计数或生成重复节点。
    assert manager.record_pd_import(
        "decode",
        ["hash-0", "hash-1"],
        [70, 71],
        token_ids,
        block_size=128,
    ) == 2
    assert state.cached_blocks["hash-0"].ref_count == ref_count_before_repeat


def test_cross_worker_candidate_aligns_existing_blocks_by_prefix_position():
    """相同 token 位置的跨进程 hash 不同时，不应重复传输目标已有的前缀块。"""
    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("source", "vllm", "model-a")
    manager.register_worker("target", "vllm", "model-a")
    token_ids = list(range(manager.block_size * 3))
    manager.process_event(
        CacheEvent(
            event_type="store",
            worker_id="source",
            block_hashes=["source-0", "source-1", "source-2"],
            token_ids=token_ids,
            engine_specific={
                "worker_id": "source",
                "block_size": manager.block_size,
                "physical_block_ids": [17, 18, 19],
            },
        )
    )
    manager.process_event(
        CacheEvent(
            event_type="store",
            worker_id="target",
            block_hashes=["target-0", "target-1"],
            token_ids=token_ids[: manager.block_size * 2],
            engine_specific={
                "worker_id": "target",
                "block_size": manager.block_size,
                "physical_block_ids": [70, 71],
            },
        )
    )

    candidate = manager.get_cross_worker_transfer_candidate(
        "target",
        token_ids,
        candidate_worker_ids=["source"],
        limit=8,
    )

    assert candidate is not None
    assert candidate.block_hashes == ("source-2",)
    assert candidate.source_block_ids == (19,)
    assert candidate.target_matched_blocks == 2


def test_cross_worker_candidate_skips_transfer_when_target_prefix_is_complete():
    """目标已有完整 token 前缀时，即使两端 hash 不同也不应创建传输候选。"""
    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("source", "vllm", "model-a")
    manager.register_worker("target", "vllm", "model-a")
    token_ids = list(range(manager.block_size * 2))
    for worker_id, hash_prefix, block_ids in (
        ("source", "source", [17, 18]),
        ("target", "target", [70, 71]),
    ):
        manager.process_event(
            CacheEvent(
                event_type="store",
                worker_id=worker_id,
                block_hashes=[f"{hash_prefix}-0", f"{hash_prefix}-1"],
                token_ids=token_ids,
                engine_specific={
                    "worker_id": worker_id,
                    "block_size": manager.block_size,
                    "physical_block_ids": block_ids,
                },
            )
        )

    candidate = manager.get_cross_worker_transfer_candidate(
        "target",
        token_ids,
        candidate_worker_ids=["source"],
        limit=8,
    )

    assert candidate is None


def test_cross_worker_plan_scans_each_worker_prefix_once(monkeypatch):
    """完整公共前缀无须传输时，每个 Worker 的 token 路径只能扫描一次。

    参数:
        monkeypatch: pytest 提供的方法替换工具，用于统计快照构建次数。

    返回:
        无返回值；重复扫描或错误生成候选时由断言报告失败。
    """

    manager = KVCacheManager(enable_radix_tree=True)
    token_ids = list(range(manager.block_size * 4))
    for worker_index in range(4):
        worker_id = f"worker-{worker_index}"
        manager.register_worker(worker_id, "vllm", "model-a")
        manager.process_event(
            CacheEvent(
                event_type="store",
                worker_id=worker_id,
                block_hashes=[
                    f"{worker_id}-hash-{block_index}"
                    for block_index in range(4)
                ],
                token_ids=token_ids,
                engine_specific={
                    "worker_id": worker_id,
                    "block_size": manager.block_size,
                    "physical_block_ids": [
                        worker_index * 10 + block_index
                        for block_index in range(4)
                    ],
                },
            )
        )

    original = manager.get_worker_prefix_snapshot
    snapshot_calls: list[str] = []

    def tracked_snapshot(worker_id: str, request_token_ids: list[int]):
        snapshot_calls.append(worker_id)
        return original(worker_id, request_token_ids)

    monkeypatch.setattr(manager, "get_worker_prefix_snapshot", tracked_snapshot)

    plan = manager.plan_cross_worker_transfer(
        "worker-0",
        token_ids,
        candidate_worker_ids=["worker-1", "worker-2", "worker-3"],
    )

    assert plan.candidate is None
    assert plan.diagnostics["target_matched_blocks"] == 4
    assert sorted(snapshot_calls) == [
        "worker-0",
        "worker-1",
        "worker-2",
        "worker-3",
    ]


def test_cross_worker_candidate_counts_hash_aliases_once_per_prefix_position():
    """同一 token 边界存在跨进程 hash 别名时，每个逻辑 block 只能计数一次。"""
    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("source", "vllm", "model-a")
    manager.register_worker("target", "vllm", "model-a")
    token_ids = list(range(manager.block_size * 2))
    for block_hashes, block_ids in (
        (["transferred-0", "transferred-1"], [17, 18]),
        (["local-0", "local-1"], [59, 60]),
    ):
        manager.process_event(
            CacheEvent(
                event_type="store",
                worker_id="source",
                block_hashes=block_hashes,
                token_ids=token_ids,
                engine_specific={
                    "worker_id": "source",
                    "block_size": manager.block_size,
                    "physical_block_ids": block_ids,
                },
            )
        )
    manager.process_event(
        CacheEvent(
            event_type="store",
            worker_id="target",
            block_hashes=["target-0", "target-1"],
            token_ids=token_ids,
            engine_specific={
                "worker_id": "target",
                "block_size": manager.block_size,
                "physical_block_ids": [70, 71],
            },
        )
    )

    assert manager.get_matched_block_hashes("source", token_ids) == [
        "local-0",
        "local-1",
    ]
    assert manager.get_matched_block_location_counts("source", token_ids) == {
        CacheLocation.NPU: 2
    }
    assert (
        manager.get_cross_worker_transfer_candidate(
            "target",
            token_ids,
            candidate_worker_ids=["source"],
            limit=8,
        )
        is None
    )


def test_cross_worker_transfer_diagnostics_reports_missing_physical_mapping():
    """候选为空时应明确指出源 NPU block 缺少物理编号。"""
    manager = KVCacheManager(enable_radix_tree=True)
    manager.register_worker("source", "vllm", "model-a")
    manager.register_worker("target", "vllm", "model-a")
    token_ids = list(range(manager.block_size))
    manager.process_event(
        CacheEvent(
            event_type="store",
            worker_id="source",
            block_hashes=["hash-0"],
            token_ids=token_ids,
            engine_specific={
                "worker_id": "source",
                "block_size": manager.block_size,
            },
        )
    )

    diagnostics = manager.diagnose_cross_worker_transfer(
        "target",
        token_ids,
        candidate_worker_ids=["source"],
    )

    assert diagnostics["sources"][0]["matched_blocks"] == 1
    assert diagnostics["sources"][0]["npu_blocks"] == 1
    assert diagnostics["sources"][0]["physical_mapped_npu_blocks"] == 0
    assert diagnostics["sources"][0]["first_blocker"] == "missing_physical_block_id"


def test_worker_reported_block_size_corrects_fallback_radix_boundary():
    """真实尺寸事件应修正原生事件按 Router 默认值建立的错误 block 边界。

    返回:
        无返回值；旧节点未清理或候选无法命中时由断言报告失败。
    """

    manager = KVCacheManager(block_size=16, enable_radix_tree=True)
    manager.register_worker("source", "vllm", "model-a")
    manager.register_worker("target", "vllm", "model-a")
    token_ids = list(range(128))

    manager.process_event(
        CacheEvent(
            event_type="store",
            worker_id="source",
            block_hashes=["hash-128"],
            token_ids=token_ids,
            engine_specific={"worker_id": "source", "block_size": 16},
        )
    )
    source_state = manager.index.pod_states["source"]
    wrong_node = source_state.radix_tree_root
    for token_id in token_ids[:16]:
        wrong_node = wrong_node.children[token_id]
    assert "hash-128" in wrong_node.block_ids

    manager.process_event(
        CacheEvent(
            event_type="store",
            worker_id="source",
            block_hashes=["hash-128"],
            token_ids=token_ids,
            engine_specific={
                "worker_id": "source",
                "block_size": 128,
                "physical_block_ids": [9],
            },
        )
    )

    assert "hash-128" not in wrong_node.block_ids
    assert source_state.block_node_map["hash-128"].get_token_ids() == token_ids
    candidate = manager.get_cross_worker_transfer_candidate(
        "target",
        token_ids,
        candidate_worker_ids=["source"],
    )
    assert candidate is not None
    assert candidate.block_hashes == ("hash-128",)
    assert candidate.source_block_ids == (9,)
