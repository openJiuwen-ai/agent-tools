"""测试 KV 事件功能"""

from openjiuwentools.infer_router.kv_cache import CacheEvent, KVCacheManager


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

        # 测试获取事件生成器统计
        stats = kv_cache_manager.get_event_generator_stats()
        assert "total_workers" in stats
        assert "total_tokens" in stats

        # 测试清空事件
        clear_events = kv_cache_manager.clear_worker_events("test_worker")
        assert len(clear_events) > 0
        assert clear_events[0].event_type == "Cleared"

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
    assert events[0]["event_type"] == "Cleared"
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
