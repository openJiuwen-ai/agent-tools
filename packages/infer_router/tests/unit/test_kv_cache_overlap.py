import logging

from openjiuwentools.infer_router.kv_cache.event import CacheEvent
from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager

logger = logging.getLogger(__name__)


def run_test(name, test_func):
    """运行单个测试"""
    try:
        test_func()
        logger.info(f"✓ {name}")
        return True
    except AssertionError as e:
        logger.error(f"✗ {name}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ {name}: Unexpected error - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_initial_overlap_is_zero():
    """测试初始状态下overlap为0"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    test_tokens = [
        60,
        124,
        117,
        115,
        101,
        114,
        124,
        62,
        105,
        32,
        119,
        97,
        110,
        116,
        32,
        116,
    ]

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
    test_tokens = [
        60,
        124,
        117,
        115,
        101,
        114,
        124,
        62,
        105,
        32,
        119,
        97,
        110,
        116,
        32,
        116,
    ]

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


def test_same_request_twice_overlap():
    """测试同一个请求发送两次时，第二次应该有overlap"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    test_tokens = [
        60,
        124,
        117,
        115,
        101,
        114,
        124,
        62,
        105,
        32,
        119,
        97,
        110,
        116,
        32,
        116,
    ]

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
    test_tokens = [
        60,
        124,
        117,
        115,
        101,
        114,
        62,
        105,
        32,
        119,
        97,
        110,
        116,
        100,
        99,
        98,
    ]

    kv_cache_manager.register_worker("worker-5-decode", "vllm", "Qwen3-8B")

    if kv_cache_manager.event_generator:
        events = kv_cache_manager.event_generator.generate_events("worker-5-decode", test_tokens)

        assert len(events) > 0, "应该生成事件"

        for event in events:
            kv_cache_manager.event_manager.process_event(event)

        overlap_scores = kv_cache_manager.find_matches(token_ids=test_tokens, model="Qwen3-8B")

        logger.info(f"  生成的事件数量: {len(events)}")
        logger.info(
            f"  Worker-5-decode的缓存块数量: {len(kv_cache_manager.index.pod_states['worker-5-decode'].cached_blocks)}"
        )
        logger.info(f"  Overlap scores: {overlap_scores}")

        assert overlap_scores.get("worker-5-decode", 0) >= 1, "生成事件并处理后应该有overlap"
    else:
        logger.info("  跳过（event_generator未启用）")


def test_group_based_routing_with_overlap():
    """测试基于组的路由和overlap"""
    kv_cache_manager = KVCacheManager(enable_radix_tree=True)
    test_tokens = [
        60,
        124,
        117,
        115,
        101,
        114,
        124,
        62,
        105,
        32,
        119,
        97,
        110,
        116,
        32,
        116,
    ]

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
    logger.info("=== 运行 KV 缓存重叠检测测试 ===")
    logger.info("")

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
        logger.info(f"测试: {name}")
        if run_test(name, test_func):
            passed += 1
        else:
            failed += 1
        logger.info("")

    logger.info(f"=== 测试完成: {passed} 通过, {failed} 失败 ===")
