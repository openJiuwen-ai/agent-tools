"""测试Prefill worker的KV缓存重叠检测"""

import logging

from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.kv_cache.kv_event_generator import KVEventGenerator
from openjiuwentools.infer_router.kv_cache.kv_event_manager import KVEventManager
from openjiuwentools.infer_router.preprocess.preprocessor import Preprocessor
from openjiuwentools.infer_router.schemas.agent_hints import ChatCompletionRequest

logger = logging.getLogger(__name__)


def test_prefill_overlap():
    """测试prefill worker的overlap"""
    logger.info("=== 测试Prefill Worker Overlap ===")
    logger.info("")

    # 初始化组件
    event_manager = KVEventManager()
    event_generator = KVEventGenerator()
    kvcache_manager = KVCacheManager(
        event_manager=event_manager,
        event_generator=event_generator,
        enable_radix_tree=True,
    )

    # 注册工作器（包括prefill和decode）
    workers = [
        ("worker-1-prefill", "prefill"),
        ("worker-2-decode", "decode"),
        ("worker-3-decode", "decode"),
        ("worker-4-prefill", "prefill"),
        ("worker-5-decode", "decode"),
        ("worker-6-decode", "decode"),
    ]

    for worker_id, worker_type in workers:
        kvcache_manager.register_worker(worker_id, "vllm", "Qwen3-8B")
        logger.info(f"注册工作器: {worker_id} ({worker_type})")

    logger.info("")

    # 创建测试请求
    chat_request = ChatCompletionRequest(
        model="Qwen3-8B",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=50,
        stream=False,
    )

    preprocessor = Preprocessor()
    route_hint = preprocessor.process(chat_request, None, "test-request-1")

    logger.info(f"Token IDs: {route_hint.token_ids}")
    logger.info(f"Token数量: {len(route_hint.token_ids)}")
    logger.info(f"Block大小: {kvcache_manager.block_size}")
    logger.info("")

    # 第一次请求 - 所有worker的overlap应该为0
    logger.info("=== 第一次请求 ===")
    overlap_scores1 = kvcache_manager.find_matches(
        token_ids=route_hint.token_ids,
        model=route_hint.model,
    )
    logger.info(f"Overlap scores: {overlap_scores1}")

    for worker_id, _worker_type in workers:
        assert overlap_scores1.get(worker_id, 0) == 0, f"第一次请求时 {worker_id} 的overlap应为0"
    logger.info("✓ 第一次请求所有worker的overlap都是0")
    logger.info("")

    # 模拟存储缓存到prefill worker
    logger.info("=== 存储缓存到prefill worker ===")
    if event_generator and route_hint.token_ids:
        events = event_generator.generate_events(
            "worker-1-prefill",  # 注意：这里存储到prefill worker
            route_hint.token_ids,
        )
        logger.info(f"生成的事件数量: {len(events)}")
        for event in events:
            logger.info(f"  事件类型: {event.event_type}")
            logger.info(f"  Worker ID: {event.worker_id}")
            logger.info(f"  Block hashes: {event.block_hashes}")
            kvcache_manager.event_manager.process_event(event)

    # 第二次请求 - prefill worker应该有overlap
    logger.info("")
    logger.info("=== 第二次请求 ===")
    overlap_scores2 = kvcache_manager.find_matches(
        token_ids=route_hint.token_ids,
        model=route_hint.model,
    )
    logger.info(f"Overlap scores: {overlap_scores2}")

    # 检查prefill worker的overlap
    prefill_workers_with_cache = ["worker-1-prefill"]  # 只有这个worker存储了缓存
    prefill_workers_without_cache = ["worker-4-prefill"]  # 这个worker没有存储缓存
    decode_workers = [
        "worker-2-decode",
        "worker-3-decode",
        "worker-5-decode",
        "worker-6-decode",
    ]

    logger.info("")
    logger.info("=== 结果分析 ===")
    for worker in prefill_workers_with_cache:
        score = overlap_scores2.get(worker, 0)
        if score > 0:
            logger.info(f"✓ {worker}: overlap={score} (预期)")
        else:
            logger.error(f"✗ {worker}: overlap={score} (应该大于0)")

    for worker in prefill_workers_without_cache:
        score = overlap_scores2.get(worker, 0)
        if score == 0:
            logger.info(f"✓ {worker}: overlap={score} (预期 - 没有存储缓存)")
        else:
            logger.error(f"✗ {worker}: overlap={score} (应该为0)")

    for worker in decode_workers:
        score = overlap_scores2.get(worker, 0)
        if score == 0:
            logger.info(f"✓ {worker}: overlap={score} (预期)")
        else:
            logger.error(f"✗ {worker}: overlap={score} (应该为0)")

    # 验证结果
    assert overlap_scores2.get("worker-1-prefill", 0) > 0, "worker-1-prefill应该有overlap"


if __name__ == "__main__":
    test_prefill_overlap()
