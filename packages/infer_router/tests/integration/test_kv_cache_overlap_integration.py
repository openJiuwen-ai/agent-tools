"""测试KV缓存重叠检测的集成测试"""

import logging
from unittest.mock import MagicMock

import pytest

from openjiuwentools.infer_router.api.server import (
    KVCacheManager,
    KVEventGenerator,
    KVEventManager,
    Router,
    Scheduler,
    WorkerManager,
)
from openjiuwentools.infer_router.preprocess.preprocessor import Preprocessor
from openjiuwentools.infer_router.schemas.agent_hints import ChatCompletionRequest

logger = logging.getLogger(__name__)


class MockWorker:
    """Mock工作器类"""

    def __init__(self, worker_id, url, worker_type):
        self.worker_id = worker_id
        self.url = url
        self.worker_type = worker_type
        self.kv_addr = f"http://{worker_id}:8080"
        self.dp_rank = 0
        self.model = "Qwen3-8B"
        self.engine = "vllm"


@pytest.fixture(scope="class")
def test_env():
    """测试环境fixture"""
    # 初始化组件
    event_manager = KVEventManager()
    event_generator = KVEventGenerator()
    kvcache_manager = KVCacheManager(
        event_manager=event_manager,
        event_generator=event_generator,
        enable_radix_tree=True,
    )
    worker_manager = WorkerManager(kv_cache_manager=kvcache_manager)
    workload_manager = MagicMock()
    scheduler = Scheduler(kvcache_manager)
    router = Router(kvcache_manager, worker_manager)
    preprocessor = Preprocessor()

    # 注册mock工作器（2个group，每个group有1P2D）
    workers = [
        # Group 1
        MockWorker("worker-1-prefill", "http://worker-1:8080", "prefill"),
        MockWorker("worker-2-decode", "http://worker-2:8080", "decode"),
        MockWorker("worker-3-decode", "http://worker-3:8080", "decode"),
        # Group 2
        MockWorker("worker-4-prefill", "http://worker-4:8080", "prefill"),
        MockWorker("worker-5-decode", "http://worker-5:8080", "decode"),
        MockWorker("worker-6-decode", "http://worker-6:8080", "decode"),
    ]

    worker_manager.set_workload_manager(workload_manager)
    worker_manager.register_event_handlers()

    for worker in workers:
        kvcache_manager.register_worker(
            worker_id=worker.worker_id,
            engine_type=worker.engine,
            model=worker.model,
        )

    return {
        "event_manager": event_manager,
        "event_generator": event_generator,
        "kvcache_manager": kvcache_manager,
        "worker_manager": worker_manager,
        "workload_manager": workload_manager,
        "scheduler": scheduler,
        "router": router,
        "preprocessor": preprocessor,
        "workers": workers,
    }


class TestKVOverlapIntegration:
    """集成测试KV缓存重叠检测"""

    @staticmethod
    def _create_chat_request(message="Hello, how are you?"):
        """创建聊天请求"""
        return ChatCompletionRequest(
            model="Qwen3-8B",
            messages=[{"role": "user", "content": message}],
            max_tokens=50,
            stream=False,
        )

    def test_same_request_twice_overlap(self, test_env):
        """测试同一个请求发送两次时，第二次应该有overlap"""
        chat_request = self._create_chat_request()
        kvcache_manager = test_env["kvcache_manager"]
        preprocessor = test_env["preprocessor"]
        event_generator = test_env["event_generator"]
        workers = test_env["workers"]

        route_hint = preprocessor.process(chat_request, None, "test-request-1")

        logger.info(f"Token IDs: {route_hint.token_ids}")
        logger.info(f"Number of tokens: {len(route_hint.token_ids)}")

        # 第一次请求
        overlap_scores1 = kvcache_manager.find_matches(
            token_ids=route_hint.token_ids,
            model=route_hint.model,
        )
        logger.info(f"第一次请求的overlap scores: {overlap_scores1}")

        for worker in workers:
            assert overlap_scores1.get(worker.worker_id, 0) == 0, f"第一次请求时 {worker.worker_id} 的overlap应为0"

        # 模拟处理请求（生成事件并处理）
        if event_generator and route_hint.token_ids:
            events = event_generator.generate_events(
                "worker-2-decode",
                route_hint.token_ids,
            )
            logger.info(f"生成的事件数量: {len(events)}")
            for event in events:
                logger.info(f"  事件类型: {event.event_type}, block_hashes: {event.block_hashes}")
                kvcache_manager.event_manager.process_event(event)

        # 第二次请求
        overlap_scores2 = kvcache_manager.find_matches(
            token_ids=route_hint.token_ids,
            model=route_hint.model,
        )
        logger.info(f"第二次请求的overlap scores: {overlap_scores2}")

        # 验证第二次请求时worker-2-decode有overlap
        assert overlap_scores2.get("worker-2-decode", 0) >= 1, "第二次请求时worker-2-decode的overlap应该大于0"

    def test_group_based_routing_with_overlap(self, test_env):
        """测试基于组的路由和overlap"""
        chat_request = self._create_chat_request()
        kvcache_manager = test_env["kvcache_manager"]
        preprocessor = test_env["preprocessor"]
        event_generator = test_env["event_generator"]

        route_hint = preprocessor.process(chat_request, None, "test-request-2")

        # 为Group 1的decode worker存储缓存
        if event_generator and route_hint.token_ids:
            events = event_generator.generate_events(
                "worker-2-decode",
                route_hint.token_ids,
            )
            for event in events:
                kvcache_manager.event_manager.process_event(event)

        # 为Group 2的decode worker存储缓存
        if event_generator and route_hint.token_ids:
            events = event_generator.generate_events(
                "worker-5-decode",
                route_hint.token_ids,
            )
            for event in events:
                kvcache_manager.event_manager.process_event(event)

        overlap_scores = kvcache_manager.find_matches(
            token_ids=route_hint.token_ids,
            model=route_hint.model,
        )

        logger.info(f"Group路由测试的overlap scores: {overlap_scores}")

        assert overlap_scores.get("worker-2-decode", 0) >= 1, "worker-2-decode应该有overlap"
        assert overlap_scores.get("worker-5-decode", 0) >= 1, "worker-5-decode应该有overlap"
        assert overlap_scores.get("worker-1-prefill", 0) == 0, "worker-1-prefill不应该有overlap"
        assert overlap_scores.get("worker-4-prefill", 0) == 0, "worker-4-prefill不应该有overlap"

    def test_cache_event_flow(self, test_env):
        """测试缓存事件流程"""
        chat_request = self._create_chat_request()
        kvcache_manager = test_env["kvcache_manager"]
        preprocessor = test_env["preprocessor"]
        event_generator = test_env["event_generator"]

        route_hint = preprocessor.process(chat_request, None, "test-request-3")

        logger.info(f"Token IDs长度: {len(route_hint.token_ids)}")

        # 生成并处理缓存事件
        if event_generator and route_hint.token_ids:
            events = event_generator.generate_events(
                "worker-2-decode",
                route_hint.token_ids,
            )
            logger.info(f"生成的事件: {len(events)}")

            for event in events:
                logger.info(f"处理事件: {event.event_type}")
                kvcache_manager.event_manager.process_event(event)

        # 验证缓存状态已更新
        pod_state = kvcache_manager.index.pod_states.get("worker-2-decode")
        assert pod_state is not None, "worker-2-decode应该已注册"
        assert pod_state.radix_tree_root is not None, "radix tree应该已初始化"
