"""测试KV缓存重叠检测的集成测试"""

import logging
from unittest.mock import MagicMock

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

    worker_id: str
    url: str
    worker_type: str
    kv_addr: str
    dp_rank: int
    model: str
    engine: str

    def __init__(self, worker_id: str, url: str, worker_type: str):
        self.worker_id = worker_id
        self.url = url
        self.worker_type = worker_type
        self.kv_addr = f"http://{worker_id}:8080"
        self.dp_rank = 0
        self.model = "Qwen3-8B"
        self.engine = "vllm"


class TestKVOverlapIntegration:
    """集成测试KV缓存重叠检测"""

    event_manager: KVEventManager | None = None
    event_generator: KVEventGenerator | None = None
    kvcache_manager: KVCacheManager | None = None
    worker_manager: WorkerManager | None = None
    workload_manager: MagicMock | None = None
    scheduler: Scheduler | None = None
    router: Router | None = None
    preprocessor: Preprocessor | None = None
    workers: list | None = None

    def setup_method(self):
        """设置测试环境"""
        # 初始化组件
        self.event_manager = KVEventManager()
        self.event_generator = KVEventGenerator()
        self.kvcache_manager = KVCacheManager(
            event_manager=self.event_manager,
            event_generator=self.event_generator,
            enable_radix_tree=True,
        )
        self.worker_manager = WorkerManager(kv_cache_manager=self.kvcache_manager)
        self.workload_manager = MagicMock()
        self.worker_manager.set_workload_manager(self.workload_manager)
        self.worker_manager.register_event_handlers()

        self.scheduler = Scheduler(self.kvcache_manager)
        self.router = Router(self.kvcache_manager, self.worker_manager)
        self.preprocessor = Preprocessor()

        # 注册mock工作器（2个group，每个group有1P2D）
        self.workers = [
            # Group 1
            MockWorker("worker-1-prefill", "http://worker-1:8080", "prefill"),
            MockWorker("worker-2-decode", "http://worker-2:8080", "decode"),
            MockWorker("worker-3-decode", "http://worker-3:8080", "decode"),
            # Group 2
            MockWorker("worker-4-prefill", "http://worker-4:8080", "prefill"),
            MockWorker("worker-5-decode", "http://worker-5:8080", "decode"),
            MockWorker("worker-6-decode", "http://worker-6:8080", "decode"),
        ]

        for worker in self.workers:
            self.kvcache_manager.register_worker(
                worker_id=worker.worker_id,
                engine_type=worker.engine,
                model=worker.model,
            )

    @staticmethod
    def _create_chat_request(message="Hello, how are you?"):
        """创建聊天请求"""
        return ChatCompletionRequest(
            model="Qwen3-8B",
            messages=[{"role": "user", "content": message}],
            max_tokens=50,
            stream=False,
        )

    def test_same_request_twice_overlap(self):
        """测试同一个请求发送两次时，第二次应该有overlap"""
        chat_request = self._create_chat_request()

        route_hint = self.preprocessor.process(chat_request, None, "test-request-1")

        logging.info(f"Token IDs: {route_hint.token_ids}")
        logging.info(f"Number of tokens: {len(route_hint.token_ids)}")

        # 第一次请求
        overlap_scores1 = self.kvcache_manager.find_matches(
            token_ids=route_hint.token_ids,
            model=route_hint.model,
        )
        logging.info(f"第一次请求的overlap scores: {overlap_scores1}")

        for worker in self.workers:
            assert overlap_scores1.get(worker.worker_id, 0) == 0, (
                f"第一次请求时 {worker.worker_id} 的overlap应为0"
            )

        # 模拟处理请求（生成事件并处理）
        if self.event_generator and route_hint.token_ids:
            events = self.event_generator.generate_events(
                "worker-2-decode",
                route_hint.token_ids,
            )
            logging.info(f"生成的事件数量: {len(events)}")
            for event in events:
                logging.info(f"  事件类型: {event.event_type}, block_hashes: {event.block_hashes}")
                self.kvcache_manager.event_manager.process_event(event)

        # 第二次请求
        overlap_scores2 = self.kvcache_manager.find_matches(
            token_ids=route_hint.token_ids,
            model=route_hint.model,
        )
        logging.info(f"第二次请求的overlap scores: {overlap_scores2}")

        # 验证第二次请求时worker-2-decode有overlap
        assert overlap_scores2.get("worker-2-decode", 0) >= 1, (
            "第二次请求时worker-2-decode的overlap应该大于0"
        )

    def test_group_based_routing_with_overlap(self):
        """测试基于组的路由和overlap"""
        chat_request = self._create_chat_request()
        route_hint = self.preprocessor.process(chat_request, None, "test-request-2")

        # 为Group 1的decode worker存储缓存
        if self.event_generator and route_hint.token_ids:
            events = self.event_generator.generate_events(
                "worker-2-decode",
                route_hint.token_ids,
            )
            for event in events:
                self.kvcache_manager.event_manager.process_event(event)

        # 为Group 2的decode worker存储缓存
        if self.event_generator and route_hint.token_ids:
            events = self.event_generator.generate_events(
                "worker-5-decode",
                route_hint.token_ids,
            )
            for event in events:
                self.kvcache_manager.event_manager.process_event(event)

        overlap_scores = self.kvcache_manager.find_matches(
            token_ids=route_hint.token_ids,
            model=route_hint.model,
        )

        logging.info(f"Group路由测试的overlap scores: {overlap_scores}")

        assert overlap_scores.get("worker-2-decode", 0) >= 1, "worker-2-decode应该有overlap"
        assert overlap_scores.get("worker-5-decode", 0) >= 1, "worker-5-decode应该有overlap"
        assert overlap_scores.get("worker-1-prefill", 0) == 0, "worker-1-prefill不应该有overlap"
        assert overlap_scores.get("worker-4-prefill", 0) == 0, "worker-4-prefill不应该有overlap"

    def test_cache_event_flow(self):
        """测试缓存事件流程"""
        chat_request = self._create_chat_request()
        route_hint = self.preprocessor.process(chat_request, None, "test-request-3")

        logging.info(f"Token IDs长度: {len(route_hint.token_ids)}")
        logging.info(f"Block size: {self.kvcache_manager.block_size}")

        if len(route_hint.token_ids) < self.kvcache_manager.block_size:
            # 如果token数量不足一个block，补充一些token
            route_hint.token_ids.extend(list(range(100, 100 + self.kvcache_manager.block_size)))

        logging.info(f"补充后的Token IDs长度: {len(route_hint.token_ids)}")

        # 生成事件
        events = self.event_generator.generate_events(
            "worker-2-decode",
            route_hint.token_ids,
        )

        logging.info(f"生成的事件数量: {len(events)}")

        # 处理事件（使用kvcache_manager的event_manager，它已经注册了处理器）
        for event in events:
            logging.info(f"  事件类型: {event.event_type}")
            logging.info(f"  Worker ID: {event.worker_id}")
            logging.info(f"  Engine specific: {event.engine_specific}")
            logging.info(f"  Block hashes: {event.block_hashes}")
            self.kvcache_manager.event_manager.process_event(event)

        # 验证缓存状态
        pod_states = self.kvcache_manager.index.pod_states
        logging.info(f"Pod状态: {pod_states}")

        if "worker-2-decode" in pod_states:
            radix_tree_root = pod_states["worker-2-decode"].radix_tree_root
            logging.info("worker-2-decode的Radix Tree已构建")
            assert radix_tree_root is not None, "Radix Tree应该已构建"
            assert len(radix_tree_root.children) > 0, "Radix Tree应该有子节点"


if __name__ == "__main__":
    logging.info("=== 运行KV缓存重叠集成测试 ===")

    test = TestKVOverlapIntegration()
    test.setup_method()

    logging.info("1. 测试同一个请求发送两次时，第二次应该有overlap")
    try:
        test.test_same_request_twice_overlap()
        logging.info("   ✓ 通过")
    except AssertionError as e:
        logging.info(f"   ✗ 失败: {e}")
    except Exception as e:
        logging.info(f"   ✗ 异常: {e}")
        logging.exception("Traceback:")

    test.setup_method()

    logging.info("2. 测试基于组的路由和overlap")
    try:
        test.test_group_based_routing_with_overlap()
        logging.info("   ✓ 通过")
    except AssertionError as e:
        logging.info(f"   ✗ 失败: {e}")
    except Exception as e:
        logging.info(f"   ✗ 异常: {e}")
        logging.exception("Traceback:")

    test.setup_method()

    logging.info("3. 测试缓存事件流程")
    try:
        test.test_cache_event_flow()
        logging.info("   ✓ 通过")
    except AssertionError as e:
        logging.info(f"   ✗ 失败: {e}")
    except Exception as e:
        logging.info(f"   ✗ 异常: {e}")
        logging.exception("Traceback:")

    logging.info("=== 测试完成 ===")
