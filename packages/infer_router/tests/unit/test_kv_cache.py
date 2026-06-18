"""测试KV缓存管理"""

from openjiuwentools.infer_router.kv_cache.kv_cache import (
    CacheStateUpdate,
    EngineType,
    KVCacheIndex,
    KVCacheManager,
    PrefixCacheScorer,
)


class TestKVCacheManager:
    """测试KV缓存管理器"""

    kv_cache_manager: KVCacheManager
    worker_id: str
    model: str

    def setup_method(self):
        """设置测试环境"""
        self.kv_cache_manager = KVCacheManager(block_size=16)
        self.worker_id = "test-worker-1"
        self.model = "test-model"

    def test_register_worker(self):
        """测试注册工作器"""
        self.kv_cache_manager.register_worker(self.worker_id, "vllm", self.model)
        # 验证工作器已注册
        assert self.worker_id in self.kv_cache_manager.index.pod_states

    def test_update_cache_state(self):
        """测试更新缓存状态"""
        # 注册工作器
        self.kv_cache_manager.register_worker(self.worker_id, "vllm", self.model)

        # 测试store事件 - 使用token_ids而不是block_hashes
        test_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.kv_cache_manager.update_cache_state(
            CacheStateUpdate(
                worker_id=self.worker_id,
                event_type="store",
                token_count=16,
                location="npu",
                token_ids=test_tokens,
            )
        )

        # 验证Radix Tree已更新
        pod_state = self.kv_cache_manager.index.pod_states[self.worker_id]
        assert pod_state.radix_tree_root is not None
        assert len(pod_state.radix_tree_root.children) > 0

    def test_compute_decay_factor(self):
        """测试计算衰减因子"""
        # 测试不同的输出token情况
        assert self.kv_cache_manager.compute_decay_factor(0, 100) == 1.0
        assert self.kv_cache_manager.compute_decay_factor(50, 100) == 0.5
        assert self.kv_cache_manager.compute_decay_factor(100, 100) == 0.0
        assert self.kv_cache_manager.compute_decay_factor(150, 100) == 0.0

    def test_update_block_decay_factor(self):
        """测试更新缓存块的衰减因子（已移除block支持）"""
        # 此测试已过时，因为我们不再使用block
        pass

    def test_get_block_effective_weight(self):
        """测试获取缓存块的有效权重（已移除block支持）"""
        # 此测试已过时，因为我们不再使用block
        pass

    def test_get_cache_stats(self):
        """测试获取缓存统计信息"""
        # 注册工作器
        self.kv_cache_manager.register_worker(self.worker_id, "vllm", self.model)

        # 获取统计信息
        stats = self.kv_cache_manager.get_cache_stats()
        assert stats["total_pods"] >= 1


class TestPrefixCacheScorer:
    """测试前缀缓存评分器"""

    scorer: PrefixCacheScorer

    def setup_method(self):
        """设置测试环境"""
        self.scorer = PrefixCacheScorer(block_size=16)

    def test_compute_block_hash(self):
        """测试计算块哈希"""
        token_ids = [1, 2, 3, 4, 5]
        block_hash = self.scorer.compute_block_hash(token_ids)
        assert isinstance(block_hash, str)
        assert len(block_hash) > 0

    def test_score_vllm(self):
        """测试VLLM评分"""

        # 创建模拟的PodCacheState
        class MockPodCacheState:
            def __init__(self):
                self.engine_type = EngineType.VLLM
                self.cached_blocks = {}
                self.radix_tree_root = None

        pod_state = MockPodCacheState()

        # 测试没有缓存的情况
        score = self.scorer.score_vllm(pod_state, [1, 2, 3, 4, 5])
        assert score == 0.0


class TestKVCacheIndex:
    """测试KV缓存索引"""

    index: KVCacheIndex
    worker_id: str
    model: str

    def setup_method(self):
        """设置测试环境"""
        self.index = KVCacheIndex(block_size=16)
        self.worker_id = "test-worker-1"
        self.model = "test-model"

    def test_register_pod(self):
        """测试注册Pod"""
        self.index.register_pod(self.worker_id, EngineType.VLLM, self.model)
        assert self.worker_id in self.index.pod_states

    def test_unregister_pod(self):
        """测试注销Pod"""
        self.index.register_pod(self.worker_id, EngineType.VLLM, self.model)
        self.index.unregister_pod(self.worker_id)
        assert self.worker_id not in self.index.pod_states

    def test_get_pods_for_model(self):
        """测试获取指定模型的Pod列表"""
        self.index.register_pod(self.worker_id, EngineType.VLLM, self.model)
        pods = self.index.get_pods_for_model(self.model)
        assert self.worker_id in pods

    def test_set_session_affinity(self):
        """测试设置会话亲和性"""
        self.index.register_pod(self.worker_id, EngineType.VLLM, self.model)
        session_id = "test-session-1"
        self.index.set_session_affinity(session_id, self.worker_id)
        assert self.index.get_session_affinity(session_id) == self.worker_id
