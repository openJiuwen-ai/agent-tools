"""测试 CTS 路由算法在不同分发场景下的行为"""

import logging
from unittest.mock import Mock

import pytest

from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.routing.router import WorkloadWeightedAlgorithm
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint, WorkerInfo, WorkerType


@pytest.fixture
def kv_cache_manager():
    """创建 KVCacheManager 的 Mock 实例"""
    manager = Mock(spec=KVCacheManager)
    manager.block_size = 16
    return manager


@pytest.fixture
def cts_algorithm(kv_cache_manager):
    """创建 WorkloadWeightedAlgorithm 实例"""
    return WorkloadWeightedAlgorithm(kv_cache_manager)


@pytest.fixture
def two_groups_workers():
    """创建两组 1P2D 的 worker 配置（全局 P/D 池）

    Prefill pool: worker-1-prefill, worker-2-prefill
    Decode pool: worker-1-decode-1, worker-1-decode-2, worker-2-decode-1, worker-2-decode-2
    """
    prefill_workers = [
        WorkerInfo(
            worker_id="worker-1-prefill",
            model="Qwen3-8B",
            url="http://localhost:8001/v1",
            available_memory=1000000,
            current_load=50,
            cached_prefixes=[],
            engine_type="vllm",
            worker_type=WorkerType.PREFILL,
            group="group1",
        ),
        WorkerInfo(
            worker_id="worker-2-prefill",
            model="Qwen3-8B",
            url="http://localhost:8004/v1",
            available_memory=1000000,
            current_load=50,
            cached_prefixes=[],
            engine_type="vllm",
            worker_type=WorkerType.PREFILL,
            group="group2",
        ),
    ]

    decode_workers = [
        WorkerInfo(
            worker_id="worker-1-decode-1",
            model="Qwen3-8B",
            url="http://localhost:8002/v1",
            available_memory=1000000,
            current_load=50,
            cached_prefixes=[],
            engine_type="vllm",
            worker_type=WorkerType.DECODE,
            group="group1",
        ),
        WorkerInfo(
            worker_id="worker-1-decode-2",
            model="Qwen3-8B",
            url="http://localhost:8003/v1",
            available_memory=1000000,
            current_load=50,
            cached_prefixes=[],
            engine_type="vllm",
            worker_type=WorkerType.DECODE,
            group="group1",
        ),
        WorkerInfo(
            worker_id="worker-2-decode-1",
            model="Qwen3-8B",
            url="http://localhost:8005/v1",
            available_memory=1000000,
            current_load=50,
            cached_prefixes=[],
            engine_type="vllm",
            worker_type=WorkerType.DECODE,
            group="group2",
        ),
        WorkerInfo(
            worker_id="worker-2-decode-2",
            model="Qwen3-8B",
            url="http://localhost:8006/v1",
            available_memory=1000000,
            current_load=50,
            cached_prefixes=[],
            engine_type="vllm",
            worker_type=WorkerType.DECODE,
            group="group2",
        ),
    ]

    all_workers = {w.worker_id: w for w in prefill_workers + decode_workers}

    return prefill_workers, decode_workers, all_workers


def create_requests_with_overlap():
    """创建10个请求，q1 的 token 重叠度最高，递减到 q10 的 10%

    每个请求 50-200 个 token
    q1: 90% overlap, 200 tokens
    q2: 80% overlap, 180 tokens
    q3: 70% overlap, 160 tokens
    q4: 60% overlap, 140 tokens
    q5: 50% overlap, 120 tokens
    q6: 40% overlap, 100 tokens
    q7: 30% overlap, 90 tokens
    q8: 20% overlap, 80 tokens
    q9: 10% overlap, 60 tokens
    q10: 10% overlap, 50 tokens
    """
    requests = []
    base_token_count = 200

    for i in range(1, 11):
        overlap = max(0.1, 1.0 - (i - 1) * 0.1)  # 90%, 80%, 70%, ... 10%
        token_count = max(50, base_token_count - (i - 1) * 20)  # 200, 180, 160, ...

        request = RouteHint(
            request_id=f"q{i}",
            model="Qwen3-8B",
            priority=5,
            estimated_output_tokens=128,
            prefix_id=f"q{i}",  # 每个请求有唯一的 prefix_id
            token_ids=list(range(token_count)),  # 生成指定数量的 token
            next_turn_prefill=False,
            total_requests=1,
            iat=100,
        )
        requests.append((f"q{i}", request, overlap, token_count))

    return requests


class TestCTSWorkerPairSelection:
    """测试 CTS 算法在 worker pair 选择场景下的行为（全局 P/D 池模式）"""

    @staticmethod
    def test_sequential_dispatch_q1_to_q10_twice(
        cts_algorithm, two_groups_workers, kv_cache_manager
    ):
        """测试用例1：所有请求按顺序分发 q1-q10，q1-q10 两遍"""
        prefill_workers, decode_workers, all_workers = two_groups_workers
        requests = create_requests_with_overlap()

        logging.info("\n" + "=" * 80)
        logging.info("测试用例1：顺序分发 q1-q10 两遍")
        logging.info("=" * 80)

        for round_num in range(2):
            logging.info(f"\n--- 第 {round_num + 1} 轮分发 ---")
            for req_id, request, overlap, token_count in requests:
                def mock_find_matches(token_ids, model, token_count_val=token_count):
                    result = {}
                    for worker_id in all_workers:
                        if "worker-1-" in worker_id:
                            result[worker_id] = int(token_count_val * 0.9)
                        else:
                            result[worker_id] = int(token_count_val * 0.5)
                    return result

                kv_cache_manager.find_matches = Mock(side_effect=mock_find_matches)

                try:
                    prefill_worker, decode_worker = cts_algorithm.select_worker_pair(
                        prefill_workers, decode_workers, request
                    )

                    assert prefill_worker.worker_type == WorkerType.PREFILL
                    assert decode_worker.worker_type == WorkerType.DECODE

                    load_increase = token_count / 10
                    prefill_worker.current_load = min(
                        100, prefill_worker.current_load + load_increase
                    )
                    decode_worker.current_load = min(
                        100, decode_worker.current_load + load_increase
                    )

                except Exception as e:
                    logging.info(f"{req_id} 分发失败: {e}")
                    raise

    @staticmethod
    def test_reverse_dispatch_q10_to_q1_twice(
        cts_algorithm, two_groups_workers, kv_cache_manager
    ):
        """测试用例2：所有请求按逆序分发 q10-q1，q10-q1 两遍"""
        prefill_workers, decode_workers, all_workers = two_groups_workers
        requests = create_requests_with_overlap()

        for round_num in range(2):
            for req_id, request, overlap, token_count in reversed(requests):
                def mock_find_matches(token_ids, model, token_count_val=token_count):
                    result = {}
                    for worker_id in all_workers:
                        if "worker-1-" in worker_id:
                            result[worker_id] = int(token_count_val * 0.9)
                        else:
                            result[worker_id] = int(token_count_val * 0.5)
                    return result

                kv_cache_manager.find_matches = Mock(side_effect=mock_find_matches)

                try:
                    prefill_worker, decode_worker = cts_algorithm.select_worker_pair(
                        prefill_workers, decode_workers, request
                    )

                    assert prefill_worker.worker_type == WorkerType.PREFILL
                    assert decode_worker.worker_type == WorkerType.DECODE

                    load_increase = token_count / 10
                    prefill_worker.current_load = min(
                        100, prefill_worker.current_load + load_increase
                    )
                    decode_worker.current_load = min(
                        100, decode_worker.current_load + load_increase
                    )

                except Exception as e:
                    logging.info(f"{req_id} 分发失败: {e}")
                    raise

    @staticmethod
    def test_shuffled_dispatch_with_duplicates(
        cts_algorithm, two_groups_workers, kv_cache_manager
    ):
        """测试用例3：请求复制两份然后打乱顺序分发"""
        import random

        prefill_workers, decode_workers, all_workers = two_groups_workers
        requests = create_requests_with_overlap()

        all_requests = []
        for round_num in range(2):
            for req_id, _request, overlap, token_count in requests:
                shuffled_request = RouteHint(
                    request_id=f"{req_id}_round{round_num}",
                    model="Qwen3-8B",
                    priority=5,
                    estimated_output_tokens=128,
                    prefix_id=req_id,
                    token_ids=list(range(token_count)),
                    next_turn_prefill=False,
                    total_requests=1,
                    iat=100,
                )
                all_requests.append((req_id, shuffled_request, overlap, token_count))

        random.seed(42)
        random.shuffle(all_requests)

        for dispatch_id, (req_id, request, overlap, token_count) in enumerate(all_requests, 1):
            def mock_find_matches(token_ids, model):
                result = {}
                for worker_id in all_workers:
                    if "worker-1-" in worker_id:
                        result[worker_id] = int(token_ids.__len__() * 0.9)
                    else:
                        result[worker_id] = int(token_ids.__len__() * 0.5)
                return result

            kv_cache_manager.find_matches = Mock(side_effect=mock_find_matches)

            try:
                prefill_worker, decode_worker = cts_algorithm.select_worker_pair(
                    prefill_workers, decode_workers, request
                )

                assert prefill_worker.worker_type == WorkerType.PREFILL
                assert decode_worker.worker_type == WorkerType.DECODE

                load_increase = token_count / 10
                prefill_worker.current_load = min(100, prefill_worker.current_load + load_increase)
                decode_worker.current_load = min(100, decode_worker.current_load + load_increase)

            except Exception as e:
                logging.info(f"{req_id} (round {dispatch_id}) 分发失败: {e}")
                raise


class TestCTSLoadBalancing:
    """测试 CTS 算法的负载均衡能力"""

    @staticmethod
    def test_load_balance_with_different_worker_loads(cts_algorithm, kv_cache_manager):
        """测试算法在高负载情况下的负载均衡"""
        from openjiuwentools.infer_router.worker.workload_manager import WorkerWorkload

        workers = {
            "high-load": WorkerInfo(
                worker_id="high-load",
                model="test-model",
                url="http://localhost:8001/v1",
                available_memory=1000000,
                current_load=95,
                cached_prefixes=[],
                engine_type="vllm",
            ),
            "low-load": WorkerInfo(
                worker_id="low-load",
                model="test-model",
                url="http://localhost:8002/v1",
                available_memory=1000000,
                current_load=5,
                cached_prefixes=[],
                engine_type="vllm",
            ),
        }

        route_hint = RouteHint(
            request_id="test-1",
            model="test-model",
            priority=5,
            estimated_output_tokens=128,
            prefix_id="test-prefix",
            token_ids=list(range(100)),
            next_turn_prefill=False,
            total_requests=1,
            iat=100,
        )

        kv_cache_manager.find_matches = Mock(return_value={"high-load": 50, "low-load": 50})

        mock_worker_manager = Mock()
        high_workload = WorkerWorkload("high-load")
        high_workload.pending_tokens = 10000
        high_workload.pending_osl = 1000
        low_workload = WorkerWorkload("low-load")
        low_workload.pending_tokens = 1000
        low_workload.pending_osl = 100

        def get_workload(worker_id):
            if worker_id == "high-load":
                return high_workload
            else:
                return low_workload

        mock_worker_manager.workload_manager.get_workload = Mock(side_effect=get_workload)
        cts_algorithm.worker_manager = mock_worker_manager

        selection_count = {"high-load": 0, "low-load": 0}
        num_iterations = 100

        for _ in range(num_iterations):
            selected = cts_algorithm.select_worker(list(workers.values()), route_hint)
            selection_count[selected.worker_id] += 1

        assert selection_count["low-load"] > selection_count["high-load"], (
            f"低负载 worker 应该被选择更多，但 "
            f"low-load={selection_count['low-load']}, "
            f"high-load={selection_count['high-load']}"
        )


class TestCTSScoreValidation:
    """测试 CTS 算法分数计算的正确性"""

    @staticmethod
    def test_all_scores_are_non_negative(cts_algorithm, kv_cache_manager):
        """验证所有计算出的分数都是非负的"""
        workers = {
            f"worker-{i}": WorkerInfo(
                worker_id=f"worker-{i}",
                model="test-model",
                url=f"http://localhost:800{i}/v1",
                available_memory=1000000,
                current_load=i * 10,
                cached_prefixes=[],
                engine_type="vllm",
            )
            for i in range(1, 6)
        }

        requests = create_requests_with_overlap()

        all_scores = []

        for req_id, request, _overlap, token_count in requests:
            def mock_find_matches(token_ids, model, token_count_val=token_count):
                return {f"worker-{i}": int(token_count_val * (0.5 + i * 0.1)) for i in range(1, 6)}

            kv_cache_manager.find_matches = Mock(side_effect=mock_find_matches)

            for worker in workers.values():
                score = cts_algorithm.calculate_score(worker, list(workers.values()), request)
                all_scores.append((req_id, worker.worker_id, score))

                assert score >= 0, (
                    f"分数不应为负: req={req_id}, worker={worker.worker_id}, score={score}"
                )

        negative_scores = [s for s in all_scores if s[2] < 0]
        assert len(negative_scores) == 0, f"发现 {len(negative_scores)} 个负分数"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
