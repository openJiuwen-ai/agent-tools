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
    """创建两组 1P2D 的 worker 配置

    Group 1: worker-1-prefill, worker-1-decode-1, worker-1-decode-2
    Group 2: worker-2-prefill, worker-2-decode-1, worker-2-decode-2
    """
    workers = {}

    # Group 1
    workers["worker-1-prefill"] = WorkerInfo(
        worker_id="worker-1-prefill",
        model="Qwen3-8B",
        url="http://localhost:8001/v1",
        available_memory=1000000,
        current_load=50,
        cached_prefixes=[],
        engine_type="vllm",
        worker_type=WorkerType.PREFILL,
        group="group1",
    )
    workers["worker-1-decode-1"] = WorkerInfo(
        worker_id="worker-1-decode-1",
        model="Qwen3-8B",
        url="http://localhost:8002/v1",
        available_memory=1000000,
        current_load=50,
        cached_prefixes=[],
        engine_type="vllm",
        worker_type=WorkerType.DECODE,
        group="group1",
    )
    workers["worker-1-decode-2"] = WorkerInfo(
        worker_id="worker-1-decode-2",
        model="Qwen3-8B",
        url="http://localhost:8003/v1",
        available_memory=1000000,
        current_load=50,
        cached_prefixes=[],
        engine_type="vllm",
        worker_type=WorkerType.DECODE,
        group="group1",
    )

    # Group 2
    workers["worker-2-prefill"] = WorkerInfo(
        worker_id="worker-2-prefill",
        model="Qwen3-8B",
        url="http://localhost:8004/v1",
        available_memory=1000000,
        current_load=50,
        cached_prefixes=[],
        engine_type="vllm",
        worker_type=WorkerType.PREFILL,
        group="group2",
    )
    workers["worker-2-decode-1"] = WorkerInfo(
        worker_id="worker-2-decode-1",
        model="Qwen3-8B",
        url="http://localhost:8005/v1",
        available_memory=1000000,
        current_load=50,
        cached_prefixes=[],
        engine_type="vllm",
        worker_type=WorkerType.DECODE,
        group="group2",
    )
    workers["worker-2-decode-2"] = WorkerInfo(
        worker_id="worker-2-decode-2",
        model="Qwen3-8B",
        url="http://localhost:8006/v1",
        available_memory=1000000,
        current_load=50,
        cached_prefixes=[],
        engine_type="vllm",
        worker_type=WorkerType.DECODE,
        group="group2",
    )

    groups = {
        "group1": ["worker-1-prefill", "worker-1-decode-1", "worker-1-decode-2"],
        "group2": ["worker-2-prefill", "worker-2-decode-1", "worker-2-decode-2"],
    }

    return workers, groups


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
    """测试 CTS 算法在 worker pair 选择场景下的行为"""

    @staticmethod
    def test_sequential_dispatch_q1_to_q10_twice(
        cts_algorithm, two_groups_workers, kv_cache_manager
    ):
        """测试用例1：所有请求按顺序分发 q1-q10，q1-q10 两遍

        验证：
        1. 分发节点是否正确（应该选择分数最高的 group）
        2. 过程中的分数计算是否正确
        3. 重叠度高的请求是否倾向于选择相同节点
        4. 每次分发后更新 worker 状态（current_load），后续请求基于更新后的状态来分发
        """
        workers, groups = two_groups_workers
        requests = create_requests_with_overlap()

        # 打印测试信息
        logging.info("\n" + "=" * 80)
        logging.info("测试用例1：顺序分发 q1-q10 两遍")
        logging.info("=" * 80)
        logging.info(
            f"{'请求ID':<8} {'Token数':<8} {'重叠度':<10} "
            f"{'选中的Prefill':<20} {'选中的Decode':<20} "
            f"{'Prefill分数':<15} {'Decode分数':<15} "
            f"{'Prefill负载':<12} {'Decode负载':<12}"
        )
        logging.info("-" * 100)

        # 记录初始负载
        initial_loads = {wid: w.current_load for wid, w in workers.items()}
        logging.info(f"初始负载: {initial_loads}")

        # 分发两遍
        for round_num in range(2):
            logging.info(f"\n--- 第 {round_num + 1} 轮分发 ---")
            for req_id, request, overlap, token_count in requests:
                # 设置 KV cache manager 的 mock 返回值
                # 让 group1 和 group2 的 worker 有不同的重叠度，以便观察选择行为
                def mock_find_matches(token_ids, model, token_count_val=token_count):
                    """模拟不同请求在不同 worker 上的重叠度"""
                    # group1 的 worker 有更高的重叠度，应该更容易被选择
                    result = {}
                    for worker_id in workers:
                        if "worker-1-" in worker_id:
                            # group1 worker 有更高的重叠度
                            result[worker_id] = int(token_count_val * 0.9)
                        else:
                            # group2 worker 重叠度稍低
                            result[worker_id] = int(token_count_val * 0.5)
                    return result

                kv_cache_manager.find_matches = Mock(side_effect=mock_find_matches)

                try:
                    prefill_worker, decode_worker = cts_algorithm.select_worker_pair(
                        workers, groups, request
                    )

                    # 计算分数以供验证
                    prefill_score = cts_algorithm.calculate_score(
                        prefill_worker,
                        [
                            w
                            for w in workers.values()
                            if w.worker_type in (WorkerType.PREFILL, WorkerType.COMBINED)
                        ],
                        request,
                    )
                    decode_score = cts_algorithm.calculate_score(
                        decode_worker,
                        [
                            w
                            for w in workers.values()
                            if w.worker_type in (WorkerType.DECODE, WorkerType.COMBINED)
                        ],
                        request,
                    )

                    logging.info(
                        f"{req_id:<8} {token_count:<8} {overlap:.0%}     "
                        f"{prefill_worker.worker_id:<20} "
                        f"{decode_worker.worker_id:<20} "
                        f"{prefill_score:< 15.3f} {decode_score:< 15.3f} "
                        f"{prefill_worker.current_load:< 12} "
                        f"{decode_worker.current_load:< 12}"
                    )

                    # 验证：分数应该是正数
                    assert prefill_score >= 0, (
                        f"{prefill_worker.worker_id} 的分数 {prefill_score} 不应为负"
                    )
                    assert decode_score >= 0, (
                        f"{decode_worker.worker_id} 的分数 {decode_score} 不应为负"
                    )

                    # 验证：选中的 worker 应该在同一个 group
                    selected_group = None
                    for group_name, worker_ids in groups.items():
                        if (
                            prefill_worker.worker_id in worker_ids
                            and decode_worker.worker_id in worker_ids
                        ):
                            selected_group = group_name
                            break
                    assert selected_group is not None, (
                        "选中的 prefill 和 decode worker 应该在同一个 group"
                    )

                    # 更新 worker 的负载状态
                    # 模拟请求分发后，worker 的负载增加
                    # 增加的负载与 token 数量成正比
                    load_increase = token_count / 10  # 假设每个 token 增加 0.1 的负载
                    prefill_worker.current_load = min(
                        100, prefill_worker.current_load + load_increase
                    )
                    decode_worker.current_load = min(
                        100, decode_worker.current_load + load_increase
                    )

                except Exception as e:
                    logging.info(f"{req_id} 分发失败: {e}")
                    raise

        # 打印最终负载
        final_loads = {wid: w.current_load for wid, w in workers.items()}
        logging.info(f"\n最终负载: {final_loads}")

        logging.info("\n" + "=" * 80)
        logging.info("测试用例1 完成")
        logging.info("=" * 80)

    @staticmethod
    def test_reverse_dispatch_q10_to_q1_twice(
        cts_algorithm, two_groups_workers, kv_cache_manager
    ):
        """测试用例2：所有请求按逆序分发 q10-q1，q10-q1 两遍

        验证：
        1. 分发节点是否正确（应该选择分数最高的 group）
        2. 过程中的分数计算是否正确
        3. 即使逆序分发，算法仍能正确工作
        4. 每次分发后更新 worker 状态（current_load），后续请求基于更新后的状态来分发
        """
        workers, groups = two_groups_workers
        requests = create_requests_with_overlap()

        # 打印测试信息
        logging.info("\n" + "=" * 100)
        logging.info("测试用例2：逆序分发 q10-q1 两遍")
        logging.info("=" * 100)
        logging.info(
            f"{'请求ID':<8} {'Token数':<8} {'重叠度':<10} "
            f"{'选中的Prefill':<20} {'选中的Decode':<20} "
            f"{'Prefill分数':<15} {'Decode分数':<15} "
            f"{'Prefill负载':<12} {'Decode负载':<12}"
        )
        logging.info("-" * 100)

        # 记录初始负载
        initial_loads = {wid: w.current_load for wid, w in workers.items()}
        logging.info(f"初始负载: {initial_loads}")

        # 分发两遍（逆序）
        for round_num in range(2):
            logging.info(f"\n--- 第 {round_num + 1} 轮分发 ---")
            for req_id, request, overlap, token_count in reversed(requests):
                # 设置 KV cache manager 的 mock 返回值
                def mock_find_matches(token_ids, model, token_count_val=token_count):
                    """模拟不同请求在不同 worker 上的重叠度"""
                    # group1 的 worker 有更高的重叠度，应该更容易被选择
                    result = {}
                    for worker_id in workers:
                        if "worker-1-" in worker_id:
                            # group1 worker 有更高的重叠度
                            result[worker_id] = int(token_count_val * 0.9)
                        else:
                            # group2 worker 重叠度稍低
                            result[worker_id] = int(token_count_val * 0.5)
                    return result

                kv_cache_manager.find_matches = Mock(side_effect=mock_find_matches)

                try:
                    prefill_worker, decode_worker = cts_algorithm.select_worker_pair(
                        workers, groups, request
                    )

                    # 计算分数以供验证
                    prefill_score = cts_algorithm.calculate_score(
                        prefill_worker,
                        [
                            w
                            for w in workers.values()
                            if w.worker_type in (WorkerType.PREFILL, WorkerType.COMBINED)
                        ],
                        request,
                    )
                    decode_score = cts_algorithm.calculate_score(
                        decode_worker,
                        [
                            w
                            for w in workers.values()
                            if w.worker_type in (WorkerType.DECODE, WorkerType.COMBINED)
                        ],
                        request,
                    )

                    logging.info(
                        f"{req_id:<8} {token_count:<8} {overlap:.0%}     "
                        f"{prefill_worker.worker_id:<20} "
                        f"{decode_worker.worker_id:<20} "
                        f"{prefill_score:< 15.3f} {decode_score:< 15.3f} "
                        f"{prefill_worker.current_load:< 12} "
                        f"{decode_worker.current_load:< 12}"
                    )

                    # 验证：分数应该是正数
                    assert prefill_score >= 0, (
                        f"{prefill_worker.worker_id} 的分数 {prefill_score} 不应为负"
                    )
                    assert decode_score >= 0, (
                        f"{decode_worker.worker_id} 的分数 {decode_score} 不应为负"
                    )

                    # 验证：选中的 worker 应该在同一个 group
                    selected_group = None
                    for group_name, worker_ids in groups.items():
                        if (
                            prefill_worker.worker_id in worker_ids
                            and decode_worker.worker_id in worker_ids
                        ):
                            selected_group = group_name
                            break
                    assert selected_group is not None, (
                        "选中的 prefill 和 decode worker 应该在同一个 group"
                    )

                    # 更新 worker 的负载状态
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

        # 打印最终负载
        final_loads = {wid: w.current_load for wid, w in workers.items()}
        logging.info(f"\n最终负载: {final_loads}")

        logging.info("\n" + "=" * 80)
        logging.info("测试用例2 完成")
        logging.info("=" * 80)

    @staticmethod
    def test_shuffled_dispatch_with_duplicates(
        cts_algorithm, two_groups_workers, kv_cache_manager
    ):
        """测试用例3：请求复制两份然后打乱顺序分发

        验证：
        1. 分发节点是否正确（应该选择分数最高的 group）
        2. 过程中的分数计算是否正确
        3. 相同的请求（相同的 prefix_id）是否倾向于选择相同的节点
        4. 每次分发后更新 worker 状态（current_load），后续请求基于更新后的状态来分发
        """
        import random

        workers, groups = two_groups_workers
        requests = create_requests_with_overlap()

        # 复制两份并打乱顺序
        all_requests = []
        for round_num in range(2):
            for req_id, _request, overlap, token_count in requests:
                # 创建新的 RouteHint，但使用相同的 prefix_id
                shuffled_request = RouteHint(
                    request_id=f"{req_id}_round{round_num}",
                    model="Qwen3-8B",
                    priority=5,
                    estimated_output_tokens=128,
                    prefix_id=req_id,  # 保持相同的 prefix_id
                    token_ids=list(range(token_count)),
                    next_turn_prefill=False,
                    total_requests=1,
                    iat=100,
                )
                all_requests.append((req_id, shuffled_request, overlap, token_count))

        # 打乱顺序
        random.seed(42)  # 固定随机种子以便复现
        random.shuffle(all_requests)

        # 打印测试信息
        logging.info("\n" + "=" * 110)
        logging.info("测试用例3：打乱顺序分发（每请求两份）")
        logging.info("=" * 110)
        logging.info(
            f"{'分发ID':<12} {'原始请求':<8} {'Token数':<8} "
            f"{'重叠度':<10} {'选中的Prefill':<20} "
            f"{'选中的Decode':<20} {'Prefill分数':<15} "
            f"{'Decode分数':<15} {'Prefill负载':<12} "
            f"{'Decode负载':<12}"
        )
        logging.info("-" * 140)

        # 记录初始负载
        initial_loads = {wid: w.current_load for wid, w in workers.items()}
        logging.info(f"初始负载: {initial_loads}\n")

        # 记录每个原始请求的选择结果
        request_selection_history = {req_id: [] for req_id, _, _, _ in requests}

        for dispatch_id, (req_id, request, overlap, token_count) in enumerate(all_requests, 1):
            # 设置 KV cache manager 的 mock 返回值
            def mock_find_matches(token_ids, model):
                """模拟不同请求在不同 worker 上的重叠度"""
                # group1 的 worker 有更高的重叠度，应该更容易被选择
                result = {}
                for worker_id in workers:
                    if "worker-1-" in worker_id:
                        # group1 worker 有更高的重叠度
                        result[worker_id] = int(token_ids.__len__() * 0.9)
                    else:
                        # group2 worker 重叠度稍低
                        result[worker_id] = int(token_ids.__len__() * 0.5)
                return result

            kv_cache_manager.find_matches = Mock(side_effect=mock_find_matches)

            try:
                prefill_worker, decode_worker = cts_algorithm.select_worker_pair(
                    workers, groups, request
                )

                # 计算分数以供验证
                prefill_score = cts_algorithm.calculate_score(
                    prefill_worker,
                    [
                        w
                        for w in workers.values()
                        if w.worker_type in (WorkerType.PREFILL, WorkerType.COMBINED)
                    ],
                    request,
                )
                decode_score = cts_algorithm.calculate_score(
                    decode_worker,
                    [
                        w
                        for w in workers.values()
                        if w.worker_type in (WorkerType.DECODE, WorkerType.COMBINED)
                    ],
                    request,
                )

                logging.info(
                    f"{dispatch_id:<12} {req_id:<8} "
                    f"{token_count:<8} {overlap:.0%}     "
                    f"{prefill_worker.worker_id:<20} "
                    f"{decode_worker.worker_id:<20} "
                    f"{prefill_score:< 15.3f} {decode_score:< 15.3f} "
                    f"{prefill_worker.current_load:< 12} "
                    f"{decode_worker.current_load:< 12}"
                )

                # 记录选择历史
                request_selection_history[req_id].append(
                    {
                        "round": dispatch_id,
                        "prefill": prefill_worker.worker_id,
                        "decode": decode_worker.worker_id,
                        "prefill_score": prefill_score,
                        "decode_score": decode_score,
                    }
                )

                # 验证：分数应该是正数
                assert prefill_score >= 0, (
                    f"{prefill_worker.worker_id} 的分数 {prefill_score} 不应为负"
                )
                assert decode_score >= 0, (
                    f"{decode_worker.worker_id} 的分数 {decode_score} 不应为负"
                )

                # 验证：选中的 worker 应该在同一个 group
                selected_group = None
                for group_name, worker_ids in groups.items():
                    if (
                        prefill_worker.worker_id in worker_ids
                        and decode_worker.worker_id in worker_ids
                    ):
                        selected_group = group_name
                        break
                assert selected_group is not None, (
                    "选中的 prefill 和 decode worker 应该在同一个 group"
                )

                # 更新 worker 的负载状态
                load_increase = token_count / 10
                prefill_worker.current_load = min(100, prefill_worker.current_load + load_increase)
                decode_worker.current_load = min(100, decode_worker.current_load + load_increase)

            except Exception as e:
                logging.info(f"{req_id} (round {dispatch_id}) 分发失败: {e}")
                raise

        # 打印最终负载
        final_loads = {wid: w.current_load for wid, w in workers.items()}
        logging.info(f"\n最终负载: {final_loads}")

        # 验证相同 prefix_id 的请求是否选择了相同的节点
        logging.info("\n--- 相同 prefix_id 的选择一致性分析 ---")
        for req_id, history in request_selection_history.items():
            if len(history) >= 2:
                prefill_choices = {h["prefill"] for h in history}
                decode_choices = {h["decode"] for h in history}
                logging.info(
                    f"{req_id}: Prefill 选择 {len(prefill_choices)} 种, Decode 选择 {len(decode_choices)} 种"
                )

        logging.info("\n" + "=" * 80)
        logging.info("测试用例3 完成")
        logging.info("=" * 80)


class TestCTSLoadBalancing:
    """测试 CTS 算法的负载均衡能力"""

    @staticmethod
    def test_load_balance_with_different_worker_loads(cts_algorithm, kv_cache_manager):
        """测试算法在高负载情况下的负载均衡"""
        from openjiuwentools.infer_router.worker.workload_manager import WorkerWorkload

        # 创建两个负载差异很大的 worker
        workers = {
            "high-load": WorkerInfo(
                worker_id="high-load",
                model="test-model",
                url="http://localhost:8001/v1",
                available_memory=1000000,
                current_load=95,  # 高负载
                cached_prefixes=[],
                engine_type="vllm",
            ),
            "low-load": WorkerInfo(
                worker_id="low-load",
                model="test-model",
                url="http://localhost:8002/v1",
                available_memory=1000000,
                current_load=5,  # 低负载
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

        # 设置相同的重叠度
        kv_cache_manager.find_matches = Mock(return_value={"high-load": 50, "low-load": 50})

        # 设置 worker_manager 和 workload 来模拟负载差异
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

        # 多次选择，验证是否倾向于选择低负载的 worker
        selection_count = {"high-load": 0, "low-load": 0}
        num_iterations = 100

        for _ in range(num_iterations):
            selected = cts_algorithm.select_worker(list(workers.values()), route_hint)
            selection_count[selected.worker_id] += 1

        logging.info(f"\n负载均衡测试结果（共 {num_iterations} 次）：")
        high_pct = selection_count["high-load"] / num_iterations * 100
        low_pct = selection_count["low-load"] / num_iterations * 100
        logging.info(
            f"  high-load 被选中: {selection_count['high-load']} 次 "
            f"({high_pct:.1f}%)"
        )
        logging.info(
            f"  low-load 被选中: {selection_count['low-load']} 次 "
            f"({low_pct:.1f}%)"
        )

        # 低负载 worker 应该被选择更多次
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
                current_load=i * 10,  # 10%, 20%, ...
                cached_prefixes=[],
                engine_type="vllm",
            )
            for i in range(1, 6)
        }

        requests = create_requests_with_overlap()

        all_scores = []

        for req_id, request, _overlap, token_count in requests:
            # 设置 KV cache manager 的 mock
            def mock_find_matches(token_ids, model, token_count_val=token_count):
                return {f"worker-{i}": int(token_count_val * (0.5 + i * 0.1)) for i in range(1, 6)}

            kv_cache_manager.find_matches = Mock(side_effect=mock_find_matches)

            for worker in workers.values():
                score = cts_algorithm.calculate_score(worker, list(workers.values()), request)
                all_scores.append((req_id, worker.worker_id, score))

                # 验证分数非负
                assert score >= 0, (
                    f"分数不应为负: req={req_id}, worker={worker.worker_id}, score={score}"
                )

        # 打印分数统计
        logging.info(f"\n分数统计（共 {len(all_scores)} 个分数）：")
        min_score = min(s[2] for s in all_scores)
        max_score = max(s[2] for s in all_scores)
        avg_score = sum(s[2] for s in all_scores) / len(all_scores)
        logging.info(f"  最小值: {min_score:.3f}")
        logging.info(f"  最大值: {max_score:.3f}")
        logging.info(f"  平均值: {avg_score:.3f}")

        # 所有分数都应该是非负的
        negative_scores = [s for s in all_scores if s[2] < 0]
        assert len(negative_scores) == 0, f"发现 {len(negative_scores)} 个负分数"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
