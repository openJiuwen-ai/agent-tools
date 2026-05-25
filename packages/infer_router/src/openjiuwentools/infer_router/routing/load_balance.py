import time
from dataclasses import dataclass

from loguru import logger

from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint, WorkerInfo


@dataclass
class SharedContext:
    """_prepare_shared_context 返回的共享上下文"""

    overlap_scores: dict
    osl: str
    iat: str
    last_w: str | None
    last_group: str | None
    reuse_after: int
    decode_cost: float
    iat_factor: float
    temp: float


@dataclass
class PrefixUpdateParams:
    """update_prefix 所需参数"""

    prefix_id: str
    worker_id: str
    initial_reuse_budget: int | None = None
    decode_cost: float = 2.0
    prefill_cost: float = 0.0
    iat_factor: float = 1.0
    group_id: str | None = None


@dataclass
class FeatureParams:
    """feature_vector 所需参数"""

    worker: WorkerInfo
    overlap: float
    last_w: str | None
    reuse_after: int
    decode_cost: float
    prefill_cost: float
    iat_factor: float
    last_group: str | None = None


@dataclass
class ScoreParams:
    """calculate_score 的预计算参数"""

    overlap_scores: dict | None = None
    last_w: str | None = None
    reuse_after: int | None = None
    decode_cost: float | None = None
    iat_factor: float | None = None
    last_group: str | None = None


class LoadBalancingAlgorithm:
    """负载均衡算法基类"""

    def select_worker(self, workers: list[WorkerInfo], route_hint: RouteHint) -> WorkerInfo:
        """选择工作器"""
        pass

    def select_worker_pair(
        self,
        workers: dict[str, WorkerInfo],
        groups: dict[str, list[str]],
        route_hint: RouteHint,
    ) -> tuple[WorkerInfo, WorkerInfo]:
        """选择最佳的prefill-decode工作器对

        Args:
            workers: 所有可用工作器字典（worker_id -> WorkerInfo）
            groups: group字典（group_name -> list[worker_id]）
            route_hint: 路由提示信息

        Returns:
            (prefill_worker, decode_worker) - 最佳工作器对
        """
        pass


class RoundRobinAlgorithm(LoadBalancingAlgorithm):
    """轮询算法 — 按 group 轮询，组内 prefill/decode 各自轮询"""

    def __init__(self):
        self.current_index = -1
        self.group_index = -1
        self.prefill_indices: dict[str, int] = {}
        self.decode_indices: dict[str, int] = {}

    def select_worker(self, workers: list[WorkerInfo], route_hint: RouteHint) -> WorkerInfo:
        self.current_index = (self.current_index + 1) % len(workers)
        return workers[self.current_index]

    def select_worker_pair(
        self,
        workers: dict[str, WorkerInfo],
        groups: dict[str, list[str]],
        route_hint: RouteHint,
    ) -> tuple[WorkerInfo, WorkerInfo]:
        from openjiuwentools.infer_router.schemas.agent_hints import WorkerType

        group_list = list(groups.keys())
        if not group_list:
            raise ValueError("No groups available")

        attempts = len(group_list)
        while attempts > 0:
            self.group_index = (self.group_index + 1) % len(group_list)
            group_name = group_list[self.group_index]
            workers_in_group = [workers[wid] for wid in groups[group_name] if wid in workers]

            prefill_workers = [w for w in workers_in_group if w.worker_type == WorkerType.PREFILL]
            decode_workers = [w for w in workers_in_group if w.worker_type == WorkerType.DECODE]

            if not prefill_workers or not decode_workers:
                attempts -= 1
                continue

            pi = self.prefill_indices.get(group_name, -1)
            pi = (pi + 1) % len(prefill_workers)
            self.prefill_indices[group_name] = pi

            di = self.decode_indices.get(group_name, -1)
            di = (di + 1) % len(decode_workers)
            self.decode_indices[group_name] = di

            return prefill_workers[pi], decode_workers[di]

        raise ValueError("No valid prefill or decode workers available")


class WorkloadWeightedAlgorithm(LoadBalancingAlgorithm):
    """Contextual Thompson Sampling 路由算法

    基于 router_algor.md 文档实现的智能路由算法，包含：
    - KV 缓存感知
    - 上下文汤普森采样（LinTS）
    - 工作负载平衡
    - 亲和性和粘性
    - 自适应探索/利用平衡
    """

    def __init__(self, kv_cache_manager: KVCacheManager, worker_manager=None):
        self.kv_cache_manager = kv_cache_manager
        self.worker_manager = worker_manager

        # 算法参数配置（来自 router_algor.md）
        self.affinity_base = 0.30
        self.affinity_reuse_weight = 0.15
        self.affinity_iat_weight = 0.20
        self.base_ts_weight = 0.10
        self.sticky_load_floor = 0.70

        self.temp_base = 1.0
        self.temp_min = 0.15
        self.temp_max = 2.0

        self.switch_cost_base = 0.20
        self.switch_cost_reuse = 0.08
        self.switch_cost_iat = 0.05

        self.queue_penalty_weight = 0.50
        self.npu_penalty_weight = 1.00
        self.outstanding_work_weight = 0.45
        self.job_npu_coupling_weight = 0.40
        self.job_queue_coupling_weight = 0.20

        self.prefill_token_scale = 1024.0
        self.prefill_weight = 1.0

        self.lints_lambda = 1.0
        self.lints_v = 0.25
        self.lints_forget = 0.995

        # LinTS 模型参数（每个工作器一个）
        self.lints_models = {}  # worker_id -> (A, b) 参数

        # Beta bandit 参数（用于探索）
        self.beta_params = {}  # worker_id -> (alpha, beta)

        # 前缀跟踪
        self.prefix_tracking = {}  # prefix_id -> (last_worker, last_group, reuse_budget)
        self.prefix_meta = {}  # prefix_id -> {decode_cost, prefill_cost, iat_factor}

    def _get_worker_workload(self, worker_id: str):
        """获取 worker 的工作负载信息"""
        if (
            self.worker_manager
            and hasattr(self.worker_manager, "workload_manager")
            and self.worker_manager.workload_manager
        ):
            return self.worker_manager.workload_manager.get_workload(worker_id)
        return None

    def _prepare_shared_context(self, route_hint: RouteHint) -> SharedContext:
        """计算所有 worker 共享的上下文参数

        Returns:
            SharedContext 数据类，包含 overlap_scores、osl、iat 等字段
        """
        # 步骤1：提取请求参数（所有 worker 共享）
        prefix_id = route_hint.prefix_id
        token_ids = route_hint.token_ids or []

        estimated_osl = self.estimate_osl(route_hint.estimated_output_tokens)
        estimated_iat = self.estimate_iat(route_hint.iat)

        osl = self.norm_level(estimated_osl, "MEDIUM")
        iat = self.norm_level(estimated_iat, "MEDIUM")

        last_w, last_group, reuse_budget = self.get_prefix(prefix_id)
        reuse_after = max(int(reuse_budget), 0)
        decode_cost = self.decode_cost(osl)
        iat_factor = self.iat_factor(iat)

        # 计算温度（用于参考，但不直接影响分数）
        temp = self.temp_base / (1.0 + float(reuse_after) * iat_factor)
        temp = min(max(temp, self.temp_min), self.temp_max)

        # 计算 KV 缓存重叠（所有 worker 共享，一次性计算）
        overlap_scores = self.kv_cache_manager.find_matches(
            token_ids=token_ids,
            model=route_hint.model,
        )
        logger.info(
            "overlap_scores: %s, len tokens ids=%d, osl=%d, iat=%s,"
            " last_w=%s, last_group=%s, reuse_after=%s,"
            " decode_cost=%.2f, iat_factor=%.2f, temp=%.2f",
            overlap_scores, len(token_ids), osl, iat,
            last_w, last_group, reuse_after,
            decode_cost, iat_factor, temp,
        )

        return SharedContext(
            overlap_scores=overlap_scores,
            osl=osl,
            iat=iat,
            last_w=last_w,
            last_group=last_group,
            reuse_after=reuse_after,
            decode_cost=decode_cost,
            iat_factor=iat_factor,
            temp=temp,
        )

    def select_worker_pair(
        self,
        workers: dict[str, WorkerInfo],
        groups: dict[str, list[str]],
        route_hint: RouteHint,
    ) -> tuple[WorkerInfo, WorkerInfo]:
        from openjiuwentools.infer_router.schemas.agent_hints import WorkerType

        logger.info(
            f"[ALGO: CTS] Starting worker pair selection for {len(groups)} groups, "
            f"prefix_id={route_hint.prefix_id}"
        )

        if not groups:
            raise ValueError("No groups available")

        # 阶段1：预计算共享上下文（find_matches + 参数计算）
        t0 = time.perf_counter()
        ctx = self._prepare_shared_context(route_hint)
        t_context = time.perf_counter()

        # 阶段2：per-group 打分
        best_pair = None
        best_score = float("-inf")

        score_params = ScoreParams(
            overlap_scores=ctx.overlap_scores,
            last_w=ctx.last_w,
            reuse_after=ctx.reuse_after,
            decode_cost=ctx.decode_cost,
            iat_factor=ctx.iat_factor,
            last_group=ctx.last_group,
        )

        for group_name, worker_ids in groups.items():
            workers_in_group = [workers[wid] for wid in worker_ids if wid in workers]

            prefill_workers = [
                w
                for w in workers_in_group
                if w.worker_type in (WorkerType.PREFILL, WorkerType.COMBINED)
            ]
            decode_workers = [
                w
                for w in workers_in_group
                if w.worker_type in (WorkerType.DECODE, WorkerType.COMBINED)
            ]

            if not prefill_workers or not decode_workers:
                continue

            prefill_scores = {
                w.worker_id: self.calculate_score(
                    w,
                    prefill_workers,
                    route_hint,
                    score_params=score_params,
                )
                for w in prefill_workers
            }
            decode_scores = {
                w.worker_id: self.calculate_score(
                    w,
                    decode_workers,
                    route_hint,
                    score_params=score_params,
                )
                for w in decode_workers
            }

            prefill_best_id = max(prefill_scores, key=prefill_scores.get)
            decode_best_id = max(decode_scores, key=decode_scores.get)

            prefill_best_score = prefill_scores[prefill_best_id]
            decode_best_score = decode_scores[decode_best_id]

            average_score = (2 * prefill_best_score * decode_best_score) / (
                prefill_best_score + decode_best_score
            )

            logger.info(
                f"[ALGO: CTS] Group {group_name}: prefill={prefill_best_id}({prefill_best_score:.3f}), "
                f"decode={decode_best_id}({decode_best_score:.3f}), avg_score={average_score:.3f}"
            )

            if average_score > best_score:
                best_score = average_score
                best_pair = (workers[prefill_best_id], workers[decode_best_id])

        t_scoring = time.perf_counter()

        if not best_pair:
            raise ValueError("No valid worker pair found")

        # 阶段3：前缀跟踪更新
        prefix_id = route_hint.prefix_id
        if prefix_id:
            token_ids = route_hint.token_ids or []
            prefill_overlap = (
                float(ctx.overlap_scores.get(best_pair[0].worker_id, 0.0)) / len(token_ids)
                if token_ids
                else 0.0
            )
            chosen_prefill_cost = self.prefill_cost_for_worker(token_ids, prefill_overlap)
            chosen_decode_cost = self.decode_cost(ctx.osl)

            chosen_group = best_pair[0].group
            self.update_prefix(PrefixUpdateParams(
                prefix_id=prefix_id,
                worker_id=best_pair[0].worker_id,
                initial_reuse_budget=route_hint.total_requests,
                decode_cost=chosen_decode_cost,
                prefill_cost=chosen_prefill_cost,
                iat_factor=ctx.iat_factor,
                group_id=chosen_group,
            ))

        t_end = time.perf_counter()

        context_ms = (t_context - t0) * 1000
        scoring_ms = (t_scoring - t_context) * 1000
        prefix_ms = (t_end - t_scoring) * 1000
        total_ms = (t_end - t0) * 1000

        logger.info(
            f"[ALGO: CTS] Selected {best_pair[0].worker_id} + {best_pair[1].worker_id}, "
            f"score={best_score:.3f} | "
            f"context(find_matches)={context_ms:.2f}ms, scoring={scoring_ms:.2f}ms, "
            f"prefix_update={prefix_ms:.2f}ms, total={total_ms:.2f}ms"
        )

        return best_pair

    def norm_level(self, level: str, default: str = "MEDIUM") -> str:
        """规范化级别参数"""
        if not level:
            return default
        level = level.upper()
        if level in ["LOW", "MEDIUM", "HIGH"]:
            return level
        return default

    def decode_cost(self, osl: str) -> float:
        """计算解码成本"""
        osl = self.norm_level(osl, "MEDIUM")
        cost_map = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0}
        return cost_map.get(osl, 2.0)

    def iat_factor(self, iat: str) -> float:
        """计算 IAT 因子"""
        iat = self.norm_level(iat, "MEDIUM")
        factor_map = {"LOW": 1.5, "MEDIUM": 1.0, "HIGH": 0.6}
        return factor_map.get(iat, 1.0)

    def get_prefix(self, prefix_id: str | None) -> tuple[str | None, str | None, int]:
        """获取前缀跟踪信息"""
        if not prefix_id or prefix_id not in self.prefix_tracking:
            return None, None, 0

        entry = self.prefix_tracking[prefix_id]
        if len(entry) == 3:
            last_worker, last_group, reuse_budget = entry
        else:
            last_worker, reuse_budget = entry
            last_group = None
        return last_worker, last_group, reuse_budget

    def update_prefix(self, params: PrefixUpdateParams):
        """更新前缀跟踪信息"""
        if not params.prefix_id:
            return

        if params.initial_reuse_budget is not None and params.prefix_id not in self.prefix_tracking:
            reuse_budget = params.initial_reuse_budget
        else:
            _, _, reuse_budget = self.get_prefix(params.prefix_id)
        self.prefix_tracking[params.prefix_id] = (
            params.worker_id, params.group_id, max(0, reuse_budget - 1),
        )
        self.prefix_meta[params.prefix_id] = {
            "decode_cost": float(params.decode_cost),
            "prefill_cost": float(params.prefill_cost),
            "iat_factor": float(params.iat_factor),
        }

    def prefill_cost_for_worker(self, token_ids: list[int] | None, overlap: float) -> float:
        """计算预填充成本"""
        if not token_ids:
            return 0.0

        input_len = len(token_ids)
        uncached_tokens = max(0, input_len * (1.0 - overlap))
        return (uncached_tokens / self.prefill_token_scale) * self.prefill_weight

    def _worker_outstanding(self, worker_id: str) -> tuple[int, float]:
        """计算 worker 的待处理工作量"""

        reuse_total = 0
        work_total = 0.0
        for pid, info in self.prefix_tracking.items():
            if not isinstance(info, tuple) or len(info) < 2:
                continue
            if len(info) == 3:
                last_worker, _, reuse_budget = info
            else:
                last_worker, reuse_budget = info
            if last_worker != worker_id:
                continue
            r = int(reuse_budget) if reuse_budget else 0
            reuse_total += r
            meta = self.prefix_meta.get(pid)
            if meta:
                work_total += (
                    float(r)
                    * (float(meta.get("decode_cost", 2.0)) + float(meta.get("prefill_cost", 0.0)))
                    * float(meta.get("iat_factor", 1.0))
                )
        return reuse_total, work_total

    def feature_vector(self, params: FeatureParams) -> list[float]:
        """构建特征向量（11维）"""
        import math

        worker = params.worker
        npu_usage = worker.current_load / 100.0
        queue_depth = float(worker.queue_depth)
        running_requests = float(worker.running_requests)
        _, work_out = self._worker_outstanding(worker.worker_id)

        penalties = (
            self.npu_penalty_weight * npu_usage
            + self.queue_penalty_weight * queue_depth
            + self.outstanding_work_weight * max(0.0, work_out)
        )

        inverse_load = 1.0 / (1.0 + penalties)
        affinity = 1.0 if (params.last_group and params.last_group == worker.group) else 0.0

        normalized_decode_cost = min(params.decode_cost / 3.0, 1.0)
        normalized_prefill_cost = math.tanh(params.prefill_cost)
        outstanding_norm = math.tanh(0.1 * work_out)
        reuse_norm = math.tanh(0.25 * float(max(params.reuse_after, 0)))
        queue_norm = math.tanh(0.1 * queue_depth)
        running_norm = math.tanh(0.05 * running_requests)

        return [
            1.0,
            inverse_load,
            params.overlap,
            affinity,
            outstanding_norm,
            normalized_decode_cost,
            normalized_prefill_cost,
            min(params.iat_factor / 1.5, 1.0),
            reuse_norm,
            queue_norm,
            running_norm,
        ]

    def lints_sample(self, worker_id: str, x: list[float]) -> float:
        """Linear Thompson Sampling 采样"""
        if worker_id not in self.lints_models:
            d = len(x)
            # 初始化 b_vec 为正值，确保基础分数为正
            # 这样即使没有历史数据，也不会产生大幅负值
            b_vec = [0.5 if i == 0 else 0.0 for i in range(d)]
            self.lints_models[worker_id] = (
                [[self.lints_lambda if i == j else 0.0 for j in range(d)] for i in range(d)],
                b_vec,
            )

        a, b_vec = self.lints_models[worker_id]

        # 计算后验分布参数
        try:
            # 简化的 Thompson Sampling：使用线性模型
            # score = x^T * w，其中 w ~ N(mean, cov)
            import random

            # 计算后验均值
            # mean = A^{-1} * b
            # 这里简化计算，直接使用b作为均值
            mean = b_vec

            # 添加高斯噪声进行探索
            # 使用较小的噪声，确保分数相对稳定
            score = 0.0
            for i, xi in enumerate(x):
                noise = random.gauss(0, self.lints_v * 0.5)  # 减小噪声以避免负分
                score += xi * (mean[i] + noise)

            # 确保分数非负
            return max(0.0, score)

        except Exception as e:
            logger.warning(f"[ALGO: CTS] LinTS sampling failed for {worker_id}: {e}")
            return random.uniform(0, 1)

    def ts_sample(self, worker_id: str) -> float:
        """Beta bandit 采样（探索奖励）"""
        if worker_id not in self.beta_params:
            self.beta_params[worker_id] = (1.0, 1.0)  # 均匀先验

        alpha, beta = self.beta_params[worker_id]
        import random

        return random.betavariate(alpha, beta)

    def load_score(self, worker: WorkerInfo, job_cost_total: float) -> float:
        """计算负载得分"""
        npu_usage = worker.current_load / 100.0
        queue_depth = float(worker.queue_depth)
        outstanding_work = worker.current_load / 100.0

        penalty = (
            self.npu_penalty_weight * npu_usage
            + self.queue_penalty_weight * queue_depth
            + self.outstanding_work_weight * outstanding_work
            + self.job_npu_coupling_weight * job_cost_total * npu_usage
            + self.job_queue_coupling_weight * job_cost_total * queue_depth
        )

        return 1.0 / (1.0 + max(0.0, penalty))

    def calculate_score(
        self,
        worker: WorkerInfo,
        workers: list[WorkerInfo],
        route_hint: RouteHint,
        score_params: ScoreParams | None = None,
    ) -> float:
        """计算单个工作器的分数

        Args:
            worker: 要计算分数的工作器
            workers: 所有可用工作器列表
            route_hint: 路由提示信息
            score_params: 预计算的共享参数（可选），未提供时自动计算

        Returns:
            工作器的分数（越高越好）
        """
        logger.debug(f"[ALGO: CTS] Calculating score for worker {worker.worker_id}")

        token_ids = route_hint.token_ids or []

        # 如果没有预计算的共享参数，则自己计算
        if score_params is None:
            score_params = ScoreParams()

        overlap_scores = score_params.overlap_scores
        last_w = score_params.last_w
        reuse_after = score_params.reuse_after
        decode_cost = score_params.decode_cost
        iat_factor = score_params.iat_factor
        last_group = score_params.last_group

        has_missing = (
            overlap_scores is None or reuse_after is None
            or decode_cost is None or iat_factor is None
        )
        if has_missing:
            ctx = self._prepare_shared_context(route_hint)
            overlap_scores = ctx.overlap_scores
            last_w = ctx.last_w
            last_group = ctx.last_group
            reuse_after = ctx.reuse_after
            decode_cost = ctx.decode_cost
            iat_factor = ctx.iat_factor

        # 步骤3：计算各项分数组成（worker 特有）
        # 3.1 KV缓存重叠
        overlap = (
            float(overlap_scores.get(worker.worker_id, 0.0)) / len(token_ids) if token_ids else 0.0
        )
        logger.debug(f"[ALGO: CTS] {worker.worker_id}: overlap={overlap:.3f}")

        # 3.2 预填充成本
        prefill_cost = self.prefill_cost_for_worker(token_ids, overlap)
        logger.debug(f"[ALGO: CTS] {worker.worker_id}: prefill_cost={prefill_cost:.3f}")

        # 3.3 总作业成本
        job_cost_total = decode_cost + prefill_cost
        logger.debug(f"[ALGO: CTS] {worker.worker_id}: job_cost_total={job_cost_total:.3f}")

        # 3.4 特征向量
        x = self.feature_vector(FeatureParams(
            worker=worker,
            overlap=overlap,
            last_w=last_w,
            reuse_after=reuse_after,
            decode_cost=decode_cost,
            prefill_cost=prefill_cost,
            iat_factor=iat_factor,
            last_group=last_group,
        ))
        logger.debug(f"[ALGO: CTS] {worker.worker_id}: features={[f'{v:.3f}' for v in x]}")

        # 3.5 LinTS 评分
        lints_score = self.lints_sample(worker.worker_id, x)
        logger.debug(f"[ALGO: CTS] {worker.worker_id}: lints_score={lints_score:.3f}")

        # 3.6 探索奖励
        explore_w = self.base_ts_weight / (1.0 + float(reuse_after) * iat_factor)
        explore_bonus = explore_w * self.ts_sample(worker.worker_id)
        logger.debug(f"[ALGO: CTS] {worker.worker_id}: explore_bonus={explore_bonus:.3f}")

        val = lints_score + explore_bonus
        logger.debug(
            "[ALGO: CTS] %s: val_before_modifiers=%.3f,"
            " last_group=%s, reuse_after=%s",
            worker.worker_id, val, last_group, reuse_after,
        )

        is_same_group = last_group is not None and worker.group == last_group

        # 3.7 亲和性奖励（group 级别）
        if is_same_group and reuse_after > 0:
            affinity_bonus = (
                self.affinity_base
                + self.affinity_reuse_weight * float(reuse_after)
                + self.affinity_iat_weight * iat_factor
            ) * (0.5 + 0.5 * overlap)
            val += affinity_bonus
            logger.debug(f"[ALGO: CTS] {worker.worker_id}: affinity_bonus={affinity_bonus:.3f} (group={worker.group})")

        # 3.8 切换惩罚（group 级别）
        if last_group is not None and not is_same_group and reuse_after > 0:
            raw_switch_penalty = (
                self.switch_cost_base
                + self.switch_cost_reuse * float(reuse_after)
                + self.switch_cost_iat * iat_factor
            )

            val -= raw_switch_penalty
            logger.debug(
                f"[ALGO: CTS] {worker.worker_id}: switch_penalty={raw_switch_penalty:.3f}, "
                f"worker_group={worker.group}, last_group={last_group}, "
                f"val_after_penalty={val:.3f}"
            )

        # 3.9 负载平衡
        load_mod = self.load_score(worker, job_cost_total)
        if is_same_group and reuse_after > 0:
            load_mod = max(load_mod, self.sticky_load_floor)
        val *= load_mod
        logger.debug(
            f"[ALGO: CTS] {worker.worker_id}: load_mod={load_mod:.3f}, final_score={val:.3f}"
        )

        # 3.10 合理性检查
        if val != val or val == float("inf") or val == float("-inf"):
            val = -1e9
            logger.warning(f"[ALGO: CTS] {worker.worker_id}: Invalid score, set to -1e9")

        logger.debug(f"[ALGO: CTS] Worker {worker.worker_id} score: {val:.3f}")

        return float(val)

    def select_worker(self, workers: list[WorkerInfo], route_hint: RouteHint) -> WorkerInfo:
        """Combined 模式路由：优先 overlap+affinity，overlap 差别 <10% 时回退到负载选择"""
        logger.info(
            f"[ALGO: CTS] Starting combined-mode selection for {len(workers)} workers"
        )

        if not workers:
            raise ValueError("No workers available")

        t0 = time.perf_counter()

        ctx = self._prepare_shared_context(route_hint)
        t_context = time.perf_counter()

        token_ids = route_hint.token_ids or []
        token_len = len(token_ids) if token_ids else 1

        sp = ScoreParams(
            overlap_scores=ctx.overlap_scores,
            last_w=ctx.last_w,
            reuse_after=ctx.reuse_after,
            decode_cost=ctx.decode_cost,
            iat_factor=ctx.iat_factor,
            last_group=ctx.last_group,
        )

        worker_scores = {}
        for w in workers:
            score = self.calculate_score(
                w, workers, route_hint,
                score_params=sp,
            )
            raw_overlap = float(ctx.overlap_scores.get(w.worker_id, 0.0))
            overlap_ratio = raw_overlap / token_len
            worker_scores[w.worker_id] = {
                "score": score,
                "overlap": raw_overlap,
                "overlap_ratio": overlap_ratio,
            }

        max_overlap = max(s["overlap"] for s in worker_scores.values())

        # overlap 差值在 token_len 的 10% 以内视为"差别不大"
        overlap_threshold = token_len * 0.10
        top_overlap_workers = [
            w for w in workers
            if max_overlap - worker_scores.get(w.worker_id, {}).get("overlap", 0.0) <= overlap_threshold
        ]

        if len(top_overlap_workers) > 1:
            import math
            import random

            loads = []
            for w in top_overlap_workers:
                workload = self._get_worker_workload(w.worker_id)
                try:
                    pending_tokens = int(workload.pending_tokens) if workload else 0
                    pending_osl = int(workload.pending_osl) if workload else 0
                    prompt_ratio = float(getattr(workload, "prompt_ratio", 1.0)) if workload else 1.0
                    completion_ratio = (
                        float(getattr(workload, "completion_ratio", 1.0)) if workload else 1.0
                    )
                except (TypeError, ValueError):
                    pending_tokens = 0
                    pending_osl = 0
                    prompt_ratio = 1.0
                    completion_ratio = 1.0
                loads.append(pending_tokens / prompt_ratio + pending_osl / completion_ratio)

            neg_loads = [-ld for ld in loads]
            max_neg = max(neg_loads)
            exp_scores = [math.exp((s - max_neg) / ctx.temp) for s in neg_loads]
            sum_exp = sum(exp_scores)
            probs = [e / sum_exp for e in exp_scores]

            r = random.random()
            cum = 0.0
            chosen_idx = 0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    chosen_idx = i
                    break
            chosen_worker = top_overlap_workers[chosen_idx]
            selection_reason = "load_tiebreak"
        else:
            chosen_worker = max(workers, key=lambda w: worker_scores.get(w.worker_id, {}).get("score", 0.0))
            selection_reason = "overlap_affinity"

        t_scoring = time.perf_counter()

        chosen_worker_id = chosen_worker.worker_id
        chosen_info = worker_scores.get(chosen_worker_id, {"score": 0.0, "overlap": 0.0, "overlap_ratio": 0.0})

        logger.debug("[ALGO: CTS] Worker scores (combined mode):")
        for w in workers:
            s = worker_scores.get(w.worker_id, {"score": 0.0, "overlap": 0.0, "overlap_ratio": 0.0})
            logger.debug(
                f"[ALGO: CTS]   {w.worker_id}: score={s['score']:.3f}, "
                f"overlap={s['overlap']:.0f}/{token_len} ({s['overlap_ratio']:.2%})"
                f"{' <- SELECTED' if w.worker_id == chosen_worker_id else ''}"
            )
        logger.debug(
            f"[ALGO: CTS] Selection reason: {selection_reason}, "
            f"max_overlap={max_overlap:.0f}, threshold={overlap_threshold:.0f}, "
            f"top_overlap_count={len(top_overlap_workers)}"
        )

        overlap_ratio = chosen_info["overlap_ratio"]
        chosen_prefill_cost = self.prefill_cost_for_worker(token_ids, overlap_ratio)
        chosen_decode_cost = self.decode_cost(ctx.osl)

        prefix_id = route_hint.prefix_id
        if prefix_id:
            self.update_prefix(PrefixUpdateParams(
                prefix_id=prefix_id,
                worker_id=chosen_worker_id,
                initial_reuse_budget=route_hint.total_requests,
                decode_cost=chosen_decode_cost,
                prefill_cost=chosen_prefill_cost,
                iat_factor=ctx.iat_factor,
                group_id=chosen_worker.group,
            ))

        t_end = time.perf_counter()

        context_ms = (t_context - t0) * 1000
        scoring_ms = (t_scoring - t_context) * 1000
        total_ms = (t_end - t0) * 1000

        logger.info(
            f"[ALGO: CTS] Selected {chosen_worker_id} ({selection_reason}) | "
            f"score={chosen_info['score']:.3f}, overlap={chosen_info['overlap']:.0f}, "
            f"context={context_ms:.2f}ms, scoring={scoring_ms:.2f}ms, total={total_ms:.2f}ms"
        )
        return chosen_worker

    def estimate_osl(self, estimated_output_tokens: int | None) -> str:
        """估算输出序列长度级别"""
        if not estimated_output_tokens:
            return "MEDIUM"

        if estimated_output_tokens <= 128:
            return "LOW"
        elif estimated_output_tokens <= 512:
            return "MEDIUM"
        else:
            return "HIGH"

    def estimate_iat(self, iat: int | None) -> str:
        """估算到达时间间隔级别"""
        if not iat:
            return "MEDIUM"

        if iat <= 100:
            return "LOW"
        elif iat <= 1000:
            return "MEDIUM"
        else:
            return "HIGH"

    def update_feedback(self, worker_id: str, reward: float):
        """更新反馈（用于强化学习）"""
        # 更新 Beta bandit 参数
        if worker_id not in self.beta_params:
            self.beta_params[worker_id] = (1.0, 1.0)

        alpha, beta = self.beta_params[worker_id]
        if reward > 0:
            self.beta_params[worker_id] = (alpha + 1, beta)
        else:
            self.beta_params[worker_id] = (alpha, beta + 1)

        # 更新 LinTS 模型（简化版本）
        if worker_id in self.lints_models:
            a, b = self.lints_models[worker_id]
            # 简化的在线更新
            for i, _ in enumerate(b):
                b[i] = b[i] * self.lints_forget + (reward if i == 0 else 0)
            self.lints_models[worker_id] = (a, b)

        logger.info(f"[ALGO: CTS] Updated feedback for {worker_id}: reward={reward:.3f}")
