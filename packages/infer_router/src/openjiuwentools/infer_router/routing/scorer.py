"""插件式 Scorer 框架。

仿照 llm-d EndpointPickerConfig 的设计，每个 Scorer 是一个独立插件：
  - 实现 score() 方法，对每个 worker 返回 [0, 1] 分数
  - 通过 YAML 配置声明启用哪些 scorer 及其权重
  - PluginScorerAlgorithm 按加权求和 + argmax 选择 worker

配置示例（config.yaml）：

    scoring_plugins:
      - type: prefix-cache-scorer
        weight: 3.0
      - type: cache-location-scorer
        weight: 3.0
        parameters:
          npu_weight: 1.0
          ram_weight: 0.6
          disk_weight: 0.2
      - type: kv-cache-utilization-scorer
        weight: 2.0
      - type: queue-scorer
        weight: 2.0
      - type: latency-predict-scorer
        weight: 5.0
        parameters:
          prediction_server_url: "http://localhost:8001"
          training_server_url: "http://localhost:8000"
          predict_timeout: 0.5
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from loguru import logger

from openjiuwentools.infer_router.kv_cache.event import CacheLocation
from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.routing.load_balance import LoadBalancingAlgorithm
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint, WorkerInfo, WorkerType


# ---------------------------------------------------------------------------
# Scorer 基类
# ---------------------------------------------------------------------------


class Scorer:
    """Scorer 插件基类，每个插件对 worker 打分 [0, 1]。"""

    scorer_type: str = ""

    def __init__(self, name: str | None = None, parameters: dict | None = None):
        self.name = name or self.scorer_type

    def score(
        self,
        worker: WorkerInfo,
        all_workers: list[WorkerInfo],
        route_hint: RouteHint,
        context: ScoringContext,
    ) -> float:
        """对单个 worker 打分，返回 [0, 1]。"""
        raise NotImplementedError


class ScoringContext:
    """在一次路由决策中所有 scorer 共享的上下文（只计算一次）。"""

    def __init__(
        self,
        overlap_scores: dict,
        total_tokens: int,
        any_hit: bool,
        worker_ids: list[str],
        inflight_counts: dict[str, int] | None = None,
        location_token_counts: dict[str, dict[CacheLocation, int]] | None = None,
        overlap_source: str = "none",
        overlap_timing: dict[str, float] | None = None,
    ):
        self.overlap_scores = overlap_scores
        self.total_tokens = total_tokens
        self.any_hit = any_hit
        self.worker_ids = worker_ids
        self.inflight_counts = inflight_counts or {}
        self.location_token_counts = location_token_counts or {}
        self.overlap_source = overlap_source
        self.overlap_timing = overlap_timing or {}


# ---------------------------------------------------------------------------
# Scorer 注册表
# ---------------------------------------------------------------------------

_SCORER_REGISTRY: dict[str, type[Scorer]] = {}


def register_scorer(cls: type[Scorer]) -> type[Scorer]:
    """Register a scorer class without wrapping or replacing its identity."""
    if cls.scorer_type:
        _SCORER_REGISTRY[cls.scorer_type] = cls
    return cls


def get_scorer_registry() -> dict[str, type[Scorer]]:
    return _SCORER_REGISTRY


def create_scorer(scorer_type: str, name: str | None = None, parameters: dict | None = None) -> Scorer:
    """工厂函数：按 type 创建 Scorer 实例。"""
    cls = _SCORER_REGISTRY.get(scorer_type)
    if cls is None:
        raise ValueError(f"Unknown scorer type '{scorer_type}'. Available: {list(_SCORER_REGISTRY.keys())}")
    return cls(name=name or scorer_type, parameters=parameters)


# ---------------------------------------------------------------------------
# 内置 Scorer 实现
# ---------------------------------------------------------------------------


class PrefixCacheScorer(Scorer):
    """前缀缓存命中率评分：overlap / total_tokens。"""

    scorer_type = "prefix-cache-scorer"

    def score(self, worker, all_workers, route_hint, context):
        if context.total_tokens == 0:
            return 0.0
        return min(
            float(context.overlap_scores.get(worker.worker_id, 0.0)) / context.total_tokens,
            1.0,
        )


register_scorer(PrefixCacheScorer)


class CacheLocationScorer(Scorer):
    """按连续前缀 KV block 所在层级计算缓存质量分数。

    参数:
        name: scorer 实例名称。
        parameters: 可选权重字典，支持 npu_weight、ram_weight/cpu_weight
            和 disk_weight/remote_weight，取值会限制在 [0, 1]。

    返回:
        ``score`` 为单个 Worker 返回 [0, 1] 分数；完整 NPU 命中为 1，
        RAM/DISK 命中按配置权重折算，未命中部分为 0。
    """

    scorer_type = "cache-location-scorer"

    def __init__(self, name=None, parameters=None):
        super().__init__(name, parameters)
        params = parameters or {}
        self.npu_weight = self._weight(params.get("npu_weight", 1.0))
        self.ram_weight = self._weight(params.get("ram_weight", params.get("cpu_weight", 0.6)))
        self.disk_weight = self._weight(params.get("disk_weight", params.get("remote_weight", 0.2)))

    @staticmethod
    def _weight(value: Any) -> float:
        """将配置权重转换并限制到合法范围。

        参数:
            value: YAML parameters 中的数字或可转换为数字的字符串。

        返回:
            限制在 [0, 1] 范围内的浮点权重。
        """

        return max(0.0, min(float(value), 1.0))

    def score(self, worker, all_workers, route_hint, context):
        """计算一个 Worker 的 location 加权前缀命中分数。

        参数:
            worker: 当前待评分 Worker。
            all_workers: 本次调度的全部候选 Worker。
            route_hint: 当前请求路由提示。
            context: 包含各 Worker location 命中 token 数的共享评分上下文。

        返回:
            location 加权命中 token 数除以请求总 token 数，范围为 [0, 1]。
        """

        if context.total_tokens <= 0:
            return 0.0
        counts = context.location_token_counts.get(worker.worker_id, {})
        weighted_tokens = (
            counts.get(CacheLocation.NPU, 0) * self.npu_weight
            + counts.get(CacheLocation.CPU, 0) * self.ram_weight
            + counts.get(CacheLocation.REMOTE, 0) * self.disk_weight
        )
        return max(0.0, min(weighted_tokens / context.total_tokens, 1.0))


register_scorer(CacheLocationScorer)


class KVCacheUtilizationScorer(Scorer):
    """KV 缓存利用率评分：空闲比例越高得分越高。"""

    scorer_type = "kv-cache-utilization-scorer"

    def score(self, worker, all_workers, route_hint, context):
        return max(0.0, 1.0 - worker.current_load / 100.0)


register_scorer(KVCacheUtilizationScorer)


class QueueScorer(Scorer):
    """队列深度评分：queue_depth 越低得分越高（min-max 归一化）。"""

    scorer_type = "queue-scorer"

    def score(self, worker, all_workers, route_hint, context):
        queue_sizes = [w.queue_depth for w in all_workers]
        min_q, max_q = min(queue_sizes), max(queue_sizes)
        if max_q == min_q:
            return 1.0
        return (max_q - worker.queue_depth) / (max_q - min_q)


register_scorer(QueueScorer)


class NoHitLRUScorer(Scorer):
    """冷请求 LRU 扩散评分：无缓存命中时优先选最近最少使用的 worker。"""

    scorer_type = "no-hit-lru-scorer"

    def __init__(self, name=None, parameters=None):
        super().__init__(name, parameters)
        self._cold_lru: list[str] = []

    def score(self, worker, all_workers, route_hint, context):
        if context.any_hit:
            return 0.5
        total = len(context.worker_ids)
        if total <= 1:
            return 1.0
        never_used = [wid for wid in context.worker_ids if wid not in self._cold_lru]
        if worker.worker_id not in self._cold_lru:
            rank = never_used.index(worker.worker_id)
        else:
            rank = len(never_used) + self._cold_lru.index(worker.worker_id)
        return max(0.0, 1.0 - float(rank) / float(total - 1))

    def update_lru(self, worker_id: str):
        if worker_id in self._cold_lru:
            self._cold_lru.remove(worker_id)
        self._cold_lru.append(worker_id)


register_scorer(NoHitLRUScorer)


class ActiveRequestScorer(Scorer):
    """活跃请求数评分：running_requests 越少得分越高。"""

    scorer_type = "active-request-scorer"

    def score(self, worker, all_workers, route_hint, context):
        running = [w.running_requests for w in all_workers]
        min_r, max_r = min(running), max(running)
        if max_r == min_r:
            return 1.0
        return (max_r - worker.running_requests) / (max_r - min_r)


register_scorer(ActiveRequestScorer)


class LoadAwareScorer(Scorer):
    """负载感知评分：综合 current_load、queue_depth、running_requests。"""

    scorer_type = "load-aware-scorer"

    def __init__(self, name=None, parameters=None):
        super().__init__(name, parameters)
        params = parameters or {}
        self.queue_threshold = params.get("queue_threshold", 10)

    def score(self, worker, all_workers, route_hint, context):
        load_score = max(0.0, 1.0 - worker.current_load / 100.0)
        queue_score = max(0.0, 1.0 - worker.queue_depth / max(self.queue_threshold, 1))
        return 0.5 * load_score + 0.5 * queue_score


register_scorer(LoadAwareScorer)


class InflightScorer(Scorer):
    """Router侧inflight请求数评分：使用平滑绝对函数避免相邻值被极端放大。

    与 queue-scorer / active-request-scorer 不同，inflight 计数在 router 侧
    实时更新（路由时 +1，完成时 -1），不依赖 worker ZMQ 上报，无延迟。
    """

    scorer_type = "inflight-scorer"

    def __init__(self, name=None, parameters=None):
        super().__init__(name, parameters)
        params = parameters or {}
        self.scale = max(float(params.get("scale", 8.0)), 1.0)

    def score(self, worker, all_workers, route_hint, context):
        mine = max(float(context.inflight_counts.get(worker.worker_id, 0)), 0.0)
        return 1.0 / (1.0 + mine / self.scale)


register_scorer(InflightScorer)


class LatencyPredictScorer(Scorer):
    """延迟预测评分：调用外部 prediction_server 预测 TTFT+TPOT，归一化为 [0, 1]。

    预测失败时返回 0.5（中性分数），不阻塞其他 scorer 的评分。
    """

    scorer_type = "latency-predict-scorer"

    def __init__(self, name=None, parameters=None):
        super().__init__(name, parameters)
        params = parameters or {}
        self.prediction_server_url = params.get("prediction_server_url", "http://localhost:8001").rstrip("/")
        self.training_server_url = params.get("training_server_url", "http://localhost:8000").rstrip("/")
        self.predict_timeout = params.get("predict_timeout", 0.5)
        self.training_report_enabled = params.get("training_report_enabled", True)
        self.ttft_weight = params.get("ttft_weight", 0.8)
        self.tpot_weight = params.get("tpot_weight", 0.2)
        self._sync_client = None
        self._async_client = None
        self._last_predictions: dict[str, dict] = {}

    def set_predictions(self, predictions: dict[str, dict] | None) -> None:
        """Publish one bulk prediction batch for subsequent worker scoring."""
        self._last_predictions = predictions or {}

    def _get_sync_client(self):
        import httpx

        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.predict_timeout)
        return self._sync_client

    def _get_async_client(self):
        import httpx

        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.predict_timeout)
        return self._async_client

    @staticmethod
    def _build_pred_request(worker, route_hint, overlap_scores):
        token_ids = route_hint.token_ids or []
        input_len = len(token_ids)
        overlap = float(overlap_scores.get(worker.worker_id, 0.0))
        prefix_score = (overlap / input_len) if input_len > 0 else 0.0

        pod_type = ""
        if worker.worker_type == WorkerType.PREFILL:
            pod_type = "prefill"
        elif worker.worker_type == WorkerType.DECODE:
            pod_type = "decode"

        return {
            "kv_cache_percentage": min(worker.current_load / 100.0, 1.0),
            "input_token_length": input_len,
            "num_request_waiting": worker.queue_depth,
            "num_request_running": worker.running_requests,
            "num_tokens_generated": 1,
            "prefix_cache_score": min(max(prefix_score, 0.0), 1.0),
            "pod_type": pod_type,
        }

    def predict_bulk(self, workers, route_hint, overlap_scores) -> dict[str, dict] | None:
        """批量预测所有 worker 的延迟，缓存结果供 score() 使用。"""

        requests = [self._build_pred_request(w, route_hint, overlap_scores) for w in workers]
        try:
            client = self._get_sync_client()
            resp = client.post(
                f"{self.prediction_server_url}/predict/bulk/strict",
                json={"requests": requests},
            )
            if resp.status_code == 200:
                preds = resp.json().get("predictions", [])
                if len(preds) == len(workers):
                    result = {}
                    for w, pred in zip(workers, preds):
                        result[w.worker_id] = pred
                    logger.info("[LatencyPredictScorer] Predicted {} workers", len(workers))
                    return result
            logger.warning(
                "[LatencyPredictScorer] Prediction server returned {}",
                resp.status_code,
            )
        except Exception as e:
            logger.warning("[LatencyPredictScorer] Prediction failed: {}", e)
        return None

    def score(self, worker, all_workers, route_hint, context):
        pred = self._last_predictions.get(worker.worker_id)
        if pred is None:
            return 0.5

        ttft = pred.get("ttft_ms", 0.0)
        tpot = pred.get("tpot_ms", 0.0)

        all_ttfts, all_tpots = [], []
        for w in all_workers:
            p = self._last_predictions.get(w.worker_id)
            if p:
                all_ttfts.append(p.get("ttft_ms", 0.0))
                all_tpots.append(p.get("tpot_ms", 0.0))

        if not all_ttfts:
            return 0.5

        min_ttft, max_ttft = min(all_ttfts), max(all_ttfts)
        min_tpot, max_tpot = min(all_tpots), max(all_tpots)
        ttft_range = max_ttft - min_ttft
        tpot_range = max_tpot - min_tpot

        # 归一化权重；某个维度无差异时权重归零
        alpha, beta = self.ttft_weight, self.tpot_weight
        if ttft_range <= 1e-9 and tpot_range > 1e-9:
            alpha, beta = 0.0, 1.0
        elif tpot_range <= 1e-9 and ttft_range > 1e-9:
            alpha, beta = 1.0, 0.0
        else:
            total = alpha + beta
            if total > 0:
                alpha, beta = alpha / total, beta / total

        # 分别 min-max 归一化，延迟越低分数越高
        n_ttft = (max_ttft - ttft) / ttft_range if ttft_range > 1e-9 else 0.5
        n_tpot = (max_tpot - tpot) / tpot_range if tpot_range > 1e-9 else 0.5

        return max(0.0, min(1.0, alpha * n_ttft + beta * n_tpot))

    async def report_training_data(
        self,
        *,
        worker: WorkerInfo,
        route_hint: RouteHint,
        overlap_scores: dict,
        actual_ttft_ms: float,
        actual_tpot_ms: float,
        completion_tokens: int,
    ):
        logger.info(
            "[LatencyPredictScorer] report_training_data called: worker={}, enabled={}, url={}",
            worker.worker_id,
            self.training_report_enabled,
            self.training_server_url,
        )
        if not self.training_report_enabled:
            logger.info("[LatencyPredictScorer] training_report_enabled=False, skipped")
            return

        entry = self._build_pred_request(worker, route_hint, overlap_scores)
        entry["actual_ttft_ms"] = max(0.0, actual_ttft_ms)
        entry["actual_tpot_ms"] = max(0.0, actual_tpot_ms)
        entry["num_tokens_generated"] = max(completion_tokens, 1)

        try:
            client = self._get_async_client()
            resp = await client.post(
                f"{self.training_server_url}/add_training_data_bulk",
                json={"entries": [entry]},
                timeout=2.0,
            )
            logger.info(
                "[LatencyPredictScorer] Training data reported for {}, status={}, response={}",
                worker.worker_id,
                resp.status_code,
                resp.text,
            )
        except Exception as e:
            logger.info("[LatencyPredictScorer] Training report failed: {}", e)

    def report_training_data_fire_and_forget(self, *, loop: asyncio.AbstractEventLoop | None = None, **kwargs):
        """异步上报训练数据（fire-and-forget）。

        支持两种调用场景：
        - 在 asyncio 事件循环线程内调用（api/server.py）：直接 create_task
        - 在非事件循环线程调用（data_plane/server.py 的 Rust 回调）：通过 loop 参数跨线程调度
        """
        try:
            running_loop = asyncio.get_running_loop()
            logger.info("[LatencyPredictScorer] fire_and_forget: using running loop (create_task)")
            running_loop.create_task(self.report_training_data(**kwargs))
        except RuntimeError:
            if loop is not None:
                logger.info("[LatencyPredictScorer] fire_and_forget: cross-thread via run_coroutine_threadsafe")
                future = asyncio.run_coroutine_threadsafe(self.report_training_data(**kwargs), loop)
                future.add_done_callback(
                    lambda f: logger.info(
                        "[LatencyPredictScorer] fire_and_forget result: {}",
                        f.exception() or "ok",
                    )
                )
            else:
                logger.info("[LatencyPredictScorer] fire_and_forget: no event loop available, skipped")


register_scorer(LatencyPredictScorer)


def _log_scoring_details(
    request_id: str,
    phase: str,
    details: list[tuple[str, float, list[tuple[str, float, float]]]],
    workers: list[WorkerInfo],
    context: ScoringContext,
    selected_worker_id: str,
    diagnostic_workers: list[WorkerInfo] | None = None,
    diagnostic_inflight_counts: dict[str, int] | None = None,
) -> None:
    """记录每个 worker 的评分明细，帮助诊断负载不均问题。"""
    workers_by_id = {worker.worker_id: worker for worker in workers}
    candidate_parts = []
    for wid, total, breakdown in details:
        worker = workers_by_id[wid]
        location_counts = context.location_token_counts.get(wid, {})
        score_parts = [
            f"{scorer_type}:{raw:.3f}*{weight:.1f}={raw * weight:.3f}" for scorer_type, raw, weight in breakdown
        ]
        candidate_parts.append(
            f"{wid}(selected={wid == selected_worker_id},total={total:.3f},"
            f"load={worker.current_load:.3f},queue={worker.queue_depth},"
            f"running={worker.running_requests},"
            f"inflight={context.inflight_counts.get(wid, 0)},"
            f"overlap_tokens={context.overlap_scores.get(wid, 0)},"
            f"npu_tokens={location_counts.get(CacheLocation.NPU, 0)},"
            f"ram_tokens={location_counts.get(CacheLocation.CPU, 0)},"
            f"disk_tokens={location_counts.get(CacheLocation.REMOTE, 0)},"
            f"scores=[{','.join(score_parts)}])"
        )
    scored_worker_ids = set(workers_by_id)
    not_scored_parts = []
    diagnostic_inflight_counts = diagnostic_inflight_counts or {}
    for worker in diagnostic_workers or []:
        if worker.worker_id in scored_worker_ids:
            continue
        not_scored_parts.append(
            f"{worker.worker_id}(eligible=false,load={worker.current_load:.3f},"
            f"queue={worker.queue_depth},running={worker.running_requests},"
            f"inflight={diagnostic_inflight_counts.get(worker.worker_id, 0)})"
        )
    logger.info(
        "[PluginScorerDetail] request={} phase={} any_hit={} candidates={} not_scored=[{}]",
        request_id,
        phase,
        context.any_hit,
        ";".join(candidate_parts),
        ";".join(not_scored_parts),
    )


# ---------------------------------------------------------------------------
# PluginScorerAlgorithm — 插件式算法入口
# ---------------------------------------------------------------------------


class PluginScorerAlgorithm(LoadBalancingAlgorithm):
    """插件式评分算法：从配置加载 scorer 列表及权重，加权求和选 worker。

    支持 PD 分离模式下 prefill/decode 使用独立的 scorer 配置。

    config.yaml 示例：
        scoring_plugins:          # combined 模式 / 回退默认
          - type: prefix-cache-scorer
            weight: 3.0
          - type: queue-scorer
            weight: 2.0

        prefill_scoring_plugins:  # PD 分离时 prefill 专用
          - type: prefix-cache-scorer
            weight: 3.0
          - type: queue-scorer
            weight: 1.0

        decode_scoring_plugins:   # PD 分离时 decode 专用
          - type: queue-scorer
            weight: 3.0
          - type: inflight-scorer
            weight: 2.0
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        worker_manager=None,
        *,
        plugin_configs: list[dict] | None = None,
        prefill_plugin_configs: list[dict] | None = None,
        decode_plugin_configs: list[dict] | None = None,
    ):
        self.kv_cache_manager = kv_cache_manager
        self.worker_manager = worker_manager
        self._inflight_tracker = None

        self.weighted_scorers = self._load_scorers(plugin_configs, "combined")
        if not self.weighted_scorers:
            logger.warning("[PluginScorer] No scorers configured, using defaults")
            self.weighted_scorers = [
                (PrefixCacheScorer(), 3.0),
                (KVCacheUtilizationScorer(), 2.0),
                (QueueScorer(), 2.0),
                (NoHitLRUScorer(), 2.0),
            ]

        self.prefill_scorers = self._load_scorers(prefill_plugin_configs, "prefill")
        self.decode_scorers = self._load_scorers(decode_plugin_configs, "decode")

    @staticmethod
    def _load_scorers(
        configs: list[dict] | None,
        profile: str,
    ) -> list[tuple[Scorer, float]]:
        """从配置列表加载 scorer 实例。"""
        if not configs:
            return []
        scorers: list[tuple[Scorer, float]] = []
        for cfg in configs:
            scorer_type = cfg.get("type", "")
            weight = float(cfg.get("weight", 1.0))
            name = cfg.get("name", scorer_type)
            parameters = cfg.get("parameters")
            scorer = create_scorer(scorer_type, name=name, parameters=parameters)
            scorers.append((scorer, weight))
            logger.info(
                "[PluginScorer] Loaded scorer '{}' (type={}, weight={:.1f}, profile={})",
                name,
                scorer_type,
                weight,
                profile,
            )
        return scorers

    def set_inflight_tracker(self, tracker) -> None:
        """注入 Router 侧的 InflightTracker，供 InflightScorer 使用。"""
        self._inflight_tracker = tracker

    def _build_context(
        self,
        workers: list[WorkerInfo],
        route_hint: RouteHint,
        scorers: list[tuple[Scorer, float]] | None = None,
    ) -> ScoringContext:
        token_ids = route_hint.token_ids or []
        total_tokens = len(token_ids)
        active_scorers = scorers or self.weighted_scorers
        needs_overlap = any(
            isinstance(scorer, (PrefixCacheScorer, LatencyPredictScorer)) for scorer, _ in active_scorers
        )
        needs_location = any(isinstance(scorer, CacheLocationScorer) for scorer, _ in active_scorers)
        overlap_timing: dict[str, float] = {}
        precomputed = route_hint.precomputed_overlap_scores
        if not token_ids or not needs_overlap:
            overlap_scores = {}
            overlap_source = "skipped"
        elif precomputed is not None:
            worker_ids = {worker.worker_id for worker in workers}
            overlap_scores = {worker_id: score for worker_id, score in precomputed.items() if worker_id in worker_ids}
            overlap_source = "reused"
        else:
            worker_ids = {worker.worker_id for worker in workers}
            overlap_scores = self.kv_cache_manager.find_matches(
                token_ids=token_ids,
                model=route_hint.model,
                include_workers=worker_ids,
                timing=overlap_timing,
            )
            overlap_source = "computed"

        location_token_counts: dict[str, dict[CacheLocation, int]] = {}
        if token_ids and needs_location:
            prefix_snapshots = getattr(route_hint, "precomputed_prefix_snapshots", None) or {}
            for worker in workers:
                snapshot = prefix_snapshots.get(worker.worker_id)
                if snapshot is None:
                    counts = self.kv_cache_manager.get_matched_block_location_token_counts(
                        worker.worker_id,
                        token_ids,
                    )
                else:
                    counts = self.kv_cache_manager.get_matched_block_location_token_counts(
                        worker.worker_id,
                        token_ids,
                        snapshot=snapshot,
                    )
                location_token_counts[worker.worker_id] = counts

        any_hit = False
        if total_tokens > 0:
            max_ratio = max(
                (float(overlap_scores.get(w.worker_id, 0.0)) / total_tokens for w in workers),
                default=0.0,
            )
            any_hit = max_ratio > 0.01
        if not any_hit and location_token_counts:
            any_hit = any(sum(worker_counts.values()) > 0 for worker_counts in location_token_counts.values())

        inflight_counts = {}
        if self._inflight_tracker:
            inflight_counts = {w.worker_id: self._inflight_tracker.get(w.worker_id) for w in workers}

        return ScoringContext(
            overlap_scores=overlap_scores,
            total_tokens=total_tokens,
            any_hit=any_hit,
            worker_ids=[w.worker_id for w in workers],
            inflight_counts=inflight_counts,
            location_token_counts=location_token_counts,
            overlap_source=overlap_source,
            overlap_timing=overlap_timing,
        )

    def _pre_score_hook(self, workers, route_hint, context, scorers=None):
        """在评分前调用需要批量预计算的 scorer（如 LatencyPredictScorer）。"""
        for scorer, _ in scorers or self.weighted_scorers:
            if isinstance(scorer, LatencyPredictScorer):
                predictions = scorer.predict_bulk(
                    workers,
                    route_hint,
                    context.overlap_scores,
                )
                scorer.set_predictions(predictions)

    @staticmethod
    def _score_worker_with(
        worker: WorkerInfo,
        all_workers: list[WorkerInfo],
        route_hint: RouteHint,
        context: ScoringContext,
        scorers: list[tuple[Scorer, float]],
    ) -> float:
        total = 0.0
        for scorer, weight in scorers:
            raw = scorer.score(worker, all_workers, route_hint, context)
            clamped = max(0.0, min(1.0, raw))
            total += weight * clamped
        return total

    def _score_worker(
        self,
        worker: WorkerInfo,
        all_workers: list[WorkerInfo],
        route_hint: RouteHint,
        context: ScoringContext,
    ) -> float:
        return self._score_worker_with(
            worker,
            all_workers,
            route_hint,
            context,
            self.weighted_scorers,
        )

    def _select_worker_with_profile(
        self,
        workers: list[WorkerInfo],
        route_hint: RouteHint,
        scorers: list[tuple[Scorer, float]],
        phase: str,
    ) -> WorkerInfo:
        if not workers:
            raise ValueError("No workers available")
        if len(workers) == 1:
            return workers[0]

        t0 = time.perf_counter()
        shuffled = list(workers)
        random.shuffle(shuffled)
        context = self._build_context(shuffled, route_hint, scorers)
        self._pre_score_hook(shuffled, route_hint, context, scorers)

        best_worker = shuffled[0]
        best_score = float("-inf")
        details: list[tuple[str, float, list[tuple[str, float, float]]]] = []
        for w in shuffled:
            breakdown = []
            for scorer, weight in scorers:
                raw = scorer.score(w, shuffled, route_hint, context)
                clamped = max(0.0, min(1.0, raw))
                breakdown.append((scorer.scorer_type, clamped, weight))
            score = sum(value * weight for _, value, weight in breakdown)
            details.append((w.worker_id, score, breakdown))
            logger.debug("[PluginScorer] {}: score={:.3f}", w.worker_id, score)
            if score > best_score:
                best_score = score
                best_worker = w

        if phase == "prefill" and not context.any_hit and context.inflight_counts:
            min_inflight = min(context.inflight_counts.get(worker.worker_id, 0) for worker in shuffled)
            least_inflight_workers = [
                worker for worker in shuffled if context.inflight_counts.get(worker.worker_id, 0) == min_inflight
            ]
            if best_worker not in least_inflight_workers:
                score_by_worker = {worker_id: score for worker_id, score, _ in details}
                best_worker = max(
                    least_inflight_workers,
                    key=lambda worker: score_by_worker[worker.worker_id],
                )
                best_score = score_by_worker[best_worker.worker_id]

        diagnostic_workers = shuffled
        if self.worker_manager is not None:
            if phase == "prefill":
                diagnostic_workers = self.worker_manager.get_all_healthy_prefill_workers(route_hint.model)
            elif phase == "decode":
                diagnostic_workers = self.worker_manager.get_all_healthy_decode_workers(route_hint.model)
        diagnostic_inflight_counts = {}
        if self._inflight_tracker is not None:
            diagnostic_inflight_counts = {
                worker.worker_id: self._inflight_tracker.get(worker.worker_id) for worker in diagnostic_workers
            }
        _log_scoring_details(
            route_hint.request_id,
            phase,
            details,
            shuffled,
            context,
            best_worker.worker_id,
            diagnostic_workers,
            diagnostic_inflight_counts,
        )

        if not context.any_hit:
            for scorer, _ in scorers:
                if isinstance(scorer, NoHitLRUScorer):
                    scorer.update_lru(best_worker.worker_id)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[PluginScorer] Selected {} phase={} score={:.3f} took={:.2f}ms",
            best_worker.worker_id,
            phase,
            best_score,
            elapsed_ms,
        )
        return best_worker

    def select_worker(self, workers: list[WorkerInfo], route_hint: RouteHint) -> WorkerInfo:
        return self._select_worker_with_profile(
            workers,
            route_hint,
            self.weighted_scorers,
            "combined",
        )

    def select_worker_for_phase(
        self,
        workers: list[WorkerInfo],
        route_hint: RouteHint,
        *,
        phase: str,
    ) -> WorkerInfo:
        if phase == "prefill":
            scorers = self.prefill_scorers or self.weighted_scorers
        elif phase == "decode":
            scorers = self.decode_scorers or self.weighted_scorers
        else:
            raise ValueError(f"Unknown scoring phase: {phase}")
        return self._select_worker_with_profile(
            workers,
            route_hint,
            scorers,
            phase,
        )

    def select_worker_pair(
        self,
        prefill_workers: list[WorkerInfo],
        decode_workers: list[WorkerInfo],
        route_hint: RouteHint,
    ) -> tuple[WorkerInfo, WorkerInfo]:
        if not prefill_workers or not decode_workers:
            raise ValueError("No valid prefill or decode workers available")

        t0 = time.perf_counter()

        p_scorers = self.prefill_scorers or self.weighted_scorers
        d_scorers = self.decode_scorers or self.weighted_scorers

        p_shuffled = list(prefill_workers)
        random.shuffle(p_shuffled)
        d_shuffled = list(decode_workers)
        random.shuffle(d_shuffled)

        p_prepare_start = time.perf_counter()
        p_context = self._build_context(p_shuffled, route_hint, p_scorers)
        self._pre_score_hook(p_shuffled, route_hint, p_context, p_scorers)
        p_prepare_ms = (time.perf_counter() - p_prepare_start) * 1000

        d_prepare_start = time.perf_counter()
        d_context = self._build_context(d_shuffled, route_hint, d_scorers)
        self._pre_score_hook(d_shuffled, route_hint, d_context, d_scorers)
        d_prepare_ms = (time.perf_counter() - d_prepare_start) * 1000

        scoring_start = time.perf_counter()
        best_prefill, best_p = None, float("-inf")
        p_details: list[tuple[str, float, list[tuple[str, float, float]]]] = []
        for w in p_shuffled:
            breakdown = []
            for scorer, weight in p_scorers:
                raw = scorer.score(w, p_shuffled, route_hint, p_context)
                clamped = max(0.0, min(1.0, raw))
                breakdown.append((scorer.scorer_type, clamped, weight))
            s = sum(sc * wt for _, sc, wt in breakdown)
            p_details.append((w.worker_id, s, breakdown))
            if s > best_p:
                best_p = s
                best_prefill = w

        # 冷前缀没有缓存收益可权衡时，优先使用Router实时维护的最小
        # inflight集合，再让配置的插件总分决定同负载Worker间的落点。
        # 这避免滞后的queue指标或cold LRU把突发请求堆到更忙的P节点。
        if not p_context.any_hit and p_context.inflight_counts:
            min_inflight = min(p_context.inflight_counts.get(worker.worker_id, 0) for worker in p_shuffled)
            least_inflight_workers = [
                worker for worker in p_shuffled if p_context.inflight_counts.get(worker.worker_id, 0) == min_inflight
            ]
            if best_prefill not in least_inflight_workers:
                score_by_worker = {worker_id: score for worker_id, score, _ in p_details}
                previous_prefill = best_prefill
                best_prefill = max(
                    least_inflight_workers,
                    key=lambda worker: score_by_worker[worker.worker_id],
                )
                best_p = score_by_worker[best_prefill.worker_id]
                logger.info(
                    "[PluginScorer] Cold prefill load guard request={} "
                    "previous={} previous_inflight={} selected={} "
                    "selected_inflight={} min_inflight={}",
                    route_hint.request_id,
                    previous_prefill.worker_id,
                    p_context.inflight_counts.get(previous_prefill.worker_id, 0),
                    best_prefill.worker_id,
                    p_context.inflight_counts.get(best_prefill.worker_id, 0),
                    min_inflight,
                )

        best_decode, best_d = None, float("-inf")
        d_details: list[tuple[str, float, list[tuple[str, float, float]]]] = []
        for w in d_shuffled:
            breakdown = []
            for scorer, weight in d_scorers:
                raw = scorer.score(w, d_shuffled, route_hint, d_context)
                clamped = max(0.0, min(1.0, raw))
                breakdown.append((scorer.scorer_type, clamped, weight))
            s = sum(sc * wt for _, sc, wt in breakdown)
            d_details.append((w.worker_id, s, breakdown))
            if s > best_d:
                best_d = s
                best_decode = w

        if not best_prefill or not best_decode:
            raise ValueError("No valid worker pair found")
        scoring_ms = (time.perf_counter() - scoring_start) * 1000

        detail_log_start = time.perf_counter()
        diagnostic_prefill_workers = p_shuffled
        diagnostic_decode_workers = d_shuffled
        if self.worker_manager is not None:
            diagnostic_prefill_workers = self.worker_manager.get_all_healthy_prefill_workers(route_hint.model)
            diagnostic_decode_workers = self.worker_manager.get_all_healthy_decode_workers(route_hint.model)
        diagnostic_inflight_counts = {}
        if self._inflight_tracker is not None:
            diagnostic_inflight_counts = {}
            for selected_worker in [*diagnostic_prefill_workers, *diagnostic_decode_workers]:
                diagnostic_inflight_counts[selected_worker.worker_id] = self._inflight_tracker.get(
                    selected_worker.worker_id
                )
        _log_scoring_details(
            route_hint.request_id,
            "prefill",
            p_details,
            p_shuffled,
            p_context,
            best_prefill.worker_id,
            diagnostic_prefill_workers,
            diagnostic_inflight_counts,
        )
        _log_scoring_details(
            route_hint.request_id,
            "decode",
            d_details,
            d_shuffled,
            d_context,
            best_decode.worker_id,
            diagnostic_decode_workers,
            diagnostic_inflight_counts,
        )
        detail_log_ms = (time.perf_counter() - detail_log_start) * 1000

        if not p_context.any_hit:
            for scorer, _ in p_scorers:
                if isinstance(scorer, NoHitLRUScorer):
                    scorer.update_lru(best_prefill.worker_id)
        if not d_context.any_hit:
            for scorer, _ in d_scorers:
                if isinstance(scorer, NoHitLRUScorer):
                    scorer.update_lru(best_decode.worker_id)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[PluginScorer] Selected pair {} (p={:.3f}) + {} (d={:.3f}) took={:.2f}ms",
            best_prefill.worker_id,
            best_p,
            best_decode.worker_id,
            best_d,
            elapsed_ms,
        )
        logger.info(
            "[PluginScorerTiming] request={} p_prepare={:.2f}ms p_overlap={} "
            "p_match={:.2f}ms p_lock_wait={:.2f}ms d_prepare={:.2f}ms "
            "d_overlap={} d_match={:.2f}ms d_lock_wait={:.2f}ms "
            "scoring={:.2f}ms detail_log={:.2f}ms total={:.2f}ms",
            route_hint.request_id,
            p_prepare_ms,
            p_context.overlap_source,
            p_context.overlap_timing.get("total_ms", 0.0),
            p_context.overlap_timing.get("lock_wait_ms", 0.0),
            d_prepare_ms,
            d_context.overlap_source,
            d_context.overlap_timing.get("total_ms", 0.0),
            d_context.overlap_timing.get("lock_wait_ms", 0.0),
            scoring_ms,
            detail_log_ms,
            elapsed_ms,
        )
        return best_prefill, best_decode

    def get_latency_predict_scorer(self) -> LatencyPredictScorer | None:
        """返回 LatencyPredictScorer 实例（供训练数据上报使用），若未配置则返回 None。"""
        all_scorers = self.weighted_scorers + self.prefill_scorers + self.decode_scorers
        for scorer, _ in all_scorers:
            if isinstance(scorer, LatencyPredictScorer):
                return scorer
        return None
