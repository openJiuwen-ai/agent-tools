"""基于延迟预测的负载均衡算法。

通过外部 latency-predictor 服务（training_server + prediction_server）预测
每个 worker 处理当前请求的 TTFT 和 TPOT，选择预测总延迟最低的 worker。

特征映射（router 指标 → predictor 输入）：
  - worker.current_load / 100.0       → kv_cache_percentage
  - len(token_ids)                    → input_token_length
  - worker.queue_depth                → num_request_waiting
  - worker.running_requests           → num_request_running
  - estimated_output_tokens           → num_tokens_generated
  - overlap / len(token_ids)          → prefix_cache_score
  - worker.worker_type                → pod_type
"""

import asyncio
import time
from typing import Any

import httpx
from loguru import logger

from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.routing.load_balance import LoadBalancingAlgorithm
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint, WorkerInfo, WorkerType


class LatencyPredictAlgorithm(LoadBalancingAlgorithm):
    """基于延迟预测选择 worker 的负载均衡算法。

    对每个候选 worker 调用 prediction_server 获取预测的 TTFT 和 TPOT，
    按 total_latency = ttft + tpot * estimated_output_tokens 选择最优 worker。
    预测失败时回退到最低负载选择。

    请求完成后通过 report_training_data() 异步上报训练样本到 training_server，
    持续优化预测模型。
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        worker_manager=None,
        *,
        prediction_server_url: str = "http://localhost:8001",
        training_server_url: str = "http://localhost:8000",
        predict_timeout: float = 0.5,
        training_report_enabled: bool = True,
    ):
        self.kv_cache_manager = kv_cache_manager
        self.worker_manager = worker_manager
        self.prediction_server_url = prediction_server_url.rstrip("/")
        self.training_server_url = training_server_url.rstrip("/")
        self.predict_timeout = predict_timeout
        self.training_report_enabled = training_report_enabled

        self._http_client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.predict_timeout)
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.predict_timeout)
        return self._http_client

    # ------------------------------------------------------------------
    # 特征构造
    # ------------------------------------------------------------------

    def _build_prediction_request(
        self,
        worker: WorkerInfo,
        route_hint: RouteHint,
        overlap_scores: dict,
    ) -> dict:
        """构造 prediction_server 的请求体。"""
        token_ids = route_hint.token_ids or []
        input_token_length = len(token_ids)
        overlap = float(overlap_scores.get(worker.worker_id, 0.0))
        prefix_cache_score = (overlap / input_token_length) if input_token_length > 0 else 0.0

        pod_type = ""
        if worker.worker_type == WorkerType.PREFILL:
            pod_type = "prefill"
        elif worker.worker_type == WorkerType.DECODE:
            pod_type = "decode"

        return {
            "kv_cache_percentage": min(worker.current_load / 100.0, 1.0),
            "input_token_length": input_token_length,
            "num_request_waiting": worker.queue_depth,
            "num_request_running": worker.running_requests,
            "num_tokens_generated": route_hint.estimated_output_tokens or 128,
            "prefix_cache_score": min(max(prefix_cache_score, 0.0), 1.0),
            "pod_type": pod_type,
        }

    def _compute_overlap_scores(
        self, workers: list[WorkerInfo], route_hint: RouteHint,
    ) -> dict:
        token_ids = route_hint.token_ids or []
        if not token_ids:
            return {}
        return self.kv_cache_manager.find_matches(
            token_ids=token_ids, model=route_hint.model,
        )

    # ------------------------------------------------------------------
    # 批量预测
    # ------------------------------------------------------------------

    def _predict_bulk_sync(self, requests: list[dict]) -> list[dict] | None:
        """同步批量预测（select_worker / select_worker_pair 是同步调用的）。"""
        try:
            client = self._get_sync_client()
            resp = client.post(
                f"{self.prediction_server_url}/predict/bulk/strict",
                json={"requests": requests},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("predictions", [])
            logger.warning(
                "[LatencyPredict] Prediction server returned {}: {}",
                resp.status_code, resp.text[:200],
            )
            return None
        except Exception as e:
            logger.warning("[LatencyPredict] Prediction request failed: {}", e)
            return None

    # ------------------------------------------------------------------
    # 延迟估算
    # ------------------------------------------------------------------

    def _estimate_total_latency(self, prediction: dict, osl: int) -> float:
        """total_latency = ttft + tpot * osl"""
        ttft = prediction.get("ttft_ms", 0.0)
        tpot = prediction.get("tpot_ms", 0.0)
        return ttft + tpot * max(osl, 1)

    def _fallback_select(self, workers: list[WorkerInfo]) -> WorkerInfo:
        """预测失败时回退：选负载最低的 worker。"""
        return min(workers, key=lambda w: (w.queue_depth, w.running_requests, w.current_load))

    # ------------------------------------------------------------------
    # select_worker（combined 模式）
    # ------------------------------------------------------------------

    def select_worker(self, workers: list[WorkerInfo], route_hint: RouteHint) -> WorkerInfo:
        if not workers:
            raise ValueError("No workers available")
        if len(workers) == 1:
            return workers[0]

        t0 = time.perf_counter()

        overlap_scores = self._compute_overlap_scores(workers, route_hint)
        osl = route_hint.estimated_output_tokens or 128

        pred_requests = [
            self._build_prediction_request(w, route_hint, overlap_scores)
            for w in workers
        ]

        predictions = self._predict_bulk_sync(pred_requests)

        if not predictions or len(predictions) != len(workers):
            logger.warning("[LatencyPredict] Fallback to lowest-load selection")
            return self._fallback_select(workers)

        best_idx = 0
        best_latency = float("inf")
        for i, (w, pred) in enumerate(zip(workers, predictions)):
            latency = self._estimate_total_latency(pred, osl)
            logger.debug(
                "[LatencyPredict] {}: ttft={:.1f}ms tpot={:.2f}ms total={:.1f}ms "
                "(kv={:.0f}% wait={} run={} prefix={:.2f})",
                w.worker_id,
                pred.get("ttft_ms", 0), pred.get("tpot_ms", 0), latency,
                w.current_load, w.queue_depth, w.running_requests,
                pred_requests[i]["prefix_cache_score"],
            )
            if latency < best_latency:
                best_latency = latency
                best_idx = i

        chosen = workers[best_idx]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[LatencyPredict] Selected {} latency={:.1f}ms took={:.2f}ms",
            chosen.worker_id, best_latency, elapsed_ms,
        )
        return chosen

    # ------------------------------------------------------------------
    # select_worker_pair（disagg 模式）
    # ------------------------------------------------------------------

    def select_worker_pair(
        self,
        prefill_workers: list[WorkerInfo],
        decode_workers: list[WorkerInfo],
        route_hint: RouteHint,
    ) -> tuple[WorkerInfo, WorkerInfo]:
        if not prefill_workers or not decode_workers:
            raise ValueError("No valid prefill or decode workers available")

        t0 = time.perf_counter()

        all_workers_map = {w.worker_id: w for w in prefill_workers}
        for w in decode_workers:
            all_workers_map.setdefault(w.worker_id, w)
        all_candidates = list(all_workers_map.values())

        overlap_scores = self._compute_overlap_scores(all_candidates, route_hint)
        osl = route_hint.estimated_output_tokens or 128

        pred_requests = [
            self._build_prediction_request(w, route_hint, overlap_scores)
            for w in all_candidates
        ]
        predictions = self._predict_bulk_sync(pred_requests)

        pred_map: dict[str, dict] = {}
        if predictions and len(predictions) == len(all_candidates):
            for w, pred in zip(all_candidates, predictions):
                pred_map[w.worker_id] = pred

        best_prefill, best_ttft = None, float("inf")
        for w in prefill_workers:
            if w.worker_id in pred_map:
                ttft = pred_map[w.worker_id].get("ttft_ms", float("inf"))
            else:
                ttft = float(w.queue_depth * 100 + w.running_requests * 10)
            if ttft < best_ttft:
                best_ttft = ttft
                best_prefill = w

        best_decode, best_decode_lat = None, float("inf")
        for w in decode_workers:
            if w.worker_id in pred_map:
                lat = self._estimate_total_latency(pred_map[w.worker_id], osl)
            else:
                lat = float(w.queue_depth * 100 + w.running_requests * 10)
            if lat < best_decode_lat:
                best_decode_lat = lat
                best_decode = w

        if not best_prefill or not best_decode:
            raise ValueError("No valid worker pair found")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[LatencyPredict] Selected pair {} + {} score={:.1f}ms took={:.2f}ms",
            best_prefill.worker_id, best_decode.worker_id,
            best_ttft + best_decode_lat, elapsed_ms,
        )
        return best_prefill, best_decode

    # ------------------------------------------------------------------
    # 训练数据上报（请求完成后异步调用）
    # ------------------------------------------------------------------

    async def report_training_data(
        self,
        *,
        worker: WorkerInfo,
        route_hint: RouteHint,
        overlap_scores: dict,
        actual_ttft_ms: float,
        actual_tpot_ms: float,
        completion_tokens: int,
    ) -> None:
        """请求完成后上报训练样本到 training_server。"""
        if not self.training_report_enabled:
            return

        token_ids = route_hint.token_ids or []
        input_token_length = len(token_ids)
        overlap = float(overlap_scores.get(worker.worker_id, 0.0))
        prefix_cache_score = (overlap / input_token_length) if input_token_length > 0 else 0.0

        pod_type = ""
        if worker.worker_type == WorkerType.PREFILL:
            pod_type = "prefill"
        elif worker.worker_type == WorkerType.DECODE:
            pod_type = "decode"

        entry = {
            "kv_cache_percentage": min(worker.current_load / 100.0, 1.0),
            "input_token_length": input_token_length,
            "num_request_waiting": worker.queue_depth,
            "num_request_running": worker.running_requests,
            "actual_ttft_ms": max(0.0, actual_ttft_ms),
            "actual_tpot_ms": max(0.0, actual_tpot_ms),
            "num_tokens_generated": max(completion_tokens, 1),
            "prefix_cache_score": min(max(prefix_cache_score, 0.0), 1.0),
            "pod_type": pod_type,
        }

        try:
            client = self._get_async_client()
            resp = await client.post(
                f"{self.training_server_url}/add_training_data_bulk",
                json={"entries": [entry]},
                timeout=2.0,
            )
            if resp.status_code != 202:
                logger.debug(
                    "[LatencyPredict] Training report got {}: {}",
                    resp.status_code, resp.text[:200],
                )
        except Exception as e:
            logger.debug("[LatencyPredict] Training report failed (non-critical): {}", e)

    def report_training_data_fire_and_forget(self, **kwargs) -> None:
        """非阻塞上报：创建后台任务发送训练数据。"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.report_training_data(**kwargs))
        except RuntimeError:
            pass
