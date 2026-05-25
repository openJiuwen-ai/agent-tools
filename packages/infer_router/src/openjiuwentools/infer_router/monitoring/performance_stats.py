import asyncio
import time
from collections import deque
from dataclasses import dataclass

from loguru import logger


@dataclass
class RequestTimings:
    """请求时间度量参数"""

    request_id: str
    start_time: float
    end_time: float
    router_dispatch_duration: float
    prefill_duration: float
    decode_duration: float
    response_return_duration: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    osl: int = 0


@dataclass
class RequestStats:
    """请求统计数据"""

    request_id: str
    start_time: float
    end_time: float
    e2e_duration: float
    router_dispatch_duration: float
    prefill_duration: float
    decode_duration: float
    response_return_duration: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    osl: int = 0


class PerformanceStats:
    """性能统计类"""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.requests: deque[RequestStats] = deque()
        self._cleanup_task: asyncio.Task | None = None

    async def start(self):
        """启动统计服务"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        asyncio.create_task(self._print_stats_loop())
        logger.info(f"Performance stats started with {self.window_seconds}s window")

    async def stop(self):
        """停止统计服务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Performance stats stopped")

    async def _cleanup_loop(self):
        """清理过期数据的循环"""
        while True:
            try:
                await asyncio.sleep(1)
                self._cleanup_old_requests()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _print_stats_loop(self):
        """打印统计数据的循环"""
        while True:
            try:
                await asyncio.sleep(self.window_seconds)
                self._print_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in print stats loop: {e}")

    def _cleanup_old_requests(self):
        """清理过期的请求数据"""
        now = time.time()
        cutoff = now - self.window_seconds
        while self.requests and self.requests[0].end_time < cutoff:
            self.requests.popleft()

    def record_request(self, timings: RequestTimings):
        """记录请求统计数据

        Args:
            timings: 请求时间度量参数

        """
        e2e_duration = timings.end_time - timings.start_time

        stats = RequestStats(
            request_id=timings.request_id,
            start_time=timings.start_time,
            end_time=timings.end_time,
            e2e_duration=e2e_duration,
            router_dispatch_duration=timings.router_dispatch_duration,
            prefill_duration=timings.prefill_duration,
            decode_duration=timings.decode_duration,
            response_return_duration=timings.response_return_duration,
            prompt_tokens=timings.prompt_tokens,
            completion_tokens=timings.completion_tokens,
            total_tokens=timings.total_tokens,
            osl=timings.osl,
        )

        self.requests.append(stats)
        self._cleanup_old_requests()

    @staticmethod
    def _calculate_percentiles(values: list[float]) -> dict:
        """计算分位数"""
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        percentiles = {}
        for p in [50, 70, 90, 95, 99, 100]:
            idx = int((p / 100) * (n - 1))
            if idx >= n:
                idx = n - 1
            percentiles[p] = sorted_values[idx]

        return percentiles

    def _print_stats(self):
        """打印统计数据"""
        self._cleanup_old_requests()

        if not self.requests:
            logger.info("[PERFORMANCE STATS] No requests in the last minute")
            return

        e2e_durations = [req.e2e_duration for req in self.requests]
        router_dispatch_durations = [req.router_dispatch_duration for req in self.requests]
        prefill_durations = [req.prefill_duration for req in self.requests]
        decode_durations = [req.decode_duration for req in self.requests]
        response_return_durations = [req.response_return_duration for req in self.requests]

        e2e_percentiles = self._calculate_percentiles(e2e_durations)
        router_dispatch_percentiles = self._calculate_percentiles(router_dispatch_durations)
        prefill_percentiles = self._calculate_percentiles(prefill_durations)
        decode_percentiles = self._calculate_percentiles(decode_durations)
        response_return_percentiles = self._calculate_percentiles(response_return_durations)

        total_requests = len(self.requests)

        logger.info(f"[PERFORMANCE STATS] Last {self.window_seconds}s: Requests={total_requests}")

        logger.info(
            f"[PERFORMANCE STATS] E2E Duration (s): "
            f"50%={e2e_percentiles.get(50, 0):.3f}, "
            f"70%={e2e_percentiles.get(70, 0):.3f}, "
            f"90%={e2e_percentiles.get(90, 0):.3f}, "
            f"95%={e2e_percentiles.get(95, 0):.3f}, "
            f"99%={e2e_percentiles.get(99, 0):.3f}, "
            f"100%={e2e_percentiles.get(100, 0):.3f}"
        )

        logger.info(
            f"[PERFORMANCE STATS] Router Dispatch Duration (s): "
            f"50%={router_dispatch_percentiles.get(50, 0):.3f}, "
            f"70%={router_dispatch_percentiles.get(70, 0):.3f}, "
            f"90%={router_dispatch_percentiles.get(90, 0):.3f}, "
            f"95%={router_dispatch_percentiles.get(95, 0):.3f}, "
            f"99%={router_dispatch_percentiles.get(99, 0):.3f}, "
            f"100%={router_dispatch_percentiles.get(100, 0):.3f}"
        )

        logger.info(
            f"[PERFORMANCE STATS] Prefill Duration (s): "
            f"50%={prefill_percentiles.get(50, 0):.3f}, "
            f"70%={prefill_percentiles.get(70, 0):.3f}, "
            f"90%={prefill_percentiles.get(90, 0):.3f}, "
            f"95%={prefill_percentiles.get(95, 0):.3f}, "
            f"99%={prefill_percentiles.get(99, 0):.3f}, "
            f"100%={prefill_percentiles.get(100, 0):.3f}"
        )

        logger.info(
            f"[PERFORMANCE STATS] Decode Duration (s): "
            f"50%={decode_percentiles.get(50, 0):.3f}, "
            f"70%={decode_percentiles.get(70, 0):.3f}, "
            f"90%={decode_percentiles.get(90, 0):.3f}, "
            f"95%={decode_percentiles.get(95, 0):.3f}, "
            f"99%={decode_percentiles.get(99, 0):.3f}, "
            f"100%={decode_percentiles.get(100, 0):.3f}"
        )

        logger.info(
            f"[PERFORMANCE STATS] Response Return Duration (s): "
            f"50%={response_return_percentiles.get(50, 0):.3f}, "
            f"70%={response_return_percentiles.get(70, 0):.3f}, "
            f"90%={response_return_percentiles.get(90, 0):.3f}, "
            f"95%={response_return_percentiles.get(95, 0):.3f}, "
            f"99%={response_return_percentiles.get(99, 0):.3f}, "
            f"100%={response_return_percentiles.get(100, 0):.3f}"
        )

        prompt_tokens_list = [req.prompt_tokens for req in self.requests if req.prompt_tokens > 0]
        completion_tokens_list = [
            req.completion_tokens for req in self.requests if req.completion_tokens > 0
        ]
        total_tokens_list = [req.total_tokens for req in self.requests if req.total_tokens > 0]

        if prompt_tokens_list:
            prompt_avg = sum(prompt_tokens_list) / len(prompt_tokens_list)
            prompt_sum = sum(prompt_tokens_list)
            prompt_min = min(prompt_tokens_list)
            prompt_max = max(prompt_tokens_list)
            prompt_percentiles = self._calculate_percentiles([float(x) for x in prompt_tokens_list])
            logger.info(
                f"[TOKEN STATS] Prompt Tokens: count={len(prompt_tokens_list)}, "
                f"sum={prompt_sum}, avg={prompt_avg:.1f}, min={prompt_min}, max={prompt_max}, "
                f"50%={prompt_percentiles.get(50, 0):.0f}, 90%={prompt_percentiles.get(90, 0):.0f}, "
                f"99%={prompt_percentiles.get(99, 0):.0f}"
            )

        if completion_tokens_list:
            completion_avg = sum(completion_tokens_list) / len(completion_tokens_list)
            completion_sum = sum(completion_tokens_list)
            completion_min = min(completion_tokens_list)
            completion_max = max(completion_tokens_list)
            completion_percentiles = self._calculate_percentiles(
                [float(x) for x in completion_tokens_list]
            )
            logger.info(
                f"[TOKEN STATS] Completion Tokens: count={len(completion_tokens_list)}, "
                f"sum={completion_sum}, avg={completion_avg:.1f}, min={completion_min}, max={completion_max}, "
                f"50%={completion_percentiles.get(50, 0):.0f}, 90%={completion_percentiles.get(90, 0):.0f}, "
                f"99%={completion_percentiles.get(99, 0):.0f}"
            )

        if total_tokens_list:
            total_avg = sum(total_tokens_list) / len(total_tokens_list)
            total_sum = sum(total_tokens_list)
            total_min = min(total_tokens_list)
            total_max = max(total_tokens_list)
            total_percentiles = self._calculate_percentiles([float(x) for x in total_tokens_list])
            logger.info(
                f"[TOKEN STATS] Total Tokens: count={len(total_tokens_list)}, "
                f"sum={total_sum}, avg={total_avg:.1f}, min={total_min}, max={total_max}, "
                f"50%={total_percentiles.get(50, 0):.0f}, 90%={total_percentiles.get(90, 0):.0f}, "
                f"99%={total_percentiles.get(99, 0):.0f}"
            )

        osl_diff_list = []
        osl_diff_ratio_list = []
        for req in self.requests:
            if req.osl > 0 and req.completion_tokens > 0:
                abs_diff = req.osl - req.completion_tokens
                ratio = abs_diff / req.completion_tokens
                osl_diff_list.append(float(abs_diff))
                osl_diff_ratio_list.append(ratio)

        if osl_diff_list:
            diff_avg = sum(osl_diff_list) / len(osl_diff_list)
            diff_min = min(osl_diff_list)
            diff_max = max(osl_diff_list)
            diff_percentiles = self._calculate_percentiles(osl_diff_list)
            logger.info(
                f"[OSL DIFF STATS] OSL vs CompletionTokens Abs Diff: "
                f"count={len(osl_diff_list)}, avg={diff_avg:.1f}, min={diff_min:.1f}, max={diff_max:.1f}, "
                f"50%={diff_percentiles.get(50, 0):.1f}, 80%={diff_percentiles.get(80, 0):.1f}, "
                f"90%={diff_percentiles.get(90, 0):.1f}, 95%={diff_percentiles.get(95, 0):.1f}, "
                f"99%={diff_percentiles.get(99, 0):.1f}"
            )

        if osl_diff_ratio_list:
            ratio_avg = sum(osl_diff_ratio_list) / len(osl_diff_ratio_list)
            ratio_min = min(osl_diff_ratio_list)
            ratio_max = max(osl_diff_ratio_list)
            ratio_percentiles = self._calculate_percentiles(osl_diff_ratio_list)
            logger.info(
                f"[OSL DIFF STATS] OSL vs CompletionTokens Diff Ratio: "
                f"count={len(osl_diff_ratio_list)}, avg={ratio_avg:.3f}, min={ratio_min:.3f}, max={ratio_max:.3f}, "
                f"50%={ratio_percentiles.get(50, 0):.3f}, 80%={ratio_percentiles.get(80, 0):.3f}, "
                f"90%={ratio_percentiles.get(90, 0):.3f}, 95%={ratio_percentiles.get(95, 0):.3f}, "
                f"99%={ratio_percentiles.get(99, 0):.3f}"
            )


performance_stats = PerformanceStats()
