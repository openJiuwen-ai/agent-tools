"""Worker 端 Prometheus 指标采集模块，对接 vLLM StatLogger 接口。

提供三个核心组件：
- WorkerBackendMetrics: 定义并管理 worker 级别的 Prometheus Gauge 指标
- WorkerStatLogger: 实现 vLLM StatLoggerBase，在每次调度迭代后更新指标
- StatLoggerFactory: vLLM 要求的工厂类，负责创建 WorkerStatLogger 实例

指标流向::

    vLLM Scheduler → WorkerStatLogger.record() → WorkerBackendMetrics (Prometheus Gauge)
                                                → MetricsPublisher (ZMQ) → router
"""

import logging
import re
from typing import Optional

from prometheus_client import CollectorRegistry, Gauge, REGISTRY, generate_latest
from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

logger = logging.getLogger("worker.metrics")

WORKER_COMPONENT_REGISTRY = CollectorRegistry()

_VLLM_PREFIX_RE = re.compile(r"^(# (HELP|TYPE) )?(vllm:)")

_LABEL_MODEL = "model"
_LABEL_COMPONENT = "worker_component"
_LABEL_DP_RANK = "dp_rank"


class WorkerBackendMetrics:
    """Worker vLLM 后端的 Prometheus 指标集合。

    - 管理 KV cache 总块数、GPU 缓存使用率、模型加载时间等 Gauge 指标
    - 每个指标按 model、component、dp_rank 标签区分
    - 使用独立的 CollectorRegistry 避免与 vLLM 内置指标冲突
    """

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = WORKER_COMPONENT_REGISTRY,
        model_name: str = "",
        component_name: str = "",
    ) -> None:
        self.model_name = model_name
        self.component_name = component_name

        self.total_blocks = Gauge(
            "worker_component_kv_cache_total_blocks",
            "Total number of KV cache blocks available on the worker.",
            labelnames=[_LABEL_MODEL, _LABEL_COMPONENT, _LABEL_DP_RANK],
            registry=registry,
        )
        self.gpu_cache_usage_percent = Gauge(
            "worker_component_gpu_cache_usage_percent",
            "GPU cache usage as a percentage (0.0-1.0).",
            labelnames=[_LABEL_MODEL, _LABEL_COMPONENT, _LABEL_DP_RANK],
            registry=registry,
        )
        self.model_load_time = Gauge(
            "worker_component_model_load_time_seconds",
            "Model load time in seconds.",
            labelnames=[_LABEL_MODEL, _LABEL_COMPONENT],
            registry=registry,
        )

    def _build_labels(self, extra: dict) -> dict:
        base = {_LABEL_MODEL: self.model_name, _LABEL_COMPONENT: self.component_name}
        base[_LABEL_DP_RANK] = extra.get("dp_rank", "0")
        return base

    def set_total_blocks(self, labels: dict, value: int) -> None:
        self.total_blocks.labels(**self._build_labels(labels)).set(value)

    def set_gpu_cache_usage(self, labels: dict, value: float) -> None:
        self.gpu_cache_usage_percent.labels(**self._build_labels(labels)).set(value)

    def set_model_load_time(self, value: float) -> None:
        self.model_load_time.labels(
            **{_LABEL_MODEL: self.model_name, _LABEL_COMPONENT: self.component_name}
        ).set(value)


class WorkerStatLogger(StatLoggerBase):
    """vLLM StatLogger 实现，在每次调度迭代后采集 KV cache 和请求指标。

    - 实现 vLLM 的 StatLoggerBase 接口，通过 record() 接收调度统计
    - 更新 Prometheus Gauge 指标（KV cache 使用率、总块数等）
    - 通过 get_metrics() 向 MetricsPublisher 提供运行时指标快照
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        dp_rank: int = 0,
        component_gauges: Optional[WorkerBackendMetrics] = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.dp_rank = dp_rank
        self.component_gauges = component_gauges or WorkerBackendMetrics()
        self.num_gpu_block = 1
        self.kv_cache_usage = 0.0
        self.num_running_reqs = 0
        self.num_waiting_reqs = 0

    def set_num_gpu_block(self, num_blocks: int) -> None:
        self.num_gpu_block = num_blocks

    def record(
        self,
        scheduler_stats: SchedulerStats | None = None,
        iteration_stats: IterationStats | None = None,
        mm_cache_stats=None,
        engine_idx: int = 0,
    ) -> None:
        if scheduler_stats is None:
            return
        self.kv_cache_usage = scheduler_stats.kv_cache_usage
        self.num_running_reqs = scheduler_stats.num_running_reqs
        self.num_waiting_reqs = scheduler_stats.num_waiting_reqs
        self._flush_gauges()

    def _flush_gauges(self) -> None:
        labels = self._rank_labels()
        self.component_gauges.set_total_blocks(labels, self.num_gpu_block)
        self.component_gauges.set_gpu_cache_usage(labels, self.kv_cache_usage)

    def _rank_labels(self) -> dict:
        return {"dp_rank": str(self.dp_rank)}

    @property
    def active_decode_blocks(self) -> int:
        return int(self.num_gpu_block * self.kv_cache_usage)

    def get_metrics(self) -> dict:
        return {
            "num_gpu_block": self.num_gpu_block,
            "kv_cache_usage": self.kv_cache_usage,
            "active_decode_blocks": self.active_decode_blocks,
            "num_running_reqs": self.num_running_reqs,
            "num_waiting_reqs": self.num_waiting_reqs,
        }

    def init_publish(self) -> None:
        self.kv_cache_usage = 0.0
        self.num_gpu_block = 1
        self._flush_gauges()

    def log_engine_initialized(self) -> None:
        blocks = getattr(self.vllm_config.cache_config, "num_gpu_blocks", None)
        if blocks:
            self.num_gpu_block = blocks
        logger.info(
            "WorkerStatLogger ready, num_gpu_blocks=%d, Prometheus metrics active",
            self.num_gpu_block,
        )


class StatLoggerFactory:
    """WorkerStatLogger 的工厂类，供 vLLM 引擎初始化时调用。

    - vLLM 通过 __call__(vllm_config, dp_rank) 创建 stat logger 实例
    - 保留已创建的 logger 引用，供外部组件查询指标和设置参数
    """

    def __init__(
        self,
        component_gauges: Optional[WorkerBackendMetrics] = None,
    ) -> None:
        self.component_gauges = component_gauges
        self.created_logger: Optional[WorkerStatLogger] = None

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        stat_logger = WorkerStatLogger(
            vllm_config=vllm_config,
            dp_rank=dp_rank,
            component_gauges=self.component_gauges,
        )
        self.created_logger = stat_logger
        return stat_logger

    def get_metrics(self) -> dict:
        if self.created_logger:
            return self.created_logger.get_metrics()
        return {}

    def _forward(self, method: str, *args) -> None:
        if self.created_logger is not None:
            getattr(self.created_logger, method)(*args)

    def set_num_gpu_blocks_all(self, num_blocks: int) -> None:
        self._forward("set_num_gpu_block", num_blocks)

    def init_publish(self) -> None:
        self._forward("init_publish")


def collect_all_metrics() -> bytes:
    """收集 worker 自定义指标与 vLLM 内置指标（vllm: 前缀），合并为 Prometheus 文本格式返回。"""
    worker_output = generate_latest(WORKER_COMPONENT_REGISTRY)

    vllm_raw = generate_latest(REGISTRY).decode("utf-8")
    vllm_lines = [
        line for line in vllm_raw.split("\n")
        if line.strip() and _VLLM_PREFIX_RE.match(line)
    ]
    vllm_output = ("\n".join(vllm_lines) + "\n").encode("utf-8") if vllm_lines else b""

    return worker_output + vllm_output
