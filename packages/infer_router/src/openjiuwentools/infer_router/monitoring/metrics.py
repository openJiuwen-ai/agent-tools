from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from openjiuwentools.infer_router.config.config import settings


class Metrics:
    """监控指标收集类"""

    def __init__(self):
        # 请求相关指标
        self.requests_total = Counter(
            "router_requests_total",
            "Total number of requests received",
            ["endpoint", "method", "status"],
        )

        self.request_duration_seconds = Histogram(
            "router_request_duration_seconds",
            "Request processing duration in seconds",
            ["endpoint", "method"],
        )

        # 路由相关指标
        self.routed_requests_total = Counter(
            "router_routed_requests_total",
            "Total number of routed requests",
            ["worker_id", "model"],
        )

        # 调度相关指标
        self.scheduled_requests_total = Counter(
            "router_scheduled_requests_total",
            "Total number of scheduled requests",
            ["strategy", "worker_id"],
        )

        self.queue_size = Gauge("router_queue_size", "Current request queue size", ["worker_id"])

        # KV缓存相关指标
        self.cache_hits_total = Counter("router_cache_hits_total", "Total number of cache hits")

        self.cache_misses_total = Counter(
            "router_cache_misses_total", "Total number of cache misses"
        )

        self.cache_blocks_total = Gauge("router_cache_blocks_total", "Total number of cache blocks")

        self.cache_aging_blocks_total = Gauge(
            "router_cache_aging_blocks_total", "Total number of aging cache blocks"
        )

        self.cache_fresh_blocks_total = Gauge(
            "router_cache_fresh_blocks_total", "Total number of fresh cache blocks"
        )

        # 工作器相关指标
        self.workers_total = Gauge("router_workers_total", "Total number of workers")

        self.healthy_workers_total = Gauge(
            "router_healthy_workers_total", "Total number of healthy workers"
        )

        self.worker_response_time_seconds = Histogram(
            "router_worker_response_time_seconds", "Worker response time in seconds", ["worker_id"]
        )

        # Agent Hints相关指标
        self.requests_with_agent_hints_total = Counter(
            "router_requests_with_agent_hints_total", "Total number of requests with agent hints"
        )

        self.priority_distribution = Histogram(
            "router_request_priority_distribution", "Distribution of request priorities"
        )

        self.estimated_output_tokens_distribution = Histogram(
            "router_estimated_output_tokens_distribution", "Distribution of estimated output tokens"
        )

        # 事件相关指标
        self.event_counter = Counter(
            "router_kv_event_count",
            "Number of KV cache events processed",
            ["event_type", "worker_id"],
        )
        self.event_latency = Histogram(
            "router_kv_event_processing_time",
            "Time taken to process KV cache events",
            ["event_type"],
        )

        self._metrics_server = None

    @staticmethod
    def start_metrics_server():
        """启动指标服务器"""
        if settings.enable_metrics:
            try:
                logger.info(f"Starting metrics server on port {settings.metrics_port}")
                start_http_server(settings.metrics_port)
            except OSError as e:
                logger.warning(
                    f"Failed to start metrics server on port {settings.metrics_port}: {e}. "
                    "Metrics will not be available."
                )

    def record_request(self, endpoint: str, method: str, status: int):
        """记录请求"""
        self.requests_total.labels(endpoint=endpoint, method=method, status=status).inc()

    def record_request_duration(self, endpoint: str, method: str, duration: float):
        """记录请求持续时间"""
        self.request_duration_seconds.labels(endpoint=endpoint, method=method).observe(duration)

    def record_routed_request(self, worker_id: str, model: str):
        """记录路由请求"""
        self.routed_requests_total.labels(worker_id=worker_id, model=model).inc()

    def record_scheduled_request(self, strategy: str, worker_id: str):
        """记录调度请求"""
        self.scheduled_requests_total.labels(strategy=strategy, worker_id=worker_id).inc()

    def update_queue_size(self, worker_id: str, size: int):
        """更新队列大小"""
        self.queue_size.labels(worker_id=worker_id).set(size)

    def record_cache_hit(self):
        """记录缓存命中"""
        self.cache_hits_total.inc()

    def record_cache_miss(self):
        """记录缓存未命中"""
        self.cache_misses_total.inc()

    def update_cache_stats(self, total_blocks: int, aging_blocks: int, fresh_blocks: int):
        """更新缓存统计信息"""
        self.cache_blocks_total.set(total_blocks)
        self.cache_aging_blocks_total.set(aging_blocks)
        self.cache_fresh_blocks_total.set(fresh_blocks)

    def update_worker_stats(self, total_workers: int, healthy_workers: int):
        """更新工作器统计信息"""
        self.workers_total.set(total_workers)
        self.healthy_workers_total.set(healthy_workers)

    def record_worker_response_time(self, worker_id: str, response_time: float):
        """记录工作器响应时间"""
        self.worker_response_time_seconds.labels(worker_id=worker_id).observe(response_time)

    def record_agent_hints_request(self):
        """记录带有Agent Hints的请求"""
        self.requests_with_agent_hints_total.inc()

    def record_priority(self, priority: int):
        """记录请求优先级"""
        self.priority_distribution.observe(priority)

    def record_estimated_output_tokens(self, tokens: int):
        """记录预期输出token数"""
        self.estimated_output_tokens_distribution.observe(tokens)

    def record_event(self, event_type: str, worker_id: str):
        """记录事件"""
        self.event_counter.labels(event_type=event_type, worker_id=worker_id).inc()

    def record_event_latency(self, event_type: str, latency: float):
        """记录事件处理延迟"""
        self.event_latency.labels(event_type=event_type).observe(latency)


# 创建全局指标实例
metrics = Metrics()
