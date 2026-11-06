import os
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings


def load_yaml_config(config_path: str | None = None) -> dict[str, Any]:
    """加载 YAML 配置文件

    Args:
        config_path: 配置文件路径，如果为 None 则尝试从环境变量 CONFIG_PATH 获取

    Returns:
        配置字典

    """
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config.yaml")

    config_file = Path(config_path)
    if not config_file.exists():
        return {}

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def flatten_config(config: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """将嵌套的配置字典扁平化

    Args:
        config: 嵌套的配置字典
        parent_key: 父级键名

    Returns:
        扁平化的配置字典

    """
    items: list[tuple[str, Any]] = []
    for key, value in config.items():
        new_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


class Settings(BaseSettings):
    """系统配置类"""

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # 工作器配置
    worker_discovery_interval: int = 30
    worker_health_check_interval: int = 30  # 增加间隔，减少健康检查频率
    worker_health_check_timeout: int = 10  # 增加超时时间
    worker_health_check_max_failures: int = 3  # 允许连续失败次数
    request_forward_timeout: int = 120

    # 工作器发现配置
    worker_discovery_type: str = "config"
    worker_config_path: str = "workers.json"

    # etcd配置
    etcd_hosts: list[str] = ["localhost"]
    etcd_port: int = 2379
    etcd_prefix: str = "/jiuwen/workers"
    etcd_user: str | None = None
    etcd_password: str | None = None
    etcd_enable_watch: bool = False

    # 调度配置
    default_scheduling_strategy: str = "FCFS"

    # KV缓存配置
    kv_cache_max_blocks: int = 1000
    kv_cache_aging_block_factor: float = 0.3
    kv_cache_decay_factor: float = 0.9
    kv_cache_block_size: int = 16  # 旧事件缺少 block_size 时的兼容兜底值
    kv_cache_enable_session_affinity: bool = True
    kv_cache_enable_radix_tree: bool = False  # 是否启用 Radix Tree 前缀匹配
    router_prefix_match_decode_workers: bool = True  # Prefix 匹配是否包含 D Worker
    kvbm_enabled: bool = True  # 是否启用独立 jiuwen-kvbm 管理面适配
    kvbm_disk_root: str = "/tmp/jiuwen-kvbm-router"  # KVBM 本地磁盘后端目录
    kvbm_ram_max_bytes: int | None = None  # KVBM RAM 后端容量上限
    kvbm_disk_max_bytes: int | None = None  # KVBM 磁盘后端容量上限
    kvbm_router_prefetch_enabled: bool = True  # 是否在路由转发前触发 worker 侧 KVBM 预取
    kvbm_router_prefetch_timeout_ms: int = 500  # 从 Router 发起时起算的请求级预取 deadline
    kvbm_router_prefetch_max_blocks: int = 8  # 单次请求最多预取的 block 数
    kvbm_router_prefetch_wait_for_result: bool = False  # 是否同步等待 worker 预取结果
    kvbm_router_transfer_enabled: bool = False  # 是否自动触发跨 Worker LLM-DataDist/HIXL 传输
    kvbm_router_transfer_max_blocks: int = 8  # 单次跨 Worker 传输的 block 上限
    kvbm_router_transfer_timeout_ms: int = 3000  # pull-and-publish 请求预算
    kvbm_router_link_timeout_ms: int = 3000  # 首次建立 LLM-DataDist 链路的超时
    kvbm_router_datadist_status_ttl_seconds: float = 10.0  # endpoint 状态缓存时间
    kvbm_router_source_lease_ttl_ms: int = 30000  # 源 block pin lease 生存时间
    kvbm_router_reservation_ttl_ms: int = 120000  # 目标 BlockPool 预留生存时间
    kvbm_router_pd_enabled: bool = False  # 是否由 Rust 数据面自动编排 P/D KVBM handoff
    kvbm_pd_block_ownership: str = "scheduler"  # D目标block所有权；默认与Worker Scheduler模式一致
    kvbm_router_pd_timeout_ms: int = 10000  # P/D export、gate、pull 和 attach 的单阶段超时
    kvbm_router_pd_export_timeout_ms: int = 120000  # P排队并完成KV捕获的请求级等待上限
    router_pd_prefill_admission_mode: str = "auto"  # auto按token/block预算准入，fixed仅使用并发上限
    router_pd_prefill_max_inflight: int | None = 0  # 每个P Worker硬上限；0表示自动模式不限制
    kvbm_router_pd_prefill_max_inflight: int = 8  # 兼容旧配置；通用配置未设置时使用
    router_pd_prefill_block_admission_enabled: bool = True  # 是否启用P Worker KV block预算准入
    router_pd_prefill_block_headroom_ratio: float = 0.10  # P Worker预留的KV block安全余量比例
    router_pd_decode_admission_mode: str = "auto"  # auto按实时block预算准入，fixed仅使用并发上限
    router_pd_decode_max_inflight: int = 0  # 每个D Worker硬上限；0表示自动模式不限制
    router_pd_decode_block_admission_enabled: bool = True  # 是否启用D Worker KV block预算准入
    router_pd_decode_block_headroom_ratio: float = 0.10  # D Worker预留的KV block安全余量比例
    router_pd_decode_initial_window_blocks: int = 2  # attach后视为已开始落地的Decode block数
    router_pd_block_pool_metrics_ttl_seconds: float = 2.0  # engine-core真实空闲block指标有效期
    router_pd_decode_capacity_confirmation_enabled: bool = True  # 派发前向D engine-core确认最新容量
    router_pd_decode_capacity_confirmation_timeout_ms: int = 50  # D容量确认HTTP超时
    router_pd_decode_stale_capacity_confirmation_timeout_ms: int = 150  # 容量快照过期后的复核超时
    router_pd_decode_hotspot_spill_enabled: bool = False  # HIXL D热点有限分流；默认保持现有亲和调度
    router_pd_decode_hotspot_spill_hot_min_running: int | None = None  # 热点D最少Running数
    router_pd_decode_hotspot_spill_cold_max_running: int | None = None  # 冷D最多Running数
    router_pd_decode_hotspot_spill_running_gap: int | None = None  # 仅未配具体Running数时生效
    router_pd_decode_hotspot_spill_tpot_ratio: float = 1.25  # 热点D TPOT至少为冷D的倍数
    kvbm_router_pd_decode_direct_enabled: bool = False  # 默认严格PD；显式开启后允许D补算Prompt尾部
    kvbm_router_pd_decode_direct_min_npu_tokens: int = 1024  # 允许 D 直达所需的最少连续 NPU token
    kvbm_router_pd_decode_direct_min_ratio: float = 0.8  # 连续 NPU 前缀占 prompt 的最小比例

    # KV事件配置
    kv_event_mode: str = "inner_event"  # "inner_event" 或 "worker_event"
    worker_token_capacity: int = 1000000  # 每个工作器的最大 token 容量
    worker_timeout: int = 300  # 工作器失联超时时间（秒）
    kv_event_buffer_size: int = 1000  # 事件缓冲区大小

    # 认证配置
    api_key: str | None = None
    enable_auth: bool = False

    # 监控配置
    enable_metrics: bool = True
    metrics_port: int = 8001

    # 负载均衡配置
    load_balancing_algorithm: str = "weighted"
    prefill_load_balancing_algorithm: str | None = None
    decode_load_balancing_algorithm: str | None = None

    # 插件式评分配置
    scoring_plugins: list[dict[str, Any]] = []
    prefill_scoring_plugins: list[dict[str, Any]] = []
    decode_scoring_plugins: list[dict[str, Any]] = []

    # 延迟预测服务配置（latency_predict 算法或 latency-predict-scorer 插件使用）
    latency_prediction_server_url: str = "http://localhost:8001"
    latency_training_server_url: str = "http://localhost:8000"
    latency_predict_timeout: float = 0.5
    latency_training_report_enabled: bool = True

    # 容错配置
    retry_attempts: int = 3
    retry_delay: float = 0.5

    # Admission control 配置
    admission_mode: str = "static"  # "static": 固定并发上限; "dynamic": 基于 worker 上报的 waiting/running 比值
    max_concurrency_per_worker: int = 100
    concurrency_headroom: float = 0.0
    admission_saturation_ratio: float = 0.2  # dynamic 模式下 waiting/running >= 此值则限流
    router_queue_timeout: float = 30.0
    router_queue_poll_interval: float = 0.1

    # 性能优化配置
    uvicorn_workers: int = 1  # uvicorn worker 进程数
    http_pool_connections: int = 500  # 增加连接池大小
    http_pool_max_keepalive: int = 100  # 增加keepalive连接数
    http_keepalive_expiry: float = 30.0  # 增加keepalive过期时间
    request_rate_limit: int = 10000
    request_burst_limit: int = 1000
    enable_response_cache: bool = True
    response_cache_ttl: int = 300
    response_cache_max_size: int = 1000
    max_concurrent_requests: int = 1000  # 增加最大并发请求数

    # 请求默认值
    # 默认不改写客户端请求，由后端按模型上下文动态决定生成上限。
    # 仅在部署方显式配置整数时，才为缺失/null 的请求补充 max_tokens。
    default_max_tokens: int | None = None
    default_ignore_eos: bool = False

    # Tokenizer配置
    tokenizer_load_from_file: bool = False
    tokenizer_local_dir: str | None = None  # 本地tokenizer目录
    tokenizer_prefix_cache_enabled: bool = True
    tokenizer_prefix_cache_max_entries: int = 2048
    tokenizer_prefix_cache_max_tokens: int = 2_000_000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @classmethod
    def from_yaml(cls, config_path: str | None = None) -> "Settings":
        """从 YAML 配置文件创建配置实例

        Args:
            config_path: 配置文件路径

        Returns:
            配置实例

        """
        yaml_config = load_yaml_config(config_path)
        return _with_yaml_defaults(cls, cls(), yaml_config)


def _with_yaml_defaults(
    settings_cls: type[Settings],
    settings_instance: Settings,
    yaml_config: dict[str, Any],
) -> Settings:
    """按文档声明的优先级合并 YAML：环境变量（及 .env）高于 YAML，高于默认值。

    ``Settings(**yaml_flat)`` 会让 init 参数压过环境变量（pydantic-settings 中
    init > env），与 docs/deployment.md「配置优先级」相反。这里先用
    ``Settings()`` 让环境变量生效，再只把环境变量**没有**设置的字段用 YAML
    覆盖，从而保持 环境变量 > YAML > 默认值。
    """
    if not yaml_config:
        return settings_instance

    flat_config = flatten_config(yaml_config)
    known_fields = set(settings_cls.model_fields)
    overrides = {
        key: value
        for key, value in flat_config.items()
        if key in known_fields and key not in settings_instance.model_fields_set
    }
    if not overrides:
        return settings_instance

    merged = settings_instance.model_dump()
    merged.update(overrides)
    return settings_cls(**merged)


_global_settings: Settings | None = None


class SettingsProxy:
    """设置代理类，用于动态访问全局设置"""

    @classmethod
    def _get_settings(cls) -> "Settings":
        global _global_settings
        if _global_settings is None:
            _global_settings = create_settings()
        return _global_settings

    def __getattr__(self, name):
        return getattr(self._get_settings(), name)

    def __setattr__(self, name, value):
        if name == "_internal":
            super().__setattr__(name, value)
            return
        setattr(self._get_settings(), name, value)


def create_settings(config_path: str | None = None) -> Settings:
    """创建配置实例

    优先级：环境变量 > YAML 配置文件 > 默认值

    Args:
        config_path: 配置文件路径

    Returns:
        配置实例

    """
    yaml_config = load_yaml_config(config_path)
    return _with_yaml_defaults(Settings, Settings(), yaml_config)


def set_global_settings(new_settings: Settings) -> None:
    """设置全局配置实例

    Args:
        new_settings: 新的配置实例

    """
    global _global_settings
    _global_settings = new_settings


def get_global_settings() -> Settings:
    """获取全局配置实例

    Returns:
        配置实例

    """
    global _global_settings
    if _global_settings is None:
        _global_settings = create_settings()
    return _global_settings


settings = SettingsProxy()
