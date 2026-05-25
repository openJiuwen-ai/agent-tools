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
    kv_cache_block_size: int = 16
    kv_cache_enable_session_affinity: bool = True
    kv_cache_enable_radix_tree: bool = False  # 是否启用 Radix Tree 前缀匹配

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

    # 容错配置
    retry_attempts: int = 3
    retry_delay: float = 0.5

    # 性能优化配置
    http_pool_connections: int = 500  # 增加连接池大小
    http_pool_max_keepalive: int = 100  # 增加keepalive连接数
    http_keepalive_expiry: float = 30.0  # 增加keepalive过期时间
    request_rate_limit: int = 10000
    request_burst_limit: int = 1000
    enable_response_cache: bool = True
    response_cache_ttl: int = 300
    response_cache_max_size: int = 1000
    max_concurrent_requests: int = 1000  # 增加最大并发请求数

    # Tokenizer配置
    tokenizer_load_from_file: bool = False
    tokenizer_local_dir: str | None = None  # 本地tokenizer目录

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
        if yaml_config:
            flat_config = flatten_config(yaml_config)
            return cls(**flat_config)
        return cls()


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
    if yaml_config:
        flat_config = flatten_config(yaml_config)
        return Settings(**flat_config)
    return Settings()


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
