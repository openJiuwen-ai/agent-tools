"""Worker YAML 配置加载，将 YAML 配置映射为与 argparse 兼容的 Namespace。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


_DEFAULTS = {
    "model": None,
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": None,
    "max_num_seqs": 256,
    "dtype": "auto",
    "quantization": None,
    "trust_remote_code": False,
    "enable_prefix_caching": False,
    "prefix_caching_hash_algo": "sha256",
    "prefix_caching_hash_seed": 0,
    "disable_log_requests": False,
    "kv_events_config": None,
    "kv_transfer_config": None,
    "host": "127.0.0.1",
    "api_key": None,
    "max_request_body_bytes": 64 * 1024 * 1024,
    "port": 8001,
    "served_model_name": None,
    "uvicorn_log_level": "info",
    "request_plane": "http",
    "etcd_endpoints": None,
    "service_name": None,
    "registry_ttl": 10,
    "kv_relay_endpoint": None,
    "worker_mode": "aggregated",
    "metrics_interval": 1.0,
    "log_level": "INFO",
    "tool_call_parser": None,
    "reasoning_parser": None,
    "enable_auto_tools": None,
    "auto_detect_parsers": True,
    "kvbm_worker_enabled": False,
    "kvbm_worker_disk_root": "/tmp/jiuwen-kvbm-worker",
    "kvbm_clean_disk_on_start": True,
    "kvbm_npu_high_watermark_blocks": 0,
    "kvbm_ram_high_watermark_bytes": 0,
    "kvbm_worker_interval": 5.0,
    "kvbm_block_size": 16,
    "kvbm_worker_provider": "auto",
    "kvbm_worker_diagnostics": False,
    "kvbm_worker_locator_depth": 16,
    "kvbm_auto_register_physical_blocks": False,
    "kvbm_auto_enforce_watermarks_on_init": False,
    "kvbm_prefetch_host": "127.0.0.1",
    "kvbm_prefetch_port": None,
    "kvbm_control_url": "",
    "kvbm_reservation_port": 0,
    "kvbm_transfer_max_inflight": 1,
    "kvbm_transfer_retention_seconds": 300.0,
    "kvbm_benchmark_enabled": False,
    "kvbm_benchmark_tcp_host": "127.0.0.1",
    "kvbm_benchmark_tcp_port": 0,
    "kvbm_benchmark_max_blocks": 64,
    "kvbm_data_backend": "local",
    "kvbm_llm_datadist_mode": "python",
    "kvbm_llm_datadist_role": "mix",
    "kvbm_llm_datadist_cluster_id": 0,
    "kvbm_llm_datadist_listen_ip": "127.0.0.1",
    "kvbm_llm_datadist_listen_port": 0,
    "kvbm_llm_datadist_listen_port_base": 19000,
    "kvbm_llm_datadist_transfer_backend": "hixl",
    "kvbm_llm_datadist_enable_cache_manager": True,
    "kvbm_llm_datadist_enable_remote_cache_accessible": False,
    "kvbm_pd_block_ownership": "scheduler",
}

_JSON_FIELDS = {"kv_events_config", "kv_transfer_config"}
_KEY_ALIASES = {
    # 与 vLLM 官方命名保持兼容，内部继续使用既有字段名。
    "enable_auto_tool_choice": "enable_auto_tools",
}


def load_worker_config(config_path: str) -> argparse.Namespace:
    """从 YAML 文件加载 worker 配置，返回与 parse_args() 兼容的 Namespace。

    YAML 示例::

        model: /root/models/Qwen3-0.6B
        port: 8010
        trust_remote_code: true
        enable_prefix_caching: true
        prefix_caching_hash_algo: sha256
        prefix_caching_hash_seed: 0
        gpu_memory_utilization: 0.3
        worker_mode: prefill
        kv_relay_endpoint: "tcp://*:9010"
        kv_events_config:
          enable_kv_cache_events: true
          publisher: zmq
          endpoint: "tcp://*:5557"
        kv_transfer_config:
          kv_connector: MooncakeConnector
          kv_role: kv_producer
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Worker config not found: {config_path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    merged = dict(_DEFAULTS)
    for key, value in raw.items():
        norm_key = key.replace("-", "_")
        norm_key = _KEY_ALIASES.get(norm_key, norm_key)
        if norm_key in _JSON_FIELDS and isinstance(value, dict):
            merged[norm_key] = json.dumps(value)
        else:
            merged[norm_key] = value

    if merged["model"] is None:
        raise ValueError("'model' is required in worker config")

    return argparse.Namespace(**merged)
