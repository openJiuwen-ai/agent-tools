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
    "disable_log_requests": False,
    "kv_events_config": None,
    "kv_transfer_config": None,
    "host": "0.0.0.0",
    "port": 8001,
    "served_model_name": None,
    "uvicorn_log_level": "info",
    "request_plane": "http",
    "etcd_endpoints": None,
    "service_name": None,
    "registry_ttl": 10,
    "kv_relay_endpoint": None,
    "worker_mode": "aggregated",
    "metrics_interval": 5.0,
}

_JSON_FIELDS = {"kv_events_config", "kv_transfer_config"}


def load_worker_config(config_path: str) -> argparse.Namespace:
    """从 YAML 文件加载 worker 配置，返回与 parse_args() 兼容的 Namespace。

    YAML 示例::

        model: /root/models/Qwen3-0.6B
        port: 8010
        trust_remote_code: true
        enable_prefix_caching: true
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
        if norm_key in _JSON_FIELDS and isinstance(value, dict):
            merged[norm_key] = json.dumps(value)
        else:
            merged[norm_key] = value

    if merged["model"] is None:
        raise ValueError("'model' is required in worker config")

    return argparse.Namespace(**merged)
