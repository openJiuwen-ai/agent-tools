"""Worker YAML 配置加载测试。"""

import argparse
import os
import sys
import types
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.modules.setdefault("uvloop", types.ModuleType("uvloop"))

from openjiuwentools.infer_router.vllm.main import (  # noqa: E402
    _configure_prefix_caching_hash_seed,
    _router_worker_type,
    _start_worker_publisher,
)
from openjiuwentools.infer_router.vllm.worker_config import load_worker_config


def test_load_worker_config_preserves_llm_datadist_options(tmp_path) -> None:
    """LLM-DataDist 配置应以 argparse 属性形式完整传递给 worker 入口。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        无返回值；配置值不符合预期时由断言报告失败。
    """

    config_path = tmp_path / "worker.yaml"
    config_path.write_text(
        "\n".join(
            (
                "model: /models/test",
                "enable_prefix_caching: true",
                "prefix_caching_hash_algo: sha256",
                "prefix_caching_hash_seed: 17",
                "kvbm_data_backend: llm_datadist",
                "kvbm_control_url: http://10.0.0.7:18007",
                "kvbm_reservation_port: 29107",
                "kvbm_transfer_max_inflight: 2",
                "kvbm_transfer_retention_seconds: 900",
                "kvbm_benchmark_enabled: true",
                "kvbm_benchmark_tcp_host: 10.0.0.7",
                "kvbm_benchmark_tcp_port: 29007",
                "kvbm_benchmark_max_blocks: 8",
                "kvbm_llm_datadist_mode: python",
                "kvbm_llm_datadist_role: mix",
                "kvbm_llm_datadist_cluster_id: 7",
                "kvbm_llm_datadist_listen_ip: 10.0.0.7",
                "kvbm_llm_datadist_listen_port: 19107",
                "kvbm_llm_datadist_listen_port_base: 19100",
                "kvbm_llm_datadist_transfer_backend: hixl",
                "kvbm_llm_datadist_enable_cache_manager: true",
                "kvbm_llm_datadist_enable_remote_cache_accessible: true",
                "kvbm_pd_block_ownership: legacy",
            )
        ),
        encoding="utf-8",
    )

    config = load_worker_config(str(config_path))

    assert config.enable_prefix_caching is True
    assert config.prefix_caching_hash_algo == "sha256"
    assert config.prefix_caching_hash_seed == 17
    assert config.kvbm_data_backend == "llm_datadist"
    assert config.kvbm_control_url == "http://10.0.0.7:18007"
    assert config.kvbm_reservation_port == 29107
    assert config.kvbm_transfer_max_inflight == 2
    assert config.kvbm_transfer_retention_seconds == 900
    assert config.kvbm_benchmark_enabled is True
    assert config.kvbm_benchmark_tcp_host == "10.0.0.7"
    assert config.kvbm_benchmark_tcp_port == 29007
    assert config.kvbm_benchmark_max_blocks == 8
    assert config.kvbm_llm_datadist_mode == "python"
    assert config.kvbm_llm_datadist_role == "mix"
    assert config.kvbm_llm_datadist_cluster_id == 7
    assert config.kvbm_llm_datadist_listen_ip == "10.0.0.7"
    assert config.kvbm_llm_datadist_listen_port == 19107
    assert config.kvbm_llm_datadist_listen_port_base == 19100
    assert config.kvbm_llm_datadist_transfer_backend == "hixl"
    assert config.kvbm_llm_datadist_enable_cache_manager is True
    assert config.kvbm_llm_datadist_enable_remote_cache_accessible is True
    assert config.kvbm_pd_block_ownership == "legacy"


def test_load_worker_config_uses_stable_kvbm_defaults(tmp_path) -> None:
    """未显式配置时应使用稳定hash和Scheduler原生block所有权。

    参数:
        tmp_path: pytest 提供的临时目录。

    返回:
        无返回值；默认hash或PD block所有权不正确时由断言报告失败。
    """

    config_path = tmp_path / "worker.yaml"
    config_path.write_text("model: /models/test\n", encoding="utf-8")

    config = load_worker_config(str(config_path))

    assert config.prefix_caching_hash_algo == "sha256"
    assert config.prefix_caching_hash_seed == 0
    assert config.kvbm_worker_provider == "auto"
    assert config.kvbm_pd_block_ownership == "scheduler"
    assert config.enable_auto_tools is None
    assert config.auto_detect_parsers is True


@pytest.mark.parametrize(
    "config_key",
    ["enable_auto_tools", "enable_auto_tool_choice", "enable-auto-tool-choice"],
)
def test_load_worker_config_accepts_tool_choice_aliases(tmp_path, config_key) -> None:
    """Worker YAML 应兼容项目原字段和 vLLM 官方字段名。"""
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(
        "\n".join(
            (
                "model: /models/Qwen3-4B",
                "tool_call_parser: hermes",
                "reasoning_parser: qwen3",
                f"{config_key}: true",
            )
        ),
        encoding="utf-8",
    )

    config = load_worker_config(str(config_path))

    assert config.tool_call_parser == "hermes"
    assert config.reasoning_parser == "qwen3"
    assert config.enable_auto_tools is True


def test_configure_prefix_caching_hash_seed_overrides_environment(monkeypatch) -> None:
    """配置函数应在 vLLM 初始化前写入统一的 hash 链种子。

    参数:
        monkeypatch: pytest 提供的环境变量隔离工具。

    返回:
        无返回值；环境变量未按配置覆盖时由断言报告失败。
    """

    monkeypatch.setenv("PYTHONHASHSEED", "99")
    args = argparse.Namespace(prefix_caching_hash_seed=17)

    configured = _configure_prefix_caching_hash_seed(args)

    assert configured == "17"
    assert os.environ["PYTHONHASHSEED"] == "17"


def test_configure_prefix_caching_hash_seed_rejects_invalid_value() -> None:
    """超出 Python 支持范围的 seed 应在启动引擎前失败。

    参数:
        无。

    返回:
        无返回值；未抛出 ValueError 时由 pytest 报告失败。
    """

    args = argparse.Namespace(prefix_caching_hash_seed=4294967296)

    with pytest.raises(ValueError, match="prefix_caching_hash_seed"):
        _configure_prefix_caching_hash_seed(args)


@pytest.mark.parametrize(
    ("worker_mode", "worker_type"),
    (
        ("prefill", "prefill"),
        ("decode", "decode"),
        ("aggregated", "combined"),
        ("unknown", "combined"),
    ),
)
def test_router_worker_type_matches_worker_mode(
    worker_mode: str,
    worker_type: str,
) -> None:
    """Worker 注册信息应使用 Router 可以识别的 P/D 类型。

    参数:
        worker_mode: Worker YAML 中的运行模式。
        worker_type: 预期写入 etcd 的 Router Worker 类型。

    返回:
        无返回值；映射不一致时由断言报告失败。
    """

    assert _router_worker_type(worker_mode) == worker_type


@pytest.mark.asyncio
async def test_decode_worker_starts_kv_event_relay_when_enabled() -> None:
    """D节点启用KV事件后应与P节点一样启动原生事件中继。

    参数:
        无。

    返回:
        无返回值；D节点未创建KvEventPublisher时由断言报告失败。
    """

    worker_publisher = Mock()
    worker_publisher.start = AsyncMock()
    kv_event_publisher = Mock()
    metrics_publisher = Mock()
    with (
        patch(
            "openjiuwentools.infer_router.vllm.worker_publisher.WorkerPublisher",
            return_value=worker_publisher,
        ),
        patch(
            "openjiuwentools.infer_router.vllm.worker_publisher.KvEventPublisher",
            return_value=kv_event_publisher,
        ) as kv_event_publisher_cls,
        patch(
            "openjiuwentools.infer_router.vllm.worker_publisher.MetricsPublisher",
            return_value=metrics_publisher,
        ),
    ):
        result = await _start_worker_publisher(
            {
                "enable_kv_cache_events": True,
                "endpoint": "tcp://*:5557",
                "topic": "",
            },
            relay_endpoint="tcp://*:9010",
            worker_id="decode-0",
            worker_mode="decode",
        )

    assert result == (worker_publisher, kv_event_publisher, metrics_publisher)
    worker_publisher.start.assert_awaited_once_with()
    kv_event_publisher_cls.assert_called_once_with(
        worker_publisher=worker_publisher,
        sub_endpoint="tcp://127.0.0.1:5557",
        topic="",
        worker_id="decode-0",
    )
    kv_event_publisher.start.assert_called_once_with()
    metrics_publisher.start.assert_called_once_with()


@pytest.mark.asyncio
async def test_decode_worker_without_kv_event_config_only_starts_metrics() -> None:
    """D节点未启用KV事件时不应额外启动事件中继。

    参数:
        无。

    返回:
        无返回值；未配置时仍创建KvEventPublisher则由断言报告失败。
    """

    worker_publisher = Mock()
    worker_publisher.start = AsyncMock()
    metrics_publisher = Mock()
    with (
        patch(
            "openjiuwentools.infer_router.vllm.worker_publisher.WorkerPublisher",
            return_value=worker_publisher,
        ),
        patch(
            "openjiuwentools.infer_router.vllm.worker_publisher.KvEventPublisher",
        ) as kv_event_publisher_cls,
        patch(
            "openjiuwentools.infer_router.vllm.worker_publisher.MetricsPublisher",
            return_value=metrics_publisher,
        ),
    ):
        _worker, kv_event_publisher, _metrics = await _start_worker_publisher(
            None,
            relay_endpoint="tcp://*:9010",
            worker_id="decode-0",
            worker_mode="decode",
        )

    assert kv_event_publisher is None
    kv_event_publisher_cls.assert_not_called()
