# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
from __future__ import annotations

import socket
import types

import pytest

from jiuwen_kvbm.benchmark.runner import run_transfer_benchmark
from jiuwen_kvbm.benchmark.standalone import (
    _comparison_output,
    _parse_args,
    _validate_target_arguments,
)
from jiuwen_kvbm.benchmark.tcp_staging import _recv_json, _send_json
from jiuwen_kvbm.integrations import vllm_ascend_018 as integration


def test_transfer_benchmark_reports_latency_and_effective_bandwidth() -> None:
    """统一基准统计应输出延迟和按有效 KV 字节计算的带宽。

    返回:
        无返回值；统计不符合预期时由断言报告失败。
    """

    samples = iter((2.0, 4.0, 6.0))
    result = run_transfer_benchmark(
        lambda: {"elapsed_ms": next(samples), "copy_ms": 1.0},
        transferred_bytes=1_000_000,
        warmup=0,
        iterations=3,
    )

    assert result["latency_ms"]["p50"] == 4.0
    assert result["effective_gbps"]["p50"] == 2.0
    assert result["phase_ms"]["copy_ms"]["mean"] == 1.0
    assert result["total_transferred_bytes"] == 3_000_000


def test_tcp_staging_json_frame_round_trip() -> None:
    """TCP staging 长度前缀 JSON 帧应可无损往返。

    返回:
        无返回值；协议编码不一致时由断言报告失败。
    """

    left, right = socket.socketpair()
    try:
        _send_json(left, {"version": 1, "source_blocks": [1, 2]})
        assert _recv_json(right) == {"version": 1, "source_blocks": [1, 2]}
    finally:
        left.close()
        right.close()


def test_hixl_benchmark_uses_llm_datadist_pull(monkeypatch) -> None:
    """HIXL 基准应重复调用真实 LLM-DataDist pull_blocks 边界。

    参数:
        monkeypatch: pytest 提供的环境和全局变量替换工具。

    返回:
        无返回值；调用次数或统计错误时由断言报告失败。
    """

    class Provider:
        @staticmethod
        def describe_block(block_id: int) -> object:
            return types.SimpleNamespace(nbytes=1024 + block_id * 0)

    class Backend:
        def __init__(self) -> None:
            self.calls = 0

        def pull_blocks(self, cache_id: int, mapping: object) -> None:
            assert cache_id == 7
            assert tuple(mapping.prompt_blocks) == (1, 2)
            assert tuple(mapping.decoder_blocks) == (11, 12)
            self.calls += 1

    backend = Backend()
    control = types.SimpleNamespace(provider=Provider())
    monkeypatch.setenv("JIUWEN_KVBM_BENCHMARK_ENABLED", "1")
    monkeypatch.setattr(integration, "_ENGINE_CORE_LLM_DATADIST_CACHE_ID", 7)

    result = integration._benchmark_pull_response(
        control,
        backend,
        {
            "backend": "hixl",
            "remote_cluster_id": 9,
            "source_blocks": [1, 2],
            "destination_blocks": [11, 12],
            "warmup": 1,
            "iterations": 2,
        },
    )

    assert result["ok"] is True
    assert result["backend"] == "hixl"
    assert result["transferred_bytes_per_iteration"] == 2048
    assert backend.calls == 3


def test_standalone_target_arguments_and_digest_comparison() -> None:
    """独立 benchmark 应解析目标端参数并报告两条路径的摘要一致性。

    返回:
        无返回值；参数默认值或比较输出错误时由断言报告失败。
    """

    args = _parse_args(
        [
            "target",
            "--device-id",
            "1",
            "--listen-ip",
            "10.0.0.1",
            "--source-ip",
            "10.0.0.1",
            "--source-tcp-host",
            "10.0.0.1",
        ]
    )
    _validate_target_arguments(args)
    result = {
        "latency_ms": {"p50": 2.0},
        "effective_gbps": {"p50": 10.0},
    }

    output = _comparison_output(
        args,
        result,
        {**result, "latency_ms": {"p50": 20.0}},
        hixl_digests=["same"],
        tcp_digests=["same"],
    )

    assert args.blocks == 4
    assert output["comparison"]["hixl_vs_tcp_latency_speedup_p50"] == 10.0
    assert output["validation"]["digests_match"] is True


def test_standalone_rejects_same_source_and_target_cluster() -> None:
    """独立 benchmark 应拒绝源端与目标端使用相同 cluster ID。

    返回:
        无返回值；未拒绝冲突 cluster ID 时测试失败。
    """

    args = _parse_args(
        [
            "target",
            "--device-id",
            "1",
            "--cluster-id",
            "0",
            "--source-cluster-id",
            "0",
            "--listen-ip",
            "10.0.0.1",
            "--source-ip",
            "10.0.0.1",
            "--source-tcp-host",
            "10.0.0.1",
        ]
    )

    with pytest.raises(ValueError, match="cluster IDs must be different"):
        _validate_target_arguments(args)
