from __future__ import annotations

import sys

import argparse
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

from jiuwen_kvbm.accessor import VllmAscend018TensorProvider
from jiuwen_kvbm.benchmark.runner import run_transfer_benchmark
from jiuwen_kvbm.benchmark.tcp_staging import TcpStagingClient, start_tcp_staging_server
from jiuwen_kvbm.transfer import (
    LLMDataDistBlockMapping,
    LLMDataDistCacheDesc,
    LLMDataDistCacheKey,
    LLMDataDistCluster,
    LLMDataDistConfig,
    LLMDataDistEndpoint,
    LLMDataDistRole,
    create_llm_datadist_backend,
)

logger = logging.getLogger("jiuwen.kvbm.benchmark.standalone")


def write_report(*values, file=None, flush=False):
    """Write CLI result data without routing it through diagnostic logging."""
    stream = sys.stdout if file is None else file
    stream.write(" ".join(str(value) for value in values) + "\n")
    if flush:
        stream.flush()


@dataclass
class _StandaloneRuntime:
    """独立 benchmark 的 NPU tensor、provider 和 LLM-DataDist 资源。"""

    provider: VllmAscend018TensorProvider
    backend: Any
    cache_id: int
    access_lock: threading.Lock


def main() -> int:
    """运行不依赖 vLLM Worker 的 HIXL/TCP 对比工具。

    返回:
        命令执行成功时返回 0；初始化、传输或正确性校验失败时返回 1。
    """

    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "source":
            return _run_source(args)
        return _run_target(args)
    except KeyboardInterrupt:
        return 130
    except Exception:
        logger.exception("Standalone HIXL/TCP benchmark failed")
        return 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析独立 benchmark 命令行参数。

    参数:
        argv: 可选参数列表；为空时读取当前进程命令行。

    返回:
        包含 source 或 target 子命令配置的 argparse Namespace。
    """

    parser = argparse.ArgumentParser(description="Benchmark LLM-DataDist/HIXL against TCP staging without vLLM workers")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("source", help="启动源端 NPU cache 和 TCP staging 服务")
    _add_tensor_arguments(source)
    _add_local_endpoint_arguments(source, default_cluster_id=0, default_port=19000)
    source.add_argument("--tcp-host", default="0.0.0.0")
    source.add_argument("--tcp-port", type=int, default=29011)
    source.add_argument("--max-blocks", type=int, default=64)

    target = subparsers.add_parser("target", help="在目标 NPU 上运行 HIXL/TCP 对比")
    _add_tensor_arguments(target)
    _add_local_endpoint_arguments(target, default_cluster_id=1, default_port=19001)
    target.add_argument("--source-cluster-id", type=int, default=0)
    target.add_argument("--source-ip", required=True, help="源端 HIXL 可达 IP")
    target.add_argument("--source-hixl-port", type=int, default=19000)
    target.add_argument("--source-tcp-host", required=True, help="源端 TCP staging 可达 IP")
    target.add_argument("--source-tcp-port", type=int, default=29011)
    target.add_argument("--blocks", type=int, default=4, help="每次传输的 block 数")
    target.add_argument("--warmup", type=int, default=5)
    target.add_argument("--iterations", type=int, default=50)
    target.add_argument("--timeout-ms", type=int, default=30000)
    return parser.parse_args(argv)


def _add_tensor_arguments(parser: argparse.ArgumentParser) -> None:
    """向子命令增加测试 KV tensor 布局参数。

    参数:
        parser: 待增加参数的 argparse 子命令 parser。

    返回:
        无返回值。
    """

    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )


def _add_local_endpoint_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_cluster_id: int,
    default_port: int,
) -> None:
    """向子命令增加本地 LLM-DataDist endpoint 参数。

    参数:
        parser: 待增加参数的 argparse 子命令 parser。
        default_cluster_id: 当前角色默认 cluster ID。
        default_port: 当前角色默认 HIXL 监听端口。

    返回:
        无返回值。
    """

    parser.add_argument("--cluster-id", type=int, default=default_cluster_id)
    parser.add_argument("--listen-ip", required=True, help="本端 HIXL/HCCN 可达 IP")
    parser.add_argument("--listen-port", type=int, default=default_port)


def _run_source(args: argparse.Namespace) -> int:
    """启动独立源端并等待目标端发起 HIXL 或 TCP 拉取。

    参数:
        args: source 子命令解析结果。

    返回:
        正常停止时返回 0。
    """

    runtime = _create_runtime(args, role=LLMDataDistRole.PROMPT, initialize_source=True)
    server = None
    stop = threading.Event()

    def _request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        server, _thread = start_tcp_staging_server(
            runtime.provider,
            args.tcp_host,
            args.tcp_port,
            runtime.access_lock,
            max_blocks=args.max_blocks,
        )
        output = {
            "ok": True,
            "role": "source",
            "cluster_id": args.cluster_id,
            "device_id": args.device_id,
            "hixl_endpoint": f"{args.listen_ip}:{args.listen_port}",
            "tcp_endpoint": f"{args.tcp_host}:{args.tcp_port}",
            "cache_id": runtime.cache_id,
            "tensor": runtime.provider.diagnostics(0),
        }
        write_report(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
        while not stop.wait(1.0):
            pass
        return 0
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        runtime.backend.finalize()


def _run_target(args: argparse.Namespace) -> int:
    """初始化目标 NPU，并依次执行 HIXL 与 TCP staging benchmark。

    参数:
        args: target 子命令解析结果。

    返回:
        两种传输都成功且结果摘要一致时返回 0，否则抛出异常并由 main 返回 1。
    """

    _validate_target_arguments(args)
    runtime = _create_runtime(args, role=LLMDataDistRole.DECODER, initialize_source=False)
    source_blocks = list(range(args.blocks))
    destination_blocks = list(range(args.blocks))
    transferred_bytes = sum(runtime.provider.describe_block(block_id).nbytes for block_id in destination_blocks)
    cluster = LLMDataDistCluster(
        remote_cluster_id=args.source_cluster_id,
        remote_ip=args.source_ip,
        remote_port=args.source_hixl_port,
        local_ip=args.listen_ip,
        local_port=args.listen_port,
    )
    try:
        runtime.backend.link_clusters([cluster], timeout_ms=args.timeout_ms)
        mapping = LLMDataDistBlockMapping(
            cache_key=LLMDataDistCacheKey(
                key=f"cluster:{args.source_cluster_id}:model:0",
                prompt_cluster_id=args.source_cluster_id,
                prompt_batch_index=0,
            ),
            prompt_blocks=tuple(source_blocks),
            decoder_blocks=tuple(destination_blocks),
        )

        hixl_result = run_transfer_benchmark(
            lambda: _run_hixl_once(runtime, mapping),
            transferred_bytes=transferred_bytes,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        hixl_digests = _block_digests(runtime.provider, destination_blocks)
        _zero_blocks(runtime, destination_blocks)

        with TcpStagingClient(
            args.source_tcp_host,
            args.source_tcp_port,
            runtime.provider,
            runtime.access_lock,
            timeout_seconds=max(args.timeout_ms / 1000.0, 1.0),
        ) as tcp_client:
            tcp_result = run_transfer_benchmark(
                lambda: tcp_client.pull_blocks(
                    source_blocks,
                    destination_blocks,
                    expected_nbytes=transferred_bytes,
                ),
                transferred_bytes=transferred_bytes,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        tcp_digests = _block_digests(runtime.provider, destination_blocks)
        output = _comparison_output(
            args,
            hixl_result,
            tcp_result,
            hixl_digests=hixl_digests,
            tcp_digests=tcp_digests,
        )
        write_report(json.dumps(output, ensure_ascii=False, indent=2))
        if not output["validation"]["digests_match"]:
            raise RuntimeError("HIXL and TCP destination block digests do not match")
        return 0
    finally:
        try:
            runtime.backend.unlink_clusters(
                [cluster],
                timeout_ms=args.timeout_ms,
                force=True,
            )
        except Exception:
            logger.warning("Failed to unlink standalone benchmark cluster", exc_info=True)
        runtime.backend.finalize()


def _create_runtime(
    args: argparse.Namespace,
    *,
    role: LLMDataDistRole,
    initialize_source: bool,
) -> _StandaloneRuntime:
    """分配模拟 KV cache tensor 并注册到 LLM-DataDist。

    参数:
        args: 包含 tensor 布局、NPU 和 endpoint 配置的命令行参数。
        role: 当前 benchmark 进程的 LLM-DataDist 角色。
        initialize_source: 为 True 时用确定性数据填充源 block；否则清零目标 block。

    返回:
        持有 provider、backend、cache ID 和访问锁的运行时对象。
    """

    kv_caches = _allocate_kv_caches(args, initialize_source=initialize_source)
    provider = VllmAscend018TensorProvider(
        kv_caches,
        device_id=args.device_id,
        delete_mode="zero",
    )
    config = LLMDataDistConfig(
        enabled=True,
        endpoint=LLMDataDistEndpoint(
            cluster_id=args.cluster_id,
            role=role,
            listen_ip=args.listen_ip,
            listen_port=args.listen_port,
            device_id=args.device_id,
        ),
        transfer_backend="hixl",
        enable_cache_manager=True,
    )
    backend = create_llm_datadist_backend(config, mode="python")
    backend.initialize()
    diagnostics = provider.diagnostics(0)
    handle = backend.register_cache(
        LLMDataDistCacheDesc(
            placement="device",
            data_type=provider.blocks_cache_dtype(),
            shape=provider.blocks_cache_shape(),
            num_tensors=int(diagnostics["layer_count"]) * 2,
            tensor_addresses=provider.tensor_base_addresses(),
            tensor_num_per_layer=2,
            layer_count=int(diagnostics["layer_count"]),
        ),
        [
            LLMDataDistCacheKey(
                key=f"cluster:{args.cluster_id}:model:0",
                prompt_cluster_id=args.cluster_id,
                prompt_batch_index=0,
            )
        ],
    )
    return _StandaloneRuntime(
        provider=provider,
        backend=backend,
        cache_id=handle.cache_id,
        access_lock=threading.Lock(),
    )


def _allocate_kv_caches(
    args: argparse.Namespace,
    *,
    initialize_source: bool,
) -> dict[str, tuple[Any, Any]]:
    """在指定 NPU 上分配与 vLLM-Ascend 相同布局的测试 KV tensor。

    参数:
        args: 包含 device_id、shape 和 dtype 的命令行参数。
        initialize_source: 是否写入按层区分的确定性源数据。

    返回:
        layer 名到 `(key_tensor, value_tensor)` 的字典。
    """

    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("standalone benchmark requires torch and torch_npu") from exc

    dimensions = (
        args.num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
    )
    if any(int(value) <= 0 for value in dimensions) or args.num_layers <= 0:
        raise ValueError("all tensor dimensions and num_layers must be positive")
    torch.npu.set_device(args.device_id)
    dtype = getattr(torch, args.dtype)
    device = f"npu:{args.device_id}"
    kv_caches: dict[str, tuple[Any, Any]] = {}
    for layer_id in range(args.num_layers):
        key = torch.empty(dimensions, dtype=dtype, device=device)
        value = torch.empty(dimensions, dtype=dtype, device=device)
        if initialize_source:
            for block_id in range(args.num_blocks):
                key[block_id].fill_(float((layer_id * 2 + block_id + 1) % 97))
                value[block_id].fill_(float((layer_id * 2 + block_id + 2) % 97))
        else:
            key.zero_()
            value.zero_()
        kv_caches[f"model.layers.{layer_id}.self_attn.attn"] = (key, value)
    torch.npu.synchronize()
    return kv_caches


def _run_hixl_once(
    runtime: _StandaloneRuntime,
    mapping: LLMDataDistBlockMapping,
) -> dict[str, float]:
    """执行一次同步 LLM-DataDist/HIXL block 拉取并计时。

    参数:
        runtime: 目标端独立 benchmark 运行时。
        mapping: 源 block 到目标 block 的映射。

    返回:
        包含 elapsed_ms 的单次采样字典。
    """

    started = time.perf_counter()
    runtime.backend.pull_blocks(runtime.cache_id, mapping)
    _synchronize_npu()
    return {"elapsed_ms": (time.perf_counter() - started) * 1000.0}


def _zero_blocks(runtime: _StandaloneRuntime, block_ids: list[int]) -> None:
    """在 TCP benchmark 前清零目标物理 block。

    参数:
        runtime: 目标端独立 benchmark 运行时。
        block_ids: 需要清零的目标 block ID。

    返回:
        无返回值。
    """

    with runtime.access_lock:
        for block_id in block_ids:
            runtime.provider.delete_block(block_id)
    _synchronize_npu()


def _block_digests(
    provider: VllmAscend018TensorProvider,
    block_ids: list[int],
) -> list[str]:
    """计算目标 block 摘要用于比较两条传输路径的正确性。

    参数:
        provider: 目标 NPU tensor provider。
        block_ids: 需要计算摘要的物理 block ID。

    返回:
        与 block_ids 顺序一致的 SHA-256 字符串列表。
    """

    return [str(provider.block_digest(block_id)["sha256"]) for block_id in block_ids]


def _comparison_output(
    args: argparse.Namespace,
    hixl_result: dict[str, Any],
    tcp_result: dict[str, Any],
    *,
    hixl_digests: list[str],
    tcp_digests: list[str],
) -> dict[str, Any]:
    """组合 HIXL、TCP 性能结果与正确性校验结果。

    参数:
        args: target 子命令参数。
        hixl_result: HIXL benchmark 汇总。
        tcp_result: TCP staging benchmark 汇总。
        hixl_digests: HIXL 完成后的目标 block 摘要。
        tcp_digests: TCP 完成后的目标 block 摘要。

    返回:
        可直接输出为 JSON 的完整结果。
    """

    hixl_p50 = float(hixl_result["latency_ms"]["p50"])
    tcp_p50 = float(tcp_result["latency_ms"]["p50"])
    return {
        "ok": True,
        "mode": "standalone",
        "source_cluster_id": args.source_cluster_id,
        "target_cluster_id": args.cluster_id,
        "block_count": args.blocks,
        "results": {
            "hixl": hixl_result,
            "tcp_staging": tcp_result,
        },
        "comparison": {
            "hixl_vs_tcp_latency_speedup_p50": tcp_p50 / hixl_p50,
            "hixl_latency_p50_ms": hixl_p50,
            "tcp_staging_latency_p50_ms": tcp_p50,
        },
        "validation": {
            "digests_match": hixl_digests == tcp_digests,
            "hixl_sha256": hixl_digests,
            "tcp_staging_sha256": tcp_digests,
        },
    }


def _validate_target_arguments(args: argparse.Namespace) -> None:
    """校验目标端 block 数与 tensor 容量。

    参数:
        args: target 子命令参数。

    返回:
        参数合法时无返回值。
    """

    if args.blocks <= 0:
        raise ValueError("blocks must be positive")
    if args.blocks > args.num_blocks:
        raise ValueError("blocks must not exceed num_blocks")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if args.source_cluster_id == args.cluster_id:
        raise ValueError("source and target cluster IDs must be different")


def _synchronize_npu() -> None:
    """等待当前 NPU 上的异步操作完成。

    返回:
        无返回值。
    """

    import torch

    torch.npu.synchronize()


if __name__ == "__main__":
    raise SystemExit(main())
