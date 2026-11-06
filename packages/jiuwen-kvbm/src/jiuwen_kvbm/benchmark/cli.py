from __future__ import annotations

import sys
from urllib.parse import urljoin

import argparse
import json
import urllib.error
import urllib.request
from typing import Any


def write_report(*values, file=None, flush=False):
    """Write CLI result data without routing it through diagnostic logging."""
    stream = sys.stdout if file is None else file
    stream.write(" ".join(str(value) for value in values) + "\n")
    if flush:
        stream.flush()


def main() -> int:
    """运行 HIXL/TCP staging 对比命令行工具。

    返回:
        所有请求成功时返回进程码 0；任一请求失败时返回 1。
    """

    args = _parse_args()
    common = {
        "source_blocks": _parse_block_ids(args.source_blocks),
        "destination_blocks": _parse_block_ids(args.destination_blocks),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timeout_ms": args.timeout_ms,
    }
    backends = ["hixl", "tcp_staging"] if args.backend == "both" else [args.backend]
    results: dict[str, dict[str, Any]] = {}
    success = True
    for backend in backends:
        payload = {**common, "backend": backend}
        if backend == "hixl":
            payload.update(
                {
                    "remote_cluster_id": args.remote_cluster_id,
                    "model_id": args.model_id,
                }
            )
        else:
            payload.update(
                {
                    "source_host": args.source_host,
                    "source_port": args.source_port,
                }
            )
        result = _post_json(
            urljoin(args.target_url.rstrip("/") + "/", "datadist/benchmark/pull"),
            payload,
            timeout_seconds=max(1.0, args.http_timeout_seconds),
        )
        results[backend] = result
        success = success and bool(result.get("ok"))

    output: dict[str, Any] = {"results": results}
    if all(name in results and results[name].get("ok") for name in ("hixl", "tcp_staging")):
        hixl_p50 = float(results["hixl"]["latency_ms"]["p50"])
        tcp_p50 = float(results["tcp_staging"]["latency_ms"]["p50"])
        output["comparison"] = {
            "hixl_vs_tcp_latency_speedup_p50": tcp_p50 / hixl_p50,
            "hixl_latency_p50_ms": hixl_p50,
            "tcp_staging_latency_p50_ms": tcp_p50,
        }
    write_report(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if success else 1


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回:
        argparse Namespace。
    """

    parser = argparse.ArgumentParser(
        description="Compare LLM-DataDist/HIXL with CPU RAM TCP staging for real KV blocks"
    )
    parser.add_argument(
        "--target-url",
        required=True,
        help="目标 worker engine-core KVBM 控制地址，例如 http://127.0.0.1:18020",
    )
    parser.add_argument(
        "--backend",
        choices=["hixl", "tcp_staging", "both"],
        default="both",
    )
    parser.add_argument("--source-blocks", required=True, help="逗号分隔的源物理 block ID")
    parser.add_argument(
        "--destination-blocks",
        required=True,
        help="逗号分隔的目标物理 block ID",
    )
    parser.add_argument("--remote-cluster-id", type=int, default=0)
    parser.add_argument("--model-id", type=int, default=0)
    parser.add_argument("--source-host", default="127.0.0.1")
    parser.add_argument("--source-port", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--http-timeout-seconds", type=float, default=600.0)
    return parser.parse_args()


def _parse_block_ids(value: str) -> list[int]:
    """解析逗号分隔的物理 block ID。

    参数:
        value: 例如 10,11,12 的字符串。

    返回:
        非空整数列表。
    """

    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("block ID list must not be empty")
    return result


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """向 engine-core 基准入口发送 JSON 请求。

    参数:
        url: 完整 HTTP URL。
        payload: JSON 请求体。
        timeout_seconds: HTTP 总超时秒数。

    返回:
        服务端 JSON 响应；HTTP 错误响应也会尽量解析并返回。
    """

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read().decode("utf-8"))
        except Exception:
            return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    raise SystemExit(main())
