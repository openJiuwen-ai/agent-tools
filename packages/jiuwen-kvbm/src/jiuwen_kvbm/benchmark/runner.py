from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Any


def run_transfer_benchmark(
    operation: Callable[[], dict[str, Any] | None],
    *,
    transferred_bytes: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """重复执行一次传输操作并生成统一性能统计。

    参数:
        operation: 单次传输函数。函数可以返回包含 elapsed_ms 的字典；若未返回
            elapsed_ms，则调用方必须在函数内部补齐，避免把准备阶段计入传输时间。
        transferred_bytes: 每次迭代搬运的有效 KV 数据字节数。
        warmup: 正式采样前的预热次数，预热结果不进入统计。
        iterations: 正式采样次数。

    返回:
        包含延迟分位数、有效带宽分位数和可选阶段耗时的字典。

    异常:
        ValueError: 参数无效，或者 operation 没有返回有效 elapsed_ms。
    """

    if transferred_bytes <= 0:
        raise ValueError("transferred_bytes must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    for _ in range(warmup):
        operation()

    samples: list[dict[str, Any]] = []
    for _ in range(iterations):
        sample = dict(operation() or {})
        elapsed_ms = float(sample.get("elapsed_ms", 0.0))
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise ValueError("operation must return a positive elapsed_ms")
        sample["elapsed_ms"] = elapsed_ms
        sample["effective_gbps"] = transferred_bytes * 8.0 / elapsed_ms / 1_000_000.0
        samples.append(sample)

    latency = [sample["elapsed_ms"] for sample in samples]
    bandwidth = [sample["effective_gbps"] for sample in samples]
    result: dict[str, Any] = {
        "warmup": warmup,
        "iterations": iterations,
        "transferred_bytes_per_iteration": transferred_bytes,
        "total_transferred_bytes": transferred_bytes * iterations,
        "latency_ms": _summarize(latency),
        "effective_gbps": _summarize(bandwidth),
    }
    phases = _summarize_numeric_fields(samples, excluded={"elapsed_ms", "effective_gbps"})
    if phases:
        result["phase_ms"] = phases
    return result


def _summarize(values: list[float]) -> dict[str, float]:
    """汇总一组浮点采样值。

    参数:
        values: 非空浮点采样列表。

    返回:
        包含 min、mean、p50、p95、p99 和 max 的统计字典。
    """

    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    """使用线性插值计算分位数。

    参数:
        ordered: 已按升序排列的非空采样列表。
        quantile: 0 到 1 之间的目标分位。

    返回:
        线性插值得到的分位数。
    """

    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_numeric_fields(
    samples: list[dict[str, Any]],
    *,
    excluded: set[str],
) -> dict[str, dict[str, float]]:
    """汇总所有采样都包含的附加数值字段。

    参数:
        samples: operation 返回的采样字典。
        excluded: 不作为阶段耗时汇总的字段名。

    返回:
        字段名到统计字典的映射；没有公共数值字段时返回空字典。
    """

    if not samples:
        return {}
    common = set(samples[0]).difference(excluded)
    for sample in samples[1:]:
        common.intersection_update(sample)
    result: dict[str, dict[str, float]] = {}
    for name in sorted(common):
        values = [sample[name] for sample in samples]
        if all(isinstance(value, int | float) for value in values):
            result[name] = _summarize([float(value) for value in values])
    return result
