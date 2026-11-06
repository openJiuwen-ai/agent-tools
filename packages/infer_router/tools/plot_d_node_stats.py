#!/usr/bin/env python3
"""绘制 PD 分离下 4 个 D 节点的 cache usage 与吞吐变化曲线。

从每个 D 节点的日志（默认 d1.log ~ d4.log）中解析 vLLM 引擎统计行：
    Avg prompt throughput: X tokens/s, Avg generation throughput: Y tokens/s,
    Running: N reqs, Waiting: M reqs, GPU KV cache usage: Z%,
    Prefix cache hit rate: W%
输出三张 PNG：
    <prefix>_cache_usage.png   —— 各 D 节点 GPU KV cache usage(%) 曲线
    <prefix>_throughput.png    —— 各 D 节点 Avg generation throughput 曲线
    <prefix>_running.png       —— 各 D 节点 Running 请求数曲线

为便于观察负载是否均匀，每张图采用分面布局：每个 D 节点独占一行
子图并共享 y 轴范围，本节点彩色高亮、其余节点灰色衬托，虚线为本
节点活跃均值（只统计 running/waiting>0 的样本，空载段的 0 值不
拉低平均），每行标注 busy 占比；控制台同时输出各节点活跃均值与
跨节点倾斜倍数 imbalance(max/min)。曲线本身仍绘制完整时间线。

时间轴为相对全部节点最早时间戳的秒数，便于对齐比较各节点负载。
兼容两种时间戳格式：
    2026-09-16 14:03:05,123 INFO [jiuwen.worker] ...   （worker 主进程）
    INFO 09-16 14:03:05 [loggers.py:123] ...           （vLLM engine-core）

用法：
    python plot_d_node_stats.py d1.log d2.log d3.log d4.log
    python plot_d_node_stats.py -o bench1 d*.log --year 2026
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
NUMBER = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

FULL_TS_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:[.,](\d{1,6}))?(?!\d)")
SHORT_TS_RE = re.compile(r"(?<![\d-])(\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2})(?:[.,](\d{1,6}))?(?!\d)")

# 引擎统计行的唯一锚点：只有该行会出现 "GPU KV cache usage:"。
STATS_ANCHOR_RE = re.compile(r"GPU KV cache usage:")
GEN_TPS_RE = re.compile(rf"Avg generation throughput:\s*({NUMBER})\s*tokens/s")
PROMPT_TPS_RE = re.compile(rf"Avg prompt throughput:\s*({NUMBER})\s*tokens/s")
KV_USAGE_RE = re.compile(rf"GPU KV cache usage:\s*({NUMBER})\s*%")
PREFIX_HIT_RE = re.compile(rf"Prefix cache hit rate:\s*({NUMBER})\s*%")
RUNNING_RE = re.compile(rf"Running:\s*({NUMBER})\s*reqs")
WAITING_RE = re.compile(rf"Waiting:\s*({NUMBER})\s*reqs")


def write_report(*values, file=None, flush=False):
    """Write CLI result data without routing it through diagnostic logging."""
    stream = sys.stdout if file is None else file
    stream.write(" ".join(str(value) for value in values) + "\n")
    if flush:
        stream.flush()


def parse_timestamp(line: str, default_year: int) -> datetime | None:
    match = FULL_TS_RE.search(line)
    if match:
        date, clock, frac = match.groups()
        micro = int((frac or "0").ljust(6, "0")[:6])
        return datetime.fromisoformat(f"{date}T{clock}.{micro:06d}")
    match = SHORT_TS_RE.search(line)
    if match:
        month, day, clock, frac = match.groups()
        micro = int((frac or "0").ljust(6, "0")[:6])
        return datetime.fromisoformat(f"{default_year}-{month}-{day}T{clock}.{micro:06d}")
    return None


def search_float(line: str, pattern: re.Pattern[str]) -> float | None:
    match = pattern.search(line)
    return float(match.group(1)) if match else None


def parse_log(
    path: Path,
    default_year: int,
) -> list[dict[str, float | datetime]]:
    """提取该日志中所有引擎统计样本（时间戳 + 各指标）。"""
    samples: list[dict[str, float | datetime]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = ANSI_RE.sub("", raw)
            if not STATS_ANCHOR_RE.search(line):
                continue
            timestamp = parse_timestamp(line, default_year)
            if timestamp is None:
                continue
            samples.append(
                {
                    "ts": timestamp,
                    "kv_usage": search_float(line, KV_USAGE_RE),
                    "gen_tps": search_float(line, GEN_TPS_RE),
                    "prompt_tps": search_float(line, PROMPT_TPS_RE),
                    "prefix_hit": search_float(line, PREFIX_HIT_RE),
                    "running": search_float(line, RUNNING_RE),
                    "waiting": search_float(line, WAITING_RE),
                }
            )
    # 按时间排序（vLLM 多 DP rank 或乱序写日志时保证曲线单调）。
    samples.sort(key=lambda s: s["ts"])
    return samples


def dedupe(samples: list[dict[str, float | datetime]]) -> list[dict]:
    """合并同一秒内重复的统计样本（取均值）。"""
    if not samples:
        return []
    buckets: dict[datetime, list[dict]] = {}
    for sample in samples:
        buckets.setdefault(sample["ts"], []).append(sample)
    merged = []
    for ts, group in sorted(buckets.items()):
        merged.append(
            {
                key: (
                    ts
                    if key == "ts"
                    else sum(g[key] for g in group if g[key] is not None)
                    / max(1, sum(1 for g in group if g[key] is not None))
                )
                for key in group[0]
            }
        )
    return merged


def summarize(name: str, values: list[float]) -> str:
    if not values:
        return f"  {name}: no data"
    ordered = sorted(values)
    return (
        f"  {name}: count={len(values)} avg={sum(values) / len(values):.2f} max={ordered[-1]:.2f} min={ordered[0]:.2f}"
    )


def is_active(sample: dict[str, float | datetime]) -> bool:
    """该样本是否处于服务期：running/waiting 任一不为 0。

    D 节点空载时统计行仍会打印（吞吐为 0），若计入均值会拉低平均、
    放大空载越久越失真的问题，因此均值与倾斜倍数只统计活跃样本。
    字段缺失时视为活跃，避免丢数据。
    """
    running = sample.get("running") or 0.0
    waiting = sample.get("waiting") or 0.0
    return running > 0 or waiting > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot per-D-node KV cache usage and throughput curves.")
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="D 节点日志路径，例如 d1.log d2.log d3.log d4.log",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        default="d_nodes",
        help="输出 PNG 文件名前缀（默认 d_nodes）",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now(tz=timezone.utc).astimezone().year,
        help="短时间戳（mm-dd）缺少年份时的补全年份，默认当前年份",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="对曲线做 N 点滑动平均（0 = 原始数据）",
    )
    args = parser.parse_args()

    nodes: dict[str, list[dict]] = {}
    for path in args.logs:
        if not path.exists():
            write_report(f"error: log file not found: {path}", file=sys.stderr)
            return 1
        samples = dedupe(parse_log(path, args.year))
        label = path.stem
        nodes[label] = samples

    valid = {label: s for label, s in nodes.items() if s}
    if not valid:
        write_report(
            "error: no engine stats lines (containing 'GPU KV cache usage') found in any log",
            file=sys.stderr,
        )
        return 1
    for label, samples in nodes.items():
        if not samples:
            write_report(f"warning: no samples parsed from {label}", file=sys.stderr)

    t0 = min(samples[0]["ts"] for samples in valid.values())
    write_report(f"time origin (earliest sample): {t0.isoformat()}")

    def active_values(samples: list[dict], key: str) -> list[float]:
        return [
            s[key]
            for s in samples
            if s[key] is not None and is_active(s)  # type: ignore[operator]
        ]

    busy_ratios: dict[str, float] = {}
    for label in valid:
        series = valid[label]
        active_count = sum(1 for s in series if is_active(s))
        busy = active_count / len(series) if series else 0.0
        busy_ratios[label] = busy
        span = (series[-1]["ts"] - series[0]["ts"]).total_seconds()
        write_report(f"{label}: samples={len(series)} active={active_count} busy={busy * 100:.0f}% span={span:.0f}s")
        write_report(summarize("kv_cache_usage(%, active)", active_values(series, "kv_usage")))
        write_report(summarize("gen_throughput(tokens/s, active)", active_values(series, "gen_tps")))
        hit = [s["prefix_hit"] for s in series if s["prefix_hit"] is not None]
        write_report(summarize("prefix_cache_hit(%, cumulative)", hit))

    def series_xy(samples: list[dict], key: str) -> tuple[list[float], list[float]]:
        xs = [(s["ts"] - t0).total_seconds() for s in samples if s[key] is not None]
        ys = [s[key] for s in samples if s[key] is not None]  # type: ignore[misc]
        if args.smooth > 1 and len(ys) > args.smooth:
            window = args.smooth
            ys = [
                sum(ys[max(0, i - window + 1):i + 1]) / len(ys[max(0, i - window + 1):i + 1])
                for i in range(len(ys))
            ]
        return xs, ys

    figures = [
        (
            f"{args.output_prefix}_cache_usage.png",
            "kv_usage",
            "D-node GPU KV cache usage (%)",
            100.0,
        ),
        (
            f"{args.output_prefix}_throughput.png",
            "gen_tps",
            "D-node avg generation throughput (tokens/s)",
            None,
        ),
        (
            f"{args.output_prefix}_running.png",
            "running",
            "D-node running requests",
            None,
        ),
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for filename, key, title, fixed_ymax in figures:
        series_map = {label: series_xy(valid[label], key) for label in valid}
        # 均值只统计活跃样本（running/waiting > 0），空载段的 0 值
        # 不再拉低平均；虚线与倾斜倍数均基于该活跃均值。
        means = {}
        for selected_label, selected_active in (
            (selected_label, active_values(valid[selected_label], key)) for selected_label in valid
        ):
            means[selected_label] = sum(selected_active) / len(selected_active) if selected_active else None

        # 跨节点活跃均值与倾斜倍数：直接回答负载是否均匀。
        measured = {label: m for label, m in means.items() if m is not None}
        if measured:
            lo, hi = min(measured.values()), max(measured.values())
            ratio = (hi / lo) if lo > 0 else float("inf")
            descriptions = []
            for label in series_map:
                mean = means.get(label)
                descriptions.append(f"{label}=" + (f"{mean:.1f}" if mean is not None else "n/a"))
            detail = ", ".join(descriptions)
            write_report(f"{key} cross-node active mean: {detail} imbalance(max/min)={ratio:.2f}x")
        else:
            write_report(f"{key} cross-node active mean: no active samples")

        data_max = max((max(ys) for _xs, ys in series_map.values() if ys), default=1.0)
        if fixed_ymax is not None:
            ymax = fixed_ymax
        else:
            # 全空载（data_max=0）时给一个非零上限，避免退化坐标轴。
            ymax = data_max * 1.08 if data_max > 0 else 1.0

        # 每个节点独占一行（共享 y 轴范围）：本节点彩色高亮，其余节点
        # 灰色衬托，虚线为本节点活跃均值——曲线交杂时也能直接看出
        # 各节点负载水平差异。
        fig, axes = plt.subplots(
            nrows=len(series_map),
            figsize=(12, 2.8 * len(series_map)),
            dpi=150,
            sharex=True,
            layout="constrained",
        )
        axes_list = [axes] if len(series_map) == 1 else list(axes)
        for ax, (idx, (label, (xs, ys))) in zip(axes_list, enumerate(series_map.items())):
            color = colors[idx % len(colors)]
            for other, (oxs, oys) in series_map.items():
                if other == label:
                    continue
                ax.plot(oxs, oys, color="0.82", linewidth=0.9, zorder=1)
            ax.plot(xs, ys, color=color, linewidth=1.5, zorder=3)
            mean = means.get(label)
            if mean is not None and mean > 0:
                ax.axhline(
                    mean,
                    color=color,
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.8,
                    zorder=2,
                )
            mean_txt = f"{mean:.1f}" if mean is not None else "n/a"
            ax.set_ylim(0, ymax)
            ax.set_ylabel(
                f"{label}\navg={mean_txt} busy={busy_ratios[label] * 100:.0f}%",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
        axes_list[0].set_title(f"{title}   (t0={t0.isoformat()}; gray=other nodes, dashed=this node's active avg)")
        axes_list[-1].set_xlabel("time since t0 (s)")
        fig.savefig(filename)
        plt.close(fig)
        write_report(f"saved: {filename}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
