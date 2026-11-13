#!/usr/bin/env python3
"""Map msopprof OPPROF basic-block hot spots back to source lines.

The script intentionally uses only files emitted by ``msprof op``/OPPROF:

* ``OpBasicInfo*.csv`` and ``PipeUtilization*.csv`` for op-level context.
* ``dump/fdata`` for basic-block execution counts.
* ``dump/*bbbmap*`` for optional basic-block ids.
* ``dump/aicore_binary.o`` plus CANN ``llvm-objdump`` for address -> line info.

This is useful when MindStudio's ``visualize_data.bin`` points to hot source
lines and we want a command-line, reproducible check from the profiled binary.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_OBJDUMP_CANDIDATES = (
    "/usr/local/Ascend/cann-8.5.1/bin/llvm-objdump",
    "/usr/local/Ascend/cann-9.2.0/bin/llvm-objdump",
    "/usr/local/Ascend/ascend-toolkit/latest/bin/llvm-objdump",
)

ADDR_RE = re.compile(r"^\s*([0-9a-fA-F]+):")
SOURCE_RE = re.compile(r"^(.+):(\d+)$")
BBMAP_RE = re.compile(r"BBB id:(\d+)\s+Offset:\s+0x([0-9a-fA-F]+)")

PIPE_METRICS = (
    "aic_time(us)",
    "aic_cube_time(us)",
    "aic_scalar_time(us)",
    "aic_mte1_time(us)",
    "aic_mte2_time(us)",
    "aic_mte3_time(us)",
    "aic_fixpipe_time(us)",
    "aiv_time(us)",
    "aiv_vec_time(us)",
    "aiv_scalar_time(us)",
    "aiv_mte2_time(us)",
    "aiv_mte3_time(us)",
)


def find_objdump(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("llvm-objdump")
    if found:
        return found
    for candidate in DEFAULT_OBJDUMP_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("CANN llvm-objdump not found")


def is_opprof_op_dir(path: Path) -> bool:
    return (
        (path / "dump/fdata").is_file()
        and (
            (path / "dump/aicore_binary.o").is_file()
            or (path / "dump/kernel_data/aicore_binary.o").is_file()
        )
    )


def op_dir_matches_filter(path: Path, op_filter: str | None) -> bool:
    if not op_filter:
        return True
    if op_filter in str(path):
        return True

    op_basic_path = find_first(path, ("OpBasicInfo*.csv",))
    for row in read_csv_rows(op_basic_path):
        if op_filter in row.get("Op Name", ""):
            return True

    text_path = path / "dump/op_basic_info.txt"
    if text_path.is_file():
        try:
            return op_filter in text_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
    return False


def discover_op_dirs(root: Path, op_filter: str | None) -> list[Path]:
    if is_opprof_op_dir(root):
        dirs = [root]
    else:
        dirs = sorted({p.parent.parent for p in root.rglob("dump/fdata")
                       if (p.parent / "aicore_binary.o").is_file()})
    if op_filter:
        dirs = [p for p in dirs if op_dir_matches_filter(p, op_filter)]
    return dirs


def find_first(path: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def float_or_none(value: str | None) -> float | None:
    if value is None or value in ("", "NA"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize_pipe(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_sub_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sub_block[row.get("sub_block_id", "unknown")].append(row)

    summary: dict[str, Any] = {}
    for sub_block, sub_rows in sorted(by_sub_block.items()):
        metric_summary: dict[str, Any] = {"rows": len(sub_rows)}
        for metric in PIPE_METRICS:
            values = [
                parsed
                for row in sub_rows
                if (parsed := float_or_none(row.get(metric))) is not None
            ]
            if values:
                metric_summary[metric] = {
                    "avg": mean(values),
                    "max": max(values),
                }
        summary[sub_block] = metric_summary
    return summary


def parse_fdata(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            launch, kernel, offset_text, count_text = parts
            try:
                records.append({
                    "launch": int(launch),
                    "kernel": kernel,
                    "offset": int(offset_text, 16),
                    "count": int(count_text),
                })
            except ValueError:
                continue
    return records


def parse_bbbmap(dump_dir: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for path in sorted(dump_dir.glob("*bbbmap*")):
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                match = BBMAP_RE.search(line)
                if match:
                    mapping[int(match.group(2), 16)] = int(match.group(1))
    return mapping


def run_objdump(objdump: str, elf: Path, timeout_s: int) -> str:
    return subprocess.check_output(
        [objdump, "--disassemble-aicore", "--line-numbers", str(elf)],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=timeout_s,
    )


def parse_objdump_lines(output: str) -> tuple[list[int], dict[int, str]]:
    current_source = "unknown"
    addr_to_source: dict[int, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        source_match = SOURCE_RE.match(stripped)
        if source_match and not ADDR_RE.match(stripped):
            current_source = stripped
            continue
        addr_match = ADDR_RE.match(line)
        if addr_match:
            addr_to_source[int(addr_match.group(1), 16)] = current_source
    addrs = sorted(addr_to_source)
    return addrs, addr_to_source


def source_for_offset(
    offset: int,
    sorted_addrs: list[int],
    addr_to_source: dict[int, str],
) -> str:
    if offset in addr_to_source:
        return addr_to_source[offset]
    idx = bisect.bisect_right(sorted_addrs, offset) - 1
    if idx < 0:
        return "unknown"
    return addr_to_source[sorted_addrs[idx]]


def repo_source_for(source: str, source_root: Path) -> str | None:
    match = SOURCE_RE.match(source)
    if not match:
        return None
    raw_path = Path(match.group(1))
    line = match.group(2)

    src_match = re.search(r"/src/([^/]+)/([^/]+)$", str(raw_path))
    if src_match:
        op_dir, filename = src_match.groups()
        rel_dirs = (
            Path("csrc") / op_dir / "op_kernel",
            Path("csrc") / op_dir / "op_kernel" / "arch35",
            Path("csrc") / op_dir / "op_host",
            Path("packages/openjiuwen_ascend_ops/csrc") / op_dir / "op_kernel",
            Path("packages/openjiuwen_ascend_ops/csrc") / op_dir / "op_kernel" / "arch35",
            Path("packages/openjiuwen_ascend_ops/csrc") / op_dir / "op_host",
        )
        for rel_dir in rel_dirs:
            candidate = source_root / rel_dir / filename
            if candidate.is_file():
                return f"{candidate}:{line}"

    if raw_path.is_file() and source_root in raw_path.parents:
        return f"{raw_path}:{line}"
    return None


def line_key(source: str, source_root: Path) -> tuple[str, bool]:
    repo_source = repo_source_for(source, source_root)
    if repo_source:
        return repo_source, True
    return source, False


def collect_op(
    op_dir: Path,
    objdump: str,
    source_root: Path,
    timeout_s: int,
    top_bb: int,
    debug_elf: Path | None,
) -> dict[str, Any]:
    dump_dir = op_dir / "dump"
    fdata_path = dump_dir / "fdata"
    elf_path = dump_dir / "aicore_binary.o"
    if not elf_path.is_file():
        elf_path = dump_dir / "kernel_data/aicore_binary.o"
    op_basic_path = find_first(op_dir, ("OpBasicInfo*.csv",))
    pipe_path = find_first(op_dir, ("PipeUtilization*.csv",))

    fdata = parse_fdata(fdata_path)
    bbbmap = parse_bbbmap(dump_dir)
    line_elf_path = debug_elf if debug_elf is not None else elf_path
    objdump_output = run_objdump(objdump, line_elf_path, timeout_s)
    sorted_addrs, addr_to_source = parse_objdump_lines(objdump_output)

    lines: dict[str, dict[str, Any]] = {}
    hot_records: list[dict[str, Any]] = []
    for record in fdata:
        count = record["count"]
        if count <= 0:
            continue
        offset = record["offset"]
        source = source_for_offset(offset, sorted_addrs, addr_to_source)
        display_source, is_repo = line_key(source, source_root)
        item = lines.setdefault(display_source, {
            "source": display_source,
            "raw_sources": set(),
            "is_repo": is_repo,
            "count": 0,
            "basic_blocks": 0,
            "offsets": [],
        })
        item["raw_sources"].add(source)
        item["is_repo"] = item["is_repo"] or is_repo
        item["count"] += count
        item["basic_blocks"] += 1
        item["offsets"].append({
            "offset": offset,
            "count": count,
            "bbb_id": bbbmap.get(offset),
        })
        hot_records.append({
            "offset": offset,
            "count": count,
            "bbb_id": bbbmap.get(offset),
            "source": display_source,
            "raw_source": source,
        })

    for item in lines.values():
        item["raw_sources"] = sorted(item["raw_sources"])
        item["offsets"] = sorted(
            item["offsets"], key=lambda x: x["count"], reverse=True)[:8]

    line_list = sorted(lines.values(), key=lambda x: x["count"], reverse=True)
    hot_records = sorted(hot_records, key=lambda x: x["count"], reverse=True)[:top_bb]
    return {
        "op_dir": str(op_dir),
        "profile_elf": str(elf_path),
        "line_elf": str(line_elf_path),
        "op_basic": read_csv_rows(op_basic_path),
        "pipe": summarize_pipe(read_csv_rows(pipe_path)),
        "fdata_records": len(fdata),
        "source_line_count": len(line_list),
        "top_lines": line_list,
        "top_basic_blocks": hot_records,
    }


def compact_line(line: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": line["source"],
        "count": line["count"],
        "basic_blocks": line["basic_blocks"],
        "offsets": line["offsets"],
    }


def print_op_summary(op: dict[str, Any], top_lines: int, top_repo_lines: int) -> None:
    print(f"\nOP dir: {op['op_dir']}")
    print(f"  profile_elf={op['profile_elf']}")
    print(f"  line_elf={op['line_elf']}")
    if op["op_basic"]:
        for row in op["op_basic"]:
            print(
                "  "
                f"op={row.get('Op Name', 'NA')} "
                f"type={row.get('Op Type', 'NA')} "
                f"duration_us={row.get('Task Duration(us)', 'NA')} "
                f"block_dim={row.get('Block Dim', 'NA')} "
                f"mix_block_dim={row.get('Mix Block Dim', 'NA')}"
            )
    else:
        print("  OpBasicInfo: missing")

    if op["pipe"]:
        print("  PipeUtilization(avg/max us):")
        for sub_block, metrics in op["pipe"].items():
            pieces = [f"{sub_block} rows={metrics['rows']}"]
            for metric in PIPE_METRICS:
                stat = metrics.get(metric)
                if stat:
                    pieces.append(f"{metric}={stat['avg']:.3f}/{stat['max']:.3f}")
            print("    " + " ".join(pieces))

    print(f"  fdata_records={op['fdata_records']} source_lines={op['source_line_count']}")

    print("  Top source lines:")
    for line in op["top_lines"][:top_lines]:
        print(
            "    "
            f"count={line['count']} bb={line['basic_blocks']} {line['source']}"
        )
        for off in line["offsets"][:3]:
            bbb = "NA" if off["bbb_id"] is None else off["bbb_id"]
            print(f"      offset=0x{off['offset']:x} count={off['count']} bbb={bbb}")

    repo_lines = [line for line in op["top_lines"] if line["is_repo"]]
    if repo_lines:
        print("  Top repo source lines:")
        for line in repo_lines[:top_repo_lines]:
            print(
                "    "
                f"count={line['count']} bb={line['basic_blocks']} {line['source']}"
            )
            for off in line["offsets"][:3]:
                bbb = "NA" if off["bbb_id"] is None else off["bbb_id"]
                print(f"      offset=0x{off['offset']:x} count={off['count']} bbb={bbb}")

    print("  Top basic blocks:")
    for row in op["top_basic_blocks"]:
        bbb = "NA" if row["bbb_id"] is None else row["bbb_id"]
        print(
            "    "
            f"count={row['count']} offset=0x{row['offset']:x} "
            f"bbb={bbb} {row['source']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map OPPROF fdata hot basic blocks to source lines."
    )
    parser.add_argument("opprof_dir", help="OPPROF root or nested op profile dir")
    parser.add_argument("--op-filter", help="Only analyze op dirs containing this text")
    parser.add_argument("--top-lines", type=int, default=12)
    parser.add_argument("--top-repo-lines", type=int, default=12)
    parser.add_argument("--top-bb", type=int, default=16)
    parser.add_argument("--source-root", default=str(Path.cwd()))
    parser.add_argument("--llvm-objdump", help="Path to CANN llvm-objdump")
    parser.add_argument(
        "--debug-elf",
        help=(
            "Use this object for addr2line while keeping fdata from OPPROF. "
            "Useful when the profiled dump object is stripped but a same-hash "
            "debug-line build object is available."
        ),
    )
    parser.add_argument("--objdump-timeout", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.opprof_dir).resolve()
    source_root = Path(args.source_root).resolve()
    objdump = find_objdump(args.llvm_objdump)
    debug_elf = Path(args.debug_elf).resolve() if args.debug_elf else None
    if debug_elf is not None and not debug_elf.is_file():
        raise FileNotFoundError(f"--debug-elf does not exist: {debug_elf}")
    op_dirs = discover_op_dirs(root, args.op_filter)
    if not op_dirs:
        raise FileNotFoundError(f"No OPPROF op dirs found under {root}")

    ops = [
        collect_op(
            op_dir,
            objdump,
            source_root,
            args.objdump_timeout,
            args.top_bb,
            debug_elf,
        )
        for op_dir in op_dirs
    ]

    result = {
        "opprof_root": str(root),
        "source_root": str(source_root),
        "llvm_objdump": objdump,
        "ops": ops,
    }

    if args.json:
        serializable = dict(result)
        serializable["ops"] = []
        for op in result["ops"]:
            serializable["ops"].append({
                **op,
                "top_lines": [compact_line(line) for line in op["top_lines"]],
            })
        print(json.dumps(serializable, indent=2, sort_keys=True))
        return 0

    print(f"OPPROF root: {result['opprof_root']}")
    print(f"llvm-objdump: {result['llvm_objdump']}")
    print(f"op dirs: {len(result['ops'])}")
    for op in result["ops"]:
        print_op_summary(op, args.top_lines, args.top_repo_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
