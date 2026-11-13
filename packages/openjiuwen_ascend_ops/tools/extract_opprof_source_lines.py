#!/usr/bin/env python3
"""Extract MindStudio source-line counters from an OPPROF visualize_data.bin.

MindStudio Insight stores the source-view table in ``visualize_data.bin`` as
JSON objects embedded in a binary file. This script extracts the ``Files`` /
``Lines`` records directly, so line-level "Instructions Executed" values can be
checked without opening the UI.

Typical usage:

* Show MindStudio-style source-line counters for one source range:

  ``python packages/openjiuwen_ascend_ops/tools/extract_opprof_source_lines.py OPPROF_DIR \
      --source turboquant_pack_kv_for_cache4bit.cpp \
      --line-start 975 --line-end 1004``

* Summarize system bottlenecks by parsed C/C++ function range:

  ``python packages/openjiuwen_ascend_ops/tools/extract_opprof_source_lines.py OPPROF_DIR \
      --source turboquant_pack_kv_for_cache4bit.cpp \
      --group-by function --top 20``

* Emit machine-readable data for reports or profile comparisons:

  ``python packages/openjiuwen_ascend_ops/tools/extract_opprof_source_lines.py OPPROF_DIR \
      --source turboquant_pack_kv_for_cache4bit.cpp \
      --group-by function --top 20 --format json``

Inputs:

* ``OPPROF_DIR`` can be an OPPROF directory, a parent directory containing
  multiple OPPROF directories, or a direct ``visualize_data.bin`` path.
* ``--source`` accepts a basename or path suffix. For generated paths such as
  ``.../src/<op>/<file>.cpp``, function grouping resolves the matching repo
  source under ``csrc/<op>/op_kernel`` or ``csrc/<op>/op_host``. Override the
  repository root with ``--source-root`` when running from another directory.

Output fields:

* ``Instructions Executed`` comes directly from MindStudio source-view data.
* ``Profile%`` is a function's instruction count divided by all extracted
  source-line instructions in the profile.
* ``Selected%`` is divided only by the lines selected by ``--source`` and
  optional line range filters.
* ``Est(us)`` is ``OpBasicInfo.csv`` task duration multiplied by ``Profile%``.
  This is an instruction-share estimate for bottleneck ranking, not a hardware
  measured per-function wall time.
* ``Hot Lines`` lists the highest-instruction source lines within each
  function as ``line:instructions``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


JSON_PREFIX = b'{"Cores":'


@dataclass
class LineStat:
    source: str
    line: int
    instructions: int = 0
    process_bytes: int | float | None = None
    l2_hit_rate: int | float | None = None
    record_count: int = 0
    ranges: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class FunctionRange:
    name: str
    start_line: int
    end_line: int
    source_path: Path | None


@dataclass
class FunctionStat:
    visualize_file: Path
    source: str
    function: str
    start_line: int | None
    end_line: int | None
    instructions: int = 0
    record_count: int = 0
    lines: list[LineStat] = field(default_factory=list)


def discover_visualize_files(root: Path) -> list[Path]:
    if root.is_file() and root.name == "visualize_data.bin":
        return [root]
    if (root / "visualize_data.bin").is_file():
        return [root / "visualize_data.bin"]
    return sorted(root.rglob("visualize_data.bin"))


def extract_json_object(data: bytes, start: int) -> tuple[bytes | None, int]:
    """Return the JSON object starting at ``start`` and the next scan offset."""
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(data)):
        byte = data[index]
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue

        if byte == ord('"'):
            in_string = True
        elif byte == ord("{"):
            depth += 1
        elif byte == ord("}"):
            depth -= 1
            if depth == 0:
                return data[start : index + 1], index + 1

    return None, len(data)


def iter_mindstudio_json(data: bytes) -> Iterator[tuple[int, dict[str, Any]]]:
    offset = 0
    while True:
        start = data.find(JSON_PREFIX, offset)
        if start < 0:
            return

        raw, offset = extract_json_object(data, start)
        if raw is None or b'"Files"' not in raw:
            continue

        try:
            yield start, json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue


def source_matches(source: str, source_filter: str | None) -> bool:
    if not source_filter:
        return True

    normalized_source = source.replace("\\", "/")
    normalized_filter = source_filter.replace("\\", "/")
    if normalized_source.endswith(normalized_filter):
        return True
    return Path(normalized_source).name == normalized_filter


def line_in_range(line: int, line_start: int | None, line_end: int | None) -> bool:
    if line_start is not None and line < line_start:
        return False
    if line_end is not None and line > line_end:
        return False
    return True


def sum_counter(value: Any) -> int:
    if isinstance(value, list):
        return sum(sum_counter(item) for item in value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_line_stats(
    visualize_file: Path,
) -> dict[tuple[str, int], LineStat]:
    data = visualize_file.read_bytes()
    stats: dict[tuple[str, int], LineStat] = {}

    for _offset, obj in iter_mindstudio_json(data):
        for file_obj in obj.get("Files", []):
            source = str(file_obj.get("Source", ""))
            for line_obj in file_obj.get("Lines", []):
                try:
                    line = int(line_obj["Line"])
                except (KeyError, TypeError, ValueError):
                    continue

                key = (source, line)
                stat = stats.setdefault(key, LineStat(source=source, line=line))
                stat.instructions += sum_counter(line_obj.get("Instructions Executed"))
                stat.process_bytes = line_obj.get("Process Bytes", stat.process_bytes)
                stat.l2_hit_rate = line_obj.get("L2Cache Hit Rate", stat.l2_hit_rate)
                stat.record_count += 1
                for address_range in line_obj.get("Address Range", []):
                    if (
                        isinstance(address_range, list)
                        and len(address_range) == 2
                        and all(isinstance(item, str) for item in address_range)
                    ):
                        stat.ranges.append(address_range)

    return stats


def filter_line_stats(
    stats: dict[tuple[str, int], LineStat],
    source_filter: str | None,
    line_start: int | None,
    line_end: int | None,
) -> list[LineStat]:
    return [
        stat
        for stat in stats.values()
        if source_matches(stat.source, source_filter)
        and line_in_range(stat.line, line_start, line_end)
    ]


def source_without_line(source: str) -> str:
    match = re.match(r"^(.+):\d+$", source)
    return match.group(1) if match else source


def resolve_source_path(source: str, source_root: Path) -> Path | None:
    raw_path = Path(source_without_line(source))
    if raw_path.is_file():
        return raw_path

    normalized = str(raw_path).replace("\\", "/")
    src_match = re.search(r"/src/([^/]+)/([^/]+)$", normalized)
    if src_match:
        op_dir, filename = src_match.groups()
        candidates = [
            source_root / "csrc" / op_dir / "op_kernel" / filename,
            source_root / "csrc" / op_dir / "op_kernel" / "arch35" / filename,
            source_root / "csrc" / op_dir / "op_host" / filename,
            source_root / "packages/openjiuwen_ascend_ops/csrc" / op_dir / "op_kernel" / filename,
            source_root
            / "packages/openjiuwen_ascend_ops/csrc"
            / op_dir
            / "op_kernel"
            / "arch35"
            / filename,
            source_root / "packages/openjiuwen_ascend_ops/csrc" / op_dir / "op_host" / filename,
            source_root
            / "tools/_turboquant4bit_cann_ops_custom/vendors/vllm-ascend/op_impl"
            / "ai_core/tbe/vllm-ascend_impl/ascendc"
            / op_dir
            / filename,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def strip_comments(line: str, in_block_comment: bool) -> tuple[str, bool]:
    out: list[str] = []
    i = 0
    while i < len(line):
        if in_block_comment:
            end = line.find("*/", i)
            if end < 0:
                return "".join(out), True
            i = end + 2
            in_block_comment = False
            continue

        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            in_block_comment = True
            i += 2
            continue

        out.append(line[i])
        i += 1
    return "".join(out), in_block_comment


def remove_string_literals(line: str) -> str:
    out: list[str] = []
    quote: str | None = None
    escaped = False
    for char in line:
        if quote is not None:
            out.append(" ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ('"', "'"):
            quote = char
            out.append(" ")
        else:
            out.append(char)
    return "".join(out)


CONTROL_KEYWORDS = {
    "catch",
    "do",
    "else",
    "for",
    "if",
    "switch",
    "while",
}


def extract_function_name(signature: str) -> str | None:
    compact = " ".join(signature.replace("\n", " ").split())
    open_idx = compact.rfind("(")
    if open_idx < 0:
        return None

    prefix = compact[:open_idx].rstrip()
    match = re.search(r"([A-Za-z_~][A-Za-z0-9_:~]*)\s*$", prefix)
    if not match:
        return None

    name = match.group(1)
    leaf_name = name.rsplit("::", 1)[-1]
    if leaf_name in CONTROL_KEYWORDS:
        return None
    return name


def is_function_signature(signature: str) -> bool:
    compact = " ".join(signature.split())
    if "(" not in compact or ")" not in compact:
        return False
    if compact.startswith(("return ", "using ", "typedef ")):
        return False
    return extract_function_name(compact) is not None


def parse_function_ranges(source_path: Path | None) -> list[FunctionRange]:
    if source_path is None or not source_path.is_file():
        return []

    lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ranges: list[FunctionRange] = []
    pending_lines: list[str] = []
    pending_start: int | None = None
    current_name: str | None = None
    current_start = 0
    current_target_depth = 0
    depth = 0
    in_block_comment = False

    for line_no, raw_line in enumerate(lines, start=1):
        code, in_block_comment = strip_comments(raw_line, in_block_comment)
        code = remove_string_literals(code)
        stripped = code.strip()
        opens = code.count("{")
        closes = code.count("}")
        depth_before = depth

        if current_name is not None:
            depth += opens - closes
            if depth <= current_target_depth:
                ranges.append(FunctionRange(
                    name=current_name,
                    start_line=current_start,
                    end_line=line_no,
                    source_path=source_path,
                ))
                current_name = None
            continue

        if stripped.startswith("}"):
            pending_lines.clear()
            pending_start = None

        if stripped and not stripped.startswith("#"):
            if pending_start is None:
                pending_start = line_no
            pending_lines.append(stripped)

        if opens:
            signature = " ".join(pending_lines).split("{", 1)[0]
            function_name = extract_function_name(signature)
            if function_name is not None and is_function_signature(signature):
                current_name = function_name
                current_start = pending_start or line_no
                current_target_depth = depth_before
            pending_lines.clear()
            pending_start = None
        elif ";" in stripped or stripped.endswith(":"):
            pending_lines.clear()
            pending_start = None

        depth += opens - closes
        if current_name is not None and depth <= current_target_depth:
            ranges.append(FunctionRange(
                name=current_name,
                start_line=current_start,
                end_line=line_no,
                source_path=source_path,
            ))
            current_name = None

    return ranges


def function_for_line(ranges: list[FunctionRange], line: int) -> FunctionRange | None:
    matches = [item for item in ranges if item.start_line <= line <= item.end_line]
    if not matches:
        return None
    return max(matches, key=lambda item: item.start_line)


def read_task_duration_us(visualize_file: Path) -> float | None:
    op_basic_info = visualize_file.parent / "OpBasicInfo.csv"
    if not op_basic_info.is_file():
        return None

    with op_basic_info.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get("Task Duration(us)")
            if value in (None, "", "NA"):
                return None
            try:
                return float(value)
            except ValueError:
                return None
    return None


def build_function_stats(
    visualize_file: Path,
    selected_stats: list[LineStat],
    source_root: Path,
) -> list[FunctionStat]:
    ranges_by_source: dict[str, list[FunctionRange]] = {}
    function_stats: dict[tuple[str, str, int | None, int | None], FunctionStat] = {}

    for stat in selected_stats:
        if stat.source not in ranges_by_source:
            ranges_by_source[stat.source] = parse_function_ranges(
                resolve_source_path(stat.source, source_root)
            )
        function_range = function_for_line(ranges_by_source[stat.source], stat.line)
        function = function_range.name if function_range else "<unknown>"
        start_line = function_range.start_line if function_range else None
        end_line = function_range.end_line if function_range else None
        key = (stat.source, function, start_line, end_line)
        function_stat = function_stats.setdefault(
            key,
            FunctionStat(
                visualize_file=visualize_file,
                source=stat.source,
                function=function,
                start_line=start_line,
                end_line=end_line,
            ),
        )
        function_stat.instructions += stat.instructions
        function_stat.record_count += stat.record_count
        function_stat.lines.append(stat)

    return sorted(function_stats.values(), key=lambda item: item.instructions, reverse=True)


def stat_to_json(stat: LineStat, visualize_file: Path) -> dict[str, Any]:
    return {
        "visualize_data": str(visualize_file),
        "source": stat.source,
        "line": stat.line,
        "instructions_executed": stat.instructions,
        "record_count": stat.record_count,
        "address_ranges": stat.ranges,
        "l2_cache_hit_rate": stat.l2_hit_rate,
        "process_bytes": stat.process_bytes,
    }


def function_stat_to_json(
    stat: FunctionStat,
    profile_total_instructions: int,
    selected_total_instructions: int,
    task_duration_us: float | None,
    hot_lines: int,
) -> dict[str, Any]:
    profile_fraction = (
        stat.instructions / profile_total_instructions
        if profile_total_instructions > 0
        else 0.0
    )
    selected_fraction = (
        stat.instructions / selected_total_instructions
        if selected_total_instructions > 0
        else 0.0
    )
    return {
        "visualize_data": str(stat.visualize_file),
        "source": stat.source,
        "function": stat.function,
        "start_line": stat.start_line,
        "end_line": stat.end_line,
        "instructions_executed": stat.instructions,
        "record_count": stat.record_count,
        "profile_percent": profile_fraction * 100.0,
        "selected_percent": selected_fraction * 100.0,
        "estimated_time_us": (
            task_duration_us * profile_fraction
            if task_duration_us is not None
            else None
        ),
        "hot_lines": [
            {
                "line": line.line,
                "instructions_executed": line.instructions,
                "record_count": line.record_count,
            }
            for line in sorted(
                stat.lines,
                key=lambda item: item.instructions,
                reverse=True,
            )[:hot_lines]
        ],
    }


def print_table(
    all_stats: list[tuple[Path, LineStat]],
    show_ranges: bool,
) -> None:
    if not all_stats:
        print("No matching source-line records found.")
        return

    current_group: tuple[Path, str] | None = None
    for visualize_file, stat in sorted(
        all_stats,
        key=lambda item: (str(item[0]), item[1].source, item[1].line),
    ):
        group = (visualize_file, stat.source)
        if group != current_group:
            if current_group is not None:
                print()
            print(f"visualize_data: {visualize_file}")
            print(f"Source: {stat.source}")
            if show_ranges:
                print("Line  Instructions Executed  Records  Address Ranges")
            else:
                print("Line  Instructions Executed  Records  Range Count")
            current_group = group

        if show_ranges:
            ranges = ",".join(f"{start}-{end}" for start, end in stat.ranges)
            print(f"{stat.line:<5} {stat.instructions:<22} {stat.record_count:<7} {ranges}")
        else:
            print(f"{stat.line:<5} {stat.instructions:<22} {stat.record_count:<7} {len(stat.ranges)}")


def print_function_table(
    function_stats_by_file: list[
        tuple[Path, list[FunctionStat], int, int, float | None]
    ],
    top: int,
    hot_lines: int,
) -> None:
    printed_any = False
    for visualize_file, function_stats, profile_total, selected_total, task_us in function_stats_by_file:
        if not function_stats:
            continue

        printed_any = True
        print(f"visualize_data: {visualize_file}")
        if task_us is not None:
            print(f"Task Duration(us): {task_us:.3f}")
        print(f"Profile source instructions: {profile_total}")
        print(f"Selected source instructions: {selected_total}")
        print(
            "Profile%  Selected%  Est(us)    Instructions Executed  "
            "Records  Lines       Function  Hot Lines"
        )

        rows = function_stats[:top] if top > 0 else function_stats
        for stat in rows:
            profile_percent = (
                stat.instructions * 100.0 / profile_total
                if profile_total > 0
                else 0.0
            )
            selected_percent = (
                stat.instructions * 100.0 / selected_total
                if selected_total > 0
                else 0.0
            )
            estimated_us = (
                task_us * profile_percent / 100.0
                if task_us is not None
                else None
            )
            line_range = (
                f"{stat.start_line}-{stat.end_line}"
                if stat.start_line is not None and stat.end_line is not None
                else "unknown"
            )
            hot_line_text = ", ".join(
                f"{line.line}:{line.instructions}"
                for line in sorted(
                    stat.lines,
                    key=lambda item: item.instructions,
                    reverse=True,
                )[:hot_lines]
            )
            est_text = f"{estimated_us:.3f}" if estimated_us is not None else "NA"
            print(
                f"{profile_percent:>7.2f}  {selected_percent:>9.2f}  "
                f"{est_text:>9}  {stat.instructions:<22} "
                f"{stat.record_count:<7} {line_range:<11} "
                f"{stat.function:<36} {hot_line_text}"
            )
        print()

    if not printed_any:
        print("No matching source-line records found.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract MindStudio source-line stats from OPPROF visualize_data.bin."
    )
    parser.add_argument("opprof_dir", help="OPPROF dir, parent dir, or visualize_data.bin path")
    parser.add_argument(
        "--source",
        help="Source basename/suffix to match, for example turboquant_pack_kv_for_cache4bit.cpp",
    )
    parser.add_argument("--line-start", type=int, help="First source line to include")
    parser.add_argument("--line-end", type=int, help="Last source line to include")
    parser.add_argument(
        "--group-by",
        choices=("line", "function"),
        default="line",
        help="Aggregate by source line or by parsed C/C++ function range.",
    )
    parser.add_argument(
        "--source-root",
        default=str(Path.cwd()),
        help="Repository root used to resolve generated /src/<op>/<file> paths.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Limit function rows; 0 prints all rows.",
    )
    parser.add_argument(
        "--hot-lines",
        type=int,
        default=3,
        help="Number of hottest source lines to show per function.",
    )
    parser.add_argument(
        "--show-ranges",
        action="store_true",
        help="Print address ranges instead of only the range count.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format.",
    )
    args = parser.parse_args()

    visualize_files = discover_visualize_files(Path(args.opprof_dir).resolve())
    if not visualize_files:
        raise FileNotFoundError(f"No visualize_data.bin found under {args.opprof_dir}")

    all_stats: list[tuple[Path, LineStat]] = []
    function_stats_by_file: list[
        tuple[Path, list[FunctionStat], int, int, float | None]
    ] = []
    source_root = Path(args.source_root).resolve()

    for visualize_file in visualize_files:
        stats = parse_line_stats(visualize_file)
        selected_stats = filter_line_stats(
            stats,
            source_filter=args.source,
            line_start=args.line_start,
            line_end=args.line_end,
        )
        all_stats.extend((visualize_file, stat) for stat in selected_stats)
        if args.group_by == "function":
            profile_total = sum(stat.instructions for stat in stats.values())
            selected_total = sum(stat.instructions for stat in selected_stats)
            task_us = read_task_duration_us(visualize_file)
            function_stats = build_function_stats(
                visualize_file,
                selected_stats,
                source_root,
            )
            function_stats_by_file.append((
                visualize_file,
                function_stats,
                profile_total,
                selected_total,
                task_us,
            ))
    all_stats.sort(key=lambda item: (str(item[0]), item[1].source, item[1].line))

    if args.format == "json":
        rows: list[dict[str, Any]]
        if args.group_by == "function":
            rows = []
            for _visualize_file, function_stats, profile_total, selected_total, task_us in function_stats_by_file:
                selected_functions = (
                    function_stats[:args.top] if args.top > 0 else function_stats
                )
                rows.extend(
                    function_stat_to_json(
                        stat,
                        profile_total,
                        selected_total,
                        task_us,
                        args.hot_lines,
                    )
                    for stat in selected_functions
                )
        else:
            rows = [
                stat_to_json(stat, visualize_file)
                for visualize_file, stat in all_stats
            ]
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.group_by == "function":
        print_function_table(
            function_stats_by_file,
            top=args.top,
            hot_lines=args.hot_lines,
        )
    else:
        print_table(all_stats, show_ranges=args.show_ranges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
