#!/usr/bin/env python3
"""Extract MindStudio subblock_detail counters from OPPROF visualize_data.bin."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from extract_opprof_source_lines import discover_visualize_files, extract_json_object


ADVICE_PREFIX = b'{"advice":'


def split_filters(values: list[str] | None) -> set[str]:
    filters: set[str] = set()
    for value in values or []:
        filters.update(item.strip() for item in value.split(",") if item.strip())
    return filters


def iter_subblock_objects(data: bytes) -> Iterator[tuple[int, dict[str, Any]]]:
    offset = 0
    detail_index = 0
    while True:
        start = data.find(ADVICE_PREFIX, offset)
        if start < 0:
            return

        raw, offset = extract_json_object(data, start)
        if raw is None or b'"subblock_detail"' not in raw:
            continue

        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

        yield detail_index, obj
        detail_index += 1


def row_matches(
    row: dict[str, Any],
    block_ids: set[str],
    block_types: set[str],
    names: set[str],
    units: set[str],
) -> bool:
    if block_ids and str(row.get("block_id", "")) not in block_ids:
        return False
    if block_types and str(row.get("block_type", "")) not in block_types:
        return False
    if names and str(row.get("name", "")) not in names:
        return False
    if units and str(row.get("unit", "")) not in units:
        return False
    return True


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    block_ids = split_filters(args.block_id)
    block_types = split_filters(args.block_type)
    names = split_filters(args.name)
    units = split_filters(args.unit)
    rows: list[dict[str, Any]] = []

    for input_path in args.opprof_dir:
        visualize_files = discover_visualize_files(Path(input_path).resolve())
        if not visualize_files:
            raise FileNotFoundError(f"No visualize_data.bin found under {input_path}")

        for visualize_file in visualize_files:
            data = visualize_file.read_bytes()
            for detail_index, obj in iter_subblock_objects(data):
                for item in obj.get("subblock_detail", []):
                    if not isinstance(item, dict):
                        continue
                    if not row_matches(item, block_ids, block_types, names, units):
                        continue
                    rows.append({
                        "opprof": visualize_file.parent.name,
                        "visualize_data": str(visualize_file),
                        "detail_index": detail_index,
                        "block_id": item.get("block_id", ""),
                        "block_type": item.get("block_type", ""),
                        "name": item.get("name", ""),
                        "unit": item.get("unit", ""),
                        "value": item.get("value", ""),
                        "origin_value": item.get("origin_value", ""),
                    })
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No matching subblock_detail records found.")
        return

    headers = ["opprof", "detail_index", "block_id", "block_type", "name", "unit", "value", "origin_value"]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def print_csv(rows: list[dict[str, Any]]) -> None:
    headers = ["opprof", "detail_index", "block_id", "block_type", "name", "unit", "value", "origin_value", "visualize_data"]
    writer = csv.DictWriter(sys.stdout, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract MindStudio subblock_detail metrics from OPPROF visualize_data.bin."
    )
    parser.add_argument("opprof_dir", nargs="+", help="OPPROF dir, parent dir, or visualize_data.bin path")
    parser.add_argument("--block-id", action="append", help="Block id filter; comma-separated values are allowed")
    parser.add_argument("--block-type", action="append", help="Block type filter, for example cube0,vector1")
    parser.add_argument("--name", action="append", help="Metric name filter; may be repeated")
    parser.add_argument("--unit", action="append", help="Unit id filter; comma-separated values are allowed")
    parser.add_argument("--format", choices=("table", "csv", "json"), default="table")
    args = parser.parse_args()

    rows = collect_rows(args)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.format == "csv":
        print_csv(rows)
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
