#!/usr/bin/env python3
"""Extract MindStudio table_detail metrics from OPPROF visualize_data.bin."""

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


def iter_table_objects(data: bytes) -> Iterator[dict[str, Any]]:
    offset = 0
    while True:
        start = data.find(ADVICE_PREFIX, offset)
        if start < 0:
            return

        raw, offset = extract_json_object(data, start)
        if raw is None or b'"table_detail"' not in raw:
            continue

        try:
            yield json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue


def row_matches(
    row: dict[str, Any],
    block_ids: set[str],
    table_names: set[str],
    names: set[str],
) -> bool:
    if block_ids and str(row.get("block_id", "")) not in block_ids:
        return False
    if table_names and str(row.get("table_name", "")) not in table_names:
        return False
    if names and str(row.get("name", "")) not in names:
        return False
    return True


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    block_ids = split_filters(args.block_id)
    table_names = split_filters(args.table_name)
    names = split_filters(args.name)
    rows: list[dict[str, Any]] = []

    for input_path in args.opprof_dir:
        visualize_files = discover_visualize_files(Path(input_path).resolve())
        if not visualize_files:
            raise FileNotFoundError(f"No visualize_data.bin found under {input_path}")

        for visualize_file in visualize_files:
            data = visualize_file.read_bytes()
            for obj in iter_table_objects(data):
                block_id = obj.get("block_id", "")
                for table in obj.get("table_detail", []):
                    if not isinstance(table, dict):
                        continue
                    table_name = table.get("table_name", "")
                    headers = [str(header) for header in table.get("header_name", [])[1:]]
                    for item in table.get("row", []):
                        if not isinstance(item, dict):
                            continue
                        values = item.get("value", [])
                        value_map = {
                            header: value
                            for header, value in zip(headers, values, strict=False)
                        }
                        row = {
                            "opprof": visualize_file.parent.name,
                            "visualize_data": str(visualize_file),
                            "block_id": block_id,
                            "table_name": table_name,
                            "name": item.get("name", ""),
                            **value_map,
                        }
                        if row_matches(row, block_ids, table_names, names):
                            rows.append(row)
    return rows


def ordered_headers(rows: list[dict[str, Any]]) -> list[str]:
    base_headers = ["opprof", "block_id", "table_name", "name"]
    extra_headers: list[str] = []
    for row in rows:
        for key in row:
            if key in base_headers or key == "visualize_data":
                continue
            if key not in extra_headers:
                extra_headers.append(key)
    return base_headers + extra_headers


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No matching table_detail records found.")
        return

    headers = ordered_headers(rows)
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    for row in rows:
        print("  ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def print_csv(rows: list[dict[str, Any]]) -> None:
    headers = ordered_headers(rows) + ["visualize_data"]
    writer = csv.DictWriter(sys.stdout, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract MindStudio table_detail metrics from OPPROF visualize_data.bin."
    )
    parser.add_argument("opprof_dir", nargs="+", help="OPPROF dir, parent dir, or visualize_data.bin path")
    parser.add_argument("--block-id", action="append", help="Block id filter; comma-separated values are allowed")
    parser.add_argument("--table-name", action="append", help="Table name filter, for example 'Vector Core1'")
    parser.add_argument("--name", action="append", help="Metric name filter; may be repeated")
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
