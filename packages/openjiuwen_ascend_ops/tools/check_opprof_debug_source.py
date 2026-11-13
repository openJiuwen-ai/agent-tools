#!/usr/bin/env python3
"""Check whether an OPPROF directory carries source-line debug data.

MindStudio Insight can show source for a profiled custom op only when the
profile dump contains a device kernel object with debug line information. The
host runner and libcust_opapi.so are not enough: AscendC kernel source lines
live in the per-op kernel object under ``dump/aicore_binary.o`` or in the
runtime OPP kernel ``*.o``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


SOURCE_HINTS = (
    b"op_kernel",
    b"pack_k8v4_aiv.h",
    b"rq_uniform_pack_k8v4",
    b"csrc/turboquant",
    b"/src/turboquant",
    b"turboquant_pack_kv_for_cache4bit.cpp",
    b"turboquant_attention_paged4bit.cpp",
    b"decode_device.h",
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def discover_op_dirs(root: Path) -> list[Path]:
    if (root / "dump/fdata").is_file():
        return [root]
    return sorted({path.parent.parent for path in root.rglob("dump/fdata")})


def op_name(op_dir: Path) -> str:
    text = read_text(op_dir / "dump/op_basic_info.txt")
    match = re.search(r"^Op Name=(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    for csv_path in sorted(op_dir.glob("OpBasicInfo*.csv")):
        lines = read_text(csv_path).splitlines()
        if len(lines) >= 2:
            return lines[1].split(",", 1)[0].strip()
    return ""


def elf_sections(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result = subprocess.run(
        ["readelf", "-SW", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    sections: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.search(r"\]\s+(\.\S+)", line)
        if match:
            sections.add(match.group(1))
    return sections


def has_source_hint(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as f:
        tail = b""
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                return False
            data = tail + chunk
            if any(hint in data for hint in SOURCE_HINTS):
                return True
            tail = data[-256:]


def candidate_runtime_debug_elf(root: Path, op: str, soc_dir: str) -> Path | None:
    hash_name = op
    for suffix in (
        "_0_mix_aic",
        "_1_mix_aic",
        "_mix_aic",
        "_0_mix_aiv",
        "_1_mix_aiv",
        "_mix_aiv",
    ):
        hash_name = hash_name.removesuffix(suffix)
    if not hash_name:
        return None

    if hash_name.startswith("TurboquantPackKvForCache4bit_"):
        subdir = "turboquant_pack_kv_for_cache4bit"
    elif hash_name.startswith("TurboquantAttentionPaged4bit_"):
        subdir = "turboquant_attention_paged4bit"
    elif hash_name.startswith("RqUniformPackK8v4_"):
        subdir = "rq_uniform_pack_k8v4"
    else:
        return None

    candidates = [
        root
        / "ztmp/cann_ops_install/vendors/vllm-ascend/op_impl/ai_core/tbe/kernel"
        / soc_dir
        / subdir
        / f"{hash_name}.o",
        root
        / "packages/openjiuwen_ascend_ops/pysrc/openjiuwentools/openjiuwen_ascend_ops"
        / "_cann_ops_custom/vendors/vllm-ascend/op_impl/ai_core/tbe/kernel"
        / soc_dir
        / subdir
        / f"{hash_name}.o",
        root
        / "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_impl/ai_core/tbe/kernel"
        / soc_dir
        / subdir
        / f"{hash_name}.o",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report whether OPPROF carries kernel debug source data."
    )
    parser.add_argument("opprof_dir", help="OPPROF root or a parent containing OPPROF dirs")
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--soc-dir", default="ascend910b")
    parser.add_argument(
        "--require-insight-source",
        action="store_true",
        help="Exit non-zero unless each op dir has debug line data and source hints in visualize_data.bin.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    opprof_root = Path(args.opprof_dir).resolve()
    op_dirs = discover_op_dirs(opprof_root)
    if not op_dirs:
        raise FileNotFoundError(f"No OPPROF dump/fdata found under {opprof_root}")

    all_ok = True
    for op_dir in op_dirs:
        dump_elf = op_dir / "dump/aicore_binary.o"
        sections = elf_sections(dump_elf)
        has_debug_line = ".debug_line" in sections
        has_debug_info = ".debug_info" in sections
        visualize_has_source = has_source_hint(op_dir / "visualize_data.bin")
        dump_has_source = has_source_hint(dump_elf)
        op = op_name(op_dir)
        external = candidate_runtime_debug_elf(root, op, args.soc_dir)

        print(f"OP dir: {op_dir}")
        print(f"  op={op or 'unknown'}")
        print(f"  dump_elf={dump_elf}")
        print(f"  dump_elf_size={dump_elf.stat().st_size if dump_elf.is_file() else 0}")
        print(f"  dump_has_debug_line={has_debug_line}")
        print(f"  dump_has_debug_info={has_debug_info}")
        print(f"  dump_has_source_hint={dump_has_source}")
        print(f"  visualize_has_source_hint={visualize_has_source}")
        if has_debug_line and visualize_has_source:
            status = "ok"
        elif has_debug_line:
            status = "likely_ok_no_visualize_hint"
        else:
            status = "missing_dump_debug_line"
        print(f"  insight_source_status={status}")
        all_ok = all_ok and status == "ok"
        if external is not None:
            external_sections = elf_sections(external)
            print(f"  external_debug_elf={external}")
            print(f"  external_has_debug_line={'.debug_line' in external_sections}")
            print(
                "  hotspot_cmd=python packages/openjiuwen_ascend_ops/tools/msopprof_hotspot_lines.py "
                f"{op_dir} --debug-elf {external}"
            )
        print()
    if args.require_insight_source and not all_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
