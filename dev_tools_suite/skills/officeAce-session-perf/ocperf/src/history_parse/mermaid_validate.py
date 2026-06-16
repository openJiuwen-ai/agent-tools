"""Validate Mermaid flowchart syntax for generated E2E diagrams."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def sanitize_mermaid_label(text: str, *, max_len: int = 48) -> str:
    """Make label safe inside Mermaid node brackets."""
    s = str(text or "")
    s = s.replace('"', "'").replace("[", "(").replace("]", ")")
    s = re.sub(r"[#;]", "", s)
    s = s.replace("\n", " ").replace("\r", " ")
    return s.strip()[:max_len] or "?"


def validate_mermaid(source: str) -> dict[str, Any]:
    """Static validation; returns {ok, errors, warnings, stats}."""
    errors: list[str] = []
    warnings: list[str] = []
    src = (source or "").strip()
    lines = src.splitlines()

    if not src:
        errors.append("Mermaid 内容为空")
        return _result(errors, warnings)

    header = lines[0].strip() if lines else ""
    if not re.match(r"^flowchart\s+(TD|LR|BT|RL)\b", header, re.I):
        errors.append(f"首行须为 flowchart TD/LR 等，当前: {header[:40]!r}")

    subgraph_n = len(re.findall(r"^\s*subgraph\s+", src, re.M | re.I))
    end_n = len(re.findall(r"^\s*end\s*$", src, re.M | re.I))
    if subgraph_n != end_n:
        errors.append(f"subgraph/end 不匹配: subgraph={subgraph_n}, end={end_n}")

    node_defs = re.findall(r'\b\w+\[["\']', src)
    if not node_defs and "empty[" not in src:
        warnings.append("未检测到节点定义")

    for i, line in enumerate(lines, 1):
        if re.search(r'\w+\[["\']', line) and line.count("[") != line.count("]"):
            errors.append(f"第 {i} 行节点方括号可能未闭合: {line[:80]!r}")

    dup_ids = _duplicate_node_ids(src)
    if dup_ids:
        warnings.append(f"重复节点 ID（可能影响渲染）: {', '.join(dup_ids[:5])}")

    if len(lines) > 500:
        warnings.append(f"行数较多 ({len(lines)})，浏览器渲染可能较慢")

    return _result(errors, warnings, extra={"lines": len(lines), "nodes": len(node_defs)})


def _duplicate_node_ids(src: str) -> list[str]:
    ids = re.findall(r"\b(e\d+)\[", src)
    seen: set[str] = set()
    dups: list[str] = []
    for nid in ids:
        if nid in seen:
            dups.append(nid)
        seen.add(nid)
    return dups


def _result(errors: list[str], warnings: list[str], extra: dict | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
    if extra:
        out["stats"] = extra
    return out


def try_render_mermaid_cli(mmd_path: Path, *, timeout_sec: int = 90) -> dict[str, Any]:
    """Optional: render via mmdc (mermaid-cli) if installed."""
    mmdc = shutil.which("mmdc")
    if not mmdc:
        return {"render_ok": None, "render_note": "mmdc 未安装，仅做静态校验"}

    with tempfile.TemporaryDirectory() as tmp:
        out_svg = Path(tmp) / "probe.svg"
        try:
            proc = subprocess.run(
                [mmdc, "-i", str(mmd_path), "-o", str(out_svg), "-b", "transparent"],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            if proc.returncode == 0 and out_svg.is_file() and out_svg.stat().st_size > 0:
                return {"render_ok": True, "render_note": "mmdc 渲染成功"}
            err = (proc.stderr or proc.stdout or "").strip()[:500]
            return {"render_ok": False, "render_note": f"mmdc 失败: {err}"}
        except subprocess.TimeoutExpired:
            return {"render_ok": False, "render_note": "mmdc 超时"}
        except OSError as e:
            return {"render_ok": False, "render_note": str(e)}


def validate_mermaid_file(mmd_path: Path, *, try_cli: bool = True) -> dict[str, Any]:
    source = mmd_path.read_text(encoding="utf-8")
    report = validate_mermaid(source)
    report["mmd_path"] = str(mmd_path.resolve())
    if try_cli:
        report.update(try_render_mermaid_cli(mmd_path))
    return report


def write_validation_report(out_dir: Path, safe_name: str, report: dict[str, Any]) -> Path:
    import json

    path = out_dir / f"execution_flow_mermaid_{safe_name}_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
