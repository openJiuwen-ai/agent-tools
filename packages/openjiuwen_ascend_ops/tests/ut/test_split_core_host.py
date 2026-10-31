"""CPU host UTs for rewritten SplitCore + kv_quant InferShape/tiling helpers.

No NPU required. Builds and runs:
  - tests/ut/host/test_split_core.cpp        (+ production split_core.cpp)
  - tests/ut/host/test_kv_quant_infershape.cpp (mirrors InferShape control flow)
  - tests/ut/host/test_kv_quant_tiling_math.cpp (mirrors MLA PlanState / CalcBalance)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_UT = Path(__file__).resolve().parent
_HOST_DIR = _UT / "host"
_SPLIT_DIR = _UT.parents[1] / "csrc" / "common_fia_split"
_SPLIT_CPP = _SPLIT_DIR / "split_core.cpp"
_SPLIT_H = _SPLIT_DIR / "split_core.h"
_SPLIT_HOST = _HOST_DIR / "test_split_core.cpp"
_INFER_HOST = _HOST_DIR / "test_kv_quant_infershape.cpp"
_TILING_HOST = _HOST_DIR / "test_kv_quant_tiling_math.cpp"


def _cxx() -> str:
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        pytest.skip("g++/c++ not available")
    return cxx


def _compile_and_run(cxx: str, sources: list[Path], includes: list[Path], out_bin: Path, ok_msg: str):
    cmd = [cxx, "-std=c++17", "-O0", "-g"]
    for inc in includes:
        cmd.append(f"-I{inc}")
    cmd.extend(str(s) for s in sources)
    cmd.extend(["-o", str(out_bin)])
    compile_res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert compile_res.returncode == 0, (
        f"compile failed\ncmd={' '.join(cmd)}\n"
        f"stdout:\n{compile_res.stdout}\nstderr:\n{compile_res.stderr}"
    )
    run_res = subprocess.run([str(out_bin)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0, (
        f"run failed\nstdout:\n{run_res.stdout}\nstderr:\n{run_res.stderr}"
    )
    assert ok_msg in run_res.stdout


@pytest.mark.golden
def test_split_core_host_all_rewritten_methods(tmp_path: Path):
    """Hard host coverage for split_core.cpp rewritten methods."""
    assert _SPLIT_HOST.is_file(), f"missing {_SPLIT_HOST}"
    assert _SPLIT_CPP.is_file() and _SPLIT_H.is_file()
    _compile_and_run(
        _cxx(),
        sources=[_SPLIT_HOST, _SPLIT_CPP],
        includes=[_SPLIT_DIR],
        out_bin=tmp_path / "test_split_core",
        ok_msg="all checks passed",
    )


@pytest.mark.golden
def test_kv_quant_infershape_host_all_rewritten_methods(tmp_path: Path):
    """Hard host coverage for rewritten InferShape helpers (TND/BSND/unknown)."""
    assert _INFER_HOST.is_file(), f"missing {_INFER_HOST}"
    _compile_and_run(
        _cxx(),
        sources=[_INFER_HOST],
        includes=[],
        out_bin=tmp_path / "test_kv_quant_infershape",
        ok_msg="all checks passed",
    )


@pytest.mark.golden
def test_kv_quant_tiling_math_host_planstate_and_balance(tmp_path: Path):
    """Hard host coverage for rewritten MLA PlanState math + CalcBalance + attr gates."""
    assert _TILING_HOST.is_file(), f"missing {_TILING_HOST}"
    _compile_and_run(
        _cxx(),
        sources=[_TILING_HOST],
        includes=[],
        out_bin=tmp_path / "test_kv_quant_tiling_math",
        ok_msg="all checks passed",
    )
