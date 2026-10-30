"""Shared fixtures for rq_uniform UT (contract / golden / perf)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[2]
_REPO = _PKG.parents[1]
# Import path for openjiuwentools: set via pytest.ini pythonpath=pysrc (avoid sys.path.insert).

# Prefer local OPP install before importing the wheel (kernel binaries).
_OPP_CANDIDATES = [
    _REPO / "ztmp" / "cann_ops_install" / "vendors" / "vllm-ascend",
    _PKG / "pysrc" / "openjiuwentools" / "openjiuwen_ascend_ops" / "_cann_ops_custom" / "vendors" / "vllm-ascend",
]


def _bootstrap_opp() -> Path | None:
    import ctypes

    for opp in _OPP_CANDIDATES:
        if not opp.is_dir():
            continue
        opapi = opp / "op_api" / "lib" / "libcust_opapi_jw.so"
        if opapi.is_file():
            try:
                ctypes.CDLL(str(opapi), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
        if str(opp) not in existing:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
                str(opp) if not existing else f"{opp}:{existing}"
            )
        return opp
    return None


_bootstrap_opp()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "npu: needs Ascend NPU")
    config.addinivalue_line("markers", "golden: numerical correctness")
    config.addinivalue_line("markers", "perf: latency smoke")


@pytest.fixture(scope="session")
def openjiuwen_ops():
    """Import package (registers torch.ops.openjiuwen.*) or skip."""
    try:
        import openjiuwentools.openjiuwen_ascend_ops  # noqa: F401
        import torch
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"openjiuwen_ascend_ops unavailable: {exc}")
    if not hasattr(torch.ops, "openjiuwen"):
        pytest.skip("torch.ops.openjiuwen not registered")
    return torch.ops.openjiuwen


@pytest.fixture(scope="session")
def npu_device(openjiuwen_ops):
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("torch.npu is not available")
    return torch.device("npu")
