"""Packaging discovery must never pick up setuptools' own build/ staging tree.

Building a wheel creates ``build/lib/jiuwen_vllm_affinity/...`` inside the
project. With an unqualified package discovery the second build sees that
staging directory as a namespace package and ships a duplicate copy of the
plugin (plus a top-level ``build`` package) in the wheel.
"""

import tomllib
from pathlib import Path

import pytest

setuptools = pytest.importorskip("setuptools")
find_namespace_packages = setuptools.find_namespace_packages

PKG_ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("v0.13.0", "v0.17.0", "v0.18.0", "v0.21.0")


def _find_config(version: str) -> dict:
    pyproject = PKG_ROOT / version / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["tool"]["setuptools"]["packages"]["find"]


@pytest.mark.parametrize("version", VERSIONS)
def test_build_staging_tree_is_not_discovered(version, tmp_path):
    cfg = _find_config(version)

    (tmp_path / "jiuwen_vllm_affinity").mkdir()
    (tmp_path / "jiuwen_vllm_affinity" / "__init__.py").write_text("", encoding="utf-8")
    staging = tmp_path / "build" / "lib" / "jiuwen_vllm_affinity"
    staging.mkdir(parents=True)
    (staging / "__init__.py").write_text("", encoding="utf-8")

    found = find_namespace_packages(
        where=str(tmp_path),
        include=cfg.get("include"),
        exclude=cfg.get("exclude", ()),
    )

    assert "jiuwen_vllm_affinity" in found
    assert not [name for name in found if name == "build" or name.startswith("build.")]
