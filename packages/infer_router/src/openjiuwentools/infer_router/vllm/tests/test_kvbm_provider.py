"""Tests for automatic KVBM worker provider selection."""

from __future__ import annotations

import argparse
from importlib import metadata
import sys
import types

import pytest

from openjiuwentools.infer_router.vllm.kvbm_provider import (
    KVBMProviderSpec,
    ResolvedKVBMProvider,
    normalize_kvbm_provider,
    resolve_kvbm_worker_provider,
)


def _versions(**versions: str):
    def _read(package: str) -> str:
        try:
            return versions[package.replace("-", "_")]
        except KeyError as exc:
            raise metadata.PackageNotFoundError(package) from exc

    return _read


@pytest.mark.parametrize("requested", ["", "auto"])
def test_auto_detects_registered_vllm_ascend_018_provider(requested: str) -> None:
    resolved = resolve_kvbm_worker_provider(
        requested,
        version_reader=_versions(vllm_ascend="0.18.0", vllm="0.18.2"),
    )

    assert resolved.name == "vllm-ascend-0.18"
    assert resolved.spec.multiprocess_installer == (
        "install_vllm_ascend_018_multiprocess_hooks"
    )
    assert resolved.spec.engine_core_hook is True
    assert resolved.spec.event_relay is True


def test_auto_accepts_release_candidate_version_suffixes() -> None:
    resolved = resolve_kvbm_worker_provider(
        "auto",
        version_reader=_versions(
            vllm_ascend="0.18.0rc2+ascend",
            vllm="0.18.0rc1",
        ),
    )

    assert resolved.name == "vllm-ascend-0.18"


def test_explicit_provider_alias_is_normalized_and_validated() -> None:
    resolved = resolve_kvbm_worker_provider(
        "vllm_ascend_018",
        version_reader=_versions(vllm_ascend="0.18.1", vllm="0.18.0"),
    )

    assert normalize_kvbm_provider("vllm_ascend_018") == "vllm-ascend-0.18"
    assert resolved.requested == "vllm-ascend-0.18"


def test_rejects_mismatched_engine_release_lines() -> None:
    with pytest.raises(RuntimeError, match="different major/minor release lines"):
        resolve_kvbm_worker_provider(
            "auto",
            version_reader=_versions(vllm_ascend="0.23.0", vllm="0.18.0"),
        )


def test_rejects_explicit_provider_that_differs_from_installed_version() -> None:
    with pytest.raises(RuntimeError, match="does not match installed"):
        resolve_kvbm_worker_provider(
            "vllm-ascend-0.18",
            version_reader=_versions(vllm_ascend="0.23.0", vllm="0.23.1"),
        )


def test_auto_detects_registered_vllm_ascend_023_provider() -> None:
    resolved = resolve_kvbm_worker_provider(
        "auto",
        version_reader=_versions(vllm_ascend="0.23.0", vllm="0.23.1"),
    )

    assert resolved.name == "vllm-ascend-0.23"
    assert resolved.spec.multiprocess_installer == (
        "install_vllm_ascend_023_multiprocess_hooks"
    )


def test_missing_distribution_has_clear_error() -> None:
    with pytest.raises(RuntimeError, match="installed package 'vllm-ascend'"):
        resolve_kvbm_worker_provider("auto", version_reader=_versions(vllm="0.18.0"))


def test_unknown_provider_name_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="unknown KVBM worker provider"):
        normalize_kvbm_provider("vllm-ascend-latest")


def test_worker_startup_dispatches_detected_023_installer(monkeypatch) -> None:
    from jiuwen_kvbm import integrations
    from openjiuwentools.infer_router.vllm import kvbm_provider

    sys.modules.setdefault("uvloop", types.ModuleType("uvloop"))
    from openjiuwentools.infer_router.vllm.main import (
        _install_kvbm_engine_core_hook,
    )

    calls = []
    spec = KVBMProviderSpec(
        name="vllm-ascend-0.23",
        version_line=(0, 23),
        multiprocess_installer="install_vllm_ascend_023_multiprocess_hooks",
        engine_core_hook=True,
        event_relay=True,
    )
    resolved = ResolvedKVBMProvider(
        requested="auto",
        spec=spec,
        vllm_ascend_version="0.23.0",
        vllm_version="0.23.0",
    )
    monkeypatch.setattr(
        kvbm_provider,
        "resolve_kvbm_worker_provider",
        lambda requested: resolved,
    )
    monkeypatch.setattr(
        integrations,
        "install_vllm_ascend_023_multiprocess_hooks",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    args = argparse.Namespace(
        kvbm_worker_enabled=True,
        kvbm_worker_provider="auto",
        port=8001,
    )

    _install_kvbm_engine_core_hook(args)

    assert args.kvbm_worker_provider == "vllm-ascend-0.23"
    assert len(calls) == 1
    assert calls[0]["prefetch_port"] == 18001
