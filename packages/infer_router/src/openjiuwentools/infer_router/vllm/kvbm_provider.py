"""KVBM worker provider version detection and adapter registry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import metadata
from typing import Callable


AUTO_PROVIDER = "auto"


@dataclass(frozen=True)
class KVBMProviderSpec:
    """A registered worker integration for one vLLM-Ascend release line."""

    name: str
    version_line: tuple[int, int]
    multiprocess_installer: str
    engine_core_hook: bool
    event_relay: bool


@dataclass(frozen=True)
class ResolvedKVBMProvider:
    """Provider selected from configuration and installed package metadata."""

    requested: str
    spec: KVBMProviderSpec
    vllm_ascend_version: str
    vllm_version: str

    @property
    def name(self) -> str:
        return self.spec.name


_PROVIDER_SPECS = {
    "vllm-ascend-0.18": KVBMProviderSpec(
        name="vllm-ascend-0.18",
        version_line=(0, 18),
        multiprocess_installer="install_vllm_ascend_018_multiprocess_hooks",
        engine_core_hook=True,
        event_relay=True,
    ),
    "vllm-ascend-0.23": KVBMProviderSpec(
        name="vllm-ascend-0.23",
        version_line=(0, 23),
        multiprocess_installer="install_vllm_ascend_023_multiprocess_hooks",
        engine_core_hook=True,
        event_relay=True,
    ),
}

# Known release lines are kept separate from registered adapters so detection can
# report an unsupported version precisely instead of silently selecting an old hook.
_KNOWN_PROVIDERS = {
    (0, 18): "vllm-ascend-0.18",
    (0, 23): "vllm-ascend-0.23",
}

_PROVIDER_ALIASES = {
    "": AUTO_PROVIDER,
    AUTO_PROVIDER: AUTO_PROVIDER,
    "vllm-ascend-0.18": "vllm-ascend-0.18",
    "vllm_ascend_018": "vllm-ascend-0.18",
    "vllm-ascend018": "vllm-ascend-0.18",
    "vllm-ascend-0.23": "vllm-ascend-0.23",
    "vllm_ascend_023": "vllm-ascend-0.23",
    "vllm-ascend023": "vllm-ascend-0.23",
}


def normalize_kvbm_provider(value: object) -> str:
    """Normalize a configured provider name without importing engine packages."""

    provider = str(value or "").strip().lower()
    normalized = _PROVIDER_ALIASES.get(provider)
    if normalized is None:
        supported = ", ".join(sorted({AUTO_PROVIDER, *_KNOWN_PROVIDERS.values()}))
        raise RuntimeError(
            f"unknown KVBM worker provider {provider!r}; expected one of: {supported}"
        )
    return normalized


def get_kvbm_provider_spec(value: object) -> KVBMProviderSpec | None:
    """Return the registered adapter spec for an already resolved provider."""

    provider = normalize_kvbm_provider(value)
    if provider == AUTO_PROVIDER:
        return None
    return _PROVIDER_SPECS.get(provider)


def _version_line(package: str, version: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(
            f"cannot parse installed {package} version {version!r}; expected MAJOR.MINOR"
        )
    return int(match.group(1)), int(match.group(2))


def _read_version(
    package: str,
    version_reader: Callable[[str], str],
) -> str:
    try:
        return version_reader(package)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"KVBM worker provider detection requires installed package {package!r}"
        ) from exc


def resolve_kvbm_worker_provider(
    requested: object,
    *,
    version_reader: Callable[[str], str] = metadata.version,
) -> ResolvedKVBMProvider:
    """Resolve and validate the KVBM adapter from installed package versions."""

    configured = normalize_kvbm_provider(requested)
    ascend_version = _read_version("vllm-ascend", version_reader)
    vllm_version = _read_version("vllm", version_reader)
    ascend_line = _version_line("vllm-ascend", ascend_version)
    vllm_line = _version_line("vllm", vllm_version)
    if ascend_line != vllm_line:
        raise RuntimeError(
            "incompatible KVBM engine packages: "
            f"vllm-ascend={ascend_version} and vllm={vllm_version} use different "
            "major/minor release lines"
        )

    detected = _KNOWN_PROVIDERS.get(ascend_line)
    if detected is None:
        raise RuntimeError(
            "no KVBM worker provider is known for installed versions "
            f"vllm-ascend={ascend_version}, vllm={vllm_version}"
        )
    if configured != AUTO_PROVIDER and configured != detected:
        raise RuntimeError(
            f"configured KVBM worker provider {configured!r} does not match installed "
            f"vllm-ascend={ascend_version}, vllm={vllm_version}; detected {detected!r}"
        )

    selected = detected if configured == AUTO_PROVIDER else configured
    spec = _PROVIDER_SPECS.get(selected)
    if spec is None:
        raise RuntimeError(
            f"detected {selected!r} from vllm-ascend={ascend_version} and "
            f"vllm={vllm_version}, but this build has no registered KVBM adapter for it"
        )
    return ResolvedKVBMProvider(
        requested=configured,
        spec=spec,
        vllm_ascend_version=ascend_version,
        vllm_version=vllm_version,
    )
