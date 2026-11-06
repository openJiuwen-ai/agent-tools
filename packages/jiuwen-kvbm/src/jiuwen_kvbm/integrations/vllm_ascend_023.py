"""vLLM-Ascend 0.23 integration built on the shared KVBM hook contracts."""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

import jiuwen_kvbm.integrations.vllm_ascend_018 as _shared

logger = logging.getLogger("jiuwen.kvbm.vllm_ascend_023")

_ENGINE_CORE_PATCH_ATTR = "_jiuwen_kvbm_023_entrypoint_hook_installed"
_WORKER_PATCH_ATTR = "_jiuwen_kvbm_023_worker_entrypoint_hook_installed"
_WORKER_ORIGINAL_ATTR = "_jiuwen_kvbm_023_original_worker_main"


def _missing_runtime_contracts() -> list[str]:
    """Return missing private APIs required by the 0.23 integration."""

    missing_modules = []
    for module_name in (
        "vllm_ascend.worker.model_runner_v1",
        "vllm_ascend.patch.platform.patch_multiproc_executor",
        "vllm_ascend.patch.platform.patch_balance_schedule",
    ):
        try:
            if find_spec(module_name) is None:
                missing_modules.append(module_name)
        except (ImportError, AttributeError, ValueError):
            missing_modules.append(module_name)
    if missing_modules:
        return [f"module {module_name}" for module_name in missing_modules]

    try:
        from vllm.distributed.kv_events import ZmqEventPublisher
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.sched.output import NewRequestData
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.engine.core import EngineCoreProc
        from vllm.v1.executor.multiproc_executor import WorkerProc
    except Exception as exc:
        return [f"runtime imports: {type(exc).__name__}: {exc}"]

    contracts = {
        "EngineCoreProc.run_engine_core": (EngineCoreProc, "run_engine_core"),
        "WorkerProc.worker_main": (WorkerProc, "worker_main"),
        "ZmqEventPublisher.publish": (ZmqEventPublisher, "publish"),
        "BlockPool.cache_full_blocks": (BlockPool, "cache_full_blocks"),
        "BlockPool.get_new_blocks": (BlockPool, "get_new_blocks"),
        "BlockPool.free_blocks": (BlockPool, "free_blocks"),
        "KVCacheManager.get_computed_blocks": (
            KVCacheManager,
            "get_computed_blocks",
        ),
        "KVCacheManager.take_events": (KVCacheManager, "take_events"),
        "KVCacheManager.free": (KVCacheManager, "free"),
        "Scheduler.schedule": (Scheduler, "schedule"),
        "Scheduler.add_request": (Scheduler, "add_request"),
        "Scheduler.update_from_output": (Scheduler, "update_from_output"),
        "NewRequestData.from_request": (NewRequestData, "from_request"),
    }
    return [name for name, (owner, attr) in contracts.items() if not hasattr(owner, attr)]


def validate_vllm_ascend_023_runtime() -> None:
    """Fail before engine startup when a required 0.23 private API is absent."""

    missing = _missing_runtime_contracts()
    if missing:
        raise RuntimeError("vLLM-Ascend 0.23 KVBM compatibility check failed; missing: " + ", ".join(missing))


def install_vllm_ascend_023_engine_core_hook(**kwargs: Any) -> bool:
    """Install the shared model-runner hook after validating 0.23 APIs."""

    validate_vllm_ascend_023_runtime()
    installed = _shared.install_vllm_ascend_018_engine_core_hook(
        provider_label="0.23",
        **kwargs,
    )
    if installed:
        logger.info("Jiuwen KVBM vLLM-Ascend 0.23 engine-core hook installed")
    return installed


def _install_worker_entrypoint_hook() -> bool:
    """Patch the base WorkerProc entrypoint used by vLLM-Ascend 0.23."""

    from vllm.v1.executor.multiproc_executor import WorkerProc

    if getattr(WorkerProc, _WORKER_PATCH_ATTR, False):
        return True
    original = WorkerProc.worker_main
    setattr(WorkerProc, _WORKER_ORIGINAL_ATTR, original)
    WorkerProc.worker_main = staticmethod(jiuwen_kvbm_vllm_ascend_023_worker_main)
    setattr(WorkerProc, _WORKER_PATCH_ATTR, True)
    logger.info("Jiuwen KVBM vLLM-Ascend 0.23 WorkerProc entrypoint hook installed")
    return True


def _install_child_process_hooks() -> None:
    """Install scheduler and worker hooks inside the EngineCore subprocess."""

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    installed = {
        "worker_entrypoint": _install_worker_entrypoint_hook(),
        "model_runner": _shared.install_vllm_ascend_018_engine_core_hook(
            diagnostics=_shared.diagnostics_enabled_from_env(),
            provider_label="0.23",
        ),
        "event_publisher": _shared.install_vllm_ascend_018_kv_event_publisher_hook(),
        "block_pool": _shared.install_vllm_ascend_018_block_pool_hook(),
        "kv_cache_manager": _shared.install_vllm_ascend_018_kv_cache_manager_hook(),
        "scheduler": _shared.install_vllm_ascend_018_scheduler_diagnostics_hook(),
    }
    failed = [name for name, ok in installed.items() if not ok]
    if failed:
        raise RuntimeError("vLLM-Ascend 0.23 EngineCore hook installation failed: " + ", ".join(failed))


def jiuwen_kvbm_vllm_ascend_023_run_engine_core(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """EngineCore entrypoint that installs 0.23 hooks before Ascend startup."""

    _install_child_process_hooks()
    from vllm_ascend.patch.platform.patch_balance_schedule import run_engine_core

    return run_engine_core(*args, **kwargs)


def jiuwen_kvbm_vllm_ascend_023_worker_main(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Worker entrypoint used by the 0.23 base WorkerProc target."""

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    installed = {
        "model_runner": _shared.install_vllm_ascend_018_engine_core_hook(
            diagnostics=_shared.diagnostics_enabled_from_env(),
            provider_label="0.23",
        ),
        "event_publisher": _shared.install_vllm_ascend_018_kv_event_publisher_hook(),
        "block_pool": _shared.install_vllm_ascend_018_block_pool_hook(),
        "kv_cache_manager": _shared.install_vllm_ascend_018_kv_cache_manager_hook(),
    }
    failed = [name for name, ok in installed.items() if not ok]
    if failed:
        raise RuntimeError("vLLM-Ascend 0.23 Worker hook installation failed: " + ", ".join(failed))

    from vllm.v1.executor.multiproc_executor import WorkerProc

    original = getattr(WorkerProc, _WORKER_ORIGINAL_ATTR, None)
    if original is None:
        original = WorkerProc.worker_main
    if original is jiuwen_kvbm_vllm_ascend_023_worker_main:
        raise RuntimeError("vLLM-Ascend 0.23 WorkerProc original entrypoint was lost")
    return original(*args, **kwargs)


def install_vllm_ascend_023_multiprocess_hooks(**kwargs: Any) -> bool:
    """Install the frontend EngineCore entrypoint for vLLM-Ascend 0.23."""

    validate_vllm_ascend_023_runtime()
    if not _shared.install_vllm_ascend_018_multiprocess_hooks(**kwargs):
        return False

    import vllm.v1.engine.core as core_module
    from vllm.v1.engine.core import EngineCoreProc

    EngineCoreProc.run_engine_core = staticmethod(jiuwen_kvbm_vllm_ascend_023_run_engine_core)
    setattr(EngineCoreProc, _ENGINE_CORE_PATCH_ATTR, True)
    core_module.EngineCoreProc.run_engine_core = EngineCoreProc.run_engine_core
    logger.info("Jiuwen KVBM vLLM-Ascend 0.23 multiprocess hook installed")
    return True


# These shared hooks use the common BlockPool, KVCacheManager and Scheduler
# contracts while tolerating release-specific optional methods.
install_vllm_ascend_023_block_pool_hook = _shared.install_vllm_ascend_018_block_pool_hook
install_vllm_ascend_023_kv_cache_manager_hook = _shared.install_vllm_ascend_018_kv_cache_manager_hook
install_vllm_ascend_023_kv_event_publisher_hook = _shared.install_vllm_ascend_018_kv_event_publisher_hook
install_vllm_ascend_023_scheduler_diagnostics_hook = _shared.install_vllm_ascend_018_scheduler_diagnostics_hook


def get_bound_vllm_ascend_023_dataplane() -> Any:
    return _shared.get_bound_vllm_ascend_018_dataplane()


def get_bound_vllm_ascend_023_physical_block_resolver() -> Any:
    return _shared.get_bound_vllm_ascend_018_physical_block_resolver()
