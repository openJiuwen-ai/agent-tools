# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
"""Compatibility tests for the vLLM-Ascend 0.23 integration."""

from __future__ import annotations

import sys
import types

from jiuwen_kvbm.integrations import vllm_ascend_023 as integration


def _install_fake_worker_modules(monkeypatch, worker_proc: type) -> None:
    vllm_module = types.ModuleType("vllm")
    vllm_v1_module = types.ModuleType("vllm.v1")
    executor_module = types.ModuleType("vllm.v1.executor")
    multiproc_module = types.ModuleType("vllm.v1.executor.multiproc_executor")
    multiproc_module.WorkerProc = worker_proc
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.executor", executor_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.executor.multiproc_executor",
        multiproc_module,
    )


def test_multiprocess_hook_replaces_engine_core_entrypoint(monkeypatch) -> None:
    class EngineCoreProc:
        @staticmethod
        def run_engine_core(*args, **kwargs):
            return None

    vllm_module = types.ModuleType("vllm")
    vllm_v1_module = types.ModuleType("vllm.v1")
    engine_module = types.ModuleType("vllm.v1.engine")
    core_module = types.ModuleType("vllm.v1.engine.core")
    core_module.EngineCoreProc = EngineCoreProc
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core_module)
    monkeypatch.setattr(integration, "validate_vllm_ascend_023_runtime", lambda: None)
    monkeypatch.setattr(
        integration._shared,
        "install_vllm_ascend_018_multiprocess_hooks",
        lambda **kwargs: True,
    )

    installed = integration.install_vllm_ascend_023_multiprocess_hooks()

    assert installed is True
    assert EngineCoreProc.run_engine_core is integration.jiuwen_kvbm_vllm_ascend_023_run_engine_core


def test_worker_hook_patches_base_worker_proc_used_by_023(monkeypatch) -> None:
    calls = []

    class WorkerProc:
        @staticmethod
        def worker_main(*args, **kwargs):
            calls.append((args, kwargs))
            return "worker-result"

    _install_fake_worker_modules(monkeypatch, WorkerProc)
    ascend_module = types.ModuleType("vllm_ascend")
    utils_module = types.ModuleType("vllm_ascend.utils")

    def adapt_patch_stub(**kwargs):
        return None

    utils_module.adapt_patch = adapt_patch_stub
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils_module)
    for name in (
        "install_vllm_ascend_018_engine_core_hook",
        "install_vllm_ascend_018_kv_event_publisher_hook",
        "install_vllm_ascend_018_block_pool_hook",
        "install_vllm_ascend_018_kv_cache_manager_hook",
    ):
        monkeypatch.setattr(integration._shared, name, lambda **kwargs: True)

    assert integration._install_worker_entrypoint_hook() is True
    result = WorkerProc.worker_main("request", rank=1)

    assert result == "worker-result"
    assert calls == [(("request",), {"rank": 1})]


def test_shared_block_pool_hook_supports_023_without_insert_method() -> None:
    class Block:
        block_id = 7
        block_hash = None

    class BlockPool:
        def __init__(self):
            self.free_block_queue = []
            self.cached_blocks = {}

        @staticmethod
        def get_new_blocks(count):
            return []

        @staticmethod
        def free_blocks(blocks):
            return None

        @staticmethod
        def touch(blocks):
            return None

        @staticmethod
        def reset_prefix_cache():
            return True

        def cache_full_blocks(self, request, blocks, *args, **kwargs):
            blocks[0].block_hash = "hash-7"
            self.cached_blocks[blocks[0].block_hash] = blocks[0]
            return 1

    assert integration.install_vllm_ascend_023_block_pool_hook(block_pool_cls=BlockPool)
    block = Block()
    pool = BlockPool()

    assert pool.cache_full_blocks(object(), [block], 0, 1) == 1
    assert block.block_hash == "hash-7"
    assert pool.cached_blocks["hash-7"] is block
