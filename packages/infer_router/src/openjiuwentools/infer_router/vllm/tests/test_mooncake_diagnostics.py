# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
from __future__ import annotations

import logging
import sys
import time
import types
from types import SimpleNamespace

sys.modules.setdefault("uvloop", types.ModuleType("uvloop"))

from openjiuwentools.infer_router.vllm import mooncake_diagnostics
from openjiuwentools.infer_router.vllm.main import (
    _install_mooncake_diagnostics_hook,
)
from openjiuwentools.infer_router.vllm.mooncake_diagnostics import (
    install_mooncake_scheduler_diagnostics_hook,
    restore_mooncake_scheduler_diagnostics_hook,
)
from openjiuwentools.infer_router.vllm.transport import (
    _attach_kv_transfer_params,
    _sampling_extra_args,
)


def test_transport_forwards_and_returns_native_connector_metadata():
    transfer_params = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
    }
    assert _sampling_extra_args({"kv_transfer_params": transfer_params}) == {
        "kv_transfer_params": transfer_params,
    }
    assert _sampling_extra_args({}) is None

    output_params = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_block_ids": [1, 2],
    }
    response = _attach_kv_transfer_params(
        {"id": "cmpl-1"},
        SimpleNamespace(kv_transfer_params=output_params),
    )
    assert response["kv_transfer_params"] == output_params


def test_async_remote_wait_restores_computed_tokens_for_ascend_018(caplog):
    request = SimpleNamespace(
        request_id="engine-request-ascend-018",
        trace_headers={"x-jiuwen-request-id": "cmpl-request-ascend-018"},
        arrival_time=time.time(),
        num_computed_tokens=0,
        status=SimpleNamespace(name="WAITING"),
    )

    class FakeConnector:
        @staticmethod
        def get_num_new_matched_tokens(_request, _local_tokens):
            return 100, True

    scheduler = SimpleNamespace(
        connector=FakeConnector(),
        requests={request.request_id: request},
    )
    caplog.set_level(logging.INFO)
    assert mooncake_diagnostics._install_connector_diagnostics_hook(
        scheduler,
        logging.getLogger("test.mooncake"),
    )
    assert scheduler.connector.get_num_new_matched_tokens(request, 20) == (
        100,
        True,
    )

    # Restore only after allocate_slots succeeded and the request entered wait.
    assert request.num_computed_tokens == 0
    request.status = SimpleNamespace(name="WAITING_FOR_REMOTE_KVS")
    mooncake_diagnostics._record_remote_waiting_requests(scheduler, time.time())

    assert request.num_computed_tokens == 120
    assert "[MooncakeCompatibility] restored async remote computed tokens" in caplog.text


def test_scheduler_hook_patches_runtime_schedule_override(caplog):
    request = SimpleNamespace(
        request_id="engine-request-runtime",
        trace_headers={"x-jiuwen-request-id": "cmpl-request-runtime"},
        arrival_time=time.time() - 0.01,
    )

    class BaseScheduler:
        def __init__(self):
            self.running = [request]
            self.waiting = []
            self.requests = {}
            self.remote_kv_ready = True
            self.scheduled_tokens = {}

        def add_request(self, new_request):
            self.requests[new_request.request_id] = new_request

        def _update_waiting_for_remote_kv(self, _request):
            return self.remote_kv_ready

        def schedule(self):
            return SimpleNamespace(num_scheduled_tokens=dict(self.scheduled_tokens))

    class RuntimeScheduler(BaseScheduler):
        def _update_waiting_for_remote_kv(self, request):
            return super()._update_waiting_for_remote_kv(request)

        def schedule(self):
            self._update_waiting_for_remote_kv(request)
            return SimpleNamespace(
                num_scheduled_tokens={request.request_id: 1},
            )

    original_runtime_schedule = RuntimeScheduler.__dict__["schedule"]
    original_runtime_update = RuntimeScheduler.__dict__["_update_waiting_for_remote_kv"]
    caplog.set_level(logging.INFO)
    assert install_mooncake_scheduler_diagnostics_hook(BaseScheduler)
    try:
        scheduler = RuntimeScheduler()
        scheduler.add_request(request)
        assert scheduler.requests == {request.request_id: request}
        assert RuntimeScheduler.schedule is not original_runtime_schedule
        assert RuntimeScheduler._update_waiting_for_remote_kv is not original_runtime_update
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {request.request_id: 1}
    finally:
        restore_mooncake_scheduler_diagnostics_hook(BaseScheduler)
        RuntimeScheduler.schedule = original_runtime_schedule
        RuntimeScheduler._update_waiting_for_remote_kv = original_runtime_update

    assert RuntimeScheduler.schedule is original_runtime_schedule
    assert RuntimeScheduler._update_waiting_for_remote_kv is original_runtime_update
    base = BaseScheduler()
    base.scheduled_tokens = {"base-request": 2}
    assert base.schedule().num_scheduled_tokens == {"base-request": 2}
    base.remote_kv_ready = False
    assert base._update_waiting_for_remote_kv(request) is False
    assert "Mooncake runtime scheduler diagnostics hook installed" in caplog.text
    assert "update_owner=" in caplog.text
    assert "request=cmpl-request-runtime" in caplog.text


def test_setup_accepts_vllm_config_object(monkeypatch, caplog):
    installed = []
    monkeypatch.setattr(
        mooncake_diagnostics,
        "install_mooncake_engine_core_diagnostics",
        lambda **_kwargs: installed.append(True) or True,
    )
    args = SimpleNamespace(
        worker_mode="decode",
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeConnector",
            kv_role="kv_consumer",
        ),
        kvbm_worker_enabled=False,
    )

    caplog.set_level(logging.INFO)
    _install_mooncake_diagnostics_hook(args)

    assert installed == [True]
    assert "[MooncakeDiagnosticsSetup] enabled=True reason=installed" in caplog.text


def test_engine_core_entrypoint_avoids_ascend_patch_recursion(monkeypatch):
    calls = []

    def original_run_engine_core(value):
        calls.append("original")
        return f"original:{value}"

    class FakeEngineCoreProc:
        run_engine_core = staticmethod(original_run_engine_core)

    core_module = types.ModuleType("vllm.v1.engine.core")
    core_module.EngineCoreProc = FakeEngineCoreProc
    engine_module = types.ModuleType("vllm.v1.engine")
    engine_module.core = core_module
    v1_module = types.ModuleType("vllm.v1")
    v1_module.engine = engine_module
    vllm_module = types.ModuleType("vllm")
    vllm_module.v1 = v1_module

    balance_module = types.ModuleType("vllm_ascend.patch.platform.patch_balance_schedule")

    def ascend_run_engine_core(*args, **kwargs):
        calls.append("ascend")
        return balance_module._ORIGINAL_RUN_ENGINE_CORE(*args, **kwargs)

    balance_module.run_engine_core = ascend_run_engine_core
    platform_module = types.ModuleType("vllm_ascend.patch.platform")
    platform_module.patch_balance_schedule = balance_module
    patch_module = types.ModuleType("vllm_ascend.patch")
    patch_module.platform = platform_module
    ascend_module = types.ModuleType("vllm_ascend")
    ascend_module.patch = patch_module
    ascend_utils_module = types.ModuleType("vllm_ascend.utils")

    def adapt_patch(*, is_global_patch):
        assert is_global_patch is True
        # vLLM-Ascend captures the currently installed Mooncake wrapper as
        # its original entrypoint unless the wrapper repairs this reference.
        balance_module._ORIGINAL_RUN_ENGINE_CORE = FakeEngineCoreProc.run_engine_core
        FakeEngineCoreProc.run_engine_core = staticmethod(ascend_run_engine_core)

    ascend_utils_module.adapt_patch = adapt_patch
    fake_modules = {
        "vllm": vllm_module,
        "vllm.v1": v1_module,
        "vllm.v1.engine": engine_module,
        "vllm.v1.engine.core": core_module,
        "vllm_ascend": ascend_module,
        "vllm_ascend.utils": ascend_utils_module,
        "vllm_ascend.patch": patch_module,
        "vllm_ascend.patch.platform": platform_module,
        "vllm_ascend.patch.platform.patch_balance_schedule": balance_module,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        mooncake_diagnostics,
        "install_mooncake_scheduler_diagnostics_hook",
        lambda **_kwargs: True,
    )

    assert mooncake_diagnostics.install_mooncake_engine_core_diagnostics()
    result = FakeEngineCoreProc.run_engine_core("request")

    assert result == "original:request"
    assert calls == ["ascend", "original"]
    assert balance_module._ORIGINAL_RUN_ENGINE_CORE is original_run_engine_core
