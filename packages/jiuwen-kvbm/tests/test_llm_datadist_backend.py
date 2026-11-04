# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
from __future__ import annotations

import http.client
import json
import sys
import threading
import time
import types
import urllib.request
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from unittest.mock import Mock

import pytest

from jiuwen_kvbm.accessor import FakeTensorAccessor, VllmAscend018TensorProvider
from jiuwen_kvbm.integrations import vllm_ascend_018 as integration
from jiuwen_kvbm.integrations.vllm_ascend_018 import _llm_datadist_config_from_env
from jiuwen_kvbm.runtime import WorkerKVBMConfig, WorkerKVBMRuntime
from jiuwen_kvbm.transfer import (
    FakeLLMDataDistBackend,
    LLMDataDistBlockMapping,
    LLMDataDistBackendError,
    LLMDataDistCacheDesc,
    LLMDataDistCacheKey,
    LLMDataDistCluster,
    LLMDataDistConfig,
    LLMDataDistEndpoint,
    LLMDataDistRole,
    PythonLLMDataDistBackend,
)
from jiuwen_kvbm.transfer.llm_datadist import _raise_cluster_statuses


def _hash_descriptor(packed_hash, external_hash):
    return {"packed_hex": packed_hash.hex(), "external_hash": str(external_hash), "group_id": 0}


def _source_block_group(index, block_hash):
    return {
        "logical_index": index,
        "group_id": 0,
        "block_id": 7 + index,
        "block_hash": block_hash,
    }


def _source_lease_block(block_id, block_hash):
    return {"block_id": block_id, "block_hash": {"packed_hex": block_hash.hex()}}


def test_model_step_diagnostics_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("JIUWEN_KVBM_MODEL_STEP_SAMPLE_EVERY", raising=False)
    cls = type("Runner", (), {"execute_model": lambda self, output: output})
    original = cls.execute_model
    assert not integration._install_model_step_diagnostics(cls, Mock())
    assert cls.execute_model is original


def test_model_step_diagnostics_sample_bound_and_preserve_errors(monkeypatch):
    monkeypatch.setenv("JIUWEN_KVBM_MODEL_STEP_SAMPLE_EVERY", "2")
    monkeypatch.setenv("JIUWEN_KVBM_MODEL_STEP_MAX_SAMPLES", "2")

    def execute(self, output):
        if output is None:
            raise RuntimeError("execute failed")
        return output

    cls = type("Runner", (), {"execute_model": execute})
    log = Mock()
    assert integration._install_model_step_diagnostics(cls, log)
    assert not integration._install_model_step_diagnostics(cls, log)
    runner = cls()
    output = types.SimpleNamespace(num_scheduled_tokens={"r1": 1, "r2": 1})
    assert runner.execute_model(output) is output
    with pytest.raises(RuntimeError, match="execute failed"):
        runner.execute_model(None)
    for _ in range(6):
        assert runner.execute_model(output) is output
    records = [call for call in log.info.call_args_list if call.args[0].startswith("[KVBMModelStepTiming]")]
    assert len(records) == 2
    assert records[0].args[-1] is False
    assert records[1].args[-1] is True


def test_fake_llm_datadist_backend_records_cache_and_pull() -> None:
    """Fake LLM-DataDist backend 应记录 cache 注册和 block pull 请求。"""

    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()

    desc = LLMDataDistCacheDesc(
        placement="device",
        data_type="bf16",
        shape=(128, 8, 128),
        num_tensors=72,
        layer_count=36,
    )
    key = LLMDataDistCacheKey("prefix-a", prompt_cluster_id=1)
    handle = backend.register_cache(desc, [key])
    mapping = LLMDataDistBlockMapping(
        cache_key=key,
        prompt_blocks=(1, 2),
        decoder_blocks=(10, 11),
    )

    backend.pull_blocks(handle.cache_id, mapping)

    assert backend.health()["ok"] is True
    assert handle.cache_id == 1
    assert backend.pull_requests == [(1, mapping)]


def test_worker_runtime_stats_exposes_llm_datadist_health(tmp_path: Path) -> None:
    """Worker runtime stats 应暴露可选 LLM-DataDist backend 状态。"""

    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()
    runtime = WorkerKVBMRuntime(
        WorkerKVBMConfig(
            worker_id="worker-0",
            disk_root=tmp_path,
            data_backend="llm_datadist",
        ),
        accessor=FakeTensorAccessor({}),
        llm_datadist_backend=backend,
    )

    stats = runtime.stats()

    assert stats["data_backend"] == "llm_datadist"
    assert stats["llm_datadist"]["mode"] == "fake"
    assert stats["llm_datadist"]["endpoint"]["listen_port"] == 19000


def test_llm_datadist_config_uses_device_id_for_default_port(monkeypatch) -> None:
    """engine-core LLM-DataDist 配置应按 device_id 推导默认监听端口。"""

    class Provider:
        device_id = None

        @staticmethod
        def diagnostics(block_id: int = 0) -> dict[str, str]:
            return {"key_device": "npu:3", "value_device": "npu:3"}

    monkeypatch.setenv("JIUWEN_KVBM_LLM_DATADIST_ROLE", "decoder")
    monkeypatch.setenv("JIUWEN_KVBM_LLM_DATADIST_CLUSTER_ID", "42")
    monkeypatch.setenv("JIUWEN_KVBM_LLM_DATADIST_LISTEN_IP", "10.0.0.8")
    monkeypatch.setenv("JIUWEN_KVBM_LLM_DATADIST_LISTEN_PORT", "0")
    monkeypatch.setenv("JIUWEN_KVBM_LLM_DATADIST_LISTEN_PORT_BASE", "19100")

    config = _llm_datadist_config_from_env(Provider())  # type: ignore[arg-type]

    assert config.endpoint.role == LLMDataDistRole.DECODER
    assert config.endpoint.cluster_id == 42
    assert config.endpoint.listen_ip == "10.0.0.8"
    assert config.endpoint.device_id == 3
    assert config.endpoint.listen_port == 19103


def test_python_llm_datadist_backend_initializes_object_api(monkeypatch) -> None:
    """Python backend 应优先兼容 CANN 示例中的 LLMDataDist 对象式 API。"""

    calls: list[tuple[str, object]] = []

    class LLMRole:
        MIX = "Mix"

    class LLMConfig:
        def __init__(self) -> None:
            self.device_id = -1
            self.listen_ip_info = ""
            self.enable_cache_manager = False
            self.enable_remote_cache_accessible = False

        def generate_options(self) -> dict[str, str]:
            return {
                "llm.DeviceId": str(self.device_id),
                "llm.ListenIpInfo": self.listen_ip_info,
                "llm.EnableCacheManager": "1" if self.enable_cache_manager else "0",
            }

    class LLMDataDist:
        def __init__(self, role: object, cluster_id: int) -> None:
            calls.append(("ctor", (role, cluster_id)))
            self.cache_manager = object()

        @staticmethod
        def init(options: dict[str, str]) -> int:
            calls.append(("init", dict(options)))
            return 0

        @staticmethod
        def finalize() -> int:
            calls.append(("finalize", None))
            return 0

    module = types.ModuleType("fake_llm_datadist")
    module.LLMRole = LLMRole
    module.LLMConfig = LLMConfig
    module.LLMDataDist = LLMDataDist
    monkeypatch.setitem(sys.modules, "fake_llm_datadist", module)

    backend = PythonLLMDataDistBackend(_fake_config(), module_name="fake_llm_datadist")
    backend.initialize()
    backend.finalize()

    assert calls[0] == ("ctor", ("Mix", 0))
    assert calls[1] == (
        "init",
        {
            "llm.DeviceId": "0",
            "llm.ListenIpInfo": "127.0.0.1:19000",
            "llm.EnableCacheManager": "1",
        },
    )
    assert calls[2] == ("finalize", None)


def test_python_llm_datadist_backend_registers_and_pulls_blocks(monkeypatch) -> None:
    """Python backend 应按 CANN object API 注册 blocks cache 并执行 pull_blocks。"""

    calls: list[tuple[str, object]] = []

    class LLMRole:
        MIX = "Mix"

    class LLMConfig:
        def __init__(self) -> None:
            self.device_id = -1
            self.enable_cache_manager = False
            self.enable_remote_cache_accessible = False

        def generate_options(self) -> dict[str, str]:
            return {"llm.DeviceId": str(self.device_id)}

    class DataType:
        DT_BF16 = "bf16"

    class Placement:
        DEVICE = "device"

    class BlocksCacheKey:
        def __init__(self, cluster_id: int, model_id: int = 0) -> None:
            self.cluster_id = cluster_id
            self.model_id = model_id

    class CacheDesc:
        def __init__(
            self,
            num_tensors: int,
            shape: list[int],
            data_type: object,
            placement: object,
        ) -> None:
            self.num_tensors = num_tensors
            self.shape = shape
            self.data_type = data_type
            self.placement = placement

    class Cache:
        cache_id = 7

    class CacheManager:
        @staticmethod
        def register_blocks_cache(
            cache_desc: CacheDesc,
            addrs: list[int],
            blocks_cache_key: BlocksCacheKey,
            remote_accessible: bool | None = None,
        ) -> Cache:
            calls.append(
                (
                    "register_blocks_cache",
                    (
                        cache_desc.num_tensors,
                        cache_desc.shape,
                        cache_desc.data_type,
                        cache_desc.placement,
                        addrs,
                        blocks_cache_key.cluster_id,
                        blocks_cache_key.model_id,
                        remote_accessible,
                    ),
                )
            )
            return Cache()

        @staticmethod
        def pull_blocks(
            src_cache_key: BlocksCacheKey,
            dst_cache: Cache,
            src_blocks: list[int],
            dst_blocks: list[int],
            **kwargs: object,
        ) -> None:
            calls.append(
                (
                    "pull_blocks",
                    (
                        src_cache_key.cluster_id,
                        src_cache_key.model_id,
                        dst_cache.cache_id,
                        src_blocks,
                        dst_blocks,
                        kwargs,
                    ),
                )
            )

    class LLMDataDist:
        def __init__(self, role: object, cluster_id: int) -> None:
            self.cache_manager = CacheManager()

        @staticmethod
        def init(options: dict[str, str]) -> None:
            return None

    module = types.ModuleType("fake_llm_datadist_blocks")
    module.LLMRole = LLMRole
    module.LLMConfig = LLMConfig
    module.LLMDataDist = LLMDataDist
    module.DataType = DataType
    module.Placement = Placement
    module.BlocksCacheKey = BlocksCacheKey
    module.CacheDesc = CacheDesc
    monkeypatch.setitem(sys.modules, "fake_llm_datadist_blocks", module)

    backend = PythonLLMDataDistBackend(_fake_config(), module_name="fake_llm_datadist_blocks")
    backend.initialize()
    desc = LLMDataDistCacheDesc(
        placement="device",
        data_type="torch.bfloat16",
        shape=(16, 128, 8, 128),
        num_tensors=4,
        tensor_addresses=(101, 102, 201, 202),
        layer_count=2,
    )
    key = LLMDataDistCacheKey("local", prompt_cluster_id=3)
    handle = backend.register_cache(desc, [key])
    backend.pull_blocks(
        handle.cache_id,
        LLMDataDistBlockMapping(
            cache_key=LLMDataDistCacheKey("remote", prompt_cluster_id=9),
            prompt_blocks=(1, 2),
            decoder_blocks=(11, 12),
            layer_range=(0, 2),
        ),
    )

    assert handle.cache_id == 7
    assert calls[0] == (
        "register_blocks_cache",
        (4, [16, 128, 8, 128], "bf16", "device", [101, 102, 201, 202], 3, 0, None),
    )
    assert calls[1] == (
        "pull_blocks",
        (
            9,
            0,
            7,
            [1, 2],
            [11, 12],
            {
                "tensor_num_per_layer": 2,
                "src_layer_range": range(0, 2),
                "dst_layer_range": range(0, 2),
            },
        ),
    )


def test_vllm_ascend_provider_exports_llm_datadist_addresses() -> None:
    """vLLM-Ascend provider 应按 layer K/V 顺序导出 LLM-DataDist 地址。"""

    kv_caches = {
        "layer0": (
            _FakeKVTensor((4, 2), "torch.bfloat16", 101),
            _FakeKVTensor((4, 2), "torch.bfloat16", 102),
        ),
        "layer1": (
            _FakeKVTensor((4, 2), "torch.bfloat16", 201),
            _FakeKVTensor((4, 2), "torch.bfloat16", 202),
        ),
    }
    provider = VllmAscend018TensorProvider(kv_caches)

    assert provider.tensor_base_addresses() == (101, 102, 201, 202)
    assert provider.blocks_cache_shape() == (4, 2)
    assert provider.blocks_cache_dtype() == "torch.bfloat16"


def test_python_llm_datadist_backend_links_and_unlinks_clusters(monkeypatch) -> None:
    """Python backend 应按 CANN API 映射 cluster 建链和断链参数。"""

    calls: list[tuple[str, object]] = []

    class LLMStatusCode(Enum):
        LLM_SUCCESS = "success"
        LLM_ALREADY_LINK = "already_link"

    class LLMRole:
        MIX = "Mix"

    class LLMClusterInfo:
        def __init__(self) -> None:
            self.remote_cluster_id = -1
            self.local: tuple[str, int] | None = None
            self.remote: tuple[str, int] | None = None

        def append_local_ip_info(self, ip: str, port: int) -> None:
            self.local = (ip, port)

        def append_remote_ip_info(self, ip: str, port: int) -> None:
            self.remote = (ip, port)

    class LLMDataDist:
        def __init__(self, role: object, cluster_id: int) -> None:
            return None

        @staticmethod
        def init(options: dict[str, str]) -> None:
            return None

        @staticmethod
        def link_clusters(clusters: list[LLMClusterInfo], *, timeout: int):
            cluster = clusters[0]
            calls.append(("link", (cluster.remote_cluster_id, cluster.local, cluster.remote, timeout)))
            if len([call for call in calls if call[0] == "link"]) == 1:
                return LLMStatusCode.LLM_SUCCESS, [LLMStatusCode.LLM_SUCCESS]
            return LLMStatusCode.LLM_ALREADY_LINK, [LLMStatusCode.LLM_ALREADY_LINK]

        @staticmethod
        def unlink_clusters(
            clusters: list[LLMClusterInfo],
            *,
            timeout: int,
            force: bool,
        ):
            calls.append(("unlink", (clusters[0].remote_cluster_id, timeout, force)))
            return LLMStatusCode.LLM_SUCCESS, [LLMStatusCode.LLM_SUCCESS]

    module = types.ModuleType("fake_llm_datadist_clusters")
    module.LLMRole = LLMRole
    module.LLMClusterInfo = LLMClusterInfo
    module.LLMDataDist = LLMDataDist
    monkeypatch.setitem(sys.modules, "fake_llm_datadist_clusters", module)

    backend = PythonLLMDataDistBackend(
        _fake_config(),
        module_name="fake_llm_datadist_clusters",
    )
    backend.initialize()
    cluster = LLMDataDistCluster(
        remote_cluster_id=9,
        local_ip="10.0.0.1",
        local_port=26000,
        remote_ip="10.0.0.2",
        remote_port=26001,
    )
    backend.link_clusters([cluster], timeout_ms=5000)
    backend.link_clusters([cluster], timeout_ms=5000)
    assert backend.health()["linked_cluster_ids"] == [9]
    backend.unlink_clusters([cluster], timeout_ms=4000, force=True)

    assert calls == [
        ("link", (9, ("10.0.0.1", 26000), ("10.0.0.2", 26001), 5000)),
        ("link", (9, ("10.0.0.1", 26000), ("10.0.0.2", 26001), 5000)),
        ("unlink", (9, 4000, True)),
    ]
    assert backend.health()["linked_cluster_ids"] == []


def test_engine_core_pd_handoff_protocol_exposes_state_and_rollback(monkeypatch) -> None:
    """Worker 控制面应提供幂等 begin、受控 advance、status 和 rollback。"""

    monkeypatch.setattr(
        integration,
        "_ENGINE_CORE_DATAPLANE",
        types.SimpleNamespace(),
    )
    monkeypatch.setattr(
        integration,
        "_PROCESS_PD_HANDOFFS",
        integration.PDHandoffManager(max_inflight=4),
    )
    payload = {
        "handoff_id": "pd-request-1",
        "request_id": "request-1",
        "source_worker_id": "prefill-0",
        "target_worker_id": "decode-0",
        "model_id": "model-a",
        "prompt_token_count": 1024,
    }

    begun = integration._handle_engine_core_datadist_request(
        "/datadist/pd/handoffs/begin",
        payload,
    )
    replay = integration._handle_engine_core_datadist_request(
        "/datadist/pd/handoffs/begin",
        payload,
    )
    advanced = integration._handle_engine_core_datadist_request(
        "/datadist/pd/handoffs/advance",
        {
            "handoff_id": "pd-request-1",
            "expected_state": "prefill_running",
            "state": "prefill_ready",
            "block_hashes": ["hash-1"],
            "source_blocks": [7],
        },
    )
    status = integration._handle_engine_core_datadist_request(
        "/datadist/pd/handoffs/status",
        {"handoff_id": "pd-request-1"},
    )
    rollback = integration._handle_engine_core_datadist_request(
        "/datadist/pd/handoffs/rollback",
        {"handoff_id": "pd-request-1", "reason": "decode unavailable"},
    )

    assert begun["ok"] is True
    assert begun["created"] is True
    assert replay["created"] is False
    assert advanced["handoff"]["state"] == "prefill_ready"
    assert status["handoff"]["source_blocks"] == [7]
    assert rollback["accepted"] is True
    assert rollback["handoff"]["state"] == "rolling_back"


def test_pd_source_begin_prepares_scheduler_partial_capture(monkeypatch) -> None:
    """Router 标记 source 角色时，Worker 应在 P 请求到达前注册捕获意图。"""

    monkeypatch.setattr(
        integration,
        "_PROCESS_PD_HANDOFFS",
        integration.PDHandoffManager(max_inflight=4),
    )
    dispatched = []

    def dispatch(path, payload):
        dispatched.append((path, payload))
        return {"ok": True, "created": True, "captured": False}

    monkeypatch.setattr(integration, "_dispatch_block_pool_request", dispatch)

    result = integration._pd_handoff_begin_response(
        {
            "handoff_id": "pd-source-prepare-1",
            "request_id": "request-source-1",
            "source_worker_id": "prefill-0",
            "target_worker_id": "decode-0",
            "model_id": "model-a",
            "prompt_token_count": 6300,
            "local_role": "source",
        }
    )

    assert result["ok"] is True
    assert dispatched == [
        (
            "/datadist/pd/export/prepare/local",
            {
                "handoff_id": "pd-source-prepare-1",
                "request_id": "request-source-1",
                "prompt_token_count": 6300,
                "ttl_ms": 120_000,
            },
        )
    ]


def test_legacy_scheduler_kv_bridge_owns_all_pd_control_boundaries(monkeypatch) -> None:
    """现有P/D本地控制入口必须统一经过legacy SchedulerKVBridge。"""

    monkeypatch.setenv("JIUWEN_KVBM_PD_BLOCK_OWNERSHIP", "legacy")
    monkeypatch.setattr(integration, "_SCHEDULER_KV_BRIDGE", None)
    operations = {
        "_local_pd_export_prepare": "prepare_source",
        "_local_pd_export": "export_source",
        "_release_pd_source_export_intent": "release_source",
        "_local_pd_import_prepare": "prepare_target",
        "_local_pd_import_plan": "plan_target",
        "_local_pd_import_attach": "complete_target",
        "_local_pd_import_release": "abort_target",
        "_local_pd_import_status": "target_status",
    }
    calls = []

    for function_name, operation in operations.items():
        if function_name == "_release_pd_source_export_intent":
            monkeypatch.setattr(
                integration,
                function_name,
                lambda request_id, handoff_id, operation=operation: {
                    "ok": True,
                    "operation": operation,
                    "request_id": request_id,
                    "handoff_id": handoff_id,
                },
            )
        else:
            monkeypatch.setattr(
                integration,
                function_name,
                lambda payload, operation=operation: {
                    "ok": True,
                    "operation": operation,
                    "payload": payload,
                },
            )

    paths = {
        "/datadist/pd/export/prepare/local": "prepare_source",
        "/datadist/pd/export/local": "export_source",
        "/datadist/pd/export/release/local": "release_source",
        "/datadist/pd/import/prepare/local": "prepare_target",
        "/datadist/pd/import/plan/local": "plan_target",
        "/datadist/pd/import/attach/local": "complete_target",
        "/datadist/pd/import/release/local": "abort_target",
        "/datadist/pd/import/status/local": "target_status",
    }
    for path, operation in paths.items():
        result = integration._handle_local_block_reservation_request(
            path,
            {"request_id": "request-1", "handoff_id": "handoff-1"},
        )
        calls.append(result["operation"])

    assert calls == list(paths.values())
    assert integration._scheduler_kv_bridge().diagnostics() == {
        "ownership_mode": "legacy",
        "implementation": "_LegacyVllmSchedulerKVBridge",
        "scheduler_owned_destination_blocks": False,
        "uses_external_reservation": True,
    }


def test_scheduler_owned_bridge_selects_native_target_operations(monkeypatch) -> None:
    """scheduler模式必须选择原生plan实现，不能退回legacy reservation。"""

    monkeypatch.setenv("JIUWEN_KVBM_PD_BLOCK_OWNERSHIP", "scheduler")
    monkeypatch.setattr(integration, "_SCHEDULER_KV_BRIDGE", None)
    monkeypatch.setattr(
        integration,
        "_local_scheduler_owned_pd_import_plan",
        lambda payload: {"ok": True, "operation": "scheduler_plan", "payload": payload},
    )
    monkeypatch.setattr(
        integration,
        "_local_pd_import_attach",
        lambda payload: {"ok": True, "operation": "scheduler_commit", "payload": payload},
    )

    planned = integration._handle_local_block_reservation_request(
        "/datadist/pd/import/plan/local",
        {"request_id": "request-1", "handoff_id": "handoff-1"},
    )
    committed = integration._handle_local_block_reservation_request(
        "/datadist/pd/import/attach/local",
        {"request_id": "request-1", "handoff_id": "handoff-1"},
    )

    assert planned["operation"] == "scheduler_plan"
    assert committed["operation"] == "scheduler_commit"
    assert committed["payload"]["ownership_mode"] == "scheduler"
    assert integration._scheduler_kv_bridge().diagnostics() == {
        "ownership_mode": "scheduler",
        "implementation": "_VllmSchedulerOwnedKVBridge",
        "scheduler_owned_destination_blocks": True,
        "uses_external_reservation": False,
    }


def test_finalize_failure_clears_initialized_state():
    backend = PythonLLMDataDistBackend(_fake_config())
    backend.initialized = True
    backend.module = types.SimpleNamespace()

    def fail_finalize():
        raise RuntimeError("injected finalize failure")

    backend.datadist = types.SimpleNamespace(finalize=fail_finalize)
    with pytest.raises(RuntimeError, match="finalize failure"):
        backend.finalize()
    assert backend.initialized is False


def test_scheduler_owned_allocation_commit_and_rollback(monkeypatch) -> None:
    """D目标block应由请求原生持有，HIXL完成后提交，失败时由vLLM归还。"""

    class BlockPool:
        @staticmethod
        def get_num_free_blocks():
            return 7

    class KVCacheManager:
        num_kv_cache_groups = 1

        def __init__(self):
            self.block_pool = BlockPool()
            self.request_blocks = {}
            self.cache_calls = []
            self.free_calls = []
            self.allocate_calls = []
            self.computed_blocks = object()
            self.computed_tokens = 256

        def get_computed_blocks(self, request):
            return self.computed_blocks, self.computed_tokens

        def allocate_slots(self, request, num_new_tokens, **kwargs):
            self.allocate_calls.append((num_new_tokens, kwargs))
            self.request_blocks[request.request_id] = ([10, 11, 12],)
            return object()

        def get_block_ids(self, request_id):
            return self.request_blocks[request_id]

        def cache_blocks(self, request, num_computed_tokens):
            self.cache_calls.append((request.request_id, num_computed_tokens))

        def free(self, request):
            self.free_calls.append(request.request_id)
            self.request_blocks.pop(request.request_id, None)

    token_ids = list(range(300))
    request_hashes = [b"scheduler-owned-full-0" * 2, b"scheduler-owned-full-1" * 2]
    request = types.SimpleNamespace(
        request_id="decode-scheduler-owned-1",
        block_hashes=request_hashes,
        prompt_token_ids=token_ids,
        num_tokens=300,
        num_computed_tokens=0,
        num_cached_tokens=-1,
        num_external_computed_tokens=0,
        cache_salt=None,
        lora_request=None,
        mm_features=None,
        prompt_embeds=None,
    )
    manager = KVCacheManager()
    scheduler = types.SimpleNamespace(
        requests={request.request_id: request},
        running=[],
        kv_cache_manager=manager,
        block_size=128,
    )
    allocation = integration._scheduler_owned_pd_target_allocation(
        scheduler,
        request,
        handoff_id="pd-scheduler-owned-1",
    )
    gate = integration._PDTargetRequestGate(
        handoff_id="pd-scheduler-owned-1",
        request_id=request.request_id,
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 30,
        request_seen=True,
        engine_request_id=request.request_id,
        scheduler_allocation=allocation,
    )
    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {request.request_id: gate})
    monkeypatch.setattr(integration, "_OBSERVED_SCHEDULERS", [scheduler])
    monkeypatch.setattr(
        integration,
        "_process_block_reservations",
        lambda: pytest.fail("scheduler-owned path must not use external reservations"),
    )
    fingerprints = tuple(integration._token_block_fingerprints(token_ids, 128))
    partial_fingerprint = integration._partial_token_fingerprint(token_ids, 128)
    plan_task = integration._PDTargetPlanTask(
        request_id=request.request_id,
        handoff_id=gate.handoff_id,
        block_groups=allocation.block_groups,
        token_block_fingerprints=fingerprints,
        partial_token_fingerprint=partial_fingerprint,
        partial_valid_tokens=44,
        semantic_plain_text=True,
        reservation_ttl_ms=0,
        timeout_seconds=1,
        ownership_mode=integration.SchedulerKVOwnershipMode.SCHEDULER,
    )

    plan = integration._scheduler_plan_owned_pd_import(scheduler, plan_task)
    attach_task = integration._PDTargetAttachTask(
        request_id=request.request_id,
        handoff_id=gate.handoff_id,
        reservation_id=plan["reservation"]["reservation_id"],
        block_hashes=tuple(
            item.get("block_hash") or {"request_scoped_partial": True} for item in allocation.block_groups
        ),
        block_groups=allocation.block_groups,
        timeout_seconds=1,
        token_block_fingerprints=fingerprints,
        partial_token_fingerprint=partial_fingerprint,
        partial_valid_tokens=44,
        semantic_plain_text=True,
        ownership_mode=integration.SchedulerKVOwnershipMode.SCHEDULER,
    )
    committed = integration._scheduler_commit_owned_pd_request(scheduler, attach_task)

    assert manager.allocate_calls[0][0] == 0
    assert manager.allocate_calls[0][1]["num_external_computed_tokens"] == 44
    assert manager.allocate_calls[0][1]["delay_cache_blocks"] is True
    assert plan["scheduler_owned"] is True
    assert plan["planned_destination_blocks"] == [10, 11, 12]
    assert plan["transfer_positions"] == [2]
    assert plan["reservation"]["block_ids"] == [12]
    assert committed["computed_tokens"] == 299
    assert committed["request_scoped_destination_blocks"] == [12]
    assert request.num_computed_tokens == 299
    assert manager.cache_calls == [(request.request_id, 256)]
    assert manager.free_calls == []

    rollback_request = types.SimpleNamespace(**vars(request))
    rollback_request.request_id = "decode-scheduler-owned-rollback"
    rollback_request.num_computed_tokens = 0
    rollback_request.num_cached_tokens = -1
    scheduler.requests[rollback_request.request_id] = rollback_request
    rollback_allocation = integration._scheduler_owned_pd_target_allocation(
        scheduler,
        rollback_request,
        handoff_id="pd-scheduler-owned-rollback",
    )
    rollback_gate = integration._PDTargetRequestGate(
        handoff_id="pd-scheduler-owned-rollback",
        request_id=rollback_request.request_id,
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 30,
        request_seen=True,
        engine_request_id=rollback_request.request_id,
        scheduler_allocation=rollback_allocation,
    )

    assert (
        integration._release_scheduler_owned_pd_allocation(
            rollback_gate,
            reason="test rollback",
        )
        is True
    )
    assert manager.free_calls == [rollback_request.request_id]
    assert rollback_allocation.released is True


def test_pd_export_resolves_request_hashes_and_pins_real_block_pool(monkeypatch) -> None:
    """P export 应按 request_id 解析真实物理块，并通过 BlockPool 引用计数 pin。"""

    external_hashes = (b"pd-export-block-000000000000001", b"pd-export-block-000000000000002")
    packed_hashes = tuple(value + (0).to_bytes(4, "big") for value in external_hashes)

    class Block:
        def __init__(self, block_id, block_hash):
            self.block_id = block_id
            self.block_hash = block_hash
            self.ref_cnt = 0

    blocks = [Block(0, packed_hashes[0]), Block(1, packed_hashes[1])]

    class BlockPool:
        def __init__(self):
            self.blocks = blocks
            self.cached_block_hash_to_block = types.SimpleNamespace(
                _cache={packed_hashes[index]: block for index, block in enumerate(blocks)}
            )

        @staticmethod
        def get_num_free_blocks():
            return 2

        @staticmethod
        def get_new_blocks(count):
            return []

        @staticmethod
        def touch(selected):
            for block in selected:
                block.ref_cnt += 1

        @staticmethod
        def free_blocks(selected):
            for block in selected:
                block.ref_cnt -= 1

    pool = BlockPool()
    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()
    control = types.SimpleNamespace(
        llm_datadist_backend=backend,
        block_reservations=types.SimpleNamespace(
            stats=lambda: {"pool_ready": True},
        ),
    )
    monkeypatch.setattr(integration, "_OBSERVED_BLOCK_POOLS", [pool])
    monkeypatch.setattr(
        integration,
        "_PD_REQUEST_BLOCK_HASHES",
        {},
    )
    monkeypatch.setattr(
        integration,
        "_PD_REQUEST_TOKEN_BLOCK_FINGERPRINTS",
        {},
    )
    integration._record_pd_request_block_hashes(
        types.SimpleNamespace(
            request_id="prefill-request-1-8a773759",
            block_hashes=list(external_hashes),
            prompt_token_ids=list(range(256)),
        ),
        block_size=128,
    )
    monkeypatch.setattr(integration, "_PROCESS_SOURCE_BLOCK_LEASES", None)
    monkeypatch.setattr(integration, "_PROCESS_PD_HANDOFFS", None)
    monkeypatch.setattr(integration, "_ENGINE_CORE_DATAPLANE", control)
    monkeypatch.setattr(integration, "_ENGINE_CORE_LLM_DATADIST_CACHE_ID", 7)

    begun = integration._handle_engine_core_datadist_request(
        "/datadist/pd/handoffs/begin",
        {
            "handoff_id": "pd-export-1",
            "request_id": "prefill-request-1",
            "source_worker_id": "prefill-0",
            "target_worker_id": "decode-0",
            "model_id": "model-a",
            "prompt_token_count": 256,
        },
    )
    exported = integration._handle_engine_core_datadist_request(
        "/datadist/pd/export",
        {"handoff_id": "pd-export-1", "ttl_ms": 30_000},
    )

    assert begun["ok"] is True
    assert exported["ok"] is True
    assert exported["handoff"]["state"] == "source_pinned"
    assert exported["export"]["source_blocks"] == [0, 1]
    assert exported["export"]["logical_block_count"] == 2
    assert exported["export"]["physical_block_count"] == 2
    assert exported["export"]["semantic_plain_text"] is True
    assert len(exported["export"]["token_block_fingerprints"]) == 2
    assert exported["export"]["control_timing"]["engine_core_proxy_ms"] >= 0
    assert exported["export"]["control_timing"]["engine_core_queue_ms"] >= 0
    assert exported["export"]["control_timing"]["engine_core_execution_ms"] >= 0
    assert exported["export"]["control_timing"]["response_build_ms"] >= 0
    assert [block.ref_cnt for block in blocks] == [1, 1]

    replay = integration._handle_engine_core_datadist_request(
        "/datadist/pd/export",
        {"handoff_id": "pd-export-1"},
    )
    released = integration._handle_engine_core_datadist_request(
        "/datadist/pd/export/release",
        {"handoff_id": "pd-export-1", "reason": "test cleanup"},
    )

    assert replay["already_exported"] is True
    assert released["ok"] is True
    assert released["handoff"]["state"] == "rolled_back"
    assert [block.ref_cnt for block in blocks] == [0, 0]


def test_pd_export_captures_and_pins_request_scoped_partial_block(
    monkeypatch,
) -> None:
    """P 请求释放前应捕获 49 个完整 block 和最后 28-token partial block。"""

    external_hashes = tuple(bytes([index + 1]) * 32 for index in range(49))

    class Block:
        def __init__(self, block_id):
            self.block_id = block_id
            self.ref_cnt = 1
            self.block_hash = external_hashes[block_id] + (0).to_bytes(4, "big") if block_id < 49 else None

    class BlockPool:
        def __init__(self):
            self.blocks = [Block(index) for index in range(50)]

        @staticmethod
        def touch(selected):
            for block in selected:
                block.ref_cnt += 1

        @staticmethod
        def free_blocks(selected):
            for block in selected:
                block.ref_cnt -= 1

    pool = BlockPool()
    manager = types.SimpleNamespace(
        block_size=128,
        block_pool=pool,
        get_blocks=lambda _request_id: types.SimpleNamespace(blocks=(pool.blocks,)),
    )
    request = types.SimpleNamespace(
        request_id="partial-request-1-8a773759",
        block_hashes=list(external_hashes),
        prompt_token_ids=list(range(6300)),
    )
    monkeypatch.setattr(integration, "_PD_SOURCE_EXPORT_INTENTS", {})
    monkeypatch.setattr(
        integration,
        "_PROCESS_SOURCE_BLOCK_LEASES",
        integration.VllmSourceBlockLeaseManager(lambda: pool),
    )

    prepared = integration._local_pd_export_prepare(
        {
            "handoff_id": "pd-partial-1",
            "request_id": "partial-request-1",
            "prompt_token_count": 6300,
            "ttl_ms": 30_000,
        }
    )
    intent = integration._PD_SOURCE_EXPORT_INTENTS["partial-request-1"]
    intent.created_at = time.monotonic() - 0.010
    intent.scheduler_seen_at = time.monotonic() - 0.005
    intent.scheduler_seen_cycle_id = 11
    intent.scheduler_running_request_count = 3
    intent.scheduler_waiting_request_count = 2
    intent.first_scheduled_cycle_id = 13
    intent.scheduler_wait_cycles = 1
    intent.first_scheduled_token_count = 28
    intent.first_scheduled_running_request_count = 4
    intent.first_scheduled_waiting_request_count = 1
    intent.first_scheduled_cycle_ms = 31.0
    monkeypatch.setattr(
        integration,
        "_OBSERVED_SCHEDULERS",
        [types.SimpleNamespace(_jiuwen_kvbm_pd_scheduler_cycle_id=13)],
    )
    integration._capture_pd_source_request_blocks(
        manager,
        request,
        integration.logger,
    )
    pool.free_blocks(reversed(pool.blocks))
    exported = integration._local_pd_export({"request_id": "partial-request-1", "ttl_ms": 30_000})

    assert prepared["ok"] is True
    assert exported["ok"] is True
    assert exported["source_blocks"] == list(range(50))
    assert exported["full_logical_block_count"] == 49
    assert exported["partial_valid_tokens"] == 28
    assert exported["block_hashes"][-1] == {
        "request_scoped_partial": True,
        "logical_index": 49,
        "group_id": 0,
        "valid_token_count": 28,
    }
    assert exported["partial_token_fingerprint"]
    assert exported["p_dispatch_to_scheduler_seen_ms"] > 0
    assert exported["p_scheduler_queue_cycles"] == 1
    assert exported["p_scheduler_seen_cycle_id"] == 11
    assert exported["p_first_scheduled_cycle_id"] == 13
    assert exported["p_kv_ready_cycle_id"] == 13
    assert exported["p_scheduler_compute_cycles"] == 1
    assert exported["p_first_scheduled_token_count"] == 28
    assert exported["p_scheduler_running_request_count"] == 3
    assert exported["p_scheduler_waiting_request_count"] == 2
    assert exported["p_first_scheduled_running_request_count"] == 4
    assert exported["p_first_scheduled_waiting_request_count"] == 1
    assert exported["p_first_scheduled_cycle_ms"] == 31.0
    assert exported["p_scheduler_seen_to_kv_ready_ms"] > 0
    assert exported["p_kv_ready_to_export_ready_ms"] >= 0
    assert intent.capture_ready.is_set()
    assert all(block.ref_cnt == 1 for block in pool.blocks)

    integration._process_source_block_leases().release(exported["lease"]["lease_id"])
    assert all(block.ref_cnt == 0 for block in pool.blocks)


def test_pd_export_waits_for_engine_core_capture_ready(monkeypatch) -> None:
    """export 长轮询应由 engine-core capture-ready 事件唤醒并返回捕获错误。"""

    intent = integration._PDSourceExportIntent(
        handoff_id="pd-export-wait-1",
        request_id="prefill-wait-1",
        prompt_token_count=128,
        ttl_ms=30_000,
        created_at=time.monotonic(),
    )
    monkeypatch.setattr(
        integration,
        "_PD_SOURCE_EXPORT_INTENTS",
        {"prefill-wait-1": intent},
    )

    def complete_capture() -> None:
        time.sleep(0.01)
        intent.error = "capture failed for test"
        intent.capture_complete = True
        intent.capture_ready.set()

    thread = threading.Thread(target=complete_capture)
    thread.start()
    result = integration._local_pd_export({"request_id": "prefill-wait-1", "wait_timeout_ms": 1_000})
    thread.join()

    assert result["ok"] is False
    assert result["error"] == ("P request-scoped block capture failed: capture failed for test")


def test_pd_export_returns_retryable_pending_before_capture(monkeypatch) -> None:
    """P capture尚未完成时应返回可重试暂态，而不是最终失败。"""

    intent = integration._PDSourceExportIntent(
        handoff_id="pd-export-pending-1",
        request_id="prefill-pending-1",
        prompt_token_count=128,
        ttl_ms=30_000,
        created_at=time.monotonic(),
    )
    monkeypatch.setattr(
        integration,
        "_PD_SOURCE_EXPORT_INTENTS",
        {"prefill-pending-1": intent},
    )

    result = integration._local_pd_export({"request_id": "prefill-pending-1", "wait_timeout_ms": 0})

    assert result["ok"] is False
    assert result["pending"] is True
    assert result["retryable"] is True
    assert result["handoff_id"] == "pd-export-pending-1"


def test_pd_source_capture_completion_publishes_router_event(monkeypatch) -> None:
    """P capture完成与失败都应发布可重放的Router事件。"""

    broker = Mock()
    monkeypatch.setattr(integration, "_PD_CONTROL_EVENTS", broker)
    ready = integration._PDSourceExportIntent(
        handoff_id="pd-export-ready-1",
        request_id="prefill-ready-1",
        prompt_token_count=128,
        ttl_ms=30_000,
        created_at=time.monotonic(),
    )
    failed = integration._PDSourceExportIntent(
        handoff_id="pd-export-failed-1",
        request_id="prefill-failed-1",
        prompt_token_count=128,
        ttl_ms=30_000,
        created_at=time.monotonic(),
    )

    integration._complete_pd_source_capture(ready)
    integration._complete_pd_source_capture(failed, error="capture failed")

    assert ready.capture_complete is True
    assert ready.capture_ready.is_set()
    assert failed.error == "capture failed"
    broker.publish.assert_any_call(
        "EXPORT_READY",
        handoff_id="pd-export-ready-1",
        request_id="prefill-ready-1",
        result={"ok": True, "error": None},
    )
    broker.publish.assert_any_call(
        "EXPORT_FAILED",
        handoff_id="pd-export-failed-1",
        request_id="prefill-failed-1",
        result={"ok": False, "error": "capture failed"},
    )


def test_pd_scheduler_attach_uses_d_hash_when_token_fingerprints_match(monkeypatch) -> None:
    """P/D 内部 hash 不同时应按 token 指纹校验，并使用 D hash 发布目标块。"""

    source_hashes = (b"pd-source-block-00000000000001", b"pd-source-block-00000000000002")
    target_hashes = (b"pd-target-block-00000000000001", b"pd-target-block-00000000000002")
    source_packed_hashes = tuple(value + (0).to_bytes(4, "big") for value in source_hashes)
    target_packed_hashes = tuple(value + (0).to_bytes(4, "big") for value in target_hashes)

    class Block:
        def __init__(self, block_id):
            self.block_id = block_id
            self.ref_cnt = 0
            self.block_hash = None

    class HashMap:
        def __init__(self):
            self._cache = {}

        def insert(self, block_hash, block):
            self._cache[block_hash] = block

        def pop(self, block_hash, _block_id):
            return self._cache.pop(block_hash, None)

    class BlockPool:
        def __init__(self):
            self.blocks = [Block(10), Block(11), Block(12)]
            self.free = list(self.blocks)
            self.enable_caching = True
            self.cached_block_hash_to_block = HashMap()

        def get_num_free_blocks(self):
            return len(self.free)

        def get_new_blocks(self, count):
            selected = self.free[:count]
            del self.free[:count]
            for block in selected:
                block.ref_cnt += 1
            return selected

        def touch(self, selected):
            for block in selected:
                if block.ref_cnt == 0:
                    self.free.remove(block)
                block.ref_cnt += 1

        def free_blocks(self, selected):
            for block in selected:
                block.ref_cnt -= 1
                if block.ref_cnt == 0 and block not in self.free:
                    self.free.append(block)

    class Coordinator:
        def __init__(self, pool):
            self.pool = pool
            self.req_to_blocks = {}

        def allocate_new_computed_blocks(
            self,
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
            num_external_computed_tokens,
        ):
            assert num_local_computed_tokens == 256
            assert num_external_computed_tokens == 0
            blocks = list(new_computed_blocks[0])
            for block in blocks:
                block.ref_cnt += 1
            self.req_to_blocks[request_id] = blocks

        def free(self, request_id):
            for block in self.req_to_blocks.pop(request_id, []):
                self.pool.free_blocks([block])

    pool = BlockPool()
    coordinator = Coordinator(pool)

    class KVCacheManager:
        num_kv_cache_groups = 1

        def __init__(self):
            self.coordinator = coordinator

        @staticmethod
        def free(request):
            coordinator.free(request.request_id)

    request = types.SimpleNamespace(
        request_id="decode-request-1",
        block_hashes=list(target_hashes),
        prompt_token_ids=list(range(300)),
        num_computed_tokens=0,
        num_cached_tokens=-1,
        num_tokens=300,
    )
    scheduler = types.SimpleNamespace(
        requests={request.request_id: request},
        running=[],
        kv_cache_manager=KVCacheManager(),
        block_size=128,
    )
    existing = pool.get_new_blocks(1)[0]
    existing.block_hash = target_packed_hashes[0]
    pool.cached_block_hash_to_block.insert(target_packed_hashes[0], existing)
    pool.free_blocks([existing])
    reservation_manager = integration.VllmBlockPoolReservationManager(lambda: pool)
    reservation = reservation_manager.reserve(2)
    assert reservation_manager._existing_cached_block_ids(
        pool,
        target_packed_hashes[0],
    ) == [existing.block_id]
    monkeypatch.setattr(
        integration,
        "_PROCESS_BLOCK_RESERVATIONS",
        reservation_manager,
    )
    task = integration._PDTargetAttachTask(
        request_id=request.request_id,
        handoff_id="pd-target-1",
        reservation_id=reservation.reservation_id,
        block_hashes=tuple({"packed_hex": value.hex()} for value in source_packed_hashes),
        block_groups=tuple(
            {
                "logical_index": index,
                "group_id": 0,
                "block_id": index,
                "block_hash": {
                    "packed_hex": source_packed_hashes[index].hex(),
                    "external_hash": str(source_hashes[index]),
                    "group_id": 0,
                },
            }
            for index in range(2)
        ),
        timeout_seconds=1.0,
        token_block_fingerprints=tuple(integration._token_block_fingerprints(list(range(300)), 128)[:2]),
        semantic_plain_text=True,
    )

    result = integration._scheduler_attach_pd_request(scheduler, task)

    assert result["ok"] is True
    assert result["computed_tokens"] == 256
    assert request.num_computed_tokens == 256
    assert request.num_cached_tokens == 256
    assert [block.block_id for block in coordinator.req_to_blocks[request.request_id]] == [10, 12]
    assert result["destination_blocks"] == [10, 12]
    assert result["reused_published_blocks"] == [True, False]
    assert [block.ref_cnt for block in pool.blocks] == [1, 0, 1]
    assert [block.block_hash for block in pool.blocks] == [
        target_packed_hashes[0],
        None,
        target_packed_hashes[1],
    ]
    assert [value["packed_hex"] for value in result["block_hashes"]] == [value.hex() for value in target_packed_hashes]
    assert reservation.state.value == "attached"


def test_pd_hash_mismatch_rejects_requests_with_extra_cache_semantics() -> None:
    """带 cache salt 的请求不能仅凭 token 指纹接受不同的 P/D 内部 hash。"""

    source_hash = b"pd-source-block-00000000000001"
    target_hash = b"pd-target-block-00000000000001"
    token_ids = list(range(128))
    request = types.SimpleNamespace(
        block_hashes=[target_hash],
        prompt_token_ids=token_ids,
        cache_salt="tenant-a",
    )
    source_fingerprints = tuple(integration._token_block_fingerprints(token_ids, 128))
    block_groups = (
        {
            "logical_index": 0,
            "group_id": 0,
            "block_hash": {
                "external_hash": str(source_hash),
            },
        },
    )

    with pytest.raises(RuntimeError, match="P export hash does not match"):
        integration._validate_pd_target_request_hashes(
            request,
            (target_hash,),
            block_groups,
            source_fingerprints,
            semantic_plain_text=True,
            block_size=128,
        )


def test_pd_scheduler_attach_uses_partial_block_without_publishing_it(
    monkeypatch,
) -> None:
    """D attach 应使用 44-token partial KV，但只发布前两个完整 block。"""

    target_hashes = (b"target-full-0" * 3, b"target-full-1" * 3)

    class Block:
        def __init__(self, block_id):
            self.block_id = block_id
            self.ref_cnt = 0
            self.block_hash = None

    class HashMap:
        def __init__(self):
            self._cache = {}

        def insert(self, block_hash, block):
            self._cache[block_hash] = block

        def pop(self, block_hash, _block_id):
            return self._cache.pop(block_hash, None)

    class BlockPool:
        def __init__(self):
            self.blocks = [Block(10), Block(11), Block(12)]
            self.free = list(self.blocks)
            self.enable_caching = True
            self.cached_block_hash_to_block = HashMap()

        def get_num_free_blocks(self):
            return len(self.free)

        def get_new_blocks(self, count):
            selected = self.free[:count]
            del self.free[:count]
            for block in selected:
                block.ref_cnt += 1
            return selected

        def touch(self, selected):
            for block in selected:
                if block.ref_cnt == 0:
                    self.free.remove(block)
                block.ref_cnt += 1

        def free_blocks(self, selected):
            for block in selected:
                block.ref_cnt -= 1
                if block.ref_cnt == 0 and block not in self.free:
                    self.free.append(block)

    class Coordinator:
        def __init__(self, pool):
            self.pool = pool
            self.req_to_blocks = {}

        def allocate_new_computed_blocks(
            self,
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
            num_external_computed_tokens,
        ):
            assert num_local_computed_tokens == 256
            assert num_external_computed_tokens == 44
            blocks = list(new_computed_blocks[0])
            for block in blocks:
                block.ref_cnt += 1
            self.req_to_blocks[request_id] = blocks

        def free(self, request_id):
            self.pool.free_blocks(self.req_to_blocks.pop(request_id, []))

    pool = BlockPool()
    coordinator = Coordinator(pool)
    kv_cache_manager = types.SimpleNamespace(
        num_kv_cache_groups=1,
        coordinator=coordinator,
        free=lambda request: coordinator.free(request.request_id),
    )
    token_ids = list(range(300))
    request = types.SimpleNamespace(
        request_id="decode-partial-1",
        block_hashes=list(target_hashes),
        prompt_token_ids=token_ids,
        num_computed_tokens=0,
        num_cached_tokens=-1,
        num_tokens=300,
        cache_salt=None,
        lora_request=None,
        mm_features=None,
        prompt_embeds=None,
    )
    scheduler = types.SimpleNamespace(
        requests={request.request_id: request},
        running=[],
        kv_cache_manager=kv_cache_manager,
        block_size=128,
    )
    reservation_manager = integration.VllmBlockPoolReservationManager(lambda: pool)
    monkeypatch.setattr(
        integration,
        "_PROCESS_BLOCK_RESERVATIONS",
        reservation_manager,
    )
    full_fingerprints = tuple(integration._token_block_fingerprints(token_ids, 128))
    partial_fingerprint = integration._partial_token_fingerprint(token_ids, 128)
    packed_hashes = tuple(value + b"\x00\x00\x00\x00" for value in target_hashes)
    cached_blocks = pool.get_new_blocks(2)
    for block_hash, block in zip(packed_hashes, cached_blocks, strict=True):
        block.block_hash = block_hash
        pool.cached_block_hash_to_block.insert(block_hash, block)
    pool.free_blocks(cached_blocks)
    block_groups = (
        {
            "logical_index": 0,
            "group_id": 0,
            "block_hash": {"external_hash": str(target_hashes[0])},
        },
        {
            "logical_index": 1,
            "group_id": 0,
            "block_hash": {"external_hash": str(target_hashes[1])},
        },
        {
            "logical_index": 2,
            "group_id": 0,
            "request_scoped_partial": True,
            "valid_token_count": 44,
            "block_hash": None,
        },
    )
    plan = integration._scheduler_plan_pd_import(
        scheduler,
        integration._PDTargetPlanTask(
            request_id=request.request_id,
            handoff_id="pd-partial-attach-1",
            block_groups=block_groups,
            token_block_fingerprints=full_fingerprints,
            partial_token_fingerprint=partial_fingerprint,
            partial_valid_tokens=44,
            semantic_plain_text=True,
            reservation_ttl_ms=120_000,
            timeout_seconds=1.0,
        ),
    )
    task = integration._PDTargetAttachTask(
        request_id=request.request_id,
        handoff_id="pd-partial-attach-1",
        reservation_id=plan["reservation"]["reservation_id"],
        block_hashes=(
            {"packed_hex": packed_hashes[0].hex()},
            {"packed_hex": packed_hashes[1].hex()},
            {
                "request_scoped_partial": True,
                "logical_index": 2,
                "group_id": 0,
                "valid_token_count": 44,
            },
        ),
        block_groups=block_groups,
        timeout_seconds=1.0,
        token_block_fingerprints=full_fingerprints,
        partial_token_fingerprint=partial_fingerprint,
        partial_valid_tokens=44,
        semantic_plain_text=True,
    )

    result = integration._scheduler_attach_pd_request(scheduler, task)

    assert result["computed_tokens"] == 299
    assert result["partial_valid_tokens"] == 44
    assert plan["reused_blocks"] == 2
    assert plan["transfer_blocks"] == 1
    assert plan["transfer_positions"] == [2]
    assert request.num_computed_tokens == 299
    assert result["published_destination_blocks"] == []
    assert result["reused_block_count"] == 2
    assert result["transferred_block_count"] == 1
    assert result["request_scoped_destination_blocks"] == [12]
    assert pool.blocks[2].block_hash is None
    assert len(pool.cached_block_hash_to_block._cache) == 2


def test_block_reservation_proxy_timeout_exceeds_scheduler_budget() -> None:
    """内部 HTTP 包络超时应晚于 scheduler attach 自身的截止时间。"""

    assert integration._block_reservation_proxy_timeout_seconds({}) == 10.0
    assert integration._block_reservation_proxy_timeout_seconds({"timeout_ms": 10_000}) == 11.0
    assert integration._block_reservation_proxy_timeout_seconds({"timeout_ms": 30_000}) == 31.0


def test_pd_import_orchestrates_reserve_hixl_attach_and_source_release(monkeypatch) -> None:
    """D import 应同步 attach，并在 ACK 后异步推进状态和释放 P。"""

    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()
    handle = backend.register_cache(
        LLMDataDistCacheDesc(
            placement="device",
            data_type="bf16",
            shape=(128, 8, 128),
            num_tensors=2,
        )
    )
    control = types.SimpleNamespace(llm_datadist_backend=backend)
    monkeypatch.setattr(integration, "_PROCESS_PD_HANDOFFS", None)
    monkeypatch.setattr(integration, "_ENGINE_CORE_DATAPLANE", control)
    monkeypatch.setattr(integration, "_ENGINE_CORE_LLM_DATADIST_CACHE_ID", handle.cache_id)
    monkeypatch.setattr(
        integration,
        "_async_metadata_publisher",
        lambda: types.SimpleNamespace(submit=lambda _task: {"queued": True, "queued_count": 1}),
    )
    reconciliations = []
    monkeypatch.setattr(
        integration,
        "_PD_RECONCILE_EXECUTOR",
        types.SimpleNamespace(submit=reconciliations.append),
    )
    dispatches = []

    def dispatch(path, payload):
        dispatches.append((path, payload))
        if path == "/datadist/pd/import/prepare/local":
            return {
                "ok": True,
                "created": False,
                "gate": {"request_seen": True},
            }
        if path == "/datadist/pd/import/plan/local":
            return {
                "ok": True,
                "planned_destination_blocks": [20, 21, 101],
                "transfer_positions": [2],
                "reservation": {
                    "reservation_id": "destination-reservation-1",
                    "block_ids": [101],
                },
            }
        if path == "/datadist/reservations/renew":
            return {"ok": True}
        if path == "/datadist/pd/import/attach/local":
            return {
                "ok": True,
                "request_id": "decode-request-2",
                "destination_blocks": [20, 21, 101],
                "computed_tokens": 384,
                "reused_published_blocks": [True, True, False],
                "reservation": {"state": "attached"},
            }
        raise AssertionError(path)

    remote_requests = []

    def remote(_base_url, path, payload):
        remote_requests.append((path, payload))
        if path == "/datadist/pd/export/release":
            return {"ok": True, "handoff": {"state": "source_released"}}
        return {"ok": True, "lease": {"state": "active"}}

    monkeypatch.setattr(integration, "_dispatch_block_pool_request", dispatch)
    monkeypatch.setattr(integration, "_post_remote_control", remote)
    integration._handle_engine_core_datadist_request(
        "/datadist/pd/handoffs/begin",
        {
            "handoff_id": "pd-import-1",
            "request_id": "decode-request-2",
            "source_worker_id": "prefill-0",
            "target_worker_id": "decode-0",
            "model_id": "model-a",
            "prompt_token_count": 384,
        },
    )
    external_hashes = [
        b"pd-import-block-0000000000000001",
        b"pd-import-block-0000000000000002",
        b"pd-import-block-0000000000000003",
    ]
    packed_hashes = [value + (0).to_bytes(4, "big") for value in external_hashes]

    hash_pairs = list(zip(packed_hashes, external_hashes, strict=True))
    result = integration._handle_engine_core_datadist_request(
        "/datadist/pd/import",
        {
            "handoff_id": "pd-import-1",
            "request_id": "decode-request-2",
            "source_control_url": "http://10.0.0.1:18011",
            "remote_cluster_id": 1,
            "source_export": {
                "request_id": "decode-request-2",
                "source_blocks": [7, 8, 9],
                "source_lease_id": "source-lease-pd-1",
                "block_hashes": [
                    _hash_descriptor(packed_hash, external_hash) for packed_hash, external_hash in hash_pairs
                ],
                "block_groups": [
                    _source_block_group(index, _hash_descriptor(packed_hash, external_hash))
                    for index, (packed_hash, external_hash) in enumerate(hash_pairs)
                ],
            },
        },
    )

    assert result["ok"] is True
    assert result["handoff"]["state"] == "decode_accepted"
    assert result["remote_reconciliation"] == {"queued": True}
    assert result["destination_blocks"] == [20, 21, 101]
    assert result["reused_block_count"] == 2
    assert result["transferred_block_count"] == 1
    assert result["transfer_positions"] == [2]
    assert backend.pull_requests[0][1].prompt_blocks == (9,)
    assert backend.pull_requests[0][1].decoder_blocks == (101,)
    assert [path for path, _payload in dispatches] == [
        "/datadist/pd/import/prepare/local",
        "/datadist/pd/import/plan/local",
        "/datadist/reservations/renew",
        "/datadist/pd/import/attach/local",
    ]
    assert remote_requests == []

    reconciliations[0]()

    assert integration._process_pd_handoffs().get("pd-import-1").state.value == ("source_released")
    assert "/datadist/pd/export/release" in [path for path, _payload in remote_requests]


def test_pd_import_uses_scheduler_owned_blocks_without_reservation_guard(
    monkeypatch,
) -> None:
    """Scheduler所有权模式应直接向请求block执行HIXL pull并跳过外部续租。"""

    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()
    handle = backend.register_cache(
        LLMDataDistCacheDesc(
            placement="device",
            data_type="bf16",
            shape=(128, 8, 128),
            num_tensors=2,
        )
    )
    control = types.SimpleNamespace(llm_datadist_backend=backend)
    monkeypatch.setattr(integration, "_PROCESS_PD_HANDOFFS", None)
    monkeypatch.setattr(integration, "_ENGINE_CORE_DATAPLANE", control)
    monkeypatch.setattr(integration, "_ENGINE_CORE_LLM_DATADIST_CACHE_ID", handle.cache_id)
    monkeypatch.setattr(
        integration,
        "_TargetBlockReservationGuard",
        lambda *_args, **_kwargs: pytest.fail("scheduler-owned import must not create a reservation guard"),
    )
    monkeypatch.setattr(
        integration,
        "_async_metadata_publisher",
        lambda: types.SimpleNamespace(submit=lambda _task: {"queued": True, "queued_count": 1}),
    )
    reconciliations = []
    monkeypatch.setattr(
        integration,
        "_PD_RECONCILE_EXECUTOR",
        types.SimpleNamespace(submit=reconciliations.append),
    )
    dispatches = []

    def dispatch(path, payload):
        dispatches.append((path, payload))
        if path == "/datadist/pd/import/prepare/local":
            return {"ok": True, "created": False, "gate": {"request_seen": True}}
        if path == "/datadist/pd/import/plan/local":
            return {
                "ok": True,
                "scheduler_owned": True,
                "planned_destination_blocks": [20, 21, 101],
                "transfer_positions": [2],
                "reservation": {
                    "reservation_id": "scheduler:pd-scheduler-import-1",
                    "block_ids": [101],
                },
            }
        if path == "/datadist/pd/import/attach/local":
            assert payload["ownership_mode"] == "scheduler"
            return {
                "ok": True,
                "request_id": "decode-request-scheduler",
                "destination_blocks": [20, 21, 101],
                "computed_tokens": 384,
                "reused_published_blocks": [True, True, False],
                "published_destination_blocks": [101],
                "published_block_hashes": [payload["block_hashes"][2]],
            }
        raise AssertionError(path)

    remote_requests = []

    def remote(_base_url, path, payload):
        remote_requests.append((path, payload))
        return {"ok": True}

    monkeypatch.setattr(integration, "_dispatch_block_pool_request", dispatch)
    monkeypatch.setattr(integration, "_post_remote_control", remote)
    integration._handle_engine_core_datadist_request(
        "/datadist/pd/handoffs/begin",
        {
            "handoff_id": "pd-scheduler-import-1",
            "request_id": "decode-request-scheduler",
            "source_worker_id": "prefill-0",
            "target_worker_id": "decode-0",
            "model_id": "model-a",
            "prompt_token_count": 384,
        },
    )
    external_hashes = [
        b"pd-scheduler-block-0000000000001",
        b"pd-scheduler-block-0000000000002",
        b"pd-scheduler-block-0000000000003",
    ]
    packed_hashes = [value + (0).to_bytes(4, "big") for value in external_hashes]
    source_export = {
        "request_id": "decode-request-scheduler",
        "source_blocks": [7, 8, 9],
        "source_lease_id": "source-lease-scheduler-1",
        "block_hashes": [
            _hash_descriptor(packed_hash, external_hash)
            for packed_hash, external_hash in zip(packed_hashes, external_hashes, strict=True)
        ],
    }
    source_export["block_groups"] = [
        _source_block_group(index, block_hash) for index, block_hash in enumerate(source_export["block_hashes"])
    ]

    result = integration._handle_engine_core_datadist_request(
        "/datadist/pd/import",
        {
            "handoff_id": "pd-scheduler-import-1",
            "request_id": "decode-request-scheduler",
            "source_control_url": "http://10.0.0.1:18011",
            "remote_cluster_id": 1,
            "source_export": source_export,
        },
    )

    assert result["ok"] is True
    assert result["destination_blocks"] == [20, 21, 101]
    assert result["transfer_positions"] == [2]
    assert backend.pull_requests[0][1].prompt_blocks == (9,)
    assert backend.pull_requests[0][1].decoder_blocks == (101,)
    assert [path for path, _payload in dispatches] == [
        "/datadist/pd/import/prepare/local",
        "/datadist/pd/import/plan/local",
        "/datadist/pd/import/attach/local",
    ]
    assert reconciliations
    assert remote_requests == []


def test_engine_core_datadist_control_links_pulls_and_unlinks(monkeypatch) -> None:
    """engine-core 控制入口应驱动 backend 完成建链、block pull 和断链。"""

    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()
    handle = backend.register_cache(
        LLMDataDistCacheDesc(
            placement="device",
            data_type="bf16",
            shape=(16, 8, 128),
            num_tensors=2,
        )
    )
    control = types.SimpleNamespace(llm_datadist_backend=backend)
    monkeypatch.setattr(integration, "_ENGINE_CORE_DATAPLANE", control)
    monkeypatch.setattr(integration, "_ENGINE_CORE_LLM_DATADIST_CACHE_ID", handle.cache_id)
    cluster_fields = {
        "remote_cluster_id": 9,
        "local_ip": "10.0.0.1",
        "local_port": 26000,
        "remote_ip": "10.0.0.2",
        "remote_port": 26001,
    }

    linked = integration._handle_engine_core_datadist_request(
        "/datadist/link",
        cluster_fields,
    )
    linked_again = integration._handle_engine_core_datadist_request(
        "/datadist/link",
        cluster_fields,
    )
    pulled = integration._handle_engine_core_datadist_request(
        "/datadist/pull",
        {
            "remote_cluster_id": 9,
            "source_blocks": [1, 2],
            "destination_blocks": [11, 12],
            "layer_range": [0, 1],
        },
    )
    unlinked = integration._handle_engine_core_datadist_request(
        "/datadist/unlink",
        {**cluster_fields, "force": True},
    )

    assert linked == {"ok": True, "remote_cluster_id": 9}
    assert linked_again == {
        "ok": True,
        "remote_cluster_id": 9,
        "already_linked": True,
    }
    assert pulled["pulled_count"] == 2
    assert backend.pull_requests[0][0] == handle.cache_id
    assert backend.pull_requests[0][1].prompt_blocks == (1, 2)
    assert backend.pull_requests[0][1].decoder_blocks == (11, 12)
    assert unlinked == {"ok": True, "remote_cluster_id": 9}


def test_engine_core_datadist_transaction_reserves_pulls_and_publishes(
    monkeypatch,
) -> None:
    """事务 pull 应使用预留目标 ID，并在传输后发布 hash 和 NPU 位置。"""

    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()
    handle = backend.register_cache(
        LLMDataDistCacheDesc(
            placement="device",
            data_type="bf16",
            shape=(16, 8, 128),
            num_tensors=2,
        )
    )
    registered: list[tuple[int, str, int, bool]] = []
    relayed: list[tuple[list[object], bool]] = []
    requests: list[tuple[str, dict[str, object]]] = []
    packed_hashes = [
        b"a" * 32 + (0).to_bytes(4, "big"),
        b"b" * 32 + (0).to_bytes(4, "big"),
    ]

    class SourceLease:
        """测试用远端源 lease guard。"""

        def __init__(self, url: str, block_ids: list[int], *, ttl_ms: int) -> None:
            assert url == "http://source:18011"
            assert block_ids == [1, 2]
            assert ttl_ms == 30_000

        @staticmethod
        def acquire() -> dict[str, object]:
            return {
                "lease_id": "source-lease-1",
                "block_hashes": [
                    {"packed_hex": packed_hashes[0].hex()},
                    {"packed_hex": packed_hashes[1].hex()},
                ],
            }

        @staticmethod
        def ensure_valid() -> None:
            return None

        @staticmethod
        def renew_now() -> None:
            return None

        @staticmethod
        def release() -> dict[str, object]:
            return {"ok": True, "lease": {"state": "released"}}

    def dispatch(path: str, payload: dict[str, object]) -> dict[str, object]:
        requests.append((path, payload))
        if path == "/datadist/blocks/metadata":
            return {
                "ok": True,
                "blocks": [
                    _source_lease_block(block_id, block_hash)
                    for block_id, block_hash in zip([11, 12], packed_hashes, strict=True)
                ],
            }
        if path == "/datadist/source-leases/acquire":
            return {
                "ok": True,
                "lease": {
                    "lease_id": "target-metadata-lease-1",
                    "block_hashes": [{"packed_hex": packed_hash.hex()} for packed_hash in packed_hashes],
                },
            }
        if path == "/datadist/source-leases/release":
            return {"ok": True, "lease": {"state": "released"}}
        if path.endswith("/reserve"):
            return {
                "ok": True,
                "reservation": {
                    "reservation_id": "reservation-1",
                    "block_ids": [11, 12],
                    "state": "reserved",
                },
            }
        if path.endswith("/renew"):
            return {
                "ok": True,
                "reservation": {
                    "reservation_id": "reservation-1",
                    "block_ids": [11, 12],
                    "state": "reserved",
                },
            }
        assert path.endswith("/publish")
        return {
            "ok": True,
            "reservation": {
                "reservation_id": "reservation-1",
                "block_ids": [11, 12],
                "state": "published",
            },
        }

    def register_batch(
        block_ids: list[int],
        block_hashes: list[str],
        *,
        token_count: int,
        active: bool,
    ) -> dict[str, object]:
        registered.extend(
            (block_id, block_hash, token_count, active)
            for block_id, block_hash in zip(block_ids, block_hashes, strict=True)
        )
        return {
            "event_type": "store",
            "location": "npu",
            "block_hashes": list(block_hashes),
            "token_count": token_count,
            "engine_specific": {"physical_block_ids": list(block_ids)},
        }

    control = types.SimpleNamespace(
        llm_datadist_backend=backend,
        config=types.SimpleNamespace(block_size=16),
        register_published_physical_blocks=register_batch,
        register_physical_block=lambda *_args, **_kwargs: pytest.fail(
            "successful batch registration must not use per-block fallback"
        ),
    )
    monkeypatch.setattr(integration, "_ENGINE_CORE_DATAPLANE", control)
    monkeypatch.setattr(integration, "_ENGINE_CORE_LLM_DATADIST_CACHE_ID", handle.cache_id)
    monkeypatch.setattr(integration, "_dispatch_block_pool_request", dispatch)
    monkeypatch.setattr(integration, "_RemoteSourceLeaseGuard", SourceLease)
    monkeypatch.setattr(
        integration,
        "_PROCESS_TRANSFER_TRANSACTIONS",
        integration.TransferTransactionManager(max_inflight=1),
    )
    monkeypatch.setattr(
        integration,
        "_enqueue_runtime_events",
        lambda events, *, include_store=False: relayed.append((list(events), include_store)),
    )
    payload = {
        "transfer_id": "transfer-success-1",
        "remote_cluster_id": 9,
        "source_control_url": "http://source:18011",
        "source_blocks": [1, 2],
        "block_hashes": [
            {"packed_hex": packed_hashes[0].hex()},
            {"packed_hex": packed_hashes[1].hex()},
        ],
        "layer_range": [0, 1],
    }
    result = integration._handle_engine_core_datadist_request(
        "/datadist/transfers/pull",
        payload,
    )
    assert integration._wait_for_async_metadata_publish()
    replay = integration._handle_engine_core_datadist_request(
        "/datadist/transfers/pull",
        payload,
    )

    assert result["ok"] is True
    assert result["destination_blocks"] == [11, 12]
    assert result["metadata_publish"]["mode"] == "async"
    assert result["source_lease"]["lease_id"] == "source-lease-1"
    assert result["source_lease"]["release"]["ok"] is True
    assert {
        "lease_acquire",
        "reserve",
        "hixl_pull",
        "lease_renew",
        "reservation_renew",
        "publish",
        "metadata_register",
        "metadata_event_enqueue",
        "metadata_publish",
        "lease_release",
        "worker_dataplane_total",
        "worker_total",
    }.issubset(result["phase_ms"])
    assert all(value >= 0 for value in result["phase_ms"].values())
    assert backend.pull_requests[0][1].prompt_blocks == (1, 2)
    assert backend.pull_requests[0][1].decoder_blocks == (11, 12)
    assert requests[2][0] == "/datadist/reservations/publish"
    assert requests[2][1]["block_hashes"] == [
        {"packed_hex": packed_hashes[0].hex()},
        {"packed_hex": packed_hashes[1].hex()},
    ]
    assert [item[0] for item in registered] == [11, 12]
    assert all(item[2:] == (16, False) for item in registered)
    assert relayed[0][1] is True
    assert len(relayed[0][0]) == 1
    assert relayed[0][0][0]["block_hashes"] == [
        str(integration._external_block_hash_string(packed_hashes[0][:-4])),
        str(integration._external_block_hash_string(packed_hashes[1][:-4])),
    ]
    assert result["transfer"]["state"] == "completed"
    assert replay["ok"] is True
    assert replay["idempotent_replay"] is True
    assert replay["transfer"]["state"] == "completed"
    assert len(backend.pull_requests) == 1
    assert len(requests) == 6
    assert requests[3] == (
        "/datadist/blocks/metadata",
        {"block_ids": [11, 12], "allow_missing": True},
    )
    assert requests[4][0] == "/datadist/source-leases/acquire"
    assert requests[5] == (
        "/datadist/source-leases/release",
        {"lease_id": "target-metadata-lease-1"},
    )


def test_engine_core_datadist_duplicate_publish_is_successful_skip(
    monkeypatch,
) -> None:
    """目标 hash 已被并发发布时应回滚冗余预留并按成功跳过。"""

    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()
    handle = backend.register_cache(
        LLMDataDistCacheDesc(
            placement="device",
            data_type="bf16",
            shape=(16, 8, 128),
            num_tensors=2,
        )
    )
    requests: list[str] = []
    packed_hash = b"d" * 32 + (0).to_bytes(4, "big")

    def dispatch(path: str, payload: dict[str, object]) -> dict[str, object]:
        requests.append(path)
        if path.endswith("/reserve"):
            return {
                "ok": True,
                "reservation": {
                    "reservation_id": "reservation-duplicate",
                    "block_ids": [71],
                    "state": "reserved",
                },
            }
        if path.endswith("/renew"):
            return {
                "ok": True,
                "reservation": {
                    "reservation_id": "reservation-duplicate",
                    "block_ids": [71],
                    "state": "reserved",
                },
            }
        assert path.endswith("/publish")
        return {
            "ok": True,
            "skipped": True,
            "reason": "target_already_published",
            "already_published": True,
            "existing_block_ids": [10],
            "reservation": {
                "reservation_id": "reservation-duplicate",
                "block_ids": [71],
                "state": "rolled_back",
            },
        }

    class SourceLease:
        """测试用已持有源 block lease。"""

        @staticmethod
        def renew_now() -> None:
            """模拟源 lease 续租成功。

            返回:
                无返回值。
            """

    control = types.SimpleNamespace(
        config=types.SimpleNamespace(block_size=16),
        register_physical_block=lambda *_args, **_kwargs: pytest.fail(
            "duplicate publish must not register discarded destination blocks"
        ),
    )
    transactions = integration.TransferTransactionManager(max_inflight=1)
    transactions.begin(
        transfer_id="transfer-duplicate",
        request_fingerprint="duplicate",
        source_blocks=[11],
    )
    monkeypatch.setattr(integration, "_ENGINE_CORE_LLM_DATADIST_CACHE_ID", handle.cache_id)
    monkeypatch.setattr(integration, "_PROCESS_TRANSFER_TRANSACTIONS", transactions)
    monkeypatch.setattr(integration, "_dispatch_block_pool_request", dispatch)

    result = integration._pull_leased_blocks_and_publish(
        control,
        backend,
        {
            "remote_cluster_id": 9,
            "source_blocks": [11],
        },
        source_blocks=[11],
        hash_descriptors=[{"packed_hex": packed_hash.hex()}],
        source_lease=SourceLease(),
        transfer_id="transfer-duplicate",
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "target_already_published"
    assert result["published_count"] == 0
    assert result["destination_blocks"] == []
    assert result["discarded_destination_blocks"] == [71]
    assert result["existing_block_ids"] == [10]
    assert backend.pull_requests[0][1].decoder_blocks == (71,)
    assert requests == [
        "/datadist/reservations/reserve",
        "/datadist/reservations/renew",
        "/datadist/reservations/publish",
    ]


def test_engine_core_datadist_transaction_rolls_back_failed_pull(monkeypatch) -> None:
    """DataDist pull 失败时应归还目标 reservation，且不执行 publish。"""

    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()
    handle = backend.register_cache(
        LLMDataDistCacheDesc(
            placement="device",
            data_type="bf16",
            shape=(16, 8, 128),
            num_tensors=2,
        )
    )
    requests: list[str] = []
    packed_hash = b"a" * 32 + (0).to_bytes(4, "big")

    class SourceLease:
        """测试用远端源 lease guard。"""

        def __init__(self, url: str, block_ids: list[int], *, ttl_ms: int) -> None:
            return None

        @staticmethod
        def acquire() -> dict[str, object]:
            return {
                "lease_id": "source-lease-2",
                "block_hashes": [{"packed_hex": packed_hash.hex()}],
            }

        @staticmethod
        def ensure_valid() -> None:
            return None

        @staticmethod
        def renew_now() -> None:
            return None

        @staticmethod
        def release() -> dict[str, object]:
            return {"ok": True, "lease": {"state": "released"}}

    def failed_pull(cache_id: int, mapping: LLMDataDistBlockMapping) -> None:
        raise LLMDataDistBackendError("injected pull failure")

    def dispatch(path: str, payload: dict[str, object]) -> dict[str, object]:
        requests.append(path)
        if path.endswith("/reserve"):
            return {
                "ok": True,
                "reservation": {
                    "reservation_id": "reservation-2",
                    "block_ids": [13],
                    "state": "reserved",
                },
            }
        assert path.endswith("/rollback")
        return {
            "ok": True,
            "reservation": {
                "reservation_id": "reservation-2",
                "block_ids": [13],
                "state": "rolled_back",
            },
        }

    control = types.SimpleNamespace(
        llm_datadist_backend=backend,
        config=types.SimpleNamespace(block_size=16),
    )
    monkeypatch.setattr(backend, "pull_blocks", failed_pull)
    monkeypatch.setattr(integration, "_ENGINE_CORE_DATAPLANE", control)
    monkeypatch.setattr(integration, "_ENGINE_CORE_LLM_DATADIST_CACHE_ID", handle.cache_id)
    monkeypatch.setattr(integration, "_dispatch_block_pool_request", dispatch)
    monkeypatch.setattr(integration, "_RemoteSourceLeaseGuard", SourceLease)

    result = integration._handle_engine_core_datadist_request(
        "/datadist/transfers/pull",
        {
            "remote_cluster_id": 9,
            "source_control_url": "http://source:18011",
            "source_blocks": [1],
            "block_hashes": [{"packed_hex": packed_hash.hex()}],
        },
    )

    assert result["ok"] is False
    assert "injected pull failure" in result["error"]
    assert result["rollback"]["ok"] is True
    assert result["source_lease"]["release"]["ok"] is True
    assert requests == [
        "/datadist/reservations/reserve",
        "/datadist/reservations/rollback",
    ]


def test_engine_core_datadist_cancel_rolls_back_before_publish(monkeypatch) -> None:
    """HIXL 阻塞期间的取消应可查询，并在 publish 前回滚目标 block。"""

    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()
    handle = backend.register_cache(
        LLMDataDistCacheDesc(
            placement="device",
            data_type="bf16",
            shape=(16, 8, 128),
            num_tensors=2,
        )
    )
    pull_started = threading.Event()
    allow_pull_to_return = threading.Event()
    requests: list[str] = []
    results: list[dict[str, object]] = []
    packed_hash = b"c" * 32 + (0).to_bytes(4, "big")

    class SourceLease:
        """测试用远端源 lease guard。"""

        def __init__(self, url: str, block_ids: list[int], *, ttl_ms: int) -> None:
            return None

        @staticmethod
        def acquire() -> dict[str, object]:
            return {
                "lease_id": "source-lease-cancel",
                "block_hashes": [{"packed_hex": packed_hash.hex()}],
            }

        @staticmethod
        def renew_now() -> None:
            return None

        @staticmethod
        def release() -> dict[str, object]:
            return {"ok": True, "lease": {"state": "released"}}

    def blocking_pull(cache_id: int, mapping: LLMDataDistBlockMapping) -> None:
        pull_started.set()
        assert allow_pull_to_return.wait(timeout=2.0)

    def dispatch(path: str, payload: dict[str, object]) -> dict[str, object]:
        requests.append(path)
        if path.endswith("/reserve"):
            return {
                "ok": True,
                "reservation": {
                    "reservation_id": "reservation-cancel",
                    "block_ids": [14],
                    "state": "reserved",
                },
            }
        assert path.endswith("/rollback")
        return {
            "ok": True,
            "reservation": {
                "reservation_id": "reservation-cancel",
                "block_ids": [14],
                "state": "rolled_back",
            },
        }

    control = types.SimpleNamespace(
        llm_datadist_backend=backend,
        config=types.SimpleNamespace(block_size=16),
    )
    monkeypatch.setattr(backend, "pull_blocks", blocking_pull)
    monkeypatch.setattr(integration, "_ENGINE_CORE_DATAPLANE", control)
    monkeypatch.setattr(integration, "_ENGINE_CORE_LLM_DATADIST_CACHE_ID", handle.cache_id)
    monkeypatch.setattr(integration, "_dispatch_block_pool_request", dispatch)
    monkeypatch.setattr(integration, "_RemoteSourceLeaseGuard", SourceLease)
    monkeypatch.setattr(
        integration,
        "_PROCESS_TRANSFER_TRANSACTIONS",
        integration.TransferTransactionManager(max_inflight=1),
    )

    payload = {
        "transfer_id": "transfer-cancel-1",
        "remote_cluster_id": 9,
        "source_control_url": "http://source:18011",
        "source_blocks": [1],
    }
    thread = threading.Thread(
        target=lambda: results.append(
            integration._handle_engine_core_datadist_request(
                "/datadist/transfers/pull",
                payload,
            )
        )
    )
    thread.start()
    assert pull_started.wait(timeout=2.0)

    status = integration._handle_engine_core_datadist_request(
        "/datadist/transfers/status",
        {"transfer_id": "transfer-cancel-1"},
    )
    cancelled = integration._handle_engine_core_datadist_request(
        "/datadist/transfers/cancel",
        {"transfer_id": "transfer-cancel-1"},
    )
    allow_pull_to_return.set()
    thread.join(timeout=2.0)

    assert status["transfer"]["state"] == "transferring"
    assert cancelled["accepted"] is True
    assert thread.is_alive() is False
    assert results[0]["ok"] is False
    assert results[0]["cancelled"] is True
    assert results[0]["transfer"]["state"] == "cancelled"
    assert requests == [
        "/datadist/reservations/reserve",
        "/datadist/reservations/rollback",
    ]


def test_engine_core_datadist_transaction_rejects_lost_source_lease(
    monkeypatch,
) -> None:
    """pull 完成前 lease 丢失时应回滚目标 block，不能执行 publish。"""

    backend = FakeLLMDataDistBackend(_fake_config())
    backend.initialize()
    handle = backend.register_cache(
        LLMDataDistCacheDesc(
            placement="device",
            data_type="bf16",
            shape=(16, 8, 128),
            num_tensors=2,
        )
    )
    packed_hash = b"a" * 32 + (0).to_bytes(4, "big")
    requests: list[str] = []

    class LostSourceLease:
        """模拟传输期间续租失败的源 lease guard。"""

        def __init__(self, url: str, block_ids: list[int], *, ttl_ms: int) -> None:
            return None

        @staticmethod
        def acquire() -> dict[str, object]:
            return {
                "lease_id": "lost-source-lease",
                "block_hashes": [{"packed_hex": packed_hash.hex()}],
            }

        @staticmethod
        def renew_now() -> None:
            raise RuntimeError("source block lease lost during transfer")

        @staticmethod
        def release() -> dict[str, object]:
            return {"ok": True, "lease": {"state": "expired"}}

    def dispatch(path: str, payload: dict[str, object]) -> dict[str, object]:
        requests.append(path)
        if path.endswith("/reserve"):
            return {
                "ok": True,
                "reservation": {
                    "reservation_id": "reservation-lost",
                    "block_ids": [14],
                    "state": "reserved",
                },
            }
        assert path.endswith("/rollback")
        return {
            "ok": True,
            "reservation": {
                "reservation_id": "reservation-lost",
                "block_ids": [14],
                "state": "rolled_back",
            },
        }

    control = types.SimpleNamespace(
        llm_datadist_backend=backend,
        config=types.SimpleNamespace(block_size=16),
    )
    monkeypatch.setattr(integration, "_ENGINE_CORE_DATAPLANE", control)
    monkeypatch.setattr(integration, "_ENGINE_CORE_LLM_DATADIST_CACHE_ID", handle.cache_id)
    monkeypatch.setattr(integration, "_dispatch_block_pool_request", dispatch)
    monkeypatch.setattr(integration, "_RemoteSourceLeaseGuard", LostSourceLease)

    result = integration._handle_engine_core_datadist_request(
        "/datadist/transfers/pull",
        {
            "remote_cluster_id": 9,
            "source_control_url": "http://source:18011",
            "source_blocks": [1],
        },
    )

    assert result["ok"] is False
    assert "lease lost" in result["error"]
    assert requests == [
        "/datadist/reservations/reserve",
        "/datadist/reservations/rollback",
    ]


def test_block_metadata_exports_exact_packed_hash(monkeypatch) -> None:
    """源端 metadata 接口应无损导出 group ID 在内的 vLLM 完整 hash。"""

    packed_hash = b"a" * 32 + (7).to_bytes(4, "big")
    block = types.SimpleNamespace(block_hash=packed_hash)
    pool = types.SimpleNamespace(blocks=[block])
    monkeypatch.setattr(integration, "_latest_reservable_block_pool", lambda: pool)

    result = integration._local_block_metadata({"block_ids": [0]})

    descriptor = result["blocks"][0]["block_hash"]
    assert result["ok"] is True
    assert descriptor["packed_hex"] == packed_hash.hex()
    assert descriptor["group_id"] == 7
    assert integration._decode_packed_block_hash(descriptor) == packed_hash


def test_block_metadata_allows_async_validation_to_observe_reused_blocks(
    monkeypatch,
) -> None:
    """异步校验应把已经释放的 block 返回为空，而不是让整个批次失败。"""

    packed_hash = b"a" * 32 + (0).to_bytes(4, "big")
    pool = types.SimpleNamespace(
        blocks=[
            types.SimpleNamespace(block_hash=packed_hash),
            types.SimpleNamespace(block_hash=None),
        ]
    )
    monkeypatch.setattr(integration, "_latest_reservable_block_pool", lambda: pool)

    result = integration._local_block_metadata({"block_ids": [0, 1, 99], "allow_missing": True})

    assert result["ok"] is True
    assert result["blocks"][0]["block_hash"]["packed_hex"] == packed_hash.hex()
    assert result["blocks"][1:] == [
        {"block_id": 1, "block_hash": None},
        {"block_id": 99, "block_hash": None},
    ]


def test_async_metadata_publish_skips_blocks_reused_before_registration(
    monkeypatch,
) -> None:
    """后台任务只应注册仍持有预期完整 hash 的目标物理 block。"""

    packed_hashes = [b"a" * 36, b"b" * 36]
    registered: list[tuple[list[int], list[str]]] = []
    relayed: list[list[object]] = []
    control = types.SimpleNamespace(
        config=types.SimpleNamespace(block_size=16),
        register_published_physical_blocks=lambda block_ids, block_hashes, **_kwargs: (
            registered.append((list(block_ids), list(block_hashes)))
            or {
                "event_type": "store",
                "location": "npu",
                "block_hashes": list(block_hashes),
                "engine_specific": {"physical_block_ids": list(block_ids)},
            }
        ),
    )

    def dispatch(path: str, _payload: dict[str, object]) -> dict[str, object]:
        if path == "/datadist/blocks/metadata":
            return {
                "ok": True,
                "blocks": [
                    {
                        "block_id": 11,
                        "block_hash": {"packed_hex": packed_hashes[0].hex()},
                    },
                    {"block_id": 12, "block_hash": None},
                ],
            }
        if path == "/datadist/source-leases/acquire":
            return {
                "ok": True,
                "lease": {
                    "lease_id": "metadata-lease",
                    "block_hashes": [{"packed_hex": packed_hashes[0].hex()}],
                },
            }
        assert path == "/datadist/source-leases/release"
        return {"ok": True}

    monkeypatch.setattr(integration, "_dispatch_block_pool_request", dispatch)
    monkeypatch.setattr(
        integration,
        "_enqueue_runtime_events",
        lambda events, *, include_store=False: relayed.extend(events) if include_store else None,
    )
    task = integration._PublishedMetadataTask(
        transfer_id="transfer-stale",
        control=control,
        block_ids=(11, 12),
        packed_hashes=tuple(value.hex() for value in packed_hashes),
        external_hashes=("hash-a", "hash-b"),
    )

    integration._publish_published_block_metadata(task)

    assert registered == [([11], ["hash-a"])]
    assert len(relayed) == 1
    assert relayed[0]["block_hashes"] == ["hash-a"]


def test_async_metadata_publisher_deduplicates_pending_blocks(monkeypatch) -> None:
    """相同 block 和完整 hash 在后台执行期间只应排队一次。"""

    started = threading.Event()
    release = threading.Event()
    published: list[str] = []

    def publish(task: integration._PublishedMetadataTask) -> None:
        started.set()
        assert release.wait(2.0)
        published.append(task.transfer_id)

    monkeypatch.setattr(integration, "_publish_published_block_metadata", publish)
    publisher = integration._AsyncPublishedMetadataPublisher(max_pending_batches=2)
    task = integration._PublishedMetadataTask(
        transfer_id="transfer-deduplicate",
        control=object(),
        block_ids=(11,),
        packed_hashes=("aa",),
        external_hashes=("hash-a",),
    )
    try:
        first = publisher.submit(task)
        assert started.wait(2.0)
        duplicate = publisher.submit(task)
        release.set()
        assert publisher.wait_for_idle(2.0)
    finally:
        release.set()
        publisher.close()

    assert first["queued_count"] == 1
    assert duplicate == {
        "queued": True,
        "duplicate_count": 1,
        "queued_count": 0,
        "reason": "already_pending",
    }
    assert published == ["transfer-deduplicate"]


def test_remote_source_lease_guard_acquires_renews_and_releases(monkeypatch) -> None:
    """远端 guard 应在 publish 前同步续租，并在完成后释放 lease。"""

    calls: list[tuple[str, dict[str, object]]] = []
    packed_hash = b"a" * 36

    def post(
        base_url: str,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        assert base_url == "http://source:18011"
        calls.append((path, payload))
        if path.endswith("/acquire"):
            return {
                "ok": True,
                "lease": {
                    "lease_id": "lease-guard",
                    "block_hashes": [{"packed_hex": packed_hash.hex()}],
                },
            }
        return {"ok": True, "lease": {"lease_id": "lease-guard"}}

    monkeypatch.setattr(integration, "_post_remote_control", post)
    guard = integration._RemoteSourceLeaseGuard(
        "http://source:18011/",
        [1],
        ttl_ms=30_000,
    )

    lease = guard.acquire()
    guard.renew_now()
    released = guard.release()

    assert lease["lease_id"] == "lease-guard"
    assert released["ok"] is True
    assert [path for path, _ in calls] == [
        "/datadist/source-leases/acquire",
        "/datadist/source-leases/renew",
        "/datadist/source-leases/release",
    ]


def test_engine_core_datadist_digest_does_not_require_transfer_backend(monkeypatch) -> None:
    """digest 调试接口应直接读取 provider，不依赖 LLM-DataDist backend。"""

    class Provider:
        @staticmethod
        def block_digest(block_id: int, *, include_layers: bool = False):
            return {
                "block_id": block_id,
                "sha256": f"digest-{block_id}",
                "layers": [] if include_layers else None,
            }

    control = types.SimpleNamespace(provider=Provider(), llm_datadist_backend=None)
    monkeypatch.setattr(integration, "_ENGINE_CORE_DATAPLANE", control)

    result = integration._handle_engine_core_datadist_request(
        "/datadist/digest",
        {"block_ids": [1, 2], "include_layers": True},
    )

    assert result["ok"] is True
    assert [block["sha256"] for block in result["blocks"]] == ["digest-1", "digest-2"]


def test_cluster_status_validation_rejects_non_numeric_failure_enum() -> None:
    """pybind 风格枚举无法 int 转换时仍应识别逐 cluster 建链失败。"""

    class LLMStatusCode(Enum):
        LLM_SUCCESS = "success"
        LLM_LINK_FAILED = "link_failed"

    cluster = LLMDataDistCluster(
        remote_cluster_id=9,
        local_ip="10.0.0.1",
        local_port=26000,
        remote_ip="10.0.0.2",
        remote_port=26001,
    )

    with pytest.raises(LLMDataDistBackendError, match="remote_cluster_id=9"):
        _raise_cluster_statuses(
            (LLMStatusCode.LLM_SUCCESS, [LLMStatusCode.LLM_LINK_FAILED]),
            "LLMDataDist.link_clusters",
            [cluster],
        )


def test_external_source_hash_validation_detects_reused_physical_block() -> None:
    """Router 观察到的 hash 与 lease 快照不一致时必须拒绝继续传输。"""
    packed_hash = b"actual-hash" + (16).to_bytes(4, byteorder="little")

    with pytest.raises(
        ValueError,
        match="source external block hashes changed",
    ):
        integration._validate_expected_external_source_hashes(
            {"external_block_hashes": ["stale-hash"]},
            [{"packed_hex": packed_hash.hex()}],
        )


def test_transfer_deadline_rejects_late_publish() -> None:
    """已经超过请求 deadline 的传输必须在 publish 前失败并进入回滚路径。"""
    with pytest.raises(
        integration.TransferCancelledError,
        match="deadline expired",
    ):
        integration._raise_if_transfer_deadline_expired({"deadline_epoch_ms": int(time.time() * 1000) - 1})


def test_pd_control_event_broker_replays_only_matching_handoff() -> None:
    """PD事件订阅应按handoff隔离，并重放订阅前已发布的事件。"""

    broker = integration._PDControlEventBroker()
    first = broker.publish(
        "REQUEST_SEEN",
        handoff_id="handoff-a",
        request_id="request-a",
    )
    broker.publish(
        "REQUEST_SEEN",
        handoff_id="handoff-b",
        request_id="request-b",
    )

    subscriber, history = broker.subscribe(
        handoff_id="handoff-a",
        after_sequence=0,
    )
    assert history == [first]
    completed = broker.publish(
        "IMPORT_COMPLETE",
        handoff_id="handoff-a",
        request_id="request-a",
        result={"ok": True},
    )
    assert subscriber.get(timeout=1.0) == completed
    broker.unsubscribe(handoff_id="handoff-a", subscriber=subscriber)


def test_async_pd_import_is_idempotent_and_publishes_result(monkeypatch) -> None:
    """重复提交同一handoff只能执行一次，并通过事件和结果接口返回终态。"""

    monkeypatch.setenv("JIUWEN_KVBM_PD_BLOCK_OWNERSHIP", "legacy")
    broker = integration._PDControlEventBroker()
    calls: list[str] = []
    gate_renewals: list[dict] = []

    def run_import(control, backend, payload):
        calls.append(str(payload["handoff_id"]))
        return {"ok": True, "destination_blocks": [7, 8]}

    monkeypatch.setattr(integration, "_PD_CONTROL_EVENTS", broker)
    monkeypatch.setattr(integration, "_PD_IMPORT_FUTURES", {})
    monkeypatch.setattr(integration, "_pd_import_response", run_import)
    monkeypatch.setattr(
        integration,
        "_dispatch_block_pool_request",
        lambda _path, request: gate_renewals.append(dict(request)) or {"ok": True, "gate": {"request_seen": True}},
    )
    subscriber, _history = broker.subscribe(handoff_id="handoff-a")
    payload = {"handoff_id": "handoff-a", "request_id": "request-a"}

    first = integration._submit_pd_import_response(object(), object(), payload)
    second = integration._submit_pd_import_response(object(), object(), payload)
    event = subscriber.get(timeout=1.0)
    result = integration._pd_import_result_response({"handoff_id": "handoff-a", "wait_timeout_ms": 1000})

    assert first["accepted"] is True
    assert second["already_submitted"] is True
    assert event["event_type"] == "IMPORT_COMPLETE"
    assert event["result"] == result
    assert calls == ["handoff-a"]
    assert len(gate_renewals) == 1
    assert gate_renewals[0]["require_existing"] is True
    broker.unsubscribe(handoff_id="handoff-a", subscriber=subscriber)


def test_scheduler_pd_import_waits_without_occupying_executor(monkeypatch) -> None:
    """Scheduler容量未就绪时应先登记，分配成功后再投递真实import。"""

    monkeypatch.setenv("JIUWEN_KVBM_PD_BLOCK_OWNERSHIP", "scheduler")
    broker = integration._PDControlEventBroker()
    calls: list[str] = []
    gate = integration._PDTargetRequestGate(
        handoff_id="handoff-capacity",
        request_id="request-capacity",
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 120.0,
        request_seen=True,
        engine_request_id="request-capacity-engine",
        preplan_capacity_waiting=True,
    )

    def run_import(control, backend, payload):
        calls.append(str(payload["handoff_id"]))
        return {"ok": True, "destination_blocks": [21]}

    monkeypatch.setattr(integration, "_PD_CONTROL_EVENTS", broker)
    monkeypatch.setattr(integration, "_PD_IMPORT_FUTURES", {})
    monkeypatch.setattr(integration, "_PD_SCHEDULER_IMPORT_SUBMISSIONS", {})
    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {gate.request_id: gate})
    monkeypatch.setattr(integration, "_pd_import_response", run_import)
    monkeypatch.setattr(
        integration,
        "_dispatch_block_pool_request",
        lambda _path, _request: {
            "ok": True,
            "gate": {"request_seen": True},
        },
    )
    subscriber, _history = broker.subscribe(handoff_id=gate.handoff_id)

    accepted = integration._submit_pd_import_response(
        object(),
        object(),
        {"handoff_id": gate.handoff_id, "request_id": gate.request_id},
    )

    assert accepted["accepted"] is True
    assert accepted["waiting_for_scheduler_capacity"] is True
    assert calls == []

    gate.scheduler_allocation = types.SimpleNamespace(released=False)
    assert integration._activate_scheduler_import_submission(gate.handoff_id) is True
    result = integration._pd_import_result_response({"handoff_id": gate.handoff_id, "wait_timeout_ms": 1000})
    event = subscriber.get(timeout=1.0)

    assert result["ok"] is True
    assert calls == [gate.handoff_id]
    assert event["event_type"] == "IMPORT_COMPLETE"
    broker.unsubscribe(handoff_id=gate.handoff_id, subscriber=subscriber)


def test_scheduler_cycle_retries_capacity_and_activates_import(monkeypatch) -> None:
    """BlockPool释放容量后Scheduler周期应完成分配并唤醒等待事务。"""

    monkeypatch.setenv("JIUWEN_KVBM_PD_BLOCK_OWNERSHIP", "scheduler")
    broker = integration._PDControlEventBroker()
    gate = integration._PDTargetRequestGate(
        handoff_id="handoff-retry",
        request_id="request-retry",
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 120.0,
        request_seen=True,
        engine_request_id="request-retry-engine",
        preplan_capacity_waiting=True,
        preplan_error="insufficient free vLLM blocks",
    )
    request = types.SimpleNamespace(request_id=gate.engine_request_id)
    scheduler = types.SimpleNamespace(requests={request.request_id: request})
    allocation = types.SimpleNamespace(
        planned_destination_blocks=(1, 2, 3),
        transfer_positions=(2,),
        allocation_ms=1.25,
    )
    activated: list[str] = []
    monkeypatch.setattr(integration, "_PD_CONTROL_EVENTS", broker)
    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {gate.request_id: gate})
    monkeypatch.setattr(
        integration,
        "_scheduler_owned_pd_target_allocation",
        lambda _scheduler, _request, *, handoff_id: allocation,
    )
    monkeypatch.setattr(
        integration,
        "_activate_scheduler_import_submission",
        lambda handoff_id: activated.append(handoff_id) or True,
    )
    subscriber, _history = broker.subscribe(handoff_id=gate.handoff_id)

    try:
        integration._retry_scheduler_owned_target_allocations(
            scheduler,
            integration.logger,
        )
        event = subscriber.get(timeout=1.0)
    finally:
        broker.unsubscribe(
            handoff_id=gate.handoff_id,
            subscriber=subscriber,
        )

    assert gate.scheduler_allocation is allocation
    assert gate.preplan_capacity_waiting is False
    assert gate.preplan_error == ""
    assert activated == [gate.handoff_id]
    assert event["event_type"] == "TARGET_BLOCKS_ALLOCATED"
    assert event["result"] == {
        "scheduler_allocation_ready": True,
        "allocated_blocks": 3,
        "transfer_blocks": 1,
        "capacity_generation": -1,
    }


def test_scheduler_capacity_retry_waits_for_new_block_pool_generation(
    monkeypatch,
) -> None:
    """容量generation未变时不应在每个Scheduler周期重试分配。"""

    monkeypatch.setenv("JIUWEN_KVBM_PD_BLOCK_OWNERSHIP", "scheduler")
    block_pool = types.SimpleNamespace()
    setattr(block_pool, integration._BLOCK_POOL_CAPACITY_GENERATION_ATTR, 100)
    gate = integration._PDTargetRequestGate(
        handoff_id="handoff-generation",
        request_id="request-generation",
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 120.0,
        request_seen=True,
        engine_request_id="request-generation-engine",
        preplan_capacity_waiting=True,
        last_capacity_retry_generation=100,
    )
    request = types.SimpleNamespace(request_id=gate.engine_request_id)
    scheduler = types.SimpleNamespace(
        requests={request.request_id: request},
        kv_cache_manager=types.SimpleNamespace(block_pool=block_pool),
    )
    allocation = types.SimpleNamespace(
        planned_destination_blocks=(1,),
        transfer_positions=(0,),
        allocation_ms=0.5,
        capacity_generation=101,
    )
    calls: list[int] = []
    monkeypatch.setattr(integration, "_PD_SCHEDULER_IMPORT_SUBMISSIONS", {})
    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {gate.request_id: gate})
    monkeypatch.setattr(
        integration,
        "_scheduler_owned_pd_target_allocation",
        lambda *_args, **_kwargs: calls.append(1) or allocation,
    )
    monkeypatch.setattr(
        integration,
        "_activate_scheduler_import_submission",
        lambda _handoff_id: True,
    )

    integration._retry_scheduler_owned_target_allocations(
        scheduler,
        integration.logger,
    )
    assert calls == []

    setattr(block_pool, integration._BLOCK_POOL_CAPACITY_GENERATION_ATTR, 101)
    integration._retry_scheduler_owned_target_allocations(
        scheduler,
        integration.logger,
    )
    assert calls == [1]
    assert gate.scheduler_allocation is allocation


def test_gated_request_is_restored_to_keep_scheduler_mailbox_live(
    monkeypatch,
) -> None:
    """等待plan时D请求必须回到waiting，以便EngineCore继续处理mailbox。"""

    class RequestQueue:
        def __init__(self, requests: list[object]) -> None:
            self.requests = list(requests)

        def remove_requests(self, requests: list[object]) -> None:
            removed = {id(request) for request in requests}
            self.requests = [request for request in self.requests if id(request) not in removed]

        def prepend_request(self, request: object) -> None:
            self.requests.insert(0, request)

    gate = integration._PDTargetRequestGate(
        handoff_id="handoff-mailbox-live",
        request_id="request-mailbox-live",
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 120.0,
        request_seen=True,
        engine_request_id="request-mailbox-live-engine",
        preplan_capacity_waiting=True,
    )
    request = types.SimpleNamespace(request_id=gate.engine_request_id)
    waiting = RequestQueue([request])
    skipped_waiting = RequestQueue([])
    scheduler = types.SimpleNamespace(
        requests={request.request_id: request},
        waiting=waiting,
        skipped_waiting=skipped_waiting,
    )
    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {gate.request_id: gate})

    gated = integration._remove_gated_pd_requests_from_queues(scheduler)
    assert gated == [request]
    assert waiting.requests == []

    integration._restore_gated_pd_requests(scheduler, gated)
    assert waiting.requests == [request]


def test_scheduler_capacity_deadline_fails_import_and_releases_gate(
    monkeypatch,
) -> None:
    """Scheduler容量竞态超过短截止时间后应结束事务并释放D gate。"""
    monkeypatch.setenv("JIUWEN_KVBM_PD_BLOCK_OWNERSHIP", "scheduler")
    submission = integration._PDSchedulerImportSubmission(
        handoff_id="handoff-expired",
        request_id="request-expired",
        control=object(),
        backend=object(),
        payload={},
        result_future=Future(),
        created_at=time.monotonic() - 10.0,
        capacity_deadline_at=time.monotonic() - 1.0,
    )
    failed: list[str] = []
    released: list[dict] = []
    monkeypatch.setattr(
        integration,
        "_PD_SCHEDULER_IMPORT_SUBMISSIONS",
        {submission.handoff_id: submission},
    )
    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {})
    monkeypatch.setattr(
        integration,
        "_fail_pending_scheduler_import",
        lambda handoff_id, *, error: failed.append(f"{handoff_id}:{error}") or True,
    )
    monkeypatch.setattr(
        integration,
        "_local_pd_import_release",
        lambda payload: released.append(payload) or {"released": True},
    )

    integration._retry_scheduler_owned_target_allocations(
        types.SimpleNamespace(requests={}),
        integration.logger,
    )

    assert failed == ["handoff-expired:timed out waiting for vLLM Scheduler target blocks"]
    assert released == [
        {
            "request_id": submission.request_id,
            "handoff_id": submission.handoff_id,
        }
    ]


def test_pd_control_event_http_stream_delivers_worker_notification(monkeypatch) -> None:
    """Worker长连接接口应立即向Router写出scheduler事件。"""

    broker = integration._PDControlEventBroker()
    monkeypatch.setattr(integration, "_PD_CONTROL_EVENTS", broker)
    server = integration.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        integration._EngineCorePrefetchHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    response = urllib.request.urlopen(
        f"http://{host}:{port}/datadist/pd/events?handoff_id=handoff-a",
        timeout=2.0,
    )
    try:
        broker.publish(
            "REQUEST_SEEN",
            handoff_id="handoff-a",
            request_id="request-a",
        )
        event = json.loads(response.readline().decode("utf-8"))
        assert event["event_type"] == "REQUEST_SEEN"
        assert event["request_id"] == "request-a"
    finally:
        response.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_pd_control_event_broker_replays_p_request_seen() -> None:
    """P请求先进入Scheduler时，后连接的Router仍应收到可重放事件。"""

    broker = integration._PDControlEventBroker()
    broker.publish(
        "P_REQUEST_SEEN",
        handoff_id="handoff-p-seen",
        request_id="request-p-seen",
        result={"scheduler_seen_cycle_id": 7},
    )

    subscriber, history = broker.subscribe(handoff_id="handoff-p-seen")
    try:
        assert len(history) == 1
        assert history[0]["event_type"] == "P_REQUEST_SEEN"
        assert history[0]["result"]["scheduler_seen_cycle_id"] == 7
    finally:
        broker.unsubscribe(
            handoff_id="handoff-p-seen",
            subscriber=subscriber,
        )


def test_event_drain_consumes_body_before_keep_alive_reuse() -> None:
    """events/drain必须消费POST body，避免下次方法被解析为{}POST。"""
    server = integration.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        integration._EngineCorePrefetchHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=2.0)
    try:
        connection.request(
            "POST",
            "/events/drain",
            body="{}",
            headers={"Content-Type": "application/json"},
        )
        first = connection.getresponse()
        first.read()

        connection.request(
            "POST",
            "/not-found",
            body="{}",
            headers={"Content-Type": "application/json"},
        )
        second = connection.getresponse()
        second.read()

        assert first.status == 200
        assert second.status == 404
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_block_pool_capacity_endpoint_avoids_full_backend_status(monkeypatch) -> None:
    """轻量容量接口不应调用LLM-DataDist health等完整状态逻辑。"""

    class _Control:
        llm_datadist_backend = None

    monkeypatch.setattr(integration, "_ENGINE_CORE_DATAPLANE", _Control())
    monkeypatch.setattr(
        integration,
        "_dispatch_block_pool_request",
        lambda path, payload: {
            "ok": True,
            "free_blocks": 321,
            "ownership_mode": "scheduler",
            "_control_timing": {"engine_core_execution_ms": 0.0},
        },
    )

    result = integration._handle_engine_core_datadist_request(
        "/datadist/block-pool/capacity",
        {},
    )

    assert result["ok"] is True
    assert result["block_pool"]["ready"] is True
    assert result["block_pool"]["free_blocks"] == 321
    assert result["block_pool"]["ownership_mode"] == "scheduler"
    assert result["control_timing"]["engine_core_proxy_ms"] >= 0
    assert result["control_timing"]["engine_core_execution_ms"] == 0
    assert result["control_timing"]["engine_core_queue_and_transport_ms"] >= 0


def test_capacity_http_response_exposes_control_phases(monkeypatch) -> None:
    monkeypatch.setattr(
        integration,
        "_handle_engine_core_datadist_request",
        lambda path, payload: {
            "ok": True,
            "control_timing": {"engine_core_proxy_ms": 2.0},
            "block_pool": {"ready": True, "free_blocks": 30},
        },
    )
    server = integration.ThreadingHTTPServer(("127.0.0.1", 0), integration._EngineCorePrefetchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    try:
        body = json.dumps({"request_id": "r1", "handoff_id": "pd-r1"})
        connection.request(
            "POST", "/datadist/block-pool/capacity", body=body, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 200
        assert result["control_timing"]["engine_core_proxy_ms"] == 2
        assert result["control_timing"]["handler_execution_ms"] >= 0
        assert result["control_timing"]["handler_request_read_ms"] >= 0
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _fake_config() -> LLMDataDistConfig:
    """构造测试用 LLM-DataDist 配置。

    返回:
        Fake backend 使用的配置对象。
    """

    return LLMDataDistConfig(
        enabled=True,
        endpoint=LLMDataDistEndpoint(
            cluster_id=0,
            role=LLMDataDistRole.MIX,
            listen_ip="127.0.0.1",
            listen_port=19000,
            device_id=0,
        ),
    )


class _FakeKVTensor:
    """测试用最小 tensor 对象。"""

    def __init__(self, shape: tuple[int, ...], dtype: str, address: int) -> None:
        """创建 fake tensor。

        参数:
            shape: tensor shape。
            dtype: dtype 字符串。
            address: data_ptr 返回的地址。

        返回:
            无返回值。
        """

        self.shape = shape
        self.dtype = dtype
        self.device = "npu:0"
        self._address = address

    def data_ptr(self) -> int:
        """返回 fake tensor 地址。

        返回:
            构造时传入的地址。
        """

        return self._address
