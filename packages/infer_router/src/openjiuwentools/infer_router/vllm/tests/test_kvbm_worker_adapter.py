# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
import argparse
import types

import pytest

pytest.importorskip("jiuwen_kvbm")

from openjiuwentools.infer_router.vllm.kvbm_worker_adapter import (
    VllmWorkerKVBMAdapter,
    VllmWorkerKVBMConfig,
)
from openjiuwentools.infer_router.vllm.worker_publisher import (
    MetricsPublisher,
    _event_to_dict,
)
from jiuwen_kvbm.accessor import FakeTensorAccessor


class Provider:
    def __init__(self):
        self.accessor = FakeTensorAccessor({1: b"payload", 2: b"payload-2"})

    def read_block(self, block_id: int):
        return self.accessor.read_block(block_id)

    def write_block(self, block_id: int, payload):
        self.accessor.write_block(block_id, payload)

    def describe_block(self, block_id: int):
        return self.accessor.describe_block(block_id)

    def delete_block(self, block_id: int):
        self.accessor.delete_block(block_id)


class FakeBlock:
    def __init__(self, values):
        self.values = list(values)
        self.device = "npu"
        self.dtype = "float32"
        self.shape = (len(self.values),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return FakeBlock(self.values)

    def to(self, device=None, dtype=None):
        return self

    def copy_(self, other):
        self.values = list(other.values)
        return self

    def numel(self):
        return len(self.values)

    @staticmethod
    def element_size():
        return 4


class FakeCache:
    def __init__(self, blocks):
        self.blocks = [FakeBlock(block) for block in blocks]
        self.shape = (len(self.blocks),) + self.blocks[0].shape
        self.dtype = self.blocks[0].dtype
        self.device = self.blocks[0].device

    def __getitem__(self, index):
        return self.blocks[index]


def _config(tmp_path, enabled=True, provider="", diagnostics=False):
    return VllmWorkerKVBMConfig(
        enabled=enabled,
        worker_id="worker-1",
        disk_root=tmp_path,
        clean_disk_on_start=True,
        npu_high_watermark_blocks=0,
        ram_high_watermark_bytes=0,
        interval=60.0,
        block_size=16,
        provider=provider,
        diagnostics=diagnostics,
        locator_depth=16,
    )


@pytest.mark.asyncio
async def test_adapter_is_inactive_without_provider_factory(tmp_path):
    """未提供 provider_factory 时，adapter 不访问 vLLM-Ascend 私有结构。"""
    adapter = VllmWorkerKVBMAdapter(_config(tmp_path), engine=object())

    await adapter.start()

    assert not adapter.active
    assert adapter.inactive_reason == "provider_factory_missing"
    assert adapter.runtime is None


@pytest.mark.asyncio
async def test_adapter_starts_with_version_independent_provider(tmp_path):
    """提供版本无关 provider 后，adapter 应能启动 WorkerKVBMRuntime。"""
    provider = Provider()
    adapter = VllmWorkerKVBMAdapter(
        _config(tmp_path),
        engine=object(),
        provider_factory=lambda _engine: provider,
    )

    await adapter.start()
    try:
        assert adapter.active
        adapter.on_block_stored(1, "hash-1", token_count=16)
        assert adapter.runtime is not None
        assert adapter.runtime.stats()["total_blocks"] == 1
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_adapter_applies_local_kv_events_with_physical_block_ids(tmp_path):
    """本地 KV 事件 sink 应能把物理 block 事件接入 worker KVBM runtime。"""
    provider = Provider()
    adapter = VllmWorkerKVBMAdapter(
        _config(tmp_path),
        engine=object(),
        provider_factory=lambda _engine: provider,
    )

    await adapter.start()
    try:
        events = adapter.apply_kv_events(
            [
                {
                    "type": "BlockStored",
                    "data": {
                        "physical_block_ids": [1],
                        "block_hashes": ["hash-1"],
                        "block_size": 16,
                    },
                },
                {
                    "type": "BlockMoved",
                    "data": {
                        "physical_block_id": 1,
                        "block_hash": "hash-1",
                        "to": "ram",
                    },
                },
            ]
        )

        assert [event.event_type for event in events] == ["store", "move"]
        assert adapter.runtime.stats()["ram_blocks"] == 1
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_adapter_ignores_hash_only_local_kv_event(tmp_path):
    """缺少物理 block id 的 KV 事件不应触发 worker 数据面注册。"""
    provider = Provider()
    adapter = VllmWorkerKVBMAdapter(
        _config(tmp_path),
        engine=object(),
        provider_factory=lambda _engine: provider,
    )

    await adapter.start()
    try:
        result = adapter.apply_kv_event(
            {
                "type": "BlockStored",
                "data": {
                    "block_hashes": ["hash-only"],
                    "block_size": 16,
                },
            }
        )

        assert result is None
        assert adapter.runtime.stats()["total_blocks"] == 0
    finally:
        await adapter.stop()


def test_config_from_args_maps_worker_flags(tmp_path):
    """CLI/YAML 参数应能映射为 worker KVBM 配置。"""
    args = argparse.Namespace(
        kvbm_worker_enabled=True,
        kvbm_worker_disk_root=str(tmp_path),
        kvbm_clean_disk_on_start=False,
        kvbm_npu_high_watermark_blocks=8,
        kvbm_ram_high_watermark_bytes=1024,
        kvbm_worker_interval=3.0,
        kvbm_block_size=32,
        kvbm_worker_provider="vllm-ascend-0.18",
        kvbm_worker_diagnostics=True,
        kvbm_worker_locator_depth=20,
    )

    config = VllmWorkerKVBMConfig.from_args(args, "worker-1")

    assert config.enabled is True
    assert config.disk_root == tmp_path
    assert config.clean_disk_on_start is False
    assert config.npu_high_watermark_blocks == 8
    assert config.ram_high_watermark_bytes == 1024
    assert config.interval == 3.0
    assert config.block_size == 32
    assert config.provider == "vllm-ascend-0.18"
    assert config.diagnostics is True
    assert config.locator_depth == 20


@pytest.mark.asyncio
async def test_adapter_auto_locates_vllm_ascend_018_provider(tmp_path):
    """配置 provider 后，adapter 应能从 engine 对象树自动定位 kv_caches。"""

    class Runner:
        def __init__(self):
            self.kv_caches = {
                "layer.0": (
                    FakeCache([[1, 2, 3, 4], [5, 6, 7, 8]]),
                    FakeCache([[9, 10, 11, 12], [13, 14, 15, 16]]),
                )
            }

    class Worker:
        def __init__(self):
            self.model_runner = Runner()

    class Engine:
        def __init__(self):
            self.worker = Worker()

    adapter = VllmWorkerKVBMAdapter(
        _config(tmp_path, provider="vllm-ascend-0.18", diagnostics=True),
        engine=Engine(),
    )

    await adapter.start()
    try:
        assert adapter.active
        assert adapter.runtime is not None
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_adapter_defers_to_engine_core_hook_for_async_mp_client(tmp_path):
    """AsyncMPClient 多进程模式下，外层 adapter 应让位给 engine-core hook。"""

    AsyncMPClient = type("AsyncMPClient", (), {})
    AsyncMPClient.__module__ = "vllm.v1.engine.core_client"

    class Engine:
        def __init__(self):
            self.engine_core = AsyncMPClient()

    adapter = VllmWorkerKVBMAdapter(
        _config(tmp_path, provider="vllm-ascend-0.18", diagnostics=True),
        engine=Engine(),
    )

    await adapter.start()

    assert not adapter.active
    assert adapter.runtime is None
    assert adapter.inactive_reason == "engine_core_hook_mode"


def test_metrics_publisher_merges_extra_metrics_source():
    """MetricsPublisher 应合并 KVBM adapter 提供的额外指标。"""
    publisher = MetricsPublisher(worker_publisher=object(), worker_id="worker-1")

    class Extra:
        @staticmethod
        def get_metrics():
            return {"kvbm_worker_active": True}

    publisher.set_extra_metrics_source(Extra())

    metrics = publisher._collect_metrics()
    assert metrics["kvbm_worker_active"] is True


def test_worker_publisher_infers_block_size_from_self_describing_store_event():
    """原生 store 事件可等分 token/hash 时应自动携带 block_size。

    返回:
        无返回值；发布字典缺少自动识别尺寸时由断言报告失败。
    """

    event = types.SimpleNamespace(
        token_ids=list(range(256)),
        block_hashes=["hash-0", "hash-1"],
        parent_block_hash=None,
    )

    payload = _event_to_dict(event)

    assert payload["block_size"] == 128
