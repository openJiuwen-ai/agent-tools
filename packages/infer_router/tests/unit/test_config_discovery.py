"""测试配置文件发现功能"""

import json
from pathlib import Path

import pytest

from openjiuwentools.infer_router.discovery.config_discovery import ConfigDiscovery
from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType


@pytest.mark.asyncio
async def test_config_discovery_json(sample_worker_config_json: Path):
    """测试从JSON配置文件发现工作器"""
    discovery = ConfigDiscovery(config_path=str(sample_worker_config_json))
    workers = await discovery.discover()

    assert len(workers) == 2
    assert all(isinstance(w, WorkerInfo) for w in workers)

    worker1 = workers[0]
    assert worker1.worker_id == "test-worker-1"
    assert worker1.model == "test-model"
    assert worker1.url == "http://localhost:8001/v1"
    assert worker1.available_memory == 1000000
    assert worker1.current_load == 10
    assert worker1.cached_prefixes == ["prefix-1", "prefix-2"]

    worker2 = workers[1]
    assert worker2.worker_id == "test-worker-2"
    assert worker2.model == "test-model"
    assert worker2.url == "http://localhost:8002/v1"


@pytest.mark.asyncio
async def test_config_discovery_yaml(sample_worker_config_yaml: Path):
    """测试从YAML配置文件发现工作器"""
    discovery = ConfigDiscovery(config_path=str(sample_worker_config_yaml))
    workers = await discovery.discover()

    assert len(workers) == 2
    assert all(isinstance(w, WorkerInfo) for w in workers)

    worker1 = workers[0]
    assert worker1.worker_id == "test-worker-1"
    assert worker1.model == "test-model"
    assert worker1.url == "http://localhost:8001/v1"
    assert worker1.available_memory == 1000000
    assert worker1.current_load == 10
    assert worker1.cached_prefixes == ["prefix-1", "prefix-2"]


@pytest.mark.asyncio
async def test_config_discovery_file_not_found():
    """测试配置文件不存在的情况"""
    discovery = ConfigDiscovery(config_path="nonexistent.json")
    workers = await discovery.discover()

    assert len(workers) == 0


@pytest.mark.asyncio
async def test_config_discovery_invalid_json(tmp_path: Path):
    """测试无效的JSON格式"""
    config_file = tmp_path / "invalid.json"
    config_file.write_text("{ invalid json }")

    discovery = ConfigDiscovery(config_path=str(config_file))
    workers = await discovery.discover()

    assert len(workers) == 0


@pytest.mark.asyncio
async def test_config_discovery_missing_fields(invalid_worker_config: Path):
    """测试缺少必需字段的配置"""
    discovery = ConfigDiscovery(config_path=str(invalid_worker_config))
    workers = await discovery.discover()

    assert len(workers) == 0


@pytest.mark.asyncio
async def test_config_discovery_unsupported_format(tmp_path: Path):
    """测试不支持的文件格式"""
    config_file = tmp_path / "workers.txt"
    config_file.write_text("workers: []")

    discovery = ConfigDiscovery(config_path=str(config_file))
    workers = await discovery.discover()

    assert len(workers) == 0


@pytest.mark.asyncio
async def test_config_discovery_start_stop(sample_worker_config_json: Path):
    """测试启动和停止发现服务"""
    discovery = ConfigDiscovery(config_path=str(sample_worker_config_json))

    await discovery.start()
    await discovery.stop()


@pytest.mark.asyncio
async def test_config_discovery_default_values(tmp_path: Path):
    """测试默认值"""
    config_file = tmp_path / "workers.json"
    config_file.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "worker_id": "test-worker",
                        "model": "test-model",
                        "url": "http://localhost:8001/v1",
                    }
                ]
            }
        )
    )

    discovery = ConfigDiscovery(config_path=str(config_file))
    workers = await discovery.discover()

    assert len(workers) == 1
    worker = workers[0]
    assert worker.total_tokens == 1000000
    assert worker.current_load == 0
    assert worker.cached_prefixes == []


@pytest.mark.asyncio
async def test_config_discovery_with_worker_type_and_group(
    sample_worker_config_with_types: Path,
):
    """测试解析包含worker_type和group的配置"""
    discovery = ConfigDiscovery(config_path=str(sample_worker_config_with_types))
    workers = await discovery.discover()

    assert len(workers) == 5

    worker_map = {w.worker_id: w for w in workers}

    prefill_worker = worker_map["prefill-worker-1"]
    assert prefill_worker.worker_type == WorkerType.PREFILL
    assert prefill_worker.group == "group-a"

    decode_worker = worker_map["decode-worker-1"]
    assert decode_worker.worker_type == WorkerType.DECODE
    assert decode_worker.group == "group-a"

    combined_worker1 = worker_map["combined-worker-1"]
    assert combined_worker1.worker_type == WorkerType.COMBINED
    assert combined_worker1.group == "group-b"

    combined_worker2 = worker_map["combined-worker-2"]
    assert combined_worker2.worker_type == WorkerType.COMBINED
    assert combined_worker2.group == "group-b"

    default_worker = worker_map["default-worker"]
    assert default_worker.worker_type == WorkerType.COMBINED
    assert default_worker.group == "default"


@pytest.mark.asyncio
async def test_config_discovery_invalid_worker_type(tmp_path: Path):
    """测试无效的worker_type值"""
    config_file = tmp_path / "workers.json"
    config_file.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "worker_id": "test-worker",
                        "model": "test-model",
                        "url": "http://localhost:8001/v1",
                        "worker_type": "invalid_type",
                    }
                ]
            }
        )
    )

    discovery = ConfigDiscovery(config_path=str(config_file))
    workers = await discovery.discover()

    assert len(workers) == 1
    worker = workers[0]
    assert worker.worker_type == WorkerType.COMBINED
