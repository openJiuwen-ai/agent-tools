"""测试工作器管理器"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType
from openjiuwentools.infer_router.worker.worker_manager import (
    WorkerManager,
    WorkerStatus,
)


@pytest.fixture
def worker_manager():
    """创建工作器管理器实例"""
    return WorkerManager()


@pytest.fixture
def sample_workers():
    """创建示例工作器列表"""
    return [
        WorkerInfo(
            worker_id="worker-1",
            model="test-model",
            url="http://localhost:8001/v1",
            available_memory=1000000,
            current_load=10,
            cached_prefixes=["prefix-1"],
        ),
        WorkerInfo(
            worker_id="worker-2",
            model="test-model",
            url="http://localhost:8002/v1",
            available_memory=800000,
            current_load=5,
            cached_prefixes=["prefix-2"],
        ),
    ]


@pytest.mark.asyncio
async def test_worker_manager_start_stop(worker_manager: WorkerManager):
    """测试工作器管理器启动和停止"""
    with patch.object(worker_manager, "create_discovery") as mock_create:
        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(return_value=[])
        mock_create.return_value = mock_discovery

        await worker_manager.start()
        assert worker_manager.discovery_task is not None
        assert worker_manager.health_check_task is not None

        await worker_manager.stop()
        # 任务被取消后会进入 cancelling 状态
        assert worker_manager.discovery_task.cancelling() or worker_manager.discovery_task.done()
        assert worker_manager.health_check_task.cancelling() or worker_manager.health_check_task.done()


@pytest.mark.asyncio
async def test_worker_manager_discover_workers(
    worker_manager: WorkerManager,
    sample_workers: list[WorkerInfo],
    sample_worker_config_json: Path,
):
    """测试工作器发现"""
    with patch.object(worker_manager, "create_discovery") as mock_create:
        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(return_value=sample_workers)
        mock_create.return_value = mock_discovery

        worker_manager.discovery = mock_discovery
        await worker_manager.discover_workers()

        assert len(worker_manager.workers) == 2
        assert "worker-1" in worker_manager.workers
        assert "worker-2" in worker_manager.workers
        assert len(worker_manager.worker_statuses) == 2


@pytest.mark.asyncio
async def test_worker_manager_remove_workers(worker_manager: WorkerManager, sample_workers: list[WorkerInfo]):
    """测试移除不再存在的工作器"""
    worker_manager.workers = {w.worker_id: w for w in sample_workers}
    worker_manager.worker_statuses = {
        w.worker_id: WorkerStatus(worker_id=w.worker_id, last_health_check=0, is_healthy=True, response_time=0)
        for w in sample_workers
    }

    with patch.object(worker_manager, "create_discovery") as mock_create:
        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(return_value=[sample_workers[0]])
        mock_create.return_value = mock_discovery

        worker_manager.discovery = mock_discovery
        await worker_manager.discover_workers()

        assert len(worker_manager.workers) == 1
        assert "worker-1" in worker_manager.workers
        assert "worker-2" not in worker_manager.workers


@pytest.mark.asyncio
async def test_worker_manager_health_check(worker_manager: WorkerManager, sample_workers: list[WorkerInfo]):
    """测试健康检查"""
    worker_manager.workers = {w.worker_id: w for w in sample_workers}

    # Mock HTTP 客户端
    with patch.object(worker_manager, "http_client") as mock_client:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        await worker_manager.check_all_workers_health()

    for worker_id in worker_manager.workers:
        assert worker_id in worker_manager.worker_statuses
        status = worker_manager.worker_statuses[worker_id]
        assert isinstance(status, WorkerStatus)
        assert status.is_healthy is True


def test_worker_manager_get_healthy_workers(worker_manager: WorkerManager, sample_workers: list[WorkerInfo]):
    """测试获取健康工作器"""
    worker_manager.workers = {w.worker_id: w for w in sample_workers}
    worker_manager.worker_statuses = {
        w.worker_id: WorkerStatus(worker_id=w.worker_id, last_health_check=0, is_healthy=True, response_time=0)
        for w in sample_workers
    }

    healthy_workers = worker_manager.get_healthy_workers("test-model")
    assert len(healthy_workers) == 2

    healthy_workers = worker_manager.get_healthy_workers("other-model")
    assert len(healthy_workers) == 0


def test_worker_manager_get_worker(worker_manager: WorkerManager, sample_workers: list[WorkerInfo]):
    """测试获取单个工作器"""
    worker_manager.workers = {w.worker_id: w for w in sample_workers}

    worker = worker_manager.get_worker("worker-1")
    assert worker is not None
    assert worker.worker_id == "worker-1"

    worker = worker_manager.get_worker("nonexistent")
    assert worker is None


def test_worker_manager_get_all_workers(worker_manager: WorkerManager, sample_workers: list[WorkerInfo]):
    """测试获取所有工作器"""
    worker_manager.workers = {w.worker_id: w for w in sample_workers}

    all_workers = worker_manager.get_all_workers()
    assert len(all_workers) == 2
    assert "worker-1" in all_workers
    assert "worker-2" in all_workers


def test_worker_manager_create_discovery_config():
    """测试创建配置文件发现器"""
    from openjiuwentools.infer_router.discovery import ConfigDiscovery

    with patch("openjiuwentools.infer_router.worker.worker_manager.settings") as mock_settings:
        mock_settings.worker_discovery_type = "config"
        mock_settings.worker_config_path = "workers.json"

        manager = WorkerManager()
        discovery = manager.create_discovery()

        assert isinstance(discovery, ConfigDiscovery)


def test_worker_manager_create_discovery_etcd():
    """测试创建etcd发现器"""
    from openjiuwentools.infer_router.discovery import EtcdDiscovery

    with patch("openjiuwentools.infer_router.worker.worker_manager.settings") as mock_settings:
        mock_settings.worker_discovery_type = "etcd"
        mock_settings.etcd_hosts = ["localhost"]
        mock_settings.etcd_port = 2379
        mock_settings.etcd_prefix = "/test/workers"
        mock_settings.etcd_user = None
        mock_settings.etcd_password = None

        manager = WorkerManager()
        discovery = manager.create_discovery()

        assert isinstance(discovery, EtcdDiscovery)


@pytest.fixture
def workers_with_groups():
    """创建带有group和worker_type的工作器列表"""
    return [
        WorkerInfo(
            worker_id="prefill-1",
            model="test-model",
            url="http://localhost:8001/v1",
            available_memory=1000000,
            current_load=10,
            cached_prefixes=["prefix-1"],
            worker_type=WorkerType.PREFILL,
            group="group-a",
        ),
        WorkerInfo(
            worker_id="decode-1",
            model="test-model",
            url="http://localhost:8002/v1",
            available_memory=800000,
            current_load=5,
            cached_prefixes=["prefix-2"],
            worker_type=WorkerType.DECODE,
            group="group-a",
        ),
        WorkerInfo(
            worker_id="combined-1",
            model="test-model",
            url="http://localhost:8003/v1",
            available_memory=1200000,
            current_load=3,
            cached_prefixes=["prefix-3"],
            worker_type=WorkerType.COMBINED,
            group="group-b",
        ),
        WorkerInfo(
            worker_id="combined-2",
            model="test-model",
            url="http://localhost:8004/v1",
            available_memory=1500000,
            current_load=8,
            cached_prefixes=["prefix-4"],
            worker_type=WorkerType.COMBINED,
            group="group-b",
        ),
        WorkerInfo(
            worker_id="default-worker",
            model="test-model",
            url="http://localhost:8005/v1",
            available_memory=900000,
            current_load=15,
            cached_prefixes=[],
            worker_type=WorkerType.PREFILL,
            group="default",
        ),
    ]


def test_worker_manager_get_groups(worker_manager: WorkerManager, workers_with_groups: list[WorkerInfo]):
    """测试获取所有groups"""
    for worker in workers_with_groups:
        worker_manager.workers[worker.worker_id] = worker
        worker_manager.add_to_group(worker)

    groups = worker_manager.get_groups()
    assert len(groups) == 3
    assert "group-a" in groups
    assert "group-b" in groups
    assert "default" in groups


def test_worker_manager_get_workers_in_group(worker_manager: WorkerManager, workers_with_groups: list[WorkerInfo]):
    """测试获取指定group内的工作器"""
    for worker in workers_with_groups:
        worker_manager.workers[worker.worker_id] = worker
        worker_manager.add_to_group(worker)

    group_a_workers = worker_manager.get_workers_in_group("group-a")
    assert len(group_a_workers) == 2
    assert "prefill-1" in [w.worker_id for w in group_a_workers]
    assert "decode-1" in [w.worker_id for w in group_a_workers]

    group_b_workers = worker_manager.get_workers_in_group("group-b")
    assert len(group_b_workers) == 2
    assert "combined-1" in [w.worker_id for w in group_b_workers]
    assert "combined-2" in [w.worker_id for w in group_b_workers]


def test_worker_manager_get_prefill_workers_in_group(
    worker_manager: WorkerManager, workers_with_groups: list[WorkerInfo]
):
    """测试获取group内的prefill工作器"""
    for worker in workers_with_groups:
        worker_manager.workers[worker.worker_id] = worker
        worker_manager.add_to_group(worker)

    prefill_workers = worker_manager.get_prefill_workers_in_group("group-a", "test-model")
    assert len(prefill_workers) == 1
    assert prefill_workers[0].worker_id == "prefill-1"

    prefill_workers_b = worker_manager.get_prefill_workers_in_group("group-b", "test-model")
    assert len(prefill_workers_b) == 2
    assert "combined-1" in [w.worker_id for w in prefill_workers_b]
    assert "combined-2" in [w.worker_id for w in prefill_workers_b]


def test_worker_manager_get_decode_workers_in_group(
    worker_manager: WorkerManager, workers_with_groups: list[WorkerInfo]
):
    """测试获取group内的decode工作器"""
    for worker in workers_with_groups:
        worker_manager.workers[worker.worker_id] = worker
        worker_manager.add_to_group(worker)

    decode_workers = worker_manager.get_decode_workers_in_group("group-a", "test-model")
    assert len(decode_workers) == 1
    assert decode_workers[0].worker_id == "decode-1"

    decode_workers_b = worker_manager.get_decode_workers_in_group("group-b", "test-model")
    assert len(decode_workers_b) == 2
    assert "combined-1" in [w.worker_id for w in decode_workers_b]
    assert "combined-2" in [w.worker_id for w in decode_workers_b]


def test_worker_manager_is_combined_group(worker_manager: WorkerManager, workers_with_groups: list[WorkerInfo]):
    """测试判断组合型group"""
    for worker in workers_with_groups:
        worker_manager.workers[worker.worker_id] = worker
        worker_manager.add_to_group(worker)

    assert worker_manager.is_combined_group("group-a") is False
    assert worker_manager.is_combined_group("group-b") is True
    assert worker_manager.is_combined_group("default") is False


def test_worker_manager_get_group_model(worker_manager: WorkerManager, workers_with_groups: list[WorkerInfo]):
    """测试获取group的模型类型"""
    for worker in workers_with_groups:
        worker_manager.workers[worker.worker_id] = worker
        worker_manager.add_to_group(worker)

    assert worker_manager.get_group_model("group-a") == "test-model"
    assert worker_manager.get_group_model("group-b") == "test-model"
    assert worker_manager.get_group_model("nonexistent") is None


def test_worker_manager_validate_group_consistency(worker_manager: WorkerManager):
    """测试验证group一致性"""
    combined_worker = WorkerInfo(
        worker_id="combined-1",
        model="test-model",
        url="http://localhost:8001/v1",
        available_memory=1000000,
        current_load=10,
        cached_prefixes=[],
        worker_type=WorkerType.COMBINED,
        group="group-a",
    )
    worker_manager.workers["combined-1"] = combined_worker
    worker_manager.add_to_group(combined_worker)

    decode_worker = WorkerInfo(
        worker_id="decode-1",
        model="test-model",
        url="http://localhost:8002/v1",
        available_memory=800000,
        current_load=5,
        cached_prefixes=[],
        worker_type=WorkerType.DECODE,
        group="group-a",
    )
    assert worker_manager.validate_group_consistency(decode_worker) is False

    combined_worker2 = WorkerInfo(
        worker_id="combined-2",
        model="test-model",
        url="http://localhost:8003/v1",
        available_memory=1200000,
        current_load=3,
        cached_prefixes=[],
        worker_type=WorkerType.COMBINED,
        group="group-a",
    )
    assert worker_manager.validate_group_consistency(combined_worker2) is True

    wrong_model_worker = WorkerInfo(
        worker_id="wrong-model",
        model="other-model",
        url="http://localhost:8004/v1",
        available_memory=1000000,
        current_load=10,
        cached_prefixes=[],
        worker_type=WorkerType.COMBINED,
        group="group-a",
    )
    assert worker_manager.validate_group_consistency(wrong_model_worker) is False


def test_worker_manager_add_remove_to_group(worker_manager: WorkerManager):
    """测试添加和移除工作器到group"""
    worker1 = WorkerInfo(
        worker_id="worker-1",
        model="test-model",
        url="http://localhost:8001/v1",
        available_memory=1000000,
        current_load=10,
        cached_prefixes=[],
        group="group-a",
    )
    worker2 = WorkerInfo(
        worker_id="worker-2",
        model="test-model",
        url="http://localhost:8002/v1",
        available_memory=800000,
        current_load=5,
        cached_prefixes=[],
        group="group-a",
    )

    worker_manager.add_to_group(worker1)
    worker_manager.add_to_group(worker2)

    assert len(worker_manager.groups["group-a"]) == 2

    worker_manager.remove_from_group(worker1)
    assert len(worker_manager.groups["group-a"]) == 1

    worker_manager.remove_from_group(worker2)
    assert "group-a" not in worker_manager.groups
