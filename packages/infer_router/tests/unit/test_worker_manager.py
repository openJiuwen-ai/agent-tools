# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
"""测试工作器管理器"""

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import httpx

from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType
from openjiuwentools.infer_router.worker.worker_manager import WorkerManager, WorkerStatus


class _FakeResponse:
    """测试用 HTTP 响应对象。"""

    def __init__(self, payload: dict):
        """初始化测试响应。

        参数:
            payload: response.json() 需要返回的字典。
        """

        self._payload = payload

    @staticmethod
    def raise_for_status():
        """模拟 httpx.Response.raise_for_status。

        返回:
            无返回值。
        """

    def json(self):
        """返回测试 payload。

        返回:
            初始化时传入的字典。
        """

        return self._payload


class _FakeAsyncClient:
    """测试用异步 HTTP client。"""

    def __init__(self):
        """初始化请求记录。

        参数:
            无参数。
        """

        self.calls = []

    async def post(self, url, json=None, headers=None, timeout=None):
        """记录 POST 请求并返回固定响应。

        参数:
            url: 请求 URL。
            json: 请求 JSON 体。
            headers: 请求头。
            timeout: 请求超时配置。

        返回:
            _FakeResponse 响应对象。
        """

        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _FakeResponse({"ok": True, "prefetched_count": 1, "requested_count": 2})


class _CapacitySyncClient:
    """返回engine-core BlockPool容量的同步测试客户端。"""

    def __init__(
        self,
        free_blocks: int,
        confirmed_reusable_blocks: int = 0,
        required_new_blocks: int = 0,
        capacity_generation: int | None = None,
    ):
        """初始化容量响应。

        参数:
            free_blocks: 伪造的真实空闲block数。

        返回:
            无返回值。
        """

        self.free_blocks = free_blocks
        self.confirmed_reusable_blocks = confirmed_reusable_blocks
        self.required_new_blocks = required_new_blocks
        self.capacity_generation = capacity_generation
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        """记录容量查询并返回伪造响应。

        参数:
            url: engine-core容量接口URL。
            json: 请求JSON体。
            headers: 请求头。
            timeout: HTTP超时配置。

        返回:
            包含BlockPool容量的_FakeResponse。
        """

        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        payload = {
            "ok": True,
            "block_pool": {
                "ready": True,
                "free_blocks": self.free_blocks,
                "block_size": 128,
                "confirmed_reusable_blocks": self.confirmed_reusable_blocks,
                "required_new_blocks": self.required_new_blocks,
            },
        }
        if self.capacity_generation is not None:
            payload["block_pool"]["capacity_generation"] = self.capacity_generation
        return _FakeResponse(payload)


class _DatadistAsyncClient:
    """按接口返回 LLM-DataDist 测试响应的异步客户端。"""

    def __init__(self):
        """初始化调用记录。

        返回:
            无返回值。
        """

        self.calls = []

    async def post(self, url, json=None, headers=None, timeout=None):
        """记录请求并返回对应控制接口响应。

        参数:
            url: 控制接口 URL。
            json: JSON 请求体。
            headers: HTTP 请求头。
            timeout: HTTP 超时对象。

        返回:
            与请求路径匹配的伪响应。
        """

        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if url.endswith("/datadist/status"):
            cluster_id = 1 if "source" in url else 2
            return _FakeResponse(
                {
                    "ok": True,
                    "health": {
                        "endpoint": {
                            "cluster_id": cluster_id,
                            "listen_ip": f"10.0.0.{cluster_id}",
                            "listen_port": 19000 + cluster_id,
                        },
                        "linked_cluster_ids": [],
                    },
                }
            )
        if url.endswith("/datadist/link"):
            return _FakeResponse({"ok": True, "remote_cluster_id": 1})
        if url.endswith("/datadist/transfers/pull"):
            return _FakeResponse({"ok": True, "published_count": 2})
        return _FakeResponse({"ok": True})


class _StaleDatadistStatusClient:
    """模拟 Router 缓存未建链状态、但 Worker 实际已经建链的客户端。"""

    def __init__(self):
        """初始化目标状态查询次数和调用记录。

        返回:
            无返回值。
        """

        self.target_status_calls = 0
        self.calls = []

    async def post(self, url, json=None, headers=None, timeout=None):
        """按调用阶段返回旧状态、重复建链错误和刷新后的真实状态。

        参数:
            url: 控制接口 URL。
            json: JSON 请求体。
            headers: HTTP 请求头。
            timeout: HTTP 超时对象。

        返回:
            当前模拟阶段对应的响应。
        """

        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if url == "http://target-control:18020/datadist/status":
            self.target_status_calls += 1
            linked_ids = [] if self.target_status_calls == 1 else [1]
            return _FakeResponse(
                {
                    "ok": True,
                    "health": {
                        "endpoint": {
                            "cluster_id": 2,
                            "listen_ip": "10.0.0.2",
                            "listen_port": 19002,
                        },
                        "linked_cluster_ids": linked_ids,
                    },
                }
            )
        if url == "http://source-control:18011/datadist/status":
            return _FakeResponse(
                {
                    "ok": True,
                    "health": {
                        "endpoint": {
                            "cluster_id": 1,
                            "listen_ip": "10.0.0.1",
                            "listen_port": 19001,
                        },
                        "linked_cluster_ids": [2],
                    },
                }
            )
        if url == "http://target-control:18020/datadist/link":
            return _FakeResponse({"ok": False, "error": "duplicate link"})
        return _FakeResponse({"ok": False, "error": "unexpected request"})


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
async def test_prefetch_kv_blocks_posts_to_worker_base_url(worker_manager: WorkerManager):
    """测试 KVBM 预取请求会发到 worker 的 base URL。"""
    worker_manager.workers["worker-1"] = WorkerInfo(
        worker_id="worker-1",
        model="test-model",
        url="http://localhost:8001/v1",
        available_memory=1000000,
        current_load=0,
        cached_prefixes=[],
        api_key="token",
    )
    fake_client = _FakeAsyncClient()
    worker_manager.http_client = fake_client

    started_at_ms = int(time.time() * 1000)
    result = await worker_manager.prefetch_kv_blocks(
        "worker-1",
        ["hash-a", "hash-b"],
        request_id="req-1",
        deadline_ms=50,
    )

    assert result["ok"] is True
    assert fake_client.calls[0]["url"] == "http://localhost:8001/kvbm/prefetch"
    assert fake_client.calls[0]["json"]["block_hashes"] == ["hash-a", "hash-b"]
    deadline_epoch_ms = fake_client.calls[0]["json"]["deadline_epoch_ms"]
    assert deadline_epoch_ms >= started_at_ms + 50
    assert deadline_epoch_ms <= int(time.time() * 1000) + 50
    assert fake_client.calls[0]["headers"]["Authorization"] == "Bearer token"


@pytest.mark.asyncio
async def test_transfer_kv_blocks_links_and_starts_transaction(
    worker_manager: WorkerManager,
):
    """Router 应先建链，再调用目标 Worker 的事务 pull 接口。"""
    worker_manager.workers["source"] = WorkerInfo(
        worker_id="source",
        model="model-a",
        url="http://source:8011/v1",
        kvbm_control_url="http://source-control:18011",
    )
    worker_manager.workers["target"] = WorkerInfo(
        worker_id="target",
        model="model-a",
        url="http://target:8020/v1",
    )
    client = _DatadistAsyncClient()
    worker_manager.http_client = client

    result = await worker_manager.transfer_kv_blocks(
        "target",
        "source",
        [7, 8],
        ["hash-7", "hash-8"],
        request_id="req-1",
    )

    assert result["ok"] is True
    assert {
        "status",
        "link",
        "link_total",
        "worker_transaction_http",
        "router_transfer_total",
    }.issubset(result["phase_ms"])
    assert all(value >= 0 for value in result["phase_ms"].values())
    urls = [call["url"] for call in client.calls]
    assert "http://target:18020/datadist/link" in urls
    assert "http://target:18020/datadist/transfers/pull" in urls
    pull_call = next(call for call in client.calls if call["url"].endswith("/datadist/transfers/pull"))
    assert pull_call["json"]["source_control_url"] == "http://source-control:18011"
    assert pull_call["json"]["source_blocks"] == [7, 8]
    assert pull_call["json"]["external_block_hashes"] == ["hash-7", "hash-8"]


def test_worker_control_url_derives_default_engine_core_port() -> None:
    """控制地址缺失时应从推理端口推导默认的port+10000地址。"""

    inferred = WorkerManager._worker_control_url(
        WorkerInfo(
            worker_id="decode-derived",
            model="model-a",
            url="http://10.0.0.8:8060/v1",
        )
    )
    configured = WorkerManager._worker_control_url(
        WorkerInfo(
            worker_id="decode-configured",
            model="model-a",
            url="http://10.0.0.8:8060/v1",
            kvbm_control_url="http://10.0.0.8:29060/",
        )
    )

    assert inferred == "http://10.0.0.8:18060"
    assert configured == "http://10.0.0.8:29060"


@pytest.mark.asyncio
async def test_datadist_link_status_uses_link_timeout_and_reports_control_urls(
    worker_manager: WorkerManager,
):
    """状态检查应继承建链超时，并在失败时返回两侧实际控制地址。"""
    worker_manager.workers["source"] = WorkerInfo(
        worker_id="source",
        model="model-a",
        url="http://source:8011/v1",
        kvbm_control_url="http://source-control:18011",
    )
    worker_manager.workers["target"] = WorkerInfo(
        worker_id="target",
        model="model-a",
        url="http://target:8020/v1",
        kvbm_control_url="http://target-control:18020",
    )
    client = AsyncMock()
    client.post = AsyncMock(return_value=_FakeResponse({"ok": False, "error": "backend is not ready"}))
    worker_manager.http_client = client

    result = await worker_manager.ensure_datadist_link(
        "target",
        "source",
        timeout_ms=3500,
    )

    assert result["ok"] is False
    assert result["target_worker_id"] == "target"
    assert result["target_control_url"] == "http://target-control:18020"
    assert result["source_worker_id"] == "source"
    assert result["source_control_url"] == "http://source-control:18011"
    assert client.post.await_count == 2
    assert all(call.kwargs["timeout"].read == 3.5 for call in client.post.await_args_list)


@pytest.mark.asyncio
async def test_datadist_link_reconciles_stale_status_after_duplicate_link_error(
    worker_manager: WorkerManager,
):
    """重复建链失败后应刷新目标状态，并把已经存在的链路视为成功。"""
    worker_manager.workers["source"] = WorkerInfo(
        worker_id="source",
        model="model-a",
        url="http://source:8011/v1",
        kvbm_control_url="http://source-control:18011",
    )
    worker_manager.workers["target"] = WorkerInfo(
        worker_id="target",
        model="model-a",
        url="http://target:8020/v1",
        kvbm_control_url="http://target-control:18020",
    )
    client = _StaleDatadistStatusClient()
    worker_manager.http_client = client

    result = await worker_manager.ensure_datadist_link("target", "source")

    assert result["ok"] is True
    assert result["already_linked"] is True
    assert result["reconciled_after_link_error"] is True
    assert result["link_error"]["error"] == "duplicate link"
    assert client.target_status_calls == 2


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
        await asyncio.sleep(0)
        assert worker_manager.discovery_task.cancelled() or worker_manager.discovery_task.done()
        assert worker_manager.health_check_task.cancelled() or worker_manager.health_check_task.done()


@pytest.mark.asyncio
async def test_worker_manager_discover_workers(
    worker_manager: WorkerManager, sample_workers: list[WorkerInfo], sample_worker_config_json: Path
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
    assert len(prefill_workers_b) == 0


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
    assert len(decode_workers_b) == 0


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


def test_confirm_kvbm_block_pool_capacity_updates_worker_snapshot(
    worker_manager: WorkerManager,
):
    """派发前容量确认应使用engine-core结果刷新Worker快照。"""
    worker = WorkerInfo(
        worker_id="decode-capacity",
        model="test-model",
        url="http://127.0.0.1:8020/v1",
        kvbm_control_url="http://127.0.0.1:18020",
        worker_type=WorkerType.DECODE,
    )
    worker_manager.workers[worker.worker_id] = worker
    client = _CapacitySyncClient(free_blocks=39, capacity_generation=101)
    worker_manager._kvbm_sync_client = client

    result = worker_manager.confirm_kvbm_block_pool_capacity(
        worker.worker_id,
        timeout_ms=200,
    )

    assert result["ok"] is True
    assert result["free_blocks"] == 39
    assert result["block_size"] == 128
    assert worker.kvbm_free_blocks == 39
    assert worker.kvbm_block_size == 128
    assert worker.kvbm_capacity_generation == 101
    assert worker.kvbm_block_pool_updated_at > 0
    assert client.calls[0]["url"] == ("http://127.0.0.1:18020/datadist/block-pool/capacity")


@pytest.mark.parametrize("exception", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout])
def test_capacity_query_preserves_empty_timeout_type(worker_manager, exception):
    worker = WorkerInfo(
        worker_id="decode-timeout", model="test-model", url="http://127.0.0.1:8020", worker_type=WorkerType.DECODE
    )
    worker_manager.workers[worker.worker_id] = worker
    client = _CapacitySyncClient(free_blocks=20)

    def fail(*args, **kwargs):
        raise exception("")

    client.post = fail
    worker_manager._kvbm_sync_client = client
    result = worker_manager.confirm_kvbm_block_pool_capacity(worker.worker_id, timeout_ms=50)
    assert result["ok"] is False
    assert result["error_type"] == exception.__name__
    assert result["error"].startswith(exception.__name__)
    assert result["failure_stage"] == "http_request"


def test_confirm_kvbm_block_pool_capacity_rejects_older_generation(
    worker_manager: WorkerManager,
):
    """迟到的旧代容量响应不得覆盖Worker的新代快照。"""
    worker = WorkerInfo(
        worker_id="decode-stale-capacity",
        model="test-model",
        url="http://127.0.0.1:8020/v1",
        kvbm_control_url="http://127.0.0.1:18020",
        worker_type=WorkerType.DECODE,
        kvbm_free_blocks=20,
        kvbm_capacity_generation=102,
    )
    worker_manager.workers[worker.worker_id] = worker
    worker_manager._kvbm_sync_client = _CapacitySyncClient(
        free_blocks=669,
        capacity_generation=101,
    )

    result = worker_manager.confirm_kvbm_block_pool_capacity(
        worker.worker_id,
        timeout_ms=200,
    )

    assert result["ok"] is False
    assert "stale D Worker capacity generation" in result["error"]
    assert worker.kvbm_free_blocks == 20
    assert worker.kvbm_capacity_generation == 102


def test_confirm_kvbm_block_pool_capacity_forwards_delta_query(
    worker_manager: WorkerManager,
):
    """D容量确认应返回EngineCore真实的连续前缀复用量。"""
    worker = WorkerInfo(
        worker_id="decode-delta-capacity",
        model="test-model",
        url="http://127.0.0.1:8020/v1",
        kvbm_control_url="http://127.0.0.1:18020",
        worker_type=WorkerType.DECODE,
    )
    worker_manager.workers[worker.worker_id] = worker
    client = _CapacitySyncClient(
        free_blocks=10,
        confirmed_reusable_blocks=46,
        required_new_blocks=5,
    )
    worker_manager._kvbm_sync_client = client

    result = worker_manager.confirm_kvbm_block_pool_capacity(
        worker.worker_id,
        timeout_ms=200,
        block_hashes=["hash-0", "hash-1"],
        prompt_blocks=50,
        decode_blocks=1,
    )

    assert result["confirmed_reusable_blocks"] == 46
    assert result["required_new_blocks"] == 5
    assert client.calls[0]["json"] == {
        "block_hashes": ["hash-0", "hash-1"],
        "prompt_blocks": 50,
        "decode_blocks": 1,
        "margin_blocks": 0,
        "pending_reserved_blocks": 0,
    }


def test_confirm_kvbm_block_pool_capacity_coalesces_concurrent_queries(
    worker_manager: WorkerManager,
):
    """同一D Worker的并发容量确认应共享一次真实查询。"""
    worker = WorkerInfo(
        worker_id="decode-capacity-singleflight",
        model="test-model",
        worker_type=WorkerType.DECODE,
    )
    worker_manager.workers[worker.worker_id] = worker
    query_started = threading.Event()
    release_query = threading.Event()
    call_count = 0
    call_lock = threading.Lock()

    def fetch_capacity(worker_id, *, timeout_ms):
        nonlocal call_count
        assert worker_id == worker.worker_id
        assert timeout_ms == 500
        with call_lock:
            call_count += 1
        query_started.set()
        assert release_query.wait(timeout=1.0)
        return {"ok": True, "free_blocks": 100, "elapsed_ms": 5.0}

    worker_manager._fetch_kvbm_block_pool_capacity = fetch_capacity
    results = []

    first = threading.Thread(
        target=lambda: results.append(
            worker_manager.confirm_kvbm_block_pool_capacity(
                worker.worker_id,
                timeout_ms=500,
            )
        )
    )
    second = threading.Thread(
        target=lambda: results.append(
            worker_manager.confirm_kvbm_block_pool_capacity(
                worker.worker_id,
                timeout_ms=500,
            )
        )
    )
    first.start()
    assert query_started.wait(timeout=1.0)
    second.start()
    time.sleep(0.02)
    release_query.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert call_count == 1
    assert len(results) == 2
    assert all(result["ok"] for result in results)
    assert sorted(result["singleflight_shared"] for result in results) == [False, True]
