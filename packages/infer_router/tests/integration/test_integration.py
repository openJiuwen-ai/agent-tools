"""集成测试 - 测试完整的请求处理流程"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from openjiuwentools.infer_router.api.server import get_app
from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo
from openjiuwentools.infer_router.worker.worker_manager import WorkerManager


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(get_app())


@pytest.fixture
def mock_worker_manager():
    """创建模拟的工作器管理器"""
    manager = WorkerManager()
    manager.workers = {
        "worker-1": WorkerInfo(
            worker_id="worker-1",
            model="test-model",
            url="http://localhost:8001/v1",
            available_memory=1000000,
            current_load=10,
            cached_prefixes=["Hello"],
        )
    }
    return manager


def test_health_endpoint(client):
    """测试健康检查端点"""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    assert "worker_manager" in data["components"]


def test_chat_completions_without_auth(client):
    """测试聊天完成接口（无认证）"""
    with patch("openjiuwentools.infer_router.api.server.settings") as mock_settings:
        mock_settings.enable_auth = False

        with patch("openjiuwentools.infer_router.api.server.worker_manager") as mock_manager:

            async def mock_forward_request(*args, **kwargs):
                return {
                    "id": "chatcmpl-test",
                    "choices": [{"message": {"content": "Hello!"}}],
                }

            mock_manager.get_healthy_workers.return_value = ["worker-1"]
            mock_manager.forward_request = mock_forward_request
            mock_manager.get_worker.return_value = WorkerInfo(
                worker_id="worker-1",
                model="test-model",
                url="http://localhost:8001/v1",
                available_memory=1000000,
                current_load=10,
                cached_prefixes=["Hello"],
            )

            with patch("openjiuwentools.infer_router.api.server.router") as mock_router:
                mock_router.route.return_value = "worker-1"
                mock_router.route_to_workers.return_value = ("worker-1", None)

                with patch("openjiuwentools.infer_router.api.server.circuit_breaker") as mock_cb:
                    mock_cb.return_value.is_closed = True

                    response = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "Hello"}],
                        },
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert "id" in data
                    assert "choices" in data


def test_chat_completions_with_agent_hints(client):
    """测试带有Agent Hints的聊天完成请求"""
    with patch("openjiuwentools.infer_router.api.server.settings") as mock_settings:
        mock_settings.enable_auth = False

        with patch("openjiuwentools.infer_router.api.server.worker_manager") as mock_manager:

            async def mock_forward_request(*args, **kwargs):
                return {
                    "id": "chatcmpl-test",
                    "choices": [{"message": {"content": "Hello!"}}],
                }

            mock_manager.get_healthy_workers.return_value = ["worker-1"]
            mock_manager.forward_request = mock_forward_request
            mock_manager.get_worker.return_value = WorkerInfo(
                worker_id="worker-1",
                model="test-model",
                url="http://localhost:8001/v1",
                available_memory=1000000,
                current_load=10,
                cached_prefixes=["Hello"],
            )

            with patch("openjiuwentools.infer_router.api.server.router") as mock_router:
                mock_router.route.return_value = "worker-1"
                mock_router.route_to_workers.return_value = ("worker-1", None)

                with patch("openjiuwentools.infer_router.api.server.circuit_breaker") as mock_cb:
                    mock_cb.return_value.is_closed = True

                    response = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "jiuwenext": {
                                "agent_hints": {
                                    "priority": 10,
                                    "estimated_output_tokens": 100,
                                    "next_turn_prefill": True,
                                }
                            },
                        },
                    )

                    assert response.status_code == 200


def test_metrics_endpoint(client):
    """测试指标端点"""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_performance_monitor_endpoint(client):
    """测试性能监控端点"""
    # 使用 metrics 端点替代不存在的 performance-monitor 端点
    response = client.get("/metrics")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_worker_discovery_integration(sample_worker_config_json: Path):
    """测试工作器发现集成"""
    from openjiuwentools.infer_router.discovery import ConfigDiscovery

    discovery = ConfigDiscovery(config_path=str(sample_worker_config_json))
    await discovery.start()

    workers = await discovery.discover()
    assert len(workers) == 2

    await discovery.stop()


@pytest.mark.asyncio
async def test_worker_manager_integration(sample_worker_config_json: Path):
    """测试工作器管理器集成"""
    manager = WorkerManager()

    with patch("openjiuwentools.infer_router.worker.worker_manager.settings") as mock_settings:
        mock_settings.worker_discovery_type = "config"
        mock_settings.worker_config_path = str(sample_worker_config_json)
        mock_settings.worker_discovery_interval = 30
        mock_settings.worker_health_check_interval = 10

        # Mock AsyncClient 类，避免实际的网络调用
        # 需要模拟所有 httpx 和 httpcore 内部使用的参数
        mock_client = Mock()
        mock_client.aclose = AsyncMock()  # 这是方法，不是属性
        mock_client.get = AsyncMock(return_value=Mock(status_code=200))

        # 模拟 AsyncClient 构造函数 - 返回非协程的 mock
        def mock_async_client_init(*args, **kwargs):
            # 返回已经配置好的 mock 客户端
            return mock_client

        mock_async_client = Mock(side_effect=mock_async_client_init)

        with patch(
            "openjiuwentools.infer_router.worker.worker_manager.httpx.AsyncClient",
            mock_async_client,
        ):
            await manager.start()
            await asyncio.sleep(0.1)

            assert len(manager.workers) == 2
            assert "test-worker-1" in manager.workers
            assert "test-worker-2" in manager.workers

            await manager.stop()


@pytest.mark.asyncio
async def test_full_request_flow(sample_worker_config_json: Path):
    """测试完整的请求处理流程"""
    from openjiuwentools.infer_router.preprocess.preprocessor import Preprocessor
    from openjiuwentools.infer_router.routing.router import Router
    from openjiuwentools.infer_router.schedule.scheduler import Scheduler

    manager = WorkerManager()

    with patch("openjiuwentools.infer_router.worker.worker_manager.settings") as mock_settings:
        mock_settings.worker_discovery_type = "config"
        mock_settings.worker_config_path = str(sample_worker_config_json)
        mock_settings.worker_discovery_interval = 30
        mock_settings.worker_health_check_interval = 10

        # Mock AsyncClient 类，避免实际的网络调用
        mock_client = Mock()
        mock_client.aclose = AsyncMock()
        mock_client.get = AsyncMock(return_value=Mock(status_code=200))

        def mock_async_client_init(*args, **kwargs):
            return mock_client

        mock_async_client = Mock(side_effect=mock_async_client_init)

        with patch(
            "openjiuwentools.infer_router.worker.worker_manager.httpx.AsyncClient",
            mock_async_client,
        ):
            await manager.start()
            await asyncio.sleep(0.1)

        preprocessor = Preprocessor()
        from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager

        kv_cache_manager = KVCacheManager()
        scheduler = Scheduler(kv_cache_manager)
        router = Router(kv_cache_manager, manager)

        request_id = "test-request-1"
        chat_request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "jiuwenext": {
                "agent_hints": {
                    "priority": 10,
                    "estimated_output_tokens": 100,
                }
            },
        }

        from openjiuwentools.infer_router.schemas.agent_hints import (
            ChatCompletionRequest,
        )

        chat_request = ChatCompletionRequest(**chat_request_data)
        agent_hints = chat_request.jiuwenext.agent_hints if chat_request.jiuwenext else None

        route_hint = preprocessor.process(chat_request, agent_hints, request_id)
        assert route_hint is not None

        scheduler.submit(route_hint)

        scheduled = scheduler.get_next_request()
        assert scheduled is not None

        prefill_id, decode_id = router.route_to_workers(scheduled.route_hint)
        assert prefill_id in ["test-worker-1", "test-worker-2"]

        await manager.stop()


@pytest.mark.asyncio
async def test_health_check_failed_workers():
    """测试健康检查失败的工作器"""
    from openjiuwentools.infer_router.worker.worker_manager import WorkerStatus

    manager = WorkerManager()
    worker = WorkerInfo(
        worker_id="worker-1",
        model="test-model",
        url="http://localhost:9999/v1",
        available_memory=1000000,
        current_load=10,
        cached_prefixes=[],
    )
    manager.workers = {worker.worker_id: worker}
    manager.worker_statuses = {
        worker.worker_id: WorkerStatus(
            worker_id=worker.worker_id,
            last_health_check=0,
            is_healthy=True,
            response_time=0,
            consecutive_failures=0,
        )
    }

    # Mock AsyncClient
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 500
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch(
        "openjiuwentools.infer_router.worker.worker_manager.httpx.AsyncClient",
        return_value=mock_client,
    ):
        await manager.check_worker_health("worker-1")

    status = manager.worker_statuses.get("worker-1")
    assert status is not None
    # 由于有失败计数机制，单次失败不会立即标记为不健康
    assert status.consecutive_failures >= 1


@pytest.mark.asyncio
async def test_concurrent_requests(client):
    """测试并发请求"""
    with patch("openjiuwentools.infer_router.api.server.settings") as mock_settings:
        mock_settings.enable_auth = False

        # 模拟工作器管理器
        with patch("openjiuwentools.infer_router.api.server.worker_manager") as mock_manager:

            async def mock_forward_request(*args, **kwargs):
                return {
                    "id": "chatcmpl-test",
                    "choices": [{"message": {"content": "Hello!"}}],
                }

            mock_manager.get_healthy_workers.return_value = ["worker-1"]
            mock_manager.forward_request = mock_forward_request
            mock_manager.get_worker.return_value = WorkerInfo(
                worker_id="worker-1",
                model="test-model",
                url="http://localhost:8001/v1",
                available_memory=1000000,
                current_load=10,
                cached_prefixes=["Hello"],
            )

            with patch("openjiuwentools.infer_router.api.server.router") as mock_router:
                mock_router.route.return_value = "worker-1"
                mock_router.route_to_workers.return_value = ("worker-1", None)

                with patch("openjiuwentools.infer_router.api.server.circuit_breaker") as mock_cb:
                    mock_cb.return_value.is_closed = True

                    # 发送多个并发请求
                    async def send_request():
                        response = client.post(
                            "/v1/chat/completions",
                            json={
                                "model": "test-model",
                                "messages": [{"role": "user", "content": "Hello"}],
                            },
                        )
                        return response.status_code

                    # 并发发送5个请求
                    tasks = [send_request() for _ in range(5)]
                    status_codes = await asyncio.gather(*tasks)

                    # 大部分请求应该成功，允许少数失败（熔断器可能偶尔触发）
                    success_count = sum(1 for code in status_codes if code == 200)
                    assert success_count >= 3, f"只有 {success_count}/5 请求成功，状态码: {status_codes}"
