"""测试etcd发现模块"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from openjiuwentools.infer_router.discovery.etcd_discovery import EtcdDiscovery


@pytest.fixture
async def etcd_discovery():
    """创建etcd发现实例"""
    with patch("httpx.AsyncClient") as mock_client_class:
        # 模拟HTTP客户端
        mock_client = Mock()
        mock_client.aclose = AsyncMock()  # 使aclose是异步的
        mock_client_class.return_value = mock_client

        discovery = EtcdDiscovery(
            etcd_hosts=["localhost"],
            etcd_port=2379,
            etcd_prefix="/test/workers",
            etcd_user=None,
            etcd_password=None,
        )

        # 提前设置_client，避免None，并保存引用
        discovery.client = mock_client

        # 保存原始client引用，用于验证
        discovery.test_client = mock_client

        yield discovery


@pytest.mark.asyncio
async def test_etcd_discovery_discover(etcd_discovery):
    """测试etcd发现"""
    # 模拟etcd HTTP响应
    mock_response = Mock()
    mock_response.json.return_value = {
        "node": {
            "nodes": [
                {
                    "key": "/test/workers/worker1",
                    "value": '{"worker_id": "worker-1", "model": "test-model", "url": "http://localhost:8001/v1"}',
                },
                {
                    "key": "/test/workers/worker2",
                    "value": '{"worker_id": "worker-2", "model": "test-model", "url": "http://localhost:8002/v1"}',
                },
            ]
        }
    }
    mock_response.raise_for_status = Mock()

    etcd_discovery.client.get = AsyncMock(return_value=mock_response)

    workers = await etcd_discovery.discover()

    assert len(workers) == 2
    assert workers[0].worker_id == "worker-1"
    assert workers[1].worker_id == "worker-2"


@pytest.mark.asyncio
async def test_etcd_discovery_discover_empty(etcd_discovery):
    """测试etcd发现空结果"""
    # 模拟空响应
    mock_response = Mock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status = Mock()

    etcd_discovery.client.get = AsyncMock(return_value=mock_response)

    workers = await etcd_discovery.discover()

    assert len(workers) == 0


@pytest.mark.asyncio
async def test_etcd_discovery_discover_error(etcd_discovery):
    """测试etcd发现错误"""
    import httpx

    # 模拟HTTP错误
    etcd_discovery.client.get = AsyncMock(side_effect=httpx.HTTPError("etcd error"))

    workers = await etcd_discovery.discover()

    assert len(workers) == 0


@pytest.mark.asyncio
async def test_etcd_discovery_start_stop(etcd_discovery):
    """测试etcd发现启动和停止"""
    await etcd_discovery.start()
    await etcd_discovery.stop()

    # 验证aclose被调用（检查Mock是否记录了调用）
    assert etcd_discovery.test_client.aclose.called
    assert etcd_discovery.test_client.aclose.call_count == 1


@pytest.mark.asyncio
async def test_etcd_discovery_invalid_json(etcd_discovery):
    """测试etcd发现无效JSON"""
    # 模拟无效JSON
    mock_response = Mock()
    mock_response.json.return_value = {
        "node": {"nodes": [{"key": "/test/workers/worker1", "value": "invalid json"}]}
    }
    mock_response.raise_for_status = Mock()

    etcd_discovery.client.get = AsyncMock(return_value=mock_response)

    workers = await etcd_discovery.discover()

    assert len(workers) == 0
