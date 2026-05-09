"""测试API模块"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from openjiuwentools.infer_router.api.server import get_app
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint, WorkerInfo, WorkerType


def create_worker_info(worker_id: str) -> WorkerInfo:
    """创建WorkerInfo实例"""
    return WorkerInfo(
        worker_id=worker_id,
        model="test-model",
        url="http://localhost:8000/v1",
        available_memory=1000000,
        current_load=10,
        cached_prefixes=[],
        worker_type=WorkerType.COMBINED,
        group="test-group",
    )


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(get_app())


def test_health_endpoint(client):
    """测试健康检查端点"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data


def test_metrics_endpoint(client):
    """测试指标端点"""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_chat_completions_endpoint(client):
    """测试聊天完成端点"""

    async def mock_forward_request(*args, **kwargs):
        return {
            "id": "chatcmpl-test",
            "choices": [{"message": {"content": "Hello!"}}],
        }

    with patch("openjiuwentools.infer_router.api.server.settings") as mock_settings:
        mock_settings.enable_auth = False

        with patch("openjiuwentools.infer_router.api.server.preprocessor") as mock_preprocessor:
            mock_route_hint = RouteHint(
                priority=0,
                estimated_output_tokens=128,
                next_turn_prefill=False,
                request_id="chatcmpl-test",
                model="test-model",
                prefix_id=None,
                total_requests=10,
                iat=250,
                token_ids=[
                    60,
                    124,
                    117,
                    115,
                    101,
                    114,
                    124,
                    62,
                    72,
                    101,
                    108,
                    108,
                    111,
                ],
            )
            mock_preprocessor.process.return_value = mock_route_hint

            with patch("openjiuwentools.infer_router.api.server.worker_manager") as mock_manager:
                # 模拟有健康的工作器
                mock_manager.get_healthy_workers.return_value = ["worker-1"]
                mock_manager.forward_request = mock_forward_request
                mock_manager.get_worker.side_effect = create_worker_info

                with patch("openjiuwentools.infer_router.api.server.router") as mock_router:
                    mock_router.route.return_value = "worker-1"
                    mock_router.route_to_workers.return_value = ("worker-1", None)

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
    """测试带Agent Hints的聊天完成请求"""

    async def mock_forward_request(*args, **kwargs):
        return {
            "id": "chatcmpl-test",
            "choices": [{"message": {"content": "Hello!"}}],
        }

    with patch("openjiuwentools.infer_router.api.server.settings") as mock_settings:
        mock_settings.enable_auth = False

        with patch("openjiuwentools.infer_router.api.server.preprocessor") as mock_preprocessor:
            mock_route_hint = RouteHint(
                priority=10,
                estimated_output_tokens=100,
                next_turn_prefill=True,
                request_id="chatcmpl-test",
                model="test-model",
                prefix_id=None,
                total_requests=10,
                iat=250,
                token_ids=[
                    60,
                    124,
                    117,
                    115,
                    101,
                    114,
                    124,
                    62,
                    72,
                    101,
                    108,
                    108,
                    111,
                ],
            )
            mock_preprocessor.process.return_value = mock_route_hint

            with patch("openjiuwentools.infer_router.api.server.worker_manager") as mock_manager:
                # 模拟有健康的工作器
                mock_manager.get_healthy_workers.return_value = ["worker-1"]
                mock_manager.forward_request = mock_forward_request
                mock_manager.get_worker.side_effect = create_worker_info

                with patch("openjiuwentools.infer_router.api.server.router") as mock_router:
                    mock_router.route.return_value = "worker-1"
                    mock_router.route_to_workers.return_value = ("worker-1", None)

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


def test_chat_completions_invalid_request(client):
    """测试无效的聊天完成请求"""
    with patch("openjiuwentools.infer_router.api.server.settings") as mock_settings:
        mock_settings.enable_auth = False

        response = client.post(
            "/v1/chat/completions",
            json={
                # 缺少model字段
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 422


def test_health_routing_structure(client):
    """测试健康检查接口中的routing结构"""
    health_response = client.get("/health")
    assert health_response.status_code == 200
    health_data = health_response.json()

    # 验证routing节点存在
    assert "routing" in health_data
    routing = health_data["routing"]
    assert "total" in routing
    assert "by_worker" in routing
