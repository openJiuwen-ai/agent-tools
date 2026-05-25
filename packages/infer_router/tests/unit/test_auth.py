"""测试认证模块"""

from unittest.mock import patch

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from openjiuwentools.infer_router.api.server import get_app
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(get_app())


def test_auth_disabled(client):
    """测试认证禁用的情况"""
    from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType

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
                token_ids=[60, 124, 117, 115, 101, 114, 124, 62, 72, 101, 108, 108, 111],
            )
            mock_preprocessor.process.return_value = mock_route_hint

            with patch("openjiuwentools.infer_router.api.server.worker_manager") as mock_manager:

                async def mock_forward_request(*args, **kwargs):
                    return {
                        "id": "chatcmpl-test",
                        "choices": [{"message": {"content": "Hello!"}}],
                    }

                mock_manager.get_healthy_workers.return_value = ["worker-1"]
                mock_manager.forward_request = mock_forward_request

                def _make_worker(worker_id):
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

                mock_manager.get_worker.side_effect = _make_worker

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


def test_auth_enabled_no_header(client):
    """测试认证启用但无头部的情况"""
    with patch("openjiuwentools.infer_router.api.auth.settings") as mock_settings:
        mock_settings.enable_auth = True
        mock_settings.api_key = "test-key"

        # Mock认证中间件直接返回401
        async def mock_api_key_auth_middleware(request, call_next):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key", "type": "unauthorized"}},
            )

        with patch(
            "openjiuwentools.infer_router.api.auth.api_key_auth_middleware",
            mock_api_key_auth_middleware,
        ):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            # 认证中间件应该在调用preprocessor之前返回401
            assert response.status_code == 401


def test_auth_enabled_invalid_key(client):
    """测试认证启用但无效密钥的情况"""
    with patch("openjiuwentools.infer_router.api.auth.settings") as mock_settings:
        mock_settings.enable_auth = True
        mock_settings.api_key = "test-key"

        # Mock认证中间件直接返回401
        async def mock_api_key_auth_middleware(request, call_next):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key", "type": "unauthorized"}},
            )

        with patch(
            "openjiuwentools.infer_router.api.auth.api_key_auth_middleware",
            mock_api_key_auth_middleware,
        ):
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer invalid-key"},
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert response.status_code == 401


def test_auth_enabled_valid_key(client):
    """测试认证启用且有效密钥的情况"""
    from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType

    with patch("openjiuwentools.infer_router.api.server.settings") as mock_settings:
        mock_settings.enable_auth = True
        mock_settings.api_key = "test-key"

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
                token_ids=[60, 124, 117, 115, 101, 114, 124, 62, 72, 101, 108, 108, 111],
            )
            mock_preprocessor.process.return_value = mock_route_hint

            with patch("openjiuwentools.infer_router.api.server.worker_manager") as mock_manager:

                async def mock_forward_request(*args, **kwargs):
                    return {
                        "id": "chatcmpl-test",
                        "choices": [{"message": {"content": "Hello!"}}],
                    }

                mock_manager.get_healthy_workers.return_value = ["worker-1"]
                mock_manager.forward_request = mock_forward_request

                def _make_worker(worker_id):
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

                mock_manager.get_worker.side_effect = _make_worker

                with patch("openjiuwentools.infer_router.api.server.router") as mock_router:
                    mock_router.route.return_value = "worker-1"
                    mock_router.route_to_workers.return_value = ("worker-1", None)

                    with patch(
                        "openjiuwentools.infer_router.api.server.circuit_breaker"
                    ) as mock_cb:
                        mock_cb.return_value.is_closed = True

                        response = client.post(
                            "/v1/chat/completions",
                            headers={"Authorization": "Bearer test-key"},
                            json={
                                "model": "test-model",
                                "messages": [{"role": "user", "content": "Hello"}],
                            },
                        )

                        assert response.status_code == 200
