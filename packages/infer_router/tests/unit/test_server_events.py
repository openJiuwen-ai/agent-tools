from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openjiuwentools.infer_router.api.server import (
    DisaggWorkerParams,
    _handle_combined_worker,
    _handle_disagg_workers,
)
from openjiuwentools.infer_router.schemas.agent_hints import ChatCompletionRequest


class TestServerEvents:
    """测试 server.py 中的事件生成"""

    @pytest.fixture
    def mock_worker(self):
        """创建 mock worker"""
        worker = MagicMock()
        worker.worker_id = "test-worker"
        worker.url = "http://localhost:8000"
        return worker

    @pytest.fixture
    def mock_route_hint(self):
        """创建 mock route_hint"""
        route_hint = MagicMock()
        route_hint.token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        return route_hint

    @pytest.fixture
    def mock_chat_request(self):
        """创建 mock chat_request"""
        request = MagicMock(spec=ChatCompletionRequest)
        request.stream = False
        request.jiuwenext = None
        return request

    @pytest.mark.asyncio
    async def test_combined_worker_events(self, mock_worker, mock_route_hint, mock_chat_request):
        """测试组合型 worker 的事件生成"""
        # 创建 mock 对象
        mock_worker_manager = MagicMock()
        mock_event_generator = MagicMock()
        mock_event_manager = MagicMock()

        mock_worker_manager.get_worker = MagicMock(return_value=mock_worker)
        mock_worker_manager.forward_request = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello, world!"}}]}
        )

        mock_event_generator.generate_events = MagicMock(return_value=[])

        # 记录事件
        events_received = []

        def capture_event(event):
            events_received.append(event)

        mock_event_manager.process_event = capture_event

        with patch("openjiuwentools.infer_router.api.server.worker_manager", mock_worker_manager):
            with patch(
                "openjiuwentools.infer_router.api.server.event_generator", mock_event_generator
            ):
                with patch(
                    "openjiuwentools.infer_router.api.server.event_manager", mock_event_manager
                ):
                    await _handle_combined_worker(
                        worker_id="test-worker",
                        chat_request=mock_chat_request,
                        route_hint=mock_route_hint,
                        max_tokens=100,
                        prefill_start_time=0.0,
                    )

                    # 验证事件生成（decode_end事件现在在响应返回后生成）
                    event_types = [e.event_type for e in events_received]
                    assert "decode_start" in event_types, "应该生成 decode_start 事件"

                    # 验证 decode_start 事件内容
                    decode_start_event = next(
                        e for e in events_received if e.event_type == "decode_start"
                    )
                    assert decode_start_event.worker_id == "test-worker"
                    assert decode_start_event.token_count == 16  # route_hint.token_ids 的长度
                    assert decode_start_event.engine_specific.get("osl") == 100  # max_tokens

    @pytest.mark.asyncio
    async def test_disagg_workers_events(self, mock_route_hint, mock_chat_request):
        """测试非组合型 worker（Mooncake模式）的事件生成"""
        # 创建 mock workers
        prefill_worker = MagicMock()
        prefill_worker.worker_id = "prefill-worker"
        prefill_worker.url = "http://localhost:8001"
        prefill_worker.kv_addr = "http://localhost:8001/kv"
        prefill_worker.dp_rank = 0

        decode_worker = MagicMock()
        decode_worker.worker_id = "decode-worker"
        decode_worker.url = "http://localhost:8002"
        decode_worker.kv_addr = "http://localhost:8002/kv"

        # 创建 mock 对象
        mock_worker_manager = MagicMock()
        mock_event_generator = MagicMock()
        mock_event_manager = MagicMock()

        mock_worker_manager.get_worker = MagicMock(side_effect=[prefill_worker, decode_worker])
        mock_worker_manager.forward_request = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello, world!"}}]}
        )

        mock_event_generator.generate_events = MagicMock(return_value=[])

        # 记录事件
        events_received = []

        def capture_event(event):
            events_received.append(event)

        mock_event_manager.process_event = capture_event

        with patch("openjiuwentools.infer_router.api.server.worker_manager", mock_worker_manager):
            with patch(
                "openjiuwentools.infer_router.api.server.event_generator", mock_event_generator
            ):
                with patch(
                    "openjiuwentools.infer_router.api.server.event_manager", mock_event_manager
                ):
                    await _handle_disagg_workers(
                        DisaggWorkerParams(
                            prefill_id="prefill-worker",
                            decode_id="decode-worker",
                            max_tokens=100,
                            prefill_start_time=0.0,
                        ),
                        chat_request=mock_chat_request,
                        route_hint=mock_route_hint,
                    )

                    # 验证事件生成（decode_end事件现在在响应返回后生成）
                    event_types = [e.event_type for e in events_received]
                    assert "prefill_start" in event_types, "应该生成 prefill_start 事件"
                    assert "prefill_end" in event_types, "应该生成 prefill_end 事件"
                    assert "decode_start" in event_types, "应该生成 decode_start 事件"

                    # 验证 prefill_start 事件
                    prefill_start_event = next(
                        e for e in events_received if e.event_type == "prefill_start"
                    )
                    assert prefill_start_event.worker_id == "prefill-worker"
                    assert prefill_start_event.token_count == 16

                    # 验证 prefill_end 事件
                    prefill_end_event = next(
                        e for e in events_received if e.event_type == "prefill_end"
                    )
                    assert prefill_end_event.worker_id == "prefill-worker"
                    assert prefill_end_event.token_count == 16

                    # 验证 decode_start 事件
                    decode_start_event = next(
                        e for e in events_received if e.event_type == "decode_start"
                    )
                    assert decode_start_event.worker_id == "decode-worker"
                    assert decode_start_event.token_count == 16  # route_hint.token_ids 的长度
                    assert decode_start_event.engine_specific.get("osl") == 100

    @pytest.mark.asyncio
    async def test_disagg_workers_prefill_failure(self, mock_route_hint, mock_chat_request):
        """测试 prefill 请求失败时也会生成 prefill_end 事件"""
        # 创建 mock workers
        prefill_worker = MagicMock()
        prefill_worker.worker_id = "prefill-worker"
        prefill_worker.url = "http://localhost:8001"
        prefill_worker.kv_addr = "http://localhost:8001/kv"
        prefill_worker.dp_rank = 0

        decode_worker = MagicMock()
        decode_worker.worker_id = "decode-worker"
        decode_worker.url = "http://localhost:8002"

        # 创建 mock 对象
        mock_worker_manager = MagicMock()
        mock_event_generator = MagicMock()
        mock_event_manager = MagicMock()

        mock_worker_manager.get_worker = MagicMock(side_effect=[prefill_worker, decode_worker])
        mock_worker_manager.forward_request = AsyncMock(side_effect=Exception("Connection error"))

        mock_event_generator.generate_events = MagicMock(return_value=[])

        # 记录事件
        events_received = []

        def capture_event(event):
            events_received.append(event)

        mock_event_manager.process_event = capture_event

        with patch("openjiuwentools.infer_router.api.server.worker_manager", mock_worker_manager):
            with patch(
                "openjiuwentools.infer_router.api.server.event_generator", mock_event_generator
            ):
                with patch(
                    "openjiuwentools.infer_router.api.server.event_manager", mock_event_manager
                ):
                    with pytest.raises(Exception, match="Connection error"):
                        await _handle_disagg_workers(
                            DisaggWorkerParams(
                                prefill_id="prefill-worker",
                                decode_id="decode-worker",
                                max_tokens=100,
                                prefill_start_time=0.0,
                            ),
                            chat_request=mock_chat_request,
                            route_hint=mock_route_hint,
                        )

                    # 即使失败，也应该生成 prefill_start 和 prefill_end 事件
                    event_types = [e.event_type for e in events_received]
                    assert "prefill_start" in event_types, "应该生成 prefill_start 事件"
                    assert "prefill_end" in event_types, "即使失败也应该生成 prefill_end 事件"

    @pytest.mark.asyncio
    async def test_decode_end_with_agent_hints(self, mock_route_hint):
        """测试带有 agent_hints 的请求的 decode_start 事件（decode_end事件现在在响应返回后生成）"""
        # 创建带有 agent_hints 的请求
        chat_request = MagicMock(spec=ChatCompletionRequest)
        chat_request.stream = False

        agent_hints = MagicMock()
        agent_hints.estimated_output_tokens = 50

        jiuwenext = MagicMock()
        jiuwenext.agent_hints = agent_hints
        chat_request.jiuwenext = jiuwenext

        # 创建 mock workers
        prefill_worker = MagicMock()
        prefill_worker.worker_id = "prefill-worker"
        prefill_worker.url = "http://localhost:8001"
        prefill_worker.kv_addr = "http://localhost:8001/kv"
        prefill_worker.dp_rank = 0

        decode_worker = MagicMock()
        decode_worker.worker_id = "decode-worker"
        decode_worker.url = "http://localhost:8002"

        # 创建 mock 对象
        mock_worker_manager = MagicMock()
        mock_event_generator = MagicMock()
        mock_event_manager = MagicMock()

        mock_worker_manager.get_worker = MagicMock(side_effect=[prefill_worker, decode_worker])
        mock_worker_manager.forward_request = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello, world!"}}]}
        )

        mock_event_generator.generate_events = MagicMock(return_value=[])

        # 记录事件
        events_received = []

        def capture_event(event):
            events_received.append(event)

        mock_event_manager.process_event = capture_event

        with patch("openjiuwentools.infer_router.api.server.worker_manager", mock_worker_manager):
            with patch(
                "openjiuwentools.infer_router.api.server.event_generator", mock_event_generator
            ):
                with patch(
                    "openjiuwentools.infer_router.api.server.event_manager", mock_event_manager
                ):
                    await _handle_disagg_workers(
                        DisaggWorkerParams(
                            prefill_id="prefill-worker",
                            decode_id="decode-worker",
                            max_tokens=100,
                            prefill_start_time=0.0,
                        ),
                        chat_request=chat_request,
                        route_hint=mock_route_hint,
                    )

                    # 验证 decode_start 事件使用 agent_hints 的 estimated_output_tokens
                    decode_start_event = next(
                        e for e in events_received if e.event_type == "decode_start"
                    )
                    assert decode_start_event.engine_specific.get("osl") == 50  # 来自 agent_hints
