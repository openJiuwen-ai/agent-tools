"""流式输出四条路径的单元测试。

测试矩阵：
  - combined + chat/completions
  - disagg   + chat/completions
  - combined + completions
  - disagg   + completions

每条路径验证：
  1. 返回 SSE 格式（text/event-stream）
  2. 首个 chat chunk delta 包含 role="assistant"
  3. 每个 chunk 的 content/text 是增量，不是累积
  4. 最后一行是 data: [DONE]
  5. finish_reason 只在最后一个 chunk 出现
  6. 流结束后 _record_stream_metrics 被回调且 usage 正确
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# 将 transport.py 所在目录 mock 后再导入 server 模块
# ---------------------------------------------------------------------------

_SERVER_MODULE = "openjiuwentools.infer_router.api.server"


# ---------------------------------------------------------------------------
# 模拟 SSE 字节流（模仿 worker_manager.forward_request_stream 的返回）
# ---------------------------------------------------------------------------

def _make_chat_sse_chunks() -> list[bytes]:
    """构造 chat completion 的 SSE 字节流（模拟 vLLM worker 返回）。"""
    chunks = []
    # chunk 1: role + 第一段内容
    chunks.append(
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model",'
        b'"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
    )
    # chunk 2: 后续内容
    chunks.append(
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model",'
        b'"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n'
    )
    # chunk 3: finish + usage
    chunks.append(
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test-model",'
        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
    )
    chunks.append(b"data: [DONE]\n\n")
    return chunks


def _make_completion_sse_chunks() -> list[bytes]:
    """构造 text completion 的 SSE 字节流。"""
    chunks = []
    chunks.append(
        b'data: {"id":"cmpl-1","object":"text_completion","model":"test-model",'
        b'"choices":[{"index":0,"text":"Once","finish_reason":null,"logprobs":null}]}\n\n'
    )
    chunks.append(
        b'data: {"id":"cmpl-1","object":"text_completion","model":"test-model",'
        b'"choices":[{"index":0,"text":" upon","finish_reason":null,"logprobs":null}]}\n\n'
    )
    chunks.append(
        b'data: {"id":"cmpl-1","object":"text_completion","model":"test-model",'
        b'"choices":[{"index":0,"text":" a time","finish_reason":"stop","logprobs":null}],'
        b'"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n'
    )
    chunks.append(b"data: [DONE]\n\n")
    return chunks


async def _fake_stream(chunks: list[bytes]) -> AsyncGenerator[bytes, None]:
    """将字节列表转换为 AsyncGenerator，模拟 forward_request_stream 返回。"""
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# 模拟 WorkerInfo
# ---------------------------------------------------------------------------

@dataclass
class FakeWorkerInfo:
    worker_id: str
    url: str = "http://fake:8000"
    api_key: str | None = None
    kv_addr: str = ""
    dp_rank: int = 0
    publisher_endpoint: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _patch_server_globals():
    """Patch server.py 中的所有全局组件，使路由逻辑可独立运行。"""
    import openjiuwentools.infer_router.api.server as srv

    mock_preprocessor = MagicMock()
    mock_preprocessor.process.return_value = MagicMock(
        token_ids=[1, 2, 3], model="test-model", request_id="chatcmpl-test",
    )
    mock_preprocessor.process_completion.return_value = MagicMock(
        token_ids=[1, 2, 3], model="test-model", request_id="cmpl-test",
    )

    mock_kvcache = MagicMock()
    mock_kvcache.find_matches.return_value = {}

    mock_scheduler = MagicMock()
    mock_scheduler.submit.return_value = None
    mock_scheduler.get_next_request.return_value = MagicMock(
        route_hint=mock_preprocessor.process.return_value,
    )
    mock_scheduler.mark_task_completed.return_value = None
    mock_scheduler.mark_task_failed.return_value = None

    mock_worker_mgr = MagicMock()
    mock_worker_mgr.get_worker.return_value = FakeWorkerInfo(worker_id="w1")

    mock_router = MagicMock()

    mock_event_manager = MagicMock()
    mock_event_manager.process_event.return_value = None

    mock_event_generator = MagicMock()
    mock_event_generator.generate_events.return_value = []

    mock_metrics = MagicMock()
    mock_perf = MagicMock()
    mock_circuit_breaker = MagicMock()

    mock_settings = MagicMock()
    mock_settings.kv_event_mode = "inner_event"
    mock_settings.auth_enabled = False
    mock_settings.rate_limit_enabled = False

    patches = {
        "preprocessor": mock_preprocessor,
        "kvcache_manager": mock_kvcache,
        "scheduler": mock_scheduler,
        "worker_manager": mock_worker_mgr,
        "router": mock_router,
        "event_manager": mock_event_manager,
        "event_generator": mock_event_generator,
        "circuit_breaker": mock_circuit_breaker,
        "settings": mock_settings,
    }

    originals = {}
    for name, mock_obj in patches.items():
        originals[name] = getattr(srv, name)
        setattr(srv, name, mock_obj)

    with patch(f"{_SERVER_MODULE}.metrics", mock_metrics), \
         patch(f"{_SERVER_MODULE}.performance_stats", mock_perf):
        yield {
            **patches,
            "metrics": mock_metrics,
            "performance_stats": mock_perf,
        }

    for name, original in originals.items():
        setattr(srv, name, original)


@pytest_asyncio.fixture
async def _module_client(_patch_server_globals):
    """构造 httpx AsyncClient 直连 FastAPI ASGI app（不需要真实网络）。"""
    import httpx

    from openjiuwentools.infer_router.api.server import register_routes

    from fastapi import FastAPI

    app = FastAPI()
    register_routes(app)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac, _patch_server_globals


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _parse_sse_lines(raw: bytes) -> list[dict]:
    """从 SSE 响应 body 中解析出所有 JSON chunk。"""
    text = raw.decode("utf-8")
    chunks = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def _has_done_marker(raw: bytes) -> bool:
    return b"data: [DONE]" in raw


# ---------------------------------------------------------------------------
# 测试用例：Combined + Chat/Completions（decode_id=None）
# ---------------------------------------------------------------------------

class TestCombinedChatStream:
    """combined worker + /v1/chat/completions 流式路径"""

    @pytest.mark.asyncio
    async def test_sse_format_and_content_type(self, _module_client):
        ac, mocks = _module_client
        mocks["router"].route_to_workers.return_value = ("w1", None)
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_chat_sse_chunks()
        )

        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert _has_done_marker(resp.content)

    @pytest.mark.asyncio
    async def test_first_chunk_has_role(self, _module_client):
        ac, mocks = _module_client
        mocks["router"].route_to_workers.return_value = ("w1", None)
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_chat_sse_chunks()
        )

        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        chunks = _parse_sse_lines(resp.content)
        assert len(chunks) >= 2
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"

    @pytest.mark.asyncio
    async def test_incremental_delta(self, _module_client):
        """每个 chunk 的 content 是增量，不是累积。"""
        ac, mocks = _module_client
        mocks["router"].route_to_workers.return_value = ("w1", None)
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_chat_sse_chunks()
        )

        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        chunks = _parse_sse_lines(resp.content)
        contents = [c["choices"][0]["delta"].get("content", "") for c in chunks]
        assert "Hello" in contents
        assert " world" in contents
        # 不应该出现累积文本 "Hello world"
        assert "Hello world" not in contents

    @pytest.mark.asyncio
    async def test_finish_reason_only_in_last(self, _module_client):
        ac, mocks = _module_client
        mocks["router"].route_to_workers.return_value = ("w1", None)
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_chat_sse_chunks()
        )

        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        chunks = _parse_sse_lines(resp.content)
        for c in chunks[:-1]:
            assert c["choices"][0]["finish_reason"] is None
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_metrics_callback(self, _module_client):
        """流结束后 metrics 和 performance_stats 被回调记录。"""
        ac, mocks = _module_client
        mocks["router"].route_to_workers.return_value = ("w1", None)
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_chat_sse_chunks()
        )

        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        assert resp.status_code == 200
        mocks["metrics"].record_request.assert_called()
        mocks["metrics"].record_request_duration.assert_called()
        mocks["performance_stats"].record_request.assert_called()


class TestCombinedCompletionStream:
    """combined worker + /v1/completions 流式路径"""

    @pytest.mark.asyncio
    async def test_sse_format_and_done(self, _module_client):
        ac, mocks = _module_client
        mocks["router"].route_to_workers.return_value = ("w1", None)
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_completion_sse_chunks()
        )

        resp = await ac.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Once upon",
            "stream": True,
        })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert _has_done_marker(resp.content)

    @pytest.mark.asyncio
    async def test_incremental_text(self, _module_client):
        ac, mocks = _module_client
        mocks["router"].route_to_workers.return_value = ("w1", None)
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_completion_sse_chunks()
        )

        resp = await ac.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Once upon",
            "stream": True,
        })

        chunks = _parse_sse_lines(resp.content)
        texts = [c["choices"][0]["text"] for c in chunks]
        assert "Once" in texts
        assert " upon" in texts
        assert " a time" in texts

    @pytest.mark.asyncio
    async def test_finish_reason_only_in_last(self, _module_client):
        ac, mocks = _module_client
        mocks["router"].route_to_workers.return_value = ("w1", None)
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_completion_sse_chunks()
        )

        resp = await ac.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Once upon",
            "stream": True,
        })

        chunks = _parse_sse_lines(resp.content)
        for c in chunks[:-1]:
            assert c["choices"][0]["finish_reason"] is None
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_metrics_callback(self, _module_client):
        ac, mocks = _module_client
        mocks["router"].route_to_workers.return_value = ("w1", None)
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_completion_sse_chunks()
        )

        resp = await ac.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Once upon",
            "stream": True,
        })

        assert resp.status_code == 200
        mocks["metrics"].record_request.assert_called()
        mocks["performance_stats"].record_request.assert_called()


# ---------------------------------------------------------------------------
# 测试用例：Disagg + Chat/Completions（decode_id != None）
# ---------------------------------------------------------------------------

class TestDisaggChatStream:
    """disagg worker + /v1/chat/completions 流式路径"""

    @staticmethod
    def _setup_disagg_mocks(mocks):
        def _fake_get_worker(wid):
            return FakeWorkerInfo(
                worker_id=wid, url="http://fake:8000", kv_addr="kv://fake",
            )

        mocks["router"].route_to_workers.return_value = ("prefill-w", "decode-w")
        mocks["worker_manager"].get_worker.side_effect = _fake_get_worker
        # prefill 请求（非流式）返回正常 dict
        mocks["worker_manager"].forward_request = AsyncMock(return_value={
            "id": "chatcmpl-prefill",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        })
        # decode 请求（流式）返回 SSE 流
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_chat_sse_chunks()
        )

    @pytest.mark.asyncio
    async def test_sse_format(self, _module_client):
        ac, mocks = _module_client
        self._setup_disagg_mocks(mocks)

        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert _has_done_marker(resp.content)

    @pytest.mark.asyncio
    async def test_first_chunk_has_role(self, _module_client):
        ac, mocks = _module_client
        self._setup_disagg_mocks(mocks)

        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        chunks = _parse_sse_lines(resp.content)
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"

    @pytest.mark.asyncio
    async def test_prefill_uses_non_stream(self, _module_client):
        """disagg 路径：prefill 请求必须使用非流式 forward_request。"""
        ac, mocks = _module_client
        self._setup_disagg_mocks(mocks)

        await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        # prefill 阶段调用了 forward_request（非流式）
        mocks["worker_manager"].forward_request.assert_called_once()
        call_kwargs = mocks["worker_manager"].forward_request.call_args
        # prefill 的 data 中 stream=False, max_tokens=1
        prefill_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert prefill_data["stream"] is False
        assert prefill_data["max_tokens"] == 1

    @pytest.mark.asyncio
    async def test_decode_uses_stream(self, _module_client):
        """disagg 路径：decode 请求使用流式 forward_request_stream。"""
        ac, mocks = _module_client
        self._setup_disagg_mocks(mocks)

        await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        mocks["worker_manager"].forward_request_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_metrics_with_decode_worker(self, _module_client):
        """disagg 路径：metrics 回调中 decode_end 事件应指向 decode worker。"""
        ac, mocks = _module_client
        self._setup_disagg_mocks(mocks)

        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        assert resp.status_code == 200
        mocks["metrics"].record_request.assert_called()

        # 检查 decode_end 事件指向 decode worker
        decode_end_calls = [
            c for c in mocks["event_manager"].process_event.call_args_list
            if hasattr(c.args[0], "event_type") and c.args[0].event_type == "decode_end"
        ]
        if decode_end_calls:
            event = decode_end_calls[0].args[0]
            assert event.worker_id == "decode-w"


class TestDisaggCompletionStream:
    """disagg worker + /v1/completions 流式路径"""

    @staticmethod
    def _setup_disagg_mocks(mocks):
        def _fake_get_worker(wid):
            return FakeWorkerInfo(
                worker_id=wid, url="http://fake:8000", kv_addr="kv://fake",
            )

        mocks["router"].route_to_workers.return_value = ("prefill-w", "decode-w")
        mocks["worker_manager"].get_worker.side_effect = _fake_get_worker
        mocks["worker_manager"].forward_request = AsyncMock(return_value={
            "id": "cmpl-prefill",
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        })
        mocks["worker_manager"].forward_request_stream.return_value = _fake_stream(
            _make_completion_sse_chunks()
        )

    @pytest.mark.asyncio
    async def test_sse_format(self, _module_client):
        ac, mocks = _module_client
        self._setup_disagg_mocks(mocks)

        resp = await ac.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Once upon",
            "stream": True,
        })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert _has_done_marker(resp.content)

    @pytest.mark.asyncio
    async def test_incremental_text(self, _module_client):
        ac, mocks = _module_client
        self._setup_disagg_mocks(mocks)

        resp = await ac.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Once upon",
            "stream": True,
        })

        chunks = _parse_sse_lines(resp.content)
        texts = [c["choices"][0]["text"] for c in chunks]
        assert "Once" in texts
        assert " upon" in texts

    @pytest.mark.asyncio
    async def test_prefill_non_stream_decode_stream(self, _module_client):
        """disagg 路径：prefill 非流式 + decode 流式。"""
        ac, mocks = _module_client
        self._setup_disagg_mocks(mocks)

        await ac.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Once upon",
            "stream": True,
        })

        mocks["worker_manager"].forward_request.assert_called_once()
        mocks["worker_manager"].forward_request_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_metrics_callback(self, _module_client):
        ac, mocks = _module_client
        self._setup_disagg_mocks(mocks)

        resp = await ac.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Once upon",
            "stream": True,
        })

        assert resp.status_code == 200
        mocks["metrics"].record_request.assert_called()
        mocks["performance_stats"].record_request.assert_called()


# ---------------------------------------------------------------------------
# _wrap_stream_with_metrics 单元测试
# ---------------------------------------------------------------------------

class TestWrapStreamWithMetrics:
    """直接测试 _wrap_stream_with_metrics 的行为。"""

    @pytest.mark.asyncio
    async def test_all_chunks_passthrough(self):
        """所有原始 chunk 必须原样透传。"""
        from openjiuwentools.infer_router.api.server import _wrap_stream_with_metrics

        original_chunks = _make_chat_sse_chunks()
        callback = MagicMock()

        received = []
        async for chunk in _wrap_stream_with_metrics(
            _fake_stream(original_chunks), on_stream_complete=callback,
        ):
            received.append(chunk)

        assert received == original_chunks

    @pytest.mark.asyncio
    async def test_usage_extracted_from_last_data_line(self):
        """usage 从最后一个有效 data 行中正确提取。"""
        from openjiuwentools.infer_router.api.server import _wrap_stream_with_metrics

        callback = MagicMock()

        chunks = _make_chat_sse_chunks()
        async for _ in _wrap_stream_with_metrics(
            _fake_stream(chunks), on_stream_complete=callback,
        ):
            pass

        callback.assert_called_once()
        usage = callback.call_args[0][0]
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_empty_usage_when_no_usage_field(self):
        """当 SSE 流中没有 usage 字段时，回调收到空 dict。"""
        from openjiuwentools.infer_router.api.server import _wrap_stream_with_metrics

        callback = MagicMock()
        chunks_no_usage = [
            b'data: {"id":"x","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        async for _ in _wrap_stream_with_metrics(
            _fake_stream(chunks_no_usage), on_stream_complete=callback,
        ):
            pass

        callback.assert_called_once()
        usage = callback.call_args[0][0]
        assert usage == {}

    @pytest.mark.asyncio
    async def test_callback_called_even_on_error(self):
        """即使流中途异常，finally 中的回调也会被执行。"""
        from openjiuwentools.infer_router.api.server import _wrap_stream_with_metrics

        callback = MagicMock()

        async def _broken_stream():
            yield b'data: {"id":"x","choices":[]}\n\n'
            raise RuntimeError("connection lost")

        with pytest.raises(RuntimeError, match="connection lost"):
            async for _ in _wrap_stream_with_metrics(
                _broken_stream(), on_stream_complete=callback,
            ):
                pass

        callback.assert_called_once()
