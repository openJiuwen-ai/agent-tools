import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest_asyncio

_VLLM_DIR = str(Path(__file__).resolve().parent.parent)
if _VLLM_DIR not in sys.path:
    sys.path.append(_VLLM_DIR)

_mock_vllm_sub = MagicMock()
for mod_name in [
    "vllm.config", "vllm.v1", "vllm.v1.metrics",
    "vllm.v1.metrics.loggers", "vllm.v1.metrics.stats",
    "vllm.sampling_params", "vllm.distributed", "vllm.distributed.kv_events",
    "vllm.engine", "vllm.engine.arg_utils", "vllm.v1.engine",
    "vllm.v1.engine.async_llm",
]:
    sys.modules[mod_name] = _mock_vllm_sub

from transport import _tcp_client_handler  # noqa: E402

MOCK_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "mock reply"},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


@pytest_asyncio.fixture
async def tcp_server_port():
    """启动真实 TCP server，mock 掉 do_chat_completion，返回端口号。"""
    engine = AsyncMock()
    model_name = "test-model"

    async def mock_completion(eng, body, name):
        resp = MOCK_RESPONSE.copy()
        resp["model"] = name
        return resp

    with patch("transport.do_chat_completion", side_effect=mock_completion):
        server = await asyncio.start_server(
            lambda r, w: _tcp_client_handler(r, w, engine, model_name),
            host="127.0.0.1",
            port=0,
        )
        port = server.sockets[0].getsockname()[1]
        yield port
        server.close()
        await server.wait_closed()
