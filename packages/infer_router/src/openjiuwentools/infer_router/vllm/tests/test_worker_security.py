# White-box tests exercise the request boundary and its redacted summary.
# pylint: disable=protected-access
import json
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from openjiuwentools.infer_router.vllm.transport import WorkerRequestBoundary, _request_summary
from openjiuwentools.infer_router.vllm import transport
from openjiuwentools.infer_router.vllm.cli import _build_parser


def make_app():
    app = FastAPI()
    app.add_middleware(WorkerRequestBoundary, api_key="test-secret", max_body_size=32)

    @app.post("/test")
    async def echo(request: Request):
        return await request.json()

    return app


@pytest.mark.asyncio
async def test_worker_auth_precedes_json_parsing():
    async with AsyncClient(transport=ASGITransport(app=make_app()), base_url="http://test") as client:
        response = await client.post("/test", content=b"not json")
        assert response.status_code == 401
        response = await client.post("/test", json={"a": 1}, headers={"Authorization": "Bearer test-secret"})
        assert response.json() == {"a": 1}


@pytest.mark.asyncio
async def test_worker_limits_chunked_and_declared_bodies():
    async def chunks():
        yield b"x" * 20
        yield b"x" * 20

    headers = {"Authorization": "Bearer test-secret"}
    async with AsyncClient(transport=ASGITransport(app=make_app()), base_url="http://test") as client:
        response = await client.post("/test", content=chunks(), headers=headers)
        assert response.status_code == 413
        response = await client.post("/test", content=b"x" * 40, headers=headers)
        assert response.status_code == 413


def test_request_summary_omits_all_prompt_and_tool_content():
    body = {"messages": [{"content": "private prompt"}], "tools": [{"secret": "key"}], "cache_salt": "private"}
    summary = json.dumps(_request_summary(body))
    assert "private" not in summary
    assert "secret" not in summary
    assert "messages_count" in summary


def test_worker_cli_defaults_to_loopback():
    args = _build_parser().parse_args(["--model", "test-model"])
    assert args.host == "127.0.0.1"


@pytest.mark.asyncio
async def test_tcp_unauthorized_request_never_reaches_engine(monkeypatch):
    reader = asyncio.StreamReader()
    reader.feed_data(transport._tcp_pack_message({"api_key": "wrong", "messages": []}))
    reader.feed_eof()
    output = []
    writer = SimpleNamespace(
        get_extra_info=lambda _name: "test-peer",
        write=output.append,
        drain=AsyncMock(),
        close=lambda: None,
        wait_closed=AsyncMock(),
    )
    completion = AsyncMock()
    monkeypatch.setattr(transport, "do_chat_completion", completion)
    await transport._tcp_client_handler(reader, writer, object(), "test", api_key="correct")
    assert b"Unauthorized" in output[0]
    completion.assert_not_awaited()
