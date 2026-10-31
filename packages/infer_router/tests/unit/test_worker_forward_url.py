"""Worker request forwarding must not duplicate the API version in the URL."""

import pytest

from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo
from openjiuwentools.infer_router.worker.worker_manager import (
    WorkerManager,
    worker_endpoint_url,
)


def test_worker_endpoint_url_accepts_documented_and_bare_base_urls():
    endpoint = "/v1/chat/completions"
    assert worker_endpoint_url("http://localhost:8001/v1", endpoint) == (
        "http://localhost:8001/v1/chat/completions"
    )
    assert worker_endpoint_url("http://localhost:8001/v1/", endpoint) == (
        "http://localhost:8001/v1/chat/completions"
    )
    assert worker_endpoint_url("http://localhost:8001", endpoint) == (
        "http://localhost:8001/v1/chat/completions"
    )
    assert worker_endpoint_url("http://localhost:8001/", endpoint) == (
        "http://localhost:8001/v1/chat/completions"
    )


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True}


class _FakeClient:
    def __init__(self):
        self.posted = []

    async def post(self, url, **kwargs):
        self.posted.append(url)
        return _FakeResponse()


@pytest.mark.asyncio
async def test_forward_request_uses_one_api_version_prefix():
    manager = WorkerManager()
    manager.workers["worker-1"] = WorkerInfo(
        worker_id="worker-1",
        model="test-model",
        url="http://localhost:8001/v1",
    )
    client = _FakeClient()
    manager.http_client = client

    result = await manager.forward_request(
        "worker-1", "/v1/chat/completions", "POST", {"model": "test-model"}
    )

    assert result == {"ok": True}
    assert client.posted == ["http://localhost:8001/v1/chat/completions"]
