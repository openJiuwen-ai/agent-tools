"""Worker health checks must target the registered URL's real host/port."""

from __future__ import annotations

import pytest

from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo
from openjiuwentools.infer_router.worker.worker_manager import (
    WorkerManager,
    WorkerStatus,
)


class _FakeResponse:
    status_code = 200


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get(self, url: str, headers=None):  # noqa: ANN001
        self.calls.append(url)
        return _FakeResponse()


@pytest.mark.parametrize(
    ("registered_url", "expected_health_url"),
    [
        ("http://localhost:8001/v1", "http://localhost:8001/health"),
        ("http://localhost:8111/v1", "http://localhost:8111/health"),
        ("http://localhost:8001/v1/", "http://localhost:8001/health"),
        ("http://localhost:8002", "http://localhost:8002/health"),
    ],
)
async def test_health_url_preserves_the_registered_port(registered_url, expected_health_url):
    """rstrip("/v1") strips characters, not a suffix, so port 8001 became 800."""
    manager = WorkerManager()
    manager.workers["worker-1"] = WorkerInfo(
        worker_id="worker-1",
        model="test-model",
        url=registered_url,
    )
    manager.worker_statuses["worker-1"] = WorkerStatus(
        worker_id="worker-1",
        last_health_check=0,
        is_healthy=True,
        response_time=0,
    )
    client = _RecordingClient()
    manager.http_client = client

    await manager.check_worker_health("worker-1")

    assert client.calls == [expected_health_url]
    assert manager.worker_statuses["worker-1"].is_healthy is True
