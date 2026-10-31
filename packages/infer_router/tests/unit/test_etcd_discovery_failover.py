"""EtcdDiscovery must try every configured host, not only the first."""

import httpx
import pytest

from openjiuwentools.infer_router.discovery.etcd_discovery import EtcdDiscovery


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self):
        self.calls = []

    async def get(self, url):
        self.calls.append(url)
        if url.startswith("http://127.0.0.1:2379"):
            raise httpx.ConnectError("first etcd is down")
        return _Response(
            {"node": {"nodes": [{"value": '{"worker_id":"w1","model":"m","url":"http://w1"}'}]}}
        )


@pytest.mark.asyncio
async def test_discovery_falls_back_to_second_etcd_host() -> None:
    discovery = EtcdDiscovery(etcd_hosts=["127.0.0.1", "127.0.0.2"])
    discovery.client = _FakeClient()

    workers = await discovery.discover()

    assert [worker.worker_id for worker in workers] == ["w1"]
    assert discovery.client.calls[0].startswith("http://127.0.0.1:2379")
    assert discovery.client.calls[1].startswith("http://127.0.0.2:2379")
