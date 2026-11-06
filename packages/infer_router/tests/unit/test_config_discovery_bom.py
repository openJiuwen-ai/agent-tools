"""A UTF-8 BOM in workers.json must not hide all workers."""

import json

import pytest

from openjiuwentools.infer_router.discovery.config_discovery import ConfigDiscovery


@pytest.mark.asyncio
async def test_json_config_with_bom_is_loaded(tmp_path) -> None:
    config = tmp_path / "workers.json"
    payload = {
        "workers": [
            {"worker_id": "w1", "model": "m", "url": "http://w1", "available_memory": 1000}
        ]
    }
    config.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))

    workers = await ConfigDiscovery(str(config)).discover()

    assert [worker.worker_id for worker in workers] == ["w1"]
