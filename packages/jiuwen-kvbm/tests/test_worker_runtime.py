from pathlib import Path

from jiuwen_kvbm import BlockLocation, BlockState, WorkerKVBMConfig, WorkerKVBMRuntime
from jiuwen_kvbm.accessor import FakeTensorAccessor


def test_worker_runtime_moves_payload_from_fake_npu_to_ram_and_back(tmp_path: Path):
    """worker runtime 应能把真实 payload 从 fake NPU 搬到 RAM 后再搬回。"""
    accessor = FakeTensorAccessor({1: b"kv-payload"})
    runtime = WorkerKVBMRuntime(
        WorkerKVBMConfig(
            worker_id="worker-1",
            disk_root=tmp_path,
            npu_high_watermark_blocks=0,
        ),
        accessor=accessor,
    )

    runtime.register_block(1, "hash-1", token_count=16)
    events = runtime.enforce_watermarks()

    assert [event.location for event in events] == ["ram"]
    assert not accessor.contains(1)
    metadata = runtime.manager.get_metadata(1)
    assert metadata.location == BlockLocation.HOST_RAM
    assert metadata.state == BlockState.CACHED_RAM

    accessor.write_block(1, b"stale")
    event = runtime.prefetch_block("hash-1")

    assert event is not None
    assert event.location == "npu"
    assert event.engine_specific["from"] == "host_ram"
    assert accessor.read_block(1) == b"kv-payload"
    assert runtime.manager.get_metadata(1).location == BlockLocation.NPU


def test_worker_runtime_spills_ram_to_disk_by_watermark(tmp_path: Path):
    """RAM 超过水位时，worker runtime 应继续将冷 Block 搬到磁盘。"""
    accessor = FakeTensorAccessor({1: b"aaaa", 2: b"bbbb"})
    runtime = WorkerKVBMRuntime(
        WorkerKVBMConfig(
            worker_id="worker-1",
            disk_root=tmp_path,
            npu_high_watermark_blocks=0,
            ram_high_watermark_bytes=4,
        ),
        accessor=accessor,
    )
    runtime.register_block(1, "hash-1", token_count=16)
    runtime.register_block(2, "hash-2", token_count=16)

    events = runtime.enforce_watermarks()

    assert [event.location for event in events] == ["ram", "ram", "disk"]
    assert runtime.stats()["ram_blocks"] == 1
    assert runtime.stats()["disk_blocks"] == 1

    disk_block = runtime.manager.list_blocks(location=BlockLocation.DISK)[0]
    assert disk_block.state == BlockState.CACHED_DISK

    prefetch_event = runtime.prefetch_block(disk_block.block_hash)
    assert prefetch_event is not None
    assert prefetch_event.location == "npu"
    assert prefetch_event.engine_specific["from"] == "disk"
    assert accessor.read_block(disk_block.block_id) in {b"aaaa", b"bbbb"}


def test_runtime_event_to_dict_is_router_friendly(tmp_path: Path):
    """runtime event 应能转换为 worker 上报链路可消费的字典。"""
    accessor = FakeTensorAccessor({1: b"payload"})
    runtime = WorkerKVBMRuntime(
        WorkerKVBMConfig(worker_id="worker-1", disk_root=tmp_path),
        accessor=accessor,
    )

    event = runtime.register_block(1, "hash-1", token_count=8)
    payload = event.to_dict()

    assert payload["event_type"] == "store"
    assert payload["block_hashes"] == ["hash-1"]
    assert payload["location"] == "npu"
    assert payload["token_count"] == 8
    assert payload["engine_specific"]["worker_id"] == "worker-1"


def test_worker_runtime_batch_register_emits_one_store_event(tmp_path: Path):
    """批量注册多个 block 时应生成一条携带对齐物理 ID 的 store 事件。"""

    accessor = FakeTensorAccessor({1: b"block-1", 2: b"block-2"})
    runtime = WorkerKVBMRuntime(
        WorkerKVBMConfig(worker_id="worker-1", disk_root=tmp_path, block_size=16),
        accessor=accessor,
    )

    event = runtime.register_blocks(
        [(1, "hash-1"), (2, "hash-2")],
        token_count=16,
        active=False,
    )

    assert event.event_type == "store"
    assert event.block_hashes == ["hash-1", "hash-2"]
    assert event.engine_specific["physical_block_ids"] == [1, 2]
    assert runtime.manager.get_metadata(1).block_hash == "hash-1"
    assert runtime.manager.get_metadata(2).block_hash == "hash-2"


def test_worker_runtime_cleans_old_disk_payloads_on_start(tmp_path: Path):
    """runtime 启动时应清理旧 KVBM 磁盘 payload，并保留非托管文件。"""
    old_payload = tmp_path / "old-block-1.pkl"
    old_payload.write_bytes(b"stale")
    keep_file = tmp_path / "notes.txt"
    keep_file.write_text("keep", encoding="utf-8")

    WorkerKVBMRuntime(
        WorkerKVBMConfig(worker_id="worker-1", disk_root=tmp_path),
        accessor=FakeTensorAccessor({}),
    )

    assert not old_payload.exists()
    assert keep_file.exists()


def test_worker_runtime_can_keep_old_disk_payloads_on_start(tmp_path: Path):
    """关闭启动清理开关时，runtime 不应删除旧 KVBM 磁盘 payload。"""
    old_payload = tmp_path / "old-block-1.pkl"
    old_payload.write_bytes(b"stale")

    WorkerKVBMRuntime(
        WorkerKVBMConfig(
            worker_id="worker-1",
            disk_root=tmp_path,
            clean_disk_on_start=False,
        ),
        accessor=FakeTensorAccessor({}),
    )

    assert old_payload.exists()
