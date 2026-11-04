from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import pickle
import stat
import threading
from pathlib import Path

import pytest

from jiuwen_kvbm import KVBMManager
from jiuwen_kvbm.accessor import FakeKVAccessor
from jiuwen_kvbm.transfer import LocalDiskBackend, LocalRamBackend
from jiuwen_kvbm.transfer.payload_codec import decode_payload, encode_payload
from jiuwen_kvbm.types import KVBlockMetadata


def _create_marker(path):
    return Path(path).write_text("unsafe")


@pytest.mark.parametrize("disk", [False, True])
def test_concurrent_backend_puts_respect_capacity(tmp_path, disk):
    payload = b"x" * 64
    limit = len(encode_payload(payload)) if disk else len(payload)
    backend = LocalDiskBackend(tmp_path, max_bytes=limit) if disk else LocalRamBackend(max_bytes=limit)
    barrier = threading.Barrier(8)

    def put(index):
        barrier.wait(timeout=5)
        try:
            backend.put(KVBlockMetadata(block_id=index, block_hash=f"h{index}"), payload)
            return True
        except (OSError, MemoryError):
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(put, range(8))) == 1
    assert backend.bytes_used == limit


@pytest.mark.parametrize("ref", ["../outside.kv", "/tmp/outside.kv", "..\\outside.kv", ".", "bad.pkl"])
def test_disk_rejects_unmanaged_paths(tmp_path, ref):
    backend = LocalDiskBackend(tmp_path)
    metadata = KVBlockMetadata(block_id=1, block_hash="a", store_ref=ref)
    for method in (backend.get, backend.delete, backend.contains):
        with pytest.raises(ValueError):
            method(metadata)


def test_disk_rejects_symlinks_without_touching_target(tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    root = tmp_path / "cache"
    backend = LocalDiskBackend(root)
    (root / "linked.kv").symlink_to(outside)
    metadata = KVBlockMetadata(block_id=1, block_hash="a", store_ref="linked.kv")
    for method in (backend.get, backend.delete, backend.contains):
        with pytest.raises(ValueError):
            method(metadata)
    assert outside.read_bytes() == b"secret"


def test_pickle_is_never_executed(tmp_path):
    marker = tmp_path / "executed"

    class Malicious:
        def __init__(self, path):
            self.marker = str(path)

        def __reduce__(self):
            return _create_marker, (self.marker,)

    backend = LocalDiskBackend(tmp_path / "cache")
    metadata = KVBlockMetadata(block_id=1, block_hash="a")
    metadata.store_ref = backend.put(metadata, b"safe")
    (backend.root / metadata.store_ref).write_bytes(pickle.dumps(Malicious(marker)))
    with pytest.raises((ValueError, UnicodeDecodeError)):
        backend.get(metadata)
    assert not marker.exists()


def test_data_only_roundtrip_and_atomic_capacity_failure(tmp_path):
    payload = {"bytes": b"\x00\xff", "tuple": (1, "x"), "list": [True, None]}
    assert decode_payload(encode_payload(payload)) == payload
    backend = LocalDiskBackend(tmp_path, max_bytes=len(encode_payload(b"old")))
    metadata = KVBlockMetadata(block_id=1, block_hash="a")
    metadata.store_ref = backend.put(metadata, b"old")
    with pytest.raises(OSError):
        backend.put(metadata, b"larger payload")
    assert backend.get(metadata) == b"old"
    assert backend.contains(replace(metadata))


def test_disk_payload_permissions_are_private(tmp_path):
    backend = LocalDiskBackend(tmp_path)
    metadata = KVBlockMetadata(block_id=1, block_hash="private")
    metadata.store_ref = backend.put(metadata, b"private payload")
    path = backend.root / metadata.store_ref
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert backend.get(metadata) == b"private payload"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_concurrent_offload_only_moves_one_payload():
    accessor = FakeKVAccessor({1: b"payload"})
    ram = LocalRamBackend()
    manager = KVBMManager(accessor=accessor, ram_backend=ram)
    manager.register_npu_block(1, "hash", active=False)
    barrier = threading.Barrier(2)

    def offload():
        barrier.wait(timeout=5)
        try:
            manager.offload_to_ram(1)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(lambda _: offload(), range(2))) == 1
    assert ram.bytes_used == len(b"payload")
