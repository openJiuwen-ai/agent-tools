from jiuwen_kvbm.accessor.vllm_ascend_018 import (
    VllmAscend018BlockPayload,
    VllmAscend018LayerPayload,
)
from jiuwen_kvbm.transfer import LocalDiskBackend, LocalRamBackend
from jiuwen_kvbm.transfer.payload_codec import HostValues, decode_payload, encode_payload
from jiuwen_kvbm.types import KVBlockMetadata

import pytest


def _metadata(block_id: int = 1, block_hash: str = "hash") -> KVBlockMetadata:
    return KVBlockMetadata(block_id=block_id, block_hash=block_hash)


def test_ram_backend_put_get_delete_roundtrip():
    backend = LocalRamBackend()
    metadata = _metadata()

    metadata.store_ref = backend.put(metadata, b"payload")

    assert metadata.store_ref == "hash:1"
    assert backend.contains(metadata)
    assert backend.get(metadata) == b"payload"
    assert backend.bytes_used == len(b"payload")

    backend.delete(metadata)

    assert not backend.contains(metadata)
    assert backend.bytes_used == 0


def test_ram_backend_overwrite_updates_size():
    backend = LocalRamBackend(max_bytes=4)
    metadata = _metadata()

    metadata.store_ref = backend.put(metadata, b"1234")
    metadata.store_ref = backend.put(metadata, b"12")

    assert backend.bytes_used == 2
    assert backend.get(metadata) == b"12"


def test_ram_backend_capacity_exceeded_raises_memory_error():
    backend = LocalRamBackend(max_bytes=2)

    with pytest.raises(MemoryError):
        backend.put(_metadata(), b"123")

    assert backend.bytes_used == 0
    assert not backend.contains(_metadata())


def test_ram_backend_missing_store_ref():
    backend = LocalRamBackend()

    with pytest.raises(KeyError):
        backend.get(_metadata())

    backend.delete(_metadata())
    assert not backend.contains(_metadata())


def test_disk_backend_put_get_delete_roundtrip(tmp_path):
    backend = LocalDiskBackend(tmp_path / "cache")
    metadata = _metadata()
    payload = {"value": b"\x00\xff"}

    metadata.store_ref = backend.put(metadata, payload)

    assert metadata.store_ref == "hash-1.kv"
    assert backend.contains(metadata)
    assert backend.get(metadata) == payload
    assert backend.bytes_used == len(encode_payload(payload))

    backend.delete(metadata)

    assert not backend.contains(metadata)
    assert backend.bytes_used == 0
    assert not (backend.root / "hash-1.kv").exists()


def test_disk_backend_persists_across_instances(tmp_path):
    root = tmp_path / "cache"
    metadata = _metadata()
    metadata.store_ref = LocalDiskBackend(root).put(metadata, b"persisted")

    assert LocalDiskBackend(root).get(metadata) == b"persisted"


def test_disk_backend_sanitizes_block_hash(tmp_path):
    backend = LocalDiskBackend(tmp_path)

    ref = backend.put(_metadata(block_id=7, block_hash="a/b:c"), b"x")

    assert ref == "a_b_c-7.kv"
    assert (backend.root / ref).is_file()


def test_disk_backend_clear_managed_payloads(tmp_path):
    backend = LocalDiskBackend(tmp_path)
    for index in (1, 2):
        backend.put(_metadata(block_id=index, block_hash=f"h{index}"), b"payload")
    (tmp_path / "legacy.pkl").write_bytes(b"legacy")
    (tmp_path / "keep.txt").write_bytes(b"keep")

    assert backend.clear_managed_payloads() == 3

    assert backend.bytes_used == 0
    assert not (tmp_path / "legacy.pkl").exists()
    assert (tmp_path / "keep.txt").exists()


def test_payload_codec_roundtrip_nested_containers():
    payload = {"ratio": 1.5, "flag": True, "items": [1, (2, b"\x00"), {"nested": None}]}

    decoded = decode_payload(encode_payload(payload))

    assert decoded == payload
    assert isinstance(decoded["items"][1], tuple)


def test_payload_codec_roundtrip_vllm_ascend_block():
    block = VllmAscend018BlockPayload(
        block_id=3,
        layers=[VllmAscend018LayerPayload("layer.0", b"key", b"value")],
    )

    decoded = decode_payload(encode_payload(block))

    assert decoded == block
    assert isinstance(decoded.layers[0], VllmAscend018LayerPayload)


def test_payload_codec_roundtrip_host_values():
    decoded = decode_payload(encode_payload(HostValues([1, 2, 3])))

    assert isinstance(decoded, HostValues)
    assert decoded.values == [1, 2, 3]


def test_payload_codec_rejects_unsupported_type():
    with pytest.raises(TypeError):
        encode_payload(object())


def test_payload_codec_rejects_unknown_kind():
    with pytest.raises(ValueError):
        decode_payload(b'{"kind": "unknown"}')


def test_payload_codec_rejects_non_object_payload():
    with pytest.raises(ValueError):
        decode_payload(b"[1, 2]")


def test_payload_codec_rejects_invalid_tensor_payload():
    pytest.importorskip("torch")

    with pytest.raises(ValueError):
        decode_payload(b'{"kind": "tensor", "dtype": "float128", "shape": [1], "data": ""}')
    with pytest.raises(ValueError):
        decode_payload(b'{"kind": "tensor", "dtype": "float32", "shape": [-1], "data": ""}')
    with pytest.raises(ValueError):
        decode_payload(b'{"kind": "tensor", "dtype": "float32", "shape": [2], "data": ""}')
