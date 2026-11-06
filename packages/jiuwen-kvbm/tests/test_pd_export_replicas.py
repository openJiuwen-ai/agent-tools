# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
"""P exports one physical replica per logical position and cache group."""

from types import SimpleNamespace

import pytest

from jiuwen_kvbm.integrations import vllm_ascend_018 as integration
from jiuwen_kvbm.runtime.block_lease import VllmSourceBlockLeaseManager


def packed(value, group=0):
    return value.to_bytes(32, "big") + group.to_bytes(4, "big")


def block(block_id, block_hash, refs=0):
    return SimpleNamespace(block_id=block_id, block_hash=block_hash, ref_cnt=refs)


def test_same_group_replicas_export_once():
    key = packed(100)
    pool = SimpleNamespace(
        cached_block_hash_to_block={
            key: {
                4: block(4, key),
                8: block(8, key, refs=1),
                2: block(2, key),
            }
        }
    )
    result, missing = integration._resolve_cached_request_blocks(pool, (key[:-4],))
    assert not missing
    assert [(r["logical_index"], r["group_id"], r["block_id"]) for r in result] == [(0, 0, 8)]


def test_replicas_preserve_every_group_and_logical_order():
    first = packed(100)
    second = packed(100, 1)
    third = packed(101)
    pool = SimpleNamespace(
        cached_block_hash_to_block={
            first: [block(4, first), block(2, first)],
            second: [block(7, second), block(8, second)],
            third: block(9, third),
        }
    )
    result, missing = integration._resolve_cached_request_blocks(pool, (first[:-4], third[:-4]))
    assert not missing
    assert [(r["logical_index"], r["group_id"], r["block_id"]) for r in result] == [
        (0, 0, 2),
        (0, 1, 7),
        (1, 0, 9),
    ]


def test_stale_and_unpublished_and_null_replicas_are_not_exported():
    key = packed(100)
    null = block(3, key)
    null.is_null = True
    pool = SimpleNamespace(
        cached_block_hash_to_block={
            key: [
                block(1, packed(101)),
                block(2, None),
                null,
            ]
        }
    )
    result, missing = integration._resolve_cached_request_blocks(pool, (key[:-4],))
    assert result == []
    assert missing == [0]


def test_hash_alias_collision_fails_safe(monkeypatch):
    first, second = packed(100), packed(101)
    pool = SimpleNamespace(cached_block_hash_to_block={first: block(1, first), second: block(2, second)})
    monkeypatch.setattr(integration, "_block_hash_aliases", lambda value: ["collision"])
    with pytest.raises(RuntimeError, match="ambiguously resolves"):
        integration._resolve_cached_request_blocks(pool, (first[:-4],))


def test_block_aligned_export_pins_exactly_39_blocks(monkeypatch):
    """4992/128=39; a duplicated cache copy must not become a 40th transfer."""
    hashes = tuple(packed(i + 100) for i in range(39))
    blocks = [block(i, key) for i, key in enumerate(hashes)]
    duplicate = block(39, hashes[-1])
    blocks.append(duplicate)

    class Pool:
        def __init__(self):
            self.blocks = blocks
            self.cached_block_hash_to_block = {key: {item.block_id: item} for key, item in zip(hashes, blocks)}
            self.cached_block_hash_to_block[hashes[-1]][39] = duplicate

        @staticmethod
        def touch(values):
            assert integration._VLLM_BLOCK_POOL_LOCK._is_owned()
            for item in values:
                item.ref_cnt += 1

        @staticmethod
        def free_blocks(values):
            for item in values:
                item.ref_cnt -= 1

    pool = Pool()
    manager = VllmSourceBlockLeaseManager(lambda: pool, lock=integration._VLLM_BLOCK_POOL_LOCK)
    monkeypatch.setattr(integration, "_PD_SOURCE_EXPORT_INTENTS", {})
    monkeypatch.setattr(integration, "_PD_REQUEST_BLOCK_HASHES", {"request": tuple(key[:-4] for key in hashes)})
    monkeypatch.setattr(
        integration, "_PD_REQUEST_TOKEN_BLOCK_FINGERPRINTS", {"request": tuple("token" for _ in hashes)}
    )
    monkeypatch.setattr(integration, "_latest_reservable_block_pool", lambda: pool)
    monkeypatch.setattr(integration, "_process_source_block_leases", lambda: manager)
    result = integration._local_pd_export({"request_id": "request", "block_limit": 39})
    assert result["ok"]
    assert result["logical_block_count"] == 39
    assert result["physical_block_count"] == 39
    assert len(result["block_groups"]) == len(result["block_hashes"]) == 39
    assert result["source_blocks"] == list(range(39))
    assert duplicate.ref_cnt == 0
    assert all(item.ref_cnt == 1 for item in blocks[:39])
    manager.release(result["lease"]["lease_id"])
    assert all(item.ref_cnt == 0 for item in blocks)
