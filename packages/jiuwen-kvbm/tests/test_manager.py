# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
import pytest

from jiuwen_kvbm import BlockLocation, BlockState, KVBMManager
from jiuwen_kvbm.accessor import FakeKVAccessor
from jiuwen_kvbm.manager import PrefetchCommitRejectedError
from jiuwen_kvbm.transfer import LocalDiskBackend, LocalRamBackend


def test_register_release_and_match():
    accessor = FakeKVAccessor({1: b"kv-block"})
    manager = KVBMManager(accessor=accessor)

    handle = manager.register_npu_block(1, "hash-1", session_id="s1", token_count=16)

    assert handle.block_id == 1
    assert manager.match(["hash-1"])[0].block_id == 1
    assert manager.release("s1", ["hash-1"]) == 1

    metadata = manager.get_metadata(1)
    assert metadata.ref_count == 0
    assert metadata.state == BlockState.CACHED_NPU


def test_register_reused_physical_block_removes_stale_hash_mapping():
    """物理 block ID 被新 hash 复用时不应保留旧 hash 命中。"""

    accessor = FakeKVAccessor({1: b"kv-block"})
    manager = KVBMManager(accessor=accessor)
    manager.register_npu_block(1, "old-hash", active=False)

    manager.register_npu_block(1, "new-hash", active=False)

    assert manager.match(["old-hash"]) == []
    assert manager.match(["new-hash"])[0].block_id == 1


def test_register_npu_blocks_batches_metadata_lock_and_preserves_order():
    """批量注册应只进入一次 metadata lock，并保持输入 handle 顺序。"""

    class CountingLock:
        """测试用可重入锁替身，仅统计上下文进入次数。"""

        def __init__(self) -> None:
            self.enter_count = 0

        def __enter__(self):
            self.enter_count += 1
            return self

        @staticmethod
        def __exit__(_exc_type, _exc, _traceback) -> None:
            return None

    accessor = FakeKVAccessor({1: b"block-1", 2: b"block-2"})
    manager = KVBMManager(accessor=accessor)
    counting_lock = CountingLock()
    manager._metadata_lock = counting_lock

    handles = manager.register_npu_blocks(
        [(1, "hash-1"), (2, "hash-2")],
        token_count=16,
        active=False,
    )

    assert [handle.block_id for handle in handles] == [1, 2]
    assert counting_lock.enter_count == 1
    assert manager.get_metadata(1).block_hash == "hash-1"
    assert manager.get_metadata(2).block_hash == "hash-2"
    assert manager.get_metadata(1).state == BlockState.CACHED_NPU


def test_register_npu_blocks_skips_unchanged_metadata_description():
    """相同 block ID 与 hash 重复注册时应走幂等快速路径。"""

    class CountingAccessor(FakeKVAccessor):
        def __init__(self, blocks):
            super().__init__(blocks)
            self.describe_count = 0

        def describe_block(self, block_id):
            self.describe_count += 1
            return super().describe_block(block_id)

    accessor = CountingAccessor({1: b"block-1", 2: b"block-2"})
    manager = KVBMManager(accessor=accessor)
    manager.register_npu_blocks(
        [(1, "hash-1"), (2, "hash-2")],
        token_counts=[16, 8],
        active=False,
    )
    diagnostics = {}

    handles = manager.register_npu_blocks(
        [(1, "hash-1"), (2, "hash-2")],
        token_counts=[16, 8],
        active=False,
        diagnostics=diagnostics,
    )

    assert [handle.block_id for handle in handles] == [1, 2]
    assert accessor.describe_count == 2
    assert diagnostics.get("registered_blocks") == 0
    assert diagnostics.get("unchanged_blocks") == 2
    assert diagnostics.get("describe_blocks_ms") >= 0
    assert diagnostics.get("metadata_update_ms") >= 0


def test_register_reused_physical_block_deletes_stale_ram_payload():
    """vLLM 复用 physical block 时应删除旧 hash 对应的 RAM backing payload。"""

    accessor = FakeKVAccessor({1: b"old-payload"})
    ram = LocalRamBackend()
    manager = KVBMManager(accessor=accessor, ram_backend=ram)
    manager.register_npu_block(1, "old-hash", active=False)
    manager.offload_to_ram(1)
    assert ram.bytes_used == len(b"old-payload")
    accessor.write_block(1, b"new-payload")

    manager.register_npu_block(1, "new-hash", active=False)

    assert ram.bytes_used == 0
    assert manager.match(["old-hash"]) == []
    assert manager.get_metadata(1).block_hash == "new-hash"
    assert manager.get_metadata(1).location == BlockLocation.NPU


def test_offload_to_ram_and_prefetch():
    accessor = FakeKVAccessor({1: b"payload"})
    ram = LocalRamBackend()
    manager = KVBMManager(accessor=accessor, ram_backend=ram)
    manager.register_npu_block(1, "hash-1", active=False)

    manager.offload_to_ram(1)

    metadata = manager.get_metadata(1)
    assert metadata.location == BlockLocation.HOST_RAM
    assert metadata.state == BlockState.CACHED_RAM
    assert ram.bytes_used == len(b"payload")

    accessor.write_block(1, b"stale")
    manager.prefetch_to_npu(1)

    metadata = manager.get_metadata(1)
    assert accessor.read_block(1) == b"payload"
    assert metadata.location == BlockLocation.NPU
    assert metadata.state == BlockState.CACHED_NPU
    assert ram.bytes_used == 0


def test_prefetch_rejection_after_write_keeps_ram_source_and_removes_npu_payload():
    """最终提交被拒绝时应保留 RAM 副本，并清理刚写入的 NPU payload。"""

    accessor = FakeKVAccessor({1: b"payload"})
    ram = LocalRamBackend()
    manager = KVBMManager(accessor=accessor, ram_backend=ram)
    manager.register_npu_block(1, "hash-1", active=False)
    manager.offload_to_ram(1)
    checks = iter((True, False))

    with pytest.raises(PrefetchCommitRejectedError, match="after NPU write"):
        manager.prefetch_to_npu(1, should_commit=lambda _block: next(checks))

    metadata = manager.get_metadata(1)
    assert metadata.location == BlockLocation.HOST_RAM
    assert metadata.state == BlockState.CACHED_RAM
    assert ram.contains(metadata)
    with pytest.raises(KeyError):
        accessor.read_block(1)


def test_offload_to_disk_from_ram(tmp_path):
    accessor = FakeKVAccessor({1: b"payload"})
    ram = LocalRamBackend()
    disk = LocalDiskBackend(tmp_path)
    manager = KVBMManager(accessor=accessor, ram_backend=ram, disk_backend=disk)
    manager.register_npu_block(1, "hash-1", active=False)

    manager.offload_to_ram(1)
    manager.offload_to_disk(1)

    metadata = manager.get_metadata(1)
    assert metadata.location == BlockLocation.DISK
    assert metadata.state == BlockState.CACHED_DISK
    assert ram.bytes_used == 0
    assert disk.bytes_used > 0

    accessor.write_block(1, b"stale")
    manager.prefetch_to_npu(1)

    assert accessor.read_block(1) == b"payload"
    assert manager.get_metadata(1).state == BlockState.CACHED_NPU


def test_lru_evicts_released_blocks(tmp_path):
    accessor = FakeKVAccessor({1: b"a", 2: b"bb"})
    ram = LocalRamBackend()
    disk = LocalDiskBackend(tmp_path)
    manager = KVBMManager(accessor=accessor, ram_backend=ram, disk_backend=disk)
    manager.register_npu_block(1, "hash-1", active=False)
    manager.register_npu_block(2, "hash-2", active=False)
    manager.offload_to_ram(1)
    manager.offload_to_disk(2)

    evicted = manager.evict(limit=2)

    assert {handle.block_id for handle in evicted} == {1, 2}
    assert manager.get_metadata(1).state == BlockState.EVICTED
    assert manager.get_metadata(2).state == BlockState.EVICTED
    assert manager.stats().evicted_blocks == 2
