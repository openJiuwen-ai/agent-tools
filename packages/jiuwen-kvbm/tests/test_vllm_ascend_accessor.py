from jiuwen_kvbm.accessor import VllmAscendKVAccessor
from jiuwen_kvbm.types import MemoryRegion


class DummyProvider:
    def __init__(self):
        self.blocks = {1: b"payload"}
        self.deleted = []

    def read_block(self, block_id: int):
        return self.blocks[block_id]

    def write_block(self, block_id: int, payload):
        self.blocks[block_id] = bytes(payload)

    def describe_block(self, block_id: int):
        return MemoryRegion(device_type="npu", device_id=0, nbytes=len(self.blocks[block_id]))

    def delete_block(self, block_id: int):
        self.deleted.append(block_id)
        self.blocks.pop(block_id, None)


def test_vllm_ascend_accessor_delegates_to_provider():
    """VllmAscendKVAccessor 应只依赖 provider 协议，不依赖具体 vLLM-Ascend 版本。"""
    provider = DummyProvider()
    accessor = VllmAscendKVAccessor(provider)

    assert accessor.read_block(1) == b"payload"
    assert accessor.describe_block(1).device_type == "npu"

    accessor.write_block(2, b"new")
    assert provider.blocks[2] == b"new"

    accessor.delete_block(1)
    assert provider.deleted == [1]
