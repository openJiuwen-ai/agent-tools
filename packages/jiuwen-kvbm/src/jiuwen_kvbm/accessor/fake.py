from __future__ import annotations

from typing import Any

from jiuwen_kvbm.types import MemoryRegion


class FakeKVAccessor:
    """用于单测和本地验证的伪运行时访问器。

    参数:
        blocks: 初始 Block 数据映射，key 为 block_id，value 为测试载荷。
    """

    def __init__(self, blocks: dict[int, Any] | None = None):
        self._blocks: dict[int, Any] = dict(blocks or {})

    def read_block(self, block_id: int) -> Any:
        """读取内存字典中的 Block 载荷。

        参数:
            block_id: 测试 Block 编号。

        返回:
            对应 Block 的测试载荷。
        """
        return self._blocks[block_id]

    def write_block(self, block_id: int, payload: Any) -> None:
        """写入或覆盖内存字典中的 Block 载荷。

        参数:
            block_id: 测试 Block 编号。
            payload: 需要保存的测试载荷。

        返回:
            无返回值。
        """
        self._blocks[block_id] = payload

    def describe_block(self, block_id: int) -> MemoryRegion:
        """返回伪 NPU Block 的中性内存描述。

        参数:
            block_id: 测试 Block 编号。

        返回:
            使用 fake_npu 作为 device_type 的 MemoryRegion。
        """
        payload = self._blocks[block_id]
        return MemoryRegion(
            device_type="fake_npu",
            device_id=0,
            nbytes=_payload_size(payload),
            backend_handle=block_id,
        )

    def delete_block(self, block_id: int) -> None:
        """删除内存字典中的 Block 载荷。

        参数:
            block_id: 测试 Block 编号。

        返回:
            无返回值。
        """
        self._blocks.pop(block_id, None)


def _payload_size(payload: Any) -> int:
    """估算测试载荷大小，避免 fake accessor 引入真实 tensor 依赖。

    参数:
        payload: 需要估算大小的测试载荷。

    返回:
        估算得到的字节数。
    """
    if isinstance(payload, bytes | bytearray | memoryview):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode())
    return 0
