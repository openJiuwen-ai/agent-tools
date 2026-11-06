from __future__ import annotations

from typing import Any

from jiuwen_kvbm.types import MemoryRegion


class FakeTensorAccessor:
    """模拟 worker 侧 NPU KV tensor 的访问器。

    参数:
        blocks: 初始 Block 数据映射，key 为 block_id，value 为 bytes-like 载荷。
        device_id: 模拟 NPU 设备编号。

    说明:
        该类会在读写时复制 bytes，避免测试中因为对象引用共享掩盖真实搬运问题。
        后续接入昇腾 runtime 时，可用真实 NPU accessor 替换该类。
    """

    def __init__(self, blocks: dict[int, bytes | bytearray | memoryview] | None = None, device_id: int = 0):
        self.device_id = device_id
        self._blocks: dict[int, bytes] = {
            block_id: bytes(payload) for block_id, payload in dict(blocks or {}).items()
        }

    def read_block(self, block_id: int) -> bytes:
        """读取模拟 NPU 中的 Block 载荷。

        参数:
            block_id: 模拟 NPU Block 编号。

        返回:
            Block 载荷的 bytes 副本。

        异常:
            KeyError: block_id 不存在。
        """
        return bytes(self._blocks[block_id])

    def write_block(self, block_id: int, payload: Any) -> None:
        """写入模拟 NPU Block 载荷。

        参数:
            block_id: 模拟 NPU Block 编号。
            payload: bytes-like 载荷对象。

        返回:
            无返回值。
        """
        self._blocks[block_id] = bytes(payload)

    def describe_block(self, block_id: int) -> MemoryRegion:
        """返回模拟 NPU Block 的中性内存描述。

        参数:
            block_id: 模拟 NPU Block 编号。

        返回:
            device_type 为 npu 的 MemoryRegion。

        异常:
            KeyError: block_id 不存在。
        """
        payload = self._blocks[block_id]
        return MemoryRegion(
            device_type="npu",
            device_id=self.device_id,
            nbytes=len(payload),
            backend_handle=block_id,
        )

    def delete_block(self, block_id: int) -> None:
        """从模拟 NPU 中删除 Block 载荷。

        参数:
            block_id: 模拟 NPU Block 编号。

        返回:
            无返回值。
        """
        self._blocks.pop(block_id, None)

    def contains(self, block_id: int) -> bool:
        """判断模拟 NPU 是否仍保存指定 Block。

        参数:
            block_id: 模拟 NPU Block 编号。

        返回:
            如果 Block 仍在模拟 NPU 中，返回 True；否则返回 False。
        """
        return block_id in self._blocks
