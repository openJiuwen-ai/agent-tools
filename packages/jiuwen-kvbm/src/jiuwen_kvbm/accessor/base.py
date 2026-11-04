from __future__ import annotations

from typing import Any, Protocol

from jiuwen_kvbm.types import MemoryRegion


class KVAccessor(Protocol):
    def read_block(self, block_id: int) -> Any:
        """从推理运行时读取指定 KV Block 的载荷。

        参数:
            block_id: 推理运行时中的 Block 编号。

        返回:
            Block 的载荷对象。具体类型由运行时适配器决定。
        """

    def write_block(self, block_id: int, payload: Any) -> None:
        """将 KV Block 载荷写回推理运行时。

        参数:
            block_id: 推理运行时中的 Block 编号。
            payload: 需要写回的 Block 载荷对象。

        返回:
            无返回值。
        """

    def describe_block(self, block_id: int) -> MemoryRegion:
        """返回运行时 Block 的中性内存描述。

        参数:
            block_id: 推理运行时中的 Block 编号。

        返回:
            描述该 Block 内存位置、大小和底层句柄的 MemoryRegion。
        """

    def delete_block(self, block_id: int) -> None:
        """从推理运行时释放指定 KV Block 的载荷。

        参数:
            block_id: 推理运行时中的 Block 编号。

        返回:
            无返回值。

        说明:
            真实 NPU runtime 可以在这里释放或标记复用对应 block。某些运行时如果
            不能立即释放，可实现为空操作。
        """
