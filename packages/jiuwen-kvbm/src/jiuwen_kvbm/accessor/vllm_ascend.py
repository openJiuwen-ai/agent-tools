from __future__ import annotations

from typing import Any, Protocol

from jiuwen_kvbm.types import MemoryRegion


class BlockPayloadProvider(Protocol):
    """版本无关的 KV Block 载荷提供者协议。

    该协议是 vLLM-Ascend 私有结构和 jiuwen-kvbm 之间的隔离层。不同
    vLLM-Ascend 版本只需要实现该协议，KVBM 核心不直接访问 engine 内部字段。
    """

    def read_block(self, block_id: int) -> Any:
        """读取指定 KV Block 的载荷。

        参数:
            block_id: vLLM-Ascend 运行时中的物理 Block 编号。

        返回:
            可被 RAM/Disk backend 保存的 Block 载荷。
        """

    def write_block(self, block_id: int, payload: Any) -> None:
        """写回指定 KV Block 的载荷。

        参数:
            block_id: vLLM-Ascend 运行时中的物理 Block 编号。
            payload: 从 RAM/Disk 读取出的 Block 载荷。

        返回:
            无返回值。
        """

    def describe_block(self, block_id: int) -> MemoryRegion:
        """描述指定 KV Block 的内存区域。

        参数:
            block_id: vLLM-Ascend 运行时中的物理 Block 编号。

        返回:
            中性的 MemoryRegion 描述。
        """

    def delete_block(self, block_id: int) -> None:
        """释放或标记复用指定 KV Block。

        参数:
            block_id: vLLM-Ascend 运行时中的物理 Block 编号。

        返回:
            无返回值。
        """


class VllmAscendKVAccessor:
    """版本无关的 vLLM-Ascend KVAccessor。

    参数:
        provider: 实现 BlockPayloadProvider 协议的版本适配对象。

    说明:
        本类不探测、不读取 vLLM-Ascend 私有字段，只把调用转发给 provider。
        因此它本身不依赖 vLLM-Ascend 具体版本。
    """

    def __init__(self, provider: BlockPayloadProvider):
        self.provider = provider

    def read_block(self, block_id: int) -> Any:
        """读取指定 KV Block 的载荷。

        参数:
            block_id: vLLM-Ascend 物理 Block 编号。

        返回:
            provider 返回的 Block 载荷。
        """

        return self.provider.read_block(block_id)

    def write_block(self, block_id: int, payload: Any) -> None:
        """写回指定 KV Block 的载荷。

        参数:
            block_id: vLLM-Ascend 物理 Block 编号。
            payload: 待写回的 Block 载荷。

        返回:
            无返回值。
        """

        self.provider.write_block(block_id, payload)

    def describe_block(self, block_id: int) -> MemoryRegion:
        """描述指定 KV Block 的内存区域。

        参数:
            block_id: vLLM-Ascend 物理 Block 编号。

        返回:
            provider 返回的 MemoryRegion。
        """

        return self.provider.describe_block(block_id)

    def delete_block(self, block_id: int) -> None:
        """释放或标记复用指定 KV Block。

        参数:
            block_id: vLLM-Ascend 物理 Block 编号。

        返回:
            无返回值。
        """

        self.provider.delete_block(block_id)
