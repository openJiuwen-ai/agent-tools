from __future__ import annotations

from typing import Any, Protocol

from jiuwen_kvbm.types import KVBlockMetadata


class TransferBackend(Protocol):
    @property
    def bytes_used(self) -> int:
        """当前后端已占用的字节数。

        返回:
            后端当前持有载荷的总字节数。
        """

    def put(self, metadata: KVBlockMetadata, payload: Any) -> str:
        """存储载荷并返回后端本地引用。

        参数:
            metadata: 待存储 Block 的元数据。
            payload: 待存储的 Block 载荷。

        返回:
            后端内部可用于再次读取或删除该载荷的引用字符串。
        """

    def get(self, metadata: KVBlockMetadata) -> Any:
        """根据 metadata.store_ref 加载载荷。

        参数:
            metadata: 包含 store_ref 的 Block 元数据。

        返回:
            后端读取出的 Block 载荷。
        """

    def delete(self, metadata: KVBlockMetadata) -> None:
        """删除 metadata.store_ref 指向的载荷。

        参数:
            metadata: 包含 store_ref 的 Block 元数据。

        返回:
            无返回值。
        """

    def contains(self, metadata: KVBlockMetadata) -> bool:
        """判断后端是否仍持有该 Block 载荷。

        参数:
            metadata: 包含 store_ref 的 Block 元数据。

        返回:
            如果后端仍持有该载荷，返回 True；否则返回 False。
        """
