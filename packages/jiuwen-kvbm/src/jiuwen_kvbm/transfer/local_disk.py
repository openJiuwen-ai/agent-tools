from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from jiuwen_kvbm.types import KVBlockMetadata
from jiuwen_kvbm.transfer.payload_codec import decode_payload, encode_payload


class LocalDiskBackend:
    """本地磁盘存储后端。

    使用数据专用 JSON/tensor 编码，不执行缓存文件指定的 Python 类或函数。

    参数:
        root: 本地磁盘存储根目录。
        max_bytes: 磁盘后端允许占用的最大字节数；为 None 时不做容量限制。
    """

    def __init__(self, root: str | Path, max_bytes: int | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._sizes: dict[str, int] = {}
        self._lock = threading.RLock()

    @property
    def bytes_used(self) -> int:
        """返回当前磁盘后端保存的载荷总大小。

        返回:
            当前已保存载荷文件的总字节数。
        """
        with self._lock:
            return sum(self._sizes.values())

    def put(self, metadata: KVBlockMetadata, payload: Any) -> str:
        """将 Block 载荷序列化到本地磁盘。

        参数:
            metadata: 待保存 Block 的元数据。
            payload: 待保存的 Block 载荷。

        返回:
            磁盘后端内部引用，即相对 root 的文件名。

        异常:
            OSError: 保存后会超过 max_bytes 容量限制，或文件写入失败。
        """
        with self._lock:
            encoded = encode_payload(payload)
            ref = _safe_ref(metadata)
            size = len(encoded)
            if self.max_bytes is not None and self.bytes_used - self._sizes.get(ref, 0) + size > self.max_bytes:
                raise OSError("LocalDiskBackend capacity exceeded")
            path = self._payload_path(ref)
            fd, temporary = tempfile.mkstemp(prefix=".kvbm-", dir=self.root)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(encoded)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self._sizes[ref] = size
            return ref

    def get(self, metadata: KVBlockMetadata) -> Any:
        """从本地磁盘反序列化 Block 载荷。

        参数:
            metadata: 包含磁盘后端 store_ref 的 Block 元数据。

        返回:
            从磁盘读取出的 Block 载荷。

        异常:
            KeyError: metadata 缺少 store_ref。
            OSError: 读取文件失败。
        """
        with self._lock:
            if metadata.store_ref is None:
                raise KeyError("metadata has no store_ref")
            path = self._payload_path(metadata.store_ref)
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), mode=0o600)
            with os.fdopen(fd, "rb") as handle:
                return decode_payload(handle.read())

    def delete(self, metadata: KVBlockMetadata) -> None:
        """删除磁盘上的 Block 载荷文件。

        参数:
            metadata: 包含磁盘后端 store_ref 的 Block 元数据。

        返回:
            无返回值。
        """
        with self._lock:
            if metadata.store_ref is None:
                return
            path = self._payload_path(metadata.store_ref)
            if path.exists():
                path.unlink()
            self._sizes.pop(metadata.store_ref, None)

    def contains(self, metadata: KVBlockMetadata) -> bool:
        """判断 Block 载荷文件是否仍存在。

        参数:
            metadata: 包含磁盘后端 store_ref 的 Block 元数据。

        返回:
            如果文件仍存在，返回 True；否则返回 False。
        """
        with self._lock:
            return metadata.store_ref is not None and self._payload_path(metadata.store_ref).is_file()

    def _payload_path(self, ref: str) -> Path:
        if not isinstance(ref, str) or not ref:
            raise ValueError("Invalid cache store_ref")
        if Path(ref).name != ref or "/" in ref or "\\" in ref:
            raise ValueError("Invalid cache store_ref")
        if ref in {".", ".."} or not ref.endswith(".kv"):
            raise ValueError("Invalid cache store_ref")
        path = self.root / ref
        if path.is_symlink():
            raise ValueError("Cache payload must not be a symbolic link")
        return path

    def clear_managed_payloads(self) -> int:
        """清理当前后端根目录下由 KVBM 管理的旧载荷文件。

        参数:
            无参数。

        返回:
            成功删除的 payload 文件数量。

        说明:
            清理当前 ``*.kv`` 与旧版 ``*.pkl`` 托管缓存，不反序列化旧文件。
        """

        with self._lock:
            deleted = 0
            for path in (*self.root.glob("*.kv"), *self.root.glob("*.pkl")):
                if not path.is_file():
                    continue
                try:
                    path.unlink()
                    deleted += 1
                except FileNotFoundError:
                    continue
            self._sizes.clear()
            return deleted


def _safe_ref(metadata: KVBlockMetadata) -> str:
    """生成可安全作为文件名使用的后端引用。

    参数:
        metadata: 需要生成引用的 Block 元数据。

    返回:
        由清洗后的 block_hash 和 block_id 组成的文件名。
    """
    clean_hash = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in metadata.block_hash)
    return f"{clean_hash}-{int(metadata.block_id)}.kv"
