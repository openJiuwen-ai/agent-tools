from __future__ import annotations

import sys
from typing import Any


def estimate_payload_size(payload: Any) -> int:
    """以通用方式估算载荷大小。

    参数:
        payload: 需要估算大小的载荷对象。

    返回:
        估算得到的字节数。
    """
    if isinstance(payload, bytes | bytearray | memoryview):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode())
    nbytes = getattr(payload, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return sys.getsizeof(payload)
