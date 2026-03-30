"""将 firecrawl-py 返回的 Pydantic 模型转为可 JSON 序列化的 dict。"""

from typing import Any


def firecrawl_to_plain(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json", by_alias=True)
        except Exception:
            return obj.model_dump(by_alias=True)
    return obj
