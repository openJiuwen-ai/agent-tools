"""Data-only cache serialization; never import classes named by cache files."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from jiuwen_kvbm.accessor.vllm_ascend_018 import VllmAscend018BlockPayload, VllmAscend018LayerPayload


@dataclass
class HostValues:
    """Data-only tensor stand-in for non-torch accessor implementations."""

    values: list

    def to(self, device=None, dtype=None):
        return self


def encode_payload(value: Any) -> bytes:
    return json.dumps(_encode(value), separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode_payload(data: bytes) -> Any:
    return _decode(json.loads(data))


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, VllmAscend018BlockPayload):
        return {"kind": "block", "id": value.block_id, "layers": [_encode(layer) for layer in value.layers]}
    if isinstance(value, VllmAscend018LayerPayload):
        return {"kind": "layer", "name": value.layer_name, "key": _encode(value.key), "value": _encode(value.value)}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode(item) for item in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[_encode(key), _encode(item)] for key, item in value.items()]}
    # Torch is optional in CPU-only deployments and unit tests.
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": base64.b64encode(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).decode("ascii"),
        }
    values = getattr(value, "values", None)
    if isinstance(values, list):
        return {"kind": "host_values", "items": _encode(values)}
    raise TypeError(f"Unsupported cache payload type: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if not isinstance(value, dict):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise ValueError("Invalid cache payload")
    kind = value.get("kind")
    if kind == "bytes":
        return base64.b64decode(value["data"], validate=True)
    if kind == "block":
        return VllmAscend018BlockPayload(block_id=int(value["id"]), layers=[_decode(v) for v in value["layers"]])
    if kind == "layer":
        return VllmAscend018LayerPayload(str(value["name"]), _decode(value["key"]), _decode(value["value"]))
    if kind in {"list", "tuple"}:
        items = [_decode(v) for v in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {_decode(key): _decode(item) for key, item in value["items"]}
    if kind == "host_values":
        return HostValues(_decode(value["items"]))
    if kind == "tensor":
        import math
        import torch

        allowed = {"float16", "bfloat16", "float32", "float64", "int8", "uint8", "int16", "int32", "int64", "bool"}
        if value["dtype"] not in allowed:
            raise ValueError("Unsupported cache tensor dtype")
        shape = value["shape"]
        if not isinstance(shape, list) or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("Invalid cache tensor shape")
        dtype = getattr(torch, value["dtype"])
        data = bytearray(base64.b64decode(value["data"], validate=True))
        if len(data) != math.prod(shape) * torch.empty((), dtype=dtype).element_size():
            raise ValueError("Cache tensor size does not match shape")
        if not data:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(data, dtype=dtype).clone().reshape(shape)
    raise ValueError("Unknown cache payload kind")
