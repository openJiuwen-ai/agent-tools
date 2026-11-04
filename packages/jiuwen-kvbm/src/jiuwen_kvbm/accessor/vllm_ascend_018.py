from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jiuwen_kvbm.types import MemoryRegion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VllmAscend018LayerPayload:
    """vLLM-Ascend v0.18.0 单层 KV Block 载荷。

    属性:
        layer_name: vLLM-Ascend kv_caches 字典中的层名。
        key: 已复制到 CPU 的 key cache block。
        value: 已复制到 CPU 的 value cache block。
    """

    layer_name: str
    key: Any
    value: Any

    @property
    def nbytes(self) -> int:
        """返回该层 key/value 载荷的总字节数。

        返回:
            key cache block 和 value cache block 的字节数之和。
        """

        return _tensor_nbytes(self.key) + _tensor_nbytes(self.value)


@dataclass(frozen=True)
class VllmAscend018BlockPayload:
    """vLLM-Ascend v0.18.0 单个物理 Block 的完整载荷。

    属性:
        block_id: vLLM-Ascend 物理 Block 编号。
        layers: 每层对应的一组 key/value CPU tensor。
    """

    block_id: int
    layers: list[VllmAscend018LayerPayload]

    @property
    def nbytes(self) -> int:
        """返回该物理 block 跨所有层的总字节数。

        返回:
            所有层 key/value cache block 的字节数之和。
        """

        return sum(layer.nbytes for layer in self.layers)


class VllmAscend018TensorProvider:
    """vLLM-Ascend v0.18.0 KV cache tensor provider。

    参数:
        kv_caches: vLLM-Ascend v0.18.0 worker 侧 KV cache 字典。该版本的
            CpuNpuOffloadingHandler 使用相同结构：每个 value 可通过 [0] 取 key
            cache，通过 [1] 取 value cache，且物理 block 维度位于第 0 维。
        device_type: MemoryRegion 中记录的设备类型，默认 npu。
        device_id: NPU 设备编号；无法确定时可传 None。
        delete_mode: delete_block 的行为。默认 noop，只让 vLLM block manager 继续
            管理生命周期；zero 会将对应 NPU tensor slice 清零。
    """

    def __init__(
        self,
        kv_caches: dict[str, Any],
        *,
        device_type: str = "npu",
        device_id: int | None = None,
        delete_mode: str = "noop",
    ):
        if delete_mode not in {"noop", "zero"}:
            raise ValueError("delete_mode must be 'noop' or 'zero'")
        self.kv_caches = kv_caches
        self.device_type = device_type
        self.device_id = device_id
        self.delete_mode = delete_mode
        self._block_region_template: tuple[int, str, int] | None = None

    def diagnostics(self, block_id: int = 0) -> dict[str, Any]:
        """返回 provider 的只读诊断信息。

        参数:
            block_id: 用于验证 block 维度和估算大小的物理 Block 编号。

        返回:
            包含层数、首层 key/value shape、block_size、dtype、device 和
            block 字节数的字典。

        异常:
            IndexError: block_id 超出首层 KV cache 的 block 维范围。
            ValueError: kv_caches 为空。
        """

        return summarize_vllm_ascend_018_kv_caches(self.kv_caches, block_id=block_id)

    def tensor_base_addresses(self) -> tuple[int, ...]:
        """返回所有 KV tensor 的首地址。

        返回:
            按 layer0_key、layer0_value、layer1_key、layer1_value 顺序排列的
            data_ptr 地址元组。

        异常:
            ValueError: kv_caches 为空或 tensor 不支持 data_ptr。
        """

        if not is_vllm_ascend_018_kv_caches(self.kv_caches):
            raise ValueError("kv_caches does not match vLLM-Ascend 0.18 layout")
        addresses: list[int] = []
        for layer_cache in self.kv_caches.values():
            key_cache, value_cache = _split_layer_cache(layer_cache)
            addresses.append(_tensor_data_ptr(key_cache))
            addresses.append(_tensor_data_ptr(value_cache))
        return tuple(addresses)

    def blocks_cache_shape(self) -> tuple[int, ...]:
        """返回 LLM-DataDist blocks cache 使用的单 tensor shape。

        返回:
            首层 key cache 的完整 shape，通常为
            (num_blocks, block_size, num_kv_heads, head_dim)。

        异常:
            ValueError: kv_caches 为空或结构不符合 v0.18.0 预期。
        """

        if not is_vllm_ascend_018_kv_caches(self.kv_caches):
            raise ValueError("kv_caches does not match vLLM-Ascend 0.18 layout")
        first_cache = next(iter(self.kv_caches.values()))
        key_cache, _value_cache = _split_layer_cache(first_cache)
        return tuple(int(dim) for dim in key_cache.shape)

    def block_size_tokens(self) -> int:
        """返回真实物理 KV block 容纳的 token 数。

        返回:
            首层 key cache shape 的第 1 维，即
            (num_blocks, block_size, num_kv_heads, head_dim) 中的 block_size。

        异常:
            ValueError: KV cache shape 少于两个维度，无法识别 block token 数。
        """

        shape = self.blocks_cache_shape()
        if len(shape) < 2 or int(shape[1]) <= 0:
            raise ValueError(f"cannot infer KV block size from cache shape: {shape}")
        return int(shape[1])

    def blocks_cache_dtype(self) -> str:
        """返回 LLM-DataDist blocks cache 使用的 dtype 字符串。

        返回:
            首层 key cache 的 dtype 字符串，例如 torch.bfloat16。

        异常:
            ValueError: kv_caches 为空或结构不符合 v0.18.0 预期。
        """

        if not is_vllm_ascend_018_kv_caches(self.kv_caches):
            raise ValueError("kv_caches does not match vLLM-Ascend 0.18 layout")
        first_cache = next(iter(self.kv_caches.values()))
        key_cache, _value_cache = _split_layer_cache(first_cache)
        return str(getattr(key_cache, "dtype", ""))

    def read_block(self, block_id: int) -> VllmAscend018BlockPayload:
        """读取指定物理 Block 的所有层 KV 载荷。

        参数:
            block_id: vLLM-Ascend 物理 Block 编号。

        返回:
            包含所有层 key/value CPU tensor 的 VllmAscend018BlockPayload。

        异常:
            IndexError: block_id 超出任一层 KV cache 的 block 维范围。
        """

        layers: list[VllmAscend018LayerPayload] = []
        for layer_name, layer_cache in self.kv_caches.items():
            key_cache, value_cache = _split_layer_cache(layer_cache)
            layers.append(
                VllmAscend018LayerPayload(
                    layer_name=layer_name,
                    key=key_cache[block_id].detach().cpu().clone(),
                    value=value_cache[block_id].detach().cpu().clone(),
                )
            )
        return VllmAscend018BlockPayload(block_id=block_id, layers=layers)

    def write_block(self, block_id: int, payload: VllmAscend018BlockPayload) -> None:
        """将 CPU 载荷写回指定物理 Block。

        参数:
            block_id: vLLM-Ascend 物理 Block 编号。
            payload: read_block 返回的 CPU 载荷。

        返回:
            无返回值。

        异常:
            KeyError: payload 中的 layer_name 不存在于 kv_caches。
            IndexError: block_id 超出任一层 KV cache 的 block 维范围。
        """

        for layer in payload.layers:
            key_cache, value_cache = _split_layer_cache(self.kv_caches[layer.layer_name])
            key_block = key_cache[block_id]
            value_block = value_cache[block_id]
            key_block.copy_(layer.key.to(device=key_block.device, dtype=key_block.dtype))
            value_block.copy_(layer.value.to(device=value_block.device, dtype=value_block.dtype))

    def block_digest(self, block_id: int, *, include_layers: bool = False) -> dict[str, Any]:
        """计算指定物理 Block 跨所有层 K/V tensor 的 SHA-256 摘要。

        参数:
            block_id: vLLM-Ascend 物理 Block 编号。
            include_layers: 是否在整体摘要之外返回每层 key/value 摘要。

        返回:
            包含 block_id、sha256、nbytes 的摘要字典；include_layers 为 True
            时额外包含 layers 列表。

        说明:
            本方法会把完整物理 block 从 NPU 复制到 CPU，仅用于传输正确性调试，
            不应放入正常推理请求热路径。
        """

        payload = self.read_block(block_id)
        digest = hashlib.sha256()
        layer_results: list[dict[str, Any]] = []
        for layer in payload.layers:
            key_bytes = _tensor_bytes(layer.key)
            value_bytes = _tensor_bytes(layer.value)
            layer_digest = hashlib.sha256()
            _update_digest_part(layer_digest, layer.layer_name.encode("utf-8"))
            _update_digest_part(layer_digest, key_bytes)
            _update_digest_part(layer_digest, value_bytes)
            _update_digest_part(digest, layer.layer_name.encode("utf-8"))
            _update_digest_part(digest, key_bytes)
            _update_digest_part(digest, value_bytes)
            if include_layers:
                layer_results.append(
                    {
                        "layer_name": layer.layer_name,
                        "sha256": layer_digest.hexdigest(),
                        "key_sha256": hashlib.sha256(key_bytes).hexdigest(),
                        "value_sha256": hashlib.sha256(value_bytes).hexdigest(),
                        "nbytes": len(key_bytes) + len(value_bytes),
                    }
                )
        result: dict[str, Any] = {
            "block_id": int(block_id),
            "sha256": digest.hexdigest(),
            "nbytes": payload.nbytes,
            "layer_count": len(payload.layers),
        }
        if include_layers:
            result["layers"] = layer_results
        return result

    def describe_block(self, block_id: int) -> MemoryRegion:
        """描述指定物理 Block 的内存区域。

        参数:
            block_id: vLLM-Ascend 物理 Block 编号。

        返回:
            MemoryRegion，其中 nbytes 为该 block 跨所有层的 key/value 总字节数。

        异常:
            IndexError: block_id 超出任一层 KV cache 的 block 维范围。
        """

        if self._block_region_template is None:
            nbytes = 0
            layout_parts = []
            num_blocks: int | None = None
            for layer_name, layer_cache in self.kv_caches.items():
                key_cache, value_cache = _split_layer_cache(layer_cache)
                key_block = key_cache[block_id]
                value_block = value_cache[block_id]
                nbytes += key_block.numel() * key_block.element_size()
                nbytes += value_block.numel() * value_block.element_size()
                layout_parts.append(f"{layer_name}:{tuple(key_block.shape)}/{tuple(value_block.shape)}")
                key_shape = tuple(getattr(key_cache, "shape", ()))
                value_shape = tuple(getattr(value_cache, "shape", ()))
                if not key_shape or not value_shape:
                    raise ValueError("KV cache tensors must expose a non-empty shape")
                layer_num_blocks = min(int(key_shape[0]), int(value_shape[0]))
                num_blocks = layer_num_blocks if num_blocks is None else min(num_blocks, layer_num_blocks)
            self._block_region_template = (
                nbytes,
                "vllm-ascend-0.18:" + ",".join(layout_parts),
                num_blocks or 0,
            )
        nbytes, layout, num_blocks = self._block_region_template
        if block_id < 0 or block_id >= num_blocks:
            raise IndexError(f"block_id {block_id} is outside valid range [0, {num_blocks})")
        return MemoryRegion(
            device_type=self.device_type,
            device_id=self.device_id,
            nbytes=nbytes,
            layout=layout,
            backend_handle=block_id,
        )

    def delete_block(self, block_id: int) -> None:
        """释放或清理指定物理 Block。

        参数:
            block_id: vLLM-Ascend 物理 Block 编号。

        返回:
            无返回值。

        说明:
            默认 noop，避免绕过 vLLM block manager 破坏生命周期。测试或明确需要
            清零时可设置 delete_mode="zero"。
        """

        if self.delete_mode == "noop":
            return
        for layer_cache in self.kv_caches.values():
            key_cache, value_cache = _split_layer_cache(layer_cache)
            key_cache[block_id].zero_()
            value_cache[block_id].zero_()


def _split_layer_cache(layer_cache: Any) -> tuple[Any, Any]:
    """拆分 vLLM-Ascend v0.18.0 单层 KV cache。

    参数:
        layer_cache: vLLM-Ascend kv_caches 字典中的单层 cache 对象。

    返回:
        (key_cache, value_cache) 二元组。
    """

    return layer_cache[0], layer_cache[1]


def _tensor_bytes(tensor: Any) -> bytes:
    """将 CPU tensor 的原始存储内容转换为字节串。

    参数:
        tensor: read_block 产生的 CPU tensor；测试环境也可传入带 values 的
           轻量 fake tensor。

    返回:
        按 tensor 连续内存布局导出的字节串。
    """

    value = tensor.detach().cpu() if hasattr(tensor, "detach") else tensor
    if hasattr(value, "contiguous") and hasattr(value, "view"):
        try:
            import torch

            return value.contiguous().view(torch.uint8).numpy().tobytes()
        except Exception:
            logger.debug("Byte-view export unavailable; trying NumPy export", exc_info=True)
    if hasattr(value, "numpy"):
        try:
            return value.numpy().tobytes()
        except Exception:
            logger.debug("NumPy export unavailable; trying the test payload representation", exc_info=True)
    if hasattr(value, "values"):
        return repr(list(value.values)).encode("utf-8")
    raise TypeError(f"cannot export tensor bytes from {type(tensor).__name__}")


def _update_digest_part(digest: Any, value: bytes) -> None:
    """以长度前缀方式向摘要加入一个字段，避免字段边界歧义。

    参数:
        digest: hashlib 摘要对象。
        value: 待加入摘要的字节串。

    返回:
        无返回值。
    """

    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def is_vllm_ascend_018_kv_caches(value: Any) -> bool:
    """判断对象是否符合 vLLM-Ascend v0.18.0 kv_caches 结构。

    参数:
        value: 待检查的 Python 对象。

    返回:
        如果对象是非空 dict，且首层 cache 可通过 [0]/[1] 取得带 shape 的
        key/value tensor，则返回 True；否则返回 False。
    """

    if not isinstance(value, dict) or not value:
        return False
    try:
        first_cache = next(iter(value.values()))
        key_cache, value_cache = _split_layer_cache(first_cache)
        return hasattr(key_cache, "shape") and hasattr(value_cache, "shape")
    except Exception:
        return False


def summarize_vllm_ascend_018_kv_caches(kv_caches: dict[str, Any], *, block_id: int = 0) -> dict[str, Any]:
    """汇总 vLLM-Ascend v0.18.0 kv_caches 的只读诊断信息。

    参数:
        kv_caches: vLLM-Ascend worker 侧 KV cache 字典。
        block_id: 用于验证 block 维度和估算大小的物理 Block 编号。

    返回:
        诊断字典，包含 layer_count、first_layer、key_shape、value_shape、
        block_size、key_dtype、value_dtype、key_device、value_device、
        num_blocks 和 block_nbytes 等字段。

    异常:
        ValueError: kv_caches 为空或结构不符合 v0.18.0 预期。
        IndexError: block_id 超出首层 KV cache 的 block 维范围。
    """

    if not is_vllm_ascend_018_kv_caches(kv_caches):
        raise ValueError("kv_caches does not match vLLM-Ascend 0.18 layout")

    first_layer, first_cache = next(iter(kv_caches.items()))
    key_cache, value_cache = _split_layer_cache(first_cache)
    key_block = key_cache[block_id]
    value_block = value_cache[block_id]
    first_layer_block_nbytes = _tensor_nbytes(key_block) + _tensor_nbytes(value_block)
    total_block_nbytes = 0
    for layer_cache in kv_caches.values():
        layer_key_cache, layer_value_cache = _split_layer_cache(layer_cache)
        total_block_nbytes += _tensor_nbytes(layer_key_cache[block_id])
        total_block_nbytes += _tensor_nbytes(layer_value_cache[block_id])

    return {
        "layer_count": len(kv_caches),
        "first_layer": first_layer,
        "layer_cache_type": type(first_cache).__name__,
        "key_shape": tuple(key_cache.shape),
        "value_shape": tuple(value_cache.shape),
        "key_block_shape": tuple(key_block.shape),
        "value_block_shape": tuple(value_block.shape),
        "block_size": int(key_block.shape[0]),
        "key_dtype": str(getattr(key_cache, "dtype", "")),
        "value_dtype": str(getattr(value_cache, "dtype", "")),
        "key_device": str(getattr(key_cache, "device", "")),
        "value_device": str(getattr(value_cache, "device", "")),
        "num_blocks": int(key_cache.shape[0]),
        "first_layer_block_nbytes": int(first_layer_block_nbytes),
        "total_block_nbytes": int(total_block_nbytes),
        "block_nbytes": int(total_block_nbytes),
    }


def locate_vllm_ascend_018_kv_caches(root: Any, *, max_depth: int = 16) -> tuple[dict[str, Any], str]:
    """从真实 vLLM/vLLM-Ascend worker 对象中定位 kv_caches。

    参数:
        root: 搜索起点，通常是 Jiuwen worker 中创建的 AsyncLLM engine。
        max_depth: 从 root 向下遍历 __dict__、dict、list/tuple 的最大深度。

    返回:
        (kv_caches, path) 二元组，其中 path 是诊断用的对象路径。

    异常:
        LookupError: 在限定深度内没有找到符合 vLLM-Ascend 0.18 结构的
            kv_caches 对象。

    说明:
        该函数只读取对象的 __dict__ 和容器内容，不调用 property 或私有方法，
        目的是在真实环境里安全确认 vLLM-Ascend 持有 KV tensor 的路径。
    """

    queue: list[tuple[Any, str, int]] = [(root, "engine", 0)]
    seen: set[int] = set()
    diagnostics = VllmAscend018LocatorDiagnostics()
    mp_client_path = detect_vllm_async_mp_client(root)
    if mp_client_path is not None:
        raise LookupError(
            "vLLM-Ascend 0.18 kv_caches are not directly reachable from this AsyncLLM process; "
            f"detected vLLM AsyncMPClient at {mp_client_path}. The KV tensors live in the "
            "vLLM engine-core worker subprocess, so KVBM must be installed inside that process "
            "or integrated through a vLLM/vLLM-Ascend worker hook."
        )

    while queue:
        obj, path, depth = queue.pop(0)
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        diagnostics.record_visit(path, obj, depth)

        if is_vllm_ascend_018_kv_caches(obj):
            return obj, path
        if depth >= max_depth:
            continue

        for child_name, child in _iter_diagnostic_children(obj):
            child_path = f"{path}.{child_name}"
            diagnostics.record_child(child_path, child, depth + 1)
            if child_name == "kv_caches" and is_vllm_ascend_018_kv_caches(child):
                return child, child_path
            queue.append((child, child_path, depth + 1))

    raise LookupError(
        "vLLM-Ascend 0.18 kv_caches not found under engine "
        f"within depth {max_depth}; diagnostics={diagnostics.to_dict()}"
    )


def detect_vllm_async_mp_client(root: Any) -> str | None:
    """检测当前对象是否是 vLLM 前端 AsyncLLM 多进程客户端。

    参数:
        root: 搜索起点，通常是 Jiuwen worker 中创建的 AsyncLLM engine。

    返回:
        如果检测到 vLLM AsyncMPClient，则返回该对象路径；否则返回 None。

    说明:
        vLLM v1 默认可能将 engine-core 放在子进程中运行。此时当前进程只持有
        AsyncMPClient，真实 model_runner.kv_caches 位于子进程内，无法通过
        Python 对象遍历从前端 AsyncLLM 进程直接拿到。
    """

    engine_core = getattr(root, "engine_core", None)
    if engine_core is not None and _type_name(engine_core).endswith("vllm.v1.engine.core_client.AsyncMPClient"):
        return "engine.engine_core"
    return None


def create_vllm_ascend_018_provider_from_engine(root: Any) -> VllmAscend018TensorProvider:
    """从真实 vLLM/vLLM-Ascend engine 创建 v0.18.0 TensorProvider。

    参数:
        root: 搜索起点，通常是 Jiuwen worker 中创建的 AsyncLLM engine。

    返回:
        已绑定真实 kv_caches 的 VllmAscend018TensorProvider。

    异常:
        LookupError: 未能在 engine 下定位到 vLLM-Ascend 0.18 kv_caches。
    """

    kv_caches, _path = locate_vllm_ascend_018_kv_caches(root)
    return VllmAscend018TensorProvider(kv_caches)


class VllmAscend018LocatorDiagnostics:
    """记录 vLLM-Ascend v0.18.0 kv_caches 定位过程中的轻量诊断信息。

    说明:
        该类只保存有限数量的路径摘要，避免真实 worker 对象树过大时日志失控。
    """

    def __init__(self, *, max_items: int = 80):
        """创建定位诊断记录器。

        参数:
            max_items: 每类路径最多记录多少条。

        返回:
            无返回值。
        """

        self.max_items = max_items
        self.visited_count = 0
        self.interesting_paths: list[dict[str, Any]] = []
        self.kv_like_dicts: list[dict[str, Any]] = []

    def record_visit(self, path: str, obj: Any, depth: int) -> None:
        """记录已访问路径。

        参数:
            path: 当前对象路径。
            obj: 当前对象。
            depth: 当前搜索深度。

        返回:
            无返回值。
        """

        self.visited_count += 1
        if _is_interesting_path(path):
            self._append_limited(
                self.interesting_paths,
                {
                    "path": path,
                    "depth": depth,
                    "type": _type_name(obj),
                    "summary": _safe_object_summary(obj),
                },
            )

    def record_child(self, path: str, obj: Any, depth: int) -> None:
        """记录子对象中的疑似 KV/cache 路径。

        参数:
            path: 子对象路径。
            obj: 子对象。
            depth: 子对象深度。

        返回:
            无返回值。
        """

        if _is_interesting_path(path):
            self._append_limited(
                self.interesting_paths,
                {
                    "path": path,
                    "depth": depth,
                    "type": _type_name(obj),
                    "summary": _safe_object_summary(obj),
                },
            )
        if isinstance(obj, dict) and obj and _is_kv_like_path(path):
            first_key = next(iter(obj.keys()))
            first_value = obj[first_key]
            self._append_limited(
                self.kv_like_dicts,
                {
                    "path": path,
                    "depth": depth,
                    "size": len(obj),
                    "first_key": str(first_key),
                    "first_value_type": _type_name(first_value),
                    "first_value_summary": _safe_object_summary(first_value),
                },
            )

    def to_dict(self) -> dict[str, Any]:
        """导出日志友好的诊断信息。

        返回:
            包含访问数量、疑似路径和 KV-like dict 摘要的字典。
        """

        return {
            "visited_count": self.visited_count,
            "interesting_paths": self.interesting_paths,
            "kv_like_dicts": self.kv_like_dicts,
        }

    def _append_limited(self, target: list[dict[str, Any]], item: dict[str, Any]) -> None:
        """向列表追加元素，并控制最大长度。

        参数:
            target: 目标列表。
            item: 要追加的诊断项。

        返回:
            无返回值。
        """

        if len(target) < self.max_items:
            target.append(item)


def _is_interesting_path(path: str) -> bool:
    """判断路径名是否值得记录到定位诊断中。

    参数:
        path: 对象路径。

    返回:
        路径包含 kv、cache、runner、worker、executor、engine 等关键词时返回 True。
    """

    lower_path = path.lower()
    return any(keyword in lower_path for keyword in ("kv", "cache", "runner", "worker", "executor", "engine"))


def _is_kv_like_path(path: str) -> bool:
    """判断路径是否像 KV cache 相关路径。

    参数:
        path: 对象路径。

    返回:
        路径包含 kv 或 cache 时返回 True。
    """

    lower_path = path.lower()
    return "kv" in lower_path or "cache" in lower_path


def _type_name(obj: Any) -> str:
    """返回对象类型名。

    参数:
        obj: 任意对象。

    返回:
        module.qualname 形式的类型名。
    """

    obj_type = type(obj)
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


def _safe_object_summary(obj: Any) -> str:
    """生成不会触发大规模数据读取的对象摘要。

    参数:
        obj: 任意对象。

    返回:
        包含长度、shape、dtype、device 等元信息的字符串。
    """

    parts = []
    try:
        parts.append(f"len={len(obj)}")
    except Exception:
        logger.debug("Object length unavailable during diagnostic summary", exc_info=True)
    for name in ("shape", "dtype", "device"):
        try:
            if hasattr(obj, name):
                parts.append(f"{name}={getattr(obj, name)}")
        except Exception:
            logger.debug("Object attribute %s unavailable during diagnostic summary", name, exc_info=True)
    if isinstance(obj, dict) and obj:
        try:
            first_key = next(iter(obj.keys()))
            parts.append(f"first_key={first_key}")
            parts.append(f"first_value_type={_type_name(obj[first_key])}")
        except Exception:
            logger.debug("First mapping entry unavailable during diagnostic summary", exc_info=True)
    return ", ".join(parts)


def _tensor_nbytes(tensor: Any) -> int:
    """返回 tensor-like 对象的字节数。

    参数:
        tensor: 具有 nbytes，或 numel()/element_size() 的 tensor-like 对象。

    返回:
        估算得到的字节数。
    """

    nbytes = getattr(tensor, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return int(tensor.numel() * tensor.element_size())


def _tensor_data_ptr(tensor: Any) -> int:
    """读取 tensor 首地址。

    参数:
        tensor: torch/npu tensor 对象。

    返回:
        tensor.data_ptr() 返回的地址整数。

    异常:
        ValueError: tensor 不支持 data_ptr 或返回空地址。
    """

    data_ptr = getattr(tensor, "data_ptr", None)
    if data_ptr is None:
        raise ValueError("tensor does not expose data_ptr")
    address = int(data_ptr())
    if address <= 0:
        raise ValueError(f"tensor data_ptr is invalid: {address}")
    return address


def _iter_diagnostic_children(obj: Any) -> list[tuple[str, Any]]:
    """列出诊断搜索允许访问的子对象。

    参数:
        obj: 当前搜索对象。

    返回:
        (名称, 子对象) 列表。
    """

    if isinstance(obj, Mapping):
        return [(str(key), value) for key, value in obj.items()]
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [(str(index), value) for index, value in enumerate(obj)]
    if hasattr(obj, "__dict__"):
        return list(vars(obj).items())
    return []
