"""Tensor-provider aliases for the compatible vLLM-Ascend 0.23 KV layout."""

from jiuwen_kvbm.accessor.vllm_ascend_018 import (
    VllmAscend018BlockPayload,
    VllmAscend018LayerPayload,
    VllmAscend018TensorProvider,
    create_vllm_ascend_018_provider_from_engine,
    locate_vllm_ascend_018_kv_caches,
    summarize_vllm_ascend_018_kv_caches,
)

VllmAscend023BlockPayload = VllmAscend018BlockPayload
VllmAscend023LayerPayload = VllmAscend018LayerPayload
VllmAscend023TensorProvider = VllmAscend018TensorProvider
create_vllm_ascend_023_provider_from_engine = (
    create_vllm_ascend_018_provider_from_engine
)
locate_vllm_ascend_023_kv_caches = locate_vllm_ascend_018_kv_caches
summarize_vllm_ascend_023_kv_caches = summarize_vllm_ascend_018_kv_caches

__all__ = [
    "VllmAscend023BlockPayload",
    "VllmAscend023LayerPayload",
    "VllmAscend023TensorProvider",
    "create_vllm_ascend_023_provider_from_engine",
    "locate_vllm_ascend_023_kv_caches",
    "summarize_vllm_ascend_023_kv_caches",
]
