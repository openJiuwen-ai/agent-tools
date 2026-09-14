import torch


def bit_residual_pack_k8v4(
    kv: tuple[torch.Tensor, torch.Tensor],
    layout: tuple[torch.Tensor, torch.Tensor],
    rotation_t: torch.Tensor,
    caches: tuple[torch.Tensor, torch.Tensor],
    config: tuple[int, int],
):
    """BitResidual K8V4 pack kernel wrapper.

    Args:
        kv: (key, value) input tensors, shape [tokens, heads, head_size].
        layout: (slot_mapping, query_start_loc) scheduling tensors.
        rotation_t: rotation matrix [head_size, head_size].
        caches: (key_cache, value_cache) output cache tensors (modified in-place).
        config: (num_reqs, block_size) kernel configuration.

    Returns:
        (key_cache, value_cache) after packing.
    """
    key, value = kv
    slot_mapping, query_start_loc = layout
    key_cache, value_cache = caches
    num_reqs, block_size = config
    torch.ops.openjiuwen.bit_residual_pack_k8v4(
        key,
        value,
        slot_mapping,
        query_start_loc,
        rotation_t,
        key_cache,
        value_cache,
        num_reqs,
        block_size,
    )
    return key_cache, value_cache