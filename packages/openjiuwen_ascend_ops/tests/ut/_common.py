"""Shared layout / timing / golden helpers for rq_uniform UT."""

from __future__ import annotations

import math
import time
from typing import Callable

import torch

# ---- Full-attn pack layout (block_size % 16 == 0) ----
HEAD_SIZE = 128
BLOCK_SIZE = 16
SEED = 2026

# k8v4: key int8 Scheme A (D+4)/row; value int4 (D/2+4)/row
K8V4_KEY_BYTES_PER_ROW = HEAD_SIZE + 4  # 132
K8V4_VAL_BYTES_PER_ROW = HEAD_SIZE // 2 + 4  # 68
K8V4_KEY_TILE = 16 * K8V4_KEY_BYTES_PER_ROW  # 2112
K8V4_VAL_TILE = 16 * K8V4_VAL_BYTES_PER_ROW  # 1088

# k4v4: key+value both int4 (D/2+4)/row
K4V4_BYTES_PER_ROW = HEAD_SIZE // 2 + 4  # 68
K4V4_TILE = 16 * K4V4_BYTES_PER_ROW  # 1088

# SFA Schema A fused slot
SFA_NOPE_DIM = 512
SFA_ROPE_DIM = 64
SFA_CODE_BYTES = SFA_NOPE_DIM // 2  # 256
SFA_ROPE_BYTES = SFA_ROPE_DIM * 2  # 128
SFA_SLOT_BYTES = SFA_CODE_BYTES + SFA_ROPE_BYTES + 4  # 388
SFA_SCALE_OFF = SFA_CODE_BYTES + SFA_ROPE_BYTES  # 384
SFA_OFFSET_OFF = SFA_SCALE_OFF + 2  # 386

# MLA latent Scheme A
MLA_LATENT = 512
MLA_VECTOR_TILE = 16


def haar(dim: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(dim, dim, generator=g)
    q, r = torch.linalg.qr(a)
    s = torch.sign(torch.diag(r))
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (q * s).contiguous()


def dense_rotation(dim: int, dtype: torch.dtype, device: torch.device, seed: int = SEED + 17) -> torch.Tensor:
    return haar(dim, seed).to(device=device, dtype=dtype).contiguous()


def u8_as_fp(x: torch.Tensor, dtype: torch.dtype) -> float:
    return float(x.contiguous().view(dtype).reshape(-1)[0].float().item())


def decode_int4_row(
    block: torch.Tensor,
    pos: int,
    *,
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """y = vmin + (signed_nibble + 8) * vstep for int4 SoA row."""
    row_code = head_size // 2
    meta = 2
    code = block[pos * row_code:pos * row_code + row_code].to(torch.int32)
    low = code & 0x0F
    high = (code >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low).float()
    high = torch.where(high >= 8, high - 16, high).float()
    idx4 = torch.empty(head_size, dtype=torch.float32)
    idx4[0::2] = low
    idx4[1::2] = high
    vmin = u8_as_fp(
        block[block_size * row_code + pos * meta:block_size * row_code + pos * meta + meta],
        dtype,
    )
    vstep = u8_as_fp(
        block[
            block_size * (row_code + meta) + pos * meta:block_size * (row_code + meta) + pos * meta + meta
        ],
        dtype,
    )
    return vmin + (idx4 + 8.0) * vstep


def decode_k8v4_key_row(
    block: torch.Tensor,
    pos: int,
    *,
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Scheme A: y = base + int8(code) * step."""
    code = block[pos * head_size:(pos + 1) * head_size].view(torch.int8).float()
    base = u8_as_fp(
        block[block_size * head_size + pos * 2:block_size * head_size + pos * 2 + 2],
        dtype,
    )
    step = u8_as_fp(
        block[block_size * (head_size + 2) + pos * 2:block_size * (head_size + 2) + pos * 2 + 2],
        dtype,
    )
    return base + code * step


def build_block_table(num_tokens: int, block_size: int = BLOCK_SIZE) -> tuple[torch.Tensor, int]:
    nblocks = (num_tokens + block_size - 1) // block_size
    return torch.arange(nblocks, dtype=torch.int32).reshape(1, -1), nblocks


def pack_kv(
    ops,
    pack_name: str,
    key: torch.Tensor,
    value: torch.Tensor,
    rotation_t: torch.Tensor,
    *,
    key_stride: int,
    val_stride: int,
    block_size: int = BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = key.device
    t, h = key.shape[0], key.shape[1]
    nblocks = (t + block_size - 1) // block_size
    key_cache = torch.zeros(nblocks, h, key_stride, dtype=torch.uint8, device=device)
    value_cache = torch.zeros(nblocks, h, val_stride, dtype=torch.uint8, device=device)
    getattr(ops, pack_name)(
        key,
        value,
        torch.arange(t, dtype=torch.int32, device=device),
        torch.tensor([0, t], dtype=torch.int32, device=device),
        rotation_t,
        key_cache,
        value_cache,
        1,
        block_size,
    )
    torch.npu.synchronize()
    return key_cache, value_cache


def golden_decode_attn(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    rotation_key: torch.Tensor,
    rotation_value: torch.Tensor,
) -> torch.Tensor:
    """nq=1, single head-group: (softmax(Q R_k @ K^T) @ V) @ R_v."""
    scale = HEAD_SIZE**-0.5
    q = query[0, 0].float().cpu() @ rotation_key.float().cpu()
    scores = keys @ q * scale
    out = torch.softmax(scores, dim=-1) @ values
    return out.float() @ rotation_value.float().cpu()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float().flatten(), b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(af, bf, dim=0).item())


def timed_ms(fn: Callable[[], None], *, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3 / max(iters, 1)


def scheme_a_encode(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU Scheme A (signed store): code=int8(q-128), base=ymin+128*step."""
    ymin = rows.amin(-1)
    ymax = rows.amax(-1)
    step = ((ymax - ymin) / 255.0).clamp_min(1e-6)
    q = torch.round((rows - ymin.unsqueeze(-1)) / step.unsqueeze(-1)).clamp(0, 255)
    codes = (q - 128).to(torch.int8)
    base = (ymin + 128.0 * step).to(torch.float16)
    return codes, base, step.to(torch.float16)


def assert_finite(t: torch.Tensor, name: str = "tensor") -> None:
    if not bool(torch.isfinite(t.float()).all().item()):
        raise AssertionError(f"{name} has non-finite values")


def mla_block_stride(block_size: int, latent_dim: int = MLA_LATENT) -> int:
    return block_size * (latent_dim + 4)


def require_op(ops, name: str) -> None:
    import pytest

    if not hasattr(ops, name):
        pytest.skip(f"torch.ops.openjiuwen.{name} is not registered")
