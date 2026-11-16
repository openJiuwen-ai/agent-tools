"""A5 antiquant port: int8 KV, scale fixed at 1, no mask and no rotation.

quant_mode=2 covers decode and prefill, D=128 and D=256. The long decode
shape is the cross-core bar: one KV row is split across cubes and the
partial O / LSE must be merged. Cosine is against that merged output.

FlashDecode follows A5 SplitDequant. Prefill (s1>1) sets isPFAFlag_ and
never enables FD. Short decode stays under two sInner tiles, so FD stays off.
"""

from __future__ import annotations

import logging

import pytest
import torch

from . import _common as C

logger = logging.getLogger(__name__)

QUANT_MODE_A5_ANTIQUANT_PORT = 2

# (name, kv_len, kv_heads, q_heads, s1, head_size, flash_decode)
# short: maxActualseq=128 < 512*2, FD off.
# fd: B=1, G=1, kv=2048. loopTimes=1 < 0.4*aic and KV>=2048, SplitS2 28 -> 4.
# prefill: s1>1, PA antiquant, FD off by SplitDequant.
# d256: same short decode with D=256 / sInner=256.
ACCURACY_CASES = [
    ("short", 128, 2, 4, 1, 128, False),
    ("fd", 2048, 1, 1, 1, 128, True),
    ("prefill", 128, 2, 4, 16, 128, False),
    ("d256", 128, 2, 4, 1, 256, False),
]

_WARMUP = 10
_ITERS = 100


def _int8_codes(kv_len: int, kv_heads: int, head_size: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        -128, 128, (kv_len, kv_heads, head_size), generator=g, dtype=torch.int32
    ).to(torch.int8)


def _paged_int8(codes: torch.Tensor) -> torch.Tensor:
    """BNBD cache [blocks, kv_heads, block_size * D] of int8 bytes."""
    tokens, heads, dim = codes.shape
    nblocks = (tokens + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    cache = torch.zeros(nblocks, heads, C.BLOCK_SIZE * dim, dtype=torch.uint8)
    raw = codes.contiguous().view(torch.uint8)
    for pos in range(tokens):
        blk = pos // C.BLOCK_SIZE
        row = pos % C.BLOCK_SIZE
        cache[blk, :, row * dim:(row + 1) * dim] = raw[pos]
    return cache


def _gqa_int8(query: torch.Tensor, key_i8: torch.Tensor, value_i8: torch.Tensor) -> torch.Tensor:
    """int8 promoted with scale 1, unmasked GQA.

    query is [1, q_heads, D] (decode) or [1, q_heads, s1, D] (prefill).
    Returns the same layout as query without the batch dim: [q_heads, D] or
    [q_heads, s1, D].
    """
    head_size = key_i8.shape[-1]
    kv_heads = key_i8.shape[1]
    scale = head_size ** -0.5
    k = key_i8.float()
    v = value_i8.float()
    if query.dim() == 3:
        q_heads = query.shape[1]
        group = q_heads // kv_heads
        out = torch.empty(q_heads, head_size)
        q = query[0].float()
        for qh in range(q_heads):
            kvh = qh // group
            scores = k[:, kvh, :] @ q[qh] * scale
            out[qh] = torch.softmax(scores, dim=0) @ v[:, kvh, :]
        return out

    q_heads = query.shape[1]
    s1 = query.shape[2]
    group = q_heads // kv_heads
    out = torch.empty(q_heads, s1, head_size)
    q = query[0].float()
    for qh in range(q_heads):
        kvh = qh // group
        scores = q[qh] @ k[:, kvh, :].T * scale
        out[qh] = torch.softmax(scores, dim=-1) @ v[:, kvh, :]
    return out


def _require_npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("torch.npu is not available")


@pytest.mark.npu
@pytest.mark.golden
@pytest.mark.parametrize(
    ("name", "kv_len", "kv_heads", "q_heads", "s1", "head_size", "flash_decode"),
    ACCURACY_CASES,
    ids=[row[0] for row in ACCURACY_CASES],
)
def test_a5_antiquant_port_accuracy(
    openjiuwen_ops, name, kv_len, kv_heads, q_heads, s1, head_size, flash_decode
):
    """Kernel vs attention on the int8 values. fd must match the merged output."""
    _require_npu()
    from ._common import require_op

    require_op(openjiuwen_ops, "rq_uniform_fia_paged_k8v4")
    device = torch.device("npu")
    g = torch.Generator().manual_seed(C.SEED + kv_len + q_heads + s1 + head_size)
    if s1 == 1:
        query = torch.randn(1, q_heads, head_size, generator=g).to(torch.float16)
    else:
        query = torch.randn(1, q_heads, s1, head_size, generator=g).to(torch.float16)
    key_i8 = _int8_codes(kv_len, kv_heads, head_size, C.SEED + 3 + kv_len + head_size)
    value_i8 = _int8_codes(kv_len, kv_heads, head_size, C.SEED + 9 + kv_len + head_size)
    key_cache = _paged_int8(key_i8).to(device)
    value_cache = _paged_int8(value_i8).to(device)
    bt, _ = C.build_block_table(kv_len)
    out = openjiuwen_ops.rq_uniform_fia_paged_k8v4(
        query.to(device),
        key_cache,
        value_cache,
        bt.to(device),
        [s1],
        [kv_len],
        None,
        None,
        None,
        q_heads,
        kv_heads,
        head_size,
        C.BLOCK_SIZE,
        float(head_size ** -0.5),
        2147483647,
        2147483647,
        0,
        None,
        QUANT_MODE_A5_ANTIQUANT_PORT,
    )
    torch.npu.synchronize()
    ref = _gqa_int8(query, key_i8, value_i8)
    cos = C.cosine(out[0].cpu(), ref)
    logger.info(
        "a5 antiquant port %s kv=%d kv_h=%d q_h=%d s1=%d D=%d cos=%.4f flash_decode=%s",
        name, kv_len, kv_heads, q_heads, s1, head_size, cos, flash_decode,
    )
    assert cos > 0.99, f"{name} cosine {cos} flash_decode={flash_decode}"


def _bench_us(fn) -> float:
    for _ in range(_WARMUP):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(_ITERS):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / _ITERS


def _fill_paged_fp16(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Unquant BNBD [blocks, kv_heads, block_size, D] from the promoted int8."""
    tokens, heads, dim = values.shape
    nblocks = (tokens + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    cache = torch.zeros(nblocks, heads, C.BLOCK_SIZE, dim, dtype=torch.float16, device=device)
    fp = values.to(torch.float16)
    for pos in range(tokens):
        blk = pos // C.BLOCK_SIZE
        row = pos % C.BLOCK_SIZE
        cache[blk, :, row, :] = fp[pos].to(device)
    return cache


@pytest.mark.npu
@pytest.mark.perf
def test_a5_antiquant_port_perf(openjiuwen_ops):
    """Long decode only. Log this op and unquant FIA. Short shape is not the bar."""
    _require_npu()
    import torch_npu  # noqa: F401

    from ._common import require_op

    require_op(openjiuwen_ops, "rq_uniform_fia_paged_k8v4")
    kv_len, kv_heads, q_heads, head_size = 2048, 1, 1, 128
    device = torch.device("npu")
    g = torch.Generator().manual_seed(C.SEED + 1000 + kv_len)
    query = torch.randn(1, q_heads, head_size, generator=g).to(dtype=torch.float16, device=device)
    key_i8 = _int8_codes(kv_len, kv_heads, head_size, C.SEED + 11)
    value_i8 = _int8_codes(kv_len, kv_heads, head_size, C.SEED + 13)
    key_cache = _paged_int8(key_i8).to(device)
    value_cache = _paged_int8(value_i8).to(device)
    bt, _ = C.build_block_table(kv_len)
    bt = bt.to(device)
    scale = float(head_size ** -0.5)
    q_bnsd = query.reshape(1, q_heads, 1, head_size)
    k_fp16 = _fill_paged_fp16(key_i8, device)
    v_fp16 = _fill_paged_fp16(value_i8, device)

    def run_port():
        openjiuwen_ops.rq_uniform_fia_paged_k8v4(
            query, key_cache, value_cache, bt, [1], [kv_len], None, None, None,
            q_heads, kv_heads, head_size, C.BLOCK_SIZE, scale,
            2147483647, 2147483647, 0, None, QUANT_MODE_A5_ANTIQUANT_PORT,
        )

    def run_unquant():
        torch_npu.npu_fused_infer_attention_score(
            q_bnsd, k_fp16, v_fp16,
            actual_seq_lengths=[1],
            actual_seq_lengths_kv=[kv_len],
            num_heads=q_heads,
            num_key_value_heads=kv_heads,
            input_layout="BNSD",
            scale=scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            block_table=bt,
            block_size=C.BLOCK_SIZE,
        )

    port_us = _bench_us(run_port)
    base_us = _bench_us(run_unquant)
    logger.info(
        "[a5 antiquant port] fd kv=%d kv_h=%d q_h=%d D=%d port_us=%.3f unquant_us=%.3f warmup=%d repeat=%d",
        kv_len, kv_heads, q_heads, head_size, port_us, base_us, _WARMUP, _ITERS,
    )
    assert port_us > 0.0 and base_us > 0.0
