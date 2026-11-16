"""A5 mode1 FIA: accuracy cases (including FlashDecode) and the unquant baseline.

Accuracy does not call the A5 kernel yet. It locks the CPU reference the kernel
must match, on shapes that do and do not cross cores.

Performance baseline is stock A5 ``npu_fused_infer_attention_score`` with
unquantized fp16 paged KV. Same shapes, no pack, no quant. The quant FIA row
is skipped until ``op_kernel/arch35`` lands; it must be timed with this same
function so the two numbers share one ruler.

S2 tile is 1024 (``BR_S2_BASICSIZE_IS_1024`` in the current FIA tiling).
FlashDecode — the cross-core reduce of partial O and LSE — is recorded only
when one (batch, kv head) row is split into more than one S2 part. kv length
128/256/512 stays inside one part, so those shapes do not cover the reduce.
"""

from __future__ import annotations

import logging

import pytest
import torch

from . import _common as C

logger = logging.getLogger(__name__)

# Flip when arch35 FIA can launch. Until then the NPU golden and the quant
# perf row skip, and only the CPU reference plus the unquant baseline run.
A5_FIA_KERNEL_READY = False

# (name, kv_len, kv_heads, q_heads, crosses_cores)
# crosses_cores: kv_len > 1024 and few kv heads, so SplitCore cuts one row's
# S2 across AICs and FlashDecode merges the partials.
ACCURACY_CASES = [
    ("single_tile", 128, 2, 4, False),
    ("fd_two_parts", 2048, 1, 1, True),
    ("fd_gqa", 4096, 2, 8, True),
]

# Perf acceptance is the long row. Short rows are recorded, not the bar.
PERF_CASES = [
    ("short", 128, 2, 4),
    ("fd_gqa", 4096, 2, 8),
]

_WARMUP = 10
_ITERS = 100


def _pack_mode1_cpu(key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter mode1 SoA on CPU. key/value are [tokens, heads, D] fp16/bf16."""
    tokens, heads, head_size = key.shape
    block = C.BLOCK_SIZE
    nblocks = (tokens + block - 1) // block
    key_cache = torch.zeros(nblocks, heads, block * head_size, dtype=torch.uint8)
    code_bytes = head_size // 2
    value_cache = torch.zeros(nblocks, heads, block * (code_bytes + 4), dtype=torch.uint8)
    for head in range(heads):
        k_codes = C.encode_k8v4_key_fp8_cpu(key[:, head, :].float())
        packed, vmin, vstep = C.encode_value_u4_cpu(value[:, head, :].float())
        vmin_u8 = vmin.contiguous().view(torch.uint8).reshape(tokens, 2)
        vstep_u8 = vstep.contiguous().view(torch.uint8).reshape(tokens, 2)
        for pos in range(tokens):
            blk = pos // block
            row = pos % block
            key_cache[blk, head, row * head_size:(row + 1) * head_size] = k_codes[pos]
            value_cache[blk, head, row * code_bytes:(row + 1) * code_bytes] = packed[pos]
            vmin_at = block * code_bytes + row * 2
            vstep_at = block * (code_bytes + 2) + row * 2
            value_cache[blk, head, vmin_at:vmin_at + 2] = vmin_u8[pos]
            value_cache[blk, head, vstep_at:vstep_at + 2] = vstep_u8[pos]
    return key_cache, value_cache


def _decode_mode1_cpu(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    tokens: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the packed cache back to [tokens, heads, D] float."""
    heads = key_cache.shape[1]
    head_size = C.HEAD_SIZE
    keys = torch.empty(tokens, heads, head_size)
    values = torch.empty(tokens, heads, head_size)
    for head in range(heads):
        for pos in range(tokens):
            blk = pos // C.BLOCK_SIZE
            row = pos % C.BLOCK_SIZE
            keys[pos, head] = C.decode_k8v4_key_row_fp8(
                key_cache[blk, head], row, head_size=head_size,
            )
            values[pos, head] = C.decode_int4_row(
                value_cache[blk, head],
                row,
                block_size=C.BLOCK_SIZE,
                head_size=head_size,
                dtype=dtype,
            )
    return keys, values


def _gqa_attn(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Decode attention, no rotation. query [1, q_heads, D], kv [tokens, kv_heads, D]."""
    q_heads = query.shape[1]
    kv_heads = keys.shape[1]
    group = q_heads // kv_heads
    scale = C.HEAD_SIZE ** -0.5
    out = torch.empty(q_heads, C.HEAD_SIZE)
    q = query[0].float()
    k = keys.float()
    v = values.float()
    for qh in range(q_heads):
        kvh = qh // group
        scores = k[:, kvh, :] @ q[qh] * scale
        out[qh] = torch.softmax(scores, dim=0) @ v[:, kvh, :]
    return out


def _make_qkv(kv_len: int, kv_heads: int, q_heads: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    key = torch.randn(kv_len, kv_heads, C.HEAD_SIZE, generator=g)
    value = torch.randn(kv_len, kv_heads, C.HEAD_SIZE, generator=g)
    query = torch.randn(1, q_heads, C.HEAD_SIZE, generator=g)
    return query, key, value


@pytest.mark.parametrize(
    ("name", "kv_len", "kv_heads", "q_heads", "crosses_cores"),
    ACCURACY_CASES,
    ids=[row[0] for row in ACCURACY_CASES],
)
def test_a5_fia_cpu_reference(name, kv_len, kv_heads, q_heads, crosses_cores):
    """Layout roundtrip, plus the quant-error budget against raw fp16 attention.

    ``crosses_cores`` is the shape tag the NPU test will use. This function
    itself is one full-sequence attention; the reduce is inside the kernel.
    """
    del name, crosses_cores
    query, key, value = _make_qkv(kv_len, kv_heads, q_heads, C.SEED + kv_len + q_heads)
    key_cache, value_cache = _pack_mode1_cpu(key, value)
    dec_k, dec_v = _decode_mode1_cpu(key_cache, value_cache, kv_len, torch.float16)

    fp8_k = key.float().to(torch.float8_e4m3fn).float()
    assert torch.allclose(dec_k, fp8_k, rtol=0, atol=0), "mode1 key cache layout"

    attn_fp16 = _gqa_attn(query, key, value)
    attn_dequant = _gqa_attn(query, dec_k, dec_v)
    quant_cos = C.cosine(attn_dequant, attn_fp16)
    logger.info(
        "a5 fia cpu kv=%d kv_h=%d q_h=%d quant_cos=%.4f fd_shape=%s",
        kv_len, kv_heads, q_heads, quant_cos, kv_len > 1024,
    )
    # Budget for fp8+u4 on gaussian inputs, not the kernel bar.
    # The kernel must match attn_dequant, not this number.
    assert quant_cos > 0.90, f"quant error cosine {quant_cos} kv={kv_len}"


@pytest.mark.npu
@pytest.mark.golden
@pytest.mark.parametrize(
    ("name", "kv_len", "kv_heads", "q_heads", "crosses_cores"),
    ACCURACY_CASES,
    ids=[row[0] for row in ACCURACY_CASES],
)
def test_a5_fia_kernel_vs_dequant(openjiuwen_ops, npu_device, name, kv_len, kv_heads, q_heads, crosses_cores):
    """Kernel output vs attention on the bytes it reads. FD shapes included.

    single_tile: one S2 part, no reduce.
    fd_two_parts / fd_gqa: kv_len > 1024, partial O/LSE must be merged.
    """
    if not A5_FIA_KERNEL_READY:
        pytest.skip("A5 arch35 FIA is not landed; CPU reference is test_a5_fia_cpu_reference")

    from ._common import require_op

    require_op(openjiuwen_ops, "rq_uniform_fia_paged_k8v4")
    require_op(openjiuwen_ops, "rq_uniform_pack_k8v4")
    query, key, value = _make_qkv(kv_len, kv_heads, q_heads, C.SEED + kv_len + q_heads)
    query = query.to(device=npu_device, dtype=torch.float16)
    key = key.to(device=npu_device, dtype=torch.float16)
    value = value.to(device=npu_device, dtype=torch.float16)
    key_cache, value_cache = C.pack_kv(
        openjiuwen_ops,
        "rq_uniform_pack_k8v4",
        key,
        value,
        None,
        key_stride=C.key_tile_bytes(C.HEAD_SIZE, C.QUANT_MODE_K8_FP8),
        val_stride=C.val_tile_bytes(C.HEAD_SIZE),
        quant_mode=C.QUANT_MODE_K8_FP8,
    )
    bt, _ = C.build_block_table(kv_len)
    # Rotation tensors are unused on mode1. Pass empty placeholders only if the
    # current A2 adapter still requires them; the A5 path must not apply them.
    out = openjiuwen_ops.rq_uniform_fia_paged_k8v4(
        query,
        key_cache,
        value_cache,
        bt.to(npu_device),
        [1],
        [kv_len],
        None,
        key.new_empty(0),
        key.new_empty(0),
        q_heads,
        kv_heads,
        C.HEAD_SIZE,
        C.BLOCK_SIZE,
        float(C.HEAD_SIZE ** -0.5),
        2147483647,
        2147483647,
        0,
        None,
    )
    torch.npu.synchronize()
    dec_k, dec_v = _decode_mode1_cpu(key_cache.cpu(), value_cache.cpu(), kv_len, torch.float16)
    ref = _gqa_attn(query.cpu().float(), dec_k, dec_v)
    cos = C.cosine(out[0].cpu(), ref)
    assert cos > 0.99, f"{name} cosine {cos} fd={crosses_cores}"


def _bench_us(fn) -> float:
    """Host launch + kernel, one sync after the loop. Same ruler for both rows."""
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


def _fill_paged_fp16(key: torch.Tensor, value: torch.Tensor, device: torch.device):
    """Unquantized BNBD cache: [blocks, kv_heads, block_size, D]."""
    tokens, kv_heads, head_size = key.shape
    nblocks = (tokens + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    k_cache = torch.zeros(nblocks, kv_heads, C.BLOCK_SIZE, head_size, dtype=torch.float16, device=device)
    v_cache = torch.zeros_like(k_cache)
    for pos in range(tokens):
        blk = pos // C.BLOCK_SIZE
        row = pos % C.BLOCK_SIZE
        k_cache[blk, :, row, :] = key[pos]
        v_cache[blk, :, row, :] = value[pos]
    bt = torch.arange(nblocks, dtype=torch.int32, device=device).reshape(1, -1)
    return k_cache, v_cache, bt


@pytest.mark.npu
@pytest.mark.perf
@pytest.mark.parametrize(
    ("name", "kv_len", "kv_heads", "q_heads"),
    PERF_CASES,
    ids=[row[0] for row in PERF_CASES],
)
def test_a5_fia_unquant_baseline(name, kv_len, kv_heads, q_heads):
    """Stock FIA, no quant. ``fd_gqa`` is the number the quant kernel must match.

    Query is BNSD [1, q_heads, 1, D]. Cache is BNBD. This is the paged path in
    ops-transformer ``gqa_no_quant_bnsd_bsnd.py``. The printed microseconds
    include host launch; cube/aiv time still comes from msprof. Compare the
    quant kernel only against this function, not against pytest wall time.
    """
    pytest.importorskip("torch_npu")
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.skip("torch.npu is not available")
    device = torch.device("npu")
    query, key, value = _make_qkv(kv_len, kv_heads, q_heads, C.SEED + 1000 + kv_len)
    query = query.to(device=device, dtype=torch.float16).reshape(1, q_heads, 1, C.HEAD_SIZE)
    key = key.to(device=device, dtype=torch.float16)
    value = value.to(device=device, dtype=torch.float16)
    k_cache, v_cache, bt = _fill_paged_fp16(key, value, device)
    scale = float(C.HEAD_SIZE ** -0.5)

    def run():
        torch_npu.npu_fused_infer_attention_score(
            query,
            k_cache,
            v_cache,
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

    us = _bench_us(run)
    logger.info(
        "[baseline unquant] %s kv=%d kv_h=%d q_h=%d avg_us=%.3f warmup=%d repeat=%d",
        name, kv_len, kv_heads, q_heads, us, _WARMUP, _ITERS,
    )
    assert us > 0.0


@pytest.mark.npu
@pytest.mark.perf
@pytest.mark.parametrize(
    ("name", "kv_len", "kv_heads", "q_heads"),
    PERF_CASES,
    ids=[row[0] for row in PERF_CASES],
)
def test_a5_fia_quant_perf(openjiuwen_ops, npu_device, name, kv_len, kv_heads, q_heads):
    """Same ``_bench_us`` as the unquant row. Acceptance is ``fd_gqa``, not ``short``."""
    del openjiuwen_ops, npu_device, name, kv_len, kv_heads, q_heads
    if not A5_FIA_KERNEL_READY:
        pytest.skip("A5 arch35 FIA is not landed; keep the unquant log as the bar")
