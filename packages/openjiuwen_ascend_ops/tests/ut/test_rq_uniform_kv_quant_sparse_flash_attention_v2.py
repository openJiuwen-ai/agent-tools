"""UT for rq_uniform_kv_quant_sparse_flash_attention_v2: contract + golden + perf."""

from __future__ import annotations

import logging
import math

import pytest
import torch

from . import _common as C
from ._common import require_op

logger = logging.getLogger(__name__)

OP = "rq_uniform_kv_quant_sparse_flash_attention_v2"
PACK = "rq_uniform_pack_sfa"
BLOCK = 128
NQ = 8
# Matches source e2e golden tolerances (fp16 online softmax vs fp32 ref).
MAX_ERR_LIMIT = 5e-2


def test_sfa_attn_v2_layout_contract():
    assert C.SFA_SLOT_BYTES == 388
    assert C.SFA_NOPE_DIM + C.SFA_ROPE_DIM == 576


def test_sfa_attn_v2_registered(openjiuwen_ops):
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)


def _signed_dequant_rows(slots_u8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """q=(nibble+8)&0xF; y=offset+q*scale. Returns [T, 576] (nope+rope)."""
    t = slots_u8.shape[0]
    packed = slots_u8[:, :C.SFA_CODE_BYTES].to(torch.int16)
    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    even = torch.where(even >= 8, even - 16, even) + 8
    odd = torch.where(odd >= 8, odd - 16, odd) + 8
    q = torch.empty(t, C.SFA_NOPE_DIM, dtype=torch.float32)
    q[:, 0::2] = even.float()
    q[:, 1::2] = odd.float()
    scale = slots_u8[:, C.SFA_SCALE_OFF:C.SFA_OFFSET_OFF].contiguous().view(torch.float16).reshape(t)
    offset = slots_u8[:, C.SFA_OFFSET_OFF:C.SFA_SLOT_BYTES].contiguous().view(torch.float16).reshape(t)
    nope = offset.float().unsqueeze(-1) + q * scale.float().unsqueeze(-1)
    rope = (
        slots_u8[:, C.SFA_CODE_BYTES:C.SFA_SCALE_OFF]
        .contiguous()
        .view(dtype)
        .reshape(t, C.SFA_ROPE_DIM)
    )
    return torch.cat((nope.to(dtype), rope.to(dtype)), dim=-1)


def _cpu_attn(query: torch.Tensor, kv_slots_u8: torch.Tensor, sparse: torch.Tensor,
              scale: float, dtype: torch.dtype) -> torch.Tensor:
    """Reference: SDPA over gathered slots; output nope dims only (512)."""
    idx = sparse.view(-1).cpu()
    kv_fp = _signed_dequant_rows(kv_slots_u8[idx.long()], dtype)
    q = query.cpu().float()
    scores = torch.einsum("qnd,kd->qnk", q, kv_fp.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    v = kv_fp[:, :C.SFA_NOPE_DIM].float()
    return torch.einsum("qnk,kd->qnd", probs, v).to(dtype)


def _run_sfa_attn_v2(
    ops,
    *,
    device,
    dtype,
    sk: int,
    topk: int,
    seed: int,
    sq: int = 1,
):
    """Pack Schema-A slots then run V2 attn; returns (out, cache_u8, sparse, scale, query).

    Torch adapter is TND-shaped (rank-3). Multi-token ``sq>1`` stresses tiling Process /
    CheckSinglePara and the rewritten v2 kernel Init/Process path.
    """
    torch.manual_seed(seed)
    nope = torch.randn(sk, C.SFA_NOPE_DIM, dtype=dtype, device=device)
    rope = torch.randn(sk, C.SFA_ROPE_DIM, dtype=dtype, device=device)
    nblocks = max(1, (sk + BLOCK - 1) // BLOCK)
    cache = torch.zeros(nblocks, BLOCK, 1, C.SFA_SLOT_BYTES, dtype=torch.uint8, device=device)
    slots = torch.arange(sk, dtype=torch.int64, device=device)
    getattr(ops, PACK)(nope, rope, slots, cache, BLOCK)
    torch.npu.synchronize()

    q_dim = C.SFA_NOPE_DIM + C.SFA_ROPE_DIM
    # TND [T, N, D]; sparse_indices must be [T, 1, topk] (tiling CompareShape).
    query = torch.randn(sq, NQ, q_dim, dtype=dtype, device=device)
    sparse = (
        (torch.arange(topk, dtype=torch.int32) % sk)
        .view(1, 1, topk)
        .expand(sq, 1, topk)
        .contiguous()
        .to(device)
    )
    act_q = torch.tensor([sq], dtype=torch.int32, device=device)

    bt = torch.arange(nblocks, dtype=torch.int32, device=device).view(1, -1)
    act_kv = torch.tensor([sk], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(float(C.SFA_NOPE_DIM))
    kv = cache.view(torch.int8)
    # Modes match source e2e: quant=4, sparse_mode=3, attention_mode=2, repo=1, tile=128.
    out = getattr(ops, OP)(
        query, kv, kv, sparse, bt, act_q, act_kv, scale,
        4, 4, 1, "TND", "PA_BSND", 3, 2, 1, 128, C.SFA_ROPE_DIM, None,
    )
    torch.npu.synchronize()
    out_t = out[0] if isinstance(out, tuple) else out
    return out_t, cache, sparse, scale, query


@pytest.mark.npu
@pytest.mark.golden
@pytest.mark.parametrize("sk,topk", [(64, 32), (128, 64)], ids=["s64_k32", "s128_k64"])
def test_sfa_attn_v2_golden_vs_sdpa(openjiuwen_ops, npu_device, sk, topk):
    """Covers rewritten host tiling + kernel entry (infershape / tiling / v2.cpp)."""
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)
    device, dtype = npu_device, torch.float16
    out_t, cache, sparse, scale, query = _run_sfa_attn_v2(
        openjiuwen_ops, device=device, dtype=dtype, sk=sk, topk=topk, seed=C.SEED + 7 + sk
    )
    C.assert_finite(out_t, OP)
    assert out_t.shape[-1] == C.SFA_NOPE_DIM or out_t.reshape(1, NQ, -1).shape[-1] >= C.SFA_NOPE_DIM

    npu_rows = cache.cpu().view(-1, C.SFA_SLOT_BYTES)[:sk]
    ref = _cpu_attn(query, npu_rows, sparse, scale, dtype)
    npu_out = out_t.detach().cpu().float().reshape(1, NQ, -1)[..., :C.SFA_NOPE_DIM]
    max_err = (npu_out - ref.float()).abs().max().item()
    assert max_err <= MAX_ERR_LIMIT, f"max_err={max_err} sk={sk} topk={topk}"


@pytest.mark.npu
@pytest.mark.golden
def test_sfa_attn_v2_bf16_smoke(openjiuwen_ops, npu_device):
    """bf16 path through rewritten kernel dispatch (else-branch of dtype select)."""
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)
    if not hasattr(torch, "bfloat16"):
        pytest.skip("bfloat16 unavailable")
    device = npu_device
    out_t, _, _, _, _ = _run_sfa_attn_v2(
        openjiuwen_ops, device=device, dtype=torch.bfloat16, sk=64, topk=32, seed=C.SEED + 99
    )
    C.assert_finite(out_t, OP)


@pytest.mark.npu
@pytest.mark.golden
@pytest.mark.parametrize(
    "sk,topk,sq",
    [(128, 64, 2), (256, 128, 4)],
    ids=["s128_k64_t2", "s256_k128_t4"],
)
def test_sfa_attn_v2_tiling_kernel_hard(openjiuwen_ops, npu_device, sk, topk, sq):
    """Hard cover for rewritten tiling Process/CheckSinglePara + v2 kernel Init/Process.

    Multi-token TND + larger sparse gather forces host tiling multi-para paths and the
    rewritten kernel entry; asserts rope-shrunk out last-dim (adapter InferShape contract).
    """
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)
    device, dtype = npu_device, torch.float16
    out_t, _, _, _, query = _run_sfa_attn_v2(
        openjiuwen_ops,
        device=device,
        dtype=dtype,
        sk=sk,
        topk=topk,
        seed=C.SEED + 1000 + sk + sq,
        sq=sq,
    )
    C.assert_finite(out_t, OP)
    assert out_t.shape[:-1] == query.shape[:-1]
    assert out_t.shape[-1] == C.SFA_NOPE_DIM, f"expected nope={C.SFA_NOPE_DIM}, got {out_t.shape[-1]}"
    assert out_t.shape[0] == sq


@pytest.mark.npu
@pytest.mark.perf
def test_sfa_attn_v2_perf(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)
    device, dtype = npu_device, torch.float16
    sk, topk = 512, 128
    nope = torch.randn(sk, C.SFA_NOPE_DIM, dtype=dtype, device=device)
    rope = torch.randn(sk, C.SFA_ROPE_DIM, dtype=dtype, device=device)
    nblocks = max(1, (sk + BLOCK - 1) // BLOCK)
    cache = torch.zeros(nblocks, BLOCK, 1, C.SFA_SLOT_BYTES, dtype=torch.uint8, device=device)
    slots = torch.arange(sk, dtype=torch.int64, device=device)
    getattr(openjiuwen_ops, PACK)(nope, rope, slots, cache, BLOCK)
    query = torch.randn(1, NQ, C.SFA_NOPE_DIM + C.SFA_ROPE_DIM, dtype=dtype, device=device)
    sparse = (torch.arange(topk, dtype=torch.int32) % sk).view(1, 1, topk).to(device)
    bt = torch.arange(nblocks, dtype=torch.int32, device=device).view(1, -1)
    act_q = torch.tensor([1], dtype=torch.int32, device=device)
    act_kv = torch.tensor([sk], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(float(C.SFA_NOPE_DIM))
    kv = cache.view(torch.int8)

    def run():
        getattr(openjiuwen_ops, OP)(
            query, kv, kv, sparse, bt, act_q, act_kv, scale,
            4, 4, 1, "TND", "PA_BSND", 3, 2, 1, 128, C.SFA_ROPE_DIM, None,
        )

    ms = C.timed_ms(run)
    assert ms > 0.0
    logger.info(f"[perf] {OP}: {ms:.3f} ms (S={sk}, topk={topk})")
