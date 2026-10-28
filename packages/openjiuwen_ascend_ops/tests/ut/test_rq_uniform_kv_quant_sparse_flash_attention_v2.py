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


@pytest.mark.npu
@pytest.mark.golden
def test_sfa_attn_v2_golden_vs_sdpa(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)
    device, dtype = npu_device, torch.float16
    sk, topk = 64, 32
    torch.manual_seed(C.SEED + 7)
    nope = torch.randn(sk, C.SFA_NOPE_DIM, dtype=dtype, device=device)
    rope = torch.randn(sk, C.SFA_ROPE_DIM, dtype=dtype, device=device)
    nblocks = max(1, (sk + BLOCK - 1) // BLOCK)
    cache = torch.zeros(nblocks, BLOCK, 1, C.SFA_SLOT_BYTES, dtype=torch.uint8, device=device)
    slots = torch.arange(sk, dtype=torch.int64, device=device)
    getattr(openjiuwen_ops, PACK)(nope, rope, slots, cache, BLOCK)
    torch.npu.synchronize()

    query = torch.randn(1, NQ, C.SFA_NOPE_DIM + C.SFA_ROPE_DIM, dtype=dtype, device=device)
    # Source golden uses [1, 1, topk] sparse indices (broadcast over heads).
    sparse = (torch.arange(topk, dtype=torch.int32) % sk).view(1, 1, topk).to(device)
    bt = torch.arange(nblocks, dtype=torch.int32, device=device).view(1, -1)
    act_q = torch.tensor([1], dtype=torch.int32, device=device)
    act_kv = torch.tensor([sk], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(float(C.SFA_NOPE_DIM))
    kv = cache.view(torch.int8)
    # Modes match source e2e: quant=4, sparse_mode=3, attention_mode=2, repo=1, tile=128.
    out = getattr(openjiuwen_ops, OP)(
        query,
        kv,
        kv,
        sparse,
        bt,
        act_q,
        act_kv,
        scale,
        4,
        4,
        1,
        "TND",
        "PA_BSND",
        3,
        2,
        1,
        128,
        C.SFA_ROPE_DIM,
        None,
    )
    torch.npu.synchronize()
    out_t = out[0] if isinstance(out, tuple) else out
    C.assert_finite(out_t, OP)

    npu_rows = cache.cpu().view(-1, C.SFA_SLOT_BYTES)[:sk]
    ref = _cpu_attn(query, npu_rows, sparse, scale, dtype)
    npu_out = out_t.detach().cpu().float().reshape(1, NQ, -1)[..., :C.SFA_NOPE_DIM]
    max_err = (npu_out - ref.float()).abs().max().item()
    assert max_err <= MAX_ERR_LIMIT, f"max_err={max_err}"


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
