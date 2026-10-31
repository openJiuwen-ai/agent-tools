"""UT for rq_uniform_fia_mla_latent(+_paged): contract + golden + perf."""

from __future__ import annotations

import logging
import math

import pytest
import torch

from . import _common as C
from ._common import require_op

logger = logging.getLogger(__name__)

OP = "rq_uniform_fia_mla_latent"
OP_PAGED = "rq_uniform_fia_mla_latent_paged"
PACK = "rq_uniform_pack_mla_latent"
LATENT = C.MLA_LATENT
ROPE = 64


def test_fia_mla_latent_haar_fold_contract():
    """Absorbed Haar: ql@R^T / c@R^T attention equals unrotated (no quant)."""
    latent_dim, num_heads, num_tokens_q, num_tokens_kv, drope = 32, 2, 1, 4, 4
    rot = C.haar(latent_dim)
    rot_t = rot.T.contiguous()
    c = torch.randn(num_tokens_kv, latent_dim)
    c_rot = c @ rot_t
    ql = torch.randn(num_tokens_q, num_heads, latent_dim)
    q_pe = torch.randn(num_tokens_q, num_heads, drope)
    k_pe = torch.randn(num_tokens_kv, drope)
    ql_r = ql @ rot_t
    sc = 1.0 / math.sqrt(latent_dim + drope)
    scores = torch.einsum("qhl,tl->qht", ql_r, c_rot) + torch.einsum("qhd,td->qht", q_pe, k_pe)
    out = torch.einsum("qht,tl->qhl", torch.softmax(scores * sc, dim=-1), c_rot) @ rot
    scores_fp = torch.einsum("qhl,tl->qht", ql, c) + torch.einsum("qhd,td->qht", q_pe, k_pe)
    out_fp = torch.einsum("qht,tl->qhl", torch.softmax(scores_fp * sc, dim=-1), c)
    assert torch.allclose(out, out_fp, atol=1e-4, rtol=1e-4)


def test_fia_mla_latent_registered(openjiuwen_ops):
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, OP_PAGED)


def _placeholder_weights(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Schema placeholders: w_uk_t [H,P,512], w_uv [H,512,V]. Fuse path ignores them."""
    w_uk_t = torch.zeros(1, LATENT, LATENT, dtype=dtype, device=device)
    w_uv = torch.zeros(1, LATENT, 1, dtype=dtype, device=device)
    return w_uk_t, w_uv


def _paged_k_pe(num_blocks: int, block: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """k_pe_cache layout [num_blocks, block_size, 64]."""
    return torch.randn(num_blocks, block, ROPE, dtype=dtype, device=device)


@pytest.mark.npu
@pytest.mark.golden
def test_fia_mla_latent_paged_golden_shape(openjiuwen_ops, npu_device):
    """Smoke golden: paged fuse returns finite latent context."""
    require_op(openjiuwen_ops, OP_PAGED)
    require_op(openjiuwen_ops, PACK)
    device, dtype = npu_device, torch.float16
    h, tq, tkv, block = 2, 1, 32, 32
    torch.manual_seed(C.SEED + 10)
    rot = C.haar(LATENT).to(device=device, dtype=dtype)
    rot_t = rot.T.contiguous()
    c = torch.randn(tkv, LATENT, dtype=dtype, device=device)
    nblocks = max(1, (tkv + block - 1) // block)
    cache = torch.zeros(nblocks, C.mla_block_stride(block, LATENT), dtype=torch.uint8, device=device)
    slots = torch.arange(tkv, dtype=torch.int32, device=device)
    getattr(openjiuwen_ops, PACK)(c, rot_t, slots, cache, block)

    # Path-B fuse: q_nope is already W_UK-absorbed latent [Tq,H,512].
    q_nope = torch.randn(tq, h, LATENT, dtype=dtype, device=device)
    q_pe = torch.randn(tq, h, ROPE, dtype=dtype, device=device)
    k_pe = _paged_k_pe(nblocks, block, dtype, device)
    w_uk_t, w_uv = _placeholder_weights(dtype, device)
    bt = torch.arange(nblocks, dtype=torch.int32, device=device).view(1, -1)
    scale = 1.0 / math.sqrt(LATENT + ROPE)
    out = getattr(openjiuwen_ops, OP_PAGED)(
        q_nope,
        q_pe,
        cache,
        k_pe,
        bt,
        [tq],
        [tkv],
        w_uk_t,
        w_uv,
        rot,
        block,
        scale,
        tq,
        tkv,
        tkv,  # max_kv > 0 → fuse FIA
        None,
    )
    torch.npu.synchronize()
    C.assert_finite(out, OP_PAGED)
    assert out.numel() > 0


@pytest.mark.npu
@pytest.mark.perf
def test_fia_mla_latent_paged_perf(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP_PAGED)
    require_op(openjiuwen_ops, PACK)
    device, dtype = npu_device, torch.float16
    h, tq, tkv, block = 4, 1, 128, 128
    rot = C.haar(LATENT).to(device=device, dtype=dtype)
    rot_t = rot.T.contiguous()
    c = torch.randn(tkv, LATENT, dtype=dtype, device=device)
    nblocks = max(1, (tkv + block - 1) // block)
    cache = torch.zeros(nblocks, C.mla_block_stride(block, LATENT), dtype=torch.uint8, device=device)
    slots = torch.arange(tkv, dtype=torch.int32, device=device)
    getattr(openjiuwen_ops, PACK)(c, rot_t, slots, cache, block)
    q_nope = torch.randn(tq, h, LATENT, dtype=dtype, device=device)
    q_pe = torch.randn(tq, h, ROPE, dtype=dtype, device=device)
    k_pe = _paged_k_pe(nblocks, block, dtype, device)
    w_uk_t, w_uv = _placeholder_weights(dtype, device)
    bt = torch.arange(nblocks, dtype=torch.int32, device=device).view(1, -1)
    scale = 1.0 / math.sqrt(LATENT + ROPE)

    def run():
        getattr(openjiuwen_ops, OP_PAGED)(
            q_nope, q_pe, cache, k_pe, bt, [tq], [tkv], w_uk_t, w_uv, rot,
            block, scale, tq, tkv, tkv, None,
        )

    ms = C.timed_ms(run, warmup=1, iters=4)
    assert ms > 0.0
    logger.info(f"[perf] {OP_PAGED}: {ms:.3f} ms (Tkv={tkv})")
