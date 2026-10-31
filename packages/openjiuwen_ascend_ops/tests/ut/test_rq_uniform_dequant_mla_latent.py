"""UT for rq_uniform_dequant_mla_latent: contract + golden roundtrip + perf."""

from __future__ import annotations

import logging
import pytest
import torch

from . import _common as C
from ._common import require_op

logger = logging.getLogger(__name__)

OP = "rq_uniform_dequant_mla_latent"
PACK = "rq_uniform_pack_mla_latent"


def test_dequant_mla_latent_scheme_a_contract():
    torch.manual_seed(1)
    latent, t = 64, 20
    rot = C.haar(latent).to(torch.float16)
    c = torch.randn(t, latent, dtype=torch.float16)
    y = (c.float() @ rot.T.float()).half()
    codes, base, step = C.scheme_a_encode(y.float())
    y_back = base.float().unsqueeze(-1) + codes.float() * step.float().unsqueeze(-1)
    c_back = (y_back.half().float() @ rot.float()).half()
    cos = torch.nn.functional.cosine_similarity(c_back.flatten(), c.flatten(), dim=0)
    assert cos.item() > 0.98


def test_dequant_mla_latent_registered(openjiuwen_ops):
    require_op(openjiuwen_ops, OP)


@pytest.mark.npu
@pytest.mark.golden
def test_dequant_mla_latent_golden_roundtrip(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)
    device, dtype = npu_device, torch.float16
    latent, t, block = C.MLA_LATENT, 32, 32
    torch.manual_seed(C.SEED + 9)
    rot = C.haar(latent).to(device=device, dtype=dtype)
    rot_t = rot.T.contiguous()
    c = torch.randn(t, latent, dtype=dtype, device=device)
    stride = C.mla_block_stride(block, latent)
    cache = torch.zeros(1, stride, dtype=torch.uint8, device=device)
    slots = torch.arange(t, dtype=torch.int32, device=device)
    getattr(openjiuwen_ops, PACK)(c, rot_t, slots, cache, block)
    bt = torch.zeros(1, 1, dtype=torch.int32, device=device)
    ctx = torch.tensor([t], dtype=torch.int32, device=device)
    starts = torch.zeros(1, dtype=torch.int32, device=device)
    out = getattr(openjiuwen_ops, OP)(cache, rot, bt, ctx, starts, block, t)
    torch.npu.synchronize()
    C.assert_finite(out, OP)
    assert out.shape == (t, latent)
    cos = C.cosine(out.cpu(), c.cpu())
    assert cos > 0.95, f"roundtrip cosine {cos}"


@pytest.mark.npu
@pytest.mark.perf
def test_dequant_mla_latent_perf(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)
    device, dtype = npu_device, torch.float16
    latent, t, block = C.MLA_LATENT, 256, 128
    rot = C.haar(latent).to(device=device, dtype=dtype)
    rot_t = rot.T.contiguous()
    c = torch.randn(t, latent, dtype=dtype, device=device)
    nblocks = max(1, (t + block - 1) // block)
    cache = torch.zeros(nblocks, C.mla_block_stride(block, latent), dtype=torch.uint8, device=device)
    slots = torch.arange(t, dtype=torch.int32, device=device)
    getattr(openjiuwen_ops, PACK)(c, rot_t, slots, cache, block)
    bt = torch.arange(nblocks, dtype=torch.int32, device=device).view(1, -1)
    ctx = torch.tensor([t], dtype=torch.int32, device=device)
    starts = torch.zeros(1, dtype=torch.int32, device=device)

    def run():
        getattr(openjiuwen_ops, OP)(cache, rot, bt, ctx, starts, block, t)

    ms = C.timed_ms(run)
    assert ms > 0.0
    logger.info(f"[perf] {OP}: {ms:.3f} ms (T={t})")
