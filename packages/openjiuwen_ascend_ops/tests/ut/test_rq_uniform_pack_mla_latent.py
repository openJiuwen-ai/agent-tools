"""UT for rq_uniform_pack_mla_latent: contract + golden layout + perf."""

from __future__ import annotations

import logging
import pytest
import torch

from . import _common as C
from ._common import require_op

logger = logging.getLogger(__name__)

OP = "rq_uniform_pack_mla_latent"
LATENT = 64  # smaller than prod 512 keeps contract CPU-light; NPU uses 512-capable path
BLOCK = 32


def test_pack_mla_latent_scheme_a_contract():
    """CPU Scheme A + SoA offsets (independent of AscendC)."""
    torch.manual_seed(0)
    latent_dim, num_tokens, block_size = LATENT, 32, BLOCK
    rot_t = C.haar(latent_dim).T.contiguous().to(torch.float16)
    c = torch.randn(num_tokens, latent_dim, dtype=torch.float16)
    rows = (c.float() @ rot_t.float()).half().float()
    codes = torch.empty(num_tokens, latent_dim, dtype=torch.uint8)
    bases = torch.empty(num_tokens, dtype=torch.float16)
    steps = torch.empty(num_tokens, dtype=torch.float16)
    for lo in (0, 16):
        part = rows[lo:lo + 16]
        code, base, step = C.scheme_a_encode(part)
        codes[lo:lo + 16] = code.view(torch.uint8)
        bases[lo:lo + 16] = base
        steps[lo:lo + 16] = step
    stride = C.mla_block_stride(block_size, latent_dim)
    cache = torch.zeros(1, stride, dtype=torch.uint8)
    for pos in range(num_tokens):
        cache[0, pos * latent_dim:(pos + 1) * latent_dim] = codes[pos]
        cache[0, block_size * latent_dim + pos * 2:block_size * latent_dim + pos * 2 + 2] = (
            bases[pos].reshape(1).view(torch.uint8)
        )
        cache[0, block_size * (latent_dim + 2) + pos * 2:block_size * (latent_dim + 2) + pos * 2 + 2] = (
            steps[pos].reshape(1).view(torch.uint8)
        )
    assert int(cache[0, 16 * latent_dim]) == int(codes[16, 0])
    recon = bases.float().unsqueeze(-1) + codes.view(torch.int8).float() * steps.float().unsqueeze(-1)
    rel = ((recon - rows) ** 2).mean() / rows.pow(2).mean().clamp_min(1e-8)
    assert rel.item() < 5e-4


def test_pack_mla_latent_registered(openjiuwen_ops):
    require_op(openjiuwen_ops, OP)


@pytest.mark.npu
@pytest.mark.golden
def test_pack_mla_latent_golden_npu(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    device, dtype = npu_device, torch.float16
    latent, t, block = C.MLA_LATENT, 32, 32
    torch.manual_seed(C.SEED + 8)
    rot = C.haar(latent).to(device=device, dtype=dtype)
    rot_t = rot.T.contiguous()
    c = torch.randn(t, latent, dtype=dtype, device=device)
    stride = C.mla_block_stride(block, latent)
    cache = torch.zeros(1, stride, dtype=torch.uint8, device=device)
    slots = torch.arange(t, dtype=torch.int32, device=device)
    getattr(openjiuwen_ops, OP)(c, rot_t, slots, cache, block)
    torch.npu.synchronize()
    # Decode row 0 and check reconstruction vs rotated token.
    block_cpu = cache[0].cpu()
    code = block_cpu[0:latent].view(torch.int8).float()
    base = C.u8_as_fp(block_cpu[block * latent:block * latent + 2], dtype)
    step = C.u8_as_fp(block_cpu[block * (latent + 2):block * (latent + 2) + 2], dtype)
    y = base + code * step
    y_ref = (c[0].float().cpu() @ rot_t.float().cpu()).half().float()
    rel = ((y - y_ref) ** 2).mean() / y_ref.pow(2).mean().clamp_min(1e-8)
    assert rel.item() < 5e-3


@pytest.mark.npu
@pytest.mark.perf
def test_pack_mla_latent_perf(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    device, dtype = npu_device, torch.float16
    latent, t, block = C.MLA_LATENT, 256, 128
    rot_t = C.haar(latent).T.contiguous().to(device=device, dtype=dtype)
    c = torch.randn(t, latent, dtype=dtype, device=device)
    nblocks = max(1, (t + block - 1) // block)
    cache = torch.zeros(nblocks, C.mla_block_stride(block, latent), dtype=torch.uint8, device=device)
    slots = torch.arange(t, dtype=torch.int32, device=device)

    def run():
        getattr(openjiuwen_ops, OP)(c, rot_t, slots, cache, block)

    ms = C.timed_ms(run)
    assert ms > 0.0
    logger.info(f"[perf] {OP}: {ms:.3f} ms (T={t})")
