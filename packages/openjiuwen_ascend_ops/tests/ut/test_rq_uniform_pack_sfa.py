"""UT for rq_uniform_pack_sfa: contract + golden dequant + perf."""

from __future__ import annotations

import logging
import pytest
import torch

from . import _common as C
from ._common import require_op

logger = logging.getLogger(__name__)

OP = "rq_uniform_pack_sfa"
BLOCK = 128


def _dequant_signed_slot(slot: torch.Tensor, nope: torch.Tensor) -> float:
    """V2 signed int4: recover q=(nibble+8)&0xF then y=offset+q*scale."""
    scale = slot[C.SFA_SCALE_OFF:C.SFA_OFFSET_OFF].view(torch.float16).float()
    offset = slot[C.SFA_OFFSET_OFF:C.SFA_SLOT_BYTES].view(torch.float16).float()
    packed = slot[: C.SFA_CODE_BYTES].to(torch.int16)
    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    even = torch.where(even >= 8, even - 16, even)
    odd = torch.where(odd >= 8, odd - 16, odd)
    even = even + 8
    odd = odd + 8
    q = torch.empty(C.SFA_NOPE_DIM, dtype=torch.float32)
    q[0::2] = even.float()
    q[1::2] = odd.float()
    y = offset + q * scale
    return float((y - nope.float()).abs().max())


def test_pack_sfa_layout_contract():
    assert C.SFA_SLOT_BYTES == 388
    assert C.SFA_CODE_BYTES == 256
    assert C.SFA_SCALE_OFF == 384
    assert C.SFA_OFFSET_OFF == 386


def test_pack_sfa_registered(openjiuwen_ops):
    require_op(openjiuwen_ops, OP)


@pytest.mark.npu
@pytest.mark.golden
def test_pack_sfa_golden_dequant(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    device, dtype = npu_device, torch.float16
    t = 32
    torch.manual_seed(C.SEED + 6)
    nope = torch.randn(t, C.SFA_NOPE_DIM, dtype=dtype, device=device)
    rope = torch.randn(t, C.SFA_ROPE_DIM, dtype=dtype, device=device)
    nblocks = max(1, (t + BLOCK - 1) // BLOCK)
    cache = torch.zeros(nblocks, BLOCK, 1, C.SFA_SLOT_BYTES, dtype=torch.uint8, device=device)
    slots = torch.arange(t, dtype=torch.int64, device=device)
    getattr(openjiuwen_ops, OP)(nope, rope, slots, cache, BLOCK)
    torch.npu.synchronize()
    cache_cpu = cache.cpu()
    nope_cpu = nope.cpu()
    for i in range(t):
        blk, pos = i // BLOCK, i % BLOCK
        err = _dequant_signed_slot(cache_cpu[blk, pos, 0], nope_cpu[i])
        # 4-bit uniform quant error is bounded; catch layout/nibble bugs.
        assert err < 2.0, f"token {i} max dequant err {err}"


@pytest.mark.npu
@pytest.mark.perf
def test_pack_sfa_perf(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    device, dtype = npu_device, torch.float16
    t = 1280
    nope = torch.randn(t, C.SFA_NOPE_DIM, dtype=dtype, device=device)
    rope = torch.randn(t, C.SFA_ROPE_DIM, dtype=dtype, device=device)
    nblocks = max(1, (t + BLOCK - 1) // BLOCK)
    cache = torch.zeros(nblocks, BLOCK, 1, C.SFA_SLOT_BYTES, dtype=torch.uint8, device=device)
    slots = torch.arange(t, dtype=torch.int64, device=device)

    def run():
        getattr(openjiuwen_ops, OP)(nope, rope, slots, cache, BLOCK)

    ms = C.timed_ms(run)
    assert ms > 0.0
    logger.info(f"[perf] {OP}: {ms:.3f} ms (T={t})")
