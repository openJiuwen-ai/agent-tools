"""UT for rq_uniform_pack_k4v4: contract + golden QDQ + perf."""

from __future__ import annotations

import logging
import pytest
import torch

from . import _common as C
from ._common import require_op

logger = logging.getLogger(__name__)

OP = "rq_uniform_pack_k4v4"


def test_pack_k4v4_layout_contract():
    assert C.K4V4_TILE == 1088
    assert C.K4V4_BYTES_PER_ROW == 68


def test_pack_k4v4_registered(openjiuwen_ops):
    require_op(openjiuwen_ops, OP)


@pytest.mark.npu
@pytest.mark.golden
def test_pack_k4v4_golden_qdq(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    device, dtype = npu_device, torch.float16
    t, h = 37, 4
    torch.manual_seed(C.SEED + 1)
    key = torch.randn(t, h, C.HEAD_SIZE, dtype=dtype, device=device)
    value = torch.randn(t, h, C.HEAD_SIZE, dtype=dtype, device=device)
    eye = torch.eye(C.HEAD_SIZE, dtype=dtype, device=device)
    key_cache, value_cache = C.pack_kv(
        openjiuwen_ops, OP, key, value, eye, key_stride=C.K4V4_TILE, val_stride=C.K4V4_TILE
    )
    bt, _ = C.build_block_table(t)
    for name, cache, src in (("key", key_cache, key), ("value", value_cache, value)):
        src_cpu = src.float().cpu()
        for pos in range(t):
            blk = int(bt[0, pos // C.BLOCK_SIZE])
            off = pos % C.BLOCK_SIZE
            block = cache[blk, 0].cpu()
            y = C.decode_int4_row(
                block, off, block_size=C.BLOCK_SIZE, head_size=C.HEAD_SIZE, dtype=dtype
            )
            row_code = C.HEAD_SIZE // 2
            vstep = C.u8_as_fp(
                block[
                    C.BLOCK_SIZE * (row_code + 2) + off * 2:C.BLOCK_SIZE * (row_code + 2) + off * 2 + 2
                ],
                dtype,
            )
            err = (y - src_cpu[pos, 0]).abs().max().item()
            assert err <= vstep * 0.75 + 1e-3, f"{name} pos={pos} err={err} vstep={vstep}"


@pytest.mark.npu
@pytest.mark.perf
def test_pack_k4v4_perf(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    device, dtype = npu_device, torch.float16
    t, h = 128, 8
    key = torch.randn(t, h, C.HEAD_SIZE, dtype=dtype, device=device)
    value = torch.randn(t, h, C.HEAD_SIZE, dtype=dtype, device=device)
    rot = C.dense_rotation(C.HEAD_SIZE, dtype, device)
    nblocks = (t + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    key_cache = torch.zeros(nblocks, h, C.K4V4_TILE, dtype=torch.uint8, device=device)
    value_cache = torch.zeros(nblocks, h, C.K4V4_TILE, dtype=torch.uint8, device=device)
    slots = torch.arange(t, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, t], dtype=torch.int32, device=device)

    def run():
        getattr(openjiuwen_ops, OP)(
            key, value, slots, qsl, rot, key_cache, value_cache, 1, C.BLOCK_SIZE
        )

    ms = C.timed_ms(run)
    assert ms > 0.0
    logger.info(f"[perf] {OP}: {ms:.3f} ms (T={t}, H={h})")
