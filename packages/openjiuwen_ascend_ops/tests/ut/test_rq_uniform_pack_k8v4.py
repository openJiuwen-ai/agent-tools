"""UT for rq_uniform_pack_k8v4 mode0 (A2/A3 UNIFORM Haar+SchemeA).

A5 TQ_K8_FP8 (quant_mode=1) lives in test_rq_uniform_pack_k8v4_a5.py —
run that file alone on ascend950 to avoid MIX-path interference.
"""

from __future__ import annotations

import logging

import pytest
import torch

from . import _common as C
from ._common import require_op

logger = logging.getLogger(__name__)

OP = "rq_uniform_pack_k8v4"


def test_pack_k8v4_layout_contract():
    assert C.K8V4_KEY_TILE == 2112
    assert C.K8V4_VAL_TILE == 1088
    assert C.K8V4_KEY_BYTES_PER_ROW == 132
    assert C.key_tile_bytes(C.HEAD_SIZE, C.QUANT_MODE_UNIFORM) == 2112
    assert C.val_tile_bytes(C.HEAD_SIZE) == 1088
    assert C.BLOCK_SIZE % 16 == 0


def test_pack_k8v4_value_u4_cpu_golden():
    torch.manual_seed(C.SEED + 1)
    rows = torch.randn(4, C.HEAD_SIZE, dtype=torch.float32)
    packed, vmin, vstep = C.encode_value_u4_cpu(rows)
    assert packed.shape == (4, C.HEAD_SIZE // 2)
    block = torch.zeros(C.BLOCK_SIZE * C.K8V4_VAL_BYTES_PER_ROW, dtype=torch.uint8)
    for pos in range(4):
        block[pos * (C.HEAD_SIZE // 2):(pos + 1) * (C.HEAD_SIZE // 2)] = packed[pos]
        # 0-dim half cannot .view(uint8); keep a 1-element tensor.
        block[
            C.BLOCK_SIZE * (C.HEAD_SIZE // 2) + pos * 2:
            C.BLOCK_SIZE * (C.HEAD_SIZE // 2) + pos * 2 + 2
        ] = vmin[pos:pos + 1].contiguous().view(torch.uint8)
        block[
            C.BLOCK_SIZE * (C.HEAD_SIZE // 2 + 2) + pos * 2:
            C.BLOCK_SIZE * (C.HEAD_SIZE // 2 + 2) + pos * 2 + 2
        ] = vstep[pos:pos + 1].contiguous().view(torch.uint8)
    y = C.decode_int4_row(
        block, 0, block_size=C.BLOCK_SIZE, head_size=C.HEAD_SIZE, dtype=torch.float16
    )
    assert (y - rows[0]).abs().mean().item() < 0.35


def test_pack_k8v4_registered(openjiuwen_ops):
    require_op(openjiuwen_ops, OP)


@pytest.mark.npu
@pytest.mark.golden
def test_pack_k8v4_golden_qdq_mode0(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    device = npu_device
    dtype = torch.float16
    t, h = 32, 2
    torch.manual_seed(C.SEED)
    key = torch.randn(t, h, C.HEAD_SIZE, dtype=dtype, device=device)
    value = torch.randn(t, h, C.HEAD_SIZE, dtype=dtype, device=device)
    eye = torch.eye(C.HEAD_SIZE, dtype=dtype, device=device)
    key_cache, value_cache = C.pack_kv(
        openjiuwen_ops,
        OP,
        key,
        value,
        eye,
        key_stride=C.K8V4_KEY_TILE,
        val_stride=C.K8V4_VAL_TILE,
        quant_mode=C.QUANT_MODE_UNIFORM,
    )
    bt, _ = C.build_block_table(t)
    key_cpu, val_cpu = key.float().cpu(), value.float().cpu()
    for pos in range(t):
        blk = int(bt[0, pos // C.BLOCK_SIZE])
        off = pos % C.BLOCK_SIZE
        yk = C.decode_k8v4_key_row(
            key_cache[blk, 0].cpu(), off, block_size=C.BLOCK_SIZE, head_size=C.HEAD_SIZE, dtype=dtype
        )
        yv = C.decode_int4_row(
            value_cache[blk, 0].cpu(), off, block_size=C.BLOCK_SIZE, head_size=C.HEAD_SIZE, dtype=dtype
        )
        assert (yk - key_cpu[pos, 0]).abs().mean().item() < 0.15
        assert (yv - val_cpu[pos, 0]).abs().mean().item() < 0.25


@pytest.mark.npu
@pytest.mark.perf
def test_pack_k8v4_perf_mode0(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    device = npu_device
    dtype = torch.float16
    t, h = 128, 8
    key = torch.randn(t, h, C.HEAD_SIZE, dtype=dtype, device=device)
    value = torch.randn(t, h, C.HEAD_SIZE, dtype=dtype, device=device)
    rot = C.dense_rotation(C.HEAD_SIZE, dtype, device)
    nblocks = (t + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    key_cache = torch.zeros(nblocks, h, C.K8V4_KEY_TILE, dtype=torch.uint8, device=device)
    value_cache = torch.zeros(nblocks, h, C.K8V4_VAL_TILE, dtype=torch.uint8, device=device)
    slots = torch.arange(t, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, t], dtype=torch.int32, device=device)

    def run():
        getattr(openjiuwen_ops, OP)(
            key, value, slots, qsl, rot, key_cache, value_cache, 1, C.BLOCK_SIZE, 0
        )

    ms = C.timed_ms(run, warmup=2, iters=8)
    assert ms > 0.0
    logger.info("[perf] %s mode0: %.3f ms (T=%d, H=%d)", OP, ms, t, h)
