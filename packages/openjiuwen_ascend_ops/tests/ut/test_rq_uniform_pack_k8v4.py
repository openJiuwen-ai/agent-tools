"""UT for rq_uniform_pack_k8v4: contract + golden QDQ + perf."""

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
    assert C.BLOCK_SIZE % 16 == 0


def test_pack_k8v4_registered(openjiuwen_ops):
    require_op(openjiuwen_ops, OP)


@pytest.mark.npu
@pytest.mark.golden
def test_pack_k8v4_golden_qdq(openjiuwen_ops, npu_device):
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
        # Scheme A / int4 QDQ: relative error stays bounded under identity R.
        assert (yk - key_cpu[pos, 0]).abs().mean().item() < 0.15
        assert (yv - val_cpu[pos, 0]).abs().mean().item() < 0.25


@pytest.mark.npu
@pytest.mark.perf
def test_pack_k8v4_perf(openjiuwen_ops, npu_device):
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
            key, value, slots, qsl, rot, key_cache, value_cache, 1, C.BLOCK_SIZE
        )

    ms = C.timed_ms(run, warmup=2, iters=8)
    assert ms > 0.0
    logger.info(f"[perf] {OP}: {ms:.3f} ms (T={t}, H={h})")
