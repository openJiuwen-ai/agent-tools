"""UT for rq_uniform_fia_paged_k4v4: contract + golden vs CPU + perf."""

from __future__ import annotations

import logging
import pytest
import torch

from . import _common as C
from ._common import require_op

logger = logging.getLogger(__name__)

OP = "rq_uniform_fia_paged_k4v4"
PACK = "rq_uniform_pack_k4v4"
INT_MAX = 2147483647


def test_fia_k4v4_layout_contract():
    assert C.K4V4_TILE == 1088


def test_fia_k4v4_registered(openjiuwen_ops):
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)


@pytest.mark.npu
@pytest.mark.golden
def test_fia_k4v4_golden_vs_cpu(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)
    device, dtype = npu_device, torch.float16
    kv_h, q_h, t = 2, 4, 64
    torch.manual_seed(C.SEED + 3)
    key = torch.randn(t, kv_h, C.HEAD_SIZE, dtype=dtype, device=device)
    value = torch.randn(t, kv_h, C.HEAD_SIZE, dtype=dtype, device=device)
    query = torch.randn(1, q_h, C.HEAD_SIZE, dtype=dtype, device=device)
    rot_t = C.dense_rotation(C.HEAD_SIZE, dtype, device)
    rot = rot_t.transpose(0, 1).contiguous()
    bt_cpu, _ = C.build_block_table(t)
    bt = bt_cpu.to(device)
    key_cache, value_cache = C.pack_kv(
        openjiuwen_ops, PACK, key, value, rot_t, key_stride=C.K4V4_TILE, val_stride=C.K4V4_TILE
    )
    out = getattr(openjiuwen_ops, OP)(
        query, key_cache, value_cache, bt, [1], [t], None, rot_t, rot,
        q_h, kv_h, C.HEAD_SIZE, C.BLOCK_SIZE, float(C.HEAD_SIZE**-0.5),
        INT_MAX, INT_MAX, 0, None,
    )
    torch.npu.synchronize()
    C.assert_finite(out, OP)
    assert out.shape == (1, q_h, C.HEAD_SIZE)

    keys = torch.stack(
        [
            C.decode_int4_row(
                key_cache[int(bt_cpu[0, p // C.BLOCK_SIZE]), 0].cpu(),
                p % C.BLOCK_SIZE,
                block_size=C.BLOCK_SIZE,
                head_size=C.HEAD_SIZE,
                dtype=dtype,
            )
            for p in range(t)
        ]
    )
    vals = torch.stack(
        [
            C.decode_int4_row(
                value_cache[int(bt_cpu[0, p // C.BLOCK_SIZE]), 0].cpu(),
                p % C.BLOCK_SIZE,
                block_size=C.BLOCK_SIZE,
                head_size=C.HEAD_SIZE,
                dtype=dtype,
            )
            for p in range(t)
        ]
    )
    ref = C.golden_decode_attn(query[:, :1, :], keys, vals, rot_t, rot)
    cos = C.cosine(out[0, 0].cpu(), ref)
    assert cos > 0.90, f"cosine {cos}"


@pytest.mark.npu
@pytest.mark.perf
def test_fia_k4v4_perf(openjiuwen_ops, npu_device):
    require_op(openjiuwen_ops, OP)
    require_op(openjiuwen_ops, PACK)
    device, dtype = npu_device, torch.float16
    kv_h, q_h, t = 4, 8, 128
    key = torch.randn(t, kv_h, C.HEAD_SIZE, dtype=dtype, device=device)
    value = torch.randn(t, kv_h, C.HEAD_SIZE, dtype=dtype, device=device)
    query = torch.randn(1, q_h, C.HEAD_SIZE, dtype=dtype, device=device)
    rot_t = C.dense_rotation(C.HEAD_SIZE, dtype, device)
    rot = rot_t.transpose(0, 1).contiguous()
    bt, _ = C.build_block_table(t)
    bt = bt.to(device)
    key_cache, value_cache = C.pack_kv(
        openjiuwen_ops, PACK, key, value, rot_t, key_stride=C.K4V4_TILE, val_stride=C.K4V4_TILE
    )

    def run():
        getattr(openjiuwen_ops, OP)(
            query, key_cache, value_cache, bt, [1], [t], None, rot_t, rot,
            q_h, kv_h, C.HEAD_SIZE, C.BLOCK_SIZE, float(C.HEAD_SIZE**-0.5),
            INT_MAX, INT_MAX, 0, None,
        )

    ms = C.timed_ms(run)
    assert ms > 0.0
    logger.info(f"[perf] {OP}: {ms:.3f} ms")
