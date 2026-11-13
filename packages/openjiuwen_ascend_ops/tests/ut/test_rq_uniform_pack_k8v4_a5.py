"""UT for rq_uniform_pack_k8v4 mode1 (A5 TQ_K8_FP8 pure AIV).

Run only this file on ascend950:
  pytest tests/ut/test_rq_uniform_pack_k8v4_a5.py -v
  pytest tests/ut/test_rq_uniform_pack_k8v4_a5.py -v -m "npu and golden"
  pytest tests/ut/test_rq_uniform_pack_k8v4_a5.py -v -m "npu and perf"

Do not mix with test_rq_uniform_pack_k8v4.py (mode0 MIX) on the same session
if the device was left in a bad state by a prior vector-core fault.
"""

from __future__ import annotations

import logging
import sys

import pytest
import torch

from . import _common as C
from ._common import require_op

logger = logging.getLogger(__name__)

OP = "rq_uniform_pack_k8v4"


def _host_probe(msg: str) -> None:
    """Write past pytest capture so probes land in tee/test.log without -s."""
    sys.__stderr__.write(msg + "\n")
    sys.__stderr__.flush()


def _has_fp8() -> bool:
    return hasattr(torch, "float8_e4m3fn")


def _skip_if_not_a5() -> None:
    if not hasattr(torch, "npu") or not hasattr(torch.npu, "get_device_name"):
        return
    name = torch.npu.get_device_name(0).lower()
    if ("910b" in name or "910_93" in name) and "950" not in name and "910_95" not in name:
        pytest.skip(f"quant_mode=1 requires A5, device={name!r}")


def test_pack_k8v4_a5_layout_contract():
    assert C.K8V4_KEY_TILE_FP8 == 2048
    assert C.K8V4_KEY_BYTES_PER_ROW_FP8 == 128
    assert C.key_tile_bytes(C.HEAD_SIZE, C.QUANT_MODE_K8_FP8) == 2048
    assert C.K8V4_VAL_TILE == 1088
    # D=128 volume: mode1 key D + value (D/2+4) = 1.5D+4 = 196
    assert C.K8V4_KEY_BYTES_PER_ROW_FP8 + C.K8V4_VAL_BYTES_PER_ROW == 196


def test_pack_k8v4_a5_fp8_cpu_golden_roundtrip():
    if not _has_fp8():
        pytest.skip("torch.float8_e4m3fn unavailable")
    torch.manual_seed(C.SEED)
    rows = torch.randn(8, C.HEAD_SIZE, dtype=torch.float16)
    codes = C.encode_k8v4_key_fp8_cpu(rows)
    assert codes.dtype == torch.uint8
    assert codes.shape == (8, C.HEAD_SIZE)
    recon = codes.view(torch.float8_e4m3fn).float()
    err = (recon - rows.float()).abs().mean().item()
    assert err < 0.05


def test_pack_k8v4_a5_registered(openjiuwen_ops):
    require_op(openjiuwen_ops, OP)


@pytest.mark.npu
@pytest.mark.golden
@pytest.mark.parametrize("tokens,heads", [(32, 2), (128, 8)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_pack_k8v4_a5_golden_qdq_fp8(openjiuwen_ops, npu_device, tokens, heads, dtype):
    """A5 TQ_K8_FP8: Key = fp8_e4m3fn (half/bf16→float→fp8), Value = U4.

    tokens=128 forces m>1 per core when aivNum < tokens.
    Value quant runs in half. vmin/vstep are stored as dtype T (fp16 or bf16),
    matching FIA which reads pack meta as the query dtype.
    Two UB phases overlap the next chunk's MTE2 with the current chunk's encode.
    """
    require_op(openjiuwen_ops, OP)
    if not _has_fp8():
        pytest.skip("torch.float8_e4m3fn unavailable")
    _skip_if_not_a5()

    device = npu_device
    t, h = tokens, heads
    torch.manual_seed(C.SEED + 3 + tokens + (1 if dtype == torch.bfloat16 else 0))
    key = torch.randn(t, h, C.HEAD_SIZE, dtype=dtype, device=device)
    value = torch.randn(t, h, C.HEAD_SIZE, dtype=dtype, device=device)
    try:
        key_cache, value_cache = C.pack_kv(
            openjiuwen_ops,
            OP,
            key,
            value,
            None,
            key_stride=C.K8V4_KEY_TILE_FP8,
            val_stride=C.K8V4_VAL_TILE,
            quant_mode=C.QUANT_MODE_K8_FP8,
        )
    except Exception as exc:  # pragma: no cover - env / SoC gate
        msg = str(exc).lower()
        if "quant_mode" in msg or "a5" in msg or "dav_3510" in msg or "950" in msg:
            pytest.skip(f"mode1 not available on this SoC: {exc}")
        raise

    bt, _ = C.build_block_table(t)
    key_cpu = key.float().cpu()
    val_cpu = value.float().cpu()
    # Host bits for device L2/L3/L6 compare (bypass pytest stdout capture).
    key_u16 = key[0, 0, :4].detach().cpu().contiguous().view(torch.uint16)
    fp8_ref0 = C.encode_k8v4_key_fp8_cpu(key_cpu[0, 0:1]).view(-1)
    host_kc08 = key_cache[0, 0, :8].detach().cpu().tolist()
    _host_probe(
        f"A5 host probe key[0,0].u16[:4]={key_u16.tolist()} "
        f"fp8_ref[:8]={fp8_ref0[:8].tolist()} host_kc[0,0,:8]={host_kc08}"
    )
    for hi in range(h):
        kc = key_cache[:, hi].detach().cpu().contiguous()
        vc = value_cache[:, hi].detach().cpu().contiguous()
        knz = int((kc != 0).sum().item())
        vnz = int((vc != 0).sum().item())
        _host_probe(
            f"A5 cache head{hi}: key_nz={knz}/{kc.numel()} "
            f"key_sum={int(kc.to(torch.int64).sum().item())} val_nz={vnz}/{vc.numel()}"
        )

    for pos in range(t):
        blk = int(bt[0, pos // C.BLOCK_SIZE])
        off = pos % C.BLOCK_SIZE
        tile = key_cache[blk, 0].cpu().contiguous()
        yk = C.decode_k8v4_key_row_fp8(tile, off, head_size=C.HEAD_SIZE)
        yv = C.decode_int4_row(
            value_cache[blk, 0].cpu(),
            off,
            block_size=C.BLOCK_SIZE,
            head_size=C.HEAD_SIZE,
            dtype=dtype,  # cache vmin/vstep match query dtype T
        )
        key_err = (yk - key_cpu[pos, 0]).abs().mean().item()
        ref = C.encode_k8v4_key_fp8_cpu(key_cpu[pos, 0:1]).view(-1)
        exp_off = off * C.HEAD_SIZE
        got = tile[exp_off:exp_off + C.HEAD_SIZE]
        if key_err >= 0.08:
            # Locate ref bytes inside the tile to detect SoA mis-alignment.
            hits = []
            for shift in (0, 64, 128, 132, 256, 512):
                if shift + C.HEAD_SIZE > tile.numel():
                    continue
                win = tile[shift:shift + C.HEAD_SIZE]
                match = int((win == ref).sum().item())
                hits.append((shift, match))
            # Also scan every 128B boundary.
            best_shift, best_match = -1, -1
            for shift in range(0, tile.numel() - C.HEAD_SIZE + 1, C.HEAD_SIZE):
                match = int((tile[shift:shift + C.HEAD_SIZE] == ref).sum().item())
                if match > best_match:
                    best_shift, best_match = shift, match
            logger.error(
                "A5 key QDQ fail pos=%d blk=%d off=%d exp_off=%d err=%.4f "
                "got_nz=%d ref_nz=%d got[:8]=%s ref[:8]=%s "
                "fixed_shift_hits=%s best_soa_shift=%d match=%d/%d tile_nz=%d",
                pos, blk, off, exp_off, key_err,
                int((got != 0).sum().item()), int((ref != 0).sum().item()),
                got[:8].tolist(), ref[:8].tolist(),
                hits, best_shift, best_match, C.HEAD_SIZE,
                int((tile != 0).sum().item()),
            )
        assert key_err < 0.08, (
            f"key QDQ pos={pos} blk={blk} off={off} err={key_err:.4f} "
            f"(check log for SoA shift hits)"
        )
        assert (yv - val_cpu[pos, 0]).abs().mean().item() < 0.25
        mismatch = (got != ref).sum().item()
        assert mismatch <= max(2, C.HEAD_SIZE // 32)


@pytest.mark.npu
@pytest.mark.perf
@pytest.mark.parametrize("tokens,heads", [(32, 2), (128, 8), (256, 16)])
def test_pack_k8v4_a5_perf_fp8(openjiuwen_ops, npu_device, tokens, heads):
    require_op(openjiuwen_ops, OP)
    _skip_if_not_a5()
    device = npu_device
    dtype = torch.float16
    key = torch.randn(tokens, heads, C.HEAD_SIZE, dtype=dtype, device=device)
    value = torch.randn(tokens, heads, C.HEAD_SIZE, dtype=dtype, device=device)
    nblocks = (tokens + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    key_cache = torch.zeros(nblocks, heads, C.K8V4_KEY_TILE_FP8, dtype=torch.uint8, device=device)
    value_cache = torch.zeros(nblocks, heads, C.K8V4_VAL_TILE, dtype=torch.uint8, device=device)
    slots = torch.arange(tokens, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, tokens], dtype=torch.int32, device=device)

    def run():
        getattr(openjiuwen_ops, OP)(
            key, value, slots, qsl, None, key_cache, value_cache, 1, C.BLOCK_SIZE, 1
        )

    try:
        ms = C.timed_ms(run, warmup=2, iters=8)
    except Exception as exc:
        msg = str(exc).lower()
        if "quant_mode" in msg or "a5" in msg or "dav_3510" in msg:
            pytest.skip(f"mode1 perf skipped: {exc}")
        raise
    assert ms > 0.0
    logger.info("[perf] %s mode1: %.3f ms (T=%d, H=%d)", OP, ms, tokens, heads)
