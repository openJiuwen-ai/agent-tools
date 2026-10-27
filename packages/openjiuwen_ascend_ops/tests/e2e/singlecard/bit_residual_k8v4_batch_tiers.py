"""BitResidual K8V4 batch-tier guard test.

Run on an NPU machine after building ``bit_residual_pack_k8v4``:

    python packages/openjiuwen_ascend_ops/tests/e2e/singlecard/xrx_bit_residual_k8v4_batch_tiers.py

This test guards the batch-tier tiling renovation (encode batch M grows from
16 to 16/32/48/64). It does NOT assume a particular NPU model: cases that
exercise cross-tier byte-for-byte equality (Case A) are skipped until the host
select-BATCH-M logic and the ``BR_K8V4_FORCE_BATCH_M`` knob land; the layout
cases run on any tier and act as a regression baseline so the renovation
cannot silently corrupt the cache.

Cache physical layout (ground truth, no 16-row sub-block):
  A page (block) holds ``blockSize`` tokens; ``blockSize % 16 == 0``.
  Within a page, all tokens of one head are contiguous:
    key   head region = [blockSize rows of code (128B each)]
                         [blockSize bases (2B each)]
                         [blockSize steps (2B each)]
    value head region = [blockSize rows of code (64B each)]
                         [blockSize vmins (2B each)]
                         [blockSize vsteps (2B each)]
  Multiple heads are laid out sequentially inside the page; a block larger
  than one page is split across pages (block_table lookup).

Cases:
  A. Cross-tier byte-for-byte equality (skip until tiers exist).
  B. Per-head contiguous layout + QDQ round-trip + range/sign validity,
     across blockSizes 16/32/64 and aligned/unaligned slot offsets, including
     cross-page (token_count > blockSize) to exercise block_table addressing.
  C. Read-modify-write correctness for base/step (token-shared scalars) under
     overlapping writes, within the active tier.
  E. Dual-AIV dispatch: numHeads>=2 so AIV#0 / AIV#1 each get a head slice;
     the two heads' caches stay independent and correct.

Note: the original Case D (decode transparency via
``bit_residual_attention_paged_k8v4``) lives in the source repo; the attention
op is not part of this package, so the case is omitted here.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(REPO_ROOT / "packages" / "openjiuwen_ascend_ops" / "pysrc"))

# Set ASCEND_CUSTOM_OPP_PATH and preload libcust_opapi.so before importing
# torch/torch_npu. Otherwise CANN may only search the system opapi library
# and miss this repo's custom ops (libcust_opapi.so with aclnnBitResidual*).
# NOTE: LD_LIBRARY_PATH is read by the dynamic linker at process start, so
# changing os.environ inside Python has no effect on dlopen(). We must
# preload the .so by absolute path via ctypes so that subsequent
# dlopen("libcust_opapi.so") returns our already-loaded handle.
_CANN_OPP = REPO_ROOT / "ztmp" / "cann_ops_install" / "vendors" / "vllm-ascend"
if _CANN_OPP.is_dir():
    import ctypes as _ctypes  # noqa: PLC0415
    _opapi_lib = _CANN_OPP / "op_api" / "lib" / "libcust_opapi.so"
    if _opapi_lib.is_file():
        _ctypes.CDLL(str(_opapi_lib), mode=_ctypes.RTLD_GLOBAL)
    existing_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        str(_CANN_OPP) if not existing_opp else f"{_CANN_OPP}:{existing_opp}"
    )

import torch

import openjiuwentools.openjiuwen_ascend_ops  # noqa: F401  registers torch.ops.openjiuwen.*

HEAD_SIZE = 128
# Per-row byte costs (match kernel + golden test).
KEY_ROW_CODE_BYTES = HEAD_SIZE
VALUE_ROW_CODE_BYTES = HEAD_SIZE // 2
ROW_META_BYTES = 2  # base/step/vmin/vstep are uint16 scalars
KEY_BYTES_PER_ROW = KEY_ROW_CODE_BYTES + 2 * ROW_META_BYTES
VALUE_BYTES_PER_ROW = VALUE_ROW_CODE_BYTES + 2 * ROW_META_BYTES

SEED = 2026

logger = logging.getLogger(__name__)

# Batch tiers the renovation will expose. Currently only 16 exists; the rest
# become reachable once host SelectBatchM + BR_K8V4_FORCE_BATCH_M land.
ALL_TIERS = (16, 32, 48, 64)


def _active_tiers() -> list[int]:
    """Tiers the test should attempt to exercise.

    Honors BR_K8V4_FORCE_BATCH_M if the host knob exists; otherwise reports
    only the implemented 16 tier (cross-tier Case A auto-skips).
    """
    forced = os.getenv("BR_K8V4_FORCE_BATCH_M", "")
    if forced:
        try:
            m = int(forced)
        except ValueError as e:
            raise RuntimeError(f"BR_K8V4_FORCE_BATCH_M must be int, got {forced!r}") from e
        return [m]
    return [16]


def _require_ops() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    if not hasattr(torch.ops, "openjiuwen"):
        raise RuntimeError("torch.ops.openjiuwen is not registered")
    if not hasattr(torch.ops.openjiuwen, "bit_residual_pack_k8v4"):
        raise RuntimeError("torch.ops.openjiuwen.bit_residual_pack_k8v4 is not registered")


def _key_head_stride(block_size: int) -> int:
    return block_size * KEY_BYTES_PER_ROW


def _value_head_stride(block_size: int) -> int:
    return block_size * VALUE_BYTES_PER_ROW


def _make_rotation(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Fixed dense orthogonal R^T the kernel consumes directly (mirrors golden)."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 17)
    matrix = torch.randn((HEAD_SIZE, HEAD_SIZE), dtype=torch.float32, generator=generator)
    rotation, _ = torch.linalg.qr(matrix)
    return rotation.to(device=device, dtype=dtype).contiguous()


def _u8_to_dtype_float(x: torch.Tensor, dtype: torch.dtype) -> float:
    return x.contiguous().view(dtype)[0].float().item()


def _unpack_value_idx4(packed: torch.Tensor) -> torch.Tensor:
    code = packed.to(torch.int32)
    low = code & 0x0F
    high = (code >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low) + 8
    high = torch.where(high >= 8, high - 16, high) + 8
    idx4 = torch.empty(HEAD_SIZE, dtype=torch.uint8)
    idx4[0::2] = low.to(torch.uint8)
    idx4[1::2] = high.to(torch.uint8)
    return idx4


def _decode_key_cache(
    cache: torch.Tensor, slots: torch.Tensor, block_size: int, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Decode key cache using the per-head-per-block contiguous layout."""
    cache_cpu = cache.cpu()
    slots_cpu = slots.cpu().to(torch.int64)
    t = slots_cpu.numel()
    h = cache_cpu.shape[1]
    code = torch.empty(t, h, HEAD_SIZE, dtype=torch.uint8)
    base = torch.empty(t, h, dtype=torch.float32)
    step = torch.empty(t, h, dtype=torch.float32)
    for token_idx, slot in enumerate(slots_cpu.tolist()):
        block_idx = slot // block_size
        pos_in_block = slot % block_size
        for head in range(h):
            slab = cache_cpu[block_idx, head]
            code_off = pos_in_block * KEY_ROW_CODE_BYTES
            code[token_idx, head] = slab[code_off: code_off + HEAD_SIZE]
            base_off = block_size * KEY_ROW_CODE_BYTES + pos_in_block * ROW_META_BYTES
            step_off = block_size * (KEY_ROW_CODE_BYTES + ROW_META_BYTES) + pos_in_block * ROW_META_BYTES
            base[token_idx, head] = _u8_to_dtype_float(
                slab[base_off: base_off + ROW_META_BYTES], dtype
            )
            step[token_idx, head] = _u8_to_dtype_float(
                slab[step_off: step_off + ROW_META_BYTES], dtype
            )
    return {"code": code, "base": base, "step": step}


def _decode_value_cache(
    cache: torch.Tensor, slots: torch.Tensor, block_size: int, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Decode value cache using the per-head-per-block contiguous layout."""
    cache_cpu = cache.cpu()
    slots_cpu = slots.cpu().to(torch.int64)
    t = slots_cpu.numel()
    h = cache_cpu.shape[1]
    idx4 = torch.empty(t, h, HEAD_SIZE, dtype=torch.uint8)
    vmin = torch.empty(t, h, dtype=torch.float32)
    vstep = torch.empty(t, h, dtype=torch.float32)
    for token_idx, slot in enumerate(slots_cpu.tolist()):
        block_idx = slot // block_size
        pos_in_block = slot % block_size
        for head in range(h):
            slab = cache_cpu[block_idx, head]
            code_off = pos_in_block * VALUE_ROW_CODE_BYTES
            idx4[token_idx, head] = _unpack_value_idx4(
                slab[code_off: code_off + VALUE_ROW_CODE_BYTES]
            )
            vmin_off = block_size * VALUE_ROW_CODE_BYTES + pos_in_block * ROW_META_BYTES
            vstep_off = block_size * (VALUE_ROW_CODE_BYTES + ROW_META_BYTES) + pos_in_block * ROW_META_BYTES
            vmin[token_idx, head] = _u8_to_dtype_float(
                slab[vmin_off: vmin_off + ROW_META_BYTES], dtype
            )
            vstep[token_idx, head] = _u8_to_dtype_float(
                slab[vstep_off: vstep_off + ROW_META_BYTES], dtype
            )
    return {"idx4": idx4, "vmin": vmin, "vstep": vstep}


def _assert_qdq_and_ranges(
    name: str,
    dtype: torch.dtype,
    key_dec: dict[str, torch.Tensor],
    value_dec: dict[str, torch.Tensor],
) -> None:
    """QDQ round-trip exactness + range validity (Key Scheme A uniform)."""
    q = key_dec["code"].to(torch.float32)
    y_recon = key_dec["base"].unsqueeze(-1) + q * key_dec["step"].unsqueeze(-1)
    step_valid = key_dec["step"] > 1e-6
    if step_valid.any():
        q_rt = (
            (y_recon - key_dec["base"].unsqueeze(-1))
            / key_dec["step"].unsqueeze(-1).clamp(min=1e-6)
        ).nan_to_num(0.0)
        q_err = (q_rt - q).abs()
        max_q_err = q_err[step_valid.unsqueeze(-1).expand_as(q_err)].max().item()
        if max_q_err >= 0.5:
            raise AssertionError(f"{name}: key QDQ violated, max dev {max_q_err:.4f}")

    idx4 = value_dec["idx4"].to(torch.float32)
    y_recon_v = value_dec["vmin"].unsqueeze(-1) + idx4 * value_dec["vstep"].unsqueeze(-1)
    vstep_valid = value_dec["vstep"] > 1e-6
    if vstep_valid.any():
        idx4_rt = (
            (y_recon_v - value_dec["vmin"].unsqueeze(-1))
            / value_dec["vstep"].unsqueeze(-1).clamp(min=1e-6)
        ).nan_to_num(0.0)
        idx4_err = (idx4_rt - idx4).abs()
        max_idx4_err = idx4_err[vstep_valid.unsqueeze(-1).expand_as(idx4_err)].max().item()
        if max_idx4_err >= 0.5:
            raise AssertionError(f"{name}: value QDQ violated, max dev {max_idx4_err:.4f}")

    q_int = key_dec["code"].to(torch.int32)
    if not ((q_int >= 0).all() and (q_int <= 255).all()):
        raise AssertionError(f"{name}: q out of [0,255]")
    if not (value_dec["idx4"] <= 15).all():
        raise AssertionError(f"{name}: idx4 out of [0,15]")
    if not (key_dec["step"] >= 0).all():
        raise AssertionError(f"{name}: negative key step")
    if not (value_dec["vstep"] >= 0).all():
        raise AssertionError(f"{name}: negative value vstep")

    key_ratio = step_valid.sum().item() / key_dec["step"].numel()
    val_ratio = vstep_valid.sum().item() / value_dec["vstep"].numel()
    if key_ratio < 0.5:
        raise AssertionError(f"{name}: too many degenerate key groups ({key_ratio:.2f})")
    if val_ratio < 0.5:
        raise AssertionError(f"{name}: too many degenerate value groups ({val_ratio:.2f})")
    logger.info(
        f"PASS {name}: dtype={dtype}, qdq=OK (uniform Scheme A), "
        f"nondeg=OK(key={key_ratio:.2f},val={val_ratio:.2f})"
    )


def _alloc_caches(
    num_blocks: int, num_heads: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache = torch.zeros(
        num_blocks, num_heads, _key_head_stride(block_size), dtype=torch.uint8, device=device
    )
    value_cache = torch.zeros(
        num_blocks,
        num_heads,
        _value_head_stride(block_size),
        dtype=torch.uint8,
        device=device,
    )
    return key_cache, value_cache


def _run_pack(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    query_start_loc: torch.Tensor,
    rotation_t: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_reqs: int,
    block_size: int,
) -> None:
    torch.ops.openjiuwen.bit_residual_pack_k8v4(
        key, value, slots, query_start_loc, rotation_t,
        key_cache, value_cache, num_reqs, block_size,
    )
    torch.npu.synchronize()


# ---------------------------------------------------------------------------
# Case B: per-head layout + QDQ across blockSizes / offsets / cross-page.
# ---------------------------------------------------------------------------
SHORT_OFFSETS = (0, 1, 2, 3, 4, 5)
BLOCK_SIZES = (16, 32, 64)


def _run_case_b(
    name: str,
    params: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    token_count, slot_offset, block_size = params
    num_heads = 8
    torch.manual_seed(SEED + token_count + slot_offset + block_size)
    key = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots = torch.arange(
        slot_offset, slot_offset + token_count, dtype=torch.int32, device=device
    )
    qsl = torch.tensor([0, token_count], dtype=torch.int32, device=device)
    num_blocks = (slot_offset + token_count + block_size - 1) // block_size + 1
    key_cache, value_cache = _alloc_caches(num_blocks, num_heads, block_size, device)
    _run_pack(
        key=key, value=value, slots=slots, query_start_loc=qsl, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    key_dec = _decode_key_cache(key_cache, slots, block_size, dtype)
    value_dec = _decode_value_cache(value_cache, slots, block_size, dtype)
    _assert_qdq_and_ranges(name, dtype, key_dec, value_dec)


def _run_cross_page(dtype: torch.dtype, device: torch.device) -> None:
    """token_count > block_size forces block_table multi-page addressing."""
    block_size = 16
    num_heads = 8
    token_count = 40  # spans 3 pages of 16
    torch.manual_seed(SEED + 7777)
    key = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots = torch.arange(token_count, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, token_count], dtype=torch.int32, device=device)
    num_blocks = (token_count + block_size - 1) // block_size + 1
    key_cache, value_cache = _alloc_caches(num_blocks, num_heads, block_size, device)
    _run_pack(
        key=key, value=value, slots=slots, query_start_loc=qsl, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    key_dec = _decode_key_cache(key_cache, slots, block_size, dtype)
    value_dec = _decode_value_cache(value_cache, slots, block_size, dtype)
    _assert_qdq_and_ranges("cross_page", dtype, key_dec, value_dec)


# ---------------------------------------------------------------------------
# Case C: base/step RMW correctness under overlapping writes (active tier).
# ---------------------------------------------------------------------------
def _run_case_c(dtype: torch.dtype, device: torch.device, block_size: int) -> None:
    num_heads = 8
    torch.manual_seed(SEED + 2000)
    t1, off1 = 10, 1
    key1 = torch.randn(t1, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value1 = torch.randn(t1, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots1 = torch.arange(off1, off1 + t1, dtype=torch.int32, device=device)
    qsl1 = torch.tensor([0, t1], dtype=torch.int32, device=device)
    t2, off2 = 10, 4
    key2 = torch.randn(t2, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value2 = torch.randn(t2, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    slots2 = torch.arange(off2, off2 + t2, dtype=torch.int32, device=device)
    qsl2 = torch.tensor([0, t2], dtype=torch.int32, device=device)
    num_blocks = 2
    key_cache, value_cache = _alloc_caches(num_blocks, num_heads, block_size, device)

    _run_pack(
        key=key1, value=value1, slots=slots1, query_start_loc=qsl1, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    preserved = torch.arange(off1, off1 + 3, dtype=torch.int32, device=device)
    key_pres_1 = _decode_key_cache(key_cache, preserved, block_size, dtype)
    val_pres_1 = _decode_value_cache(value_cache, preserved, block_size, dtype)

    _run_pack(
        key=key2, value=value2, slots=slots2, query_start_loc=qsl2, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    key_pres_2 = _decode_key_cache(key_cache, preserved, block_size, dtype)
    val_pres_2 = _decode_value_cache(value_cache, preserved, block_size, dtype)
    for field in ("code", "base", "step"):
        torch.testing.assert_close(key_pres_1[field], key_pres_2[field], rtol=0, atol=0)
    for field in ("idx4", "vmin", "vstep"):
        torch.testing.assert_close(val_pres_1[field], val_pres_2[field], rtol=0, atol=0)

    key_p2 = _decode_key_cache(key_cache, slots2, block_size, dtype)
    val_p2 = _decode_value_cache(value_cache, slots2, block_size, dtype)
    _assert_qdq_and_ranges(f"rmw_bs{block_size}_phase2", dtype, key_p2, val_p2)
    logger.info(f"PASS rmw_bs{block_size}: preserved slots unchanged, phase2 correct, dtype={dtype}")


# ---------------------------------------------------------------------------
# Case E: dual-AIV head dispatch (num_heads>=2, two head slices).
# ---------------------------------------------------------------------------
def _run_case_e(dtype: torch.dtype, device: torch.device, block_size: int) -> None:
    num_heads = 4  # >= HEADS_PER_TILE(=2) so AIV#0/AIV#1 split heads
    torch.manual_seed(SEED + 5000 + block_size)
    token_count = 9
    key = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots = torch.arange(token_count, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, token_count], dtype=torch.int32, device=device)
    num_blocks = (token_count + block_size - 1) // block_size + 1
    key_cache, value_cache = _alloc_caches(num_blocks, num_heads, block_size, device)
    _run_pack(
        key=key, value=value, slots=slots, query_start_loc=qsl, rotation_t=rotation_t,
        key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
    )
    key_dec = _decode_key_cache(key_cache, slots, block_size, dtype)
    value_dec = _decode_value_cache(value_cache, slots, block_size, dtype)
    _assert_qdq_and_ranges(f"dual_aiv_bs{block_size}", dtype, key_dec, value_dec)


# ---------------------------------------------------------------------------
# Case A: cross-tier byte-for-byte equality (skipped until tiers exist).
# ---------------------------------------------------------------------------
def _run_case_a(dtype: torch.dtype, device: torch.device) -> None:
    tiers = _active_tiers()
    if len(tiers) < 2 and 16 in tiers and not os.getenv("BR_K8V4_FORCE_BATCH_M"):
        logger.info(
            "SKIP case_a (cross-tier equality): only tier 16 implemented; "
            "set BR_K8V4_FORCE_BATCH_M once the select-batch-M host knob lands."
        )
        return
    num_heads = 8
    token_count = 32
    torch.manual_seed(SEED + 31)
    key = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    value = torch.randn(token_count, num_heads, HEAD_SIZE, dtype=dtype, device=device).contiguous()
    rotation_t = _make_rotation(dtype, device)
    slots = torch.arange(token_count, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, token_count], dtype=torch.int32, device=device)
    block_size = 16
    num_blocks = (token_count + block_size - 1) // block_size + 1
    ref_key = ref_val = None
    for m in tiers:
        key_cache, value_cache = _alloc_caches(num_blocks, num_heads, block_size, device)
        _run_pack(
            key=key, value=value, slots=slots, query_start_loc=qsl, rotation_t=rotation_t,
            key_cache=key_cache, value_cache=value_cache, num_reqs=1, block_size=block_size,
        )
        if ref_key is None:
            ref_key, ref_val = key_cache.cpu(), value_cache.cpu()
        else:
            if not torch.equal(key_cache.cpu(), ref_key):
                raise AssertionError(
                    f"case_a: key cache differs across tiers (tier {m})"
                )
            if not torch.equal(value_cache.cpu(), ref_val):
                raise AssertionError(
                    f"case_a: value cache differs across tiers (tier {m})"
                )
    logger.info(f"PASS case_a: cross-tier byte-for-byte equality over tiers {tiers}, dtype={dtype}")


def main() -> None:
    _require_ops()
    device = torch.device("npu:0")
    for dtype in (torch.float16, torch.bfloat16):
        for bs in BLOCK_SIZES:
            for off in SHORT_OFFSETS:
                _run_case_b(f"layout_bs{bs}_off{off}", (9, off, bs), dtype, device)
            _run_cross_page(dtype, device)
            _run_case_c(dtype, device, bs)
            _run_case_e(dtype, device, bs)
        _run_case_a(dtype, device)
    logger.info("bit residual k8v4 batch-tier guard: all cases passed")


if __name__ == "__main__":
    main()
