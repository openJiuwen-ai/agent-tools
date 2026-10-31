/**
 * RqUniform K8/V4 AIV dequant helpers (head_size=128).
 *
 * AscendC on 910B does not support Cast<int16, uint8>; widen via half then int16.
 * Metadata (base/step/vmin/vstep) must be brought into UB via DataCopyPad —
 * raw scalar GM half reads are unreliable on AIV.
 *
 * P2: batch copy contiguous same-PA-block rows (codes + meta runs), then
 * per-row / tile decode into a staged out tile (see DequantKvImpl).
 * A1: DequantKvImpl dual-buffers tmpBuff1 so MTE2/MTE3 overlap.
 * Key/Value decode: run-level TqDecodeKeyTile / TqDecodeValueTile;
 * tile=8 scratch overlays tmpBuff1 tail (dedicated dequantFp* stays 1-row).
 * Dual-AIV S2 split: both subcores dequant disjoint [0,half)/[half,s2) WS rows.
 *
 * Key quant schemes (must match pack EncodeKeyBatch):
 *   BR_KEY_UNIFORM_SCHEME=1 (A, default): y = base + q*step, q in [0,255]
 *   BR_KEY_UNIFORM_SCHEME=2 (B): y = s*(q-127.5); meta0=-127.5*s, meta1=s
 *   BR_KEY_UNIFORM_SCHEME=3 (C, legacy): y = sign*(base + q7*step),
 *     code = (q7<<1)|sign; needs 3×fp32 scratch (fp32UbC)
 * A/B use 2×fp32 scratch; C uses 3×fp32.
 *
 * Meta P1: each decode tile copies contiguous GM meta0/meta1 runs into two
 * aligned 32B UB buffers, then batch-Casts both runs.
 * This replaces 2*numRows 2B DataCopyPad transactions and 32B-per-row slots
 * with two bulk DataCopyPad transactions and 64B fixed UB staging per tile.
 * Meta P4: metadata stays in UB after Cast. Brcb expands each row scalar to
 * one FP32 block and row-broadcast Mul/Add consumes it directly, avoiding
 * GetValue and all metadata V_S synchronization.
 * Meta P9b: dequantInt8Buf_ holds up to BR_S2_SUB_MAX packed meta rows so
 * each PA run does one meta DMA+cast, then tiles only decode.
 * P14: TqApplyRowAffine — one column loop, fewer PipeBarriers.
 * P17a: TqApplyRowAffine — unroll headDim=128 (fp32 2×64 / fp16 1×128)
 *   and headDim=256 (fp32 4×64 / fp16 2×128) Mul/Add.
 * P17b-B: Value +8 folded once per PA run (vmin'=vmin+8*vstep on n rows);
 * TqDecodeValueTile consumes signed Cast(int4) with pre-folded vmin'.
 * P18 spike: meta UB ping-pong (2×768B) so next-run MTE2 can overlap
 * current-run VEC (issue after SetFlag(V_MTE3), before WaitFlag).
 * Native 16-bit affine (Scheme A/B only):
 *   half OutT: u8/int4→half → TqApplyRowAffine(half) → store half WS (no CAST #1).
 *     Meta stays half end-to-end (no half→fp32→half); stage into 32B-aligned
 *     meta0/1Align for Brcb when j0%16!=0.
 *   bf16 OutT: AscendC dav_c220 Mul/Add/Adds/Muls do NOT support bfloat16_t,
 *     and Cast has no uint8/int4→bf16 — keep fp32 affine + Cast→bf16.
 * Scheme C legacy (BR_KEY_UNIFORM_SCHEME=3): always fp32 affine.
 */
#ifndef BR_DEQUANT_DEVICE_H
#define BR_DEQUANT_DEVICE_H

#include "kernel_operator.h"
#include "tq_pack_layout.h"

namespace tq_dequant {

using namespace AscendC;
using tq_pack::BR_HEAD_SIZE;

// 1=A asymmetric uniform; 2=B symmetric uniform; 3=C legacy LSB-sign+q7.
// Default A; tests use A. Compile with -DBR_KEY_UNIFORM_SCHEME=3 for legacy.
#ifndef BR_KEY_UNIFORM_SCHEME
#define BR_KEY_UNIFORM_SCHEME 1
#endif
static constexpr uint32_t BR_KEY_UNIFORM_SCHEME_ID = BR_KEY_UNIFORM_SCHEME;
static constexpr bool BR_KEY_SCHEME_LEGACY_SIGN = (BR_KEY_UNIFORM_SCHEME == 3);

static constexpr uint32_t BR_S2_SUB_MAX = 64U;
// P9b: packed meta0/meta1 runs then FP32 cast rows (up to BR_S2_SUB_MAX).
// Layout (per slot): [0, 128) meta0 half, [128, 256) meta1 half,
// [256, 512) meta0 fp32, [512, 768) meta1 fp32. Cast sources stay 32B-aligned.
static constexpr uint32_t BR_PACKED_META_TILE_ROWS = BR_S2_SUB_MAX;
static constexpr uint32_t BR_PACKED_META_BYTES =
    BR_PACKED_META_TILE_ROWS * static_cast<uint32_t>(sizeof(half));
static constexpr uint32_t BR_META_FP32_ROW_BYTES =
    BR_PACKED_META_TILE_ROWS * static_cast<uint32_t>(sizeof(float));
static constexpr uint32_t BR_META_FP32_0_UB_OFF =
    ((2U * BR_PACKED_META_BYTES + 31U) / 32U) * 32U;
static constexpr uint32_t BR_META_FP32_1_UB_OFF =
    BR_META_FP32_0_UB_OFF + BR_META_FP32_ROW_BYTES;
static constexpr uint32_t BR_DEQUANT_UB_BYTES =
    BR_META_FP32_1_UB_OFF + BR_META_FP32_ROW_BYTES;
// P18: two independent meta slots; run r uses slot (runId % 2).
static constexpr uint32_t BR_META_SLOT_BYTES = BR_DEQUANT_UB_BYTES;
static constexpr uint32_t BR_DEQUANT_UB_BYTES_PINGPONG =
    2U * BR_META_SLOT_BYTES;

// Legacy single-row staging offsets (TqCopyMetaPair only).
static constexpr uint32_t BR_META0_UB_OFF = BR_HEAD_SIZE;
static constexpr uint32_t BR_META1_UB_OFF = BR_HEAD_SIZE + 32U;

// Pack stores base/step/vmin/vstep as 2-byte floats matching the pack input
// dtype (half or bfloat16). FIA must decode with the same type — reading bf16
// metadata as half (or vice versa) produces catastrophic dequant values.
__aicore__ inline float TqReadFp16FromUb(LocalTensor<uint8_t> ub, LocalTensor<float> scratch,
    uint32_t byteOffset)
{
    auto asHalf = ub.template ReinterpretCast<half>();
    Cast(scratch, asHalf[byteOffset / sizeof(half)], RoundMode::CAST_NONE, 1);
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
    return scratch.GetValue(0);
}

__aicore__ inline float TqReadBf16FromUb(LocalTensor<uint8_t> ub, LocalTensor<float> scratch,
    uint32_t byteOffset)
{
    auto asBf16 = ub.template ReinterpretCast<bfloat16_t>();
    Cast(scratch, asBf16[byteOffset / sizeof(bfloat16_t)], RoundMode::CAST_NONE, 1);
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
    return scratch.GetValue(0);
}

template <typename MetaT>
__aicore__ inline float TqReadMeta16FromUb(LocalTensor<uint8_t> ub, LocalTensor<float> scratch,
    uint32_t byteOffset)
{
    if constexpr (IsSameType<MetaT, bfloat16_t>::value) {
        return TqReadBf16FromUb(ub, scratch, byteOffset);
    } else {
        return TqReadFp16FromUb(ub, scratch, byteOffset);
    }
}

// Copy packed meta0/meta1 runs. GM already uses SoA layout, so each run is
// contiguous; DataCopyPad handles sub-32B tails and potentially unaligned GM.
// Caller must V_MTE2-drain prior V before this MTE2.
__aicore__ inline void TqCopyPackedMetaTile(GlobalTensor<uint8_t> srcGm,
    LocalTensor<uint8_t> dst, uint64_t meta0GmOff, uint64_t meta1GmOff,
    uint32_t numRows)
{
    const uint32_t copyBytes = numRows * static_cast<uint32_t>(sizeof(half));
    DataCopyExtParams metaParams{1, copyBytes, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    DataCopyPad(dst, srcGm[meta0GmOff], metaParams, padParams);
    DataCopyPad(dst[BR_PACKED_META_BYTES], srcGm[meta1GmOff], metaParams, padParams);
}

template <typename MetaT>
__aicore__ inline void TqCastPackedMetaToFp32(LocalTensor<uint8_t> src,
    uint32_t numRows, LocalTensor<float> dst0, LocalTensor<float> dst1)
{
    if constexpr (IsSameType<MetaT, bfloat16_t>::value) {
        auto src0 = src.template ReinterpretCast<bfloat16_t>();
        auto src1 = src[BR_PACKED_META_BYTES].template ReinterpretCast<bfloat16_t>();
        Cast(dst0, src0, RoundMode::CAST_NONE, numRows);
        Cast(dst1, src1, RoundMode::CAST_NONE, numRows);
    } else {
        auto src0 = src.template ReinterpretCast<half>();
        auto src1 = src[BR_PACKED_META_BYTES].template ReinterpretCast<half>();
        Cast(dst0, src0, RoundMode::CAST_NONE, numRows);
        Cast(dst1, src1, RoundMode::CAST_NONE, numRows);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void TqApplyRowAffine(LocalTensor<float> dst,
    LocalTensor<float> src, LocalTensor<float> offsets,
    LocalTensor<float> scales, LocalTensor<float> broadcast,
    uint32_t numRows, uint32_t headDim)
{
    // Cast→Brcb are both V-pipe; PipeBarrier after caller Cast is enough.
    constexpr uint32_t FP32_BLOCK_ELEMS = 8U;
    constexpr uint32_t FP32_REPEAT_ELEMS = 64U;

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1U;
    repeatParams.src0BlkStride = 1U;
    repeatParams.src1BlkStride = 0U;
    repeatParams.src1RepStride = 1U;

    // P17a: headDim=128 → exactly two 64-wide column chunks. Unroll Mul/Add
    // to drop the column for/if control path seen in L6 source OTHER (~25%).
    if (headDim == BR_HEAD_SIZE) {
        constexpr uint32_t ROW_STRIDE_BLOCKS = BR_HEAD_SIZE / FP32_BLOCK_ELEMS;
        repeatParams.dstRepStride = ROW_STRIDE_BLOCKS;
        repeatParams.src0RepStride = ROW_STRIDE_BLOCKS;

        Brcb(broadcast, scales, (numRows + FP32_BLOCK_ELEMS - 1U) /
            FP32_BLOCK_ELEMS, {1, FP32_BLOCK_ELEMS});
        PipeBarrier<PIPE_V>();
        Mul(dst[0], src[0], broadcast, FP32_REPEAT_ELEMS, numRows, repeatParams);
        Mul(dst[FP32_REPEAT_ELEMS], src[FP32_REPEAT_ELEMS], broadcast,
            FP32_REPEAT_ELEMS, numRows, repeatParams);

        Brcb(broadcast, offsets, (numRows + FP32_BLOCK_ELEMS - 1U) /
            FP32_BLOCK_ELEMS, {1, FP32_BLOCK_ELEMS});
        PipeBarrier<PIPE_V>();
        Add(dst[0], dst[0], broadcast, FP32_REPEAT_ELEMS, numRows, repeatParams);
        Add(dst[FP32_REPEAT_ELEMS], dst[FP32_REPEAT_ELEMS], broadcast,
            FP32_REPEAT_ELEMS, numRows, repeatParams);
        PipeBarrier<PIPE_V>();
        return;
    }

    // P17a D=256: four 64-wide fp32 columns (same 128-tile unroll, twice).
    if (headDim == 256U) {
        constexpr uint32_t ROW_STRIDE_BLOCKS = 256U / FP32_BLOCK_ELEMS;
        repeatParams.dstRepStride = ROW_STRIDE_BLOCKS;
        repeatParams.src0RepStride = ROW_STRIDE_BLOCKS;

        Brcb(broadcast, scales, (numRows + FP32_BLOCK_ELEMS - 1U) /
            FP32_BLOCK_ELEMS, {1, FP32_BLOCK_ELEMS});
        PipeBarrier<PIPE_V>();
        Mul(dst[0], src[0], broadcast, FP32_REPEAT_ELEMS, numRows, repeatParams);
        Mul(dst[FP32_REPEAT_ELEMS], src[FP32_REPEAT_ELEMS], broadcast,
            FP32_REPEAT_ELEMS, numRows, repeatParams);
        Mul(dst[2U * FP32_REPEAT_ELEMS], src[2U * FP32_REPEAT_ELEMS], broadcast,
            FP32_REPEAT_ELEMS, numRows, repeatParams);
        Mul(dst[3U * FP32_REPEAT_ELEMS], src[3U * FP32_REPEAT_ELEMS], broadcast,
            FP32_REPEAT_ELEMS, numRows, repeatParams);

        Brcb(broadcast, offsets, (numRows + FP32_BLOCK_ELEMS - 1U) /
            FP32_BLOCK_ELEMS, {1, FP32_BLOCK_ELEMS});
        PipeBarrier<PIPE_V>();
        Add(dst[0], dst[0], broadcast, FP32_REPEAT_ELEMS, numRows, repeatParams);
        Add(dst[FP32_REPEAT_ELEMS], dst[FP32_REPEAT_ELEMS], broadcast,
            FP32_REPEAT_ELEMS, numRows, repeatParams);
        Add(dst[2U * FP32_REPEAT_ELEMS], dst[2U * FP32_REPEAT_ELEMS], broadcast,
            FP32_REPEAT_ELEMS, numRows, repeatParams);
        Add(dst[3U * FP32_REPEAT_ELEMS], dst[3U * FP32_REPEAT_ELEMS], broadcast,
            FP32_REPEAT_ELEMS, numRows, repeatParams);
        PipeBarrier<PIPE_V>();
        return;
    }

    const uint32_t rowStrideBlocks = headDim / FP32_BLOCK_ELEMS;
    const uint32_t columnLoops =
        (headDim + FP32_REPEAT_ELEMS - 1U) / FP32_REPEAT_ELEMS;
    repeatParams.dstRepStride = rowStrideBlocks;
    repeatParams.src0RepStride = rowStrideBlocks;

    Brcb(broadcast, scales, (numRows + FP32_BLOCK_ELEMS - 1U) /
        FP32_BLOCK_ELEMS, {1, FP32_BLOCK_ELEMS});
    PipeBarrier<PIPE_V>();
    for (uint32_t columnLoop = 0U; columnLoop < columnLoops; ++columnLoop) {
        const uint32_t columnOffset = columnLoop * FP32_REPEAT_ELEMS;
        uint32_t columnCount = headDim - columnOffset;
        if (columnCount > FP32_REPEAT_ELEMS) {
            columnCount = FP32_REPEAT_ELEMS;
        }
        Mul(dst[columnOffset], src[columnOffset], broadcast, columnCount,
            numRows, repeatParams);
    }
    Brcb(broadcast, offsets, (numRows + FP32_BLOCK_ELEMS - 1U) /
        FP32_BLOCK_ELEMS, {1, FP32_BLOCK_ELEMS});
    PipeBarrier<PIPE_V>();
    for (uint32_t columnLoop = 0U; columnLoop < columnLoops; ++columnLoop) {
        const uint32_t columnOffset = columnLoop * FP32_REPEAT_ELEMS;
        uint32_t columnCount = headDim - columnOffset;
        if (columnCount > FP32_REPEAT_ELEMS) {
            columnCount = FP32_REPEAT_ELEMS;
        }
        Add(dst[columnOffset], dst[columnOffset], broadcast, columnCount,
            numRows, repeatParams);
    }
    PipeBarrier<PIPE_V>();
}

// fp16 path: one 128-wide column (256B/repeat). Used only when WS_T=half.
// Brcb still consumes 8 src scalars per repeat (same as fp32); each dst block
// is 16×half. Do NOT use ceil(numRows/16) — that mis-sizes the expand.
__aicore__ inline void TqApplyRowAffine(LocalTensor<half> dst,
    LocalTensor<half> src, LocalTensor<half> offsets,
    LocalTensor<half> scales, LocalTensor<half> broadcast,
    uint32_t numRows, uint32_t headDim)
{
    // Same as fp32: Cast→Brcb is V→V.
    constexpr uint32_t FP16_BLOCK_ELEMS = 16U;
    constexpr uint32_t FP16_REPEAT_ELEMS = 128U;
    constexpr uint32_t BRCB_SRC_PER_REPEAT = 8U;

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1U;
    repeatParams.src0BlkStride = 1U;
    repeatParams.src1BlkStride = 0U;
    repeatParams.src1RepStride = 1U;

    if (headDim == BR_HEAD_SIZE) {
        constexpr uint32_t ROW_STRIDE_BLOCKS = BR_HEAD_SIZE / FP16_BLOCK_ELEMS;
        repeatParams.dstRepStride = ROW_STRIDE_BLOCKS;
        repeatParams.src0RepStride = ROW_STRIDE_BLOCKS;

        Brcb(broadcast, scales, (numRows + BRCB_SRC_PER_REPEAT - 1U) /
            BRCB_SRC_PER_REPEAT, {1, BRCB_SRC_PER_REPEAT});
        PipeBarrier<PIPE_V>();
        Mul(dst, src, broadcast, FP16_REPEAT_ELEMS, numRows, repeatParams);

        Brcb(broadcast, offsets, (numRows + BRCB_SRC_PER_REPEAT - 1U) /
            BRCB_SRC_PER_REPEAT, {1, BRCB_SRC_PER_REPEAT});
        PipeBarrier<PIPE_V>();
        Add(dst, dst, broadcast, FP16_REPEAT_ELEMS, numRows, repeatParams);
        PipeBarrier<PIPE_V>();
        return;
    }

    // P17a D=256: two 128-wide fp16 columns.
    if (headDim == 256U) {
        constexpr uint32_t ROW_STRIDE_BLOCKS = 256U / FP16_BLOCK_ELEMS;
        repeatParams.dstRepStride = ROW_STRIDE_BLOCKS;
        repeatParams.src0RepStride = ROW_STRIDE_BLOCKS;

        Brcb(broadcast, scales, (numRows + BRCB_SRC_PER_REPEAT - 1U) /
            BRCB_SRC_PER_REPEAT, {1, BRCB_SRC_PER_REPEAT});
        PipeBarrier<PIPE_V>();
        Mul(dst, src, broadcast, FP16_REPEAT_ELEMS, numRows, repeatParams);
        Mul(dst[FP16_REPEAT_ELEMS], src[FP16_REPEAT_ELEMS], broadcast,
            FP16_REPEAT_ELEMS, numRows, repeatParams);

        Brcb(broadcast, offsets, (numRows + BRCB_SRC_PER_REPEAT - 1U) /
            BRCB_SRC_PER_REPEAT, {1, BRCB_SRC_PER_REPEAT});
        PipeBarrier<PIPE_V>();
        Add(dst, dst, broadcast, FP16_REPEAT_ELEMS, numRows, repeatParams);
        Add(dst[FP16_REPEAT_ELEMS], dst[FP16_REPEAT_ELEMS], broadcast,
            FP16_REPEAT_ELEMS, numRows, repeatParams);
        PipeBarrier<PIPE_V>();
        return;
    }

    const uint32_t rowStrideBlocks = headDim / FP16_BLOCK_ELEMS;
    const uint32_t columnLoops =
        (headDim + FP16_REPEAT_ELEMS - 1U) / FP16_REPEAT_ELEMS;
    repeatParams.dstRepStride = rowStrideBlocks;
    repeatParams.src0RepStride = rowStrideBlocks;

    Brcb(broadcast, scales, (numRows + BRCB_SRC_PER_REPEAT - 1U) /
        BRCB_SRC_PER_REPEAT, {1, BRCB_SRC_PER_REPEAT});
    PipeBarrier<PIPE_V>();
    for (uint32_t columnLoop = 0U; columnLoop < columnLoops; ++columnLoop) {
        const uint32_t columnOffset = columnLoop * FP16_REPEAT_ELEMS;
        uint32_t columnCount = headDim - columnOffset;
        if (columnCount > FP16_REPEAT_ELEMS) {
            columnCount = FP16_REPEAT_ELEMS;
        }
        Mul(dst[columnOffset], src[columnOffset], broadcast, columnCount,
            numRows, repeatParams);
    }
    Brcb(broadcast, offsets, (numRows + BRCB_SRC_PER_REPEAT - 1U) /
        BRCB_SRC_PER_REPEAT, {1, BRCB_SRC_PER_REPEAT});
    PipeBarrier<PIPE_V>();
    for (uint32_t columnLoop = 0U; columnLoop < columnLoops; ++columnLoop) {
        const uint32_t columnOffset = columnLoop * FP16_REPEAT_ELEMS;
        uint32_t columnCount = headDim - columnOffset;
        if (columnCount > FP16_REPEAT_ELEMS) {
            columnCount = FP16_REPEAT_ELEMS;
        }
        Add(dst[columnOffset], dst[columnOffset], broadcast, columnCount,
            numRows, repeatParams);
    }
    PipeBarrier<PIPE_V>();
}

// Copy nb meta scalars into a 32B-aligned half buffer for Brcb (half block=16).
// BOTH src and dst must be 32B-aligned — Adds faults on unaligned half src
// (e.g. metaHalf[j0] with j0%16!=0). Prefer tile starts at multiples of 16 and
// pass metaHalf[j0] directly instead of staging.
__aicore__ inline void TqStageMetaHalfAligned(LocalTensor<half> dstAlign,
    LocalTensor<half> src, uint32_t nb)
{
    Adds(dstAlign, src, static_cast<half>(0.0f), nb);
    PipeBarrier<PIPE_V>();
}

// P17b-B Value fold on native half meta: vmin' = vmin + 8*vstep (n rows).
// foldTmp must be a disjoint half buffer of at least n elements.
// vmin/vstep bases are the packed SoA runs (offset 0 → 32B-aligned).
__aicore__ inline void TqFoldValueMetaHalf(LocalTensor<half> vmin,
    LocalTensor<half> vstep, LocalTensor<half> foldTmp, uint32_t n)
{
    Muls(foldTmp, vstep, static_cast<half>(8.0f), n);
    Add(vmin, vmin, foldTmp, n);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void TqCopyMetaPair(GlobalTensor<uint8_t> srcGm, LocalTensor<uint8_t> ub,
    uint64_t meta0Off, uint64_t meta1Off)
{
    // Both half and bfloat16 metadata are 2 bytes.
    DataCopyExtParams metaParams{1, sizeof(half), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    DataCopyPad(ub[BR_META0_UB_OFF], srcGm[meta0Off], metaParams, padParams);
    DataCopyPad(ub[BR_META1_UB_OFF], srcGm[meta1Off], metaParams, padParams);
}

// fp32 affine (bf16 OutT / Scheme C legacy): Brcb needs j0%8==0; keep prior
// tile widths so tmpBuff1 scratch stays smaller and maxSub is not crushed.
static constexpr uint32_t BR_KEY_DECODE_TILE_MAX = 8U;
static constexpr uint32_t BR_VALUE_DECODE_TILE_MAX = 13U;
// Half-affine: Brcb needs 32B-aligned meta src (16×half). Tile starts must be
// multiples of 16 so metaHalf[j0] feeds Brcb directly (Adds/Cast also fault on
// unaligned half src). Do NOT reuse these for bf16 — doubles fp32 scratch.
static constexpr uint32_t BR_KEY_DECODE_TILE_MAX_HALF = 64U;
static constexpr uint32_t BR_VALUE_DECODE_TILE_MAX_HALF = 64U;
// Brcb writes 1 FP16 block (16 half) per row. Footprint = tile_rows × 16, not
// tile×headDim. Scratch in inputQue1 (64KB); tile 64 needs ~18KB scratch.
static constexpr uint32_t BR_HALF_BRCB_FOOTPRINT =
    BR_KEY_DECODE_TILE_MAX_HALF * 16U;  // 1024 half = 2048B
// Cap PA run length to BS=128 divisor (128/32=4 equal runs, no tail).
static constexpr uint32_t BR_HALF_PA_RUN_CAP = 32U;

__aicore__ inline uint32_t TqAlignUp32(uint32_t x)
{
    return (x + 31U) & ~31U;
}

#if BR_KEY_UNIFORM_SCHEME == 3
// Scheme C (legacy): y = sign * (base + q7 * step), code = (q7 << 1) | sign.
// halfScratch / scratchA/B/C: each numRows * headDim (3×fp32 + half).
template <typename OutT>
__aicore__ inline void TqDecodeKeyTile(
    LocalTensor<uint8_t> codesUb,
    LocalTensor<half> halfScratch,
    LocalTensor<float> scratchA,
    LocalTensor<float> scratchB,
    LocalTensor<float> scratchC,
    LocalTensor<OutT> outUb,
    LocalTensor<float> bases,
    LocalTensor<float> steps,
    uint32_t numRows,
    uint32_t headDim,
    uint32_t headDimAlign)
{
    const uint32_t N = numRows * headDim;

    // P13/P13b LSB-sign: And→ShiftRight and ShiftRight→Cast(sign) are
    // disjoint-UB; Duplicate→And and ShiftRight→Cast(q7) stay RAW-protected.
    Cast(halfScratch, codesUb, RoundMode::CAST_NONE, N);
    auto codeI16 = scratchA.template ReinterpretCast<int16_t>();
    Cast(codeI16, halfScratch, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();

    auto codeU16 = codeI16.template ReinterpretCast<uint16_t>();
    auto signStorage = scratchC.template ReinterpretCast<int16_t>();
    auto signStorageU16 = signStorage.template ReinterpretCast<uint16_t>();

    auto signMaskU16 = halfScratch.template ReinterpretCast<uint16_t>();
    Duplicate(signMaskU16, static_cast<uint16_t>(0x01), N);
    PipeBarrier<PIPE_V>();
    And(signStorageU16, codeU16, signMaskU16, N);

    ShiftRight(codeU16, codeU16, static_cast<uint16_t>(1), N);

    Cast(scratchB, signStorage, RoundMode::CAST_NONE, N);
    Muls(scratchB, scratchB, -2.0f, N);
    Adds(scratchB, scratchB, 1.0f, N);
    PipeBarrier<PIPE_V>();

    Cast(scratchC, codeI16, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    auto metaBroadcast = halfScratch.template ReinterpretCast<float>();
    TqApplyRowAffine(
        scratchA, scratchC, bases, steps, metaBroadcast, numRows, headDim);

    Mul(scratchA, scratchA, scratchB, N);
    PipeBarrier<PIPE_V>();
    if (headDimAlign == headDim) {
        Cast(outUb, scratchA, RoundMode::CAST_RINT, N);
        PipeBarrier<PIPE_V>();
    } else {
        for (uint32_t row = 0U; row < numRows; ++row) {
            Cast(outUb[row * headDimAlign], scratchA[row * headDim], RoundMode::CAST_RINT, headDim);
            PipeBarrier<PIPE_V>();
        }
    }
}
#else
// Scheme A/B (uniform): code stores q in [0,255] as uint8.
// A: y = base + q * step
// B: y = s * (q - 127.5) via meta0=-127.5*s, meta1=s
// halfScratch / scratchA/B: each numRows * headDim (2×fp32 + half).
// bf16 OutT: fp32 affine + Cast→OutT (avoid half→fp32→bf16).
template <typename OutT>
__aicore__ inline void TqDecodeKeyTile(
    LocalTensor<uint8_t> codesUb,
    LocalTensor<half> halfScratch,
    LocalTensor<float> scratchA,
    LocalTensor<float> scratchB,
    LocalTensor<OutT> outUb,
    LocalTensor<float> bases,
    LocalTensor<float> steps,
    uint32_t numRows,
    uint32_t headDim,
    uint32_t headDimAlign)
{
    const uint32_t N = numRows * headDim;

    // 910B: no float←uint8; widen via half then fp32 (q as float).
#if BR_KEY_UNIFORM_SCHEME == 1
    // Scheme A: pack stores signed int8 bytes (code_i8=q-128) in the same
    // byte slots. Reinterpret the underlying bytes as int8 before widening.
    auto codesI8 = codesUb.template ReinterpretCast<int8_t>();
    Cast(halfScratch, codesI8, RoundMode::CAST_NONE, N);
#else
    Cast(halfScratch, codesUb, RoundMode::CAST_NONE, N);
#endif
    Cast(scratchA, halfScratch, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();

    auto metaBroadcast = halfScratch.template ReinterpretCast<float>();
    TqApplyRowAffine(
        scratchB, scratchA, bases, steps, metaBroadcast, numRows, headDim);

    if (headDimAlign == headDim) {
        Cast(outUb, scratchB, RoundMode::CAST_RINT, N);
        PipeBarrier<PIPE_V>();
    } else {
        for (uint32_t row = 0U; row < numRows; ++row) {
            Cast(outUb[row * headDimAlign], scratchB[row * headDim], RoundMode::CAST_RINT, headDim);
            PipeBarrier<PIPE_V>();
        }
    }
}

#endif  // BR_KEY_UNIFORM_SCHEME == 3

// fp16 OutT Key path (Scheme A/B only at call sites). Always declared so
// if constexpr name lookup succeeds under Scheme C builds.
__aicore__ inline void TqDecodeKeyTileHalf(
    LocalTensor<uint8_t> codesUb,
    LocalTensor<half> halfSrc,
    LocalTensor<half> halfBroadcast,
    LocalTensor<half> outUb,
    LocalTensor<half> bases,
    LocalTensor<half> steps,
    uint32_t numRows,
    uint32_t headDim,
    uint32_t headDimAlign)
{
    (void)headDimAlign;
    const uint32_t N = numRows * headDim;
#if BR_KEY_UNIFORM_SCHEME == 1
    // Same contract as TqDecodeKeyTile: signed int8 bytes + folded base'.
    // fp16 decode is the hot path (kUseHalfAffine); uint8 Cast would treat
    // negative codes as 128..255 and drift by 256*step.
    auto codesI8 = codesUb.template ReinterpretCast<int8_t>();
    Cast(halfSrc, codesI8, RoundMode::CAST_NONE, N);
#else
    Cast(halfSrc, codesUb, RoundMode::CAST_NONE, N);
#endif
    // P2b: no PipeBarrier before Brcb — Cast→halfSrc and Brcb→broadcast are
    // disjoint; TqApplyRowAffine barriers before Mul drain both.
    TqApplyRowAffine(
        outUb, halfSrc, bases, steps, halfBroadcast, numRows, headDim);
}

// Value tile: y = vmin' + s*vstep, s = Cast(int4) in [-8,7].// Caller must pre-fold vmin' = vmin + 8*vstep (P17b-B: once per PA run).
// nibbleUb: contiguous numRows * (headDim/2) uint8 (int4 packed).
// halfScratch / scratchA/B: each numRows * headDim elements.
template <typename OutT>
__aicore__ inline void TqDecodeValueTile(
    LocalTensor<uint8_t> nibbleUb,
    LocalTensor<half> halfScratch,
    LocalTensor<float> scratchA,
    LocalTensor<float> scratchB,
    LocalTensor<OutT> outUb,
    LocalTensor<float> vmins,
    LocalTensor<float> vsteps,
    uint32_t numRows,
    uint32_t headDim,
    uint32_t headDimAlign)
{
    const uint32_t N = numRows * headDim;

    // Signed int4 → fp32; +8 already absorbed into vmin' by the caller.
    Cast(halfScratch, nibbleUb.template ReinterpretCast<int4b_t>(), RoundMode::CAST_NONE, N);
    Cast(scratchA, halfScratch, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();

    auto metaBroadcast = halfScratch.template ReinterpretCast<float>();
    TqApplyRowAffine(
        scratchB, scratchA, vmins, vsteps, metaBroadcast, numRows, headDim);

    if (headDimAlign == headDim) {
        Cast(outUb, scratchB, RoundMode::CAST_RINT, N);
        PipeBarrier<PIPE_V>();
    } else {
        for (uint32_t row = 0U; row < numRows; ++row) {
            Cast(outUb[row * headDimAlign], scratchB[row * headDim], RoundMode::CAST_RINT, headDim);
            PipeBarrier<PIPE_V>();
        }
    }
}

// fp16 OutT value path (mirror TqDecodeKeyTileHalf).
__aicore__ inline void TqDecodeValueTileHalf(
    LocalTensor<uint8_t> nibbleUb,
    LocalTensor<half> halfSrc,
    LocalTensor<half> halfBroadcast,
    LocalTensor<half> outUb,
    LocalTensor<half> vmins,
    LocalTensor<half> vsteps,
    uint32_t numRows,
    uint32_t headDim,
    uint32_t headDimAlign)
{
    (void)headDimAlign;
    const uint32_t N = numRows * headDim;
    Cast(halfSrc, nibbleUb.template ReinterpretCast<int4b_t>(), RoundMode::CAST_NONE, N);
    // P2b: same as Key half — Cast/Brcb disjoint; barrier inside affine.
    TqApplyRowAffine(
        outUb, halfSrc, vmins, vsteps, halfBroadcast, numRows, headDim);
}

}  // namespace tq_dequant

#endif
