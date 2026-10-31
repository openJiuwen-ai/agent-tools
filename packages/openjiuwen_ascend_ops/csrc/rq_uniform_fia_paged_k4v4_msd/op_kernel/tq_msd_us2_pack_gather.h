/**
 * Us2 P0-op step 4: in-kernel BR pack gather (K4V4).
 *
 * Pack-side contract (rq_uniform_pack_k4v4, Scheme A):
 *   K/V codes: signed int4 [-8, 7]; pack does NOT fold base/vmin
 *              (readers fold base' = base + 8*step)
 *
 * Gather responsibilities (decode-side):
 *   1. K codes: bmm1CopyB1 direct read (gCol=64, KVINT4)
 *   2. K/V meta: pack-tail gather + Fold +8*scale for both
 *   3. V codes: Cube BMM2 native int4 via bmm2CopyB1
 */
#ifndef BR_MSD_US2_PACK_GATHER_H
#define BR_MSD_US2_PACK_GATHER_H

#ifndef BR_MSD_US2_PACK_GATHER
#define BR_MSD_US2_PACK_GATHER 1
#endif
// Haar Q@Π / O@Π. Product default ON (matches CMake / rebuild_op.sh).
#ifndef BR_MSD_US2_APPLY_PI
#define BR_MSD_US2_APPLY_PI 1
#endif
// 1: pack-tail meta MTE2↔V ping-pong when nRows>=256 (product default).
// 0: A/B — forced single-buffer gather.
#ifndef BR_MSD_US2_META_PINGPONG
#define BR_MSD_US2_META_PINGPONG 1
#endif

#include "vendored/arch32/tq_pack_layout.h"
#include "vendored/us2/ifa_public_define.h"

namespace tq_msd_us2 {
using namespace AscendC;
using namespace tq_pack;

static constexpr uint32_t kFp32BlockElem = 8U;    // 32B / sizeof(float)
static constexpr uint32_t kFp32RepeatElem = 64U;  // 256B / sizeof(float)
static constexpr uint32_t kMetaPingRows = 128U;
static constexpr uint32_t kMetaMaxSeg = 16U;

// Pack stores unfolded vmin; Cube/combine consume signed int4 s in [-8,7].
// vmin' = vmin + 8*vstep so y = vmin' + s*vstep ≡ vmin + idx[0,15]*vstep (BRFIA).
// calCount is true n or 8-aligned n (pad floats come from DataCopyPad→Cast, not SetValue).
__aicore__ inline void FoldValMetaVminForInt4Fp32(LocalTensor<float> vmin, LocalTensor<float> vstep,
                                                 LocalTensor<float> tmp, uint32_t n)
{
    Muls(tmp, vstep, 8.0f, n);
    PipeBarrier<PIPE_V>();
    Add(vmin, vmin, tmp, n);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline uint32_t AlignFp32VecCount(uint32_t n)
{
    return (n + (kFp32BlockElem - 1U)) & ~(kFp32BlockElem - 1U);
}

__aicore__ inline uint32_t LoadBlockId(__gm__ int32_t *blockTable, uint32_t bIdx,
                                       uint32_t maxBlockNumPerBatch, uint64_t seqIdx,
                                       uint32_t blockSize)
{
    const uint64_t blockIdOffset = seqIdx / static_cast<uint64_t>(blockSize);
    return static_cast<uint32_t>(
        *(blockTable + static_cast<uint64_t>(bIdx) * maxBlockNumPerBatch + blockIdOffset));
}

__aicore__ inline void CopyU8Aligned(LocalTensor<uint8_t> dst, GlobalTensor<uint8_t> src,
                                     uint64_t srcOff, uint32_t bytes)
{
    DataCopyExtParams params{1, bytes, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    DataCopyPad(dst, src[srcOff], params, pad);
}

// GM→UB by true byte length; DataCopyPad auto-pads short tails (BRFIA TqCopyPackedMetaTile).
// META_T = pack tail meta dtype, follows query dtype (half / bfloat16_t).
template <typename META_T>
__aicore__ inline void CopyMetaHalfRun(LocalTensor<META_T> dst, GlobalTensor<uint8_t> src,
                                       uint64_t srcOff, uint32_t numRows)
{
    const uint32_t bytes = numRows * static_cast<uint32_t>(sizeof(META_T));
    DataCopyExtParams params{1, bytes, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad{true, 0, 0, 0};
    DataCopyPad(dst.template ReinterpretCast<uint8_t>(), src[srcOff], params, pad);
}

// Per-token pack meta -> fp32 UB [nRows]. Key: no fold. Value: vmin' = vmin + 8*vstep.
// Page-run DataCopy into ping-pong scratch so Cast of page i overlaps MTE2 of i+1.
// Caller's meta0H/meta1H must hold two 128-row slots when nRows>=256.
template <typename META_T>
__aicore__ inline void CopyMetaHalfPairRun(LocalTensor<META_T> dst0H, LocalTensor<META_T> dst1H,
                                           GlobalTensor<uint8_t> src, uint64_t off0, uint64_t off1,
                                           uint32_t numRows)
{
    CopyMetaHalfRun(dst0H, src, off0, numRows);
    CopyMetaHalfRun(dst1H, src, off1, numRows);
}

template <typename META_T>
__aicore__ inline void GatherBrPackMetaFp32(LocalTensor<float> dst0, LocalTensor<float> dst1,
                                            GlobalTensor<uint8_t> packGm, __gm__ int32_t *blockTable,
                                            uint32_t bIdx, uint32_t n2Idx, uint32_t kvHeadNum,
                                            uint32_t maxBlockNumPerBatch, uint32_t blockSize,
                                            uint64_t seq0, uint32_t nRows, bool isValue,
                                            LocalTensor<META_T> meta0H, LocalTensor<META_T> meta1H,
                                            LocalTensor<float> foldTmp)
{
    uint32_t segRow[kMetaMaxSeg];
    uint32_t segRun[kMetaMaxSeg];
    uint64_t segOff0[kMetaMaxSeg];
    uint64_t segOff1[kMetaMaxSeg];
    uint32_t nseg = 0;
    uint32_t row = 0;
    while (row < nRows && nseg < kMetaMaxSeg) {
        const uint64_t seq = seq0 + row;
        const uint32_t posInBlock = static_cast<uint32_t>(seq % static_cast<uint64_t>(blockSize));
        uint32_t run = blockSize - posInBlock;
        if (run > nRows - row) {
            run = nRows - row;
        }
        const uint32_t phys = LoadBlockId(blockTable, bIdx, maxBlockNumPerBatch, seq, blockSize);
        const uint64_t head = isValue ? TqValHeadBase(phys, n2Idx, kvHeadNum, blockSize)
                                      : TqKeyHeadBase(phys, n2Idx, kvHeadNum, blockSize);
        segRow[nseg] = row;
        segRun[nseg] = run;
        segOff0[nseg] = isValue ? TqValVminOffset(head, blockSize, posInBlock)
                                : TqKeyBaseOffset(head, blockSize, posInBlock);
        segOff1[nseg] = isValue ? TqValVstepOffset(head, blockSize, posInBlock)
                                : TqKeyStepOffset(head, blockSize, posInBlock);
        nseg++;
        row += run;
    }
    if (nseg == 0U) {
        return;
    }

#if BR_MSD_US2_META_PINGPONG
    const bool pingpong = (nRows >= (kMetaPingRows * 2U));
#else
    const bool pingpong = false;
#endif
    event_t evM2V[2];
    event_t evVM2[2];
    evM2V[0] = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    evM2V[1] = pingpong ? static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V)) : evM2V[0];
    evVM2[0] = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    evVM2[1] = pingpong ? static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)) : evVM2[0];
    bool slotLive[2] = {false, false};

    {
        CopyMetaHalfPairRun(meta0H, meta1H, packGm, segOff0[0], segOff1[0], segRun[0]);
        SetFlag<HardEvent::MTE2_V>(evM2V[0]);
    }
    for (uint32_t i = 0; i < nseg; ++i) {
        const uint32_t sl = pingpong ? (i & 1U) : 0U;
        WaitFlag<HardEvent::MTE2_V>(evM2V[sl]);
        if (pingpong && (i + 1U < nseg)) {
            const uint32_t nsl = (i + 1U) & 1U;
            if (slotLive[nsl]) {
                WaitFlag<HardEvent::V_MTE2>(evVM2[nsl]);
                slotLive[nsl] = false;
            }
            const uint32_t nOff = nsl * kMetaPingRows;
            CopyMetaHalfPairRun(meta0H[nOff], meta1H[nOff], packGm, segOff0[i + 1U], segOff1[i + 1U],
                                segRun[i + 1U]);
            SetFlag<HardEvent::MTE2_V>(evM2V[nsl]);
        }
        const uint32_t off = pingpong ? (sl * kMetaPingRows) : 0U;
        // DataCopyPad already zero-pads half UB to 32B; Cast 8-aligned count
        // so float base/step tails are 0 without SetValue/Duplicate (BRFIA style).
        const uint32_t runAlign = AlignFp32VecCount(segRun[i]);
        Cast(dst0[segRow[i]], meta0H[off], RoundMode::CAST_NONE, runAlign);
        Cast(dst1[segRow[i]], meta1H[off], RoundMode::CAST_NONE, runAlign);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(evVM2[sl]);
        slotLive[sl] = true;
        if (!pingpong && (i + 1U < nseg)) {
            WaitFlag<HardEvent::V_MTE2>(evVM2[0]);
            slotLive[0] = false;
            CopyMetaHalfPairRun(meta0H, meta1H, packGm, segOff0[i + 1U], segOff1[i + 1U], segRun[i + 1U]);
            SetFlag<HardEvent::MTE2_V>(evM2V[0]);
        }
    }
    for (uint32_t sl = 0; sl < (pingpong ? 2U : 1U); ++sl) {
        if (slotLive[sl]) {
            WaitFlag<HardEvent::V_MTE2>(evVM2[sl]);
        }
    }
    // K4V4: Key and Value both leave base/vmin unfolded at pack; always fold +8*scale.
    FoldValMetaVminForInt4Fp32(dst0, dst1, foldTmp, AlignFp32VecCount(nRows));
}

/**
 * score[r,s] = base[s]*qSum[r] + step[s]*score[r,s]  (qSum is Brcb [dealRow, 8]).
 *
 * BRFIA-style: expand base into tmp via Local DataCopy, then Mul/Add with
 * BinaryRepeatParams. For short S2, DataCopy length is AlignFp32VecCount
 * (32B); pad floats are zeros from DataCopyPad→Cast (no SetValue/Duplicate).
 */
__aicore__ inline void SchemeAAffineBrcb(LocalTensor<float> scores, LocalTensor<float> base,
                                         LocalTensor<float> step, LocalTensor<float> qSumBrcb,
                                         LocalTensor<float> aMaxBrcb, LocalTensor<float> tmp,
                                         uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount,
                                         bool foldAMax)
{
    if (foldAMax) {
        RowMuls(scores, scores, aMaxBrcb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
    }
    BinaryRepeatParams rp;
    rp.dstBlkStride = 1;
    rp.src0BlkStride = 1;
    rp.src1BlkStride = 0;
    rp.dstRepStride = columnCount / kFp32BlockElem;
    rp.src0RepStride = 0;
    rp.src1RepStride = 1;
    uint32_t dLoop = actualColumnCount / kFp32RepeatElem;
    uint32_t dRemain = actualColumnCount % kFp32RepeatElem;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; ++i) {
        Mul(tmp[offset], base[offset], qSumBrcb, kFp32RepeatElem, dealRowCount, rp);
        offset += kFp32RepeatElem;
    }
    if (dRemain > 0U) {
        Mul(tmp[offset], base[offset], qSumBrcb, dRemain, dealRowCount, rp);
    }
    PipeBarrier<PIPE_V>();
    rp.src0BlkStride = 1;
    rp.src1BlkStride = 1;
    rp.src0RepStride = 0;
    rp.src1RepStride = columnCount / kFp32BlockElem;
    rp.dstRepStride = columnCount / kFp32BlockElem;
    offset = 0;
    dLoop = actualColumnCount / kFp32RepeatElem;
    dRemain = actualColumnCount % kFp32RepeatElem;
    for (uint32_t i = 0; i < dLoop; ++i) {
        FusedMulAdd(scores[offset], step[offset], tmp[offset], kFp32RepeatElem, dealRowCount, rp);
        offset += kFp32RepeatElem;
    }
    if (dRemain > 0U) {
        FusedMulAdd(scores[offset], step[offset], tmp[offset], dRemain, dealRowCount, rp);
    }
    PipeBarrier<PIPE_V>();
}

/** bmm2Res[r, :] += rowSum[r]  (rowSum Brcb [dealRow, 8]). */
__aicore__ inline void AddRowSumBroadcast(LocalTensor<float> mat, LocalTensor<float> rowSumBrcb,
                                          uint32_t dealRowCount, uint32_t columnCount,
                                          uint32_t actualColumnCount)
{
    BinaryRepeatParams rp;
    rp.dstBlkStride = 1;
    rp.src0BlkStride = 1;
    rp.src1BlkStride = 0;
    rp.dstRepStride = columnCount / kFp32BlockElem;
    rp.src0RepStride = columnCount / kFp32BlockElem;
    rp.src1RepStride = 1;
    uint32_t dLoop = actualColumnCount / kFp32RepeatElem;
    uint32_t dRemain = actualColumnCount % kFp32RepeatElem;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; ++i) {
        Add(mat[offset], mat[offset], rowSumBrcb, kFp32RepeatElem, dealRowCount, rp);
        offset += kFp32RepeatElem;
    }
    if (dRemain > 0U) {
        Add(mat[offset], mat[offset], rowSumBrcb, dRemain, dealRowCount, rp);
    }
    PipeBarrier<PIPE_V>();
}

}  // namespace tq_msd_us2

#endif
