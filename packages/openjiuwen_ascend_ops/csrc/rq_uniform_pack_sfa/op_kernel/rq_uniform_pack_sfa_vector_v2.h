/**
 * rq_uniform_pack_sfa V2 — EncodeValueBatch (k4v4) half quant + int4 nibble.
 *
 * Same steps as rq_uniform_pack_k4v4_aiv.h EncodeValueBatch:
 *   WholeReduceMax/Min → step=(max-min)/15 → Brcb → Sub/Div → Adds(-8) → Cast(int4)
 * D=512 (k4v4 D=128): one vector repeat is 128 halves, so reduce/Sub/Div run
 * 4 chunks and the row stride is 32 blocks instead of 8.
 * Codes stay signed int4 ([-8, 7]). The optimized attention op adds +8 at dequant;
 * this pack does not map nibbles back to unsigned.
 */
#ifndef RQ_UNIFORM_PACK_SFA_VECTOR_V2_H
#define RQ_UNIFORM_PACK_SFA_VECTOR_V2_H

#include "kernel_operator.h"
#include "rq_uniform_pack_sfa_common.h"

using namespace AscendC;
using namespace rq_uniform_sfa;

namespace rq_uniform_pack_sfa_v2 {

constexpr uint32_t kTile = TILE_ROWS;                  // 32; y_/fat_ stay single-buffered
constexpr uint32_t kHalfPerBlock = 16;                 // 32B / sizeof(half)
constexpr uint32_t kHalfMask = 128;                    // EncodeValueBatch binaryRepeatBatchF16
constexpr uint32_t kChunks = NOPE_DIM / kHalfMask;     // 512/128 = 4
constexpr uint32_t kRowBlocks = NOPE_DIM / kHalfPerBlock;  // 32 (k4v4 D=128 → 8)
constexpr uint32_t kScalarTile = kTile * kHalfPerBlock;    // one 32B Brcb block per row
// Dual nope/rope/slots/rows + single y/fat. 910B UB is 192KB; this is 178944B.
constexpr uint32_t kUbBytes = 2U * kTile * NOPE_DIM * sizeof(half) + 2U * kTile * ROPE_DIM * sizeof(half) +
    2U * kTile * sizeof(int64_t) + 2U * kTile * UB_ROW_ALIGN + 2U * kTile * NOPE_DIM * sizeof(half) +
    kChunks * kTile * sizeof(half) + 4U * kScalarTile * sizeof(half) + kTile * CODE_BYTES;
static_assert(kUbBytes <= 180U * 1024U, "tile 32 exceeds the 910B UB budget");
constexpr half kInvLevels = static_cast<half>(1.0f / 15.0f);
constexpr half kEps = static_cast<half>(1.0e-6f);

__aicore__ inline void SyncMte2ToV()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}

__aicore__ inline void SyncVToS()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
}

__aicore__ inline void SyncSToV()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(e);
    WaitFlag<HardEvent::S_V>(e);
}

__aicore__ inline void SyncVToMte3()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);
}

__aicore__ inline void SyncSToMte3()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e);
    WaitFlag<HardEvent::S_MTE3>(e);
}

__aicore__ inline void SyncMte3ToV()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e);
    WaitFlag<HardEvent::MTE3_V>(e);
}

__aicore__ inline void SyncMte3ToS()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(e);
    WaitFlag<HardEvent::MTE3_S>(e);
}

__aicore__ inline void SyncMte3ToMte2()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e);
    WaitFlag<HardEvent::MTE3_MTE2>(e);
}

__aicore__ inline void CopyBytesToGm(
    GlobalTensor<uint8_t>& dst, uint64_t offset, LocalTensor<uint8_t>& src, uint32_t nbytes)
{
    SyncVToMte3();
    SyncSToMte3();
    DataCopyExtParams params{1, nbytes, 0, 0, 0};
    DataCopyPad(dst[offset], src, params);
    SyncMte3ToV();
    SyncMte3ToS();
    SyncMte3ToMte2();
}

template <typename T>
class VectorPackSfaV2 {
public:
    __aicore__ inline VectorPackSfaV2(
        GlobalTensor<T> nopeGm,
        GlobalTensor<T> ropeGm,
        GlobalTensor<int64_t> slotsGm,
        GlobalTensor<uint8_t> cacheGm,
        uint32_t numTokens,
        uint32_t blockSize,
        uint32_t numBlocks,
        uint32_t /*tileRows*/)
        : nopeGm_(nopeGm),
          ropeGm_(ropeGm),
          slotsGm_(slotsGm),
          cacheGm_(cacheGm),
          numTokens_(numTokens),
          blockSize_(blockSize),
          numBlocks_(numBlocks)
    {
        InitUb();
    }

    __aicore__ inline void Process(uint32_t core, uint32_t dataCores)
    {
        const event_t ev0 = static_cast<event_t>(0);
        const event_t ev1 = static_cast<event_t>(1);
        SetFlag<HardEvent::MTE3_MTE2>(ev0);
        SetFlag<HardEvent::MTE3_MTE2>(ev1);
        SetFlag<HardEvent::MTE3_V>(ev0);
        SetFlag<HardEvent::MTE3_V>(ev1);

        // Contiguous split: core c owns [c*T/N, (c+1)*T/N). Tiles of 32, tail uses the real row count.
        const uint32_t tokStart = static_cast<uint32_t>(
            (static_cast<uint64_t>(core) * numTokens_) / dataCores);
        const uint32_t tokEnd = static_cast<uint32_t>(
            (static_cast<uint64_t>(core + 1) * numTokens_) / dataCores);

        uint32_t phase = 0;
        for (uint32_t base = tokStart; base < tokEnd;) {
            const uint32_t remain = tokEnd - base;
            const uint32_t m = remain < kTile ? remain : kTile;
            const event_t ev = phase == 0 ? ev0 : ev1;
            WaitFlag<HardEvent::MTE3_MTE2>(ev);
            Load(phase, base, m);
            SetFlag<HardEvent::MTE2_V>(ev);
            SetFlag<HardEvent::MTE2_S>(ev);
            WaitFlag<HardEvent::MTE2_V>(ev);
            Quant(phase, m);
            WaitFlag<HardEvent::MTE2_S>(ev);
            WaitFlag<HardEvent::MTE3_V>(ev);
            Assemble(phase, m);
            SetFlag<HardEvent::V_MTE3>(ev);
            SetFlag<HardEvent::S_MTE3>(ev);
            WaitFlag<HardEvent::V_MTE3>(ev);
            WaitFlag<HardEvent::S_MTE3>(ev);
            Scatter(phase, m);
            SetFlag<HardEvent::MTE3_V>(ev);
            SetFlag<HardEvent::MTE3_MTE2>(ev);
            base += m;
            phase ^= 1U;
        }

        WaitFlag<HardEvent::MTE3_MTE2>(ev0);
        WaitFlag<HardEvent::MTE3_MTE2>(ev1);
        WaitFlag<HardEvent::MTE3_V>(ev0);
        WaitFlag<HardEvent::MTE3_V>(ev1);
    }

private:
    __aicore__ inline void InitUb()
    {
        for (uint32_t p = 0; p < 2; ++p) {
            pipe_.InitBuffer(bufNope_[p], kTile * NOPE_DIM * sizeof(T));
            pipe_.InitBuffer(bufRope_[p], kTile * ROPE_DIM * sizeof(T));
            pipe_.InitBuffer(bufSlots_[p], kTile * sizeof(int64_t));
            pipe_.InitBuffer(bufRows_[p], kTile * UB_ROW_ALIGN);
            nope_[p] = bufNope_[p].Get<T>();
            rope_[p] = bufRope_[p].Get<T>();
            slots_[p] = bufSlots_[p].Get<int64_t>();
            rows_[p] = bufRows_[p].Get<uint8_t>();
        }
        pipe_.InitBuffer(bufY_, kTile * NOPE_DIM * sizeof(half));
        pipe_.InitBuffer(bufFat_, kTile * NOPE_DIM * sizeof(half));
        pipe_.InitBuffer(bufPartial_, kChunks * kTile * sizeof(half));
        pipe_.InitBuffer(bufScal_, 4 * kScalarTile * sizeof(half));
        pipe_.InitBuffer(bufCodes_, kTile * CODE_BYTES);

        y_ = bufY_.Get<half>();
        fat_ = bufFat_.Get<half>();
        partial_ = bufPartial_.Get<half>();
        scal_ = bufScal_.Get<half>();
        codes_ = bufCodes_.Get<uint8_t>();
    }

    // k4v4 reduces D=128 in one WholeReduce*. D=512 is four 128-wide chunks.
    // ORDER_ONLY_VALUE + dstRepStride=1 packs one half per row. Fold the four
    // partials with Max/Min; no scalar GetValue. Repeat count is the live rows.
    // DataCopy still moves one 32B block; the scalar tail is zero so Brcb does not
    // read a stale slot. Sub/Div/Cast below use m, not the padded block.
    __aicore__ inline void ReduceChunks(
        LocalTensor<half>& dstRow, LocalTensor<half>& src, bool isMax, uint32_t m)
    {
        if (m < kTile) {
            Duplicate(partial_, static_cast<half>(0.0f), kChunks * kTile);
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t c = 0; c < kChunks; ++c) {
            auto dst = partial_[c * kTile];
            auto chunk = src[c * kHalfMask];
            if (isMax) {
                WholeReduceMax<half, false>(
                    dst, chunk, static_cast<int32_t>(kHalfMask), static_cast<int32_t>(m), 1, 1,
                    static_cast<int32_t>(kRowBlocks), ReduceOrder::ORDER_ONLY_VALUE);
            } else {
                WholeReduceMin<half, false>(
                    dst, chunk, static_cast<int32_t>(kHalfMask), static_cast<int32_t>(m), 1, 1,
                    static_cast<int32_t>(kRowBlocks), ReduceOrder::ORDER_ONLY_VALUE);
            }
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t c = 1; c < kChunks; ++c) {
            if (isMax) {
                Max(partial_, partial_, partial_[c * kTile], m);
            } else {
                Min(partial_, partial_, partial_[c * kTile], m);
            }
            PipeBarrier<PIPE_V>();
        }
        DataCopy(dstRow, partial_, kTile);
        PipeBarrier<PIPE_V>();
    }

    // Port of EncodeValueBatch. yBatch is half [m, 512], m <= kTile.
    __aicore__ inline void EncodeValueBatch(LocalTensor<half>& yBatch, uint32_t m)
    {
        const uint32_t brcbRepeats = (m + 7U) / 8U;
        const uint32_t totalElems = m * NOPE_DIM;

        auto sMaxVec = scal_;
        auto minVec = scal_[kScalarTile];
        auto stepVec = scal_[2 * kScalarTile];
        auto stepBlk = scal_[3 * kScalarTile];
        auto minBlk = sMaxVec;

        auto origBatch16 = yBatch;

        // Step 2 / 3: per-row max, min. (k4v4: one reduce of D=128.)
        ReduceChunks(sMaxVec, origBatch16, true, m);
        ReduceChunks(minVec, origBatch16, false, m);

        // Step 4: step = max((max-min)/15, eps). max dies after Sub.
        Sub(stepVec, sMaxVec, minVec, m);
        PipeBarrier<PIPE_V>();
        Muls(stepVec, stepVec, kInvLevels, m);
        PipeBarrier<PIPE_V>();
        Maxs(stepVec, stepVec, kEps, m);
        PipeBarrier<PIPE_V>();

        // Step 5: Brcb min / step. Each repeat: 8 scalars → 8 blocks.
        Brcb(minBlk, minVec, brcbRepeats, BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();
        Brcb(stepBlk, stepVec, brcbRepeats, BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();

        // Step 6 / 7: (y - min) / step. Row stride is 32 blocks (D=512), mask stays 128.
        BinaryRepeatParams rp(1, 1, 0, static_cast<uint8_t>(kRowBlocks), static_cast<uint8_t>(kRowBlocks), 1);
        for (uint32_t c = 0; c < kChunks; ++c) {
            Sub<half, false>(
                fat_[c * kHalfMask], origBatch16[c * kHalfMask], minBlk, kHalfMask, m, rp);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < kChunks; ++c) {
            Div<half, false>(
                fat_[c * kHalfMask], fat_[c * kHalfMask], stepBlk, kHalfMask, m, rp);
        }
        PipeBarrier<PIPE_V>();

        // Step 8 / 9: [0,15] → [-8,7], Cast int4 (same as k4v4 value).
        Adds(fat_, fat_, static_cast<half>(-8.0f), totalElems);
        PipeBarrier<PIPE_V>();
        auto quantI4 = codes_.template ReinterpretCast<int4b_t>();
        Cast(quantI4, fat_, RoundMode::CAST_NONE, totalElems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Load(uint32_t phase, uint32_t tokenBase, uint32_t m)
    {
        DataCopyExtParams nopeParams{
            1, static_cast<uint32_t>(m * NOPE_DIM * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(nope_[phase], nopeGm_[static_cast<uint64_t>(tokenBase) * NOPE_DIM], nopeParams, pad);

        DataCopyExtParams ropeParams{
            1, static_cast<uint32_t>(m * ROPE_DIM * sizeof(T)), 0, 0, 0};
        DataCopyPad(rope_[phase], ropeGm_[static_cast<uint64_t>(tokenBase) * ROPE_DIM], ropeParams, pad);

        DataCopyExtParams slotParams{1, static_cast<uint32_t>(m * sizeof(int64_t)), 0, 0, 0};
        DataCopyPadExtParams<int64_t> slotPad{false, 0, 0, 0};
        DataCopyPad(slots_[phase], slotsGm_[tokenBase], slotParams, slotPad);
    }

    __aicore__ inline void Quant(uint32_t phase, uint32_t m)
    {
        if constexpr (std::is_same<T, half>::value) {
            DataCopy(y_, nope_[phase], m * NOPE_DIM);
        } else {
            // 910B has no bf16→half Cast. fat_ is free until EncodeValueBatch.
            auto tmpF = fat_.template ReinterpretCast<float>();
            constexpr uint32_t kFloatChunk = (kTile * NOPE_DIM * sizeof(half)) / sizeof(float);
            for (uint32_t off = 0; off < m * NOPE_DIM; off += kFloatChunk) {
                const uint32_t n = m * NOPE_DIM - off < kFloatChunk ? m * NOPE_DIM - off : kFloatChunk;
                Cast(tmpF, nope_[phase][off], RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
                Cast(y_[off], tmpF, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
            }
        }
        PipeBarrier<PIPE_V>();
        EncodeValueBatch(y_, m);
    }

    __aicore__ inline void Assemble(uint32_t phase, uint32_t m)
    {
        auto ropeU8 = rope_[phase].template ReinterpretCast<uint8_t>();
        for (uint32_t r = 0; r < m; ++r) {
            LocalTensor<uint8_t> row = rows_[phase][r * UB_ROW_ALIGN];
            DataCopy(row, codes_[r * CODE_BYTES], CODE_BYTES);
            DataCopy(row[ROPE_OFF], ropeU8[r * ROPE_BYTES], ROPE_BYTES);
        }
        PipeBarrier<PIPE_V>();
        SyncVToS();
        auto minVec = scal_[kScalarTile];
        auto stepVec = scal_[2 * kScalarTile];
        for (uint32_t r = 0; r < m; ++r) {
            LocalTensor<uint8_t> row = rows_[phase][r * UB_ROW_ALIGN];
            auto metaH = row.template ReinterpretCast<half>();
            metaH.SetValue(SCALE_OFF / sizeof(half), stepVec.GetValue(r));
            metaH.SetValue(OFFSET_OFF / sizeof(half), minVec.GetValue(r));
        }
    }

    __aicore__ inline void Scatter(uint32_t phase, uint32_t m)
    {
        const uint64_t slotLimit = static_cast<uint64_t>(numBlocks_) * blockSize_;
        for (uint32_t r = 0; r < m; ++r) {
            const int64_t slot = slots_[phase].GetValue(r);
            if (slot < 0) {
                continue;
            }
            const uint64_t slotU = static_cast<uint64_t>(slot);
            if (slotU >= slotLimit) {
                continue;
            }
            DataCopyExtParams params{1, SLOT_BYTES, 0, 0, 0};
            DataCopyPad(cacheGm_[slotU * SLOT_BYTES], rows_[phase][r * UB_ROW_ALIGN], params);
        }
    }

    GlobalTensor<T> nopeGm_;
    GlobalTensor<T> ropeGm_;
    GlobalTensor<int64_t> slotsGm_;
    GlobalTensor<uint8_t> cacheGm_;
    uint32_t numTokens_;
    uint32_t blockSize_;
    uint32_t numBlocks_;

    TPipe pipe_;
    TBuf<TPosition::VECIN> bufNope_[2];
    TBuf<TPosition::VECIN> bufRope_[2];
    TBuf<TPosition::VECIN> bufSlots_[2];
    TBuf<TPosition::VECCALC> bufRows_[2];
    TBuf<TPosition::VECCALC> bufY_;
    TBuf<TPosition::VECCALC> bufFat_;
    TBuf<TPosition::VECCALC> bufPartial_;
    TBuf<TPosition::VECCALC> bufScal_;
    TBuf<TPosition::VECCALC> bufCodes_;

    LocalTensor<T> nope_[2];
    LocalTensor<T> rope_[2];
    LocalTensor<int64_t> slots_[2];
    LocalTensor<uint8_t> rows_[2];
    LocalTensor<half> y_;
    LocalTensor<half> fat_;
    LocalTensor<half> partial_;
    LocalTensor<half> scal_;
    LocalTensor<uint8_t> codes_;
};

template <typename T>
__aicore__ inline void Run(
    GM_ADDR nope,
    GM_ADDR rope,
    GM_ADDR slotMapping,
    GM_ADDR kvCache,
    GM_ADDR workspace,
    uint32_t numTokens,
    uint32_t blockSize,
    uint32_t numBlocks,
    uint32_t dataCores,
    uint32_t tileRows)
{
    const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
    __gm__ uint8_t* raw = reinterpret_cast<__gm__ uint8_t*>(workspace);
    SetSysWorkspace(raw);
    if (GetSysWorkSpacePtr() == nullptr || core >= dataCores) {
        return;
    }
    GlobalTensor<T> nopeGm;
    GlobalTensor<T> ropeGm;
    GlobalTensor<int64_t> slots;
    GlobalTensor<uint8_t> cache;
    nopeGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(nope), static_cast<uint64_t>(numTokens) * NOPE_DIM);
    ropeGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rope), static_cast<uint64_t>(numTokens) * ROPE_DIM);
    slots.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slotMapping), numTokens);
    cache.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(kvCache),
        static_cast<uint64_t>(numBlocks) * blockSize * SLOT_BYTES);
    VectorPackSfaV2<T> vec(
        nopeGm, ropeGm, slots, cache, numTokens, blockSize, numBlocks, tileRows);
    vec.Process(core, dataCores);
}

}  // namespace rq_uniform_pack_sfa_v2

#endif  // RQ_UNIFORM_PACK_SFA_VECTOR_V2_H
