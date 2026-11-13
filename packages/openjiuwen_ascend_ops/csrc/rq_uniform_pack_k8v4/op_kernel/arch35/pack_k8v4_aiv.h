/*!
 * \file pack_k8v4_aiv.h
 * \brief A5 TQ_K8_FP8 pure-AIV pack — mode0 Value path + Key fp8 code-only.
 *
 * Value: fat/scal/encodedBatch (mode0 EncodeValueBatch + SoA Store).
 * Key: half/bf16→float→fp8 into KeyCode only (no key meta in UB or GM).
 * Two UB phases: MTE2 of phase N+1 overlaps Vector of phase N, then MTE3.
 * Pipeline events are EVENT_ID0/1 only — do not FetchEventID those pipes.
 * Work item is one ResolvePackChunk (at most 16 rows) times one head, striped
 * across AIVs with workId % dataCores. Heads are not serial on a token owner.
 */
#pragma once

#include "kernel_operator.h"
#include "../rq_uniform_pack_k8v4_common.h"
#include "pack_k8v4_ub_layout.h"
#include "pack_k8v4_cache_layout.h"

namespace rq_uniform {
namespace arch35 {

using namespace AscendC;

constexpr uint32_t K8_A5_HALF_MASK = 128;
constexpr uint32_t K8_A5_HALF_PER_BLOCK = 16;
constexpr half K8_A5_INV_LEVELS = static_cast<half>(1.0f / 15.0f);
constexpr half K8_A5_EPS = static_cast<half>(1.0e-6f);

template <typename T>
class VectorPackK8v4A5 {
public:
    __aicore__ inline VectorPackK8v4A5(
        GlobalTensor<T> keyGm,
        GlobalTensor<T> valueGm,
        GlobalTensor<int32_t> slotsGm,
        GlobalTensor<uint8_t> keyCacheGm,
        GlobalTensor<uint8_t> valueCacheGm,
        uint32_t numTokens,
        uint32_t numHeads,
        uint32_t packD,
        uint32_t blockSize,
        uint32_t numBlocks,
        uint32_t keyStrideToken,
        uint32_t keyStrideHead,
        uint32_t valueStrideToken,
        uint32_t valueStrideHead)
        : keyGm_(keyGm),
          valueGm_(valueGm),
          slotsGm_(slotsGm),
          keyCacheGm_(keyCacheGm),
          valueCacheGm_(valueCacheGm),
          numTokens_(numTokens),
          numHeads_(numHeads),
          packD_(packD),
          blockSize_(blockSize),
          numBlocks_(numBlocks),
          keyStrideToken_(keyStrideToken),
          keyStrideHead_(keyStrideHead),
          valueStrideToken_(valueStrideToken),
          valueStrideHead_(valueStrideHead)
    {
        offs_ = K8A5MakeOffsets(packD_);
        pipe_.InitBuffer(ubBuf_, offs_.size * K8_A5_PP);
        ubBase_ = ubBuf_.Get<uint8_t>();
    }


    __aicore__ inline void Process(uint32_t core, uint32_t dataCores)
    {
        // Both phases start free for MTE2. Each store posts its phase again.
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        K8A5Walk walk;
        walk.cursor = 0;
        walk.head = 0;
        walk.workId = 0;
        walk.base = 0;
        walk.slot = 0;
        walk.validRows = 0;
        walk.haveChunk = 0;
        K8A5Chunk cur{};
        if (!NextOwned(core, dataCores, walk, cur)) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            return;
        }
        PrefetchPhase<0>(cur);
        uint32_t phase = 0;
        K8A5Chunk nxt{};
        while (NextOwned(core, dataCores, walk, nxt)) {
            // Issue the next load before encode, so MTE2(N+1) overlaps V(N).
            // Store(N) is after SetFlag V_MTE3, so it overlaps V(N+1).
            if (phase == 0) {
                PrefetchPhase<1>(nxt);
                EncodePhase<0>(cur);
                StorePhase<0>(cur);
            } else {
                PrefetchPhase<0>(nxt);
                EncodePhase<1>(cur);
                StorePhase<1>(cur);
            }
            cur = nxt;
            phase ^= 1u;
        }
        if (phase == 0) {
            EncodePhase<0>(cur);
            StorePhase<0>(cur);
        } else {
            EncodePhase<1>(cur);
            StorePhase<1>(cur);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

private:
    struct K8A5Chunk {
        uint32_t head;
        uint32_t base;
        uint32_t slot;
        uint32_t validRows;
    };

    // Shared scan cursor. Every core walks the same chunk order so workId matches.
    struct K8A5Walk {
        uint32_t cursor;
        uint32_t head;
        uint32_t workId;
        uint32_t base;
        uint32_t slot;
        uint32_t validRows;
        uint32_t haveChunk;
    };

    __aicore__ inline LocalTensor<uint8_t> PhaseBase(uint32_t phase)
    {
        return ubBase_[phase * offs_.size];
    }

    __aicore__ inline LocalTensor<half> SrcK(uint32_t phase)
    {
        return PhaseBase(phase)[offs_.offSrcK].template ReinterpretCast<half>();
    }
    __aicore__ inline LocalTensor<half> SrcV(uint32_t phase)
    {
        return PhaseBase(phase)[offs_.offSrcV].template ReinterpretCast<half>();
    }
    __aicore__ inline LocalTensor<half> Fat(uint32_t phase)
    {
        return PhaseBase(phase)[offs_.offFat].template ReinterpretCast<half>();
    }
    __aicore__ inline LocalTensor<half> Scal(uint32_t phase)
    {
        return PhaseBase(phase)[offs_.offScal].template ReinterpretCast<half>();
    }
    __aicore__ inline LocalTensor<uint8_t> KeyCode(uint32_t phase)
    {
        return PhaseBase(phase)[offs_.offKeyCode];
    }
    __aicore__ inline LocalTensor<uint8_t> ValEnc(uint32_t phase)
    {
        return PhaseBase(phase)[offs_.offValEnc];
    }

    // One ResolvePackChunk run × one head. Core owns workId % dataCores.
    __aicore__ inline bool NextOwned(
        uint32_t core, uint32_t dataCores, K8A5Walk& walk, K8A5Chunk& out)
    {
        while (walk.cursor < numTokens_ || walk.haveChunk != 0) {
            if (walk.haveChunk == 0) {
                uint32_t slot = 0;
                uint32_t validRows = 0;
                const uint32_t left = numTokens_ - walk.cursor;
                if (!ResolvePackChunk(walk.cursor, left, slot, validRows)) {
                    ++walk.cursor;
                    continue;
                }
                walk.base = walk.cursor;
                walk.slot = slot;
                walk.validRows = validRows;
                walk.head = 0;
                walk.haveChunk = 1;
            }
            if (walk.head >= numHeads_) {
                walk.cursor += walk.validRows;
                walk.haveChunk = 0;
                continue;
            }
            const uint32_t id = walk.workId;
            const uint32_t head = walk.head;
            ++walk.workId;
            ++walk.head;
            if (dataCores != 0 && id % dataCores == core) {
                out.head = head;
                out.base = walk.base;
                out.slot = walk.slot;
                out.validRows = walk.validRows;
                return true;
            }
        }
        return false;
    }

    template <uint32_t PHASE>
    __aicore__ inline void PrefetchPhase(const K8A5Chunk& chunk)
    {
        constexpr event_t kEv = PHASE == 0 ? EVENT_ID0 : EVENT_ID1;
        WaitFlag<HardEvent::MTE3_MTE2>(kEv);
        Prefetch(PHASE, chunk.base, chunk.head, chunk.validRows);
        SetFlag<HardEvent::MTE2_V>(kEv);
    }

    template <uint32_t PHASE>
    __aicore__ inline void EncodePhase(const K8A5Chunk& chunk)
    {
        constexpr event_t kEv = PHASE == 0 ? EVENT_ID0 : EVENT_ID1;
        WaitFlag<HardEvent::MTE2_V>(kEv);
        PadSrcTile(PHASE, chunk.validRows);
        PipeBarrier<PIPE_V>();
        EncodeKeyFp8(PHASE);
        EncodeValueBatch(PHASE);
        SetFlag<HardEvent::V_MTE3>(kEv);
    }

    template <uint32_t PHASE>
    __aicore__ inline void StorePhase(const K8A5Chunk& chunk)
    {
        constexpr event_t kEv = PHASE == 0 ? EVENT_ID0 : EVENT_ID1;
        WaitFlag<HardEvent::V_MTE3>(kEv);
        Store(PHASE, chunk.head, chunk.slot, chunk.validRows);
        SetFlag<HardEvent::MTE3_MTE2>(kEv);
    }

    __aicore__ inline void Prefetch(uint32_t phase, uint32_t tokBase, uint32_t head, uint32_t validRows)
    {
        auto srcK = SrcK(phase);
        auto srcV = SrcV(phase);
        auto srcKT = srcK.template ReinterpretCast<T>();
        auto srcVT = srcV.template ReinterpretCast<T>();
        for (uint32_t r = 0; r < validRows; ++r) {
            const uint64_t kOff =
                (static_cast<uint64_t>(tokBase + r) * keyStrideToken_) +
                static_cast<uint64_t>(head) * keyStrideHead_;
            const uint64_t vOff =
                (static_cast<uint64_t>(tokBase + r) * valueStrideToken_) +
                static_cast<uint64_t>(head) * valueStrideHead_;
            DataCopyExtParams p{1, static_cast<uint32_t>(packD_ * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(srcKT[r * packD_], keyGm_[kOff], p, pad);
            DataCopyPad(srcVT[r * packD_], valueGm_[vOff], p, pad);
        }
    }

    // Zero rows [validRows, K8_A5_TILE_ROWS) so encode can run a full physical tile.
    __aicore__ inline void PadSrcTile(uint32_t phase, uint32_t validRows)
    {
        if (validRows >= K8_A5_TILE_ROWS) {
            return;
        }
        const uint32_t padElems = (K8_A5_TILE_ROWS - validRows) * packD_;
        Duplicate(SrcK(phase)[validRows * packD_], static_cast<half>(0.0f), padElems);
        Duplicate(SrcV(phase)[validRows * packD_], static_cast<half>(0.0f), padElems);
    }

    // Key: code-only UB (offKeyCode). Always encodes a full 16-row tile.
    __aicore__ inline void EncodeKeyFp8(uint32_t phase)
    {
        auto src = SrcK(phase);
        auto dst = KeyCode(phase).template ReinterpretCast<fp8_e4m3fn_t>();
        auto tmp = Fat(phase).template ReinterpretCast<float>();
        const uint32_t n = K8_A5_TILE_ROWS * packD_;
        constexpr uint32_t kChunk = 256;
        for (uint32_t off = 0; off < n; off += kChunk) {
            const uint32_t cur = n - off < kChunk ? n - off : kChunk;
            if constexpr (std::is_same<T, half>::value) {
                Cast(tmp, src[off], RoundMode::CAST_NONE, cur);
            } else {
                Cast(tmp, src[off].template ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, cur);
            }
            PipeBarrier<PIPE_V>();
            Cast(dst[off], tmp, RoundMode::CAST_RINT, cur);
            PipeBarrier<PIPE_V>();
        }
    }

    // Literal port of mode0 EncodeValueBatch (rq_uniform_pack_k8v4_aiv.h).
    // m is always K8_A5_TILE_ROWS (== TQ_VECTOR_BATCH). Only Key encode differs.
    __aicore__ inline void PackApplyRowBrcbSub(
        LocalTensor<half> dst, LocalTensor<half> src, LocalTensor<half> brcb,
        uint32_t m, uint32_t packD)
    {
        constexpr uint32_t COL = 128U;
        BinaryRepeatParams p;
        p.dstBlkStride = 1U;
        p.src0BlkStride = 1U;
        p.src1BlkStride = 0U;
        p.dstRepStride = packD / 16U;
        p.src0RepStride = packD / 16U;
        p.src1RepStride = 1U;
        Sub<half, false>(dst, src, brcb, COL, m, p);
        if (packD == 256U) {
            Sub<half, false>(dst[COL], src[COL], brcb, COL, m, p);
        }
    }

    __aicore__ inline void PackApplyRowBrcbDiv(
        LocalTensor<half> dst, LocalTensor<half> src, LocalTensor<half> brcb,
        uint32_t m, uint32_t packD)
    {
        constexpr uint32_t COL = 128U;
        BinaryRepeatParams p;
        p.dstBlkStride = 1U;
        p.src0BlkStride = 1U;
        p.src1BlkStride = 0U;
        p.dstRepStride = packD / 16U;
        p.src0RepStride = packD / 16U;
        p.src1RepStride = 1U;
        Div<half, false>(dst, src, brcb, COL, m, p);
        if (packD == 256U) {
            Div<half, false>(dst[COL], src[COL], brcb, COL, m, p);
        }
    }

    // WholeReduce mask for half is 128. D=256 is two 128-wide row reduces,
    // then Max/Min of the two scalars (srcRepStride stays packD/16).
    __aicore__ inline void PackWholeReduceMax(
        LocalTensor<half> dst, LocalTensor<half> src, LocalTensor<half> tmp,
        uint32_t m, uint32_t packD)
    {
        constexpr uint32_t COL = 128U;
        constexpr uint32_t typePerBlock = ASCEND_BLOCK_BYTES / sizeof(half);
        const uint32_t srcRep = packD / typePerBlock;
        const int32_t mask = static_cast<int32_t>(packD <= COL ? packD : COL);
        WholeReduceMax<half, false>(
            dst, src, mask, m, 1, 1, srcRep, ReduceOrder::ORDER_ONLY_VALUE);
        if (packD > COL) {
            PipeBarrier<PIPE_V>();
            WholeReduceMax<half, false>(
                tmp, src[COL], mask, m, 1, 1, srcRep, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            Max(dst, dst, tmp, m);
        }
    }

    __aicore__ inline void PackWholeReduceMin(
        LocalTensor<half> dst, LocalTensor<half> src, LocalTensor<half> tmp,
        uint32_t m, uint32_t packD)
    {
        constexpr uint32_t COL = 128U;
        constexpr uint32_t typePerBlock = ASCEND_BLOCK_BYTES / sizeof(half);
        const uint32_t srcRep = packD / typePerBlock;
        const int32_t mask = static_cast<int32_t>(packD <= COL ? packD : COL);
        WholeReduceMin<half, false>(
            dst, src, mask, m, 1, 1, srcRep, ReduceOrder::ORDER_ONLY_VALUE);
        if (packD > COL) {
            PipeBarrier<PIPE_V>();
            WholeReduceMin<half, false>(
                tmp, src[COL], mask, m, 1, 1, srcRep, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            Min(dst, dst, tmp, m);
        }
    }

    __aicore__ inline void EncodeValueBatch(uint32_t phase)
    {
        // mode0: EncodeValueBatch(TQ_MANUAL_AIV_SLICE_M, yBatch) with m==16.
        constexpr uint32_t m = K8_A5_TILE_ROWS;
        constexpr uint32_t repeatsOfRows = K8_A5_TILE_ROWS / 8U;
        const uint32_t packD = packD_;
        const uint32_t totalElems = m * packD;
        const half safeDivisor = K8_A5_EPS;

        auto yBatch = SrcV(phase);
        auto fat = Fat(phase);
        auto scal = Scal(phase);
        auto encodedBatch = ValEnc(phase).template ReinterpretCast<uint16_t>();

        auto sMaxVec = scal;
        auto minVec = scal[K8_A5_SCALAR_TILE_ELEMS];
        auto stepVec = scal[2 * K8_A5_SCALAR_TILE_ELEMS];
        auto stepBlk = scal[3 * K8_A5_SCALAR_TILE_ELEMS];
        auto minBlk = sMaxVec;


        // A5-only: bf16 inputs must become half before the mode0 half pipeline.
        // mode0 receives half from AIC Fixpipe already.
        if constexpr (!std::is_same<T, half>::value) {
            auto tmpF = fat.template ReinterpretCast<float>();
            constexpr uint32_t kChunk = 256;
            for (uint32_t off = 0; off < totalElems; off += kChunk) {
                const uint32_t cur = totalElems - off < kChunk ? totalElems - off : kChunk;
                Cast(tmpF, yBatch[off].template ReinterpretCast<bfloat16_t>(),
                     RoundMode::CAST_NONE, cur);
                PipeBarrier<PIPE_V>();
                Cast(yBatch[off], tmpF, RoundMode::CAST_NONE, cur);
                PipeBarrier<PIPE_V>();
            }
        }

        auto origBatch16 = yBatch;

        PackWholeReduceMax(sMaxVec, origBatch16, stepBlk, m, packD);
        PipeBarrier<PIPE_V>();
        PackWholeReduceMin(minVec, origBatch16, stepBlk, m, packD);
        PipeBarrier<PIPE_V>();

        Sub(stepVec, sMaxVec, minVec, m);
        PipeBarrier<PIPE_V>();
        Muls(stepVec, stepVec, K8_A5_INV_LEVELS, m);
        PipeBarrier<PIPE_V>();
        Maxs(stepVec, stepVec, safeDivisor, m);
        PipeBarrier<PIPE_V>();

        Brcb(minBlk, minVec, repeatsOfRows, BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();
        Brcb(stepBlk, stepVec, repeatsOfRows, BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();

        auto quant_16 = fat;
        PackApplyRowBrcbSub(quant_16, origBatch16, minBlk, m, packD);
        PipeBarrier<PIPE_V>();
        PackApplyRowBrcbDiv(quant_16, quant_16, stepBlk, m, packD);
        PipeBarrier<PIPE_V>();

        Adds(quant_16, quant_16, static_cast<half>(-8.0f), totalElems);
        PipeBarrier<PIPE_V>();

        auto quantI4 = encodedBatch.template ReinterpretCast<int4b_t>();
        Cast(quantI4, quant_16, RoundMode::CAST_NONE, totalElems);
        PipeBarrier<PIPE_V>();

        // mode0 Step 10: quant stays in half; cache meta is dtype T so FIA
        // can read vmin/vstep as the query dtype (bf16 meta is not fp16 bits).
        auto metadataMinT = encodedBatch[
            TqValEncodedVminOff(packD) / sizeof(uint16_t)]
                                .template ReinterpretCast<T>();
        auto metadataStepT = encodedBatch[
            TqValEncodedVstepOff(packD) / sizeof(uint16_t)]
                                 .template ReinterpretCast<T>();
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            constexpr uint32_t typePerBlock = 32U / sizeof(half);
            auto metadataMinFp32 = fat.template ReinterpretCast<float>();
            auto metadataStepFp32 = metadataMinFp32[typePerBlock];
            Cast(metadataMinFp32, minVec, RoundMode::CAST_NONE, m);
            Cast(metadataStepFp32, stepVec, RoundMode::CAST_NONE, m);
            PipeBarrier<PIPE_V>();
            Cast(metadataMinT, metadataMinFp32, RoundMode::CAST_RINT, m);
            Cast(metadataStepT, metadataStepFp32, RoundMode::CAST_RINT, m);
        } else {
            Adds(metadataMinT, minVec, static_cast<half>(0.0f), m);
            Adds(metadataStepT, stepVec, static_cast<half>(0.0f), m);
        }
        PipeBarrier<PIPE_V>();
    }

    // Same rule as mode0 ResolvePackChunk: one slotMapping read, then a
    // contiguous physical run capped by the 16-row group and the PA block.
    __aicore__ inline bool ResolvePackChunk(
        uint32_t tokenStart, uint32_t tokensLeft, uint32_t& slotOut, uint32_t& validRowsOut) const
    {
        if (tokensLeft == 0 || tokenStart >= numTokens_) {
            return false;
        }
        const int32_t slot = slotsGm_.GetValue(tokenStart);
        const uint64_t slotLimit = static_cast<uint64_t>(numBlocks_) * blockSize_;
        if (slot < 0 || static_cast<uint64_t>(slot) >= slotLimit) {
            return false;
        }
        slotOut = static_cast<uint32_t>(slot);
        const uint32_t rowsInGroup = K8_A5_TILE_ROWS - (slotOut % K8_A5_TILE_ROWS);
        const uint32_t rowsInBlock = blockSize_ - (slotOut % blockSize_);
        uint32_t n = tokensLeft;
        if (n > rowsInGroup) {
            n = rowsInGroup;
        }
        if (n > rowsInBlock) {
            n = rowsInBlock;
        }
        validRowsOut = n;
        return n != 0;
    }

    // One contiguous run: UB row 0..validRows-1 → physical slot, slot+1, ...
    __aicore__ inline void Store(uint32_t phase, uint32_t head, uint32_t slot, uint32_t validRows)
    {
        auto keyCode = KeyCode(phase);
        auto valEnc = ValEnc(phase);
        const uint32_t valCodeBytes = packD_ / 2U;
        const uint32_t vminU16 = TqValEncodedVminOff(packD_) / sizeof(uint16_t);
        const uint32_t vstepU16 = TqValEncodedVstepOff(packD_) / sizeof(uint16_t);
        auto encU16 = valEnc.template ReinterpretCast<uint16_t>();
        auto vminU8 = encU16[vminU16].template ReinterpretCast<uint8_t>();
        auto vstepU8 = encU16[vstepU16].template ReinterpretCast<uint8_t>();

        const uint32_t blockIdx = slot / blockSize_;
        const uint32_t pos = slot % blockSize_;
        const uint64_t kBase = K8A5KeyHeadBase(blockIdx, head, numHeads_, blockSize_, packD_);
        const uint64_t vBase = K8A5ValHeadBase(blockIdx, head, numHeads_, blockSize_, packD_);
        DataCopyExtParams kp{1, validRows * packD_, 0, 0, 0};
        DataCopyPad(keyCacheGm_[K8A5KeyCodeOff(kBase, pos, packD_)], keyCode, kp);
        DataCopyExtParams vp{1, validRows * valCodeBytes, 0, 0, 0};
        DataCopyPad(valueCacheGm_[K8A5ValCodeOff(vBase, pos, packD_)], valEnc, vp);
        DataCopyExtParams mp{1, validRows * TQ_ROW_META_BYTES, 0, 0, 0};
        DataCopyPad(
            valueCacheGm_[K8A5ValVminOff(vBase, blockSize_, packD_, pos)], vminU8, mp);
        DataCopyPad(
            valueCacheGm_[K8A5ValVstepOff(vBase, blockSize_, packD_, pos)], vstepU8, mp);
    }

    GlobalTensor<T> keyGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<int32_t> slotsGm_;
    GlobalTensor<uint8_t> keyCacheGm_;
    GlobalTensor<uint8_t> valueCacheGm_;
    uint32_t numTokens_;
    uint32_t numHeads_;
    uint32_t packD_;
    uint32_t blockSize_;
    uint32_t numBlocks_;
    uint32_t keyStrideToken_;
    uint32_t keyStrideHead_;
    uint32_t valueStrideToken_;
    uint32_t valueStrideHead_;
    K8A5UbOffsets offs_ {};

    TPipe pipe_;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<uint8_t> ubBase_;
};

template <typename T>
__aicore__ inline void RunPackK8v4A5(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR slotMapping,
    GM_ADDR keyCache,
    GM_ADDR valueCache,
    GM_ADDR workspace,
    uint32_t numTokens,
    uint32_t numHeads,
    uint32_t packD,
    uint32_t blockSize,
    uint32_t numBlocks,
    uint32_t dataCores,
    uint32_t keyStrideToken,
    uint32_t keyStrideHead,
    uint32_t valueStrideToken,
    uint32_t valueStrideHead)
{
    const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
    if (core >= dataCores || numTokens == 0 || dataCores == 0) {
        return;
    }
    __gm__ uint8_t* raw = reinterpret_cast<__gm__ uint8_t*>(workspace);
    if (raw != nullptr) {
        SetSysWorkspace(raw);
    }
    GlobalTensor<T> keyGm;
    GlobalTensor<T> valueGm;
    GlobalTensor<int32_t> slots;
    GlobalTensor<uint8_t> keyCacheGm;
    GlobalTensor<uint8_t> valueCacheGm;
    keyGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(key),
        static_cast<uint64_t>(numTokens) * numHeads * packD);
    valueGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(value),
        static_cast<uint64_t>(numTokens) * numHeads * packD);
    slots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotMapping), numTokens);
    keyCacheGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(keyCache),
        static_cast<uint64_t>(numBlocks) * numHeads * blockSize * packD);
    valueCacheGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(valueCache),
        static_cast<uint64_t>(numBlocks) * numHeads * blockSize * TqValRowBytes(packD));
    VectorPackK8v4A5<T> vec(
        keyGm, valueGm, slots, keyCacheGm, valueCacheGm,
        numTokens, numHeads, packD, blockSize, numBlocks,
        keyStrideToken, keyStrideHead, valueStrideToken, valueStrideHead);
    vec.Process(core, dataCores);
}

}  // namespace arch35
}  // namespace rq_uniform
