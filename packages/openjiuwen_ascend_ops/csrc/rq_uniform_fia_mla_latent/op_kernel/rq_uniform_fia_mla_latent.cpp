/*
 * RqUniformFiaMlaLatent — Scheme A dequant fused as the FIA S2 producer.
 *
 * Path B (design §5 / §6): history is decoded to y, not to c. This kernel does
 * not multiply by R. keyRope stays in the fp paged cache.
 *
 * Tiling key 0 writes the full dequant y for the golden.
 * Tiling key 1 runs FiaKernelNonQuantMla. For each S2 tile the vector cores
 * dequant Scheme A into workspace y_tile[4][128, 512]; MM1/MM2 read that tile
 * for the nope half, and the rope half still comes from the paged k_pe cache.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mla_fia_device_tiling.h"
#include "vendored/arch32/fia_kernel_nonquant_mla.h"

using namespace AscendC;

namespace {

constexpr uint32_t kS2 = 128;
constexpr uint32_t kLatent = 512;
constexpr uint32_t kAivCount = 2;
constexpr uint32_t kHalfRows = kS2 / kAivCount;
constexpr uint16_t kFlagReady = 1;
constexpr uint16_t kFlagDone0 = 2;
constexpr uint16_t kFlagDone1 = 3;

template <pipe_t Pipe>
__aicore__ inline void CrossSet(uint16_t flag) {
    CrossCoreSetFlag<0x2, Pipe>(flag);
}

template <pipe_t Pipe>
__aicore__ inline void CrossWait(uint16_t flag) {
    CrossCoreWaitFlag<0x2, Pipe>(flag);
}

__aicore__ inline void SyncVToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
}

__aicore__ inline void SyncMte2ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}

__aicore__ inline void SyncMte2ToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e);
    WaitFlag<HardEvent::MTE2_S>(e);
}

__aicore__ inline void SyncSToMte2() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(e);
    WaitFlag<HardEvent::S_MTE2>(e);
}

__aicore__ inline void SyncVToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);
}

__aicore__ inline void SyncSToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e);
    WaitFlag<HardEvent::S_MTE3>(e);
}

__aicore__ inline void SyncMte3ToMte2() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e);
    WaitFlag<HardEvent::MTE3_MTE2>(e);
}

__aicore__ inline void SyncMte3ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e);
    WaitFlag<HardEvent::MTE3_V>(e);
}

__aicore__ inline void SyncMte3ToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(e);
    WaitFlag<HardEvent::MTE3_S>(e);
}

// base/step sit on a 2-byte boundary, so every GM move uses DataCopyPad.
__aicore__ inline void CopyBytesToGm(
    GlobalTensor<uint8_t>& dst, uint64_t offset, LocalTensor<uint8_t>& src, uint32_t nbytes) {
    DataCopyExtParams params{1, nbytes, 0, 0, 0};
    DataCopyPad(dst[offset], src, params);
}

__aicore__ inline void CopyBytesFromGm(
    LocalTensor<uint8_t>& dst, GlobalTensor<uint8_t>& src, uint64_t offset, uint32_t nbytes) {
    DataCopyExtParams params{1, nbytes, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    DataCopyPad(dst, src[offset], params, pad);
}

__aicore__ inline uint32_t UMin(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

__aicore__ inline uint32_t CoreId() {
    if ASCEND_IS_AIV {
        return static_cast<uint32_t>(GetBlockIdx() / GetSubBlockNum());
    }
    return static_cast<uint32_t>(GetBlockIdx());
}

__aicore__ inline uint16_t DoneFlag(uint32_t slot) {
    return slot == 0 ? kFlagDone0 : kFlagDone1;
}

// One S2 tile of fp y. MM1 reads this as K_nope and MM2 reads it as V.
// Rope is not in this buffer.
__aicore__ inline void ConsumeYTile(GlobalTensor<uint8_t>& yTileBytes, uint32_t slot, uint32_t rows) {
    (void)yTileBytes;
    (void)slot;
    (void)rows;
}

template <typename T>
class SchemeADequant {
public:
    __aicore__ inline SchemeADequant(
        GlobalTensor<uint8_t> cache,
        GlobalTensor<int32_t> blockTable,
        GlobalTensor<int64_t> contextLens,
        GlobalTensor<uint8_t> yTileBytes,
        GlobalTensor<uint8_t> yOutBytes,
        uint32_t blockSize,
        uint32_t numBlocks,
        uint32_t numReqs,
        uint32_t maxBlocks)
        : cache_(cache),
          blockTable_(blockTable),
          contextLens_(contextLens),
          yTileBytes_(yTileBytes),
          yOutBytes_(yOutBytes),
          blockSize_(blockSize),
          numBlocks_(numBlocks),
          numReqs_(numReqs),
          maxBlocks_(maxBlocks) {}

    __aicore__ inline void Process(uint32_t core, uint32_t dataCores, uint32_t aivSlice) {
        if (core >= dataCores) {
            return;
        }
        InitUb();
        const uint32_t row0 = aivSlice * kHalfRows;
        const uint32_t row1 = row0 + kHalfRows;
        for (uint32_t req = core; req < numReqs_; req += dataCores) {
            const uint32_t kvLen = static_cast<uint32_t>(contextLens_.GetValue(req));
            uint32_t prefix = 0;
            for (uint32_t prev = 0; prev < req; ++prev) {
                prefix += static_cast<uint32_t>(contextLens_.GetValue(prev));
            }
            const uint32_t nTiles = (kvLen + kS2 - 1) / kS2;
            for (uint32_t tile = 0; tile < nTiles; ++tile) {
                const uint32_t slot = tile & 1U;
                // First use of each ping-pong slot has no consumer to wait for.
                if (tile >= 2) {
                    CrossWait<PIPE_MTE2>(DoneFlag(slot));
                }
                const uint32_t tileBase = tile * kS2;
                const uint32_t rows = UMin(kS2, kvLen - tileBase);
                for (uint32_t row = row0; row < row1; ++row) {
                    if (row < rows) {
                        DequantRow(req, tileBase + row, slot, row, prefix + tileBase + row);
                    } else {
                        StoreZero(slot, row, false, 0);
                    }
                }
                // Both AIVs set the same flag. MIX 1:2 delivers READY after both.
                CrossSet<PIPE_MTE3>(kFlagReady);
            }
            // Each used slot is set once more than it is waited. Drain those
            // credits so the next request starts from an empty counter.
            if (nTiles > 0) {
                CrossWait<PIPE_MTE2>(kFlagDone0);
            }
            if (nTiles > 1) {
                CrossWait<PIPE_MTE2>(kFlagDone1);
            }
        }
    }

private:
    __aicore__ inline void InitUb() {
        // 32-byte aligned. The +32 on code is the DataCopyPad tail, not extra codes.
        pipe_.InitBuffer(bufCode_, kLatent + 32);
        pipe_.InitBuffer(bufMeta_, 32);
        pipe_.InitBuffer(bufStep_, 32);
        pipe_.InitBuffer(bufHalf_, kLatent * sizeof(half));
        pipe_.InitBuffer(bufFloat_, kLatent * sizeof(float));
        pipe_.InitBuffer(bufMetaF_, 32 * sizeof(float));
        pipe_.InitBuffer(bufOut_, kLatent * sizeof(uint16_t));
        codeU8_ = bufCode_.Get<uint8_t>();
        metaU8_ = bufMeta_.Get<uint8_t>();
        stepU8_ = bufStep_.Get<uint8_t>();
        halfRow_ = bufHalf_.Get<half>();
        floatRow_ = bufFloat_.Get<float>();
        metaF_ = bufMetaF_.Get<float>();
        outRow_ = bufOut_.Get<T>();
    }

    __aicore__ inline bool Locate(uint32_t req, uint32_t token, uint32_t& blk, uint32_t& pos) {
        const uint32_t blockIdx = token / blockSize_;
        if (blockIdx >= maxBlocks_) {
            return false;
        }
        const int32_t blockId = blockTable_.GetValue(req * maxBlocks_ + blockIdx);
        if (blockId < 0 || static_cast<uint32_t>(blockId) >= numBlocks_) {
            return false;
        }
        blk = static_cast<uint32_t>(blockId);
        pos = token % blockSize_;
        return true;
    }

    __aicore__ inline void StoreOutRow(uint32_t slot, uint32_t localRow, bool publish, uint32_t globalRow) {
        auto src = outRow_.template ReinterpretCast<uint8_t>();
        const uint32_t nbytes = kLatent * static_cast<uint32_t>(sizeof(T));
        const uint64_t tileOff = (static_cast<uint64_t>(slot) * kS2 + localRow) * nbytes;
        CopyBytesToGm(yTileBytes_, tileOff, src, nbytes);
        if (publish) {
            CopyBytesToGm(yOutBytes_, static_cast<uint64_t>(globalRow) * nbytes, src, nbytes);
        }
    }

    __aicore__ inline void StoreZero(uint32_t slot, uint32_t localRow, bool publish, uint32_t globalRow) {
        Duplicate(floatRow_, 0.0f, kLatent);
        PipeBarrier<PIPE_V>();
        Cast(outRow_, floatRow_, RoundMode::CAST_RINT, kLatent);
        SyncVToMte3();
        SyncSToMte3();
        StoreOutRow(slot, localRow, publish, globalRow);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void DequantRow(
        uint32_t req, uint32_t token, uint32_t slot, uint32_t localRow, uint32_t globalRow) {
        SyncMte3ToS();
        uint32_t blk = 0;
        uint32_t pos = 0;
        const bool ok = Locate(req, token, blk, pos);
        SyncSToMte2();
        if (!ok) {
            StoreZero(slot, localRow, true, globalRow);
            return;
        }
        // SoA slab: codes [0, BS*L), base [BS*L, BS*(L+2)), step [BS*(L+2), BS*(L+4)).
        const uint64_t stride = static_cast<uint64_t>(blockSize_) * (kLatent + 4);
        const uint64_t blockBase = static_cast<uint64_t>(blk) * stride;
        const uint64_t codeOff = blockBase + static_cast<uint64_t>(pos) * kLatent;
        const uint64_t baseOff = blockBase + static_cast<uint64_t>(blockSize_) * kLatent +
            static_cast<uint64_t>(pos) * sizeof(T);
        const uint64_t stepOff = blockBase + static_cast<uint64_t>(blockSize_) * (kLatent + 2) +
            static_cast<uint64_t>(pos) * sizeof(T);

        CopyBytesFromGm(codeU8_, cache_, codeOff, kLatent);
        CopyBytesFromGm(metaU8_, cache_, baseOff, static_cast<uint32_t>(sizeof(T)));
        CopyBytesFromGm(stepU8_, cache_, stepOff, static_cast<uint32_t>(sizeof(T)));
        SyncMte2ToV();
        SyncMte2ToS();

        // Pack stores int8(q-128) as the uint8 bit pattern (q ^ 0x80).
        // Reinterpret is that value; do not xor again.
        auto codeI8 = codeU8_.ReinterpretCast<int8_t>();
        Cast(halfRow_, codeI8, RoundMode::CAST_NONE, kLatent);
        PipeBarrier<PIPE_V>();
        Cast(floatRow_, halfRow_, RoundMode::CAST_NONE, kLatent);
        PipeBarrier<PIPE_V>();
        Cast(metaF_, metaU8_.ReinterpretCast<T>(), RoundMode::CAST_NONE, 16);
        Cast(metaF_[16], stepU8_.ReinterpretCast<T>(), RoundMode::CAST_NONE, 16);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        const float base = metaF_.GetValue(0);
        const float step = metaF_.GetValue(16);
        Muls(floatRow_, floatRow_, step, kLatent);
        PipeBarrier<PIPE_V>();
        Adds(floatRow_, floatRow_, base, kLatent);
        PipeBarrier<PIPE_V>();
        Cast(outRow_, floatRow_, RoundMode::CAST_RINT, kLatent);
        SyncVToMte3();
        SyncSToMte3();
        StoreOutRow(slot, localRow, true, globalRow);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    GlobalTensor<uint8_t> cache_;
    GlobalTensor<int32_t> blockTable_;
    GlobalTensor<int64_t> contextLens_;
    GlobalTensor<uint8_t> yTileBytes_;
    GlobalTensor<uint8_t> yOutBytes_;
    uint32_t blockSize_;
    uint32_t numBlocks_;
    uint32_t numReqs_;
    uint32_t maxBlocks_;
    TPipe pipe_;
    TBuf<TPosition::VECIN> bufCode_;
    TBuf<TPosition::VECIN> bufMeta_;
    TBuf<TPosition::VECIN> bufStep_;
    TBuf<TPosition::VECCALC> bufHalf_;
    TBuf<TPosition::VECCALC> bufFloat_;
    TBuf<TPosition::VECCALC> bufMetaF_;
    TBuf<TPosition::VECCALC> bufOut_;
    LocalTensor<uint8_t> codeU8_;
    LocalTensor<uint8_t> metaU8_;
    LocalTensor<uint8_t> stepU8_;
    LocalTensor<half> halfRow_;
    LocalTensor<float> floatRow_;
    LocalTensor<float> metaF_;
    LocalTensor<T> outRow_;
};

template <typename T>
__aicore__ inline void Run(
    GM_ADDR latentCache,
    GM_ADDR blockTable,
    GM_ADDR actualSeqKv,
    GM_ADDR attentionOut,
    GM_ADDR workspace,
    uint32_t blockSize,
    uint32_t numBlocks,
    uint32_t numReqs,
    uint32_t maxBlocks,
    uint32_t dataCores) {
    const uint32_t core = CoreId();
    SetSysWorkspace(reinterpret_cast<__gm__ uint8_t*>(workspace));
    if (GetSysWorkSpacePtr() == nullptr || core >= dataCores || blockSize == 0) {
        return;
    }
    __gm__ uint8_t* userWs = reinterpret_cast<__gm__ uint8_t*>(GetUserWorkspace(workspace));
    const uint64_t tileBytes = static_cast<uint64_t>(kS2) * kLatent * sizeof(T);
    __gm__ uint8_t* yBase = userWs + static_cast<uint64_t>(core) * tileBytes * 2;
    GlobalTensor<uint8_t> yTileBytes;
    yTileBytes.SetGlobalBuffer(yBase, tileBytes * 2);

    GlobalTensor<int64_t> lens;
    lens.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(actualSeqKv), numReqs);

    if ASCEND_IS_AIC {
        for (uint32_t req = core; req < numReqs; req += dataCores) {
            const uint32_t kvLen = static_cast<uint32_t>(lens.GetValue(req));
            const uint32_t nTiles = (kvLen + kS2 - 1) / kS2;
            for (uint32_t tile = 0; tile < nTiles; ++tile) {
                const uint32_t slot = tile & 1U;
                const uint32_t rows = UMin(kS2, kvLen - tile * kS2);
                CrossWait<PIPE_MTE2>(kFlagReady);
                ConsumeYTile(yTileBytes, slot, rows);
                CrossSet<PIPE_MTE2>(DoneFlag(slot));
            }
        }
        return;
    }
    if ASCEND_IS_AIV {
        const uint64_t stride = static_cast<uint64_t>(blockSize) * (kLatent + 4);
        GlobalTensor<uint8_t> cache;
        GlobalTensor<int32_t> table;
        cache.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(latentCache), static_cast<uint64_t>(numBlocks) * stride);
        table.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(blockTable), static_cast<uint64_t>(numReqs) * maxBlocks);
        const uint32_t aivSlice = static_cast<uint32_t>(GetSubBlockIdx() % kAivCount);
        uint64_t totalKv = 0;
        for (uint32_t req = 0; req < numReqs; ++req) {
            totalKv += static_cast<uint32_t>(lens.GetValue(req));
        }
        GlobalTensor<uint8_t> yOutBytes;
        yOutBytes.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(attentionOut), totalKv * kLatent * sizeof(T));
        SchemeADequant<T> vec(
            cache, table, lens, yTileBytes, yOutBytes, blockSize, numBlocks, numReqs, maxBlocks);
        vec.Process(core, dataCores, aivSlice);
    }
}

}  // namespace

using namespace AttentionCommon;

template <typename QT>
__aicore__ inline void InvokeMlaFia(
    GM_ADDR q_nope, GM_ADDR q_pe, GM_ADDR latent_cache, GM_ADDR k_pe_cache, GM_ADDR block_table,
    GM_ADDR actual_seq_q, GM_ADDR actual_seq_kv, GM_ADDR attention_out, GM_ADDR workspace, GM_ADDR tiling,
    const FusedInferAttentionScoreTilingData *__restrict td)
{
    using FIAT = FIAType<QT, QT, QT, QT, true, false, FIA_LAYOUT::TND, 0, false, FIA_LAYOUT::BSH, false>;
    using CubeT = FiaBlockCubeNonQuantMla<FIAT>;
    using VecT = FiaBlockVecNonQuantMla<FIAT>;
    using FdT = FiaBlockVecFlashDecode<FIAT>;

    SetSysWorkspace(reinterpret_cast<__gm__ uint8_t *>(workspace));
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    __gm__ uint8_t *user = reinterpret_cast<__gm__ uint8_t *>(GetUserWorkspace(workspace));
    // Match host FiaWorkspaceBytes(usedCoreNum, ...) — not GetBlockNum() (MIX 1:2).
    const uint32_t cores = td->baseParams.usedCoreNum;
    const uint64_t mm1e = td->workspaceParams.mm1ResSize;
    const uint64_t mm2e = td->workspaceParams.mm2ResSize;
    const uint64_t fiaBytes = static_cast<uint64_t>(cores) * 2ULL * (mm1e * 4ULL + mm1e * 2ULL + mm2e * 4ULL + mm2e * 4ULL);

    const uint32_t core = CoreId();
    __gm__ uint8_t *yTile = user + fiaBytes + static_cast<uint64_t>(core) * kMlaYSlots * 128ULL * 512ULL * 2ULL;
    TPipe pipe;
    FiaKernelNonQuantMla<FIAT, CubeT, VecT, FdT> op;
    // actual_seq_q / actual_seq_kv are host int64 GM (TND accum-Q + BY_BATCH KV).
    op.Init(
        q_nope, k_pe_cache, k_pe_cache, nullptr, nullptr, actual_seq_q, actual_seq_kv, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, block_table, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, q_pe, k_pe_cache, nullptr, nullptr, attention_out, nullptr, user, td, tiling,
        &pipe);
    if ASCEND_IS_AIC {
        op.matmulService.BindYTile(yTile);
    }
    if ASCEND_IS_AIV {
        op.vectorService.BindSchemeA(latent_cache, block_table, yTile, td->workspaceParams.dequantWsSlots);
    }
    op.Process();
}

extern "C" __global__ __aicore__ void rq_uniform_fia_mla_latent(
    GM_ADDR q_nope,
    GM_ADDR q_pe,
    GM_ADDR latent_cache,
    GM_ADDR k_pe_cache,
    GM_ADDR block_table,
    GM_ADDR actual_seq_q,
    GM_ADDR actual_seq_kv,
    GM_ADDR w_uk_t,
    GM_ADDR w_uv,
    GM_ADDR rotation,
    GM_ADDR attention_out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        return;
    }
    GET_TILING_DATA(tdGen, tiling);
    const FusedInferAttentionScoreTilingData *__restrict td =
        reinterpret_cast<const FusedInferAttentionScoreTilingData *>(&tdGen);
    if (td->baseParams.s1Size == 0 || td->baseParams.gSize == 0 || td->baseParams.headDim != kLatent ||
        td->baseParams.bSize == 0) {
        return;
    }
    (void)w_uk_t;
    (void)w_uv;
    (void)rotation;
    if (TILING_KEY_IS(1)) {
        if (td->baseParams.l2CacheOffFlag == 0) {
            InvokeMlaFia<half>(
                q_nope, q_pe, latent_cache, k_pe_cache, block_table, actual_seq_q, actual_seq_kv, attention_out,
                workspace, tiling, td);
        } else {
            InvokeMlaFia<bfloat16_t>(
                q_nope, q_pe, latent_cache, k_pe_cache, block_table, actual_seq_q, actual_seq_kv, attention_out,
                workspace, tiling, td);
        }
        return;
    }
    // Tiling key 0 publishes y for the dequant golden. Attention is key 1.
    (void)q_nope;
    (void)q_pe;
    (void)k_pe_cache;
    (void)actual_seq_q;
    const uint32_t blockSize = td->pageAttenParams.blockSize;
    const uint32_t numBlocks = td->workspaceParams.dequantWsSlots;
    const uint32_t numReqs = td->baseParams.bSize;
    const uint32_t maxBlocks = td->pageAttenParams.maxBlockNumPerBatch;
    const uint32_t dataCores = td->baseParams.usedCoreNum;
    if (td->baseParams.l2CacheOffFlag == 0) {
        Run<half>(
            latent_cache, block_table, actual_seq_kv, attention_out, workspace, blockSize, numBlocks, numReqs,
            maxBlocks, dataCores);
    } else {
        Run<bfloat16_t>(
            latent_cache, block_table, actual_seq_kv, attention_out, workspace, blockSize, numBlocks, numReqs,
            maxBlocks, dataCores);
    }
}
