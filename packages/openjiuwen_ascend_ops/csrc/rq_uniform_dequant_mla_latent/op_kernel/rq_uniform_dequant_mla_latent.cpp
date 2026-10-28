#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/resource.hpp"

using namespace AscendC;

namespace {

constexpr uint32_t kCubeGroup = 32;
constexpr uint32_t kVectorTile = 16;
constexpr uint32_t kAivCount = 2;
constexpr uint32_t kFractal = 16;
constexpr uint32_t kTile = 128;
constexpr uint32_t kMaxLatent = 512;
constexpr uint32_t kWorkspaceHead = 512 * 1024;
constexpr uint16_t kFlagReady = 1;
constexpr uint16_t kFlagDone = 2;

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

__aicore__ inline void SyncMte2ToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(e);
    WaitFlag<HardEvent::MTE2_MTE3>(e);
}

// Plain DataCopy requires a 32-byte GM and UB address. Scheme A base/step live at
// a 2-byte offset, so every vector GM transfer uses DataCopyPad from an aligned slot.
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

template <typename T>
__aicore__ inline constexpr QuantMode_t FixpipeMode() {
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return QuantMode_t::F322BF16;
    }
    return QuantMode_t::F322F16;
}

// y[32, L] @ R[L, L]. A is the vector workspace, already dense for this group.
template <typename T>
class CubeDequant {
public:
    __aicore__ inline CubeDequant(
        GlobalTensor<T> yGm, GlobalTensor<T> rotGm, GlobalTensor<T> outGm, uint32_t latentDim)
        : yGm_(yGm), rotGm_(rotGm), outGm_(outGm), latentDim_(latentDim) {}

    __aicore__ inline void Process(uint32_t core, uint32_t dataCores, uint32_t totalTokens) {
        const uint32_t groups = (totalTokens + kCubeGroup - 1) / kCubeGroup;
        if (core >= dataCores || core >= groups) {
            return;
        }
        InitEvents();
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        for (uint32_t group = core; group < groups; group += dataCores) {
            CrossWait<PIPE_MTE2>(kFlagReady);
            MultiplyGroup(resource);
            CrossSet<PIPE_FIX>(kFlagDone);
        }
        DrainEvents();
    }

private:
    static constexpr uint32_t kL0ABytes = kCubeGroup * kTile * sizeof(uint16_t);
    static constexpr uint32_t kL0BBytes = kTile * kTile * sizeof(uint16_t);
    static constexpr uint32_t kBOffset = kL0ABytes;

    __aicore__ inline void InitEvents() {
        for (uint32_t stage = 0; stage < 2; ++stage) {
            evMte1M_[stage] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>();
            evMMte1_[stage] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
            SetFlag<HardEvent::M_MTE1>(evMMte1_[stage]);
        }
        evMte1Mte2_ = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        evMte2Mte1_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
        evMFix_ = GetTPipePtr()->AllocEventID<HardEvent::M_FIX>();
        evFixM_ = GetTPipePtr()->AllocEventID<HardEvent::FIX_M>();
        SetFlag<HardEvent::MTE1_MTE2>(evMte1Mte2_);
        SetFlag<HardEvent::FIX_M>(evFixM_);
    }

    __aicore__ inline void DrainEvents() {
        for (uint32_t stage = 0; stage < 2; ++stage) {
            WaitFlag<HardEvent::M_MTE1>(evMMte1_[stage]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(evMte1M_[stage]);
            GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(evMMte1_[stage]);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(evMte1Mte2_);
        WaitFlag<HardEvent::FIX_M>(evFixM_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(evMte1Mte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(evMte2Mte1_);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_FIX>(evMFix_);
        GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(evFixM_);
    }

    __aicore__ inline void MultiplyGroup(Catlass::Arch::Resource<Catlass::Arch::AtlasA2>& resource) {
        auto aL1 = resource.l1Buf.template GetBufferByByte<T>(0);
        auto bL1 = resource.l1Buf.template GetBufferByByte<T>(kBOffset);
        auto cL0 = resource.l0CBuf.template GetBufferByByte<float>(0);

        for (uint32_t n0 = 0; n0 < latentDim_; n0 += kTile) {
            const uint32_t nExtent = UMin(kTile, latentDim_ - n0);
            WaitFlag<HardEvent::FIX_M>(evFixM_);
            uint32_t stage = 0;
            for (uint32_t k0 = 0; k0 < latentDim_; k0 += kTile) {
                const uint32_t kExtent = UMin(kTile, latentDim_ - k0);
                auto aL0 = resource.l0ABuf.template GetBufferByByte<T>(stage * kL0ABytes);
                auto bL0 = resource.l0BBuf.template GetBufferByByte<T>(stage * kL0BBytes);

                WaitFlag<HardEvent::MTE1_MTE2>(evMte1Mte2_);
                LoadA(aL1, k0, kExtent);
                LoadB(bL1, k0, n0, kExtent, nExtent);
                SetFlag<HardEvent::MTE2_MTE1>(evMte2Mte1_);
                WaitFlag<HardEvent::MTE2_MTE1>(evMte2Mte1_);

                WaitFlag<HardEvent::M_MTE1>(evMMte1_[stage]);
                LoadL0(aL0, bL0, aL1, bL1, kExtent, nExtent);
                SetFlag<HardEvent::MTE1_M>(evMte1M_[stage]);
                WaitFlag<HardEvent::MTE1_M>(evMte1M_[stage]);

                MmadParams mm{};
                mm.m = kCubeGroup;
                mm.n = nExtent;
                mm.k = kExtent;
                mm.cmatrixInitVal = (k0 == 0);
                mm.cmatrixSource = false;
                mm.unitFlag = 0;
                Mmad(cL0, aL0, bL0, mm);
                SetFlag<HardEvent::M_MTE1>(evMMte1_[stage]);
                SetFlag<HardEvent::MTE1_MTE2>(evMte1Mte2_);
                stage ^= 1U;
            }
            SetFlag<HardEvent::M_FIX>(evMFix_);
            WaitFlag<HardEvent::M_FIX>(evMFix_);
            FixpipeParamsV220 fix{};
            fix.nSize = nExtent;
            fix.mSize = kCubeGroup;
            fix.srcStride = kCubeGroup;
            fix.dstStride = latentDim_;
            fix.ndNum = 1;
            fix.unitFlag = 0;
            fix.quantPre = FixpipeMode<T>();
            fix.reluEn = false;
            Fixpipe<T, float, CFG_ROW_MAJOR>(outGm_[n0], cL0, fix);
            SetFlag<HardEvent::FIX_M>(evFixM_);
        }
        WaitFlag<HardEvent::FIX_M>(evFixM_);
        SetFlag<HardEvent::FIX_M>(evFixM_);
    }

    __aicore__ inline void LoadA(LocalTensor<T>& aL1, uint32_t k0, uint32_t kExtent) {
        for (uint32_t slice = 0; slice < kAivCount; ++slice) {
            Nd2NzParams params{};
            params.ndNum = 1;
            params.nValue = kVectorTile;
            params.dValue = kExtent;
            params.srcNdMatrixStride = 0;
            params.srcDValue = latentDim_;
            params.dstNzC0Stride = kVectorTile;
            params.dstNzNStride = 1;
            params.dstNzMatrixStride = 0;
            const uint64_t src = static_cast<uint64_t>(slice) * kVectorTile * latentDim_ + k0;
            const uint32_t dst = slice * kVectorTile * kExtent;
            DataCopy(aL1[dst], yGm_[src], params);
        }
    }

    __aicore__ inline void LoadB(
        LocalTensor<T>& bL1, uint32_t k0, uint32_t n0, uint32_t kExtent, uint32_t nExtent) {
        Nd2NzParams params{};
        params.ndNum = kExtent / kFractal;
        params.nValue = kFractal;
        params.dValue = nExtent;
        params.srcNdMatrixStride = kFractal * latentDim_;
        params.srcDValue = latentDim_;
        params.dstNzC0Stride = kFractal;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = kFractal * nExtent;
        const uint64_t src = static_cast<uint64_t>(k0) * latentDim_ + n0;
        DataCopy(bL1, rotGm_[src], params);
    }

    __aicore__ inline void LoadL0(
        LocalTensor<T>& aL0,
        LocalTensor<T>& bL0,
        LocalTensor<T>& aL1,
        LocalTensor<T>& bL1,
        uint32_t kExtent,
        uint32_t nExtent) {
        LoadData2DParams aLoad{};
        aLoad.startIndex = 0;
        aLoad.repeatTimes = (kCubeGroup / kFractal) * (kExtent / kFractal);
        aLoad.srcStride = 1;
        aLoad.dstGap = 0;
        aLoad.ifTranspose = false;
        LoadData(aL0, aL1, aLoad);

        LoadData2DParams bLoad{};
        bLoad.startIndex = 0;
        bLoad.repeatTimes = nExtent / kFractal;
        bLoad.srcStride = 1;
        bLoad.dstGap = 0;
        bLoad.ifTranspose = true;
        for (uint32_t i = 0; i < kExtent / kFractal; ++i) {
            const uint32_t off = i * nExtent * kFractal;
            LoadData(bL0[off], bL1[off], bLoad);
        }
    }

    GlobalTensor<T> yGm_;
    GlobalTensor<T> rotGm_;
    GlobalTensor<T> outGm_;
    uint32_t latentDim_;
    int8_t evMte1M_[2];
    int8_t evMMte1_[2];
    int8_t evMte1Mte2_;
    int8_t evMte2Mte1_;
    int8_t evMFix_;
    int8_t evFixM_;
};

template <typename T>
class VectorDequant {
public:
    __aicore__ inline VectorDequant(
        GlobalTensor<uint8_t> cache,
        GlobalTensor<T> rotationUnused,
        GlobalTensor<int32_t> blockTable,
        GlobalTensor<int32_t> contextLens,
        GlobalTensor<int32_t> seqStarts,
        GlobalTensor<T> yGm,
        GlobalTensor<T> outGm,
        GlobalTensor<T> cOut,
        GlobalTensor<uint8_t> yBytes,
        GlobalTensor<uint8_t> outBytes,
        GlobalTensor<uint8_t> cBytes,
        uint32_t totalTokens,
        uint32_t latentDim,
        uint32_t blockSize,
        uint32_t numBlocks,
        uint32_t numReqs,
        uint32_t maxBlocks)
        : cache_(cache),
          blockTable_(blockTable),
          contextLens_(contextLens),
          seqStarts_(seqStarts),
          yGm_(yGm),
          outGm_(outGm),
          cOut_(cOut),
          yBytes_(yBytes),
          outBytes_(outBytes),
          cBytes_(cBytes),
          totalTokens_(totalTokens),
          latentDim_(latentDim),
          blockSize_(blockSize),
          numBlocks_(numBlocks),
          numReqs_(numReqs),
          maxBlocks_(maxBlocks) {
        (void)rotationUnused;
    }

    __aicore__ inline void Process(uint32_t core, uint32_t dataCores, uint32_t aivSlice) {
        const uint32_t groups = (totalTokens_ + kCubeGroup - 1) / kCubeGroup;
        if (core >= dataCores || core >= groups) {
            return;
        }
        InitUb();
        for (uint32_t group = core; group < groups; group += dataCores) {
            const uint32_t token0 = group * kCubeGroup;
            const uint32_t valid = UMin(kCubeGroup, totalTokens_ - token0);
            const uint32_t sliceStart = aivSlice * kVectorTile;
            uint32_t rows = 0;
            if (sliceStart < valid) {
                rows = UMin(kVectorTile, valid - sliceStart);
            }
            // Pad the cube M=32 tile. Unwritten rows stay zero and are not stored.
            ZeroSlice(sliceStart);
            for (uint32_t i = 0; i < rows; ++i) {
                GatherRow(token0 + sliceStart + i, sliceStart + i);
            }
            CrossSet<PIPE_MTE3>(kFlagReady);
            CrossWait<PIPE_MTE2>(kFlagDone);
            if (rows > 0) {
                StoreSlice(token0 + sliceStart, sliceStart, rows);
            }
        }
    }

private:
    __aicore__ inline void InitUb() {
        // Each byte buffer starts 32-byte aligned. base and step each own a slot.
        pipe_.InitBuffer(bufCode_, kMaxLatent + 32);
        pipe_.InitBuffer(bufMeta_, 32);
        pipe_.InitBuffer(bufStep_, 32);
        pipe_.InitBuffer(bufHalf_, kMaxLatent * sizeof(half));
        pipe_.InitBuffer(bufFloat_, kMaxLatent * sizeof(float));
        pipe_.InitBuffer(bufMetaF_, 32 * sizeof(float));
        pipe_.InitBuffer(bufOut_, kMaxLatent * sizeof(uint16_t));
        pipe_.InitBuffer(bufCopy_, kVectorTile * kMaxLatent * sizeof(uint16_t));
        codeU8_ = bufCode_.Get<uint8_t>();
        metaU8_ = bufMeta_.Get<uint8_t>();
        stepU8_ = bufStep_.Get<uint8_t>();
        halfRow_ = bufHalf_.Get<half>();
        floatRow_ = bufFloat_.Get<float>();
        metaF_ = bufMetaF_.Get<float>();
        outRow_ = bufOut_.Get<T>();
        copyRow_ = bufCopy_.Get<T>();
    }

    __aicore__ inline bool Locate(uint32_t outRow, uint32_t& blk, uint32_t& pos) {
        uint32_t remain = outRow;
        for (uint32_t req = 0; req < numReqs_; ++req) {
            const uint32_t len = static_cast<uint32_t>(contextLens_.GetValue(req));
            if (len == 0) {
                continue;
            }
            if (remain < len) {
                const uint32_t seq = static_cast<uint32_t>(seqStarts_.GetValue(req)) + remain;
                const uint32_t blockIdx = seq / blockSize_;
                if (blockIdx >= maxBlocks_) {
                    return false;
                }
                const int32_t blockId = blockTable_.GetValue(req * maxBlocks_ + blockIdx);
                if (blockId < 0 || static_cast<uint32_t>(blockId) >= numBlocks_) {
                    return false;
                }
                blk = static_cast<uint32_t>(blockId);
                pos = seq % blockSize_;
                return true;
            }
            remain -= len;
        }
        return false;
    }

    __aicore__ inline void StoreYRow(uint32_t localRow) {
        auto src = outRow_.template ReinterpretCast<uint8_t>();
        const uint32_t nbytes = latentDim_ * static_cast<uint32_t>(sizeof(T));
        CopyBytesToGm(yBytes_, static_cast<uint64_t>(localRow) * nbytes, src, nbytes);
    }

    __aicore__ inline void ZeroSlice(uint32_t localRow0) {
        Duplicate(floatRow_, 0.0f, latentDim_);
        PipeBarrier<PIPE_V>();
        Cast(outRow_, floatRow_, RoundMode::CAST_RINT, latentDim_);
        SyncVToMte3();
        SyncSToMte3();
        for (uint32_t i = 0; i < kVectorTile; ++i) {
            StoreYRow(localRow0 + i);
        }
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void GatherRow(uint32_t outRow, uint32_t localRow) {
        SyncMte3ToS();
        uint32_t blk = 0;
        uint32_t pos = 0;
        const bool ok = Locate(outRow, blk, pos);
        SyncSToMte2();
        if (!ok) {
            Duplicate(floatRow_, 0.0f, latentDim_);
            PipeBarrier<PIPE_V>();
            Cast(outRow_, floatRow_, RoundMode::CAST_RINT, latentDim_);
            SyncVToMte3();
            SyncSToMte3();
            StoreYRow(localRow);
            SyncMte3ToMte2();
            SyncMte3ToV();
            return;
        }
        const uint64_t stride = static_cast<uint64_t>(blockSize_) * (latentDim_ + 4);
        const uint64_t blockBase = static_cast<uint64_t>(blk) * stride;
        const uint64_t codeOff = blockBase + static_cast<uint64_t>(pos) * latentDim_;
        const uint64_t baseOff = blockBase + static_cast<uint64_t>(blockSize_) * latentDim_ +
            static_cast<uint64_t>(pos) * sizeof(T);
        const uint64_t stepOff = blockBase + static_cast<uint64_t>(blockSize_) * (latentDim_ + 2) +
            static_cast<uint64_t>(pos) * sizeof(T);

        CopyBytesFromGm(codeU8_, cache_, codeOff, latentDim_);
        CopyBytesFromGm(metaU8_, cache_, baseOff, static_cast<uint32_t>(sizeof(T)));
        CopyBytesFromGm(stepU8_, cache_, stepOff, static_cast<uint32_t>(sizeof(T)));
        SyncMte2ToV();
        SyncMte2ToS();

        auto codeI8 = codeU8_.ReinterpretCast<int8_t>();
        Cast(halfRow_, codeI8, RoundMode::CAST_NONE, latentDim_);
        PipeBarrier<PIPE_V>();
        Cast(floatRow_, halfRow_, RoundMode::CAST_NONE, latentDim_);
        PipeBarrier<PIPE_V>();
        // One 32-byte slot is 16 halves. Only element 0 of each slot is live.
        Cast(metaF_, metaU8_.ReinterpretCast<T>(), RoundMode::CAST_NONE, 16);
        Cast(metaF_[16], stepU8_.ReinterpretCast<T>(), RoundMode::CAST_NONE, 16);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        const float base = metaF_.GetValue(0);
        const float step = metaF_.GetValue(16);
        Muls(floatRow_, floatRow_, step, latentDim_);
        PipeBarrier<PIPE_V>();
        Adds(floatRow_, floatRow_, base, latentDim_);
        PipeBarrier<PIPE_V>();
        Cast(outRow_, floatRow_, RoundMode::CAST_RINT, latentDim_);
        SyncVToMte3();
        SyncSToMte3();
        StoreYRow(localRow);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void StoreSlice(uint32_t outRow0, uint32_t localRow0, uint32_t rows) {
        const uint32_t nbytes = rows * latentDim_ * static_cast<uint32_t>(sizeof(T));
        auto dstUb = copyRow_.template ReinterpretCast<uint8_t>();
        CopyBytesFromGm(
            dstUb, outBytes_, static_cast<uint64_t>(localRow0) * latentDim_ * sizeof(T), nbytes);
        SyncMte2ToMte3();
        CopyBytesToGm(cBytes_, static_cast<uint64_t>(outRow0) * latentDim_ * sizeof(T), dstUb, nbytes);
        SyncMte3ToMte2();
    }

    GlobalTensor<uint8_t> cache_;
    GlobalTensor<int32_t> blockTable_;
    GlobalTensor<int32_t> contextLens_;
    GlobalTensor<int32_t> seqStarts_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> outGm_;
    GlobalTensor<T> cOut_;
    GlobalTensor<uint8_t> yBytes_;
    GlobalTensor<uint8_t> outBytes_;
    GlobalTensor<uint8_t> cBytes_;
    uint32_t totalTokens_;
    uint32_t latentDim_;
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
    TBuf<TPosition::VECCALC> bufCopy_;
    LocalTensor<uint8_t> codeU8_;
    LocalTensor<uint8_t> metaU8_;
    LocalTensor<uint8_t> stepU8_;
    LocalTensor<half> halfRow_;
    LocalTensor<float> floatRow_;
    LocalTensor<float> metaF_;
    LocalTensor<T> outRow_;
    LocalTensor<T> copyRow_;
};

template <typename T>
__aicore__ inline void Run(
    GM_ADDR latentCache,
    GM_ADDR rotation,
    GM_ADDR blockTable,
    GM_ADDR contextLens,
    GM_ADDR seqStarts,
    GM_ADDR cOut,
    GM_ADDR workspace,
    uint32_t totalTokens,
    uint32_t latentDim,
    uint32_t blockSize,
    uint32_t numBlocks,
    uint32_t numReqs,
    uint32_t maxBlocks,
    uint32_t dataCores) {
    const uint32_t core = CoreId();
    __gm__ uint8_t* raw = reinterpret_cast<__gm__ uint8_t*>(workspace);
    SetSysWorkspace(raw);
    if (GetSysWorkSpacePtr() == nullptr || core >= dataCores) {
        return;
    }
    const uint64_t oneBuf = static_cast<uint64_t>(kCubeGroup) * latentDim * sizeof(uint16_t);
    __gm__ uint8_t* base = raw + kWorkspaceHead + static_cast<uint64_t>(core) * oneBuf * 2;
    GlobalTensor<T> yGm;
    GlobalTensor<T> outGm;
    GlobalTensor<uint8_t> yBytes;
    GlobalTensor<uint8_t> outBytes;
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(base), kCubeGroup * latentDim);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(base + oneBuf), kCubeGroup * latentDim);
    yBytes.SetGlobalBuffer(base, oneBuf);
    outBytes.SetGlobalBuffer(base + oneBuf, oneBuf);

    if ASCEND_IS_AIC {
        GlobalTensor<T> rotGm;
        rotGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rotation), static_cast<uint64_t>(latentDim) * latentDim);
        TPipe pipe;
        CubeDequant<T> cube(yGm, rotGm, outGm, latentDim);
        cube.Process(core, dataCores, totalTokens);
        return;
    }
    if ASCEND_IS_AIV {
        const uint64_t stride = static_cast<uint64_t>(blockSize) * (latentDim + 4);
        GlobalTensor<uint8_t> cache;
        GlobalTensor<int32_t> table;
        GlobalTensor<int32_t> lens;
        GlobalTensor<int32_t> starts;
        GlobalTensor<T> cGm;
        GlobalTensor<uint8_t> cBytes;
        GlobalTensor<T> rotUnused;
        cache.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(latentCache), static_cast<uint64_t>(numBlocks) * stride);
        table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable), static_cast<uint64_t>(numReqs) * maxBlocks);
        lens.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(contextLens), numReqs);
        starts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqStarts), numReqs);
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cOut), static_cast<uint64_t>(totalTokens) * latentDim);
        cBytes.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(cOut),
            static_cast<uint64_t>(totalTokens) * latentDim * sizeof(T));
        const uint32_t aivSlice = static_cast<uint32_t>(GetSubBlockIdx() % kAivCount);
        VectorDequant<T> vec(
            cache, rotUnused, table, lens, starts, yGm, outGm, cGm, yBytes, outBytes, cBytes, totalTokens, latentDim,
            blockSize, numBlocks, numReqs, maxBlocks);
        vec.Process(core, dataCores, aivSlice);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void rq_uniform_dequant_mla_latent(
    GM_ADDR latent_cache,
    GM_ADDR rotation,
    GM_ADDR block_table,
    GM_ADDR context_lens,
    GM_ADDR seq_starts,
    GM_ADDR c,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        return;
    }
    GET_TILING_DATA(td, tiling);
    if (td.latentDim == 0 || td.latentDim > kMaxLatent || td.totalTokens == 0) {
        return;
    }
    if (td.dtype == 0) {
        Run<half>(
            latent_cache, rotation, block_table, context_lens, seq_starts, c, workspace, td.totalTokens, td.latentDim,
            td.blockSize, td.numBlocks, td.numReqs, td.maxBlocks, td.dataCores);
    } else {
        Run<bfloat16_t>(
            latent_cache, rotation, block_table, context_lens, seq_starts, c, workspace, td.totalTokens, td.latentDim,
            td.blockSize, td.numBlocks, td.numReqs, td.maxBlocks, td.dataCores);
    }
}
