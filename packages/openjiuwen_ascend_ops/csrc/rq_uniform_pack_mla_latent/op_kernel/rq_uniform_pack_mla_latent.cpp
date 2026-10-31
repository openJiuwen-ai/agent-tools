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
// One cube tile matches pack_k8v4: Mmad(m=32, n=128, k=128).
// L=512 is 4x4 of these tiles. K accumulation ping-pongs L0A/L0B.
constexpr uint32_t kTile = 128;
constexpr uint32_t kMaxLatent = 512;
constexpr uint32_t kWorkspaceHead = 512 * 1024;
constexpr uint16_t kFlagFree = 0;
constexpr uint16_t kFlagReady = 1;

template <pipe_t Pipe>
__aicore__ inline void CrossSet(uint16_t flag) {
    CrossCoreSetFlag<0x2, Pipe>(flag);
}

template <pipe_t Pipe>
__aicore__ inline void CrossWait(uint16_t flag) {
    CrossCoreWaitFlag<0x2, Pipe>(flag);
}

__aicore__ inline void SyncMte2ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}

__aicore__ inline void SyncVToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
}

__aicore__ inline void SyncSToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(e);
    WaitFlag<HardEvent::S_V>(e);
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

__aicore__ inline uint32_t UMin(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

__aicore__ inline uint32_t CoreId() {
    if ASCEND_IS_AIV {
        return static_cast<uint32_t>(GetBlockIdx() / GetSubBlockNum());
    }
    return static_cast<uint32_t>(GetBlockIdx());
}

__aicore__ inline void CopyBytesToGm(
    GlobalTensor<uint8_t>& dst,
    uint64_t offset,
    LocalTensor<uint8_t>& src,
    uint32_t nbytes) {
    SyncVToMte3();
    SyncSToMte3();
    DataCopyExtParams params{1, nbytes, 0, 0, 0};
    DataCopyPad(dst[offset], src, params);
    // The next row reuses this UB for both vector casts and scalar stores.
    SyncMte3ToV();
    SyncMte3ToS();
    SyncMte3ToMte2();
}

template <typename T>
class CubePack {
public:
    __aicore__ inline CubePack(
        GlobalTensor<T> cGm,
        GlobalTensor<T> rotGm,
        GlobalTensor<half> bridge,
        uint32_t numTokens,
        uint32_t latentDim)
        : cGm_(cGm), rotGm_(rotGm), bridge_(bridge), numTokens_(numTokens), latentDim_(latentDim) {}

    __aicore__ inline void Process(uint32_t core, uint32_t dataCores) {
        const uint32_t groups = (numTokens_ + kCubeGroup - 1) / kCubeGroup;
        if (core >= dataCores || core >= groups) {
            return;
        }
        // Allocate events on the user TPipe before Resource's TPipe ctor runs.
        InitEvents();
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        for (uint32_t group = core; group < groups; group += dataCores) {
            const uint32_t token0 = group * kCubeGroup;
            const uint32_t valid = UMin(kCubeGroup, numTokens_ - token0);
            CrossWait<PIPE_FIX>(kFlagFree);
            RotateGroup(resource, token0, valid);
            CrossSet<PIPE_FIX>(kFlagReady);
        }
        DrainEvents();
    }

private:
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

    __aicore__ inline void RotateGroup(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2>& resource,
        uint32_t token0,
        uint32_t valid) {
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
                LoadA(aL1, token0, valid, k0, kExtent);
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
                // Release this L0 stage only after MMAD. The next K tile
                // writes the other stage, so LoadData never hits a live L0A.
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
            fix.quantPre = QuantMode_t::F322F16;
            fix.reluEn = false;
            Fixpipe<half, float, CFG_ROW_MAJOR>(bridge_[n0], cL0, fix);
            SetFlag<HardEvent::FIX_M>(evFixM_);
        }
        // Last Fixpipe's GM store must land before the vector cores read it.
        WaitFlag<HardEvent::FIX_M>(evFixM_);
        SetFlag<HardEvent::FIX_M>(evFixM_);
    }

    __aicore__ inline void LoadA(
        LocalTensor<T>& aL1, uint32_t token0, uint32_t valid, uint32_t k0, uint32_t kExtent) {
        for (uint32_t slice = 0; slice < kAivCount; ++slice) {
            const uint32_t sliceStart = slice * kVectorTile;
            if (sliceStart >= valid) {
                continue;
            }
            const uint32_t rows = UMin(kVectorTile, valid - sliceStart);
            Nd2NzParams params{};
            params.ndNum = 1;
            params.nValue = rows;
            params.dValue = kExtent;
            params.srcNdMatrixStride = 0;
            params.srcDValue = latentDim_;
            params.dstNzC0Stride = kVectorTile;
            params.dstNzNStride = 1;
            params.dstNzMatrixStride = 0;
            const uint64_t src = (static_cast<uint64_t>(token0) + sliceStart) * latentDim_ + k0;
            const uint32_t dst = slice * kVectorTile * kExtent;
            DataCopy(aL1[dst], cGm_[src], params);
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

    static constexpr uint32_t kL0ABytes = kCubeGroup * kTile * sizeof(uint16_t);
    static constexpr uint32_t kL0BBytes = kTile * kTile * sizeof(uint16_t);
    static constexpr uint32_t kBOffset = kL0ABytes;

    GlobalTensor<T> cGm_;
    GlobalTensor<T> rotGm_;
    GlobalTensor<half> bridge_;
    uint32_t numTokens_;
    uint32_t latentDim_;
    int8_t evMte1M_[2];
    int8_t evMMte1_[2];
    int8_t evMte1Mte2_;
    int8_t evMte2Mte1_;
    int8_t evMFix_;
    int8_t evFixM_;
};

template <typename MetaT>
class VectorPack {
public:
    __aicore__ inline VectorPack(
        GlobalTensor<half> bridge,
        GlobalTensor<int32_t> slots,
        GlobalTensor<uint8_t> cache,
        uint32_t numTokens,
        uint32_t latentDim,
        uint32_t blockSize,
        uint32_t numBlocks)
        : bridge_(bridge),
          slots_(slots),
          cache_(cache),
          numTokens_(numTokens),
          latentDim_(latentDim),
          blockSize_(blockSize),
          numBlocks_(numBlocks) {}

    __aicore__ inline void Process(uint32_t core, uint32_t dataCores, uint32_t aivSlice) {
        const uint32_t groups = (numTokens_ + kCubeGroup - 1) / kCubeGroup;
        if (core >= dataCores || core >= groups) {
            return;
        }
        InitUb();
        CrossSet<PIPE_MTE2>(kFlagFree);
        for (uint32_t group = core; group < groups; group += dataCores) {
            const uint32_t token0 = group * kCubeGroup;
            const uint32_t valid = UMin(kCubeGroup, numTokens_ - token0);
            const uint32_t sliceStart = aivSlice * kVectorTile;
            uint32_t rows = 0;
            if (sliceStart < valid) {
                rows = UMin(kVectorTile, valid - sliceStart);
            }
            CrossWait<PIPE_MTE2>(kFlagReady);
            if (rows > 0) {
                const uint32_t elems = rows * latentDim_;
                DataCopy(yHalf_, bridge_[sliceStart * latentDim_], elems);
                SyncMte2ToV();
                Cast(yFloat_, yHalf_, RoundMode::CAST_NONE, elems);
                PipeBarrier<PIPE_V>();
                Encode(rows, token0 + sliceStart);
            }
            CrossSet<PIPE_MTE2>(kFlagFree);
        }
    }

private:
    __aicore__ inline void InitUb() {
        pipe_.InitBuffer(bufHalf_, kVectorTile * kMaxLatent * sizeof(half));
        pipe_.InitBuffer(bufFloat_, kVectorTile * kMaxLatent * sizeof(float));
        pipe_.InitBuffer(bufI32_, kMaxLatent * sizeof(int32_t));
        pipe_.InitBuffer(bufHalfRow_, kMaxLatent * sizeof(half));
        pipe_.InitBuffer(bufI8_, kMaxLatent * sizeof(int8_t));
        pipe_.InitBuffer(bufPartial_, kVectorTile * (kMaxLatent / 64) * sizeof(float));
        pipe_.InitBuffer(bufBase_, kVectorTile * sizeof(float));
        pipe_.InitBuffer(bufStep_, kVectorTile * sizeof(float));
        pipe_.InitBuffer(bufMeta_, 32);
        yHalf_ = bufHalf_.Get<half>();
        yFloat_ = bufFloat_.Get<float>();
        i32Row_ = bufI32_.Get<int32_t>();
        halfRow_ = bufHalfRow_.Get<half>();
        i8Row_ = bufI8_.Get<int8_t>();
        partial_ = bufPartial_.Get<float>();
        baseF_ = bufBase_.Get<float>();
        stepF_ = bufStep_.Get<float>();
        meta_ = bufMeta_.Get<uint8_t>();
    }

    __aicore__ inline void ReduceRows(LocalTensor<float>& dst, bool isMax, uint32_t rows) {
        const uint32_t nPart = latentDim_ / 64;
        if (isMax) {
            WholeReduceMax<float, false>(
                partial_, yFloat_, 64, rows * nPart, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        } else {
            WholeReduceMin<float, false>(
                partial_, yFloat_, 64, rows * nPart, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        }
        PipeBarrier<PIPE_V>();
        SyncVToS();
        for (uint32_t row = 0; row < rows; ++row) {
            float acc = partial_.GetValue(row * nPart);
            for (uint32_t i = 1; i < nPart; ++i) {
                const float v = partial_.GetValue(row * nPart + i);
                if (isMax) {
                    acc = v > acc ? v : acc;
                } else {
                    acc = v < acc ? v : acc;
                }
            }
            dst.SetValue(row, acc);
        }
    }

    __aicore__ inline void Encode(uint32_t rows, uint32_t tokenBase) {
        ReduceRows(baseF_, true, rows);
        ReduceRows(stepF_, false, rows);
        SyncSToV();
        auto broadcast = bufI32_.Get<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            const float ymax = baseF_.GetValue(row);
            const float ymin = stepF_.GetValue(row);
            float step = (ymax - ymin) * (1.0f / 255.0f);
            if (step < 1.0e-6f) {
                step = 1.0e-6f;
            }
            baseF_.SetValue(row, ymin + 128.0f * step);
            stepF_.SetValue(row, step);

            auto rowF = yFloat_[row * latentDim_];
            Duplicate(broadcast, ymin, latentDim_);
            PipeBarrier<PIPE_V>();
            Sub(rowF, rowF, broadcast, latentDim_);
            PipeBarrier<PIPE_V>();
            Duplicate(broadcast, step, latentDim_);
            PipeBarrier<PIPE_V>();
            Div(rowF, rowF, broadcast, latentDim_);
            PipeBarrier<PIPE_V>();
            Maxs(rowF, rowF, 0.0f, latentDim_);
            PipeBarrier<PIPE_V>();
            Mins(rowF, rowF, 255.0f, latentDim_);
            PipeBarrier<PIPE_V>();
            // q in [0, 255]. Scheme A stores int8(q-128), whose bits are q ^ 0x80.
            // half->int8 CAST_NONE writes zeros here, so cast to uint8 and flip bit 7.
            Cast(halfRow_, rowF, RoundMode::CAST_RINT, latentDim_);
            PipeBarrier<PIPE_V>();
            auto codeBytes = i8Row_.template ReinterpretCast<uint8_t>();
            const uint32_t castCount = (latentDim_ + 127U) & ~127U;
            SyncVToS();
            for (uint32_t i = latentDim_; i < castCount; ++i) {
                halfRow_.SetValue(i, static_cast<half>(0.0f));
            }
            SyncSToV();
            Cast(codeBytes, halfRow_, RoundMode::CAST_NONE, castCount);
            PipeBarrier<PIPE_V>();
            auto packed = codeBytes.template ReinterpretCast<uint16_t>();
            auto mask = bufI32_.Get<uint16_t>();
            uint32_t xorCount = ((latentDim_ / 2U) + 127U) & ~127U;
            if (xorCount > kMaxLatent / sizeof(uint16_t)) {
                xorCount = kMaxLatent / sizeof(uint16_t);
            }
            Duplicate(mask, static_cast<uint16_t>(0x8080), xorCount);
            PipeBarrier<PIPE_V>();
            Xor(packed, packed, mask, xorCount);
            PipeBarrier<PIPE_V>();
            WriteRow(tokenBase + row, codeBytes);
        }
        WriteMetaBlock(tokenBase, rows, true);
        WriteMetaBlock(tokenBase, rows, false);
    }

    __aicore__ inline void WriteMetaBlock(uint32_t tokenBase, uint32_t rows, bool isBase) {
        constexpr uint32_t kCastCount = 64;
        auto src = isBase ? baseF_ : stepF_;
        for (uint32_t row = 0; row < rows; ++row) {
            yFloat_.SetValue(row, src.GetValue(row));
        }
        for (uint32_t row = rows; row < kCastCount; ++row) {
            yFloat_.SetValue(row, 0.0f);
        }
        SyncSToV();
        auto metaRow = halfRow_.template ReinterpretCast<MetaT>();
        Cast(metaRow, yFloat_, RoundMode::CAST_RINT, kCastCount);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        auto aligned = meta_.template ReinterpretCast<MetaT>();
        for (uint32_t row = 0; row < rows; ++row) {
            const int32_t slot = slots_.GetValue(tokenBase + row);
            if (slot < 0) {
                continue;
            }
            const uint32_t slotU = static_cast<uint32_t>(slot);
            if (slotU >= numBlocks_ * blockSize_) {
                continue;
            }
            aligned.SetValue(0, metaRow.GetValue(row));
            const uint32_t blk = slotU / blockSize_;
            const uint32_t pos = slotU % blockSize_;
            const uint64_t stride = static_cast<uint64_t>(blockSize_) * (latentDim_ + 4);
            const uint64_t region = isBase
                ? static_cast<uint64_t>(blockSize_) * latentDim_
                : static_cast<uint64_t>(blockSize_) * (latentDim_ + 2);
            CopyBytesToGm(cache_, static_cast<uint64_t>(blk) * stride + region + static_cast<uint64_t>(pos) * 2, meta_, sizeof(MetaT));
        }
    }

    __aicore__ inline void WriteRow(uint32_t token, LocalTensor<uint8_t>& codes) {
        const int32_t slot = slots_.GetValue(token);
        if (slot < 0) {
            return;
        }
        const uint32_t slotU = static_cast<uint32_t>(slot);
        if (slotU >= numBlocks_ * blockSize_) {
            return;
        }
        const uint32_t blk = slotU / blockSize_;
        const uint32_t pos = slotU % blockSize_;
        const uint64_t stride = static_cast<uint64_t>(blockSize_) * (latentDim_ + 4);
        const uint64_t offset = static_cast<uint64_t>(blk) * stride + static_cast<uint64_t>(pos) * latentDim_;
        CopyBytesToGm(cache_, offset, codes, latentDim_);
    }

    GlobalTensor<half> bridge_;
    GlobalTensor<int32_t> slots_;
    GlobalTensor<uint8_t> cache_;
    uint32_t numTokens_;
    uint32_t latentDim_;
    uint32_t blockSize_;
    uint32_t numBlocks_;
    TPipe pipe_;
    TBuf<TPosition::VECIN> bufHalf_;
    TBuf<TPosition::VECCALC> bufFloat_;
    TBuf<TPosition::VECCALC> bufI32_;
    TBuf<TPosition::VECCALC> bufHalfRow_;
    TBuf<TPosition::VECCALC> bufI8_;
    TBuf<TPosition::VECCALC> bufPartial_;
    TBuf<TPosition::VECCALC> bufBase_;
    TBuf<TPosition::VECCALC> bufStep_;
    TBuf<TPosition::VECCALC> bufMeta_;
    LocalTensor<half> yHalf_;
    LocalTensor<float> yFloat_;
    LocalTensor<int32_t> i32Row_;
    LocalTensor<half> halfRow_;
    LocalTensor<int8_t> i8Row_;
    LocalTensor<float> partial_;
    LocalTensor<float> baseF_;
    LocalTensor<float> stepF_;
    LocalTensor<uint8_t> meta_;
};

template <typename T>
__aicore__ inline void Run(
    GM_ADDR c,
    GM_ADDR rotation,
    GM_ADDR slotMapping,
    GM_ADDR latentCache,
    GM_ADDR workspace,
    uint32_t numTokens,
    uint32_t latentDim,
    uint32_t blockSize,
    uint32_t numBlocks,
    uint32_t dataCores) {
    const uint32_t core = CoreId();
    __gm__ uint8_t* raw = reinterpret_cast<__gm__ uint8_t*>(workspace);
    SetSysWorkspace(raw);
    if (GetSysWorkSpacePtr() == nullptr || core >= dataCores) {
        return;
    }
    __gm__ uint8_t* bridgeBytes =
        raw + kWorkspaceHead + static_cast<uint64_t>(core) * kCubeGroup * latentDim * sizeof(half);
    GlobalTensor<half> bridge;
    bridge.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bridgeBytes), kCubeGroup * latentDim);

    if ASCEND_IS_AIC {
        GlobalTensor<T> cGm;
        GlobalTensor<T> rotGm;
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(c), static_cast<uint64_t>(numTokens) * latentDim);
        rotGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rotation), static_cast<uint64_t>(latentDim) * latentDim);
        TPipe pipe;
        CubePack<T> cube(cGm, rotGm, bridge, numTokens, latentDim);
        cube.Process(core, dataCores);
        return;
    }
    if ASCEND_IS_AIV {
        GlobalTensor<int32_t> slots;
        GlobalTensor<uint8_t> cache;
        slots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotMapping), numTokens);
        const uint64_t stride = static_cast<uint64_t>(blockSize) * (latentDim + 4);
        cache.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(latentCache), static_cast<uint64_t>(numBlocks) * stride);
        const uint32_t aivSlice = static_cast<uint32_t>(GetSubBlockIdx() % kAivCount);
        VectorPack<T> vec(bridge, slots, cache, numTokens, latentDim, blockSize, numBlocks);
        vec.Process(core, dataCores, aivSlice);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void rq_uniform_pack_mla_latent(
    GM_ADDR c,
    GM_ADDR rotation_t,
    GM_ADDR slot_mapping,
    GM_ADDR latent_cache,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        return;
    }
    GET_TILING_DATA(td, tiling);
    if (td.latentDim == 0 || td.latentDim > kMaxLatent || td.numTokens == 0) {
        return;
    }
    if (td.dtype == 0) {
        Run<half>(
            c, rotation_t, slot_mapping, latent_cache, workspace,
            td.numTokens, td.latentDim, td.blockSize, td.numBlocks, td.dataCores);
    } else {
        Run<bfloat16_t>(
            c, rotation_t, slot_mapping, latent_cache, workspace,
            td.numTokens, td.latentDim, td.blockSize, td.numBlocks, td.dataCores);
    }
}
