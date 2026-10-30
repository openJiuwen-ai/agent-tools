/**
 * rq_uniform_pack_sfa — pack already-rotated SFA nope+rope into 388B PA slots.
 *
 * tilingKey=0: legacy scalar EncodeOne (MIX 1:2; AIC idle; int64 slots).
 * tilingKey=1: V2 (pure AIV; int64 slots).
 *
 * Slot [Bn,Bs,1,388]:
 *   [0,256) codes u4 nibble-packed, [256,384) rope, [384,388) fp16 scale+offset
 * Dequant: y = offset + u4(code) * scale
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "rq_uniform_pack_sfa_common.h"
#include "rq_uniform_pack_sfa_vector_v2.h"

using namespace AscendC;
using namespace rq_uniform_sfa;

namespace {

constexpr uint32_t kAivPerCube = 2;

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

__aicore__ inline uint32_t CoreId()
{
    if ASCEND_IS_AIV {
        return static_cast<uint32_t>(GetBlockIdx() / GetSubBlockNum());
    }
    return static_cast<uint32_t>(GetBlockIdx());
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
class VectorPackSfa {
public:
    __aicore__ inline VectorPackSfa(
        GlobalTensor<T> nopeGm,
        GlobalTensor<T> ropeGm,
        GlobalTensor<int64_t> slotsGm,
        GlobalTensor<uint8_t> cacheGm,
        uint32_t numTokens,
        uint32_t blockSize,
        uint32_t numBlocks)
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

    __aicore__ inline void Process(uint32_t core, uint32_t dataCores, uint32_t aivSlice)
    {
        const uint32_t stride = dataCores * kAivPerCube;
        const uint32_t start = core * kAivPerCube + aivSlice;
        for (uint32_t token = start; token < numTokens_; token += stride) {
            EncodeOne(token);
        }
    }

private:
    __aicore__ inline void InitUb()
    {
        pipe_.InitBuffer(bufNope_, NOPE_DIM * sizeof(T));
        pipe_.InitBuffer(bufFloat_, NOPE_DIM * sizeof(float));
        pipe_.InitBuffer(bufPartial_, (NOPE_DIM / 64) * sizeof(float));
        pipe_.InitBuffer(bufHalfRow_, NOPE_DIM * sizeof(half));
        pipe_.InitBuffer(bufScratch_, NOPE_DIM * sizeof(float));
        pipe_.InitBuffer(bufI32_, NOPE_DIM * sizeof(int32_t));
        pipe_.InitBuffer(bufSlot_, SLOT_BYTES);
        pipe_.InitBuffer(bufRope_, ROPE_DIM * sizeof(T));
        nopeLocal_ = bufNope_.Get<T>();
        yFloat_ = bufFloat_.Get<float>();
        partial_ = bufPartial_.Get<float>();
        halfRow_ = bufHalfRow_.Get<half>();
        scratch_ = bufScratch_.Get<float>();
        i32Row_ = bufI32_.Get<int32_t>();
        slotBytes_ = bufSlot_.Get<uint8_t>();
        ropeLocal_ = bufRope_.Get<T>();
    }

    __aicore__ inline void EncodeOne(uint32_t token)
    {
        const int64_t slot = slotsGm_.GetValue(token);
        if (slot < 0) {
            return;
        }
        const uint64_t slotU = static_cast<uint64_t>(slot);
        if (slotU >= static_cast<uint64_t>(numBlocks_) * blockSize_) {
            return;
        }

        DataCopyExtParams inParams{1, static_cast<uint32_t>(NOPE_DIM * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(nopeLocal_, nopeGm_[static_cast<uint64_t>(token) * NOPE_DIM], inParams, pad);
        SyncMte2ToV();
        Cast(yFloat_, nopeLocal_, RoundMode::CAST_NONE, NOPE_DIM);
        PipeBarrier<PIPE_V>();

        constexpr uint32_t kPart = NOPE_DIM / 64;
        WholeReduceMax<float, false>(
            partial_, yFloat_, 64, kPart, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        float ymax = partial_.GetValue(0);
        for (uint32_t i = 1; i < kPart; ++i) {
            const float v = partial_.GetValue(i);
            ymax = v > ymax ? v : ymax;
        }
        SyncSToV();
        WholeReduceMin<float, false>(
            partial_, yFloat_, 64, kPart, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        float ymin = partial_.GetValue(0);
        for (uint32_t i = 1; i < kPart; ++i) {
            const float v = partial_.GetValue(i);
            ymin = v < ymin ? v : ymin;
        }
        float scale = (ymax - ymin) * (1.0f / 15.0f);
        if (scale < 1.0e-6f) {
            scale = 1.0e-6f;
        }
        const float offset = ymin;

        SyncSToV();
        Duplicate(scratch_, offset, NOPE_DIM);
        PipeBarrier<PIPE_V>();
        Sub(yFloat_, yFloat_, scratch_, NOPE_DIM);
        PipeBarrier<PIPE_V>();
        Duplicate(scratch_, scale, NOPE_DIM);
        PipeBarrier<PIPE_V>();
        Div(yFloat_, yFloat_, scratch_, NOPE_DIM);
        PipeBarrier<PIPE_V>();
        Maxs(yFloat_, yFloat_, 0.0f, NOPE_DIM);
        PipeBarrier<PIPE_V>();
        Mins(yFloat_, yFloat_, 15.0f, NOPE_DIM);
        PipeBarrier<PIPE_V>();
        Cast(halfRow_, yFloat_, RoundMode::CAST_RINT, NOPE_DIM);
        PipeBarrier<PIPE_V>();
        Cast(i32Row_, halfRow_, RoundMode::CAST_RINT, NOPE_DIM);
        PipeBarrier<PIPE_V>();
        SyncVToS();

        for (uint32_t i = 0; i < CODE_BYTES; ++i) {
            int32_t even = i32Row_.GetValue(i * 2);
            int32_t odd = i32Row_.GetValue(i * 2 + 1);
            if (even < 0) {
                even = 0;
            }
            if (even > 15) {
                even = 15;
            }
            if (odd < 0) {
                odd = 0;
            }
            if (odd > 15) {
                odd = 15;
            }
            const uint8_t packed =
                static_cast<uint8_t>((static_cast<uint32_t>(even) & 0x0Fu) |
                                     ((static_cast<uint32_t>(odd) & 0x0Fu) << 4));
            slotBytes_.SetValue(i, packed);
        }

        DataCopyExtParams ropeParams{1, static_cast<uint32_t>(ROPE_DIM * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> ropePad{false, 0, 0, 0};
        DataCopyPad(ropeLocal_, ropeGm_[static_cast<uint64_t>(token) * ROPE_DIM], ropeParams, ropePad);
        SyncMte2ToV();
        SyncVToS();
        auto ropeU8 = ropeLocal_.template ReinterpretCast<uint8_t>();
        for (uint32_t i = 0; i < ROPE_BYTES; ++i) {
            slotBytes_.SetValue(ROPE_OFF + i, ropeU8.GetValue(i));
        }

        auto metaH = slotBytes_.template ReinterpretCast<half>();
        metaH.SetValue(SCALE_OFF / sizeof(half), static_cast<half>(scale));
        metaH.SetValue(OFFSET_OFF / sizeof(half), static_cast<half>(offset));

        CopyBytesToGm(cacheGm_, slotU * SLOT_BYTES, slotBytes_, SLOT_BYTES);
    }

    GlobalTensor<T> nopeGm_;
    GlobalTensor<T> ropeGm_;
    GlobalTensor<int64_t> slotsGm_;
    GlobalTensor<uint8_t> cacheGm_;
    uint32_t numTokens_;
    uint32_t blockSize_;
    uint32_t numBlocks_;
    TPipe pipe_;
    TBuf<TPosition::VECIN> bufNope_;
    TBuf<TPosition::VECCALC> bufFloat_;
    TBuf<TPosition::VECCALC> bufPartial_;
    TBuf<TPosition::VECCALC> bufHalfRow_;
    TBuf<TPosition::VECCALC> bufScratch_;
    TBuf<TPosition::VECCALC> bufI32_;
    TBuf<TPosition::VECCALC> bufSlot_;
    TBuf<TPosition::VECIN> bufRope_;
    LocalTensor<T> nopeLocal_;
    LocalTensor<float> yFloat_;
    LocalTensor<float> partial_;
    LocalTensor<half> halfRow_;
    LocalTensor<float> scratch_;
    LocalTensor<int32_t> i32Row_;
    LocalTensor<uint8_t> slotBytes_;
    LocalTensor<T> ropeLocal_;
};

template <typename T>
__aicore__ inline void RunLegacy(
    GM_ADDR nope,
    GM_ADDR rope,
    GM_ADDR slotMapping,
    GM_ADDR kvCache,
    GM_ADDR workspace,
    uint32_t numTokens,
    uint32_t blockSize,
    uint32_t numBlocks,
    uint32_t dataCores)
{
    const uint32_t core = CoreId();
    __gm__ uint8_t* raw = reinterpret_cast<__gm__ uint8_t*>(workspace);
    SetSysWorkspace(raw);
    if (GetSysWorkSpacePtr() == nullptr || core >= dataCores) {
        return;
    }
    if ASCEND_IS_AIC {
        return;
    }
    if ASCEND_IS_AIV {
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
        const uint32_t aivSlice = static_cast<uint32_t>(GetSubBlockIdx() % kAivPerCube);
        VectorPackSfa<T> vec(nopeGm, ropeGm, slots, cache, numTokens, blockSize, numBlocks);
        vec.Process(core, dataCores, aivSlice);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void rq_uniform_pack_sfa(
    GM_ADDR nope,
    GM_ADDR rope,
    GM_ADDR slot_mapping,
    GM_ADDR kv_cache,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    if (td.numTokens == 0) {
        return;
    }

    if (TILING_KEY_IS(0)) {
        // Legacy scalar EncodeOne (MIX; int64 slots).
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        if (td.dtype == 0) {
            RunLegacy<half>(
                nope, rope, slot_mapping, kv_cache, workspace, td.numTokens, td.blockSize,
                td.numBlocks, td.dataCores);
        } else {
            RunLegacy<bfloat16_t>(
                nope, rope, slot_mapping, kv_cache, workspace, td.numTokens, td.blockSize,
                td.numBlocks, td.dataCores);
        }
        return;
    }

    if (TILING_KEY_IS(1)) {
        // V2: batch encode + vector nibble (AIV-only; int64 slots).
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        if (td.dtype == 0) {
            rq_uniform_pack_sfa_v2::Run<half>(
                nope, rope, slot_mapping, kv_cache, workspace, td.numTokens, td.blockSize,
                td.numBlocks, td.dataCores, td.tileRows);
        } else {
            rq_uniform_pack_sfa_v2::Run<bfloat16_t>(
                nope, rope, slot_mapping, kv_cache, workspace, td.numTokens, td.blockSize,
                td.numBlocks, td.dataCores, td.tileRows);
        }
    }
}
