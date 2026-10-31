/*!
 * \file kv_quant_sparse_flash_attention_service_vector_mla.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H

// TQ4 per-token scale staging reuses the upper half of kvValidSizeGm_.
// The first 1K int32 values remain the legacy valid-size ring buffer; the
// upper 1K int32 values are viewed as 4 x 512 FP16 scales (one slot per loop).
static constexpr uint32_t TQ4_SCALE_HALF_BASE = 2048U;
static constexpr uint32_t TQ4_SCALE_SLOT_STRIDE = 512U;
static_assert(TQ4_SCALE_SLOT_STRIDE * sizeof(uint16_t) <= 1024U,
              "TQ4 half staging exceeds the first 1K of tq4ScaleBuf_");
static constexpr uint32_t TQ4_SCALE_UB_F32 = 256U;
static_assert(TQ4_SCALE_UB_F32 * sizeof(float) >= 1024U, "TQ4 fp32 area must start after the half staging area");
static_assert((TQ4_SCALE_UB_F32 + TQ4_SCALE_SLOT_STRIDE) * sizeof(float) <= 4096U,
              "TQ4 fp32 scale exceeds tq4ScaleBuf_ (4K)");

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "rq_uniform_kv_quant_sparse_flash_attention_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename QSFAT>
class QSFAVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename QSFAT::kvType;
    using K_ROPE_T = typename QSFAT::kRopeType;
    using OUT_T = typename QSFAT::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;
    bool NO_AMLA = true;

    __aicore__ inline QSFAVectorService(){};
    __aicore__ inline void ProcessVec1L(const RunInfo &kvVecInfo);
    __aicore__ inline void ProcessVec2L(const RunInfo &kvVecInfo);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &kvVecConstInfo,
                                      const RqUniformKvQuantSparseFlashAttentionTilingDataMla *__restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                const GlobalTensor<K_ROPE_T> &kvMergeGm,
                                                const GlobalTensor<K_ROPE_T> &keyRopeGm,
                                                const GlobalTensor<KV_T> &keyGm,
                                                const GlobalTensor<int32_t> &blkTableGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_ROPE_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t kvDealRows, uint32_t kvColumns, uint32_t kvActualColumns);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t kvDealRows, uint32_t kvColumns, uint32_t kvActualColumns);
    // ================================Vector0==========================================
    __aicore__ inline void MergeKv(const RunInfo &kvVecRunInfo);
    __aicore__ inline int64_t GetKeyBNBOffset(int64_t realS2Idx, const RunInfo &kvVecRunInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &kvVecRunInfo);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t kvDealRows, uint32_t kvColumns,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T> &mmResUb, uint32_t kvDealRows, uint32_t kvColumns,
                                     uint64_t startId, uint64_t endId);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &kvVecRunInfo);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &kvVecRunInfo);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &kvVecRunInfo);
    // Decode packed TQ4 slots into the latent/nope workspace. Each slot is
    // 256B int4 codes followed by 64 RoPE values and one FP16 scale.
    __aicore__ inline void Tq4DequantRows(LocalTensor<KV_T> &srcTensor, LocalTensor<K_ROPE_T> &dstB16, int32_t dealRow);
    __aicore__ inline void DequantSfaRows(LocalTensor<KV_T> &srcTensor, LocalTensor<K_ROPE_T> &dstB16, int32_t dealRow);
    // Signed int4 + 8. Defined in rq_uniform_kv_quant_sparse_flash_attention_dequant_v2.h.
    __aicore__ inline void DequantSfaRowsV2(LocalTensor<KV_T> &srcTensor, LocalTensor<K_ROPE_T> &dstNope,
                                            LocalTensor<K_ROPE_T> &dstRope, int32_t row0, int32_t n);
    __aicore__ inline void CopyOutSfaV2(LocalTensor<KV_T> &srcTensor, int32_t dealRow, int64_t s2GmStartOffset,
                                        int64_t mte3Size, const RunInfo &kvVecRunInfo);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &kvVecInfo, const MSplitInfo &kvVecSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &kvVecInfo, const MSplitInfo &kvVecSplitInfo,
                                                uint32_t startRow, uint32_t kvDealRows, uint32_t kvColumns,
                                                uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &kvVecInfo, const MSplitInfo &kvVecSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t kvDealRows, uint32_t kvColumns,
                                                 uint32_t kvActualColumns);
    __aicore__ inline void ElewiseCompute(const RunInfo &kvVecInfo, const LocalTensor<T> &mmResUb, uint32_t kvDealRows,
                                          uint32_t kvColumns);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &kvVecInfo, const MSplitInfo &kvVecSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &kvVecInfo, const MSplitInfo &kvVecSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &kvVecInfo, const MSplitInfo &kvVecSplitInfo,
                                                uint32_t startRow, uint32_t kvDealRows, uint32_t kvColumns,
                                                uint32_t kvActualColumns);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &kvVecInfo, const MSplitInfo &kvVecSplitInfo,
                                            uint32_t mStartRow, uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &kvVecInfo, LocalTensor<OUT_T> &attenOutUb,
                                                uint32_t kvWorkspaceRow, uint32_t kvDealRows, uint32_t kvColumns,
                                                uint32_t kvActualColumns);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &kvVecInfo, LocalTensor<T> &kvBmm2ResultUb,
                                          uint32_t kvWorkspaceRow, uint32_t kvDealRows, uint32_t kvColumns,
                                          uint32_t kvActualColumns);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &kvVecInfo, LocalTensor<T> &kvBmm2ResultUb,
                                              uint32_t kvWorkspaceRow, uint32_t kvDealRows, uint32_t kvColumns,
                                              uint32_t kvActualColumns);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &kvVecInfo, LocalTensor<T> &kvBmm2ResultUb,
                                             uint32_t kvWorkspaceRow, uint32_t kvDealRows, uint32_t kvColumns,
                                             uint32_t kvActualColumns);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = QSFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = QSFAT::templateMode;
    static constexpr bool FLASH_DECODE = QSFAT::flashDecode;
    static constexpr QSFA_LAYOUT LAYOUT_T = QSFAT::layout;
    static constexpr QSFA_LAYOUT KV_LAYOUT_T = QSFAT::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t TQ4_SCALE_SYNC_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_512B / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T); // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                               // 32/4=8
    static constexpr uint32_t LIMIT_DEAL_ROW = 16U;
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr int32_t TQ4_DEQUANT_CHUNK = IsSameType<K_ROPE_T, bfloat16_t>::value ? 16 : 4;
    static constexpr uint32_t TQ4_NOPE_BYTES = 256U;

    const RqUniformKvQuantSparseFlashAttentionTilingDataMla *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo kvVecConstInfo = {};

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_ROPE_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<K_ROPE_T> kvMergeGm_;
    GlobalTensor<K_ROPE_T> keyRopeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;
    GlobalTensor<uint16_t> tq4ScaleGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff1;  // 32K * 2
    TBuf<> inputBuff2;  // 32K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    TBuf<> tmpBuff1;        // 32K
    TBuf<> tmpBuff2;        // 8K
    TBuf<> v0ValidSizeBuff; // 8K

    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxMaxDefaultBuff; // 1K
    TBuf<> softmaxSumDefaultBuff; // 1K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;

    TBuf<> tq4CentBuf_;
    TBuf<> tq4ByteLutBuf_;
    TBuf<> tq4STIdxBuf_;
    TBuf<> tq4ScaleBuf_;
    LocalTensor<float> tq4Cent_;
};

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, ConstInfo::BUFFER_SIZE_BYTE_8K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * kvVecConstInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * kvVecConstInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * kvVecConstInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff1.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();

    // TQ4 setup is done once per AIV. The index table is used by vectorized
    // scale export; the byte LUT packs two BF16 centroids into one uint32.
    pipe->InitBuffer(tq4STIdxBuf_, ConstInfo::BUFFER_SIZE_BYTE_512B);
    {
        LocalTensor<uint32_t> qsfaSTIdxInit = tq4STIdxBuf_.Get<uint32_t>();
        for (uint32_t i = 0; i < 128U; ++i) {
            qsfaSTIdxInit.SetValue(i, i * 32U);
        }
    }
    pipe->InitBuffer(tq4ScaleBuf_, ConstInfo::BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(tq4CentBuf_, ConstInfo::BUFFER_SIZE_BYTE_256B);
    pipe->InitBuffer(tq4ByteLutBuf_, ConstInfo::BUFFER_SIZE_BYTE_1K);
    tq4Cent_ = tq4CentBuf_.Get<float>();
    tq4Cent_.SetValue(0, 0.00547294f);
    tq4Cent_.SetValue(1, 0.01680406f);
    tq4Cent_.SetValue(2, 0.02857605f);
    tq4Cent_.SetValue(3, 0.04108622f);
    tq4Cent_.SetValue(4, 0.05492980f);
    tq4Cent_.SetValue(5, 0.07101817f);
    tq4Cent_.SetValue(6, 0.09115373f);
    tq4Cent_.SetValue(7, 0.12037795f);
    tq4Cent_.SetValue(8, -0.12091285f);
    tq4Cent_.SetValue(9, -0.09111122f);
    tq4Cent_.SetValue(10, -0.07112455f);
    tq4Cent_.SetValue(11, -0.05513602f);
    tq4Cent_.SetValue(12, -0.04132067f);
    tq4Cent_.SetValue(13, -0.02874970f);
    tq4Cent_.SetValue(14, -0.01700489f);
    tq4Cent_.SetValue(15, -0.00568677f);
    if constexpr (IsSameType<K_ROPE_T, bfloat16_t>::value) {
        LocalTensor<uint32_t> qsfaByteLut = tq4ByteLutBuf_.Get<uint32_t>();
        for (uint32_t hi = 0; hi < 16U; ++hi) {
            union {
                float f;
                uint32_t u;
            } cvtHi;
            cvtHi.f = tq4Cent_.GetValue(hi ^ 8U);
            uint32_t qsfaHiBits = (cvtHi.u + 0x7FFFU + ((cvtHi.u >> 16) & 1U)) >> 16;
            for (uint32_t lo = 0; lo < 16U; ++lo) {
                union {
                    float f;
                    uint32_t u;
                } cvtLo;
                cvtLo.f = tq4Cent_.GetValue(lo ^ 8U);
                uint32_t qsfaLoBits = (cvtLo.u + 0x7FFFU + ((cvtLo.u >> 16) & 1U)) >> 16;
                qsfaByteLut.SetValue(hi * 16U + lo, (qsfaHiBits << 16) | qsfaLoBits);
            }
        }
    }
    // Keep the aligned tail finite and avoid a non-32B vector tail write
    // when the valid sequence is shorter than 512 columns.
    Duplicate(tq4ScaleBuf_.Get<T>()[TQ4_SCALE_UB_F32], static_cast<T>(1.0), TQ4_SCALE_SLOT_STRIDE);
    PipeBarrier<PIPE_ALL>();
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitParams(
    const struct ConstInfo &kvVecConstInfo, const RqUniformKvQuantSparseFlashAttentionTilingDataMla *__restrict tilingData)
{
    this->kvVecConstInfo = kvVecConstInfo;
    this->tilingData = tilingData;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                                      const GlobalTensor<K_ROPE_T> &kvMergeGm,
                                                                      const GlobalTensor<K_ROPE_T> &keyRopeGm,
                                                                      const GlobalTensor<KV_T> &keyGm,
                                                                      const GlobalTensor<int32_t> &blkTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->keyRopeGm_ = keyRopeGm;
    this->keyGm_ = keyGm;
    this->blkTableGm_ = blkTableGm;
    this->kvValidSizeGm_ = kvValidSizeGm;
    this->tq4ScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(kvValidSizeGm.GetPhyAddr(0)));
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_ROPE_T> vec1ResGm, GlobalTensor<int32_t> actualSeqLengthsQGm,
    GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm,
    GlobalTensor<int32_t> topKGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                      GlobalTensor<T> vec2ResGm,
                                                                      GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                      GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET);
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &kvVecInfo,
                                                                             const MSplitInfo &kvVecSplitInfo,
                                                                             LocalTensor<T> &softmaxSumUb,
                                                                             LocalTensor<T> &softmaxMaxUb)
{
    if (kvVecSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t qsfaBaseOffset = kvVecSplitInfo.nBufferStartM / 2;
    size_t qsfaSize = kvVecSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t qsfaAccumTmpOutNum = CalcAccumOffset(kvVecInfo.bIdx, kvVecInfo.gS1Idx);
    uint64_t qsfaOffset =
        (qsfaAccumTmpOutNum * kvVecConstInfo.kvHeadNum * kvVecConstInfo.mBaseSize +               // taskoffset
         kvVecInfo.tndCoreStartKVSplitPos * kvVecConstInfo.kvHeadNum * kvVecConstInfo.mBaseSize + // 份数offset
         kvVecSplitInfo.nBufferStartM + kvVecSplitInfo.vecStartM) *
        FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if (kvVecInfo.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> qsfaTmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(qsfaTmp, softmaxSumUb[qsfaBaseOffset], (kvVecSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[qsfaOffset], qsfaTmp, qsfaSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        qsfaTmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(qsfaTmp, softmaxMaxUb[qsfaBaseOffset], (kvVecSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[qsfaOffset], qsfaTmp, qsfaSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[qsfaOffset], qsfaSize, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[qsfaOffset], qsfaSize, SOFTMAX_MIN_NUM);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ElewiseCompute(const RunInfo &kvVecInfo, const LocalTensor<T> &mmResUb,
                                                                uint32_t kvDealRows, uint32_t kvColumns)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), kvDealRows * kvColumns);
    if (kvVecConstInfo.keyQuantMode == QUANT_MODE::TQ4) {
        LocalTensor<T> qsfaColScale = tq4ScaleBuf_.Get<T>()[TQ4_SCALE_UB_F32];
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < kvDealRows; ++r) {
            Mul(mmResUb[r * kvColumns], mmResUb[r * kvColumns], qsfaColScale, kvColumns);
        }
        PipeBarrier<PIPE_V>();
    }
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // v0的无效值判断
        uint64_t qsfaS2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + kvVecInfo.loop % MERGE_CACHE_GM_BUF_NUM);
        uint64_t qsfaS2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + kvVecInfo.loop % MERGE_CACHE_GM_BUF_NUM);

        int64_t qsfaS2ProcessSize = kvVecInfo.actualSingleProcessSInnerSize;
        int64_t qsfaS2Pair = CeilDiv(qsfaS2ProcessSize, 2L * kvVecConstInfo.sparseBlockSize);
        int64_t qsfaS2Mid = CeilDiv(qsfaS2Pair, 2L) * 2 * kvVecConstInfo.sparseBlockSize;
        if (qsfaS2Mid > qsfaS2ProcessSize) {
            qsfaS2Mid = qsfaS2ProcessSize;
        }
        if (unlikely(qsfaS2ValidSizeFirstPart < qsfaS2Mid)) {
            int64_t qsfaS2StartCeilAlign = CeilAlign(qsfaS2ValidSizeFirstPart, 8);
            int64_t qsfaS2MidFloorAlign = qsfaS2Mid / 8 * 8;
            // 场景一 s2Mid > s2ValidSizeFirstPart + oneBlk
            // 可以推导出s2StartCeilAlign < s2Mid   第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2MidFloorAlign 第二阶段取到s2MidFloorAlign
            // 场景二 s2Mid <= s2ValidSizeFirstPart + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2Mid 第一阶段取到mid
            // s2StartCeilAlign > s2MidFloorAlign 第二阶段取到s2StartCeilAlign
            SetInfInBlk(mmResUb, kvDealRows, kvColumns, qsfaS2ValidSizeFirstPart,
                        qsfaS2StartCeilAlign >= qsfaS2Mid ? qsfaS2Mid : qsfaS2StartCeilAlign);
            SetMidInf(mmResUb, kvDealRows, kvColumns, qsfaS2StartCeilAlign, qsfaS2MidFloorAlign);
            SetInfInBlk(mmResUb, kvDealRows, kvColumns,
                        qsfaS2StartCeilAlign <= qsfaS2MidFloorAlign ? qsfaS2MidFloorAlign : qsfaS2StartCeilAlign,
                        qsfaS2Mid);
        }
        if (unlikely(qsfaS2ValidSizeSecondPart < qsfaS2ProcessSize - qsfaS2Mid)) {
            // 场景一 s2Mid + s2ValidSizeSecondPart > s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign < s2ProcessSize 第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2EndFloorAlign 第二阶段取到s2EndFloorAlign
            // 场景二 s2Mid + s2ValidSizeSecondPart <= s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2ProcessSize 第一阶段取到s2ProcessSize
            // s2StartCeilAlign > s2EndFloorAlign 第二阶段取到s2StartCeilAlign
            int64_t qsfaS2StartCeilAlign = CeilAlign(qsfaS2Mid + qsfaS2ValidSizeSecondPart, 8);
            int64_t qsfaS2EndFloorAlign = qsfaS2ProcessSize / 8 * 8;
            SetInfInBlk(mmResUb, kvDealRows, kvColumns, qsfaS2Mid + qsfaS2ValidSizeSecondPart,
                        qsfaS2StartCeilAlign >= qsfaS2ProcessSize ? qsfaS2ProcessSize : qsfaS2StartCeilAlign);
            SetMidInf(mmResUb, kvDealRows, kvColumns, qsfaS2StartCeilAlign, qsfaS2EndFloorAlign);
            SetInfInBlk(mmResUb, kvDealRows, kvColumns,
                        qsfaS2StartCeilAlign <= qsfaS2EndFloorAlign ? qsfaS2EndFloorAlign : qsfaS2StartCeilAlign,
                        qsfaS2ProcessSize);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t kvDealRows,
                                                             uint32_t kvColumns, uint64_t startId, uint64_t endId)
{
    //       startId     endId
    // x x x   0      0   0     x x x
    // 从startId到endId部分置-inf, endId、startId为endId一个blk内部的下标
    if (startId >= endId) {
        return;
    }

    uint64_t qsfaStartFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t qsfaNotComputePreMaskOneBlk = (1 << (startId - qsfaStartFloorAlignSize)) - 1;
    uint64_t qsfaNotComputePostMaskOneBlk = ~((1 << (endId - qsfaStartFloorAlignSize)) - 1);
    uint64_t qsfaNotComputeMaskOneBlk = qsfaNotComputePreMaskOneBlk ^ qsfaNotComputePostMaskOneBlk;

    uint64_t qsfaMaskOneBlk = ~qsfaNotComputeMaskOneBlk;
    uint64_t mask[1] = {qsfaMaskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (qsfaMaskOneBlk << (i * 8));
    }
    for (uint64_t qsfaRowId = 0; qsfaRowId < kvDealRows; qsfaRowId += 8) {
        Duplicate(mmResUb[qsfaRowId * kvColumns + qsfaStartFloorAlignSize], SOFTMAX_MIN_NUM, mask, 1,
                  CeilDiv(kvColumns, 8), 0);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SetMidInf(const LocalTensor<T> &mmResUb, uint32_t kvDealRows,
                                                           uint32_t kvColumns, uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    // startId        endId
    //    0      ...    0
    // 从startId到endId部分置-inf, startId、endId为32B对齐的下标
    for (uint64_t qsfaRowId = 0; qsfaRowId < kvDealRows; qsfaRowId++) {
        Duplicate(mmResUb[qsfaRowId * kvColumns + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SoftmaxFlashV2Compute(const RunInfo &kvVecInfo,
                                                                       const MSplitInfo &kvVecSplitInfo,
                                                                       LocalTensor<T> &mmResUb,
                                                                       LocalTensor<uint8_t> &softmaxTmpUb,
                                                                       uint32_t startRow, uint32_t kvDealRows,
                                                                       uint32_t kvColumns, uint32_t kvActualColumns)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = kvVecSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = kvVecInfo.loop % (kvVecConstInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset;
    if (kvVecInfo.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (kvVecInfo.loop - 1) % (kvVecConstInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
    }
    if (kvActualColumns != 0) {
        SoftMaxShapeInfo srcShape{kvDealRows, kvColumns, kvDealRows, kvActualColumns};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, QSFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, kvDealRows);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, kvDealRows);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DealBmm1ResBaseBlock(const RunInfo &kvVecInfo,
                                                                      const MSplitInfo &kvVecSplitInfo,
                                                                      uint32_t startRow, uint32_t kvDealRows,
                                                                      uint32_t kvColumns, uint32_t loopId)
{
    uint32_t qsfaComputeSize = kvDealRows * kvColumns;
    uint64_t qsfaInOutGmOffset = (kvVecInfo.loop % kvVecConstInfo.preLoadNum) * kvVecConstInfo.mmResUbSize +
                                 (kvVecSplitInfo.nBufferStartM + kvVecSplitInfo.vecStartM + startRow) * kvColumns;
    LocalTensor<MM1_OUT_T> qsfaMmResUb = inputBuff1.Get<MM1_OUT_T>();
    qsfaMmResUb = qsfaMmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(qsfaMmResUb, mm1ResGm[qsfaInOutGmOffset], qsfaComputeSize);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (loopId == 0) {
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(kvVecInfo, qsfaMmResUb, kvDealRows, kvColumns);

    PipeBarrier<PIPE_V>();
    LocalTensor<T> qsfaTmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> qsfaSoftmaxTmpUb = qsfaTmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(kvVecInfo, kvVecSplitInfo, qsfaMmResUb, qsfaSoftmaxTmpUb, startRow, kvDealRows, kvColumns,
                          kvVecInfo.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    if (kvVecConstInfo.keyQuantMode == QUANT_MODE::TQ4) {
        // MLA has K=V.  The score side consumed s_j before softmax, so the
        // value side applies the same per-column scale after softmax.
        LocalTensor<T> qsfaScaleF = tq4ScaleBuf_.Get<T>()[TQ4_SCALE_UB_F32];
        for (uint32_t r = 0; r < kvDealRows; ++r) {
            Mul(qsfaMmResUb[r * kvColumns], qsfaMmResUb[r * kvColumns], qsfaScaleF, kvColumns);
        }
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<K_ROPE_T> tmpMMResCastTensor = outputBuff1.Get<K_ROPE_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, qsfaMmResUb, AscendC::RoundMode::CAST_ROUND, qsfaComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[qsfaInOutGmOffset], tmpMMResCastTensor, qsfaComputeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec1SingleBuf(const RunInfo &kvVecInfo,
                                                                      const MSplitInfo &kvVecSplitInfo)
{
    if (kvVecSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t qsfaMSplitSize = kvVecInfo.actualSingleProcessSInnerSize == 0 ?
                                  16 :
                                  (BASE_BLOCK_MAX_ELEMENT_NUM / kvVecInfo.actualSingleProcessSInnerSizeAlign);
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    qsfaMSplitSize = qsfaMSplitSize >> 3U << 3U;

    if (qsfaMSplitSize > kvVecSplitInfo.vecDealM) {
        qsfaMSplitSize = kvVecSplitInfo.vecDealM;
    }
    uint32_t qsfaLoopCount = (kvVecSplitInfo.vecDealM + qsfaMSplitSize - 1) / qsfaMSplitSize;
    uint32_t qsfaTailSplitSize = kvVecSplitInfo.vecDealM - (qsfaLoopCount - 1) * qsfaMSplitSize;

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 256 * sizeof(int32_t);
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        DataCopyPadExtParams<int32_t> padParams;
        // 额外偏移128个元素，避免不同loop下v0和v1互相影响
        DataCopyPad(v0ValidSizeUb_[128], kvValidSizeGm_[kvVecInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)],
                    dataCopyParams, padParams);
        if (kvVecConstInfo.keyQuantMode == QUANT_MODE::TQ4) {
            DataCopyExtParams qsfaScaleInParams;
            qsfaScaleInParams.blockCount = 1;
            qsfaScaleInParams.blockLen = TQ4_SCALE_SLOT_STRIDE * sizeof(uint16_t);
            qsfaScaleInParams.srcStride = 0;
            qsfaScaleInParams.dstStride = 0;
            DataCopyPadExtParams<uint16_t> qsfaScalePad{false, 0, 0, 0};
            DataCopyPad(
                tq4ScaleBuf_.Get<uint16_t>(),
                tq4ScaleGm_[TQ4_SCALE_HALF_BASE + kvVecInfo.loop % MERGE_CACHE_GM_BUF_NUM * TQ4_SCALE_SLOT_STRIDE],
                qsfaScaleInParams, qsfaScalePad);
            // The scale was produced by MTE3 in MergeKv and is consumed by V.
            // PipeBarrier alone cannot order these two pipelines.
            SetFlag<HardEvent::MTE2_V>(TQ4_SCALE_SYNC_FLAG);
            WaitFlag<HardEvent::MTE2_V>(TQ4_SCALE_SYNC_FLAG);
            LocalTensor<half> qsfaScaleH = tq4ScaleBuf_.Get<half>();
            LocalTensor<T> qsfaScaleF = tq4ScaleBuf_.Get<T>()[TQ4_SCALE_UB_F32];
            uint32_t qsfaValidCol = kvVecInfo.actualSingleProcessSInnerSize;
            if (qsfaValidCol > 0) {
                Cast(qsfaScaleF, qsfaScaleH, AscendC::RoundMode::CAST_NONE, qsfaValidCol);
                PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<HardEvent::MTE2_S>(0);
        if (unlikely(qsfaLoopCount == 0)) {
            // scalar同步影响较大，挪到循环内部进行
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    for (uint32_t qsfaI = 0, dealSize = qsfaMSplitSize; qsfaI < qsfaLoopCount; qsfaI++) {
        if (qsfaI == (qsfaLoopCount - 1)) {
            dealSize = qsfaTailSplitSize;
        }
        DealBmm1ResBaseBlock(kvVecInfo, kvVecSplitInfo, qsfaI * qsfaMSplitSize, dealSize,
                             kvVecInfo.actualSingleProcessSInnerSizeAlign, qsfaI);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                              int64_t topkGmBaseOffset, const RunInfo &kvVecRunInfo)
{
    int64_t qsfaTopkGmIdx =
        (s2GmOffset + kvVecRunInfo.s2Idx * kvVecConstInfo.s2BaseSize) / kvVecConstInfo.sparseBlockSize;
    if (unlikely(qsfaTopkGmIdx >= kvVecConstInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx =
        topkGm_.GetValue(topkGmBaseOffset + qsfaTopkGmIdx) * static_cast<int64_t>(kvVecConstInfo.sparseBlockSize) +
        static_cast<int64_t>((s2GmOffset + kvVecRunInfo.s2Idx * kvVecConstInfo.s2BaseSize) %
                             kvVecConstInfo.sparseBlockSize);
}

template <typename QSFAT>
__aicore__ inline int64_t QSFAVectorService<QSFAT>::GetKeyBNBOffset(int64_t realS2Idx, const RunInfo &kvVecRunInfo,
                                                                    int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyBNBOffset = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / kvVecConstInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % kvVecConstInfo.kvCacheBlockSize;
        realKeyBNBOffset = blkTableGm_.GetValue(kvVecRunInfo.bIdx * kvVecConstInfo.maxBlockNumPerBatch + blkTableIdx) *
                               static_cast<int64_t>(kvVecConstInfo.kvCacheBlockSize) *
                               static_cast<int64_t>(kvVecConstInfo.kvHeadNum) +
                           blkTableOffset;
    } else {
        realKeyBNBOffset =
            (kvVecRunInfo.tensorBOffset + realS2Idx * kvVecConstInfo.kvHeadNum * kvVecConstInfo.combineHeadDim) /
            kvVecConstInfo.combineHeadDim;
    }
    return realKeyBNBOffset;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::Tq4DequantRows(LocalTensor<KV_T> &srcTensor,
                                                                LocalTensor<K_ROPE_T> &dstB16, int32_t dealRow)
{
    uint32_t HD = kvVecConstInfo.headDim;
    uint32_t ROW_BYTES =
        QSFAAlign(static_cast<uint32_t>(tilingData->baseParams.dSizeVInput), static_cast<uint32_t>(BYTE_BLOCK));
    constexpr int32_t CHUNK = TQ4_DEQUANT_CHUNK;
    constexpr uint32_t CHUNK_ELEMS = CHUNK * 512U;
    constexpr bool TQ4_FAST_BF16 = IsSameType<K_ROPE_T, bfloat16_t>::value;
    constexpr uint32_t CHUNK_BYTES = CHUNK * 256U;
    constexpr uint32_t HALF_ELEMS = TQ4_FAST_BF16 ? CHUNK_BYTES : CHUNK_ELEMS;
    constexpr uint32_t IDX_ELEMS = TQ4_FAST_BF16 ? CHUNK_BYTES : CHUNK_ELEMS;
    constexpr uint32_t COMPACT_BYTES = TQ4_FAST_BF16 ? CHUNK_BYTES : 0U;
    constexpr uint32_t SHALF_BYTE_OFF = TQ4_FAST_BF16 ? COMPACT_BYTES : CHUNK_ELEMS * sizeof(float);
    constexpr uint32_t IDX_BYTE_OFF = SHALF_BYTE_OFF + HALF_ELEMS * sizeof(half);
    static_assert(IDX_BYTE_OFF + IDX_ELEMS * sizeof(int32_t) <= ConstInfo::BUFFER_SIZE_BYTE_32K,
                  "TQ4 dequant scratch exceeds inputBuff2");

    LocalTensor<half> sHalfBase = inputBuff2.Get<half>()[SHALF_BYTE_OFF / sizeof(half)];
    LocalTensor<int32_t> idxI = inputBuff2.Get<int32_t>()[IDX_BYTE_OFF / sizeof(int32_t)];
    LocalTensor<uint32_t> idxU = inputBuff2.Get<uint32_t>()[IDX_BYTE_OFF / sizeof(uint32_t)];
    LocalTensor<int4b_t> srcI4 = srcTensor.template ReinterpretCast<int4b_t>();

    PipeBarrier<PIPE_ALL>();
    if (unlikely(dealRow <= 0)) {
        return;
    }

    if constexpr (TQ4_FAST_BF16) {
        LocalTensor<uint8_t> compactU8 = inputBuff2.Get<uint8_t>();
        LocalTensor<uint32_t> byteLut = tq4ByteLutBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> dstU32 = dstB16.template ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> srcU8 = srcTensor.template ReinterpretCast<uint8_t>();
        uint16_t nibbleBlk = static_cast<uint16_t>((HD / 2U) / BYTE_BLOCK);
        uint16_t rowGapBlk = static_cast<uint16_t>(ROW_BYTES / BYTE_BLOCK - nibbleBlk);
        for (int32_t base = 0; base < dealRow; base += CHUNK) {
            int32_t cur = (base + CHUNK <= dealRow) ? CHUNK : (dealRow - base);
            uint32_t cnt = static_cast<uint32_t>(cur) * (HD / 2U);
            DataCopyParams compactParams;
            compactParams.blockCount = static_cast<uint16_t>(cur);
            compactParams.blockLen = nibbleBlk;
            compactParams.srcStride = rowGapBlk;
            compactParams.dstStride = 0;
            DataCopy(compactU8, srcU8[base * ROW_BYTES], compactParams);
            PipeBarrier<PIPE_V>();
            Cast(sHalfBase, compactU8, RoundMode::CAST_NONE, cnt);
            PipeBarrier<PIPE_V>();
            Cast(idxI, sHalfBase, RoundMode::CAST_ROUND, cnt);
            PipeBarrier<PIPE_V>();
            ShiftLeft(idxI, idxI, static_cast<int32_t>(2), cnt);
            PipeBarrier<PIPE_V>();
            Gather(dstU32[base * (HD / 2U)], byteLut, idxU, 0, cnt);
            PipeBarrier<PIPE_V>();
        }
    } else {
        LocalTensor<float> workBase = inputBuff2.Get<float>();
        for (int32_t base = 0; base < dealRow; base += CHUNK) {
            int32_t cur = (base + CHUNK <= dealRow) ? CHUNK : (dealRow - base);
            uint32_t cnt = static_cast<uint32_t>(cur) * HD;
            for (int32_t rr = 0; rr < cur; ++rr) {
                Cast(sHalfBase[rr * HD], srcI4[(base + rr) * ROW_BYTES * 2], RoundMode::CAST_NONE, HD);
            }
            PipeBarrier<PIPE_V>();
            Adds(sHalfBase, sHalfBase, static_cast<half>(8.0f), cnt);
            PipeBarrier<PIPE_V>();
            Muls(sHalfBase, sHalfBase, static_cast<half>(4.0f), cnt);
            PipeBarrier<PIPE_V>();
            Cast(idxI, sHalfBase, RoundMode::CAST_ROUND, cnt);
            PipeBarrier<PIPE_V>();
            Gather(workBase, tq4Cent_, idxU, 0, cnt);
            PipeBarrier<PIPE_V>();
            if constexpr (IsSameType<K_ROPE_T, bfloat16_t>::value) {
                Cast(dstB16[base * HD], workBase, RoundMode::CAST_RINT, cnt);
            } else {
                Cast(dstB16[base * HD], workBase, RoundMode::CAST_ROUND, cnt);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    PipeBarrier<PIPE_ALL>();
}

// Schema A: y = offset + u4(code) * scale  (no codebook LUT). Writes [dealRow, 512] into dstB16.
template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DequantSfaRows(LocalTensor<KV_T> &srcTensor,
                                                                LocalTensor<K_ROPE_T> &dstB16, int32_t dealRow)
{
    constexpr uint32_t HD = sfa_kv::NOPE_DIM;
    constexpr uint32_t CODE_BYTES = sfa_kv::CODE_BYTES;
    uint32_t ROW_BYTES =
        QSFAAlign(static_cast<uint32_t>(tilingData->baseParams.dSizeVInput), static_cast<uint32_t>(BYTE_BLOCK));
    if (unlikely(dealRow <= 0)) {
        return;
    }
    LocalTensor<uint8_t> srcU8 = srcTensor.template ReinterpretCast<uint8_t>();
    LocalTensor<float> work = inputBuff2.Get<float>();

    for (int32_t r = 0; r < dealRow; ++r) {
        uint32_t rowOff = static_cast<uint32_t>(r) * ROW_BYTES;
        LocalTensor<half> metaBase = srcU8.template ReinterpretCast<half>();
        float scale = static_cast<float>(metaBase.GetValue((rowOff + sfa_kv::SCALE_OFF) / sizeof(half)));
        float offset = static_cast<float>(metaBase.GetValue((rowOff + sfa_kv::OFFSET_OFF) / sizeof(half)));
        for (uint32_t i = 0; i < CODE_BYTES; ++i) {
            uint8_t packed = srcU8.GetValue(rowOff + i);
            int32_t even = static_cast<int32_t>(packed & 0x0Fu);
            int32_t odd = static_cast<int32_t>((packed >> 4) & 0x0Fu);
            work.SetValue(i * 2U, offset + static_cast<float>(even) * scale);
            work.SetValue(i * 2U + 1U, offset + static_cast<float>(odd) * scale);
        }
        PipeBarrier<PIPE_ALL>();
        if constexpr (IsSameType<K_ROPE_T, bfloat16_t>::value) {
            Cast(dstB16[r * HD], work, RoundMode::CAST_RINT, HD);
        } else {
            Cast(dstB16[r * HD], work, RoundMode::CAST_ROUND, HD);
        }
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size,
                                                                int64_t mergeMte3Idx, int64_t realS2Idx,
                                                                int64_t keyBNBOffset, int64_t s2IdLimit,
                                                                const RunInfo &kvVecRunInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count = ((realS2Idx + kvVecConstInfo.sparseBlockSize > s2IdLimit) ? (s2IdLimit - realS2Idx) :
                                                                                       kvVecConstInfo.sparseBlockSize);
    DataCopyExtParams intriParams;

    intriParams.blockCount = validS2Count;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    if (kvVecConstInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
        uint32_t combineBytes =
            (kvVecConstInfo.keyQuantMode == QUANT_MODE::TQ4) ?
                (kvVecConstInfo.headDim / 2 + kvVecConstInfo.headDimRope * sizeof(K_ROPE_T) + sizeof(half)) :
            (kvVecConstInfo.keyQuantMode == QUANT_MODE::SFA_SCHEMA_A) ?
                (kvVecConstInfo.headDim / 2 + kvVecConstInfo.headDimRope * sizeof(K_ROPE_T) + 2 * sizeof(half)) :
                (kvVecConstInfo.headDim * sizeof(KV_T) + kvVecConstInfo.headDimRope * sizeof(K_ROPE_T) +
                 kvVecConstInfo.headDim / kvVecConstInfo.tileSize * sizeof(T));
        intriParams.blockLen = combineBytes;
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(
            kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) * combineDimAlign],
            keyGm_[keyBNBOffset * combineDim], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                          int64_t realS2Idx1, int64_t realS2Idx2,
                                                          const RunInfo &kvVecRunInfo)
{
    int64_t s2IdLimit = kvVecRunInfo.curActualSeqLenOri;
    if (kvVecConstInfo.sparseMode == 3) {
        s2IdLimit =
            kvVecRunInfo.curActualSeqLenOri - kvVecRunInfo.actS1Size + kvVecRunInfo.gS1Idx / kvVecConstInfo.gSize + 1;
    }

    int64_t keyBNBOffset1 = GetKeyBNBOffset(realS2Idx1, kvVecRunInfo, s2IdLimit);
    int64_t keyBNBOffset2 = GetKeyBNBOffset(realS2Idx2, kvVecRunInfo, s2IdLimit);
    if (unlikely(keyBNBOffset1 < 0 && keyBNBOffset2 < 0)) {
        return;
    }

    int64_t sparseBlockSrcStride =
        ((keyBNBOffset1 > keyBNBOffset2 ? (keyBNBOffset1 - keyBNBOffset2) : (keyBNBOffset2 - keyBNBOffset1)) -
         kvVecConstInfo.sparseBlockSize);
    uint32_t combineBytes =
        (kvVecConstInfo.keyQuantMode == QUANT_MODE::TQ4) ?
            (kvVecConstInfo.headDim / 2 + kvVecConstInfo.headDimRope * sizeof(K_ROPE_T) + sizeof(half)) :
        (kvVecConstInfo.keyQuantMode == QUANT_MODE::SFA_SCHEMA_A) ?
            (kvVecConstInfo.headDim / 2 + kvVecConstInfo.headDimRope * sizeof(K_ROPE_T) + 2 * sizeof(half)) :
            (kvVecConstInfo.headDim * sizeof(KV_T) + kvVecConstInfo.headDimRope * sizeof(K_ROPE_T) +
             kvVecConstInfo.headDim / kvVecConstInfo.tileSize * sizeof(T));
    int64_t keySrcStride = sparseBlockSrcStride * combineBytes;
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
                 realS2Idx1 + kvVecConstInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + kvVecConstInfo.sparseBlockSize >= s2IdLimit) ||
        kvVecConstInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyBNBOffset1, s2IdLimit, kvVecRunInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyBNBOffset2, s2IdLimit, kvVecRunInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyBNBOffset1 >= 0) + (keyBNBOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyBNBOffset1 > -1 ? keyBNBOffset1 : keyBNBOffset2;
        if (keyBNBOffset2 > -1 && keyBNBOffset2 < keyBNBOffset1) {
            startGmOffset = keyBNBOffset2;
        }

        // 当前仅支持COMBINE模式
        if (kvVecConstInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
            intriParams.blockLen = kvVecConstInfo.sparseBlockSize * combineBytes;
            uint32_t combineDim = combineBytes / sizeof(KV_T);
            uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign - combineDim;
            padParams.paddingValue = 0;
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) +
                                  (mte2Size - mte3Size) * combineDimAlign],
                        keyGm_[startGmOffset * combineDim], intriParams, padParams);
        }
        mte2Size += ((keyBNBOffset1 > -1) + (keyBNBOffset2 > -1)) * kvVecConstInfo.sparseBlockSize;
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                   int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                   const RunInfo &kvVecRunInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    int32_t dealRow = mte2Size - mte3Size;
    SetFlag<AscendC::HardEvent::MTE2_V>(0);
    WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    LocalTensor<half> kvTensorAsFp16 = tmpBuff1.Get<half>();
    uint64_t mask = ConstInfo::BUFFER_SIZE_BYTE_256B / sizeof(half);
    LocalTensor<KV_T> srcTensor = kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)];
    LocalTensor<K_ROPE_T> antiKvTensorAsB16 = tmpBuff1.Get<K_ROPE_T>();
    if (kvVecConstInfo.keyQuantMode == QUANT_MODE::SFA_SCHEMA_A) {
        if constexpr (QSFAT::dequantV2) {
            CopyOutSfaV2(srcTensor, dealRow, s2GmStartOffset, mte3Size, kvVecRunInfo);
            return;
        }
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        DequantSfaRows(srcTensor, antiKvTensorAsB16, dealRow);

        uint32_t rowBytes =
            QSFAAlign(static_cast<uint32_t>(tilingData->baseParams.dSizeVInput), static_cast<uint32_t>(BYTE_BLOCK));
        uint32_t mergeGmStride = 512U * kvVecConstInfo.combineHeadDim;
        uint32_t latentBytes = kvVecConstInfo.headDim * sizeof(K_ROPE_T);
        uint32_t ropeBytes = kvVecConstInfo.headDimRope * sizeof(K_ROPE_T);
        uint32_t ropeByteOff = sfa_kv::ROPE_OFF;

        DataCopyExtParams sfaDataCopyParams;
        sfaDataCopyParams.blockCount = static_cast<uint16_t>(dealRow);
        sfaDataCopyParams.blockLen = latentBytes;
        sfaDataCopyParams.srcStride = 0;
        sfaDataCopyParams.dstStride = (kvVecConstInfo.combineHeadDim - kvVecConstInfo.headDim) * sizeof(K_ROPE_T);
        uint64_t sfaGmBase = kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * mergeGmStride +
                             (s2GmStartOffset + mte3Size) * kvVecConstInfo.combineHeadDim;
        DataCopyPad(kvMergeGm_[sfaGmBase], antiKvTensorAsB16, sfaDataCopyParams);

        LocalTensor<K_ROPE_T> sfaKRopeUb = srcTensor[ropeByteOff].template ReinterpretCast<K_ROPE_T>();
        sfaDataCopyParams.blockLen = ropeBytes;
        sfaDataCopyParams.srcStride = static_cast<uint32_t>(rowBytes / BYTE_BLOCK) - ropeBytes / BYTE_BLOCK;
        sfaDataCopyParams.dstStride = (kvVecConstInfo.combineHeadDim - kvVecConstInfo.headDimRope) * sizeof(K_ROPE_T);
        DataCopyPad(kvMergeGm_[sfaGmBase + kvVecConstInfo.headDim], sfaKRopeUb, sfaDataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        return;
    } else if (kvVecConstInfo.keyQuantMode == QUANT_MODE::TQ4) {
        // MTE3 and V share tmpBuff1/tmpBuff2 across merge iterations. The
        // explicit wait is required even when PipeBarrier is present because
        // this is a cross-pipeline dependency.
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        Tq4DequantRows(srcTensor, antiKvTensorAsB16, dealRow);

        uint32_t rowBytes =
            QSFAAlign(static_cast<uint32_t>(tilingData->baseParams.dSizeVInput), static_cast<uint32_t>(BYTE_BLOCK));
        uint32_t mergeGmStride = 512U * kvVecConstInfo.combineHeadDim;
        uint32_t latentBytes = kvVecConstInfo.headDim * sizeof(K_ROPE_T);
        uint32_t ropeBytes = kvVecConstInfo.headDimRope * sizeof(K_ROPE_T);
        uint32_t ropeByteOff = kvVecConstInfo.headDim / 2U;

        // Export one FP16 scale per row to the upper half of the valid-size
        // workspace. A 32B gather block avoids scalar loads and works for any
        // dealRow <= 32.
        {
            uint32_t scaleByteOff = kvVecConstInfo.headDim / 2U + ropeBytes;
            uint16_t rowStrideBlk = static_cast<uint16_t>(rowBytes / BYTE_BLOCK);
            LocalTensor<half> scaleUb = tmpBuff2.Get<half>();
            LocalTensor<half> scaleUb32 = tmpBuff2.Get<half>()[512];
            LocalTensor<half> scaleSrc = srcTensor.template ReinterpretCast<half>()[scaleByteOff / 2U];
            DataCopyParams scaleParams;
            scaleParams.blockCount = static_cast<uint16_t>(dealRow);
            scaleParams.blockLen = 1;
            scaleParams.srcStride = static_cast<uint16_t>(rowStrideBlk - 1U);
            scaleParams.dstStride = 0;
            DataCopy(scaleUb32, scaleSrc, scaleParams);
            PipeBarrier<PIPE_V>();
            Gather(scaleUb, scaleUb32, tq4STIdxBuf_.Get<uint32_t>(), 0, static_cast<uint32_t>(dealRow));
            SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
            WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
            DataCopyExtParams scaleOutParams;
            scaleOutParams.blockCount = 1;
            scaleOutParams.blockLen = static_cast<uint32_t>(dealRow) * sizeof(uint16_t);
            scaleOutParams.srcStride = 0;
            scaleOutParams.dstStride = 0;
            DataCopyPad(
                tq4ScaleGm_[TQ4_SCALE_HALF_BASE + kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * TQ4_SCALE_SLOT_STRIDE +
                            (s2GmStartOffset + mte3Size)],
                scaleUb.template ReinterpretCast<uint16_t>(), scaleOutParams);
        }

        DataCopyExtParams tq4DataCopyParams;
        tq4DataCopyParams.blockCount = static_cast<uint16_t>(dealRow);
        tq4DataCopyParams.blockLen = latentBytes;
        tq4DataCopyParams.srcStride = 0;
        tq4DataCopyParams.dstStride = (kvVecConstInfo.combineHeadDim - kvVecConstInfo.headDim) * sizeof(K_ROPE_T);
        uint64_t tq4GmBase = kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * mergeGmStride +
                             (s2GmStartOffset + mte3Size) * kvVecConstInfo.combineHeadDim;
        DataCopyPad(kvMergeGm_[tq4GmBase], antiKvTensorAsB16, tq4DataCopyParams);

        LocalTensor<K_ROPE_T> tq4KRopeUb = srcTensor[ropeByteOff].template ReinterpretCast<K_ROPE_T>();
        tq4DataCopyParams.blockLen = ropeBytes;
        tq4DataCopyParams.srcStride = static_cast<uint32_t>(rowBytes / BYTE_BLOCK) - ropeBytes / BYTE_BLOCK;
        tq4DataCopyParams.dstStride = (kvVecConstInfo.combineHeadDim - kvVecConstInfo.headDimRope) * sizeof(K_ROPE_T);
        DataCopyPad(kvMergeGm_[tq4GmBase + kvVecConstInfo.headDim], tq4KRopeUb, tq4DataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        return;
    } else {
        if (dealRow == 1) {
            Cast(kvTensorAsFp16, srcTensor, RoundMode::CAST_NONE, mask, 4, {1, 1, 8, 4});
        } else {
            uint8_t repeatTimes = static_cast<uint8_t>(dealRow);
            Cast(kvTensorAsFp16, srcTensor, RoundMode::CAST_NONE, mask, repeatTimes,
                 {1, 1, 32, 21}); // 21=(512+64*2+32)/32
            Cast(kvTensorAsFp16[128], srcTensor[128], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21});
            Cast(kvTensorAsFp16[256], srcTensor[256], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21});
            Cast(kvTensorAsFp16[384], srcTensor[384], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21});
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<T> antiQuantScale = tmpBuff2.Get<T>();
        LocalTensor<T> oriQuantScaleTensor = srcTensor[640].template ReinterpretCast<T>();
        if (dealRow == 1) {
            Brcb(antiQuantScale, oriQuantScaleTensor, 1, {1, 4});
        } else {
            DataCopyParams params;
            params.blockCount = dealRow;
            params.blockLen = 1;
            params.dstStride = 0;
            params.srcStride = (kvVecConstInfo.headDim * sizeof(KV_T) + kvVecConstInfo.headDimRope * sizeof(K_ROPE_T)) /
                               ConstInfo::BUFFER_SIZE_BYTE_32B;
            LocalTensor<T> tmpAntiQuantScale = antiQuantScale[ConstInfo::BUFFER_SIZE_BYTE_1K];
            DataCopy(tmpAntiQuantScale, oriQuantScaleTensor, params);
            PipeBarrier<PIPE_V>();
            Brcb(antiQuantScale, tmpAntiQuantScale, dealRow, {1, 4});
        }
        PipeBarrier<PIPE_V>();
        uint32_t dealLoop = CeilDiv(dealRow, LIMIT_DEAL_ROW);
        uint32_t dealRowFp32 = LIMIT_DEAL_ROW;
        uint32_t element = LIMIT_DEAL_ROW * kvVecConstInfo.headDim;
        LocalTensor<T> kvTensorAsFp32 = inputBuff2.Get<T>();
        for (uint32_t i = 0; i < dealLoop; i++) {
            if (i == dealLoop - 1) {
                dealRowFp32 = dealRow - i * LIMIT_DEAL_ROW;
            }
            Cast(kvTensorAsFp32, kvTensorAsFp16[i * element], RoundMode::CAST_NONE,
                 static_cast<uint32_t>(dealRowFp32 * kvVecConstInfo.headDim));
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < kvVecConstInfo.tileSize / FP32_REPEAT_ELEMENT_NUM; j++) {
                Mul(kvTensorAsFp32[j * FP32_REPEAT_ELEMENT_NUM], kvTensorAsFp32[j * FP32_REPEAT_ELEMENT_NUM],
                    antiQuantScale[i * LIMIT_DEAL_ROW * 32], FP32_REPEAT_ELEMENT_NUM, 4 * dealRowFp32,
                    {1, 1, 0, 16, 16, 1});
            }
            PipeBarrier<PIPE_V>();
            if constexpr (IsSameType<K_ROPE_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
                Cast(antiKvTensorAsB16[i * element], kvTensorAsFp32, RoundMode::CAST_RINT,
                     static_cast<uint32_t>(dealRowFp32 * kvVecConstInfo.headDim));
            } else {
                Cast(antiKvTensorAsB16[i * element], kvTensorAsFp32, RoundMode::CAST_ROUND,
                     static_cast<uint32_t>(dealRowFp32 * kvVecConstInfo.headDim));
            }
            PipeBarrier<PIPE_V>();
        }
    }

    LocalTensor<K_ROPE_T> antiKvTensorAsB16Nz = outputBuff1.Get<K_ROPE_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    int dataBlocks = REPEAT_BLOCK_BYTE / BYTE_BLOCK;
    int loops = CeilDiv(dealRow, dataBlocks);
    uint64_t tail = dealRow - (loops - 1) * dataBlocks;
    uint64_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2;
    uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    uint8_t repeatTimes = static_cast<uint8_t>(kvVecConstInfo.headDim / blockElementNum);
    for (int i = 0; i < loops; i++) {
        mask = (i == loops - 1) ? tail * blockElementNum : repeatElementNum;
        Copy(antiKvTensorAsB16Nz[i * repeatElementNum], antiKvTensorAsB16[i * dataBlocks * kvVecConstInfo.headDim],
             mask, repeatTimes, {1, 32, static_cast<uint16_t>(dealRow), 1});
    }
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = kvVecConstInfo.headDim / blockElementNum;
    dataCopyParams.blockLen = dealRow * blockElementNum * sizeof(K_ROPE_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (kvVecConstInfo.s2BaseSize - dealRow) * blockElementNum * sizeof(K_ROPE_T);
    DataCopyPad(kvMergeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                           (s2GmStartOffset + mte3Size) * blockElementNum],
                antiKvTensorAsB16Nz, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    uint32_t qsfaRopeByteOff = 512;
    uint16_t qsfaRopeRowStrideBlk = 21;
    if (kvVecConstInfo.keyQuantMode == QUANT_MODE::TQ4 ||
        kvVecConstInfo.keyQuantMode == QUANT_MODE::SFA_SCHEMA_A) {
        qsfaRopeByteOff = TQ4_NOPE_BYTES; // 256; Schema A same
        qsfaRopeRowStrideBlk = 13; // ceil(388B / 32B) == ceil(386B / 32B) == 13
    }
    LocalTensor<K_ROPE_T> kRopeUb = srcTensor[qsfaRopeByteOff].template ReinterpretCast<K_ROPE_T>();
    LocalTensor<K_ROPE_T> kRopeUbNz = outputBuff2.Get<K_ROPE_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Copy(kRopeUbNz, kRopeUb, kvVecConstInfo.headDimRope, static_cast<uint8_t>(dealRow),
         {static_cast<uint16_t>(dealRow), 1, 1, qsfaRopeRowStrideBlk});
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    dataCopyParams.blockCount = kvVecConstInfo.headDimRope / blockElementNum;
    DataCopyPad(kvMergeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * 512 +
                           (s2GmStartOffset + mte3Size) * blockElementNum],
                kRopeUbNz, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

// b s1 k
template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::MergeKv(const RunInfo &kvVecRunInfo)
{
    int64_t s2ProcessSize = kvVecRunInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * kvVecConstInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t qsfaActualSeqQPrefixSum =
            (kvVecRunInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(kvVecRunInfo.bIdx - 1);
        topkGmBaseOffset += (qsfaActualSeqQPrefixSum + kvVecRunInfo.gS1Idx / kvVecConstInfo.gSize) *
                                kvVecConstInfo.kvHeadNum * kvVecConstInfo.sparseBlockCount +
                            kvVecRunInfo.n2Idx * kvVecConstInfo.sparseBlockCount;
    } else {
        topkGmBaseOffset += kvVecRunInfo.bIdx * kvVecConstInfo.qSeqSize * kvVecConstInfo.sparseBlockCount +
                            kvVecRunInfo.gS1Idx / kvVecConstInfo.gSize * kvVecConstInfo.sparseBlockCount;
    }
    int64_t qsfaMergeMte3Idx = 0;
    int64_t qsfaMte2Size = 0;
    int64_t qsfaMte3Size = 0;
    int64_t qsfaS2IdxArray0 = -1;
    int64_t qsfaS2IdxArray1 = -1;
    bool qsfaNeedWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    int64_t qsfaS2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * kvVecConstInfo.sparseBlockSize;
    int64_t qsfaS2GmLimit =
        GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * kvVecConstInfo.sparseBlockSize : s2ProcessSize;
    if (qsfaS2GmLimit > s2ProcessSize) {
        qsfaS2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = qsfaS2GmStartOffset; s2GmOffsetArray < qsfaS2GmLimit;
         s2GmOffsetArray += 2 * kvVecConstInfo.sparseBlockSize) {
        if (qsfaNeedWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % 2);
            qsfaNeedWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, qsfaS2IdxArray0, topkGmBaseOffset, kvVecRunInfo);
        if (unlikely(qsfaS2IdxArray0 < 0)) {
            CopyOutMrgeResult(qsfaMte2Size, qsfaMte3Size, qsfaS2GmStartOffset, qsfaMergeMte3Idx, kvVecRunInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % 2);
            qsfaMergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + kvVecConstInfo.sparseBlockSize, qsfaS2IdxArray1, topkGmBaseOffset, kvVecRunInfo);
        CopyInKv(qsfaMte2Size, qsfaMte3Size, qsfaMergeMte3Idx, qsfaS2IdxArray0, qsfaS2IdxArray1, kvVecRunInfo);
        if ((qsfaMte2Size - qsfaMte3Size + 2 * kvVecConstInfo.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * kvVecConstInfo.sparseBlockSize >= qsfaS2GmLimit) {
            CopyOutMrgeResult(qsfaMte2Size, qsfaMte3Size, qsfaS2GmStartOffset, qsfaMergeMte3Idx, kvVecRunInfo);
            qsfaMte3Size = qsfaMte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % 2);
            qsfaMergeMte3Idx++;
            qsfaNeedWaitMte3ToMte2 = true;
        }
    }

    if (unlikely(qsfaS2GmStartOffset + qsfaMte2Size < qsfaS2GmLimit)) {
        uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx & 1);
        LocalTensor<K_ROPE_T> mergeUb = kvMergUb_.template ReinterpretCast<K_ROPE_T>();
        Duplicate(mergeUb, static_cast<K_ROPE_T>(0.0), kvVecConstInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = kvVecConstInfo.headDim / blockElementNum;
        dataCopyParams.blockLen = blockElementNum * sizeof(K_ROPE_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (kvVecConstInfo.s2BaseSize - 1) * blockElementNum * sizeof(K_ROPE_T);
        for (int64_t s2GmOffset = qsfaS2GmStartOffset + qsfaMte2Size; s2GmOffset < qsfaS2GmLimit; s2GmOffset++) {
            DataCopyPad(
                kvMergeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + s2GmOffset * blockElementNum],
                mergeUb, dataCopyParams);
        }
        dataCopyParams.blockCount = kvVecConstInfo.headDimRope / blockElementNum;
        for (int64_t s2GmOffset = qsfaS2GmStartOffset + qsfaMte2Size; s2GmOffset < qsfaS2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                   512 * kvVecConstInfo.headDim + s2GmOffset * blockElementNum],
                        mergeUb, dataCopyParams);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx & 1);
        qsfaMergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM, qsfaMte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 128 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(kvValidSizeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + GetSubBlockIdx() * 128],
                v0ValidSizeUb_, dataCopyParams);
    return;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec1L(const RunInfo &kvVecInfo)
{
    uint32_t qsfaNBufferLoopTimes =
        (kvVecInfo.actMBaseSize + kvVecConstInfo.nBufferMBaseSize - 1) / kvVecConstInfo.nBufferMBaseSize;
    uint32_t qsfaNBufferTail = kvVecInfo.actMBaseSize - (qsfaNBufferLoopTimes - 1) * kvVecConstInfo.nBufferMBaseSize;
    for (uint32_t qsfaI = 0; qsfaI < qsfaNBufferLoopTimes; qsfaI++) {
        MSplitInfo kvVecSplitInfo;
        kvVecSplitInfo.nBufferIdx = qsfaI;
        kvVecSplitInfo.nBufferStartM = qsfaI * kvVecConstInfo.nBufferMBaseSize;
        kvVecSplitInfo.nBufferDealM =
            (qsfaI + 1 != qsfaNBufferLoopTimes) ? kvVecConstInfo.nBufferMBaseSize : qsfaNBufferTail;

        kvVecSplitInfo.vecDealM = (kvVecSplitInfo.nBufferDealM <= 16) ?
                                      kvVecSplitInfo.nBufferDealM :
                                      (((kvVecSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        kvVecSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            kvVecSplitInfo.vecStartM = kvVecSplitInfo.vecDealM;
            kvVecSplitInfo.vecDealM = kvVecSplitInfo.nBufferDealM - kvVecSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(kvVecConstInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(kvVecInfo, kvVecSplitInfo);
        CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE3>(kvVecConstInfo.syncV1C2);
        // move lse for flash decode
        if (kvVecInfo.s2Idx == kvVecInfo.curSInnerLoopTimes - 1) {
            if (kvVecInfo.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    uint32_t outIdx = kvVecInfo.loop % (kvVecConstInfo.preLoadNum);
                    auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
                    auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
                    ComputeLogSumExpAndCopyToGm(kvVecInfo, kvVecSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename QSFAT>
__aicore__ inline uint64_t QSFAVectorService<QSFAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2SingleBuf(const RunInfo &kvVecInfo,
                                                                      const MSplitInfo &kvVecSplitInfo)
{
    if (kvVecSplitInfo.vecDealM == 0) {
        return;
    }

    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / kvVecConstInfo.headDim;
    if (gPreSplitSize > kvVecSplitInfo.vecDealM) {
        gPreSplitSize = kvVecSplitInfo.vecDealM;
    }
    uint32_t loopCount = (kvVecSplitInfo.vecDealM + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t tailSplitSize = kvVecSplitInfo.vecDealM - (loopCount - 1) * gPreSplitSize;

    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(kvVecInfo, kvVecSplitInfo, i * gPreSplitSize, dealSize, kvVecConstInfo.headDim,
                             kvVecConstInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DealBmm2ResBaseBlock(const RunInfo &kvVecInfo,
                                                                      const MSplitInfo &kvVecSplitInfo,
                                                                      uint32_t startRow, uint32_t kvDealRows,
                                                                      uint32_t kvColumns, uint32_t kvActualColumns)
{
    uint32_t vec2ComputeSize = kvDealRows * kvColumns;
    uint32_t baseOffset = startRow;
    LocalTensor<T> kvBmm2ResultUb = tmpBuff1.Get<T>();
    kvBmm2ResultUb.SetSize(vec2ComputeSize);

    size_t batchBase = 0;
    uint64_t inOutBaseOffset = (kvVecSplitInfo.vecStartM + startRow) * kvColumns;
    uint64_t srcGmOffset =
        (kvVecInfo.loop % kvVecConstInfo.preLoadNum) * kvVecConstInfo.bmm2ResUbSize + inOutBaseOffset;

    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(kvBmm2ResultUb, tmpBmm2ResUb, vec2ComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    // 除第一个循环外，均需要更新中间计算结果
    if (kvVecInfo.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        uint64_t vecPre2ResGmOffset =
            ((kvVecInfo.loop - 1) % kvVecConstInfo.preLoadNum) * kvVecConstInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vecPre2ResGmOffset + batchBase], vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        LocalTensor<T> softmaxExpBrcb = tmpBuff2.Get<T>();
        Brcb(softmaxExpBrcb,
             softmaxExpUb[(kvVecInfo.loop % kvVecConstInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
             (kvVecSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpBrcb, kvDealRows, kvColumns, kvActualColumns);
        PipeBarrier<PIPE_V>();
        Add(kvBmm2ResultUb, kvBmm2ResultUb, bmm2ResPreUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    }
    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (kvVecInfo.s2Idx + 1 == kvVecInfo.curSInnerLoopTimes) {
        LocalTensor<T> softmaxSumBrcb = tmpBuff2.Get<T>();
        Brcb(softmaxSumBrcb,
             softmaxSumUb[(kvVecInfo.loop % kvVecConstInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
             (kvVecSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(kvBmm2ResultUb, kvBmm2ResultUb, softmaxSumBrcb, kvDealRows, kvColumns, kvActualColumns);

        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(kvVecInfo, kvBmm2ResultUb, kvVecSplitInfo.vecStartM + startRow, kvDealRows, kvColumns,
                       kvActualColumns);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> tmpBmm2Res = outputBuff1.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopy(tmpBmm2Res, kvBmm2ResultUb, kvDealRows * kvColumns);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t vecPre2ResGmOffset =
            (kvVecInfo.loop % kvVecConstInfo.preLoadNum) * kvVecConstInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vecPre2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2L(const RunInfo &kvVecInfo)
{
    uint32_t qsfaNBufferLoopTimes =
        (kvVecInfo.actMBaseSize + kvVecConstInfo.nBufferMBaseSize - 1) / kvVecConstInfo.nBufferMBaseSize;
    uint32_t qsfaNBufferTail = kvVecInfo.actMBaseSize - (qsfaNBufferLoopTimes - 1) * kvVecConstInfo.nBufferMBaseSize;
    for (uint32_t qsfaI = 0; qsfaI < qsfaNBufferLoopTimes; qsfaI++) {
        MSplitInfo kvVecSplitInfo;
        kvVecSplitInfo.nBufferIdx = qsfaI;
        kvVecSplitInfo.nBufferDealM =
            (qsfaI + 1 != qsfaNBufferLoopTimes) ? kvVecConstInfo.nBufferMBaseSize : qsfaNBufferTail;
        kvVecSplitInfo.nBufferStartM = qsfaI * kvVecConstInfo.nBufferMBaseSize;

        kvVecSplitInfo.vecDealM = (kvVecSplitInfo.nBufferDealM <= 16) ?
                                      kvVecSplitInfo.nBufferDealM :
                                      (((kvVecSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        kvVecSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            kvVecSplitInfo.vecStartM = kvVecSplitInfo.vecDealM;
            kvVecSplitInfo.vecDealM = kvVecSplitInfo.nBufferDealM - kvVecSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(kvVecConstInfo.syncC2V2);
        ProcessVec2SingleBuf(kvVecInfo, kvVecSplitInfo);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2Inner(const RunInfo &kvVecInfo,
                                                                  const MSplitInfo &kvVecSplitInfo, uint32_t mStartRow,
                                                                  uint32_t mDealSize)
{
    uint32_t qsfaMSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / kvVecConstInfo.headDim;
    if (qsfaMSplitSize > mDealSize) {
        qsfaMSplitSize = mDealSize;
    }

    uint32_t qsfaLoopCount = (mDealSize + qsfaMSplitSize - 1) / qsfaMSplitSize;
    uint32_t qsfaTailSplitSize = mDealSize - (qsfaLoopCount - 1) * qsfaMSplitSize;
    for (uint32_t qsfaI = 0, dealSize = qsfaMSplitSize; qsfaI < qsfaLoopCount; qsfaI++) {
        if (qsfaI == (qsfaLoopCount - 1)) {
            dealSize = qsfaTailSplitSize;
        }
        DealBmm2ResBaseBlock(kvVecInfo, kvVecSplitInfo, qsfaI * qsfaMSplitSize + mStartRow, dealSize,
                             kvVecConstInfo.headDim, kvVecConstInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::GetConfusionTransposeTiling(int64_t numR, int64_t numC,
                                                                             const uint32_t stackBufferSize,
                                                                             const uint32_t typeSize,
                                                                             ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t qsfaBlockSize = ONE_BLK_SIZE / typeSize;
    uint32_t qsfaHeight = numC;
    uint32_t qsfaWidth = numR;
    uint32_t qsfaHighBlock = qsfaHeight / BLOCK_CUBE;
    uint32_t qsfaStride = qsfaHeight * qsfaBlockSize * typeSize / ONE_BLK_SIZE;
    uint32_t qsfaRepeat = qsfaWidth / qsfaBlockSize;

    tiling.param0 = qsfaBlockSize;
    tiling.param1 = qsfaHeight;
    tiling.param2 = qsfaWidth;
    tiling.param3 = qsfaHighBlock;
    tiling.param4 = qsfaStride;
    tiling.param5 = qsfaRepeat;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::Bmm2FDDataCopyOut(const RunInfo &kvVecInfo,
                                                                   LocalTensor<T> &kvBmm2ResultUb,
                                                                   uint32_t kvWorkspaceRow, uint32_t kvDealRows,
                                                                   uint32_t kvColumns, uint32_t kvActualColumns)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, kvBmm2ResultUb, kvColumns * kvDealRows);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t kvAccumOutputCount = CalcAccumOffset(kvVecInfo.bIdx, kvVecInfo.gS1Idx);
    uint64_t offset = kvAccumOutputCount * kvVecConstInfo.kvHeadNum * kvVecConstInfo.mBaseSize *
                          kvVecConstInfo.headDim + // taskoffset
                      kvVecInfo.tndCoreStartKVSplitPos * kvVecConstInfo.kvHeadNum * kvVecConstInfo.mBaseSize *
                          kvVecConstInfo.headDim +      // 份数offset
                      kvWorkspaceRow * kvActualColumns; // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (kvVecInfo.actualSingleProcessSInnerSize == 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = kvDealRows;
        dataCopyParams.blockLen = kvActualColumns * sizeof(T);
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = (kvColumns - kvActualColumns) / (BYTE_BLOCK / sizeof(T));
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, kvDealRows * kvActualColumns, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::Bmm2DataCopyOutTrans(const RunInfo &kvVecInfo,
                                                                      LocalTensor<OUT_T> &attenOutUb,
                                                                      uint32_t kvWorkspaceRow, uint32_t kvDealRows,
                                                                      uint32_t kvColumns, uint32_t kvActualColumns)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = kvDealRows;
    dataCopyParams.blockLen = kvActualColumns * sizeof(OUT_T);
    dataCopyParams.srcStride = (kvColumns - kvActualColumns) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[kvVecInfo.attenOutOffset + kvWorkspaceRow * kvActualColumns], attenOutUb,
                dataCopyParams);
    return;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::Bmm2CastAndCopyOut(const RunInfo &kvVecInfo,
                                                                    LocalTensor<T> &kvBmm2ResultUb,
                                                                    uint32_t kvWorkspaceRow, uint32_t kvDealRows,
                                                                    uint32_t kvColumns, uint32_t kvActualColumns)
{
    LocalTensor<OUT_T> qsfaTmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(qsfaTmpBmm2ResCastTensor, kvBmm2ResultUb, AscendC::RoundMode::CAST_RINT, kvDealRows * kvColumns);
    } else {
        Cast(qsfaTmpBmm2ResCastTensor, kvBmm2ResultUb, AscendC::RoundMode::CAST_ROUND, kvDealRows * kvColumns);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(kvVecInfo, qsfaTmpBmm2ResCastTensor, kvWorkspaceRow, kvDealRows, kvColumns, kvActualColumns);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::Bmm2ResCopyOut(const RunInfo &kvVecInfo,
                                                                LocalTensor<T> &kvBmm2ResultUb, uint32_t kvWorkspaceRow,
                                                                uint32_t kvDealRows, uint32_t kvColumns,
                                                                uint32_t kvActualColumns)
{
    if constexpr (!FLASH_DECODE) {
        Bmm2CastAndCopyOut(kvVecInfo, kvBmm2ResultUb, kvWorkspaceRow, kvDealRows, kvColumns, kvActualColumns);
    } else {
        if (kvVecInfo.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(kvVecInfo, kvBmm2ResultUb, kvWorkspaceRow, kvDealRows, kvColumns, kvActualColumns);
        } else {
            Bmm2CastAndCopyOut(kvVecInfo, kvBmm2ResultUb, kvWorkspaceRow, kvDealRows, kvColumns, kvActualColumns);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub,
                                                         LocalTensor<float> src1Ub, uint32_t kvDealRows,
                                                         uint32_t kvColumns, uint32_t kvActualColumns)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t qsfaDtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t qsfaDLoop = kvActualColumns / qsfaDtypeMask;
    uint32_t qsfaDRemain = kvActualColumns % qsfaDtypeMask;

    BinaryRepeatParams qsfaRepeatParamsDiv;
    qsfaRepeatParamsDiv.src0BlkStride = 1;
    qsfaRepeatParamsDiv.src1BlkStride = 0;
    qsfaRepeatParamsDiv.dstBlkStride = 1;
    qsfaRepeatParamsDiv.src0RepStride = kvColumns / FP32_BLOCK_ELEMENT_NUM;
    qsfaRepeatParamsDiv.src1RepStride = 1;
    qsfaRepeatParamsDiv.dstRepStride = kvColumns / FP32_BLOCK_ELEMENT_NUM;
    uint32_t qsfaColumnRepeatCount = qsfaDLoop;
    if (qsfaColumnRepeatCount <= kvDealRows) {
        uint32_t qsfaOffset = 0;
        for (uint32_t qsfaI = 0; qsfaI < qsfaDLoop; qsfaI++) {
            Div(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub, qsfaDtypeMask, kvDealRows, qsfaRepeatParamsDiv);
            qsfaOffset += qsfaDtypeMask;
        }
    } else {
        BinaryRepeatParams qsfaColumnRepeatParams;
        qsfaColumnRepeatParams.src0BlkStride = 1;
        qsfaColumnRepeatParams.src1BlkStride = 0;
        qsfaColumnRepeatParams.dstBlkStride = 1;
        qsfaColumnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        qsfaColumnRepeatParams.src1RepStride = 0;
        qsfaColumnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t qsfaOffset = 0;
        for (uint32_t qsfaI = 0; qsfaI < kvDealRows; qsfaI++) {
            Div(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub[qsfaI * FP32_BLOCK_ELEMENT_NUM], qsfaDtypeMask,
                qsfaColumnRepeatCount, qsfaColumnRepeatParams);
            qsfaOffset += kvColumns;
        }
    }
    if (qsfaDRemain > 0) {
        Div(dstUb[qsfaDLoop * qsfaDtypeMask], src0Ub[qsfaDLoop * qsfaDtypeMask], src1Ub, qsfaDRemain, kvDealRows,
            qsfaRepeatParamsDiv);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub,
                                                         LocalTensor<T> src1Ub, uint32_t kvDealRows, uint32_t kvColumns,
                                                         uint32_t kvActualColumns)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t qsfaRepeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t qsfaBlockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        qsfaRepeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        qsfaBlockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t qsfaDLoop = kvActualColumns / qsfaRepeatElementNum;
    uint32_t qsfaDRemain = kvActualColumns % qsfaRepeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (kvColumns < REPEATE_STRIDE_UP_BOUND * qsfaBlockElementNum) {
        BinaryRepeatParams qsfaRepeatParams;
        qsfaRepeatParams.src0BlkStride = 1;
        qsfaRepeatParams.src1BlkStride = 0;
        qsfaRepeatParams.dstBlkStride = 1;
        qsfaRepeatParams.src0RepStride = kvColumns / qsfaBlockElementNum;
        qsfaRepeatParams.src1RepStride = 1;
        qsfaRepeatParams.dstRepStride = kvColumns / qsfaBlockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (qsfaDLoop <= kvDealRows) {
            uint32_t qsfaOffset = 0;
            for (uint32_t qsfaI = 0; qsfaI < qsfaDLoop; qsfaI++) {
                Mul(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub, qsfaRepeatElementNum, kvDealRows, qsfaRepeatParams);
                qsfaOffset += qsfaRepeatElementNum;
            }
        } else {
            BinaryRepeatParams qsfaColumnRepeatParams;
            qsfaColumnRepeatParams.src0BlkStride = 1;
            qsfaColumnRepeatParams.src1BlkStride = 0;
            qsfaColumnRepeatParams.dstBlkStride = 1;
            qsfaColumnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            qsfaColumnRepeatParams.src1RepStride = 0;
            qsfaColumnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t qsfaI = 0; qsfaI < kvDealRows; qsfaI++) {
                Mul(dstUb[qsfaI * kvColumns], src0Ub[qsfaI * kvColumns], src1Ub[qsfaI * qsfaBlockElementNum],
                    qsfaRepeatElementNum, qsfaDLoop, qsfaColumnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (qsfaDRemain > 0) {
            Mul(dstUb[qsfaDLoop * qsfaRepeatElementNum], src0Ub[qsfaDLoop * qsfaRepeatElementNum], src1Ub, qsfaDRemain,
                kvDealRows, qsfaRepeatParams);
        }
    } else {
        BinaryRepeatParams qsfaRepeatParams;
        qsfaRepeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        qsfaRepeatParams.src0BlkStride = 1;
        qsfaRepeatParams.src1RepStride = 0;
        qsfaRepeatParams.src1BlkStride = 0;
        qsfaRepeatParams.dstRepStride = 8;
        qsfaRepeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t qsfaI = 0; qsfaI < kvDealRows; qsfaI++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[qsfaI * kvColumns], src0Ub[qsfaI * kvColumns], src1Ub[qsfaI * qsfaBlockElementNum],
                qsfaRepeatElementNum, qsfaDLoop, qsfaRepeatParams);
            //  计算一行中的尾块
            if (qsfaDRemain > 0) {
                Mul(dstUb[qsfaI * kvColumns + qsfaDLoop * qsfaRepeatElementNum],
                    src0Ub[qsfaI * kvColumns + qsfaDLoop * qsfaRepeatElementNum], src1Ub[qsfaI * qsfaBlockElementNum],
                    qsfaDRemain, 1, qsfaRepeatParams);
            }
        }
    }
}

#include "rq_uniform_kv_quant_sparse_flash_attention_dequant_v2.h"

#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
