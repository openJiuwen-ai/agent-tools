/**
 * Device-side plain tiling structs for RqUniformFiaPagedK4v4.
 * Must stay binary-compatible with op_host/turboquant_fia_tiling_data.h
 * (REGISTER / BEGIN_TILING_DATA_DEF layout). Do NOT include
 * register/tilingdata_base.h here — it conflicts with AscendC DT_* macros.
 */
#ifndef TQ_FIA_DEVICE_TILING_H_
#define TQ_FIA_DEVICE_TILING_H_

#include <cstdint>

namespace optiling {

constexpr uint32_t TQ_FIA_MAX_AIC_CORE_NUM = 26;

struct TqFiaBaseParams {
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t headDim;
    uint32_t headDimRope;
    uint32_t actualSeqS1Dims;
    uint32_t actualSeqS2Dims;
    uint32_t accumQSeqFlag;
    uint32_t accumKVSeqFlag;
    float scaleValue;
    uint32_t usedCoreNum;
    uint32_t outputLayout;
    uint32_t batchContinuous;
    uint32_t softmaxLseFlag;
    uint32_t needInit;
    uint32_t slidingFlag;
    uint32_t l2CacheOffFlag;
    uint32_t isLegacyIfa;
};

struct TqFiaPageAttentionParams {
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
};

struct TqFiaMaskParams {
    uint32_t attenMaskFlag;
    uint32_t attenMaskBatchStride;
    uint32_t attenMaskStride;
    int32_t preToken;
    int32_t nextToken;
    uint32_t isRowInvalid;
    uint32_t isExistRowInvalid;
    uint32_t sparseMode;
};

struct TqFiaWorkspaceParams {
    uint32_t mm1ResSize;
    uint32_t mm2ResSize;
    uint32_t fdAccumOutSize;
    uint32_t fdLogSumExpSize;
    uint32_t dequantWsSlots;
    uint32_t dequantS2Cache;
};

struct TqFiaInnerSplitParams {
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
};

struct TqFiaOuterSplitParams {
    uint32_t bN2End[TQ_FIA_MAX_AIC_CORE_NUM];
    uint32_t gS1End[TQ_FIA_MAX_AIC_CORE_NUM];
    uint32_t s2End[TQ_FIA_MAX_AIC_CORE_NUM];
};

struct TqFiaFlashDecodeParams {
    uint32_t numOfFdHead;
    uint32_t reserved;
    uint32_t gS1BaseSizeOfFd;
    uint32_t usedVecNumOfFd;
    uint32_t bN2IdxOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM];
    uint32_t gS1IdxOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM];
    uint32_t s2SplitNumOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM];
    uint32_t s2SplitStartIdxOfCore[TQ_FIA_MAX_AIC_CORE_NUM];
    uint32_t gS1SplitNumOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM];
    uint32_t gS1LastPartSizeOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM];
    uint32_t gS1IdxEndOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM * 2];
    uint32_t gS1IdxEndOfFdHeadSplit[TQ_FIA_MAX_AIC_CORE_NUM * 2];
};

struct TqFiaPrefixParams {
    uint64_t prefixMaxLen;
    uint64_t prefixLen;
    bool prefixFlag;
};

struct TqFiaPseParams {
    uint32_t pseShiftFlag;
    uint32_t pseShiftByBatch;
    uint32_t pseShiftS1;
    uint32_t pseShiftS2;
};

struct TqFiaLeftPaddingParams {
    uint32_t qPaddingFlag;
    uint32_t kvPaddingFlag;
};

struct FusedInferAttentionScoreTilingData {
    TqFiaBaseParams baseParams;
    TqFiaPageAttentionParams pageAttenParams;
    TqFiaMaskParams maskParams;
    TqFiaWorkspaceParams workspaceParams;
    TqFiaInnerSplitParams innerSplitParams;
    TqFiaOuterSplitParams outerSplitParams;
    TqFiaFlashDecodeParams fdParams;
    TqFiaPrefixParams prefixParams;
    TqFiaPseParams pseParams;
    TqFiaLeftPaddingParams leftPaddingParams;
};

}  // namespace optiling

using optiling::FusedInferAttentionScoreTilingData;

#endif  // TQ_FIA_DEVICE_TILING_H_
