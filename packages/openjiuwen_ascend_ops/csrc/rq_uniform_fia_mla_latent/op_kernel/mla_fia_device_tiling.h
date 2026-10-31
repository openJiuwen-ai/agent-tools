/**
 * Device-side tiling for RqUniformFiaMlaLatent.
 * Layout must match op_host/rq_uniform_fia_mla_latent_tiling.h.
 * Do NOT include register/tilingdata_base.h here.
 */
#ifndef MLA_FIA_DEVICE_TILING_H_
#define MLA_FIA_DEVICE_TILING_H_

#include <cstdint>

namespace optiling {

constexpr uint32_t MLA_ABS_MAX_AIC_CORE_NUM = 26;

struct MlaAbsBaseParams {
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

struct MlaAbsPageAttentionParams {
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
};

struct MlaAbsMaskParams {
    uint32_t attenMaskFlag;
    uint32_t attenMaskBatchStride;
    uint32_t attenMaskStride;
    int32_t preToken;
    int32_t nextToken;
    uint32_t isRowInvalid;
    uint32_t isExistRowInvalid;
    uint32_t sparseMode;
};

struct MlaAbsWorkspaceParams {
    uint32_t mm1ResSize;
    uint32_t mm2ResSize;
    uint32_t fdAccumOutSize;
    uint32_t fdLogSumExpSize;
    uint32_t dequantWsSlots;
    uint32_t dequantS2Cache;
};

struct MlaAbsInnerSplitParams {
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
};

struct MlaAbsOuterSplitParams {
    uint32_t bN2End[MLA_ABS_MAX_AIC_CORE_NUM];
    uint32_t gS1End[MLA_ABS_MAX_AIC_CORE_NUM];
    uint32_t s2End[MLA_ABS_MAX_AIC_CORE_NUM];
};

struct MlaAbsFlashDecodeParams {
    uint32_t numOfFdHead;
    uint32_t reserved;
    uint32_t gS1BaseSizeOfFd;
    uint32_t usedVecNumOfFd;
    uint32_t bN2IdxOfFdHead[MLA_ABS_MAX_AIC_CORE_NUM];
    uint32_t gS1IdxOfFdHead[MLA_ABS_MAX_AIC_CORE_NUM];
    uint32_t s2SplitNumOfFdHead[MLA_ABS_MAX_AIC_CORE_NUM];
    uint32_t s2SplitStartIdxOfCore[MLA_ABS_MAX_AIC_CORE_NUM];
    uint32_t gS1SplitNumOfFdHead[MLA_ABS_MAX_AIC_CORE_NUM];
    uint32_t gS1LastPartSizeOfFdHead[MLA_ABS_MAX_AIC_CORE_NUM];
    uint32_t gS1IdxEndOfFdHead[MLA_ABS_MAX_AIC_CORE_NUM * 2];
    uint32_t gS1IdxEndOfFdHeadSplit[MLA_ABS_MAX_AIC_CORE_NUM * 2];
};

struct MlaAbsPrefixParams {
    uint64_t prefixMaxLen;
    uint64_t prefixLen;
    bool prefixFlag;
};

struct MlaAbsPseParams {
    uint32_t pseShiftFlag;
    uint32_t pseShiftByBatch;
    uint32_t pseShiftS1;
    uint32_t pseShiftS2;
};

struct MlaAbsLeftPaddingParams {
    uint32_t qPaddingFlag;
    uint32_t kvPaddingFlag;
};

// Name expected by the vendored FiaKernelNonQuantMla.
struct FusedInferAttentionScoreTilingData {
    MlaAbsBaseParams baseParams;
    MlaAbsPageAttentionParams pageAttenParams;
    MlaAbsMaskParams maskParams;
    MlaAbsWorkspaceParams workspaceParams;
    MlaAbsInnerSplitParams innerSplitParams;
    MlaAbsOuterSplitParams outerSplitParams;
    MlaAbsFlashDecodeParams fdParams;
    MlaAbsPrefixParams prefixParams;
    MlaAbsPseParams pseParams;
    MlaAbsLeftPaddingParams leftPaddingParams;
};

}  // namespace optiling

using optiling::FusedInferAttentionScoreTilingData;

#endif  // MLA_FIA_DEVICE_TILING_H_
