/**
 * Standalone FIA TQ tiling-data structs for vllm-ascend custom op.
 * Layout must match ops-transformer FusedInferAttentionScoreTilingData
 * fields consumed by fia_kernel_turboquant_p0.h (no IFA/PFA host deps).
 */
#ifndef TURBOQUANT_FIA_TILING_DATA_H_
#define TURBOQUANT_FIA_TILING_DATA_H_

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t TQ_FIA_MAX_AIC_CORE_NUM = 26;

BEGIN_TILING_DATA_DEF(TqFiaBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, headDimRope)
TILING_DATA_FIELD_DEF(uint32_t, actualSeqS1Dims)
TILING_DATA_FIELD_DEF(uint32_t, actualSeqS2Dims)
TILING_DATA_FIELD_DEF(uint32_t, accumQSeqFlag)
TILING_DATA_FIELD_DEF(uint32_t, accumKVSeqFlag)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint32_t, batchContinuous)
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
TILING_DATA_FIELD_DEF(uint32_t, needInit)
TILING_DATA_FIELD_DEF(uint32_t, slidingFlag)
TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
TILING_DATA_FIELD_DEF(uint32_t, isLegacyIfa)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TqFiaBaseParamsOp, TqFiaBaseParams)

BEGIN_TILING_DATA_DEF(TqFiaPageAttentionParams)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TqFiaPageAttentionParamsOp, TqFiaPageAttentionParams)

BEGIN_TILING_DATA_DEF(TqFiaMaskParams)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatchStride)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskStride)
TILING_DATA_FIELD_DEF(int32_t, preToken)
TILING_DATA_FIELD_DEF(int32_t, nextToken)
TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid)
TILING_DATA_FIELD_DEF(uint32_t, isExistRowInvalid)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TqFiaMaskParamsOp, TqFiaMaskParams)

BEGIN_TILING_DATA_DEF(TqFiaWorkspaceParams)
TILING_DATA_FIELD_DEF(uint32_t, mm1ResSize)
TILING_DATA_FIELD_DEF(uint32_t, mm2ResSize)
TILING_DATA_FIELD_DEF(uint32_t, fdAccumOutSize)
TILING_DATA_FIELD_DEF(uint32_t, fdLogSumExpSize)
// Dequant GM slots per AIC (K and V each). 2 = classic loop%2 ping-pong.
// When == s2LoopTimes and dequantS2Cache!=0, device indexes by s2Idx and may skip.
TILING_DATA_FIELD_DEF(uint32_t, dequantWsSlots)
TILING_DATA_FIELD_DEF(uint32_t, dequantS2Cache)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TqFiaWorkspaceParamsOp, TqFiaWorkspaceParams)

BEGIN_TILING_DATA_DEF(TqFiaInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TqFiaInnerSplitParamsOp, TqFiaInnerSplitParams)

BEGIN_TILING_DATA_DEF(TqFiaOuterSplitParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM, bN2End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM, gS1End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM, s2End)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TqFiaOuterSplitParamsOp, TqFiaOuterSplitParams)

BEGIN_TILING_DATA_DEF(TqFiaFlashDecodeParams)
TILING_DATA_FIELD_DEF(uint32_t, numOfFdHead)
TILING_DATA_FIELD_DEF(uint32_t, reserved)
TILING_DATA_FIELD_DEF(uint32_t, gS1BaseSizeOfFd)
TILING_DATA_FIELD_DEF(uint32_t, usedVecNumOfFd)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM, bN2IdxOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM, gS1IdxOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM, s2SplitNumOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM, s2SplitStartIdxOfCore)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM, gS1SplitNumOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM, gS1LastPartSizeOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, TQ_FIA_MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHeadSplit)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TqFiaFlashDecodeParamsOp, TqFiaFlashDecodeParams)

BEGIN_TILING_DATA_DEF(TqFiaPrefixParams)
TILING_DATA_FIELD_DEF(uint64_t, prefixMaxLen)
TILING_DATA_FIELD_DEF(uint64_t, prefixLen)
TILING_DATA_FIELD_DEF(bool, prefixFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TqFiaPrefixParamsOp, TqFiaPrefixParams)

BEGIN_TILING_DATA_DEF(TqFiaPseParams)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftFlag)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftByBatch)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS1)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS2)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TqFiaPseParamsOp, TqFiaPseParams)

BEGIN_TILING_DATA_DEF(TqFiaLeftPaddingParams)
TILING_DATA_FIELD_DEF(uint32_t, qPaddingFlag)
TILING_DATA_FIELD_DEF(uint32_t, kvPaddingFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(TqFiaLeftPaddingParamsOp, TqFiaLeftPaddingParams)

// Binary-compatible alias used by kernel GET_TILING_DATA_WITH_STRUCT.
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TqFiaBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaPageAttentionParams, pageAttenParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaMaskParams, maskParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaWorkspaceParams, workspaceParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaOuterSplitParams, outerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaFlashDecodeParams, fdParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaPrefixParams, prefixParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaPseParams, pseParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaLeftPaddingParams, leftPaddingParams);
END_TILING_DATA_DEF

// Keep existing REGISTER name used by OpDef / IMPL_OP_OPTILING.
BEGIN_TILING_DATA_DEF(RqUniformFiaPagedK8v4TilingData)
TILING_DATA_FIELD_DEF_STRUCT(TqFiaBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaPageAttentionParams, pageAttenParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaMaskParams, maskParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaWorkspaceParams, workspaceParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaOuterSplitParams, outerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaFlashDecodeParams, fdParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaPrefixParams, prefixParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaPseParams, pseParams);
TILING_DATA_FIELD_DEF_STRUCT(TqFiaLeftPaddingParams, leftPaddingParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(RqUniformFiaPagedK8v4,
                           RqUniformFiaPagedK8v4TilingData)

}  // namespace optiling

#endif  // TURBOQUANT_FIA_TILING_DATA_H_
