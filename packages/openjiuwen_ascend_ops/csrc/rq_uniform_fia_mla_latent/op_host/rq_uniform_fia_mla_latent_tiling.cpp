#include "rq_uniform_fia_mla_latent_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/ops_log.h"

namespace optiling {
namespace {
constexpr uint32_t kLatent = 512;
constexpr uint32_t kRope = 64;
constexpr uint32_t kS2 = 128;
constexpr uint32_t kMBaseSmall = 128;
constexpr uint32_t kMBaseLarge = 256;
constexpr uint32_t kVectorTile = 16;
constexpr uint32_t kPreload = 2;
constexpr uint64_t kLibApiWorkspaceFallback = 16ULL * 1024ULL * 1024ULL;
constexpr uint32_t kTndLayout = 3;
constexpr uint64_t kYTileBytes = 4ULL * kS2 * kLatent * sizeof(uint16_t);

uint64_t FiaWorkspaceBytes(uint32_t cores, uint32_t mm1Elems, uint32_t mm2Elems) {
    const uint64_t mm1 = static_cast<uint64_t>(mm1Elems);
    const uint64_t mm2 = static_cast<uint64_t>(mm2Elems);
    // Matches FiaKernelNonQuantMla::Init: mm1 fp32, vec1 fp16, mm2 fp32, vec2 fp32.
    return static_cast<uint64_t>(cores) * kPreload * (mm1 * 4ULL + mm1 * 2ULL + mm2 * 4ULL + mm2 * 4ULL);
}
}  // namespace

static ge::graphStatus RqUniformFiaMlaLatentTilingFunc(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(0);
    const float* scalePtr = attrs->GetAttrPointer<float>(1);
    const int64_t* numTokensPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* numHeadsPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t* nopeDimPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t* vDimPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t* numBlocksPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t* numReqsPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t* maxBlocksPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t* maxKvPtr = attrs->GetAttrPointer<int64_t>(9);
    if (blockSizePtr == nullptr || scalePtr == nullptr || numTokensPtr == nullptr || numHeadsPtr == nullptr ||
        nopeDimPtr == nullptr || vDimPtr == nullptr || numBlocksPtr == nullptr || numReqsPtr == nullptr ||
        maxBlocksPtr == nullptr || maxKvPtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs missing");
        return ge::GRAPH_FAILED;
    }
    if (*blockSizePtr <= 0 || (*blockSizePtr % kVectorTile) != 0) {
        OPS_LOG_E(nodeName, "block_size must be a positive multiple of 16");
        return ge::GRAPH_FAILED;
    }
    if (*numTokensPtr <= 0 || *numHeadsPtr <= 0 || *nopeDimPtr <= 0 || *vDimPtr <= 0 || *numBlocksPtr <= 0 ||
        *numReqsPtr <= 0 || *maxBlocksPtr <= 0) {
        OPS_LOG_E(nodeName, "shape attrs must be positive");
        return ge::GRAPH_FAILED;
    }

    // Inputs: q_nope,q_pe,latent,k_pe,block_table,actual_seq_q,actual_seq_kv,w_uk,w_uv,rotation
    auto rotDesc = context->GetInputDesc(9);
    uint32_t dtype = 0;
    if (rotDesc != nullptr && rotDesc->GetDataType() == ge::DT_BF16) {
        dtype = 1;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aicNum == 0 || aivNum < 2) {
        OPS_LOG_E(nodeName, "invalid core count for MIX 1C2V");
        return ge::GRAPH_FAILED;
    }
    const uint32_t dataCores = std::min(aicNum, aivNum / 2);
    const bool fuseFia = *maxKvPtr > 0;

    RqUniformFiaMlaLatentTilingData tilingData;
    uint32_t zeros26[MLA_ABS_MAX_AIC_CORE_NUM] = {0};
    uint32_t zeros52[MLA_ABS_MAX_AIC_CORE_NUM * 2] = {0};
    tilingData.outerSplitParams.set_bN2End(zeros26);
    tilingData.outerSplitParams.set_gS1End(zeros26);
    tilingData.outerSplitParams.set_s2End(zeros26);
    tilingData.fdParams.set_bN2IdxOfFdHead(zeros26);
    tilingData.fdParams.set_gS1IdxOfFdHead(zeros26);
    tilingData.fdParams.set_s2SplitNumOfFdHead(zeros26);
    tilingData.fdParams.set_s2SplitStartIdxOfCore(zeros26);
    tilingData.fdParams.set_gS1SplitNumOfFdHead(zeros26);
    tilingData.fdParams.set_gS1LastPartSizeOfFdHead(zeros26);
    tilingData.fdParams.set_gS1IdxEndOfFdHead(zeros52);
    tilingData.fdParams.set_gS1IdxEndOfFdHeadSplit(zeros52);
    uint32_t usedCores = dataCores;
    uint32_t tilingKey = 0;
    uint64_t userBytes = static_cast<uint64_t>(dataCores) * kYTileBytes;

    tilingData.baseParams.set_bSize(static_cast<uint32_t>(*numReqsPtr));
    tilingData.baseParams.set_n2Size(1);
    tilingData.baseParams.set_gSize(static_cast<uint32_t>(*numHeadsPtr));
    tilingData.baseParams.set_headDim(static_cast<uint32_t>(*nopeDimPtr));
    tilingData.baseParams.set_headDimRope(kRope);
    tilingData.baseParams.set_scaleValue(*scalePtr);
    tilingData.baseParams.set_outputLayout(kTndLayout);
    tilingData.baseParams.set_batchContinuous(1);
    tilingData.baseParams.set_softmaxLseFlag(0);
    tilingData.baseParams.set_needInit(0);
    tilingData.baseParams.set_slidingFlag(0);
    tilingData.baseParams.set_l2CacheOffFlag(dtype);
    tilingData.baseParams.set_isLegacyIfa(0);
    tilingData.pageAttenParams.set_blockSize(static_cast<uint32_t>(*blockSizePtr));
    tilingData.pageAttenParams.set_maxBlockNumPerBatch(static_cast<uint32_t>(*maxBlocksPtr));
    tilingData.workspaceParams.set_dequantWsSlots(static_cast<uint32_t>(*numBlocksPtr));
    tilingData.workspaceParams.set_dequantS2Cache(0);
    tilingData.workspaceParams.set_fdAccumOutSize(0);
    tilingData.workspaceParams.set_fdLogSumExpSize(0);
    tilingData.maskParams.set_attenMaskFlag(0);
    tilingData.maskParams.set_attenMaskBatchStride(0);
    tilingData.maskParams.set_attenMaskStride(0);
    tilingData.maskParams.set_preToken(0);
    tilingData.maskParams.set_nextToken(0);
    tilingData.maskParams.set_isRowInvalid(0);
    tilingData.maskParams.set_isExistRowInvalid(0);
    tilingData.maskParams.set_sparseMode(0);
    tilingData.prefixParams.set_prefixMaxLen(0);
    tilingData.prefixParams.set_prefixLen(0);
    tilingData.prefixParams.set_prefixFlag(false);
    tilingData.pseParams.set_pseShiftFlag(0);
    tilingData.pseParams.set_pseShiftByBatch(0);
    tilingData.pseParams.set_pseShiftS1(0);
    tilingData.pseParams.set_pseShiftS2(0);
    tilingData.leftPaddingParams.set_qPaddingFlag(0);
    tilingData.leftPaddingParams.set_kvPaddingFlag(0);
    tilingData.fdParams.set_numOfFdHead(0);
    tilingData.fdParams.set_reserved(0);
    tilingData.fdParams.set_gS1BaseSizeOfFd(0);
    tilingData.fdParams.set_usedVecNumOfFd(0);

    if (!fuseFia) {
        tilingData.baseParams.set_s1Size(static_cast<uint32_t>(*numTokensPtr));
        tilingData.baseParams.set_s2Size(1);
        tilingData.baseParams.set_actualSeqS1Dims(0);
        tilingData.baseParams.set_actualSeqS2Dims(0);
        tilingData.baseParams.set_accumQSeqFlag(0);
        tilingData.baseParams.set_accumKVSeqFlag(0);
        tilingData.baseParams.set_usedCoreNum(dataCores);
        tilingData.innerSplitParams.set_mBaseSize(kMBaseSmall);
        tilingData.innerSplitParams.set_s2BaseSize(kS2);
        tilingData.workspaceParams.set_mm1ResSize(0);
        tilingData.workspaceParams.set_mm2ResSize(0);
    } else {
        if (*nopeDimPtr != static_cast<int64_t>(kLatent)) {
            OPS_LOG_E(nodeName, "fused MLA requires nope dim 512");
            return ge::GRAPH_FAILED;
        }
        if (*numTokensPtr % *numReqsPtr != 0) {
            OPS_LOG_E(nodeName, "fused MLA requires uniform query length");
            return ge::GRAPH_FAILED;
        }
        const uint32_t s1 = static_cast<uint32_t>(*numTokensPtr / *numReqsPtr);
        const uint32_t heads = static_cast<uint32_t>(*numHeadsPtr);
        const uint32_t gS1 = s1 * heads;
        if (gS1 == 0 || gS1 > kMBaseLarge) {
            OPS_LOG_E(nodeName, "fused MLA query group exceeds one S1 tile");
            return ge::GRAPH_FAILED;
        }
        const uint32_t mBase = gS1 <= kMBaseSmall ? kMBaseSmall : kMBaseLarge;
        const uint32_t numReqs = static_cast<uint32_t>(*numReqsPtr);
        usedCores = std::min(dataCores, std::min(numReqs, MLA_ABS_MAX_AIC_CORE_NUM));
        if (usedCores == 0) {
            OPS_LOG_E(nodeName, "no AIC cores for fused MLA");
            return ge::GRAPH_FAILED;
        }
        uint32_t bN2End[MLA_ABS_MAX_AIC_CORE_NUM] = {0};
        uint32_t gS1End[MLA_ABS_MAX_AIC_CORE_NUM] = {0};
        uint32_t s2End[MLA_ABS_MAX_AIC_CORE_NUM] = {0};
        const uint32_t base = numReqs / usedCores;
        const uint32_t rem = numReqs % usedCores;
        uint32_t end = 0;
        for (uint32_t i = 0; i < usedCores; ++i) {
            end += base + (i < rem ? 1U : 0U);
            bN2End[i] = end;
        }
        tilingData.outerSplitParams.set_bN2End(bN2End);
        tilingData.outerSplitParams.set_gS1End(gS1End);
        tilingData.outerSplitParams.set_s2End(s2End);

        const uint32_t mm1Elems = mBase * kS2;
        const uint32_t mm2Elems = mBase * kLatent;
        tilingData.baseParams.set_s1Size(s1);
        tilingData.baseParams.set_s2Size(static_cast<uint32_t>(*maxKvPtr));
        tilingData.baseParams.set_actualSeqS1Dims(numReqs);
        tilingData.baseParams.set_actualSeqS2Dims(numReqs);
        tilingData.baseParams.set_accumQSeqFlag(1);
        tilingData.baseParams.set_accumKVSeqFlag(0);
        tilingData.baseParams.set_usedCoreNum(usedCores);
        tilingData.innerSplitParams.set_mBaseSize(mBase);
        tilingData.innerSplitParams.set_s2BaseSize(kS2);
        tilingData.workspaceParams.set_mm1ResSize(mm1Elems);
        tilingData.workspaceParams.set_mm2ResSize(mm2Elems);
        // Q/KV actual-seq live in caller GM (int64); no lens/trace workspace.
        userBytes = FiaWorkspaceBytes(usedCores, mm1Elems, mm2Elems) +
            static_cast<uint64_t>(usedCores) * kYTileBytes;
        tilingKey = 1;
    }

    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < tilingData.GetDataSize()) {
        OPS_LOG_E(nodeName, "raw tiling buffer is null or too small");
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tilingData.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OPS_LOG_E(nodeName, "workspace size buffer is null");
        return ge::GRAPH_FAILED;
    }
    size_t libapiSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (libapiSize == 0) {
        libapiSize = kLibApiWorkspaceFallback;
    }
    workspaces[0] = libapiSize + static_cast<size_t>(userBytes);
    const uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(usedCores * 2, usedCores, usedCores * 2);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RqUniformFiaMlaLatent).Tiling(RqUniformFiaMlaLatentTilingFunc);

}  // namespace optiling
