#include "rq_uniform_dequant_mla_latent_tiling.h"

#include <algorithm>

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {

constexpr uint32_t kCubeGroup = 32;
constexpr uint64_t kWorkspaceHeadBytes = 512 * 1024;
constexpr size_t kLibApiWorkspaceFallback = 16 * 1024 * 1024;
constexpr uint32_t kMaxLatentDim = 512;
constexpr uint32_t kLatentAlign = 64;

}  // namespace

static ge::graphStatus RqUniformDequantMlaLatentTilingFunc(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* latentDimPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* numBlocksPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* numReqsPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t* maxBlocksPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t* totalTokensPtr = attrs->GetAttrPointer<int64_t>(5);
    if (blockSizePtr == nullptr || latentDimPtr == nullptr || numBlocksPtr == nullptr ||
        numReqsPtr == nullptr || maxBlocksPtr == nullptr || totalTokensPtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }
    if (*blockSizePtr <= 0 || *latentDimPtr <= 0 || *numBlocksPtr <= 0 || *numReqsPtr <= 0 ||
        *maxBlocksPtr <= 0 || *totalTokensPtr <= 0) {
        OPS_LOG_E(nodeName, "block_size/latent_dim/num_blocks/num_reqs/max_blocks/total_tokens must be positive");
        return ge::GRAPH_FAILED;
    }
    const uint32_t blockSize = static_cast<uint32_t>(*blockSizePtr);
    const uint32_t latentDim = static_cast<uint32_t>(*latentDimPtr);
    const uint32_t numBlocks = static_cast<uint32_t>(*numBlocksPtr);
    const uint32_t numReqs = static_cast<uint32_t>(*numReqsPtr);
    const uint32_t maxBlocks = static_cast<uint32_t>(*maxBlocksPtr);
    const uint32_t totalTokens = static_cast<uint32_t>(*totalTokensPtr);
    if (blockSize % 16 != 0) {
        OPS_LOG_E(nodeName, "block_size must be a multiple of 16, got %u", blockSize);
        return ge::GRAPH_FAILED;
    }
    if (latentDim % kLatentAlign != 0 || latentDim > kMaxLatentDim) {
        OPS_LOG_E(nodeName, "latent_dim must be a multiple of 64 and <= 512, got %u", latentDim);
        return ge::GRAPH_FAILED;
    }

    auto rotDesc = context->GetInputDesc(1);
    if (rotDesc == nullptr) {
        OPS_LOG_E(nodeName, "rotation desc is null");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dataType = rotDesc->GetDataType();
    uint32_t dtype = 0;
    if (dataType == ge::DT_FLOAT16) {
        dtype = 0;
    } else if (dataType == ge::DT_BF16) {
        dtype = 1;
    } else {
        OPS_LOG_E(nodeName, "rotation must be fp16 or bf16");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aicNum == 0 || aivNum < 2) {
        OPS_LOG_E(nodeName, "invalid core count for MIX 1C2V: aic=%u aiv=%u", aicNum, aivNum);
        return ge::GRAPH_FAILED;
    }
    const uint32_t dataCores = std::min(aicNum, aivNum / 2);
    const uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(dataCores * 2, dataCores, dataCores * 2);

    RqUniformDequantMlaLatentTilingData tilingData;
    tilingData.set_totalTokens(totalTokens);
    tilingData.set_latentDim(latentDim);
    tilingData.set_blockSize(blockSize);
    tilingData.set_numBlocks(numBlocks);
    tilingData.set_numReqs(numReqs);
    tilingData.set_maxBlocks(maxBlocks);
    tilingData.set_dataCores(dataCores);
    tilingData.set_dtype(dtype);

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
    const uint64_t bridgeBytes =
        kWorkspaceHeadBytes + static_cast<uint64_t>(dataCores) * 2 * kCubeGroup * latentDim * sizeof(uint16_t);
    workspaces[0] = libapiSize + static_cast<size_t>(bridgeBytes);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RqUniformDequantMlaLatent).Tiling(RqUniformDequantMlaLatentTilingFunc);

}  // namespace optiling
