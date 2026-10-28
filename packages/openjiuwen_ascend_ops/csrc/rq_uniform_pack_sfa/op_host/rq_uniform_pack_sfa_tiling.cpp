#include "rq_uniform_pack_sfa_tiling.h"

#include <algorithm>
#include <cstdlib>

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {

constexpr uint32_t kSlotBytes = 388;
constexpr uint32_t kNopeDim = 512;
constexpr uint32_t kRopeDim = 64;
constexpr uint32_t kTileRows = 32;
constexpr size_t kLibApiWorkspaceFallback = 16 * 1024 * 1024;

// Force MIX legacy (tilingKey=0) for A/B: export JIUWEN_PACK_SFA_LEGACY=1
bool ForceLegacyFromEnv()
{
    const char* env = std::getenv("JIUWEN_PACK_SFA_LEGACY");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

}  // namespace

static ge::graphStatus RqUniformPackSfaTilingFunc(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* numTokensPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* numBlocksPtr = attrs->GetAttrPointer<int64_t>(2);
    if (blockSizePtr == nullptr || numTokensPtr == nullptr || numBlocksPtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }
    if (*blockSizePtr <= 0 || *numTokensPtr <= 0 || *numBlocksPtr <= 0) {
        OPS_LOG_E(nodeName, "block_size/num_tokens/num_blocks must be positive");
        return ge::GRAPH_FAILED;
    }
    const uint32_t blockSize = static_cast<uint32_t>(*blockSizePtr);
    const uint32_t numTokens = static_cast<uint32_t>(*numTokensPtr);
    const uint32_t numBlocks = static_cast<uint32_t>(*numBlocksPtr);
    if (blockSize % 16 != 0) {
        OPS_LOG_E(nodeName, "block_size must be a multiple of 16, got %u", blockSize);
        return ge::GRAPH_FAILED;
    }

    auto nopeShape = context->GetInputShape(0);
    auto ropeShape = context->GetInputShape(1);
    auto cacheShape = context->GetOutputShape(0);
    if (nopeShape == nullptr || ropeShape == nullptr || cacheShape == nullptr) {
        OPS_LOG_E(nodeName, "nope/rope/kv_cache shapes are null");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& nope = nopeShape->GetStorageShape();
    const gert::Shape& rope = ropeShape->GetStorageShape();
    const gert::Shape& cache = cacheShape->GetStorageShape();
    if (nope.GetDimNum() != 2 || nope.GetDim(0) != numTokens || nope.GetDim(1) != kNopeDim) {
        OPS_LOG_E(nodeName, "nope must be [T, 512]");
        return ge::GRAPH_FAILED;
    }
    if (rope.GetDimNum() != 2 || rope.GetDim(0) != numTokens || rope.GetDim(1) != kRopeDim) {
        OPS_LOG_E(nodeName, "rope must be [T, 64]");
        return ge::GRAPH_FAILED;
    }
    if (cache.GetDimNum() != 4 || cache.GetDim(0) != numBlocks || cache.GetDim(1) != blockSize ||
        cache.GetDim(2) != 1 || cache.GetDim(3) != kSlotBytes) {
        OPS_LOG_E(nodeName, "kv_cache must be PA_BSND [Bn, Bs, 1, 388]");
        return ge::GRAPH_FAILED;
    }

    auto nopeDesc = context->GetInputDesc(0);
    if (nopeDesc == nullptr) {
        OPS_LOG_E(nodeName, "nope desc is null");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dataType = nopeDesc->GetDataType();
    uint32_t dtype = 0;
    if (dataType == ge::DT_FLOAT16) {
        dtype = 0;
    } else if (dataType == ge::DT_BF16) {
        dtype = 1;
    } else {
        OPS_LOG_E(nodeName, "nope must be fp16 or bf16");
        return ge::GRAPH_FAILED;
    }

    auto slotDesc = context->GetInputDesc(2);
    if (slotDesc == nullptr) {
        OPS_LOG_E(nodeName, "slot_mapping desc is null");
        return ge::GRAPH_FAILED;
    }
    if (slotDesc->GetDataType() != ge::DT_INT64) {
        OPS_LOG_E(nodeName, "slot_mapping must be int64");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();

    // Default V2 (key=1). Set JIUWEN_PACK_SFA_LEGACY=1 to force MIX legacy (key=0).
    const bool useLegacy = ForceLegacyFromEnv();
    uint32_t dataCores = 0;
    uint32_t blockDim = 0;
    uint32_t tileRows = 0;
    uint64_t tilingKey = 0;

    if (useLegacy) {
        if (aicNum == 0 || aivNum < 2) {
            OPS_LOG_E(nodeName, "invalid core count for MIX 1C2V: aic=%u aiv=%u", aicNum, aivNum);
            return ge::GRAPH_FAILED;
        }
        dataCores = std::min(aicNum, aivNum / 2);
        blockDim = ascendcPlatform.CalcTschBlockDim(dataCores * 2, dataCores, dataCores * 2);
        tileRows = 0;
        tilingKey = 0;
    } else {
        if (aivNum == 0) {
            OPS_LOG_E(nodeName, "invalid AIV count: 0");
            return ge::GRAPH_FAILED;
        }
        dataCores = std::min(aivNum, std::max(numTokens, 1u));
        blockDim = dataCores;
        tileRows = kTileRows;
        tilingKey = 1;
    }

    RqUniformPackSfaTilingData tilingData;
    tilingData.set_numTokens(numTokens);
    tilingData.set_blockSize(blockSize);
    tilingData.set_numBlocks(numBlocks);
    tilingData.set_dataCores(dataCores);
    tilingData.set_dtype(dtype);
    tilingData.set_tileRows(tileRows);

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
    workspaces[0] = libapiSize;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RqUniformPackSfa).Tiling(RqUniformPackSfaTilingFunc);

}  // namespace optiling
