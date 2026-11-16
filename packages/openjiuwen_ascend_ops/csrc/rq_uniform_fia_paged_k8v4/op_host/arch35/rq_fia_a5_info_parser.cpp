#include "rq_fia_a5_info_parser.h"

#include "log/ops_log.h"
#include "rq_fia_a5_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

constexpr size_t kInputQuery = 0;
constexpr size_t kInputKey = 1;
constexpr size_t kInputValue = 2;
constexpr size_t kInputBlockTable = 3;
constexpr size_t kInputSeqQ = 4;
constexpr size_t kInputSeqKv = 5;
constexpr size_t kInputMask = 6;
constexpr size_t kInputRotationKey = 7;
constexpr size_t kInputRotationValue = 8;
constexpr size_t kAttrNumHeads = 0;
constexpr size_t kAttrNumKvHeads = 1;
constexpr size_t kAttrHeadSize = 2;
constexpr size_t kAttrBlockSize = 3;
constexpr size_t kAttrScale = 4;
constexpr size_t kAttrPreTokens = 5;
constexpr size_t kAttrNextTokens = 6;
constexpr size_t kAttrSparseMode = 7;
constexpr int64_t kSparseDefault = 2147483647;

bool TensorHasElements(const gert::StorageShape* shape)
{
    return shape != nullptr && shape->GetStorageShape().GetShapeSize() != 0;
}

}  // namespace

ge::graphStatus RqFiaA5Parse(gert::TilingContext* context, RqFiaA5Parsed& parsed)
{
    const char* nodeName = context->GetNodeName();
    auto* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OPS_LOG_E(nodeName, "platformInfo is null");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetCurNpuArch() != NpuArch::DAV_3510) {
        OPS_LOG_E(nodeName, "quant_mode=%ld requires ascend950 (DAV_3510)", kRqFiaQuantModeA5Port);
        return ge::GRAPH_FAILED;
    }
    const uint32_t aicNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    const uint32_t aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    if (aicNum == 0 || aivNum == 0 || ubSize == 0 || l1Size == 0) {
        OPS_LOG_E(nodeName, "invalid platform aic=%u aiv=%u ub=%lu l1=%lu", aicNum, aivNum, ubSize, l1Size);
        return ge::GRAPH_FAILED;
    }

    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* numHeadsPtr = attrs->GetAttrPointer<int64_t>(kAttrNumHeads);
    const int64_t* numKvHeadsPtr = attrs->GetAttrPointer<int64_t>(kAttrNumKvHeads);
    const int64_t* headSizePtr = attrs->GetAttrPointer<int64_t>(kAttrHeadSize);
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(kAttrBlockSize);
    const float* scalePtr = attrs->GetAttrPointer<float>(kAttrScale);
    if (numHeadsPtr == nullptr || numKvHeadsPtr == nullptr || headSizePtr == nullptr ||
        blockSizePtr == nullptr || scalePtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* prePtr = attrs->GetAttrPointer<int64_t>(kAttrPreTokens);
    const int64_t* nextPtr = attrs->GetAttrPointer<int64_t>(kAttrNextTokens);
    const int64_t* sparsePtr = attrs->GetAttrPointer<int64_t>(kAttrSparseMode);

    const gert::StorageShape* queryShape = context->GetInputShape(kInputQuery);
    const gert::StorageShape* keyShape = context->GetInputShape(kInputKey);
    const gert::StorageShape* valueShape = context->GetInputShape(kInputValue);
    const gert::Tensor* blockTable = context->GetInputTensor(kInputBlockTable);
    const gert::Tensor* seqQ = context->GetInputTensor(kInputSeqQ);
    const gert::Tensor* seqKv = context->GetInputTensor(kInputSeqKv);
    const gert::StorageShape* outShape = context->GetOutputShape(0);
    if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr || blockTable == nullptr ||
        seqQ == nullptr || seqKv == nullptr || outShape == nullptr) {
        OPS_LOG_E(nodeName, "required tensors are null");
        return ge::GRAPH_FAILED;
    }
    const int64_t* seqKvData = seqKv->GetData<int64_t>();
    const int64_t* seqQData = seqQ->GetData<int64_t>();
    if (seqKvData == nullptr || seqQData == nullptr) {
        OPS_LOG_E(nodeName, "actual seq length data is null");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& q = queryShape->GetStorageShape();
    uint32_t bSize = 0;
    uint32_t n1 = 0;
    uint32_t s1 = 1;
    uint32_t headDim = 0;
    if (q.GetDimNum() == 3) {
        bSize = static_cast<uint32_t>(q.GetDim(0));
        n1 = static_cast<uint32_t>(q.GetDim(1));
        s1 = 1;
        headDim = static_cast<uint32_t>(q.GetDim(2));
    } else if (q.GetDimNum() == 4) {
        bSize = static_cast<uint32_t>(q.GetDim(0));
        n1 = static_cast<uint32_t>(q.GetDim(1));
        s1 = static_cast<uint32_t>(q.GetDim(2));
        headDim = static_cast<uint32_t>(q.GetDim(3));
    } else {
        OPS_LOG_E(nodeName, "query rank %zu is not 3 or 4", q.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    const auto* qDesc = context->GetInputDesc(kInputQuery);
    if (qDesc == nullptr || qDesc->GetDataType() != ge::DT_FLOAT16) {
        OPS_LOG_E(nodeName, "A5 antiquant port query must be fp16");
        return ge::GRAPH_FAILED;
    }
    if (*headSizePtr != static_cast<int64_t>(headDim)) {
        OPS_LOG_E(nodeName, "head_size %ld does not match query D %u", *headSizePtr, headDim);
        return ge::GRAPH_FAILED;
    }

    const int64_t kvCount = seqKv->GetShapeSize();
    int64_t maxKv = 0;
    for (int64_t i = 0; i < kvCount; ++i) {
        if (seqKvData[i] > maxKv) {
            maxKv = seqKvData[i];
        }
    }

    FiaTilingInfo& info = parsed.info;
    info = FiaTilingInfo{};
    info.opName = nodeName;
    info.socVersion = platform_ascendc::SocVersion::ASCEND950;
    info.npuArch = NpuArch::DAV_3510;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, info.l2CacheSize);
    info.bSize = bSize;
    info.n1Size = n1;
    info.n2Size = static_cast<uint32_t>(*numKvHeadsPtr);
    info.s1Size = s1;
    info.s2Size = maxKv;
    info.qkHeadDim = headDim;
    info.vHeadDim = headDim;
    info.gSize = (*numKvHeadsPtr > 0 && (*numHeadsPtr % *numKvHeadsPtr) == 0) ?
                     static_cast<uint32_t>(*numHeadsPtr / *numKvHeadsPtr) :
                     0;
    info.scaleValue = *scalePtr;
    info.preToken = prePtr != nullptr ? *prePtr : kSparseDefault;
    info.nextToken = nextPtr != nullptr ? *nextPtr : kSparseDefault;
    info.sparseMode = sparsePtr != nullptr ? static_cast<int32_t>(*sparsePtr) : 0;
    info.blockSize = static_cast<int32_t>(*blockSizePtr);
    info.qLayout = FiaLayout::BNSD;
    info.outLayout = FiaLayout::BNSD;
    info.kvLayout = FiaLayout::BNSD;
    info.quantMode = FiaQuantMode::ANTI_QUANT;
    info.antiQuantFlag = true;
    info.kvStorageMode = KvStorageMode::PAGE_ATTENTION;
    info.pageAttentionFlag = true;
    info.ropeMode = RopeMode::NO_ROPE;
    info.mlaMode = MlaMode::NO_MLA;
    info.fullQuantMode = FiaFullQuantMode::NO_FULL_QUANT;
    info.inputQType = ge::DT_FLOAT16;
    info.outputType = ge::DT_FLOAT16;
    info.inputKvType = ge::DT_INT8;
    info.maxActualseq = maxKv;
    info.actualSeqLenFlag = true;
    info.actualLenQDims = static_cast<uint32_t>(seqQ->GetShapeSize());
    info.actualLenKvDims = static_cast<uint32_t>(kvCount);
    info.maxBlockNumPerBatch = blockTable->GetStorageShape().GetDimNum() >= 2 ?
                                   static_cast<uint32_t>(blockTable->GetStorageShape().GetDim(1)) :
                                   0;
    info.totalBlockNum = keyShape->GetStorageShape().GetDimNum() >= 1 ?
                             static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(0)) :
                             0;
    const uint64_t rowStride =
        static_cast<uint64_t>(info.blockSize) * static_cast<uint64_t>(headDim);
    info.hasViewStride = true;
    info.keyBnStride = static_cast<uint64_t>(info.n2Size) * rowStride;
    info.keyN2Stride = rowStride;
    info.valueBnStride = info.keyBnStride;
    info.valueN2Stride = rowStride;

    parsed.layout = "BNSD";
    parsed.pseType = 9;
    info.opParamInfo.layOut = parsed.layout.c_str();
    info.opParamInfo.pseType = &parsed.pseType;
    info.opParamInfo.query.shape = queryShape;
    info.opParamInfo.key.shape = keyShape;
    info.opParamInfo.value.shape = valueShape;
    info.opParamInfo.attenOut.shape = outShape;
    info.opParamInfo.blockTable.tensor = blockTable;
    info.opParamInfo.actualSeqLengthsQ.tensor = seqQ;
    info.opParamInfo.actualSeqLengths.tensor = seqKv;
    parsed.hasMask = TensorHasElements(context->GetOptionalInputShape(kInputMask));
    parsed.hasRotation = TensorHasElements(context->GetOptionalInputShape(kInputRotationKey)) ||
                         TensorHasElements(context->GetOptionalInputShape(kInputRotationValue));
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
