#include "rq_fia_a5_checker.h"

#include "log/ops_log.h"
#include "rq_fia_a5_tiling.h"

namespace optiling {

ge::graphStatus RqFiaA5Check(const RqFiaA5Parsed& parsed)
{
    const FiaTilingInfo& info = parsed.info;
    const char* nodeName = info.opName != nullptr ? info.opName : "RqUniformFiaPagedK8v4";
    if (info.npuArch != NpuArch::DAV_3510) {
        OPS_LOG_E(nodeName, "A5 antiquant port is ascend950 only");
        return ge::GRAPH_FAILED;
    }
    if (info.s1Size < 1) {
        OPS_LOG_E(nodeName, "A5 antiquant port requires s1>=1, got %u", info.s1Size);
        return ge::GRAPH_FAILED;
    }
    if (info.qkHeadDim != info.vHeadDim ||
        (info.qkHeadDim != kRqFiaA5HeadDim128 && info.qkHeadDim != kRqFiaA5HeadDim256)) {
        OPS_LOG_E(nodeName, "A5 antiquant port requires D=128 or 256, got Dq=%u Dv=%u",
                  info.qkHeadDim, info.vHeadDim);
        return ge::GRAPH_FAILED;
    }
    if (info.opParamInfo.blockTable.tensor == nullptr) {
        OPS_LOG_E(nodeName, "block_table is required");
        return ge::GRAPH_FAILED;
    }
    if (parsed.hasMask || parsed.hasRotation) {
        OPS_LOG_E(nodeName, "A5 antiquant port does not take a mask or a rotation");
        return ge::GRAPH_FAILED;
    }
    if (info.attenMaskFlag || info.pseShiftFlag || info.ropeMode != RopeMode::NO_ROPE) {
        OPS_LOG_E(nodeName, "mask, PSE, and rope must stay off");
        return ge::GRAPH_FAILED;
    }
    if (info.n2Size == 0 || info.n1Size == 0 || info.n1Size % info.n2Size != 0 ||
        info.gSize != info.n1Size / info.n2Size) {
        OPS_LOG_E(nodeName, "num_heads (%u) must be divisible by num_kv_heads (%u)", info.n1Size, info.n2Size);
        return ge::GRAPH_FAILED;
    }
    if (info.blockSize <= 0 || info.bSize == 0 || info.maxBlockNumPerBatch == 0) {
        OPS_LOG_E(nodeName, "invalid batch/block_table");
        return ge::GRAPH_FAILED;
    }
    if (info.actualLenKvDims != info.bSize || info.actualLenQDims != info.bSize) {
        OPS_LOG_E(nodeName, "actual seq length count must equal batch, q=%u kv=%u B=%u",
                  info.actualLenQDims, info.actualLenKvDims, info.bSize);
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor* seqKv = info.opParamInfo.actualSeqLengths.tensor;
    if (seqKv == nullptr || seqKv->GetData<int64_t>() == nullptr) {
        OPS_LOG_E(nodeName, "actual_seq_len_kv is missing");
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor* seqQ = info.opParamInfo.actualSeqLengthsQ.tensor;
    if (seqQ == nullptr || seqQ->GetData<int64_t>() == nullptr) {
        OPS_LOG_E(nodeName, "actual_seq_len_q is missing");
        return ge::GRAPH_FAILED;
    }
    const int64_t covered = static_cast<int64_t>(info.maxBlockNumPerBatch) * info.blockSize;
    const int64_t* kvLens = seqKv->GetData<int64_t>();
    const int64_t* qLens = seqQ->GetData<int64_t>();
    for (uint32_t b = 0; b < info.bSize; ++b) {
        if (qLens[b] < 1 || qLens[b] > static_cast<int64_t>(info.s1Size)) {
            OPS_LOG_E(nodeName, "actual q length %ld is outside [1, s1=%u]", qLens[b], info.s1Size);
            return ge::GRAPH_FAILED;
        }
        if (kvLens[b] < 0 || kvLens[b] > covered) {
            OPS_LOG_E(nodeName, "actual kv length %ld is outside the block table (%ld)", kvLens[b], covered);
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t expectLast = static_cast<int64_t>(info.blockSize) * static_cast<int64_t>(info.qkHeadDim);
    const gert::StorageShape* keyShape = info.opParamInfo.key.shape;
    const gert::StorageShape* valueShape = info.opParamInfo.value.shape;
    if (keyShape == nullptr || valueShape == nullptr || keyShape->GetStorageShape().GetDimNum() != 3 ||
        valueShape->GetStorageShape().GetDimNum() != 3) {
        OPS_LOG_E(nodeName, "key/value cache must be [blocks, kv_heads, block_size * D]");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& key = keyShape->GetStorageShape();
    const gert::Shape& value = valueShape->GetStorageShape();
    if (key.GetDim(1) != static_cast<int64_t>(info.n2Size) || value.GetDim(1) != static_cast<int64_t>(info.n2Size) ||
        key.GetDim(2) != expectLast || value.GetDim(2) != expectLast) {
        OPS_LOG_E(nodeName,
                  "cache last dim must be block_size * D (%ld int8), got key %ld value %ld",
                  expectLast, key.GetDim(2), value.GetDim(2));
        return ge::GRAPH_FAILED;
    }
    if (info.opParamInfo.blockTable.tensor->GetStorageShape().GetDim(0) != static_cast<int64_t>(info.bSize)) {
        OPS_LOG_E(nodeName, "block_table batch does not match query");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
