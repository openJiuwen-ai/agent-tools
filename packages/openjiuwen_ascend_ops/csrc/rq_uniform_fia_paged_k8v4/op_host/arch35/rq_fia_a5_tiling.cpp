#include "rq_fia_a5_tiling.h"

#include <cstdint>
#include <cstring>

#include "fused_infer_attention_score_tiling_impl.h"
#include "log/ops_log.h"
#include "rq_fia_a5_checker.h"
#include "rq_fia_a5_info_parser.h"
#include "rq_fia_a5_scale.h"

namespace optiling {
namespace {

ge::graphStatus AppendFp16Ones(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    gert::TilingData* raw = context->GetRawTilingData();
    if (raw == nullptr || raw->GetData() == nullptr) {
        OPS_LOG_E(nodeName, "tiling buffer is null");
        return ge::GRAPH_FAILED;
    }
    const size_t scaleOff = RqFiaA5ScaleByteOffset();
    const size_t tail = kRqFiaA5ScaleTailBytes;
    if (raw->GetCapacity() < scaleOff + tail) {
        OPS_LOG_E(nodeName, "tiling buffer capacity %zu is smaller than scale tail %zu",
                  raw->GetCapacity(), scaleOff + tail);
        return ge::GRAPH_FAILED;
    }
    auto* bytes = static_cast<uint8_t*>(raw->GetData());
    const uint16_t fp16One = 0x3c00;
    for (size_t i = 0; i < tail / sizeof(uint16_t); ++i) {
        std::memcpy(bytes + scaleOff + i * sizeof(uint16_t), &fp16One, sizeof(uint16_t));
    }
    raw->SetDataSize(scaleOff + tail);
    return ge::GRAPH_SUCCESS;
}

}  // namespace

ge::graphStatus RqFiaA5PortTiling(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    RqFiaA5Parsed parsed;
    if (RqFiaA5Parse(context, parsed) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (RqFiaA5Check(parsed) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    FusedInferAttentionScoreTilingImpl impl(context);
    if (impl.DoOpTiling(context, parsed.info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return AppendFp16Ones(context);
}

}  // namespace optiling
