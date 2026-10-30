/*!
 * \file rq_uniform_kv_quant_sparse_flash_attention_infershape.cpp
 * \brief InferShape / InferDataType for RQ-uniform KV-quant SFA (shared by V2).
 */

#include <cstring>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kQueryInputSlot = 0;
constexpr uint32_t kLayoutQueryAttrSlot = 4;
constexpr uint32_t kRopeHeadDimAttrSlot = 12;

enum class QueryLayoutKind : uint8_t { kBsnd = 0, kTnd = 1, kUnknown = 255 };

QueryLayoutKind ParseQueryLayout(const char *layoutAttr)
{
    if (layoutAttr == nullptr) {
        return QueryLayoutKind::kUnknown;
    }
    if (std::strcmp(layoutAttr, "BSND") == 0) {
        return QueryLayoutKind::kBsnd;
    }
    if (std::strcmp(layoutAttr, "TND") == 0) {
        return QueryLayoutKind::kTnd;
    }
    return QueryLayoutKind::kUnknown;
}

void CopyLeadingDims(gert::Shape &dst, const gert::Shape &src, size_t rank, size_t lastKeep)
{
    dst.SetDimNum(rank);
    for (size_t axis = 0; axis < lastKeep; ++axis) {
        dst.SetDim(axis, src.GetDim(axis));
    }
}

void ShrinkLastDimByRope(gert::Shape &dst, const gert::Shape &src, size_t lastAxis, int64_t ropeDim)
{
    const int64_t raw = src.GetDim(lastAxis);
    if (raw == -1) {
        return;
    }
    dst.SetDim(lastAxis, raw - ropeDim);
}
} // namespace

ge::graphStatus InferShapeRqUniformKvQuantSparseFlashAttention(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("RqUniformKvQuantSparseFlashAttention", "InferShapeContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *queryShape = context->GetInputShape(kQueryInputSlot);
    if (queryShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    if (attentionOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char *layoutAttr = attrs->GetAttrPointer<char>(kLayoutQueryAttrSlot);
    if (layoutAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t ropeHeadDim = *attrs->GetAttrPointer<int64_t>(kRopeHeadDimAttrSlot);

    *attentionOutShape = *queryShape;
    const QueryLayoutKind layout = ParseQueryLayout(layoutAttr);
    switch (layout) {
        case QueryLayoutKind::kBsnd: {
            constexpr size_t kRank = 4;
            CopyLeadingDims(*attentionOutShape, *queryShape, kRank, /*lastKeep=*/3);
            ShrinkLastDimByRope(*attentionOutShape, *queryShape, /*lastAxis=*/3, ropeHeadDim);
            break;
        }
        case QueryLayoutKind::kTnd:
        default: {
            // TND (and unknown fallback): drop rope from the last dim of rank-3 query.
            constexpr size_t kRank = 3;
            CopyLeadingDims(*attentionOutShape, *queryShape, kRank, /*lastKeep=*/2);
            ShrinkLastDimByRope(*attentionOutShape, *queryShape, /*lastAxis=*/2, ropeHeadDim);
            break;
        }
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeRqUniformKvQuantSparseFlashAttention(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("RqUniformKvQuantSparseFlashAttention", "InferDataTypeContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    const auto queryDtype = context->GetInputDataType(kQueryInputSlot);
    context->SetOutputDataType(0, queryDtype);
    return ge::GRAPH_SUCCESS;
}

// V1 IMPL_OP_INFERSHAPE removed: V2 binds these symbols in
// rq_uniform_kv_quant_sparse_flash_attention_v2_infershape.cpp.
} // namespace ops
