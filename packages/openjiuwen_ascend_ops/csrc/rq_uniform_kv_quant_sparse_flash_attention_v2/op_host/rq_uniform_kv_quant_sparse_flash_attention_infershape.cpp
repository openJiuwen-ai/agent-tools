/*!
 * \file rq_uniform_kv_quant_sparse_flash_attention_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
// Inputs Index
constexpr size_t kInQueryIdx = 0;
// Attributes Index
constexpr uint32_t kAttrLayoutQuery = 4;
constexpr uint32_t kAttrRopeHeadDim = 12;
// Dim Index
constexpr uint32_t kDim0 = 0;
constexpr uint32_t kDim1 = 1;
constexpr uint32_t kDim2 = 2;
constexpr uint32_t kDim3 = 3;
constexpr uint32_t kRank3 = 3;
constexpr uint32_t kRank4 = 4;

ge::graphStatus InferShapeRqUniformKvQuantSparseFlashAttention(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("RqUniformKvQuantSparseFlashAttention", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const gert::Shape *qShape = context->GetInputShape(kInQueryIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    gert::Shape *attnOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attnOutShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *layoutQueryAttr = attrs->GetAttrPointer<char>(kAttrLayoutQuery);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutQueryAttr);
    std::string layoutQueryStr = std::string(layoutQueryAttr);
    const int64_t ropeDim = *attrs->GetAttrPointer<int64_t>(kAttrRopeHeadDim);

    *attnOutShape = *qShape;
    if (layoutQueryStr == "BSND") {
        attnOutShape->SetDimNum(kRank4);
        attnOutShape->SetDim(kDim0, qShape->GetDim(kDim0));
        attnOutShape->SetDim(kDim1, qShape->GetDim(kDim1));
        attnOutShape->SetDim(kDim2, qShape->GetDim(kDim2));
        if (qShape->GetDim(kDim3) != -1) {
            attnOutShape->SetDim(kDim3, qShape->GetDim(kDim3) - ropeDim);
        }
    } else { // TND
        attnOutShape->SetDimNum(kRank3);
        attnOutShape->SetDim(kDim0, qShape->GetDim(kDim0));
        attnOutShape->SetDim(kDim1, qShape->GetDim(kDim1));
        if (qShape->GetDim(kDim2) != -1) {
            attnOutShape->SetDim(kDim2, qShape->GetDim(kDim2) - ropeDim);
        }
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeRqUniformKvQuantSparseFlashAttention(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("RqUniformKvQuantSparseFlashAttention", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const auto qDtype = context->GetInputDataType(kInQueryIdx);
    context->SetOutputDataType(0, qDtype);
    return ge::GRAPH_SUCCESS;
}

// V1 IMPL_OP_INFERSHAPE removed: V2 binds these symbols in
// rq_uniform_kv_quant_sparse_flash_attention_v2_infershape.cpp.
} // namespace ops
