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
constexpr size_t QUERY_INPUT_INDEX = 0;
// Attributes Index
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 4;
constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 12;
// Dim Index
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;

ge::graphStatus InferShapeRqUniformKvQuantSparseFlashAttention(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("RqUniformKvQuantSparseFlashAttention", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(LAYOUT_QUERY_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    const int64_t ropeHeadDim = *attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);

    *attentionOutShape = *queryShape;
    if (inputLayoutQueryPtrStr == "BSND") {
        attentionOutShape->SetDimNum(DIM_NUM_4);
        attentionOutShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
        attentionOutShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_1));
        attentionOutShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_2));
        if (queryShape->GetDim(DIM_INDEX_3) != -1) {
            attentionOutShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_3) - ropeHeadDim);
        }
    } else { // TND
        attentionOutShape->SetDimNum(DIM_NUM_3);
        attentionOutShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
        attentionOutShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_1));
        if (queryShape->GetDim(DIM_INDEX_2) != -1) {
            attentionOutShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_2) - ropeHeadDim);
        }
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeRqUniformKvQuantSparseFlashAttention(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("RqUniformKvQuantSparseFlashAttention", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

// V1 IMPL_OP_INFERSHAPE removed: V2 binds these symbols in
// rq_uniform_kv_quant_sparse_flash_attention_v2_infershape.cpp.
} // namespace ops
