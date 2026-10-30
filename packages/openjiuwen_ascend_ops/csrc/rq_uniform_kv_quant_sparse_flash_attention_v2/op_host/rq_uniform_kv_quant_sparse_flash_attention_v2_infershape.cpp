#include <register/op_impl_registry.h>

namespace ops {
ge::graphStatus InferShapeRqUniformKvQuantSparseFlashAttention(gert::InferShapeContext *context);
ge::graphStatus InferDataTypeRqUniformKvQuantSparseFlashAttention(gert::InferDataTypeContext *context);

IMPL_OP_INFERSHAPE(RqUniformKvQuantSparseFlashAttentionV2)
    .InferShape(InferShapeRqUniformKvQuantSparseFlashAttention)
    .InferDataType(InferDataTypeRqUniformKvQuantSparseFlashAttention);
} // namespace ops
