#include "register/op_def_registry.h"

namespace ops {
class BitResidualPackK8v4 : public OpDef {
public:
    explicit BitResidualPackK8v4(const char* name) : OpDef(name)
    {
        // Key/value can be large non-contiguous views. The kernel consumes
        // explicit strides and storage offsets, so keep CANN from inserting
        // another GM copy for them.
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("rotation_t")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("slot_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("query_start_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("value_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("n_vec").Int();
        this->Attr("vec_per_core").Int();
        this->Attr("num_heads").Int();
        this->Attr("block_size").Int();
        this->Attr("num_blocks").Int();
        this->Attr("num_reqs").Int();
        this->Attr("key_stride_token").Int();
        this->Attr("key_stride_head").Int();
        this->Attr("value_stride_token").Int();
        this->Attr("value_stride_head").Int();
        this->Attr("key_storage_offset").Int();
        this->Attr("value_storage_offset").Int();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");

        OpAICoreConfig aicoreConfigA2 = aicoreConfig;
        aicoreConfigA2.ExtendCfgInfo("jitCompile.flag", "static_false");

        this->AICore().AddConfig("ascend910_93", aicoreConfig);
        this->AICore().AddConfig("ascend910b", aicoreConfigA2);
    }
};

OP_ADD(BitResidualPackK8v4);
}  // namespace ops
