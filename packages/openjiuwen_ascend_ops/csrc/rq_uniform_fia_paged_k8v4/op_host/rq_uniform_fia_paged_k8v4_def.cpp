/*!
 * \file rq_uniform_fia_paged_k8v4_def.cpp
 * \brief
 */

/*
 * RqUniform FIA Paged K8V4 OpDef (standalone op: RqUniformFiaPagedK8v4).
 *
 * key/value_cache : uint8. quant_mode=0 is the A2 pack layout.
 *   quant_mode=2 is int8 BNBD, last dim = block_size * D, no scale input.
 * query / attention_out : fp16 or bf16. quant_mode=2 is fp16 only.
 * rotation_key / rotation_value : required [D, D] when quant_mode=0.
 *   quant_mode=2 ignores them (pass empty).
 * quant_mode: 0 = A2, 1 = reserved K8 fp8+U4 (rejected), 2 = A5 antiquant port.
 */

#include "register/op_def_registry.h"

namespace ops {

class RqUniformFiaPagedK8v4 : public OpDef {
public:
    explicit RqUniformFiaPagedK8v4(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_len_q")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(REQUIRED)
            .AutoContiguous();
        this->Input("actual_seq_len_kv")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(REQUIRED)
            .AutoContiguous();
        this->Input("atten_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rotation_key")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rotation_value")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("num_heads").Int();
        this->Attr("num_kv_heads").Int();
        this->Attr("head_size").Int();
        this->Attr("block_size").Int();
        this->Attr("scale_value").Float();
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(2147483647);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(2147483647);
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_mode").AttrType(REQUIRED).Int(0);

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
#ifdef VLLM_ASCEND_950_SOC_CONFIG
        this->AICore().AddConfig(VLLM_ASCEND_950_SOC_CONFIG, aicoreConfig);
#endif
    }
};

OP_ADD(RqUniformFiaPagedK8v4);

}  // namespace ops
