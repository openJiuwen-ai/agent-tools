/*!
 * \file rq_uniform_kv_quant_sparse_flash_attention_v2.cpp
 * \brief Device entry for RQ-uniform KV-quant sparse flash attention V2.
 */

#include "kernel_operator.h"
#include "rq_uniform_kv_quant_sparse_flash_attention_v2_template_tiling_key.h"
#include "rq_uniform_kv_quant_sparse_flash_attention_kernel_mla.h"

using namespace AscendC;

namespace {
struct RqSfaLaunchBuffers {
    __gm__ uint8_t *query;
    __gm__ uint8_t *key;
    __gm__ uint8_t *value;
    __gm__ uint8_t *sparseIndices;
    __gm__ uint8_t *keyScale;
    __gm__ uint8_t *valueScale;
    __gm__ uint8_t *blockTable;
    __gm__ uint8_t *actualSeqLengthsQuery;
    __gm__ uint8_t *actualSeqLengthsKV;
    __gm__ uint8_t *attentionOut;
    __gm__ uint8_t *userWorkspace;
    __gm__ uint8_t *tiling;
};

template <typename TilingStruct, typename OpClass>
__aicore__ inline void RunRqSfaMla(const RqSfaLaunchBuffers &buf, TPipe &pipeCtx)
{
    OpClass op;
    GET_TILING_DATA_WITH_STRUCT(TilingStruct, tilingHost, buf.tiling);
    const TilingStruct *__restrict tilingPtr = &tilingHost;
    op.Init(buf.query, buf.key, buf.value, buf.sparseIndices, buf.keyScale, buf.valueScale, buf.blockTable,
            buf.actualSeqLengthsQuery, buf.actualSeqLengthsKV, buf.attentionOut, buf.userWorkspace, tilingPtr,
            buf.tiling, &pipeCtx);
    op.Process();
}

template <int FlashDecode, int LayoutQ, int LayoutKv, int TemplateMode, typename QueryT, typename OutT>
__aicore__ inline void DispatchRqSfaDtype(const RqSfaLaunchBuffers &buf, TPipe &pipeCtx)
{
    using OpT = RqUniformKvQuantSparseFlashAttentionMla<
        QSFAType<QueryT, int8_t, OutT, FlashDecode, static_cast<QSFA_LAYOUT>(LayoutQ),
                 static_cast<QSFA_LAYOUT>(LayoutKv), TemplateMode, true>>;
    RunRqSfaMla<RqUniformKvQuantSparseFlashAttentionTilingDataMla, OpT>(buf, pipeCtx);
}
} // namespace

template <int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int IS_SPLIT_G,
          int IS_VEC_S2PHYADDR>
__global__ __aicore__ void rq_uniform_kv_quant_sparse_flash_attention_v2(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *keyScale, __gm__ uint8_t *valueScale, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipeCtx;
    RqSfaLaunchBuffers buf{query,
                           key,
                           value,
                           sparseIndices,
                           keyScale,
                           valueScale,
                           blocktable,
                           actualSeqLengthsQuery,
                           actualSeqLengthsKV,
                           attentionOut,
                           GetUserWorkspace(workspace),
                           tiling};
    (void)PAGE_ATTENTION;
    (void)IS_SPLIT_G;
    (void)IS_VEC_S2PHYADDR;

    constexpr bool kIsFp16Path =
        (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16);
    if constexpr (kIsFp16Path) {
        DispatchRqSfaDtype<FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, half, half>(buf, pipeCtx);
    } else {
        DispatchRqSfaDtype<FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, bfloat16_t, bfloat16_t>(buf, pipeCtx);
    }
}
