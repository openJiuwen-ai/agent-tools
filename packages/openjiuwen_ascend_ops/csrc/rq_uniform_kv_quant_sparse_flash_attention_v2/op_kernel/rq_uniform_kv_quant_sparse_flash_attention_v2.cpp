/*!
 * \file rq_uniform_kv_quant_sparse_flash_attention_v2.cpp
 * \brief
 */

/**
 * rq_uniform_kv_quant_sparse_flash_attention_v2
 *
 * Kernel sources live in this directory. QSFAType dequantV2=true selects
 * DequantSfaRowsV2 / CopyOutSfaV2. The legacy operator is not modified.
 */
#include "kernel_operator.h"
#include "rq_uniform_kv_quant_sparse_flash_attention_v2_template_tiling_key.h"
#include "rq_uniform_kv_quant_sparse_flash_attention_kernel_mla.h"

using namespace AscendC;

#define QSFA_OP_IMPL(templateClass, tilingdataClass, ...)                                          \
    do {                                                                                           \
        templateClass<QSFAType<__VA_ARGS__>> op;                                                   \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                      \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;                           \
        op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,                \
                actualSeqLengthsQuery, actualSeqLengthsKV, attentionOut, user, tiling_data, tiling,\
                &tPipe);                                                                           \
        op.Process();                                                                              \
    } while (0)

template <int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE,
          int IS_SPLIT_G, int IS_VEC_S2PHYADDR>
__global__ __aicore__ void rq_uniform_kv_quant_sparse_flash_attention_v2(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *keyScale, __gm__ uint8_t *valueScale, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    (void)PAGE_ATTENTION;
    (void)IS_SPLIT_G;
    (void)IS_VEC_S2PHYADDR;
    if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_INT8 &&
                  ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        QSFA_OP_IMPL(RqUniformKvQuantSparseFlashAttentionMla,
                     RqUniformKvQuantSparseFlashAttentionTilingDataMla, half, int8_t, half, FLASH_DECODE,
                     static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T), TEMPLATE_MODE,
                     true);
    } else {
        QSFA_OP_IMPL(RqUniformKvQuantSparseFlashAttentionMla,
                     RqUniformKvQuantSparseFlashAttentionTilingDataMla, bfloat16_t, int8_t, bfloat16_t,
                     FLASH_DECODE, static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
                     TEMPLATE_MODE, true);
    }
}
