/*
 * RqUniform FIA Paged K8V4 kernel entry for vllm-ascend.
 *
 * IO: query, key_cache, value_cache (uint8 pack), block_table,
 *     actual_seq_len_q/kv, atten_mask (optional),
 *     rotation_key, rotation_value, attention_out
 *
 * Tiling keys:
 *   0 = TND+PA, no FlashDecode
 *   1 = TND+PA, FlashDecode
 *
 * Dtype: query / rotation / out are half or bfloat16 (ORIG_DTYPE_QUERY).
 * Pack metadata (base/step/vmin/vstep) must match query dtype.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "tq_fia_device_tiling.h"

#include "vendored/arch32/fia_kernel_turboquant_p0.h"
#include "vendored/arch32/fia_block_cube_turboquant_p0.h"
#include "vendored/arch32/fia_block_vec_turboquant_p0.h"
#include "vendored/arch32/fia_block_vec_flashdecode.h"

using namespace AscendC;
using namespace AttentionCommon;

#if defined(ORIG_DTYPE_QUERY) && (ORIG_DTYPE_QUERY == DT_BF16)
using TqFiaQT = bfloat16_t;
#elif defined(DTYPE_QUERY) && (DTYPE_QUERY == DT_BF16)
using TqFiaQT = bfloat16_t;
#else
using TqFiaQT = half;
#endif

#define BR_FIA_COPY_TILING(tiling)                                                                 \
    GET_TILING_DATA(tiling_data_gen, tiling);                                                      \
    const FusedInferAttentionScoreTilingData *__restrict tiling_data =                             \
        reinterpret_cast<const FusedInferAttentionScoreTilingData *>(&tiling_data_gen)

// Q/OUT/ORIGIN = TqFiaQT; KV typed half for WS path (uint8 cache is dequanted to half WS).
#define INVOKE_BR_FIA(FLASH_DECODE)                                                                \
    do {                                                                                           \
        using FIAT = FIAType<TqFiaQT, half, TqFiaQT, TqFiaQT, true, FLASH_DECODE, FIA_LAYOUT::TND, \
                             FIA_TQ_MSE_8BIT_MODE, false, FIA_LAYOUT::BSH, false>;                 \
        using CubeT = FiaBlockCubeTurboQuantP0<FIAT>;                                               \
        using VecT = FiaBlockVecTurboQuantP0<FIAT>;                                                 \
        using FdT = FiaBlockVecFlashDecode<FIAT>;                                                   \
        FiaKernelTurboQuantP0<FIAT, CubeT, VecT, FdT> op;                                          \
        BR_FIA_COPY_TILING(tiling);                                                                \
        op.Init(query, key_cache, value_cache, /*pse*/ nullptr, atten_mask, actual_seq_len_q,      \
                actual_seq_len_kv, /*deqScale1*/ nullptr, /*quantScale1*/ nullptr,                 \
                /*deqScale2*/ nullptr, /*quantScale2*/ nullptr, /*quantOffset2*/ nullptr,          \
                rotation_key, /*antiquantOffset*/ nullptr, block_table, /*qPad*/ nullptr,        \
                /*kvPad*/ nullptr, /*keyAntiquantScale*/ nullptr, /*keyAntiquantOffset*/ nullptr,  \
                rotation_value, /*valueAntiquantOffset*/ nullptr, /*keySharedPrefix*/ nullptr,    \
                /*valueSharedPrefix*/ nullptr, /*actualSharedPrefixLen*/ nullptr,                  \
                /*queryRope*/ nullptr, /*keyRope*/ nullptr, /*keyRopeAntiquantScale*/ nullptr,    \
                /*learnableSink*/ nullptr, attention_out, /*softmaxLse*/ nullptr, workspace,       \
                tiling_data, tiling, &tPipe);                                                      \
        op.Process();                                                                              \
    } while (0)

extern "C" __global__ __aicore__ void rq_uniform_fia_paged_k8v4(
    GM_ADDR query,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR block_table,
    GM_ADDR actual_seq_len_q,
    GM_ADDR actual_seq_len_kv,
    GM_ADDR atten_mask,
    GM_ADDR rotation_key,
    GM_ADDR rotation_value,
    GM_ADDR attention_out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    AscendC::SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }

    TPipe tPipe;
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        INVOKE_BR_FIA(false);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        INVOKE_BR_FIA(true);
    }
}
