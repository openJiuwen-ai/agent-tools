/*
 * RqUniform FIA Paged K8V4 kernel entry for vllm-ascend.
 *
 * IO: query, key_cache, value_cache (uint8 pack), block_table,
 *     actual_seq_len_q/kv, atten_mask (optional),
 *     rotation_key, rotation_value, attention_out
 *
 * Tiling keys:
 *   0 = TND+PA, no FlashDecode (A2)
 *   1 = TND+PA, FlashDecode (A2)
 *   2/3 = A5 antiquant D=128 (sOuter=16, sInner=512), FD off/on
 *   4/5 = A5 antiquant D=256 (sOuter=16, sInner=256), FD off/on
 *   Prefill (s1>1) and decode both use these templates; FD is decode-only.
 *
 * Dtype: query / rotation / out are half or bfloat16 (ORIG_DTYPE_QUERY).
 * Pack metadata (base/step/vmin/vstep) must match query dtype.
 *
 * A2 vendored/arch32 and A5 arch35 must not be included in the same TU:
 * ConstInfo / RunInfo / TASK_DEAL_MODE collide across those trees.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/rq_fia_antiquant_skeleton.h"
#include "arch35/rq_fia_a5_scale.h"
#ifdef false
#undef false
#endif
#ifdef true
#undef true
#endif

#ifndef PARSE_PARAMS_AntiQuant
#define PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, ...)                                  \
    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]); \
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]); \
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);     \
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);     \
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);         \
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv)
#endif

// Stock fia_antiquant_regbase is empty unless ORIG_DTYPE_KEY is DT_INT8.
// This op's key cache is uint8 storage of int8 values, so instantiate that body here.
// config 13 = sOuter16/sInner512/D128. config 14 = sOuter16/sInner256/D256.
template <bool isFd, uint16_t config>
__aicore__ inline void InvokeA5AntiquantPort(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
    __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* blocktable, __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    __gm__ uint8_t* pseShift = nullptr;
    __gm__ uint8_t* attenMask = nullptr;
    __gm__ uint8_t* antiquantScale = tiling + kRqFiaA5ScaleByteOffset;
    __gm__ uint8_t* antiquantOffset = nullptr;
    __gm__ uint8_t* queryPaddingSize = nullptr;
    __gm__ uint8_t* kvPaddingSize = nullptr;
    __gm__ uint8_t* keySharedPrefix = nullptr;
    __gm__ uint8_t* valueSharedPrefix = nullptr;
    __gm__ uint8_t* actualSharedPrefixLen = nullptr;
    __gm__ uint8_t* softmaxLse = nullptr;
    __gm__ uint8_t* keyAntiquantScale = nullptr;
    __gm__ uint8_t* keyAntiquantOffset = nullptr;
    __gm__ uint8_t* valueAntiquantScale = nullptr;
    __gm__ uint8_t* valueAntiquantOffset = nullptr;
    __gm__ uint8_t* quantScale2 = nullptr;
    __gm__ uint8_t* quantOffset2 = nullptr;

    constexpr uint8_t inOutLayoutType = 0;
    constexpr uint8_t pseMode = 9;
    constexpr uint8_t quantMode = 0;
    constexpr bool hasAttenMask = false;
    constexpr bool hasRope = false;
    constexpr bool isPa = true;
    constexpr bool emptyTensor = false;
    constexpr bool enableKVPrefix = false;
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd,
                           emptyTensor, enableKVPrefix);
    (void)outputLayoutType;
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, float16_t, int8_t, float, float16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
        dVTemplateType, static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask,
        false, false, true, isPa, isFd, enableKVPrefix);
}

#else

#include "tq_fia_device_tiling.h"
#include "vendored/arch32/fia_kernel_turboquant_p0.h"
#include "vendored/arch32/fia_block_cube_turboquant_p0.h"
#include "vendored/arch32/fia_block_vec_turboquant_p0.h"
#include "vendored/arch32/fia_block_vec_flashdecode.h"
#ifdef false
#undef false
#endif
#ifdef true
#undef true
#endif

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

#endif

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

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        InvokeA5AntiquantPort<false, 13>(query, key_cache, value_cache, actual_seq_len_q, actual_seq_len_kv,
                                          block_table, attention_out, workspace, tiling);
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        InvokeA5AntiquantPort<true, 13>(query, key_cache, value_cache, actual_seq_len_q, actual_seq_len_kv,
                                         block_table, attention_out, workspace, tiling);
    } else if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
        InvokeA5AntiquantPort<false, 14>(query, key_cache, value_cache, actual_seq_len_q, actual_seq_len_kv,
                                          block_table, attention_out, workspace, tiling);
    } else if (TILING_KEY_IS(5)) {
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
        InvokeA5AntiquantPort<true, 14>(query, key_cache, value_cache, actual_seq_len_q, actual_seq_len_kv,
                                         block_table, attention_out, workspace, tiling);
    }
#else
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        TPipe tPipe;
        INVOKE_BR_FIA(false);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        TPipe tPipe;
        INVOKE_BR_FIA(true);
    }
#endif
}
