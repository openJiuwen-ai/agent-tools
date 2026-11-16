/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_arch35_antiquant.h
 * \brief arch35 FIA 反量化路由
 */

#ifndef FIA_ARCH35_ANTIQUANT_H_
#define FIA_ARCH35_ANTIQUANT_H_

#include "fia_arch35_common.h"
#include "fia_arch35_template_tiling_key_enum.h"
#include "flash_attention_score_antiquant_kernel.h"

#define INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(templateClass, ...) \
    do { \
        if (query == nullptr) { \
            return; \
        } \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling); \
        const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = &tilingDataIn; \
        TPipe tPipe; \
        __gm__ uint8_t *user = GetUserWorkspace(workspace); \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockCubeAntiquant<__VA_ARGS__>, \
                                      BaseApi::FABlockCubeAntiquantDummy<__VA_ARGS__>>::type; \
        using VecBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockVecAntiquantDummy<__VA_ARGS__>, \
                                      BaseApi::FABlockVecAntiquant<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op; \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, \
                queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, \
                softmaxLse, attentionOut, user, tilingData, &tPipe, antiquantScale, antiquantOffset, \
                keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, quantScale2, \
                quantOffset2); \
        op.Process(); \
    } while (0)

template <uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode, bool hasAttenMask, bool hasRope,
          uint8_t KvLayoutType, bool isFd, bool emptyTensor, bool enableKVPrefix, bool enableS1OutSplit>
inline __aicore__ void fia_antiquant_regbase(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    /*
    获取Op可用WorkSpace空间
    **/
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    constexpr bool isPa = KvLayoutType != 0;

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_INT8 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, float16_t, int8_t, float, float16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_INT4 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, float16_t, int4b_t, float, float16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_HIFLOAT8 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, float16_t, hifloat8_t, float, float16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, float16_t, fp8_e4m3fn_t, float, float16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT4_E2M1 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, float16_t, fp4x2_e2m1_t, float, float16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_INT8 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, bfloat16_t, int8_t, float, bfloat16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_INT4 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, bfloat16_t, int4b_t, float, bfloat16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_HIFLOAT8 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, bfloat16_t, hifloat8_t, float, bfloat16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_FLOAT4_E2M1 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, bfloat16_t, fp4x2_e2m1_t, float, bfloat16_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_INT8 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, bfloat16_t, int8_t, float, int8_t, ImplModeEnum::AA_HIGH_PRECISION,
        inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_INT8 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, float16_t, int8_t, float, int8_t, ImplModeEnum::AA_HIGH_PRECISION,
        inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_HIFLOAT8 && ORIG_DTYPE_ATTENTION_OUT == DT_HIFLOAT8)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, bfloat16_t, hifloat8_t, float, hifloat8_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_HIFLOAT8 && ORIG_DTYPE_ATTENTION_OUT == DT_HIFLOAT8)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, float16_t, hifloat8_t, float, hifloat8_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E4M3FN)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, bfloat16_t, fp8_e4m3fn_t, float, fp8_e4m3fn_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && \
     ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E4M3FN)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    PARSE_PARAMS_AntiQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI(
        BaseApi::FlashAttentionScoreAntiquantKernel, float16_t, fp8_e4m3fn_t, float, fp8_e4m3fn_t,
        ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), static_cast<AntiquantTypeEnum>(quantMode), hasAttenMask, false, false, true,
        isPa, isFd, enableKVPrefix);
#endif
}

#endif // FIA_ARCH35_ANTIQUANT_H_
