/*
 * Us2 IFA invoke macros (subset of incre_flash_attention_arch32.h).
 */
#pragma once

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#define BR_MSD_US2_NEED_CUBE_TILING (true)
#define BR_MSD_US2_NOT_NEED_CUBE_TILING (false)

#ifndef BR_MSD_US2_PACK_GATHER
#define BR_MSD_US2_PACK_GATHER 1
#endif
#if BR_MSD_US2_PACK_GATHER
// IFAType ANTIQUANT_MODE=1 (PER_TOKEN). Meta is pack-tensor tails (Scheme A).
#define BR_MSD_US2_ANTIQUANT_MODE , 1
#else
#define BR_MSD_US2_ANTIQUANT_MODE
#endif

#ifdef __DAV_C220_CUBE__
#define BR_MSD_US2_COPY_TILING_DATA(tiling, need_cube)                                                 \
    if constexpr (!need_cube) {                                                                        \
        return;                                                                                      \
    }                                                                                                  \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase.bmm1TilingData,                 \
                           bmm1TilingDataVar, tiling);                                                 \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase.bmm2TilingData,                 \
                           bmm2TilingDataVar, tiling);                                                 \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, bmmPiTilingData,                           \
                           bmmPiTilingDataVar, tiling);                                                \
    const IncreFlashAttentionTilingData *__restrict tiling_data = nullptr;                             \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;                                     \
    const TCubeTiling *__restrict bmmPitiling = &bmmPiTilingDataVar
#else
#define BR_MSD_US2_COPY_TILING_DATA(tiling, need_cube)                                                 \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase, tiling_data_in, tiling);       \
    const IncreFlashAttentionTilingData *__restrict tiling_data = &tiling_data_in;                     \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr;                                                \
    const TCubeTiling *__restrict bmmPitiling = nullptr
#endif

#define BR_MSD_US2_INVOKE_IFA(templateClass, ...)                                                      \
    do {                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                        \
        BR_MSD_US2_COPY_TILING_DATA(tiling, BR_MSD_US2_NEED_CUBE_TILING);                              \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling,       \
                          op.mmPi, bmmPitiling);                                                     \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                          \
        op.Init(query, key_cache, value_cache, pseShift, atten_mask, actual_seq_len_kv, block_table, \
                kvPaddingSize, attention_out, softmaxLse, user, tiling_data, tiling, &tPipe);      \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale,     \
                     antiquantOffset, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale,      \
                     valueAntiquantOffset, user);                                                      \
        op.Process();                                                                                  \
    } while (0)
