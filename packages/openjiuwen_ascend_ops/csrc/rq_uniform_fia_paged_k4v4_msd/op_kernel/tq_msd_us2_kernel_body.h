/*
 * Us2 kernel helpers (included from rq_uniform_fia_paged_k4v4_msd.cpp).
 */

#pragma once

#include "tq_msd_us2_invoke.h"
#include "tq_msd_us2_pack_gather.h"
#include "vendored/us2/ifa_public_define.h"
#include "vendored/us2/incre_flash_attention_split_Bbn2s2_Us2.h"
#include "vendored/us2/incre_flash_attention_tilingkey.h"

using namespace AscendC;

#define BR_MSD_US2_RUN_PAGED(QT, PAGE, FD)                                                             \
    do {                                                                                           \
        __gm__ uint8_t *pseShift = nullptr;                                                       \
        __gm__ uint8_t *kvPaddingSize = nullptr;                                                   \
        __gm__ uint8_t *softmaxLse = nullptr;                                                      \
        __gm__ uint8_t *deqScale1 = nullptr;                                                       \
        __gm__ uint8_t *quantScale1 = nullptr;                                                     \
        __gm__ uint8_t *deqScale2 = nullptr;                                                       \
        __gm__ uint8_t *quantScale2 = nullptr;                                                     \
        __gm__ uint8_t *quantOffset2 = nullptr;                                                    \
        __gm__ uint8_t *antiquantScale = nullptr;                                                  \
        __gm__ uint8_t *antiquantOffset = nullptr;                                                 \
        __gm__ uint8_t *keyAntiquantScale = rotation_key;                                        \
        __gm__ uint8_t *keyAntiquantOffset = nullptr;                                              \
        __gm__ uint8_t *valueAntiquantScale = rotation_value;                                       \
        __gm__ uint8_t *valueAntiquantOffset = nullptr;                                            \
        TPipe tPipe;                                                                               \
        BR_MSD_US2_INVOKE_IFA(IncreFlashAttentionAttenSplitBbn2s2Us2, QT, int4b_t, QT, QT,          \
                              PAGE, FD, LAYOUT::BNSD BR_MSD_US2_ANTIQUANT_MODE);                    \
        (void)actual_seq_len_q;                                                                    \
        (void)tiling;                                                                              \
    } while (0)
