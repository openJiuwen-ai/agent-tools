/*
 * RqUniform FIA Paged K8V4 MSD kernel entry.
 *
 * Us2 antiquant MSD (IncreFlashAttentionAttenSplitBbn2s2Us2).
 * See DESIGN.md and tq_msd_us2_kernel_body.h.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "tq_msd_us2_kernel_body.h"

// Dtype: query / rotation / out are half or bfloat16 (ORIG_DTYPE_QUERY),
// selected per build-injected dtype variant (same pattern as
// rq_uniform_fia_paged_k8v4). Pack tail meta (base/step/vmin/vstep) must
// match query dtype: reading bf16 meta as half produces garbage dequant.
#if defined(ORIG_DTYPE_QUERY) && (ORIG_DTYPE_QUERY == DT_BF16)
using TqMsdQT = bfloat16_t;
#elif defined(DTYPE_QUERY) && (DTYPE_QUERY == DT_BF16)
using TqMsdQT = bfloat16_t;
#else
using TqMsdQT = half;
#endif

extern "C" __global__ __aicore__ void rq_uniform_fia_paged_k8v4_msd(
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
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if BR_MSD_US2_PACK_GATHER
    if (TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING)) {
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING,
                         KERNEL_TYPE_MIX_AIC_1_1);
        BR_MSD_US2_RUN_PAGED(TqMsdQT, true, false);
    } else if (TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING)) {
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING,
                         KERNEL_TYPE_MIX_AIC_1_2);
        BR_MSD_US2_RUN_PAGED(TqMsdQT, true, false);
    } else if (TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING)) {
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING,
                         KERNEL_TYPE_MIX_AIC_1_2);
        BR_MSD_US2_RUN_PAGED(TqMsdQT, true, true);
    }
#else
    if (TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING)) {
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V1_TILING,
                         KERNEL_TYPE_MIX_AIC_1_1);
        BR_MSD_US2_RUN_PAGED(TqMsdQT, true, false);
    } else if (TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING)) {
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_C1V2_TILING,
                         KERNEL_TYPE_MIX_AIC_1_2);
        BR_MSD_US2_RUN_PAGED(TqMsdQT, true, false);
    } else if (TILING_KEY_IS(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING)) {
        KERNEL_TASK_TYPE(QF16_KVINT8_OUTF16_ANTIPERCHANNEL_PAGEDCACHE_FLASHDECODING_C1V2_TILING,
                         KERNEL_TYPE_MIX_AIC_1_2);
        BR_MSD_US2_RUN_PAGED(TqMsdQT, true, true);
    }
#endif
}
