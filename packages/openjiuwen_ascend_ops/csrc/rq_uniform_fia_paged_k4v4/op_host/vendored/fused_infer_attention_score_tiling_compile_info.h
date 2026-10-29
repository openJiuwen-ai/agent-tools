/*!
 * \file fused_infer_attention_score_tiling_compile_info.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_COMPILE_INFO_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_COMPILE_INFO_H
#include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.h"
#include "../../incre_flash_attention/op_host/incre_flash_attention_tiling.h"
#include "register/tilingdata_base.h"

namespace optiling {
struct FusedInferAttentionScoreCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    size_t defaultSysWorkspaceSize;
    platform_ascendc::SocVersion socShortName;
};
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_COMPILE_INFO_H