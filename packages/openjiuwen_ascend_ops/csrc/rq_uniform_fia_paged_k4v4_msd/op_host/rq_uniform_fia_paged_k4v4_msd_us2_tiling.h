#ifndef RQ_UNIFORM_FIA_PAGED_K4V4_MSD_US2_TILING_H
#define RQ_UNIFORM_FIA_PAGED_K4V4_MSD_US2_TILING_H

#include "vendored/us2/incre_flash_attention_tiling.h"

namespace optiling {

// Single registration — per-key variants share IncreFlashAttentionTilingDataV2 layout.
REGISTER_TILING_DATA_CLASS(RqUniformFiaPagedK4v4Msd, IncreFlashAttentionTilingDataV2)

}  // namespace optiling

#endif
