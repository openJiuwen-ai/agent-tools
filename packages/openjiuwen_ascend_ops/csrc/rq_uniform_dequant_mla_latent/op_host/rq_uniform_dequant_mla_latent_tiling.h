#pragma once

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RqUniformDequantMlaLatentTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
TILING_DATA_FIELD_DEF(uint32_t, latentDim);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, numReqs);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocks);
TILING_DATA_FIELD_DEF(uint32_t, dataCores);
TILING_DATA_FIELD_DEF(uint32_t, dtype);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RqUniformDequantMlaLatent, RqUniformDequantMlaLatentTilingData)

}  // namespace optiling
