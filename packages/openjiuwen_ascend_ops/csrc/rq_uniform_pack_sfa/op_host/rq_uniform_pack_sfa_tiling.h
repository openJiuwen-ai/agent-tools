#pragma once

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RqUniformPackSfaTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, dataCores);
TILING_DATA_FIELD_DEF(uint32_t, dtype);
TILING_DATA_FIELD_DEF(uint32_t, tileRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RqUniformPackSfa, RqUniformPackSfaTilingData)

}  // namespace optiling
