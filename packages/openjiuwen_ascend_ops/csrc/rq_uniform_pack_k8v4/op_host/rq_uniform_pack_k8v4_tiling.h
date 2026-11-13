#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RqUniformPackK8v4TilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
TILING_DATA_FIELD_DEF(uint32_t, nVec);
TILING_DATA_FIELD_DEF(uint32_t, vecPerCore);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, numReqs);
TILING_DATA_FIELD_DEF(uint32_t, keyStrideToken);
TILING_DATA_FIELD_DEF(uint32_t, keyStrideHead);
TILING_DATA_FIELD_DEF(uint32_t, valueStrideToken);
TILING_DATA_FIELD_DEF(uint32_t, valueStrideHead);
TILING_DATA_FIELD_DEF(uint64_t, keyStorageOffset);
TILING_DATA_FIELD_DEF(uint64_t, valueStorageOffset);
TILING_DATA_FIELD_DEF(uint32_t, dataCores);
TILING_DATA_FIELD_DEF(uint32_t, quantMode);
TILING_DATA_FIELD_DEF(uint32_t, packD);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(RqUniformPackK8v4LargeContigTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
TILING_DATA_FIELD_DEF(uint32_t, nVec);
TILING_DATA_FIELD_DEF(uint32_t, vecPerCore);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, numReqs);
TILING_DATA_FIELD_DEF(uint32_t, keyStrideToken);
TILING_DATA_FIELD_DEF(uint32_t, keyStrideHead);
TILING_DATA_FIELD_DEF(uint32_t, valueStrideToken);
TILING_DATA_FIELD_DEF(uint32_t, valueStrideHead);
TILING_DATA_FIELD_DEF(uint64_t, keyStorageOffset);
TILING_DATA_FIELD_DEF(uint64_t, valueStorageOffset);
TILING_DATA_FIELD_DEF(uint32_t, dataCores);
TILING_DATA_FIELD_DEF(uint32_t, quantMode);
TILING_DATA_FIELD_DEF(uint32_t, packD);
END_TILING_DATA_DEF;

// A5 pure-AIV: Cube unused, so omit TCubeTiling (same DEF on host/device;
// layout cannot diverge when both sides share this header). Leaner buffer.
BEGIN_TILING_DATA_DEF(RqUniformPackK8v4A5TilingData)
TILING_DATA_FIELD_DEF(uint32_t, nVec);
TILING_DATA_FIELD_DEF(uint32_t, vecPerCore);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, numReqs);
TILING_DATA_FIELD_DEF(uint32_t, keyStrideToken);
TILING_DATA_FIELD_DEF(uint32_t, keyStrideHead);
TILING_DATA_FIELD_DEF(uint32_t, valueStrideToken);
TILING_DATA_FIELD_DEF(uint32_t, valueStrideHead);
TILING_DATA_FIELD_DEF(uint64_t, keyStorageOffset);
TILING_DATA_FIELD_DEF(uint64_t, valueStorageOffset);
TILING_DATA_FIELD_DEF(uint32_t, dataCores);
TILING_DATA_FIELD_DEF(uint32_t, quantMode);
TILING_DATA_FIELD_DEF(uint32_t, packD);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RqUniformPackK8v4, RqUniformPackK8v4TilingData)
REGISTER_TILING_DATA_CLASS(RqUniformPackK8v4_1, RqUniformPackK8v4LargeContigTilingData)
REGISTER_TILING_DATA_CLASS(RqUniformPackK8v4_2, RqUniformPackK8v4A5TilingData)

}  // namespace optiling
