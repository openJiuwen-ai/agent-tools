/*
 * Byte offset of the fp16 ones appended after FlashAttentionScoreSimplifiedTilingData.
 *
 * quant_mode=2 has no scale input. Host writes 16 fp16 1s here (32-byte tail).
 * Per-tensor antiquant reads element 0 for key and element 1 for value.
 * Host and device must agree on sizeof(FlashAttentionScoreSimplifiedTilingData).
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "flash_attention_score_tiling_regbase_arch35.h"

constexpr uint32_t kRqFiaA5ScaleTailBytes = 32;
// 1112-byte tiling struct, rounded up to 32B. Constexpr so device kernels can
// use the value (bisheng treats plain inline helpers as __host__-only).
constexpr size_t kRqFiaA5ScaleByteOffset =
    (sizeof(optiling::FlashAttentionScoreSimplifiedTilingData) + static_cast<size_t>(31)) &
    ~static_cast<size_t>(31);

inline size_t RqFiaA5ScaleByteOffset()
{
    return kRqFiaA5ScaleByteOffset;
}
