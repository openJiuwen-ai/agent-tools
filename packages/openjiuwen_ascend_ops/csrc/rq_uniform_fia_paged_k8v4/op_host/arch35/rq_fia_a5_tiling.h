/*
 * A5 antiquant port entry. quant_mode=2, ascend950 only.
 * Order is the official V4 order: Parse, Check, then SplitDequant.
 */

#pragma once

#include "exe_graph/runtime/tiling_context.h"

namespace optiling {

constexpr int64_t kRqFiaQuantModeA2 = 0;
constexpr int64_t kRqFiaQuantModeK8Reserved = 1;
constexpr int64_t kRqFiaQuantModeA5Port = 2;
constexpr int64_t kRqFiaA5HeadDim128 = 128;
constexpr int64_t kRqFiaA5HeadDim256 = 256;
// Local tiling keys. PA antiquant keeps sOuter=16 (gs-merge is off with PA).
// D=128 → sInner=512 (config 13). D=256 → sInner=256 (config 14).
constexpr uint64_t kRqFiaA5KeyD128NoFd = 2;
constexpr uint64_t kRqFiaA5KeyD128Fd = 3;
constexpr uint64_t kRqFiaA5KeyD256NoFd = 4;
constexpr uint64_t kRqFiaA5KeyD256Fd = 5;

ge::graphStatus RqFiaA5PortTiling(gert::TilingContext* context);

}  // namespace optiling
