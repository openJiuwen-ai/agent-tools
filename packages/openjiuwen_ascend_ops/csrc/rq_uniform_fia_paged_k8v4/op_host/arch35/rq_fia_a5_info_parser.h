/*
 * Fills the FiaTilingInfo fields SplitDequant and SetFATilingData read.
 * Does not copy the official FiaInfoParser (that one is bound to FIA tensor indices).
 */

#pragma once

#include <cstdint>
#include <string>

#include "exe_graph/runtime/tiling_context.h"
#include "fia_tiling_info.h"

namespace optiling {

struct RqFiaA5Parsed {
    FiaTilingInfo info;
    std::string layout = "BNSD";
    int64_t pseType = 9;
    bool hasMask = false;
    bool hasRotation = false;
};

ge::graphStatus RqFiaA5Parse(gert::TilingContext* context, RqFiaA5Parsed& parsed);

}  // namespace optiling
