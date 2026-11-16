/*
 * Checks only the quant_mode=2 validation path. Does not modify FiaTilingInfo.
 */

#pragma once

#include "rq_fia_a5_info_parser.h"

namespace optiling {

ge::graphStatus RqFiaA5Check(const RqFiaA5Parsed& parsed);

}  // namespace optiling
