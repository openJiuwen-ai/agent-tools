/*
 * A5 entry for the vendored ops-transformer Antiquant skeleton.
 *
 * Kernel, cube, vec, processor, and the common headers they include all live
 * in this arch35 directory. FullQuant and the A2 vendored/arch32 kernel are
 * not on this path.
 *
 * Parsed only for __NPU_ARCH__ == 3510. Keys 2/3 instantiate the stock
 * fp16/int8 antiquant kernel. The processor formula is unchanged.
 */

#pragma once

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "fia_arch35_antiquant.h"
#endif
