/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_arch35_common.h
 * \brief arch35 FIA 公共前置
 */

#ifndef FIA_ARCH35_COMMON_H_
#define FIA_ARCH35_COMMON_H_

namespace optiling {};
using namespace AscendC;
using namespace optiling;
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef __DAV_C310_CUBE__ // CUBE 实现
#define FIA_REGBASE_COPY_TILING_DATA(tiling)                                                                     \
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = nullptr

#else // VECTOR 实现
#define FIA_REGBASE_COPY_TILING_DATA(tiling)                                                                     \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling);                        \
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = &tilingDataIn
#endif

#endif // FIA_ARCH35_COMMON_H_
