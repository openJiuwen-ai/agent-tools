/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Fused bit-residual pack for KV cache (fp16/bf16).
//
// Reference (Python):
//   y = x @ R^T                           # batched [M, D] @ [D, D]
//   key:   sign reversal + 7-bit uniform quantization
//   value: 4-bit uniform quantization, processed four rows at a time
//
// Per AICore: rotate all assigned rows with one matmul [M,128]@[128,128],
// not M separate M=1 matmuls (avoids Cube padding waste).

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/resource.hpp"
#include "bit_residual_pack_k8v4_common.h"
#include "bit_residual_pack_k8v4_aiv.h"
#include "bit_residual_pack_k8v4_aic.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void bit_residual_pack_k8v4(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR rotation_t,
    GM_ADDR slot_mapping,
    GM_ADDR query_start_loc,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);

    auto* rotationPtr = reinterpret_cast<__gm__ bit_residual::TqDataT*>(rotation_t);
    auto* slotMappingPtr = reinterpret_cast<__gm__ int32_t*>(slot_mapping);
    auto* queryStartLocPtr = reinterpret_cast<__gm__ int32_t*>(query_start_loc);
    auto* keyCachePtr = reinterpret_cast<__gm__ uint8_t*>(key_cache);
    auto* valueCachePtr = reinterpret_cast<__gm__ uint8_t*>(value_cache);

    auto* wsPtr = reinterpret_cast<__gm__ uint8_t*>(workspace);
    AscendC::SetSysWorkspace(wsPtr);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    AscendC::TPipe pipe;
    bit_residual::BitResidualPackK8v4Context<bit_residual::TqDataT> context(
        wsPtr,
        tilingData.nVec,
        tilingData.vecPerCore,
        tilingData.numHeads,
        tilingData.blockSize,
        tilingData.numBlocks,
        tilingData.dataCores,
        tilingData.numReqs,
        tilingData.keyStrideToken,
        tilingData.keyStrideHead,
        tilingData.valueStrideToken,
        tilingData.valueStrideHead,
        tilingData.keyStorageOffset,
        tilingData.valueStorageOffset);
    context.Init(
        key,
        value,
        rotationPtr,
        slotMappingPtr,
        queryStartLocPtr,
        keyCachePtr,
        valueCachePtr);
    if (!context.CanUseManualKey1()) {
        return;
    }
    if ASCEND_IS_AIV {
        const uint32_t manualGroupId = bit_residual::TqManualRawBlockIdx();
        if (manualGroupId >= context.dataCores_) {
            return;
        }
        const uint32_t aivSlice =
            AscendC::GetSubBlockIdx() % bit_residual::TQ_AIV_SUB_BLOCKS;
        bit_residual::BitResidualPackK8v4VectorService<
            bit_residual::TqDataT> service(context);
        service.Process(manualGroupId, aivSlice);
        return;
    }
    const uint32_t manualGroupId = AscendC::GetBlockIdx();
    if (manualGroupId >= context.dataCores_) {
        return;
    }
    bit_residual::BitResidualPackK8v4CubeService<
        bit_residual::TqDataT> service(context);
    service.Process(manualGroupId);
}
