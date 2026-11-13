// Fused rq-uniform pack for KV cache (fp16/bf16).
//
// tilingKey 0/1: A2/A3 MIX_AIC_1_2 — Haar rotate + Scheme A Key + U4 Value
// tilingKey 2:   A5 AIV_ONLY — TQ_K8_FP8 Key (no rotate) + U4 Value

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/resource.hpp"
#include "rq_uniform_pack_k8v4_common.h"
#include "rq_uniform_pack_k8v4_aiv.h"
#include "rq_uniform_pack_k8v4_aic.h"
// fp8 Cast is dav_3510 only. ascend910b (NPU_ARCH 2201) compiles this same
// translation unit and has no __fp8e4m3 CastIntrinsicsImpl.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/pack_k8v4_aiv.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void rq_uniform_pack_k8v4(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR rotation_t,
    GM_ADDR slot_mapping,
    GM_ADDR query_start_loc,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR workspace,
    GM_ADDR tiling) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(
            RqUniformPackK8v4A5TilingData, tilingData, tiling);
        auto* wsPtr = reinterpret_cast<__gm__ uint8_t*>(workspace);
        if (wsPtr != nullptr) {
            AscendC::SetSysWorkspace(wsPtr);
        }
        const uint32_t numTokens =
            tilingData.numHeads == 0 ? 0 : tilingData.nVec / tilingData.numHeads;
        const uint32_t packD =
            tilingData.packD != 0 ? tilingData.packD : tilingData.keyStrideHead;
#if (defined(ORIG_DTYPE_KEY) && (ORIG_DTYPE_KEY == DT_BF16)) || \
    (defined(DTYPE_KEY) && (DTYPE_KEY == DT_BF16))
        rq_uniform::arch35::RunPackK8v4A5<bfloat16_t>(
#else
        rq_uniform::arch35::RunPackK8v4A5<half>(
#endif
            key,
            value,
            slot_mapping,
            key_cache,
            value_cache,
            workspace,
            numTokens,
            tilingData.numHeads,
            packD,
            tilingData.blockSize,
            tilingData.numBlocks,
            tilingData.dataCores,
            tilingData.keyStrideToken,
            tilingData.keyStrideHead,
            tilingData.valueStrideToken,
            tilingData.valueStrideHead);
        return;
    }
#endif

    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    } else {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);

    auto* rotationPtr = reinterpret_cast<__gm__ rq_uniform::TqDataT*>(rotation_t);
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
    rq_uniform::RqUniformPackK8v4Context<rq_uniform::TqDataT> context(
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
        const uint32_t manualGroupId = rq_uniform::TqManualRawBlockIdx();
        if (manualGroupId >= context.dataCores_) {
            return;
        }
        const uint32_t aivSlice =
            AscendC::GetSubBlockIdx() % rq_uniform::TQ_AIV_SUB_BLOCKS;
        rq_uniform::RqUniformPackK8v4VectorService<
            rq_uniform::TqDataT> service(context);
        service.Process(manualGroupId, aivSlice);
        return;
    }
    const uint32_t manualGroupId = AscendC::GetBlockIdx();
    if (manualGroupId >= context.dataCores_) {
        return;
    }
    rq_uniform::RqUniformPackK8v4CubeService<
        rq_uniform::TqDataT> service(context);
    service.Process(manualGroupId);
}
