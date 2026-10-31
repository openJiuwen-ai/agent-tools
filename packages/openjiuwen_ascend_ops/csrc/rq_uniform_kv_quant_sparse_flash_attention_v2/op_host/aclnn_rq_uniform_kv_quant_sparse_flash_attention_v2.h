#ifndef ACLNN_RQ_UNIFORM_KV_QUANT_SPARSE_FLASH_ATTENTION_V2_H_
#define ACLNN_RQ_UNIFORM_KV_QUANT_SPARSE_FLASH_ATTENTION_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformKvQuantSparseFlashAttentionV2GetWorkspaceSize(
    const aclTensor* query,
    const aclTensor* key,
    const aclTensor* value,
    const aclTensor* sparseIndices,
    const aclTensor* keyDequantScale,
    const aclTensor* valueDequantScale,
    const aclTensor* blockTable,
    const aclTensor* actualSeqLengthsQuery,
    const aclTensor* actualSeqLengthsKV,
    double scaleValue,
    int64_t keyQuantMode,
    int64_t valueQuantMode,
    int64_t sparseBlockSize,
    const char* layoutQuery,
    const char* layoutKv,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t attentionMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    int64_t ropeHeadDim,
    const aclTensor* attentionOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformKvQuantSparseFlashAttentionV2(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
