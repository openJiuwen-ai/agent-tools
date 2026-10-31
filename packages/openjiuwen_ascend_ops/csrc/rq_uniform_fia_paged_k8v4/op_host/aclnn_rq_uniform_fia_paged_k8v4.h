#ifndef ACLNN_RQ_UNIFORM_FIA_PAGED_K8V4_H_
#define ACLNN_RQ_UNIFORM_FIA_PAGED_K8V4_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * RqUniform FIA Paged K8V4 fused attention (vllm-ascend custom op).
 *
 * key/value_cache : uint8 pack layout [num_blocks, num_kv_heads, packed_bytes]
 * rotation_key/value : fp16 [D, D]
 * atten_mask      : optional int8/fp16 mask (nullptr when sparse_mode=0)
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformFiaPagedK8v4GetWorkspaceSize(
    const aclTensor* query,
    const aclTensor* keyCache,
    const aclTensor* valueCache,
    const aclTensor* blockTable,
    const aclIntArray* actualSeqLenQ,
    const aclIntArray* actualSeqLenKv,
    const aclTensor* attenMask,
    const aclTensor* rotationKey,
    const aclTensor* rotationValue,
    int64_t numHeads,
    int64_t numKvHeads,
    int64_t headSize,
    int64_t blockSize,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t sparseMode,
    const aclTensor* attentionOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformFiaPagedK8v4(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
