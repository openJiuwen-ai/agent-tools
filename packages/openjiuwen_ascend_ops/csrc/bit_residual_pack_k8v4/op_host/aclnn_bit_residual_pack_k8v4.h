#ifndef ACLNN_BIT_RESIDUAL_PACK_K8V4_H_
#define ACLNN_BIT_RESIDUAL_PACK_K8V4_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnBitResidualPackK8v4GetWorkspaceSize(
    const aclTensor* key,
    const aclTensor* value,
    const aclTensor* rotationT,
    const aclTensor* slotMapping,
    const aclTensor* queryStartLoc,
    int64_t nVec,
    int64_t vecPerCore,
    int64_t numHeads,
    int64_t blockSize,
    int64_t numBlocks,
    int64_t numReqs,
    int64_t keyStrideToken,
    int64_t keyStrideHead,
    int64_t valueStrideToken,
    int64_t valueStrideHead,
    int64_t keyStorageOffset,
    int64_t valueStorageOffset,
    const aclTensor* keyCache,
    const aclTensor* valueCache,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnBitResidualPackK8v4(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
