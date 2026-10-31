#ifndef ACLNN_RQ_UNIFORM_FIA_PAGED_K4V4_MSD_H_
#define ACLNN_RQ_UNIFORM_FIA_PAGED_K4V4_MSD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * RqUniform FIA Paged K4V4 MSD (Decode MSD read path).
 * IO matches aclnnRqUniformFiaPagedK4v4.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformFiaPagedK4v4MsdGetWorkspaceSize(
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
aclnnStatus aclnnRqUniformFiaPagedK4v4Msd(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
