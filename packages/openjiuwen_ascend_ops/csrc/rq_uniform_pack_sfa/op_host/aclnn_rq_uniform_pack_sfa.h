#ifndef ACLNN_RQ_UNIFORM_PACK_SFA_H_
#define ACLNN_RQ_UNIFORM_PACK_SFA_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformPackSfaGetWorkspaceSize(
    const aclTensor* nope,
    const aclTensor* rope,
    const aclTensor* slotMapping,
    int64_t blockSize,
    int64_t numTokens,
    int64_t numBlocks,
    const aclTensor* kvCache,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformPackSfa(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
