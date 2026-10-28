#ifndef ACLNN_RQ_UNIFORM_PACK_MLA_LATENT_H_
#define ACLNN_RQ_UNIFORM_PACK_MLA_LATENT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformPackMlaLatentGetWorkspaceSize(
    const aclTensor* c,
    const aclTensor* rotationT,
    const aclTensor* slotMapping,
    int64_t blockSize,
    int64_t numTokens,
    int64_t latentDim,
    int64_t numBlocks,
    const aclTensor* latentCache,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformPackMlaLatent(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
