#ifndef ACLNN_RQ_UNIFORM_DEQUANT_MLA_LATENT_H_
#define ACLNN_RQ_UNIFORM_DEQUANT_MLA_LATENT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformDequantMlaLatentGetWorkspaceSize(
    const aclTensor* latentCache,
    const aclTensor* rotation,
    const aclTensor* blockTable,
    const aclTensor* contextLens,
    const aclTensor* seqStarts,
    int64_t blockSize,
    int64_t latentDim,
    int64_t numBlocks,
    int64_t numReqs,
    int64_t maxBlocks,
    int64_t totalTokens,
    const aclTensor* c,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformDequantMlaLatent(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
