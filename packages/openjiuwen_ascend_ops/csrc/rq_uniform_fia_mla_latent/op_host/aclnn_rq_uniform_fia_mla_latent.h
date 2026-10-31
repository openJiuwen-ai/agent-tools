
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RQ_UNIFORM_FIA_MLA_LATENT_H_
#define ACLNN_RQ_UNIFORM_FIA_MLA_LATENT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRqUniformFiaMlaLatentGetWorkspaceSize
 * parameters :
 * qNope : required
 * qPe : required
 * latentCache : required
 * kPeCache : required
 * blockTable : required
 * actualSeqQ : required
 * actualSeqKv : required
 * wUkT : required
 * wUv : required
 * rotation : required
 * blockSize : required
 * scale : required
 * numTokens : required
 * numHeads : required
 * nopeDim : required
 * vDim : required
 * numBlocks : required
 * numReqs : required
 * maxBlocks : required
 * maxKv : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformFiaMlaLatentGetWorkspaceSize(
    const aclTensor *qNope,
    const aclTensor *qPe,
    const aclTensor *latentCache,
    const aclTensor *kPeCache,
    const aclTensor *blockTable,
    const aclIntArray *actualSeqQ,
    const aclIntArray *actualSeqKv,
    const aclTensor *wUkT,
    const aclTensor *wUv,
    const aclTensor *rotation,
    int64_t blockSize,
    double scale,
    int64_t numTokens,
    int64_t numHeads,
    int64_t nopeDim,
    int64_t vDim,
    int64_t numBlocks,
    int64_t numReqs,
    int64_t maxBlocks,
    int64_t maxKv,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformFiaMlaLatentTensorGetWorkspaceSize(
    const aclTensor *qNope,
    const aclTensor *qPe,
    const aclTensor *latentCache,
    const aclTensor *kPeCache,
    const aclTensor *blockTable,
    const aclTensor *actualSeqQ,
    const aclTensor *actualSeqKv,
    const aclTensor *wUkT,
    const aclTensor *wUv,
    const aclTensor *rotation,
    int64_t blockSize,
    double scale,
    int64_t numTokens,
    int64_t numHeads,
    int64_t nopeDim,
    int64_t vDim,
    int64_t numBlocks,
    int64_t numReqs,
    int64_t maxBlocks,
    int64_t maxKv,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRqUniformFiaMlaLatent
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRqUniformFiaMlaLatent(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
