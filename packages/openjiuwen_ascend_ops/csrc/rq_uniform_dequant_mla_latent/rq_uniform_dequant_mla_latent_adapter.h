#ifndef RQ_UNIFORM_DEQUANT_MLA_LATENT_ADAPTER_H
#define RQ_UNIFORM_DEQUANT_MLA_LATENT_ADAPTER_H

#include <torch/extension.h>

namespace openjiuwen_ascend_ops {

// Gather Scheme A latent rows and restore fp c = y @ R.
// y = base + int8(code) * step, with R the inverse of pack's R^T.
// k_pe is not touched. Output rows follow request order in context_lens.
// total_tokens must equal sum(context_lens); the caller already knows it on host.
at::Tensor rq_uniform_dequant_mla_latent(
    const at::Tensor& latent_cache,
    const at::Tensor& rotation,
    const at::Tensor& block_table,
    const at::Tensor& context_lens,
    const at::Tensor& seq_starts,
    int64_t block_size,
    int64_t total_tokens);

}  // namespace openjiuwen_ascend_ops

#endif
