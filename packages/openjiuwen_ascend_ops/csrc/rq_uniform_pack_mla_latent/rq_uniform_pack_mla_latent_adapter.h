#ifndef RQ_UNIFORM_PACK_MLA_LATENT_ADAPTER_H
#define RQ_UNIFORM_PACK_MLA_LATENT_ADAPTER_H

#include <torch/extension.h>

namespace openjiuwen_ascend_ops {

// Fused MLA latent pack (MIX AIC:AIV = 1:2), same split as pack_k8v4:
//   cube rotates 32 tokens,  y = c @ R^T, fixpipe F322F16
//   two vector cores each encode 16 tokens (Key Scheme A)
//
// Cache is one uint8 slab per block:
//   [0, block_size*L)                     codes, byte = int8(q - 128)
//   [block_size*L, block_size*(L+2))      base' = min + 128*step
//   [block_size*(L+2), block_size*(L+4))  step = (max-min)/255
// Decode: y = base' + int8(code) * step.
// L must be a multiple of 64 and <= 512. block_size % 16 == 0.
void rq_uniform_pack_mla_latent(
    const at::Tensor& c,
    const at::Tensor& rotation_t,
    const at::Tensor& slot_mapping,
    at::Tensor& latent_cache,
    int64_t block_size);

}  // namespace openjiuwen_ascend_ops

#endif
