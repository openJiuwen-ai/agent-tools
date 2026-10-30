#ifndef RQ_UNIFORM_PACK_SFA_ADAPTER_H
#define RQ_UNIFORM_PACK_SFA_ADAPTER_H

#include <torch/extension.h>

namespace openjiuwen_ascend_ops {

// Pack already-rotated SFA nope [T,512] + rope [T,64] into PA_BSND
// kv_cache [Bn, Bs, 1, 388] (uint8). Decode: y = offset + u4(code) * scale.
void rq_uniform_pack_sfa(
    const at::Tensor& nope,
    const at::Tensor& rope,
    const at::Tensor& slot_mapping,
    at::Tensor& kv_cache,
    int64_t block_size);

}  // namespace openjiuwen_ascend_ops

#endif
