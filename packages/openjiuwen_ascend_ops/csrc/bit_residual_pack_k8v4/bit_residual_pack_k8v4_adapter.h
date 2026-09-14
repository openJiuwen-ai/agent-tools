#ifndef BIT_RESIDUAL_PACK_K8V4_ADAPTER_H
#define BIT_RESIDUAL_PACK_K8V4_ADAPTER_H

#include <torch/extension.h>

namespace openjiuwen_ascend_ops {

void bit_residual_pack_k8v4(
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &slot_mapping,
    const at::Tensor &query_start_loc,
    const at::Tensor &rotation_t,
    at::Tensor &key_cache,
    at::Tensor &value_cache,
    int64_t num_reqs,
    int64_t block_size);

}

#endif
