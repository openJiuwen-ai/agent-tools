#ifndef RQ_UNIFORM_FIA_PAGED_K4V4_ADAPTER_H
#define RQ_UNIFORM_FIA_PAGED_K4V4_ADAPTER_H

#include <torch/extension.h>

namespace openjiuwen_ascend_ops {

at::Tensor rq_uniform_fia_paged_k4v4(
    const at::Tensor &query,
    const at::Tensor &key_cache,
    const at::Tensor &value_cache,
    const at::Tensor &block_table,
    at::IntArrayRef actual_seq_len_q,
    at::IntArrayRef actual_seq_len_kv,
    const c10::optional<at::Tensor> &atten_mask,
    const at::Tensor &rotation_key,
    const at::Tensor &rotation_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t block_size,
    double scale_value,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t sparse_mode,
    c10::optional<at::Tensor> out_opt);

}

#endif
