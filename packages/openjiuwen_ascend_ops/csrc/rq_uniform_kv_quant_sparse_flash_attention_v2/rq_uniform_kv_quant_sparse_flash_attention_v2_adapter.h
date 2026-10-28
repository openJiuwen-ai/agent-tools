#pragma once

#include <torch/extension.h>

namespace openjiuwen_ascend_ops {

at::Tensor rq_uniform_kv_quant_sparse_flash_attention_v2(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_kv,
    double scale_value,
    int64_t key_quant_mode,
    int64_t value_quant_mode,
    int64_t sparse_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode,
    int64_t attention_mode,
    int64_t quant_scale_repo_mode,
    int64_t tile_size,
    int64_t rope_head_dim,
    const c10::optional<at::Tensor>& out);

}  // namespace openjiuwen_ascend_ops
