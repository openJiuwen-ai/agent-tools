/*!
 * \file rq_uniform_kv_quant_sparse_flash_attention_v2_adapter.cpp
 * \brief
 */

#include "rq_uniform_kv_quant_sparse_flash_attention_v2_adapter.h"

#include <limits>

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

#include "../aclnn_torch_adapter/op_api_common.h"
#include "op_host/aclnn_rq_uniform_kv_quant_sparse_flash_attention_v2.h"

namespace openjiuwen_ascend_ops {

namespace {
constexpr int64_t kQueryHeadDim = 576;
constexpr int64_t kOutHeadDim = 512;
constexpr int64_t kSlotBytes = 388;
constexpr int64_t kSchemaAQuantMode = 4;
}  // namespace

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
    const c10::optional<at::Tensor>& out)
{
    TORCH_CHECK(query.dim() == 3 && query.size(-1) == kQueryHeadDim,
                "query must be TND/BSND with last dim 576");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query must be fp16 or bf16");
    TORCH_CHECK(key.scalar_type() == at::kChar || key.scalar_type() == at::kByte,
                "key must be int8/uint8 byte container");
    TORCH_CHECK(value.is_same(key) || value.sizes() == key.sizes(),
                "value must alias key or match key shape for fused Schema A cache");
    TORCH_CHECK(key.dim() == 4 && key.size(2) == 1 && key.size(3) == kSlotBytes,
                "key/value must be PA_BSND [Bn, Bs, 1, 388]");
    TORCH_CHECK(key_quant_mode == kSchemaAQuantMode && value_quant_mode == kSchemaAQuantMode,
                "quant_mode must be 4 (Schema A); TQ4/386 is rejected");
    TORCH_CHECK(attention_mode == 2, "attention_mode must be 2 (MLA absorb)");
    TORCH_CHECK(rope_head_dim == 64, "rope_head_dim must be 64");
    TORCH_CHECK(layout_query == "TND" || layout_query == "BSND", "layout_query must be TND or BSND");
    TORCH_CHECK(layout_kv == "PA_BSND", "layout_kv must be PA_BSND");

    at::Tensor attention_out;
    if (out.has_value() && out->defined()) {
        attention_out = out.value();
        TORCH_CHECK(attention_out.sizes().vec() ==
                        std::vector<int64_t>({query.size(0), query.size(1), kOutHeadDim}),
                    "out must be [T, Nq, 512] matching query batch/heads");
    } else {
        attention_out = at::empty({query.size(0), query.size(1), kOutHeadDim}, query.options());
    }

    at::Tensor q = query.is_contiguous() ? query : query.contiguous();
    at::Tensor k = key.is_contiguous() ? key : key.contiguous();
    at::Tensor v = value.is_contiguous() ? value : value.contiguous();
    at::Tensor idx = sparse_indices.is_contiguous() ? sparse_indices : sparse_indices.contiguous();

    c10::optional<at::Tensor> bt = block_table;
    c10::optional<at::Tensor> act_q = actual_seq_lengths_query;
    c10::optional<at::Tensor> act_kv = actual_seq_lengths_kv;
    if (bt.has_value() && bt->defined() && !bt->is_contiguous()) {
        bt = bt->contiguous();
    }
    if (act_q.has_value() && act_q->defined() && !act_q->is_contiguous()) {
        act_q = act_q->contiguous();
    }
    if (act_kv.has_value() && act_kv->defined() && !act_kv->is_contiguous()) {
        act_kv = act_kv->contiguous();
    }

    const c10_npu::OptionalNPUGuard npuGuard(q.device());
    const int64_t pre_tokens = std::numeric_limits<int64_t>::max();
    const int64_t next_tokens = std::numeric_limits<int64_t>::max();
    const std::string layout_q(layout_query);
    const std::string layout_k(layout_kv);
    const char* layout_q_cstr = layout_q.c_str();
    const char* layout_k_cstr = layout_k.c_str();

    const c10::optional<at::Tensor> no_scale = c10::nullopt;
    EXEC_NPU_CMD(
        aclnnRqUniformKvQuantSparseFlashAttentionV2,
        q,
        k,
        v,
        idx,
        no_scale,
        no_scale,
        bt,
        act_q,
        act_kv,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        layout_q_cstr,
        layout_k_cstr,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
        attention_out);

    return attention_out;
}

}  // namespace openjiuwen_ascend_ops
