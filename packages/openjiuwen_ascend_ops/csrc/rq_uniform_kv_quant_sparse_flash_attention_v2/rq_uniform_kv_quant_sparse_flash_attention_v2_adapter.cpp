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
constexpr int64_t kAbsorbQDim = 576;
constexpr int64_t kAbsorbOutDim = 512;
constexpr int64_t kSchemaASlotBytes = 388;
constexpr int64_t kSchemaAMode = 4;
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
    TORCH_CHECK(query.dim() == 3 && query.size(-1) == kAbsorbQDim,
                "query must be TND/BSND with last dim 576");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query must be fp16 or bf16");
    TORCH_CHECK(key.scalar_type() == at::kChar || key.scalar_type() == at::kByte,
                "key must be int8/uint8 byte container");
    TORCH_CHECK(value.is_same(key) || value.sizes() == key.sizes(),
                "value must alias key or match key shape for fused Schema A cache");
    TORCH_CHECK(key.dim() == 4 && key.size(2) == 1 && key.size(3) == kSchemaASlotBytes,
                "key/value must be PA_BSND [Bn, Bs, 1, 388]");
    TORCH_CHECK(key_quant_mode == kSchemaAMode && value_quant_mode == kSchemaAMode,
                "quant_mode must be 4 (Schema A); TQ4/386 is rejected");
    TORCH_CHECK(attention_mode == 2, "attention_mode must be 2 (MLA absorb)");
    TORCH_CHECK(rope_head_dim == 64, "rope_head_dim must be 64");
    TORCH_CHECK(layout_query == "TND" || layout_query == "BSND", "layout_query must be TND or BSND");
    TORCH_CHECK(layout_kv == "PA_BSND", "layout_kv must be PA_BSND");

    at::Tensor attnOut;
    if (out.has_value() && out->defined()) {
        attnOut = out.value();
        TORCH_CHECK(attnOut.sizes().vec() ==
                        std::vector<int64_t>({query.size(0), query.size(1), kAbsorbOutDim}),
                    "out must be [T, Nq, 512] matching query batch/heads");
    } else {
        attnOut = at::empty({query.size(0), query.size(1), kAbsorbOutDim}, query.options());
    }

    at::Tensor q = query.is_contiguous() ? query : query.contiguous();
    at::Tensor k = key.is_contiguous() ? key : key.contiguous();
    at::Tensor v = value.is_contiguous() ? value : value.contiguous();
    at::Tensor idx = sparse_indices.is_contiguous() ? sparse_indices : sparse_indices.contiguous();

    c10::optional<at::Tensor> bt = block_table;
    c10::optional<at::Tensor> actualQLens = actual_seq_lengths_query;
    c10::optional<at::Tensor> actualKvLens = actual_seq_lengths_kv;
    if (bt.has_value() && bt->defined() && !bt->is_contiguous()) {
        bt = bt->contiguous();
    }
    if (actualQLens.has_value() && actualQLens->defined() && !actualQLens->is_contiguous()) {
        actualQLens = actualQLens->contiguous();
    }
    if (actualKvLens.has_value() && actualKvLens->defined() && !actualKvLens->is_contiguous()) {
        actualKvLens = actualKvLens->contiguous();
    }

    const c10_npu::OptionalNPUGuard deviceGuard(q.device());
    const int64_t preTokenLimit = std::numeric_limits<int64_t>::max();
    const int64_t nextTokenLimit = std::numeric_limits<int64_t>::max();
    const std::string layoutQueryStr(layout_query);
    const std::string layoutKvStr(layout_kv);
    const char* layoutQueryCStr = layoutQueryStr.c_str();
    const char* layoutKvCStr = layoutKvStr.c_str();

    const c10::optional<at::Tensor> absentScale = c10::nullopt;
    EXEC_NPU_CMD(
        aclnnRqUniformKvQuantSparseFlashAttentionV2,
        q,
        k,
        v,
        idx,
        absentScale,
        absentScale,
        bt,
        actualQLens,
        actualKvLens,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        layoutQueryCStr,
        layoutKvCStr,
        sparse_mode,
        preTokenLimit,
        nextTokenLimit,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
        attnOut);

    return attnOut;
}

}  // namespace openjiuwen_ascend_ops
