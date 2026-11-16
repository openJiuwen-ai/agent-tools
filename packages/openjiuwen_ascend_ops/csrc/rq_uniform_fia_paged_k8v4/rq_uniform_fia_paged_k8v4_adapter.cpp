#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/npu/Module.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "../aclnn_torch_adapter/op_api_common.h"

namespace openjiuwen_ascend_ops {
at::Tensor rq_uniform_fia_paged_k8v4(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    at::IntArrayRef actual_seq_len_q,
    at::IntArrayRef actual_seq_len_kv,
    const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& rotation_key,
    const c10::optional<at::Tensor>& rotation_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t block_size,
    double scale_value,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t sparse_mode,
    c10::optional<at::Tensor> out_opt,
    int64_t quant_mode)
{
    constexpr int64_t kBlockRows = 16;
    constexpr int64_t kQuantModeA2 = 0;
    constexpr int64_t kQuantModeA5Port = 2;
    TORCH_CHECK(quant_mode == kQuantModeA2 || quant_mode == kQuantModeA5Port,
                "quant_mode must be 0 (A2) or 2 (A5 antiquant port); 1 is reserved, got ", quant_mode);
    const bool a5Port = quant_mode == kQuantModeA5Port;
    TORCH_CHECK(query.is_privateuseone(), "query must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone(), "key_cache must be on NPU");
    TORCH_CHECK(value_cache.is_privateuseone(), "value_cache must be on NPU");
    if (a5Port) {
        TORCH_CHECK(query.scalar_type() == at::kHalf, "A5 antiquant port accepts fp16 query");
        TORCH_CHECK(head_size == 128 || head_size == 256,
                    "A5 antiquant port only supports head_size=128 or 256");
        TORCH_CHECK(query.dim() == 3 || query.dim() == 4,
                    "A5 antiquant port query must be [B,N,D] (decode) or [B,N,S,D] (prefill)");
        if (query.dim() == 3) {
            TORCH_CHECK(query.size(2) == head_size, "query last dim must equal head_size");
        } else {
            TORCH_CHECK(query.size(3) == head_size, "query last dim must equal head_size");
            TORCH_CHECK(query.size(2) >= 1, "prefill s1 must be >= 1");
        }
    } else {
        TORCH_CHECK(
            query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
            "RqUniform FIA accepts fp16 or bf16 query");
        TORCH_CHECK(head_size == 128 || head_size == 256, "RqUniform FIA only supports head_size=128 or 256");
    }
    const int64_t kKeyBlockStride = kBlockRows * (head_size + 4);
    const int64_t kValBlockStride = kBlockRows * (head_size / 2 + 4);
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(block_size % kBlockRows == 0,
                "block_size must be a multiple of 16 (sub-block rows)");
    TORCH_CHECK(num_heads > 0 && num_kv_heads > 0, "head counts must be > 0");
    TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");
    const bool rotKeyOk = rotation_key.has_value() && rotation_key->defined() && rotation_key->numel() > 0;
    const bool rotValueOk = rotation_value.has_value() && rotation_value->defined() && rotation_value->numel() > 0;
    if (!a5Port) {
        TORCH_CHECK(rotKeyOk && rotation_key->is_privateuseone() &&
                    rotation_key->scalar_type() == query.scalar_type() &&
                    rotation_key->dim() == 2 && rotation_key->size(0) == head_size &&
                    rotation_key->size(1) == head_size,
                    "rotation_key must be [D,D] matching query dtype");
        TORCH_CHECK(rotValueOk && rotation_value->is_privateuseone() &&
                    rotation_value->scalar_type() == query.scalar_type() &&
                    rotation_value->dim() == 2 && rotation_value->size(0) == head_size &&
                    rotation_value->size(1) == head_size,
                    "rotation_value must be [D,D] matching query dtype");
    }
    TORCH_CHECK(key_cache.dim() == 3 && value_cache.dim() == 3,
                "RqUniform caches must be [num_blocks, num_kv_heads, packed_bytes]");
    TORCH_CHECK(key_cache.size(1) == num_kv_heads && value_cache.size(1) == num_kv_heads,
                "cache num_kv_heads mismatch");
    if (a5Port) {
        const int64_t rowBytes = block_size * head_size;
        TORCH_CHECK(key_cache.size(2) == rowBytes,
                    "key_cache last dim must equal block_size * D int8 bytes");
        TORCH_CHECK(value_cache.size(2) == rowBytes,
                    "value_cache last dim must equal block_size * D int8 bytes");
    } else {
        TORCH_CHECK(key_cache.size(2) == (block_size / kBlockRows) * kKeyBlockStride,
                    "key_cache last dim must equal (block_size / 16) * key_tile");
        TORCH_CHECK(value_cache.size(2) == (block_size / kBlockRows) * kValBlockStride,
                    "value_cache last dim must equal (block_size / 16) * val_tile");
    }
    TORCH_CHECK(actual_seq_len_q.size() > 0, "actual_seq_len_q must not be empty");
    TORCH_CHECK(actual_seq_len_q.size() == actual_seq_len_kv.size(),
                "actual_seq_len_q and actual_seq_len_kv must have the same length");

    const at::Tensor query_c = query.contiguous();
    const at::Tensor key_cache_c = key_cache.contiguous();
    const at::Tensor value_cache_c = value_cache.contiguous();
    const at::Tensor block_table_c = block_table.contiguous();
    const at::Tensor rotation_key_c = rotKeyOk ? rotation_key->contiguous()
                                               : at::empty({0}, query.options());
    const at::Tensor rotation_value_c = rotValueOk ? rotation_value->contiguous()
                                                   : at::empty({0}, query.options());
    at::Tensor mask = atten_mask.has_value() && atten_mask->defined()
                          ? atten_mask->contiguous()
                          : at::empty({0}, query.options().dtype(at::kChar));

    at::Tensor out;
    if (out_opt.has_value() && out_opt->defined()) {
        TORCH_CHECK(out_opt->is_privateuseone(), "out must be on NPU");
        TORCH_CHECK(out_opt->sizes().equals(query_c.sizes()), "out shape must match query");
        TORCH_CHECK(out_opt->scalar_type() == query_c.scalar_type(), "out dtype must match query");
        TORCH_CHECK(out_opt->device() == query_c.device(), "out device must match query");
        TORCH_CHECK(out_opt->is_contiguous(), "out must be contiguous for in-place write");
        out = *out_opt;
    } else {
        out = at::empty(query_c.sizes(), query_c.options());
    }
    const c10_npu::OptionalNPUGuard npuGuard(query_c.device());
    EXEC_NPU_CMD(
        aclnnRqUniformFiaPagedK8v4,
        query_c,
        key_cache_c,
        value_cache_c,
        block_table_c,
        actual_seq_len_q,
        actual_seq_len_kv,
        mask,
        rotation_key_c,
        rotation_value_c,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        scale_value,
        pre_tokens,
        next_tokens,
        sparse_mode,
        quant_mode,
        out);
    return out;
}

}
