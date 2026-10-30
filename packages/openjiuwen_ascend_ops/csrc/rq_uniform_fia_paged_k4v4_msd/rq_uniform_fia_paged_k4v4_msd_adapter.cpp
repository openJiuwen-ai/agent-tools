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

// BOOTSTRAP: same checks/IO as BRFIA; kernel currently shares FIA TQ P0 until MSD MVP-1.
at::Tensor rq_uniform_fia_paged_k4v4_msd(
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
    c10::optional<at::Tensor> out_opt) {
    constexpr int64_t kHeadSize = 128;
    constexpr int64_t kBlockRows = 16;
    constexpr int64_t kKeyBlockStride = 1088;
    constexpr int64_t kValBlockStride = 1088;
    TORCH_CHECK(query.is_privateuseone(), "query must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone(), "key_cache must be on NPU");
    TORCH_CHECK(value_cache.is_privateuseone(), "value_cache must be on NPU");
    TORCH_CHECK(rotation_key.is_privateuseone(), "rotation_key must be on NPU");
    TORCH_CHECK(rotation_value.is_privateuseone(), "rotation_value must be on NPU");
    TORCH_CHECK(
        query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
        "RqUniform FIA MSD accepts fp16 or bf16 query");
    TORCH_CHECK(head_size == kHeadSize, "RqUniform FIA MSD only supports head_size=128");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(block_size % kBlockRows == 0,
                "block_size must be a multiple of 16 (sub-block rows)");
    TORCH_CHECK(num_heads > 0 && num_kv_heads > 0, "head counts must be > 0");
    TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");
    const int64_t us2NativeLast = block_size * kHeadSize;
    const bool us2NativePa =
        (key_cache.dim() == 4 && value_cache.dim() == 4 &&
         key_cache.size(1) == num_kv_heads && value_cache.size(1) == num_kv_heads &&
         key_cache.size(2) == block_size && value_cache.size(2) == block_size &&
         key_cache.size(3) == kHeadSize && value_cache.size(3) == kHeadSize) ||
        (key_cache.dim() == 3 && value_cache.dim() == 3 &&
         key_cache.size(1) == num_kv_heads && value_cache.size(1) == num_kv_heads &&
         key_cache.size(2) == us2NativeLast && value_cache.size(2) == us2NativeLast);
    if (us2NativePa) {
        // Bring-up: Us2 PA [num_blocks, NKV, BS, D] int8.view(uint8);
        // rotation_* are per-channel scales [NKV, D], not Haar Π.
        TORCH_CHECK(rotation_key.scalar_type() == query.scalar_type() &&
                    rotation_value.scalar_type() == query.scalar_type(),
                    "Us2-native scales must match query dtype");
        TORCH_CHECK(rotation_key.dim() == 2 && rotation_key.size(0) == num_kv_heads &&
                    rotation_key.size(1) == kHeadSize,
                    "Us2-native rotation_key must be [num_kv_heads, 128] per-channel scale");
        TORCH_CHECK(rotation_value.dim() == 2 && rotation_value.size(0) == num_kv_heads &&
                    rotation_value.size(1) == kHeadSize,
                    "Us2-native rotation_value must be [num_kv_heads, 128] per-channel scale");
    } else {
        TORCH_CHECK(rotation_key.scalar_type() == query.scalar_type() &&
                    rotation_key.dim() == 2 && rotation_key.size(0) == kHeadSize &&
                    rotation_key.size(1) == kHeadSize,
                    "rotation_key must be [128,128] matching query dtype");
        TORCH_CHECK(rotation_value.scalar_type() == query.scalar_type() &&
                    rotation_value.dim() == 2 && rotation_value.size(0) == kHeadSize &&
                    rotation_value.size(1) == kHeadSize,
                    "rotation_value must be [128,128] matching query dtype");
        TORCH_CHECK(key_cache.dim() == 3 && value_cache.dim() == 3,
                    "RqUniform caches must be [num_blocks, num_kv_heads, packed_bytes]");
        TORCH_CHECK(key_cache.size(1) == num_kv_heads && value_cache.size(1) == num_kv_heads,
                    "cache num_kv_heads mismatch");
        TORCH_CHECK(key_cache.size(2) == (block_size / kBlockRows) * kKeyBlockStride,
                    "key_cache last dim must equal (block_size / 16) * 1088");
        TORCH_CHECK(value_cache.size(2) == (block_size / kBlockRows) * kValBlockStride,
                    "value_cache last dim must equal (block_size / 16) * 1088");
    }
    TORCH_CHECK(actual_seq_len_q.size() > 0, "actual_seq_len_q must not be empty");
    TORCH_CHECK(actual_seq_len_q.size() == actual_seq_len_kv.size(),
                "actual_seq_len_q and actual_seq_len_kv must have the same length");

    const at::Tensor query_c = query.contiguous();
    const at::Tensor key_cache_c = key_cache.contiguous();
    const at::Tensor value_cache_c = value_cache.contiguous();
    const at::Tensor block_table_c = block_table.contiguous();
    const at::Tensor rotation_key_c = rotation_key.contiguous();
    const at::Tensor rotation_value_c = rotation_value.contiguous();
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
        aclnnRqUniformFiaPagedK4v4Msd,
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
        out);
    return out;
}

}
