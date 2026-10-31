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

void rq_uniform_pack_k4v4(
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &slot_mapping,
    const at::Tensor &query_start_loc,
    const at::Tensor &rotation_t,
    at::Tensor &key_cache,
    at::Tensor &value_cache,
    int64_t num_reqs,
    int64_t block_size) {
    constexpr int64_t kHeadSize = 128;
    constexpr int64_t kBlockRows = 16;
    constexpr int64_t kKeyBlockStride = 1088;
    constexpr int64_t kValBlockStride = 1088;
    TORCH_CHECK(key.is_privateuseone() && value.is_privateuseone(), "key/value must be on NPU");
    TORCH_CHECK(slot_mapping.is_privateuseone(), "slot_mapping must be on NPU");
    TORCH_CHECK(query_start_loc.is_privateuseone(), "query_start_loc must be on NPU");
    TORCH_CHECK(rotation_t.is_privateuseone(), "rotation_t must be on NPU");
    TORCH_CHECK(key_cache.is_privateuseone() && value_cache.is_privateuseone(),
                "key_cache/value_cache must be on NPU");
    TORCH_CHECK(key.device() == value.device() && key.device() == slot_mapping.device() &&
                    key.device() == query_start_loc.device() &&
                    key.device() == rotation_t.device() &&
                    key.device() == key_cache.device() && key.device() == value_cache.device(),
                "all tensors must be on the same NPU device");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key and value must have the same dtype");
    TORCH_CHECK(key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
                "RqUniform pack accepts fp16/bf16 key/value");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt, "query_start_loc must be int32");
    TORCH_CHECK(rotation_t.scalar_type() == key.scalar_type() &&
                    rotation_t.dim() == 2 && rotation_t.size(0) == kHeadSize &&
                    rotation_t.size(1) == kHeadSize,
                "rotation_t must be [128,128] and match key/value dtype");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte && value_cache.scalar_type() == at::kByte,
                "key_cache/value_cache must be uint8");
    TORCH_CHECK(key_cache.dim() == 3 && value_cache.dim() == 3,
                "RqUniform k4v4 caches must be [num_blocks, num_heads, packed_bytes]");
    TORCH_CHECK(key_cache.size(0) == value_cache.size(0) &&
                    key_cache.size(1) == value_cache.size(1),
                "key/value cache num_blocks and num_heads must match");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");
    TORCH_CHECK(num_reqs > 0, "num_reqs must be > 0");
    TORCH_CHECK(block_size % kBlockRows == 0,
                "RqUniform k4v4 cache layout requires block_size to be a multiple of 16");
    TORCH_CHECK(key_cache.size(2) == (block_size / kBlockRows) * kKeyBlockStride,
                "key_cache last dim must equal (block_size / 16) * 1088");
    TORCH_CHECK(value_cache.size(2) == (block_size / kBlockRows) * kValBlockStride,
                "value_cache last dim must equal (block_size / 16) * 1088");
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous(),
                "RqUniform caches must be contiguous");

    const int64_t head_size = key.size(-1);
    TORCH_CHECK(value.size(-1) == head_size, "key/value head_size mismatch");
    TORCH_CHECK(head_size == kHeadSize, "RqUniform pack supports head_size=128 only");
    TORCH_CHECK(key.numel() % head_size == 0 && value.numel() % head_size == 0,
                "key/value numel must align with head_size");
    const int64_t n_vec = key.numel() / head_size;
    TORCH_CHECK(value.numel() / head_size == n_vec, "key/value row count mismatch");
    const int64_t num_heads = key_cache.size(1);
    TORCH_CHECK(num_heads > 0 && key.dim() == 3 && value.dim() == 3,
                "key/value must be [num_tokens, num_heads, 128]");
    TORCH_CHECK(key.size(1) == num_heads && value.size(1) == num_heads,
                "key/value num_heads must match cache num_heads");
    const int64_t token_count = key.size(0);
    TORCH_CHECK(value.size(0) == token_count, "key/value token count mismatch");
    TORCH_CHECK(n_vec == token_count * num_heads,
                "key/value row count must equal num_tokens * num_heads");
    TORCH_CHECK(slot_mapping.numel() >= token_count, "slot_mapping length is smaller than token count");
    TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be a 1-D tensor");
    TORCH_CHECK(query_start_loc.numel() >= num_reqs + 1,
                "query_start_loc length must be at least num_reqs + 1");

    const bool key_strided_copy_supported =
        key.stride(2) == 1 && key.stride(0) > 0 && key.stride(1) > 0;
    const bool value_strided_copy_supported =
        value.stride(2) == 1 && value.stride(0) > 0 && value.stride(1) > 0;
    at::Tensor key_work = key;
    at::Tensor value_work = value;
    at::Tensor slot_work = slot_mapping;
    at::Tensor query_start_work = query_start_loc;
    at::Tensor rotation_work = rotation_t;
    if (!key_strided_copy_supported) {
        if (!key_work.is_contiguous()) {
            key_work = key_work.contiguous();
        }
        if (key_work.storage_offset() != 0) {
            key_work = key_work.clone(at::MemoryFormat::Contiguous);
        }
    }
    if (!value_strided_copy_supported) {
        if (!value_work.is_contiguous()) {
            value_work = value_work.contiguous();
        }
        if (value_work.storage_offset() != 0) {
            value_work = value_work.clone(at::MemoryFormat::Contiguous);
        }
    }
    if (!slot_work.is_contiguous()) {
        slot_work = slot_work.contiguous();
    }
    if (!query_start_work.is_contiguous()) {
        query_start_work = query_start_work.contiguous();
    }
    if (!rotation_work.is_contiguous()) {
        rotation_work = rotation_work.contiguous();
    }

    uint32_t vec_per_core = 128;
    if (n_vec < 128) {
        // Decode/chunked-prefill paths usually pack a handful of KV rows per
        // step; vecPerCore=2 matches the ori TurboQuant tuning (~41 us vs ~44).
        vec_per_core = static_cast<uint32_t>(n_vec <= 16 ? 2 : ((n_vec + 15) / 16) * 16);
        if (vec_per_core == 0) {
            vec_per_core = 16;
        }
    }
    const int64_t vec_per_core_i64 = static_cast<int64_t>(vec_per_core);
    const int64_t num_blocks = key_cache.size(0);
    const int64_t key_stride_token = key_work.stride(0);
    const int64_t key_stride_head = key_work.stride(1);
    const int64_t value_stride_token = value_work.stride(0);
    const int64_t value_stride_head = value_work.stride(1);
    // ConvertType passes tensor storage_offset to ACL, and the GM_ADDR seen by
    // the AscendC kernel is already at the logical tensor view. Keep the
    // kernel-side offset at zero to avoid applying storage_offset twice.
    const int64_t key_storage_offset = 0;
    const int64_t value_storage_offset = 0;
    constexpr int64_t kMaxKernelStride = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(key_stride_token > 0 && key_stride_head > 0 &&
                    value_stride_token > 0 && value_stride_head > 0 &&
                    key_stride_token <= kMaxKernelStride &&
                    key_stride_head <= kMaxKernelStride &&
                    value_stride_token <= kMaxKernelStride &&
                    value_stride_head <= kMaxKernelStride,
                "key/value strides must be positive and fit uint32_t");
    const c10_npu::OptionalNPUGuard npuGuard(key_work.device());
    EXEC_NPU_CMD(
        aclnnRqUniformPackK4v4,
        key_work,
        value_work,
        rotation_work,
        slot_work,
        query_start_work,
        n_vec,
        vec_per_core_i64,
        num_heads,
        block_size,
        num_blocks,
        num_reqs,
        key_stride_token,
        key_stride_head,
        value_stride_token,
        value_stride_head,
        key_storage_offset,
        value_storage_offset,
        key_cache,
        value_cache);
}

}
