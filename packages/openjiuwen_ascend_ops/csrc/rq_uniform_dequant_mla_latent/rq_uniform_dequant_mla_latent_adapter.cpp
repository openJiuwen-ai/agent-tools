#include "rq_uniform_dequant_mla_latent_adapter.h"

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

#include "../aclnn_torch_adapter/op_api_common.h"

namespace openjiuwen_ascend_ops {

at::Tensor rq_uniform_dequant_mla_latent(
    const at::Tensor& latent_cache,
    const at::Tensor& rotation,
    const at::Tensor& block_table,
    const at::Tensor& context_lens,
    const at::Tensor& seq_starts,
    int64_t block_size,
    int64_t total_tokens) {
    constexpr int64_t kLatentAlign = 64;
    constexpr int64_t kMaxLatent = 512;
    constexpr int64_t kVectorTile = 16;

    TORCH_CHECK(rotation.dim() == 2 && rotation.size(0) == rotation.size(1), "rotation must be [L, L]");
    const int64_t latent_dim = rotation.size(0);
    TORCH_CHECK(latent_dim > 0 && latent_dim % kLatentAlign == 0 && latent_dim <= kMaxLatent,
                "latent dim must be a multiple of 64 and <= 512, got ", latent_dim);
    TORCH_CHECK(rotation.scalar_type() == at::kHalf || rotation.scalar_type() == at::kBFloat16,
                "rotation must be fp16 or bf16");
    TORCH_CHECK(block_size > 0 && block_size % kVectorTile == 0,
                "block_size must be a positive multiple of 16, got ", block_size);
    TORCH_CHECK(total_tokens >= 0, "total_tokens must be non-negative, got ", total_tokens);
    TORCH_CHECK(latent_cache.scalar_type() == at::kByte, "latent_cache must be uint8");
    TORCH_CHECK(latent_cache.dim() == 2, "latent_cache must be [num_blocks, stride]");
    const int64_t num_blocks = latent_cache.size(0);
    const int64_t expect_stride = block_size * (latent_dim + 4);
    TORCH_CHECK(latent_cache.size(1) == expect_stride,
                "latent_cache stride must be block_size*(L+4)=", expect_stride, ", got ", latent_cache.size(1));
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [num_reqs, max_blocks]");
    const int64_t num_reqs = block_table.size(0);
    const int64_t max_blocks = block_table.size(1);
    TORCH_CHECK(num_reqs > 0 && max_blocks > 0, "block_table must be non-empty");
    TORCH_CHECK(context_lens.dim() == 1 && context_lens.numel() == num_reqs, "context_lens must be [num_reqs]");
    TORCH_CHECK(seq_starts.dim() == 1 && seq_starts.numel() == num_reqs, "seq_starts must be [num_reqs]");

    auto out_opts = rotation.options();
    if (total_tokens == 0) {
        return at::empty({0, latent_dim}, out_opts);
    }

    at::Tensor cache_work = latent_cache.is_contiguous() ? latent_cache : latent_cache.contiguous();
    at::Tensor rotation_work = rotation.is_contiguous() ? rotation : rotation.contiguous();
    at::Tensor table_work = block_table.to(at::kInt).contiguous();
    at::Tensor lens_work = context_lens.to(at::kInt).contiguous();
    at::Tensor starts_work = seq_starts.to(at::kInt).contiguous();
    at::Tensor c_out = at::empty({total_tokens, latent_dim}, out_opts);

    const c10_npu::OptionalNPUGuard npuGuard(cache_work.device());
    EXEC_NPU_CMD(
        aclnnRqUniformDequantMlaLatent,
        cache_work,
        rotation_work,
        table_work,
        lens_work,
        starts_work,
        block_size,
        latent_dim,
        num_blocks,
        num_reqs,
        max_blocks,
        total_tokens,
        c_out);
    return c_out;
}

}  // namespace openjiuwen_ascend_ops
