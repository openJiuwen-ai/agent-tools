#include "rq_uniform_pack_mla_latent_adapter.h"

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

#include "../aclnn_torch_adapter/op_api_common.h"

namespace openjiuwen_ascend_ops {

void rq_uniform_pack_mla_latent(
    const at::Tensor& c,
    const at::Tensor& rotation_t,
    const at::Tensor& slot_mapping,
    at::Tensor& latent_cache,
    int64_t block_size) {
    constexpr int64_t kLatentAlign = 64;
    constexpr int64_t kMaxLatent = 512;
    constexpr int64_t kVectorTile = 16;

    TORCH_CHECK(c.dim() == 2, "c must be [T, L]");
    const int64_t num_tokens = c.size(0);
    const int64_t latent_dim = c.size(1);
    TORCH_CHECK(latent_dim > 0 && latent_dim % kLatentAlign == 0 && latent_dim <= kMaxLatent,
                "latent dim must be a multiple of 64 and <= 512, got ", latent_dim);
    TORCH_CHECK(rotation_t.dim() == 2 && rotation_t.size(0) == latent_dim &&
                    rotation_t.size(1) == latent_dim,
                "rotation_t must be [L, L]");
    TORCH_CHECK(c.scalar_type() == at::kHalf || c.scalar_type() == at::kBFloat16,
                "c must be fp16 or bf16");
    TORCH_CHECK(rotation_t.scalar_type() == c.scalar_type(), "rotation_t dtype must match c");
    TORCH_CHECK(block_size > 0 && block_size % kVectorTile == 0,
                "block_size must be a positive multiple of 16, got ", block_size);
    TORCH_CHECK(latent_cache.scalar_type() == at::kByte, "latent_cache must be uint8");
    TORCH_CHECK(latent_cache.dim() == 2, "latent_cache must be [num_blocks, stride]");
    const int64_t num_blocks = latent_cache.size(0);
    const int64_t expect_stride = block_size * (latent_dim + 4);
    TORCH_CHECK(latent_cache.size(1) == expect_stride,
                "latent_cache stride must be block_size*(L+4)=", expect_stride,
                ", got ", latent_cache.size(1));
    TORCH_CHECK(latent_cache.is_contiguous(), "latent_cache must be contiguous");
    if (num_tokens == 0) {
        return;
    }

    at::Tensor c_work = c.is_contiguous() ? c : c.contiguous();
    at::Tensor rotation_work = rotation_t.is_contiguous() ? rotation_t : rotation_t.contiguous();
    at::Tensor slot_work = slot_mapping.is_contiguous() ? slot_mapping : slot_mapping.contiguous();
    const c10_npu::OptionalNPUGuard npuGuard(c_work.device());
    EXEC_NPU_CMD(
        aclnnRqUniformPackMlaLatent,
        c_work,
        rotation_work,
        slot_work,
        block_size,
        num_tokens,
        latent_dim,
        num_blocks,
        latent_cache);
}

}  // namespace openjiuwen_ascend_ops
