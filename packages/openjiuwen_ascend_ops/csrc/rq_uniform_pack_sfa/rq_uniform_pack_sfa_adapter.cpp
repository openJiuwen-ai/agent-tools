#include "rq_uniform_pack_sfa_adapter.h"

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

#include "../aclnn_torch_adapter/op_api_common.h"

namespace openjiuwen_ascend_ops {

namespace {
constexpr int64_t kNopeDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kSlotBytes = 388;
constexpr int64_t kVectorTile = 16;
}  // namespace

void rq_uniform_pack_sfa(
    const at::Tensor& nope,
    const at::Tensor& rope,
    const at::Tensor& slot_mapping,
    at::Tensor& kv_cache,
    int64_t block_size)
{
    TORCH_CHECK(nope.dim() == 2 && nope.size(1) == kNopeDim, "nope must be [T, 512]");
    const int64_t num_tokens = nope.size(0);
    TORCH_CHECK(rope.dim() == 2 && rope.size(0) == num_tokens && rope.size(1) == kRopeDim,
                "rope must be [T, 64]");
    TORCH_CHECK(nope.scalar_type() == at::kHalf || nope.scalar_type() == at::kBFloat16,
                "nope must be fp16 or bf16");
    TORCH_CHECK(rope.scalar_type() == nope.scalar_type(), "rope dtype must match nope");
    TORCH_CHECK(block_size > 0 && block_size % kVectorTile == 0,
                "block_size must be a positive multiple of 16, got ", block_size);
    TORCH_CHECK(kv_cache.scalar_type() == at::kByte, "kv_cache must be uint8");
    TORCH_CHECK(
        kv_cache.dim() == 4 && kv_cache.size(1) == block_size && kv_cache.size(2) == 1 &&
            kv_cache.size(3) == kSlotBytes,
        "kv_cache must be PA_BSND [Bn, Bs, 1, 388]");
    TORCH_CHECK(kv_cache.is_contiguous(), "kv_cache must be contiguous");
    TORCH_CHECK(
        slot_mapping.scalar_type() == at::kLong || slot_mapping.scalar_type() == at::kInt,
        "slot_mapping must be int64 (int32 is promoted)");
    const int64_t num_blocks = kv_cache.size(0);
    if (num_tokens == 0) {
        return;
    }

    at::Tensor nope_work = nope.is_contiguous() ? nope : nope.contiguous();
    at::Tensor rope_work = rope.is_contiguous() ? rope : rope.contiguous();
    at::Tensor slot_work = slot_mapping.is_contiguous() ? slot_mapping : slot_mapping.contiguous();
    if (slot_work.scalar_type() != at::kLong) {
        slot_work = slot_work.to(at::kLong);
    }
    const c10_npu::OptionalNPUGuard npuGuard(nope_work.device());
    EXEC_NPU_CMD(
        aclnnRqUniformPackSfa,
        nope_work,
        rope_work,
        slot_work,
        block_size,
        num_tokens,
        num_blocks,
        kv_cache);
}

}  // namespace openjiuwen_ascend_ops
