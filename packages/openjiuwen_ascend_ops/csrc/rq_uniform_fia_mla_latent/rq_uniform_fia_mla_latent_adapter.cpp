#include "rq_uniform_fia_mla_latent_adapter.h"

#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

#include "../aclnn_torch_adapter/op_api_common.h"

namespace openjiuwen_ascend_ops {

at::Tensor rq_uniform_fia_mla_latent(
    const at::Tensor& ql_nope,
    const at::Tensor& q_pe,
    const at::Tensor& codes,
    const at::Tensor& scales,
    const at::Tensor& k_pe,
    const at::Tensor& rotation_t,
    const at::Tensor& rotation,
    double scale) {
    TORCH_CHECK(ql_nope.dim() == 3, "ql_nope must be [Tq, H, L]");
    TORCH_CHECK(codes.dim() == 2 && codes.scalar_type() == at::kChar, "codes [Tkv, L] int8");
    const int64_t L = codes.size(1);
    const int64_t H = ql_nope.size(1);
    TORCH_CHECK(ql_nope.size(2) == L, "ql_nope last dim must equal latent L");
    TORCH_CHECK(rotation_t.sizes() == at::IntArrayRef({L, L}), "rotation_t [L, L]");
    TORCH_CHECK(rotation.sizes() == at::IntArrayRef({L, L}), "rotation [L, L]");

    auto c_rot = codes.to(at::kFloat) * scales.to(at::kFloat).unsqueeze(-1);
    auto ql = at::matmul(ql_nope.to(at::kFloat), rotation_t.to(at::kFloat));
    auto ql_h = ql.transpose(0, 1);
    auto c_h = c_rot.unsqueeze(0).expand({H, c_rot.size(0), L});
    auto q_pe_h = q_pe.to(at::kFloat).transpose(0, 1);
    auto k_pe_h = k_pe.to(at::kFloat).unsqueeze(0).expand({H, k_pe.size(0), k_pe.size(-1)});

    const double dim = static_cast<double>(L + q_pe.size(-1));
    const double sc = scale > 0 ? scale : (1.0 / std::sqrt(dim));
    auto scores = at::matmul(ql_h, c_h.transpose(-1, -2)) +
                  at::matmul(q_pe_h, k_pe_h.transpose(-1, -2));
    scores = scores * sc;
    auto probs = at::softmax(scores, -1);
    auto out = at::matmul(probs, c_h).transpose(0, 1);
    out = at::matmul(out, rotation.to(at::kFloat));
    return out.to(ql_nope.scalar_type());
}

at::Tensor rq_uniform_fia_mla_latent_paged(
    const at::Tensor& q_nope,
    const at::Tensor& q_pe,
    const at::Tensor& latent_cache,
    const at::Tensor& k_pe_cache,
    const at::Tensor& block_table,
    at::IntArrayRef actual_seq_q,
    at::IntArrayRef actual_seq_kv,
    const at::Tensor& w_uk_t,
    const at::Tensor& w_uv,
    const at::Tensor& rotation,
    int64_t block_size,
    double scale,
    int64_t num_tokens,
    int64_t total_kv,
    int64_t max_kv,
    c10::optional<at::Tensor> out_opt) {
    constexpr int64_t kLatent = 512;
    constexpr int64_t kRope = 64;
    constexpr int64_t kVectorTile = 16;

    TORCH_CHECK(q_nope.dim() == 3, "q_nope must be [Tq, H, P]");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(-1) == kRope, "q_pe must be [Tq, H, 64]");
    TORCH_CHECK(w_uk_t.dim() == 3 && w_uk_t.size(-1) == kLatent, "w_uk_t must be [H, P, 512]");
    TORCH_CHECK(w_uv.dim() == 3 && w_uv.size(1) == kLatent, "w_uv must be [H, 512, V]");
    TORCH_CHECK(rotation.dim() == 2 && rotation.size(0) == kLatent && rotation.size(1) == kLatent,
                "rotation must be [512, 512]");
    TORCH_CHECK(latent_cache.scalar_type() == at::kByte, "latent_cache must be uint8 Scheme A");
    TORCH_CHECK(block_size > 0 && block_size % kVectorTile == 0,
                "block_size must be a positive multiple of 16");
    TORCH_CHECK(num_tokens > 0 && num_tokens == q_nope.size(0), "num_tokens must equal Tq");

    const int64_t num_heads = q_nope.size(1);
    const int64_t nope_dim = q_nope.size(2);
    const int64_t v_dim = w_uv.size(2);
    const int64_t num_blocks = latent_cache.size(0);
    const int64_t num_reqs = block_table.size(0);
    const int64_t max_blocks = block_table.size(1);
    // W_UK / W_UV are not read by the current kernel. Heads may be 1 on the
    // dequant golden, which passes placeholder weights.
    TORCH_CHECK(w_uk_t.dim() == 3 && w_uv.dim() == 3, "w_uk_t and w_uv must be rank-3");
    (void)nope_dim;
    TORCH_CHECK(actual_seq_q.size() == static_cast<size_t>(num_reqs),
                "actual_seq_q must be host int[B] TND accum-Q");
    TORCH_CHECK(actual_seq_kv.size() == static_cast<size_t>(num_reqs),
                "actual_seq_kv must be host int[B]");
    TORCH_CHECK(latent_cache.size(1) == block_size * (kLatent + 4),
                "latent_cache stride must be block_size*(512+4)");
    TORCH_CHECK(total_kv > 0, "total_kv must be positive");

    at::Tensor q_work = q_nope.is_contiguous() ? q_nope : q_nope.contiguous();
    at::Tensor q_pe_work = q_pe.is_contiguous() ? q_pe : q_pe.contiguous();
    at::Tensor cache_work = latent_cache.is_contiguous() ? latent_cache : latent_cache.contiguous();
    at::Tensor k_pe_work = k_pe_cache.is_contiguous() ? k_pe_cache : k_pe_cache.contiguous();
    at::Tensor table_work = (block_table.scalar_type() == at::kInt && block_table.is_contiguous())
                                ? block_table
                                : block_table.to(at::kInt).contiguous();
    at::Tensor uk_work = w_uk_t.is_contiguous() ? w_uk_t : w_uk_t.contiguous();
    at::Tensor uv_work = w_uv.is_contiguous() ? w_uv : w_uv.contiguous();
    at::Tensor rot_work = rotation.is_contiguous() ? rotation : rotation.contiguous();
    const bool fuse_fia = max_kv > 0;
    if (fuse_fia) {
        TORCH_CHECK(nope_dim == kLatent, "fused MLA requires latent 512");
        TORCH_CHECK(num_tokens % num_reqs == 0, "fused MLA requires a uniform query length");
        TORCH_CHECK((num_tokens / num_reqs) * num_heads <= 256, "fused MLA query group exceeds one S1 tile");
        // Path B: FIA sees ql' and y. Haar stays outside the MLA kernel.
        q_work = at::matmul(q_work, rot_work.transpose(0, 1));
        if (k_pe_work.dim() == 4 && k_pe_work.size(1) == 1) {
            k_pe_work = k_pe_work.reshape({k_pe_work.size(0), k_pe_work.size(2), k_pe_work.size(3)});
        } else if (!(k_pe_work.dim() == 3 && k_pe_work.size(1) == block_size && k_pe_work.size(2) == kRope)) {
            k_pe_work = k_pe_work.reshape({-1, block_size, kRope});
        }
    }
    at::Tensor fia_out = fuse_fia ? at::empty({num_tokens, num_heads, kLatent}, q_nope.options())
                                  : at::empty({total_kv, kLatent}, q_nope.options());

    const c10_npu::OptionalNPUGuard npuGuard(q_work.device());
    // Host int[] → aclIntArray (ValueDepend), same path as Full TQ MSD.
    EXEC_NPU_CMD(
        aclnnRqUniformFiaMlaLatent,
        q_work,
        q_pe_work,
        cache_work,
        k_pe_work,
        table_work,
        actual_seq_q,
        actual_seq_kv,
        uk_work,
        uv_work,
        rot_work,
        block_size,
        scale,
        num_tokens,
        num_heads,
        nope_dim,
        v_dim,
        num_blocks,
        num_reqs,
        max_blocks,
        max_kv,
        fia_out);
    if (!fuse_fia) {
        if (out_opt.has_value() && out_opt->defined()) {
            TORCH_CHECK(out_opt->sizes().equals(fia_out.sizes()), "out shape must match dequant y");
            TORCH_CHECK(out_opt->scalar_type() == fia_out.scalar_type(), "out dtype mismatch");
            TORCH_CHECK(out_opt->device() == fia_out.device(), "out device mismatch");
            out_opt->copy_(fia_out);
            return *out_opt;
        }
        return fia_out;
    }
    // Path-B unfold: o = o' @ R, layout [H, Tq, L] for W_UV.
    auto unfolded = at::matmul(fia_out, rot_work);
    auto result = unfolded.transpose(0, 1).contiguous();
    if (out_opt.has_value() && out_opt->defined()) {
        TORCH_CHECK(out_opt->sizes().equals(result.sizes()),
                    "out must be [H, Tq, L] for fused absorb");
        TORCH_CHECK(out_opt->scalar_type() == result.scalar_type(), "out dtype mismatch");
        TORCH_CHECK(out_opt->device() == result.device(), "out device mismatch");
        TORCH_CHECK(out_opt->is_contiguous(), "out must be contiguous for in-place write");
        out_opt->copy_(result);
        return *out_opt;
    }
    return result;
}

}  // namespace openjiuwen_ascend_ops
