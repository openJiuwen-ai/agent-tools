#ifndef RQ_UNIFORM_FIA_MLA_LATENT_ADAPTER_H
#define RQ_UNIFORM_FIA_MLA_LATENT_ADAPTER_H

#include <torch/extension.h>

namespace openjiuwen_ascend_ops {

// Dense ATen golden (codes*scales, ql already W_UK-absorbed). UT / reference only.
// Product path is the paged fused AscendC op below.
at::Tensor rq_uniform_fia_mla_latent(
    const at::Tensor& ql_nope,
    const at::Tensor& q_pe,
    const at::Tensor& codes,
    const at::Tensor& scales,
    const at::Tensor& k_pe,
    const at::Tensor& rotation_t,
    const at::Tensor& rotation,
    double scale);

// Fused AscendC decode absorb (FIA MLA 910B + Q/V absorb + Scheme A TQ).
//
//   q_nope [Tq,H,P] --W_UK--> ql --R^T--> ql'
//   latent_cache Scheme A --dequant--> c  (also used as V)
//   FIA MLA(ql', q_pe, c, k_pe) --> o'
//   o' --R--> o --W_UV--> out [Tq,H,V]
//
// actual_seq_q / actual_seq_kv: host int[] (same as Full TQ MSD / native FIA
// actual_seq_qlen/kvlen). ACL-graph update rebinds these lists without H2D.
// max_kv <= 0 returns y [sum(actual_seq_kv), 512]. max_kv > 0 runs FIA on
// the dequanted y tile and returns latent context [H, Tq, 512].
// Optional ``out`` must match the return layout for in-place write (graph).
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
    c10::optional<at::Tensor> out_opt = c10::nullopt);

}  // namespace openjiuwen_ascend_ops

#endif
