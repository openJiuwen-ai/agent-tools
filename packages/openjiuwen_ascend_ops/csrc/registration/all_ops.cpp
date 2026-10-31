#include "registration.h"
#include "../rq_uniform_pack_k8v4/rq_uniform_pack_k8v4_adapter.h"
#include "../rq_uniform_pack_k4v4/rq_uniform_pack_k4v4_adapter.h"
#include "../rq_uniform_fia_paged_k8v4/rq_uniform_fia_paged_k8v4_adapter.h"
#include "../rq_uniform_fia_paged_k4v4/rq_uniform_fia_paged_k4v4_adapter.h"
#include "../rq_uniform_fia_paged_k8v4_msd/rq_uniform_fia_paged_k8v4_msd_adapter.h"
#include "../rq_uniform_fia_paged_k4v4_msd/rq_uniform_fia_paged_k4v4_msd_adapter.h"
#include "../rq_uniform_pack_mla_latent/rq_uniform_pack_mla_latent_adapter.h"
#include "../rq_uniform_pack_sfa/rq_uniform_pack_sfa_adapter.h"
#include "../rq_uniform_kv_quant_sparse_flash_attention_v2/rq_uniform_kv_quant_sparse_flash_attention_v2_adapter.h"
#include "../rq_uniform_dequant_mla_latent/rq_uniform_dequant_mla_latent_adapter.h"
#include "../rq_uniform_fia_mla_latent/rq_uniform_fia_mla_latent_adapter.h"
#include <torch/library.h>
#include <ATen/core/op_registration/op_registration.h>

namespace openjiuwen_ascend_ops {

static std::unique_ptr<c10::RegisterOperators> g_registry = nullptr;

void register_all_ops() {
    if (g_registry != nullptr) return;

    g_registry = std::make_unique<c10::RegisterOperators>();

    (*g_registry)
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_pack_k8v4(Tensor key, Tensor value, Tensor slot_mapping, Tensor query_start_loc, Tensor rotation_t, Tensor key_cache, Tensor value_cache, int num_reqs, int block_size) -> ()")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_pack_k8v4))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_pack_k4v4(Tensor key, Tensor value, Tensor slot_mapping, Tensor query_start_loc, Tensor rotation_t, Tensor key_cache, Tensor value_cache, int num_reqs, int block_size) -> ()")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_pack_k4v4))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_fia_paged_k8v4(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, int[] actual_seq_len_q, int[] actual_seq_len_kv, Tensor? atten_mask, Tensor rotation_key, Tensor rotation_value, int num_heads, int num_kv_heads, int head_size, int block_size, float scale_value, int pre_tokens, int next_tokens, int sparse_mode, Tensor? out=None) -> Tensor")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_fia_paged_k8v4))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_fia_paged_k4v4(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, int[] actual_seq_len_q, int[] actual_seq_len_kv, Tensor? atten_mask, Tensor rotation_key, Tensor rotation_value, int num_heads, int num_kv_heads, int head_size, int block_size, float scale_value, int pre_tokens, int next_tokens, int sparse_mode, Tensor? out=None) -> Tensor")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_fia_paged_k4v4))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_fia_paged_k8v4_msd(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, int[] actual_seq_len_q, int[] actual_seq_len_kv, Tensor? atten_mask, Tensor rotation_key, Tensor rotation_value, int num_heads, int num_kv_heads, int head_size, int block_size, float scale_value, int pre_tokens, int next_tokens, int sparse_mode, Tensor? out=None) -> Tensor")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_fia_paged_k8v4_msd))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_fia_paged_k4v4_msd(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, int[] actual_seq_len_q, int[] actual_seq_len_kv, Tensor? atten_mask, Tensor rotation_key, Tensor rotation_value, int num_heads, int num_kv_heads, int head_size, int block_size, float scale_value, int pre_tokens, int next_tokens, int sparse_mode, Tensor? out=None) -> Tensor")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_fia_paged_k4v4_msd))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_pack_mla_latent(Tensor c, Tensor rotation_t, Tensor slot_mapping, Tensor latent_cache, int block_size) -> ()")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_pack_mla_latent))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_pack_sfa(Tensor nope, Tensor rope, Tensor slot_mapping, Tensor kv_cache, int block_size) -> ()")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_pack_sfa))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_kv_quant_sparse_flash_attention_v2(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_kv, float scale_value, int key_quant_mode, int value_quant_mode, int sparse_block_size, str layout_query, str layout_kv, int sparse_mode, int attention_mode, int quant_scale_repo_mode, int tile_size, int rope_head_dim, Tensor? out=None) -> Tensor")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_kv_quant_sparse_flash_attention_v2))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_dequant_mla_latent(Tensor latent_cache, Tensor rotation, Tensor block_table, Tensor context_lens, Tensor seq_starts, int block_size, int total_tokens) -> Tensor")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_dequant_mla_latent))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_fia_mla_latent(Tensor ql_nope, Tensor q_pe, Tensor codes, Tensor scales, Tensor k_pe, Tensor rotation_t, Tensor rotation, float scale) -> Tensor")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_fia_mla_latent))
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::rq_uniform_fia_mla_latent_paged(Tensor q_nope, Tensor q_pe, Tensor latent_cache, Tensor k_pe_cache, Tensor block_table, int[] actual_seq_q, int[] actual_seq_kv, Tensor w_uk_t, Tensor w_uv, Tensor rotation, int block_size, float scale, int num_tokens, int total_kv, int max_kv, Tensor? out=None) -> Tensor")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::rq_uniform_fia_mla_latent_paged));
}

}
