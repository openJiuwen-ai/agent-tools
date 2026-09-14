#include "registration.h"
#include "../bit_residual_pack_k8v4/bit_residual_pack_k8v4_adapter.h"
#include <torch/library.h>
#include <ATen/core/op_registration/op_registration.h>

namespace openjiuwen_ascend_ops {

static std::unique_ptr<c10::RegisterOperators> g_registry = nullptr;

void register_all_ops() {
    if (g_registry != nullptr) return;

    g_registry = std::make_unique<c10::RegisterOperators>();

    (*g_registry)
        .op(c10::RegisterOperators::options()
            .schema("openjiuwen::bit_residual_pack_k8v4(Tensor key, Tensor value, Tensor slot_mapping, Tensor query_start_loc, Tensor rotation_t, Tensor key_cache, Tensor value_cache, int num_reqs, int block_size) -> ()")
            .kernel(c10::DispatchKey::PrivateUse1, &openjiuwen_ascend_ops::bit_residual_pack_k8v4));
}

}
