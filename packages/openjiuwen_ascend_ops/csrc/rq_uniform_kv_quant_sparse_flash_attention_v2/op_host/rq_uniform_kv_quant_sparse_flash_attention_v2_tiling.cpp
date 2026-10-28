#include "register/op_def_registry.h"
#include "rq_uniform_kv_quant_sparse_flash_attention_tiling.h"

namespace optiling {
// Bind the shared MLA tiling struct to this op so the device compile emits it.
REGISTER_TILING_DATA_CLASS(RqUniformKvQuantSparseFlashAttentionV2,
                           RqUniformKvQuantSparseFlashAttentionTilingDataMla)

// Reuse the legacy tiling. The V2 kernel decodes the same key and selects
// DequantSfaRowsV2 via QSFAType<..., /*DEQUANT_V2=*/true>.
IMPL_OP_OPTILING(RqUniformKvQuantSparseFlashAttentionV2)
    .Tiling(TilingRqUniformKvQuantSparseFlashAttention)
    .TilingParse<RqUniformKvQuantSparseFlashAttentionCompileInfo>(
        TilingPrepareForRqUniformKvQuantSparseFlashAttention);
} // namespace optiling
