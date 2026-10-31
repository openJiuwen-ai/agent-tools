set(_SFA_KV_V2_COMPILE_OPTIONS
        --cce-auto-sync=off
        -DASCENDC_DUMP=0
        -Wno-deprecated-declarations
        -Werror
)

add_ops_compile_options(
        OP_NAME RqUniformKvQuantSparseFlashAttentionV2
        OPTIONS ${_SFA_KV_V2_COMPILE_OPTIONS}
)

target_sources(op_host_aclnn PRIVATE
        rq_uniform_kv_quant_sparse_flash_attention_v2_def.cpp
)

# Shared MLA tiling/infershape (formerly in rq_uniform_kv_quant_sparse_flash_attention/)
# plus the V2 registration wrappers.
target_sources(optiling PRIVATE
        rq_uniform_kv_quant_sparse_flash_attention_tiling.cpp
        rq_uniform_kv_quant_sparse_flash_attention_v2_tiling.cpp
)

target_sources(opsproto PRIVATE
        rq_uniform_kv_quant_sparse_flash_attention_infershape.cpp
        rq_uniform_kv_quant_sparse_flash_attention_v2_infershape.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB _SFA_KV_V2_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_rq_uniform_kv_quant_sparse_flash_attention_v2.h")

install(FILES ${_SFA_KV_V2_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
