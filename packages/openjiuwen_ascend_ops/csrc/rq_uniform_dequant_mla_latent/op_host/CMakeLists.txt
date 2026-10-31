set(_MLA_DEQUANT_COMPILE_OPTIONS
        --cce-auto-sync=on
        -DASCENDC_DUMP=0
        -I${CMAKE_SOURCE_DIR}/third_party/catlass/include
        -Wno-deprecated-declarations
        -Werror
)

add_ops_compile_options(
        OP_NAME RqUniformDequantMlaLatent
        OPTIONS ${_MLA_DEQUANT_COMPILE_OPTIONS}
)

target_sources(op_host_aclnn PRIVATE
        rq_uniform_dequant_mla_latent_def.cpp
)

target_sources(optiling PRIVATE
        rq_uniform_dequant_mla_latent_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE)

file(GLOB _MLA_DEQUANT_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_rq_uniform_dequant_mla_latent.h")

install(FILES ${_MLA_DEQUANT_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
