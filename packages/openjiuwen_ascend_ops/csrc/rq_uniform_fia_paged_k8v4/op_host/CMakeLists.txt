# Vendored/arch32 is on -I for FIA headers, but do NOT name any vendored file
# the same as AscendC headers (e.g. former kernel_common.h) — that shadows
# AscendC includes mid-kernel_operator.h and breaks matmul (FixpipeConfig/TPipe).
# Key decode scheme: 1=A, 2=B, 3=C; must match pack / attention (env BR_KEY_UNIFORM_SCHEME).
set(_BR_KEY_UNIFORM_SCHEME "1")
if(DEFINED ENV{BR_KEY_UNIFORM_SCHEME})
    set(_BR_KEY_UNIFORM_SCHEME "$ENV{BR_KEY_UNIFORM_SCHEME}")
endif()

# P29: s2Base=1024 + Cube L1_V_BUFCNT=4. Default on. Disable:
#   BR_S2_BASICSIZE_IS_1024=0 bash script/.../rebuild_op.sh
# Host tiling and Cube kernel must use the same value.
set(_BR_S2_BASICSIZE_IS_1024 "1")
if(DEFINED ENV{BR_S2_BASICSIZE_IS_1024})
    set(_BR_S2_BASICSIZE_IS_1024 "$ENV{BR_S2_BASICSIZE_IS_1024}")
endif()
message(STATUS "RqUniformFiaPagedK8v4 BR_S2_BASICSIZE_IS_1024=${_BR_S2_BASICSIZE_IS_1024}")

add_ops_compile_options(
        OP_NAME RqUniformFiaPagedK8v4
        OPTIONS --cce-auto-sync=on
                -DASCENDC_DUMP=0
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
                -DBR_KEY_UNIFORM_SCHEME=${_BR_KEY_UNIFORM_SCHEME}
                -DBR_S2_BASICSIZE_IS_1024=${_BR_S2_BASICSIZE_IS_1024}
                -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
                -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/vendored
                -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/vendored/arch32
                -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35
                -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35/common
                -I${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(op_host_aclnn PRIVATE
        rq_uniform_fia_paged_k8v4_def.cpp
)

target_sources(optiling PRIVATE
        rq_uniform_fia_paged_k8v4_tiling.cpp
        arch35/fused_infer_attention_score_tiling_impl.cpp
        arch35/fia_tiling_info.cpp
        arch35/rq_fia_a5_info_parser.cpp
        arch35/rq_fia_a5_checker.cpp
        arch35/rq_fia_a5_tiling.cpp
        ${CMAKE_SOURCE_DIR}/common_fia_split/split_core.cpp
        # split_core.cpp is the SOLE TU providing optiling::SplitCore and lives
        # in csrc/common_fia_split (compiled by oscar_fia_paged_k2v2). This op
        # must NOT also target_sources that .cpp or it double-defines the
        # symbols (multiple-definition link error).
)

# Host kS2BaseSize must match Cube S2_BASICSIZE_IS_1024 (add_ops_compile_options
# only covers the device kernel).
target_compile_definitions(optiling PRIVATE
        BR_S2_BASICSIZE_IS_1024=${_BR_S2_BASICSIZE_IS_1024}
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        rq_uniform_fia_paged_k8v4_tiling.cpp
        arch35/fused_infer_attention_score_tiling_impl.cpp
        arch35/fia_tiling_info.cpp
        arch35/rq_fia_a5_info_parser.cpp
        arch35/rq_fia_a5_checker.cpp
        arch35/rq_fia_a5_tiling.cpp
    )
    target_compile_definitions(opmaster_ct PRIVATE
            BR_S2_BASICSIZE_IS_1024=${_BR_S2_BASICSIZE_IS_1024}
    )
endif ()

# err/ops_err.h lives under kv_quant_sfa_v2 (shared host-log shim). A5 builds that
# only enable this op still need that include root.
target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35/common
        ${CMAKE_SOURCE_DIR}/common_fia_split
        ${CMAKE_SOURCE_DIR}/rq_uniform_kv_quant_sparse_flash_attention_v2/op_host
)

if (NOT BUILD_OPEN_PROJECT)
    target_include_directories(opmaster_ct PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35/common
        ${CMAKE_SOURCE_DIR}/common_fia_split
        ${CMAKE_SOURCE_DIR}/rq_uniform_kv_quant_sparse_flash_attention_v2/op_host
    )
endif ()

file(GLOB _BR_FIA_PAGED_K8V4_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_rq_uniform_fia_paged_k8v4.h")

install(FILES ${_BR_FIA_PAGED_K8V4_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
