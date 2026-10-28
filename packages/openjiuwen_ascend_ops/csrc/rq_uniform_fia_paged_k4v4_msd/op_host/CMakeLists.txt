# RqUniformFiaPagedK4v4Msd — Decode MSD (Us2 antiquant + BR pack/rotate).
# Kernel + host tiling: open-source IFA Us2 skeleton.
# Do NOT compile split_core.cpp here (sole TU lives in csrc/common_fia_split,

# Product defaults: BR pack gather + Haar Q@Π/O@Π.
# Us2-native PA A/B: BR_MSD_US2_PACK_GATHER=0 (and typically APPLY_PI=0).
set(_BR_MSD_US2_PACK_GATHER "1")
if(DEFINED ENV{BR_MSD_US2_PACK_GATHER})
    set(_BR_MSD_US2_PACK_GATHER "$ENV{BR_MSD_US2_PACK_GATHER}")
endif()
set(_BR_MSD_US2_APPLY_PI "1")
if(DEFINED ENV{BR_MSD_US2_APPLY_PI})
    set(_BR_MSD_US2_APPLY_PI "$ENV{BR_MSD_US2_APPLY_PI}")
endif()
# Pack-tail meta gather: 1=MTE2↔V ping-pong when nRows>=256 (product default);
# 0=serial single-buffer A/B.
set(_BR_MSD_US2_META_PINGPONG "1")
if(DEFINED ENV{BR_MSD_US2_META_PINGPONG})
    set(_BR_MSD_US2_META_PINGPONG "$ENV{BR_MSD_US2_META_PINGPONG}")
endif()

message(STATUS "RqUniformFiaPagedK4v4Msd BR_MSD_US2_PACK_GATHER=${_BR_MSD_US2_PACK_GATHER}")
message(STATUS "RqUniformFiaPagedK4v4Msd BR_MSD_US2_APPLY_PI=${_BR_MSD_US2_APPLY_PI}")
message(STATUS "RqUniformFiaPagedK4v4Msd BR_MSD_US2_META_PINGPONG=${_BR_MSD_US2_META_PINGPONG}")

set(_MSD_KERN "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel")

set(_MSD_COMPILE_DEFS
        -DBR_MSD_US2_PACK_GATHER=${_BR_MSD_US2_PACK_GATHER}
        -DBR_MSD_US2_APPLY_PI=${_BR_MSD_US2_APPLY_PI}
        -DBR_MSD_US2_META_PINGPONG=${_BR_MSD_US2_META_PINGPONG}
)

set(_MSD_COMPILE_INCLUDES
        -I${_MSD_KERN}
        -I${_MSD_KERN}/vendored/arch32
        -I${_MSD_KERN}/vendored/us2
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -I${CMAKE_CURRENT_SOURCE_DIR}/vendored/us2
)

add_ops_compile_options(
        OP_NAME RqUniformFiaPagedK4v4Msd
        OPTIONS --cce-auto-sync=on
                -DASCENDC_DUMP=0
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
                ${_MSD_COMPILE_DEFS}
                ${_MSD_COMPILE_INCLUDES}
)

set(_MSD_TILING_KEYS
        10000000000200300
        12000000000200300
        10000000000300300
)
if(_BR_MSD_US2_PACK_GATHER STREQUAL "1")
    # ANTIPERTOKEN PAGEDCACHE C1V2 / C1V1 / FD C1V2 — KVINT4 (K4V4)
    set(_MSD_TILING_KEYS
            10000000000600400
            12000000000600400
            10000000000700400
    )
endif()

add_ops_tiling_keys(
        OP_NAME RqUniformFiaPagedK4v4Msd
        TILING_KEYS ${_MSD_TILING_KEYS}
)

target_sources(op_host_aclnn PRIVATE
        rq_uniform_fia_paged_k4v4_msd_def.cpp
)

target_sources(optiling PRIVATE rq_uniform_fia_paged_k4v4_msd_us2_tiling.cpp)

target_compile_definitions(optiling PRIVATE
        BR_MSD_US2_PACK_GATHER=${_BR_MSD_US2_PACK_GATHER}
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE rq_uniform_fia_paged_k4v4_msd_us2_tiling.cpp)
    target_compile_definitions(opmaster_ct PRIVATE
            BR_MSD_US2_PACK_GATHER=${_BR_MSD_US2_PACK_GATHER}
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/vendored/us2
        ${_MSD_KERN}
)

file(GLOB _BR_FIA_PAGED_K4V4_MSD_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_rq_uniform_fia_paged_k4v4_msd.h")

install(FILES ${_BR_FIA_PAGED_K4V4_MSD_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
