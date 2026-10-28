set(_TQ_PACK_COMPILE_OPTIONS
        --cce-auto-sync=on
        -DASCENDC_DUMP=0
        -I${CMAKE_SOURCE_DIR}/third_party/catlass/include
        -Wno-deprecated-declarations
        -Werror
)

# Key encode scheme: 1=A asymmetric, 2=B symmetric, 3=C legacy LSB-sign+q7.
# Must match FIA / attention BR_KEY_UNIFORM_SCHEME (env or default 1).
set(_BR_KEY_UNIFORM_SCHEME "1")
if(DEFINED ENV{BR_KEY_UNIFORM_SCHEME})
    set(_BR_KEY_UNIFORM_SCHEME "$ENV{BR_KEY_UNIFORM_SCHEME}")
endif()
list(APPEND _TQ_PACK_COMPILE_OPTIONS -DBR_KEY_UNIFORM_SCHEME=${_BR_KEY_UNIFORM_SCHEME})

set(RQ_UNIFORM_PACK_K4V4_DEBUG_LINE "OFF" CACHE STRING
        "Enable -g for RqUniformPackK4v4 AscendC kernel compile.")

set(_TQ_PACK_DEBUG_LINE "${RQ_UNIFORM_PACK_K4V4_DEBUG_LINE}")
if(DEFINED ENV{RQ_UNIFORM_PACK_K4V4_DEBUG_LINE})
    set(_TQ_PACK_DEBUG_LINE "$ENV{RQ_UNIFORM_PACK_K4V4_DEBUG_LINE}")
endif()
string(TOUPPER "${_TQ_PACK_DEBUG_LINE}" _TQ_PACK_DEBUG_LINE)
if(_TQ_PACK_DEBUG_LINE STREQUAL "ON" OR _TQ_PACK_DEBUG_LINE STREQUAL "1" OR
        _TQ_PACK_DEBUG_LINE STREQUAL "TRUE")
    list(APPEND _TQ_PACK_COMPILE_OPTIONS -g)
endif()

add_ops_compile_options(
        OP_NAME RqUniformPackK4v4
        OPTIONS ${_TQ_PACK_COMPILE_OPTIONS}
)

target_sources(op_host_aclnn PRIVATE
        rq_uniform_pack_k4v4_def.cpp
)

target_sources(optiling PRIVATE
        rq_uniform_pack_k4v4_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
)

target_sources(opsproto PRIVATE)

file(GLOB _TQ_PACK_TO_CACHE_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_rq_uniform_pack_k4v4.h")

install(FILES ${_TQ_PACK_TO_CACHE_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
