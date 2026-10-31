set(_SFA_PACK_COMPILE_OPTIONS
        --cce-auto-sync=on
        -DASCENDC_DUMP=0
        -Wno-deprecated-declarations
        -Werror
)

add_ops_compile_options(
        OP_NAME RqUniformPackSfa
        OPTIONS ${_SFA_PACK_COMPILE_OPTIONS}
)

target_sources(op_host_aclnn PRIVATE
        rq_uniform_pack_sfa_def.cpp
)

target_sources(optiling PRIVATE
        rq_uniform_pack_sfa_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE)

file(GLOB _SFA_PACK_Aclnn_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_rq_uniform_pack_sfa.h")

install(FILES ${_SFA_PACK_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
