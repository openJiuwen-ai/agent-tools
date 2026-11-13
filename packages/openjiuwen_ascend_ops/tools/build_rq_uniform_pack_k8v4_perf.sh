#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_DIR=$(cd "${PKG_DIR}/../.." && pwd)
OUT=${1:-"${REPO_DIR}/ztmp/rq_uniform_pack_k8v4_perf/rq_uniform_pack_k8v4_perf"}

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_PATH}/include" && -d /usr/local/Ascend/cann-8.5.1 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
fi
if [[ ! -d "${ASCEND_HOME_PATH}/include" && -d /usr/local/Ascend/cann-9.2.0 ]]; then
    ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
fi

OPP_VENDOR_DIR=${OPP_VENDOR_DIR:-"${REPO_DIR}/ztmp/cann_ops_install/vendors/vllm-ascend"}
CUSTOM_OP_LIB="${OPP_VENDOR_DIR}/op_api/lib"
HEADER_DIR="${PKG_DIR}/csrc/rq_uniform_pack_k8v4/op_host"

if [[ ! -f "${CUSTOM_OP_LIB}/libcust_opapi_jw.so" ]]; then
    echo "missing ${CUSTOM_OP_LIB}/libcust_opapi_jw.so" >&2
    echo "build and install the custom op first, or set OPP_VENDOR_DIR" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT}")"

NNOPBASE_LIB=()
if [[ -f "${ASCEND_HOME_PATH}/lib64/libnnopbase.so" ]]; then
    NNOPBASE_LIB=(-lnnopbase)
fi

g++ -std=c++17 -O2 -g -fno-omit-frame-pointer -rdynamic \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${HEADER_DIR}" \
    "${SCRIPT_DIR}/rq_uniform_pack_k8v4_perf.cpp" \
    -L"${ASCEND_HOME_PATH}/lib64" \
    -L"${CUSTOM_OP_LIB}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${CUSTOM_OP_LIB}" \
    -Wl,-rpath-link,"${ASCEND_HOME_PATH}/lib64:${CUSTOM_OP_LIB}" \
    -lcust_opapi_jw -lopapi "${NNOPBASE_LIB[@]}" -lascendcl -ldl \
    -o "${OUT}"

echo "Built ${OUT}"
