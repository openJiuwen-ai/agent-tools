#!/bin/bash
# Build CANN custom ops for openjiuwen-ascend-ops.
# This script follows the same pattern as vllm-ascend's build_aclnn.sh:
#   1. Build the CANN cmake project (host + kernel)
#   2. Generate the .run self-installing package
#   3. Install to a local directory
#   4. Copy the .so libraries to csrc/ for setup.py linking
#   5. Copy the dynamic Python tiling adapters to pysrc/
#
# Usage:
#   bash build.sh [soc_version] [install_path]
#   Default: bash build.sh ascend910b <repo_root>/ztmp/cann_ops_install

set -e

CURRENT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
BUILD_DIR=${CURRENT_DIR}/build
OUTPUT_DIR=${CURRENT_DIR}/output
PACKAGE_DIR=$(dirname "${CURRENT_DIR}")  # packages/openjiuwen_ascend_ops
PYSRC_DIR="${PACKAGE_DIR}/pysrc/openjiuwentools/openjiuwen_ascend_ops"
CSRC_DIR="${CURRENT_DIR}"

SOC_VERSION=${1:-ascend910b}
INSTALL_PATH=${2:-$(dirname "$(dirname "${PACKAGE_DIR}")")/ztmp/cann_ops_install}

USER_ID=$(id -u)

if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
fi

if [ -n "${ASCEND_HOME_PATH}" ]; then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
elif [ -n "${ASCEND_OPP_PATH}" ]; then
    ASCEND_CANN_PACKAGE_PATH=$(dirname "${ASCEND_OPP_PATH}")
elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ]; then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
elif [ -d "${DEFAULT_INSTALL_DIR}" ]; then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
else
    echo "Error: Please set ASCEND_HOME_PATH or ensure CANN is installed." >&2
    exit 1
fi

CUSTOM_OPS="bit_residual_pack_k8v4"

# Fetch catlass dependency (git submodule, same pattern as vllm-ascend build_aclnn.sh)
CATLASS_PATH="${CSRC_DIR}/third_party/catlass/include"
if [ ! -d "${CATLASS_PATH}" ]; then
    echo "dependency catlass is missing, try to fetch it..."
    # Locate the repo root (the dir holding .gitmodules): csrc -> package -> packages -> repo
    REPO_ROOT=$(dirname "$(dirname "${PACKAGE_DIR}")")
    if ! git -C "${REPO_ROOT}" submodule update --init --recursive; then
        echo "Error: failed to fetch catlass submodule" >&2
        exit 1
    fi
    if [ ! -d "${CATLASS_PATH}" ]; then
        echo "Error: catlass dependency missing after fetch" >&2
        exit 1
    fi
fi
ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
export CPATH="${ABSOLUTE_CATLASS_PATH}${CPATH:+:${CPATH}}"

# Source CANN environment
source "${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash" || true

CPU_NUM=$(($(grep "^processor" /proc/cpuinfo | wc -l) * 2))
JOB_NUM="-j${CPU_NUM}"

# Clean and configure
rm -rf "${BUILD_DIR}" "${OUTPUT_DIR}"
mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

cd "${BUILD_DIR}"
cmake "${CSRC_DIR}" \
    -DBUILD_OPEN_PROJECT=ON \
    -DASCEND_COMPUTE_UNIT="${SOC_VERSION}" \
    -DASCEND_OP_NAME="${CUSTOM_OPS}" \
    -DVENDOR_NAME=vllm-ascend \
    -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
    -DCHECK_COMPATIBLE=true

# Build the package (host + kernel)
cmake --build "${BUILD_DIR}" --target package "${JOB_NUM}"

# Locate and install .run package
RUN_FILE=$(find "${BUILD_DIR}/_CPack_Packages" -name 'CANN-custom_ops-*.run' -type f | head -1)
if [ -z "${RUN_FILE}" ]; then
    echo "Error: .run installer not found" >&2
    find "${BUILD_DIR}" -name '*.run' -type f >&2 || true
    exit 1
fi

echo "Installing custom ops to ${INSTALL_PATH}..."
rm -rf "${INSTALL_PATH}"
bash "${RUN_FILE}" --install-path="${INSTALL_PATH}"

VENDOR_DIR="${INSTALL_PATH}/vendors/vllm-ascend"

# Copy .so libraries to csrc/ for setup.py linking
echo "Copying .so libraries to csrc/..."
cp -f "${VENDOR_DIR}/op_api/lib/libcust_opapi.so" "${CSRC_DIR}/"
cp -f "${VENDOR_DIR}/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so" "${CSRC_DIR}/"
cp -f "${VENDOR_DIR}/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so" "${CSRC_DIR}/"

# Also copy them into the package dir: the binding .so has RUNPATH $ORIGIN and
# the adapter resolves aclnn symbols via dlopen("libcust_opapi.so"), so the
# libs must sit next to _openjiuwen_ascend_ops*.so for the package to work without
# a manual ctypes preload (wheel package_data picks them up from here too).
echo "Copying .so libraries to pysrc/ package dir..."
cp -f "${CSRC_DIR}/libcust_opapi.so" "${PYSRC_DIR}/"
cp -f "${CSRC_DIR}/libcust_opmaster_rt2.0.so" "${PYSRC_DIR}/"
cp -f "${CSRC_DIR}/libcust_opsproto_rt2.0.so" "${PYSRC_DIR}/"

# Copy dynamic Python tiling adapters to pysrc/
echo "Copying Python tiling adapters to pysrc/..."
for op in bit_residual_pack_k8v4; do
    PY_FILE="${VENDOR_DIR}/op_impl/ai_core/tbe/dynamic/${op}.py"
    if [ -f "${PY_FILE}" ]; then
        cp -f "${PY_FILE}" "${PYSRC_DIR}/"
    fi
done

echo "CANN custom ops build and install completed successfully!"
echo "  Install path: ${INSTALL_PATH}"
echo "  Vendor dir: ${VENDOR_DIR}"
echo "  .so in csrc/: libcust_opapi.so, libcust_opmaster_rt2.0.so, libcust_opsproto_rt2.0.so"
