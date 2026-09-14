import importlib
import os
import sys

import torch

try:
    import torch_npu
except ImportError as e:
    if "libhccl.so" in str(e) or "libascend_hal.so" in str(e):
        raise RuntimeError(
            "Failed to import torch_npu. Make sure the ASCEND environment variables are properly set:\n"
            "  export ASCEND_HOME=/usr/local/Ascend\n"
            "  export ASCEND_HOME_PATH=$ASCEND_HOME/cann-8.5.1\n"
            "  export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/aarch64-linux/lib64:"
            "$ASCEND_HOME/driver/lib64/driver:$LD_LIBRARY_PATH\n"
            "  export TORCH_DEVICE_BACKEND_AUTOLOAD=0\n"
            f"Original error: {e}"
        ) from e
    raise

package_dir = os.path.dirname(__file__)

cann_ops_custom_path = os.path.join(package_dir, "_cann_ops_custom", "vendors", "vllm-ascend")
if os.path.exists(cann_ops_custom_path):
    current_custom_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if cann_ops_custom_path not in current_custom_opp:
        if current_custom_opp:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = cann_ops_custom_path + ":" + current_custom_opp
        else:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = cann_ops_custom_path

    op_api_lib_path = os.path.join(cann_ops_custom_path, "op_api", "lib")
    if op_api_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = op_api_lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

if package_dir not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = package_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

try:
    from . import _openjiuwen_ascend_ops
    _openjiuwen_ascend_ops.register_ops()
except ImportError as e:
    raise RuntimeError(
        f"Failed to import _openjiuwen_ascend_ops. Make sure the package is built correctly. "
        f"Original error: {e}"
    ) from e

from .ops import bit_residual_pack_k8v4

__all__ = [
    "bit_residual_pack_k8v4",
]
