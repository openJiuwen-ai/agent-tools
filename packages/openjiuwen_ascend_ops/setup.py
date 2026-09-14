import os
import sys
from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CSRC_DIR = "csrc"
PYSRC_DIR = "pysrc"

CANN_HOME = os.environ.get(
    "ASCEND_TOOLKIT_HOME",
    os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"),
)
ASCEND_ROOT = os.environ.get("ASCEND_HOME", os.path.dirname(CANN_HOME))

torch_npu_include_dir = ""
for path in sys.path:
    torch_npu_path = os.path.join(path, "torch_npu", "include")
    if os.path.exists(torch_npu_path) and os.path.isdir(torch_npu_path):
        torch_npu_include_dir = torch_npu_path
        break

include_dirs = [CSRC_DIR]
if torch_npu_include_dir:
    include_dirs.append(torch_npu_include_dir)

ascend_include_dir = CANN_HOME
if os.path.exists(os.path.join(ascend_include_dir, "aarch64-linux", "include")):
    include_dirs.append(os.path.join(ascend_include_dir, "aarch64-linux", "include"))

ext_modules = []

ext_modules.append(
    CppExtension(
        "openjiuwentools.openjiuwen_ascend_ops._openjiuwen_ascend_ops",
        [
            os.path.join(CSRC_DIR, "binding", "pybind_binding.cpp"),
            os.path.join(CSRC_DIR, "registration", "all_ops.cpp"),
            os.path.join(CSRC_DIR, "aclnn_torch_adapter", "NPUBridge.cpp"),
            os.path.join(CSRC_DIR, "bit_residual_pack_k8v4", "bit_residual_pack_k8v4_adapter.cpp"),
        ],
        include_dirs=include_dirs,
        extra_compile_args=["-std=c++17"],
        extra_link_args=[
            "-L{}".format(CSRC_DIR),
            "-lcust_opapi",
            "-L{}/aarch64-linux/lib64".format(CANN_HOME),
            "-lacl_rt",
            "-L{}".format(os.path.join(torch_npu_include_dir, "..", "lib")),
            "-ltorch_npu",
            "-Wl,-rpath=$ORIGIN",
            "-Wl,-rpath={}/aarch64-linux/lib64".format(CANN_HOME),
            "-Wl,-rpath={}/driver/lib64/driver".format(ASCEND_ROOT),
            "-Wl,-rpath,{}".format(os.path.join(torch_npu_include_dir, "..", "lib")),
        ],
    )
)

setup(
    name="openjiuwentools-openjiuwen-ascend-ops",
    version="0.1.0",
    description="AscendC operators plugin for OpenJiuwenTools",
    long_description=(
        "OpenJiuwenTools Ascend affinity package, providing AscendC operator plugins "
        "and integration utilities for running OpenJiuwenTools workloads on Ascend NPU."
    ),
    long_description_content_type="text/markdown",
    url="https://gitcode.com/openJiuwen/agent-tools",
    packages=find_packages(where=PYSRC_DIR),
    package_dir={"": PYSRC_DIR},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.9.0",
        "torch-npu>=2.9.0",
    ],
    setup_requires=[
        "setuptools>=61.0",
        "wheel",
    ],
    package_data={
        "openjiuwentools.openjiuwen_ascend_ops": [
            "libcust_opapi.so",
            "libcust_opmaster_rt2.0.so",
            "libcust_opsproto_rt2.0.so",
            "bit_residual_pack_k8v4.py",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
