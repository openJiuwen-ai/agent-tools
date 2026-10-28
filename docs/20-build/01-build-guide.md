| 字段 | 值 |
| --- | --- |
| 标题 | 构建指南 |
| 修改日期 | 2026-09-14 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/build.sh, setup.py, csrc/CMakeLists.txt |
| 关联文档 | 00-index.md, 20-build/00-environment.md, 50-debugging/00-sanitizer.md, 60-profiling/00-profiling.md |

# 构建指南

构建分两层：**CANN 自定义算子工程** 产出三个 `.so` + 内核二进制 + Python tiling adapter；**setuptools** 把 C++ Torch 扩展编译为 `_openjiuwen_ascend_ops.*.so` 并打 wheel。两层产物在 `setup.py` 链接阶段汇合。

前置：先完成 [环境准备](00-environment.md)。

## 阶段 A：CANN 自定义算子工程

入口脚本 `packages/openjiuwen_ascend_ops/csrc/build.sh`。流程：

1. **解析 CANN 路径**：`ASCEND_HOME_PATH` → `ASCEND_OPP_PATH` 父目录 → 默认 toolkit 路径；root/非 root 用户分桶。
2. **拉 catlass 并注入 CPATH**：catlass 是 git submodule（`gitcode.com/cann/catlass.git`，branch `catlass-v1-stable`，pin `716fd7b`，与 vllm-ascend 一致）；`include/` 缺失时从仓库根 `git submodule update --init --recursive`；随后 `export CPATH=<catlass>/include:$CPATH`，让 AscendC 编译器在 host/kernel 编译时都能 include。
3. **source CANN 环境**：`source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash`。
4. **CMake configure**：
   ```bash
   cmake csrc/ -DBUILD_OPEN_PROJECT=ON \
     -DASCEND_COMPUTE_UNIT=ascend910b \
     -DASCEND_OP_NAME="rq_uniform_pack_k8v4" \
     -DVENDOR_NAME=vllm-ascend \
     -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=<CANN> \
     -DCHECK_COMPATIBLE=true
   ```
   `BUILD_OPEN_PROJECT=ON` 是关键开关，让 `csrc/CMakeLists.txt` 走"自定义算子包"分支（生成 `.run` + 独立 vendor 目录）。`ASCEND_OP_NAME` 由 `build.sh` 里的 `CUSTOM_OPS` 列表传入。
5. **构建 + 打 .run 自安装包**：`cmake --build build --target package`。CPack 用 `External` 生成器产出 `CANN-custom_ops-<cann_ver>-linux.aarch64.run`。
6. **安装 .run 到本地**：`bash <run> --install-path=ztmp/cann_ops_install`。安装目录结构：
   ```
   ztmp/cann_ops_install/vendors/vllm-ascend/
     op_api/lib/libcust_opapi.so
     op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
     op_impl/ai_core/tbe/op_tiling/liboptiling.so  (compat 软链)
     op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so
     op_impl/ai_core/tbe/dynamic/<op>.py           # Python tiling adapter
     op_impl/ai_core/tbe/kernel/<soc>/<op>/*.o      # 内核二进制
     op_impl/ai_core/tbe/config/<soc>/aic-<soc>-ops-info.json
     op_api/include/aclnn_<op>.h
     op_proto/inc/<op>_proto.h
   ```
7. **拷贝产物回 csrc/**：把三个 `.so` 复制到 `csrc/`（供 `setup.py` 链接）。
8. **拷贝 Python tiling adapter 到 pysrc/**：复制 `dynamic/<op>.py` 到 `pysrc/openjiuwentools/openjiuwen_ascend_ops/`，由 `package_data` 随 wheel 发布。

一行命令完成阶段 A：

```bash
cd packages/openjiuwen_ascend_ops
bash csrc/build.sh            # 默认 soc=ascend910b, install=<repo>/ztmp/cann_ops_install
# bash csrc/build.sh ascend910b ztmp/cann_ops_install   # 显式参数（路径可覆盖）
```

## 阶段 B：C++ Torch 扩展编译

入口 `setup.py`。要点：

- **CANN 路径解析**：`ASCEND_TOOLKIT_HOME` → `ASCEND_HOME_PATH` → `/usr/local/Ascend/ascend-toolkit/latest`，`ASCEND_ROOT` 取其父目录。
- **torch_npu include 自动探测**：遍历 `sys.path` 查找 `torch_npu/include`。
- **单 extension 模块** `openjiuwentools.openjiuwen_ascend_ops._openjiuwen_ascend_ops`，源文件：
  ```
  csrc/binding/pybind_binding.cpp
  csrc/registration/all_ops.cpp
  csrc/aclnn_torch_adapter/NPUBridge.cpp
  csrc/<op>/<op>_adapter.cpp
  ```
  使用 `CppExtension` + `BuildExtension`（torch.utils.cpp_extension），`-std=c++17`。
- **链接**：
  - `-Lcsrc -lcust_opapi` → 链接阶段 A 产出的 `csrc/libcust_opapi.so`
  - `-L$CANN/aarch64-linux/lib64 -lacl_rt`
  - `-L<torch_npu>/lib -ltorch_npu`
  - rpath 三连：`$ORIGIN`（让 wheel 内的 .so 找到同目录的 libcust_opapi.so）、CANN lib64、driver lib64、torch_npu lib
- **package_data**：把三个 `.so` 和算子的 `*.py`（tiling adapter，每个算子一份）随 wheel 打包，落到 `openjiuwentools.openjiuwen_ascend_ops/`。

```bash
cd packages/openjiuwen_ascend_ops

# 开发模式（推荐，产物落在 pysrc/）
pip install -e .

# uv 用户：必须关掉构建隔离（隔离环境里没有 venv 的 torch/torch_npu，
# setup.py 顶部 import torch 会直接失败；也不要把 torch 加进
# [build-system].requires——隔离环境会拉 PyPI 通用 torch，与 torch_npu ABI 不配）
uv pip install setuptools wheel          # 关隔离后构建依赖需自备
uv pip install -e . --no-build-isolation

# 或带详细输出
python setup.py build_ext --inplace
```

## 阶段 C：打 wheel 并安装

```bash
cd packages/openjiuwen_ascend_ops

python -m build
# 或 python setup.py bdist_wheel

# 产物：dist/openjiuwentools_openjiuwen_ascend_ops-0.1.0-cp311-cp311-linux_aarch64.whl
pip install dist/openjiuwentools_openjiuwen_ascend_ops-0.1.0-cp311-cp311-linux_aarch64.whl
```

wheel 内容：`_openjiuwen_ascend_ops.cpython-311-aarch64-linux-gnu.so` + 三个 `libcust_*.so` + 每个算子一份 `<op>.py`。

## 端到端流程

生产 wheel 的完整顺序：

```bash
# 先设好 CANN 环境变量（见 00-environment.md）
cd packages/openjiuwen_ascend_ops
bash csrc/build.sh                       # 阶段 A
pip install -e .                         # 阶段 B（开发模式）
# 或：python -m build && pip install dist/*.whl   # 阶段 B+C
```

性能/调试构建（带 debug line）见 [Profiling](../60-profiling/00-profiling.md) 与 [Sanitizer](../50-debugging/00-sanitizer.md)。

## 编译选项

### CMake 顶层开关

定义在 `csrc/CMakeLists.txt` 与 `csrc/cmake/config.cmake`：

| 开关 | 默认 | 作用 |
| --- | --- | --- |
| `BUILD_OPEN_PROJECT` | ON | ON=自定义算子包流程（生成 .run + 独立 vendor 目录），OFF=内嵌 built-in |
| `ENABLE_CCACHE` | ON | 启用 ccache |
| `ASCEND_COMPUTE_UNIT` | ascend910b | 目标 SoC，传给 `build.sh` 第 1 参 |
| `ASCEND_OP_NAME` | ALL | 分号分隔算子名列表或 `ALL`，控制 `op_add_subdirectory` 过滤 |
| `VENDOR_NAME` | jiuwen-ascend | vendor 目录名（`build.sh` 固定传 `vllm-ascend`） |
| `CUSTOM_ASCEND_CANN_PACKAGE_PATH` | 未设 | 覆盖 `ASCEND_CANN_PACKAGE_PATH` |
| `CHECK_COMPATIBLE` | true | 兼容性检查 |
| `CMAKE_BUILD_TYPE` | Release | 标准 CMake 构建类型，影响 `_FORTIFY_SOURCE=2` / `-g` / `-s` |
| `OP_DEBUG_CONFIG` | false | 传给 `op_build` 的 `--op-debug-config` |
| `ENABLE_OPS_HOST` / `ENABLE_OPS_KERNEL` | ON / ON | 是否构建 host / kernel（UT 测试时 kernel 关闭） |

### 算子 debug line / sanitizer 开关

每个算子 `op_host/CMakeLists.txt` 暴露一个 `<OP>_DEBUG_LINE` 缓存变量（默认 OFF，可用同名环境变量覆盖）：

| 环境变量 / CMake 变量 | 追加选项 | 说明 |
| --- | --- | --- |
| `RQ_UNIFORM_PACK_K8V4_DEBUG_LINE` | `-g` | 给 RqUniformPackK8v4 内核生成 `.debug_line` |

`-g` 生成 `.debug_line` 以支持源码行调用栈。若需 msSanitizer 全量插桩，可手动在 `add_ops_compile_options` 的 `OPTIONS` 里追加 `--cce-enable-sanitizer`。生产构建保持 unset 即可输出干净二进制。详细 sanitizer 编译选项见 [Sanitizer](../50-debugging/00-sanitizer.md)。

### 内核统一编译选项

算子 `op_host/CMakeLists.txt` 固定附加（通过 `add_ops_compile_options(OP_NAME <Op> OPTIONS ...)` 写入 `custom_compile_options.ini`，由 `op_build` 读取）：

```
--cce-auto-sync=on
-DASCENDC_DUMP=0
-I${CMAKE_SOURCE_DIR}/third_party/catlass/include
-Wno-deprecated-declarations
-Werror
```

另外，算子 CMakeLists 还有 `BR_KEY_UNIFORM_SCHEME` 编译期宏（Key 编码方案 1/2/3，默认 1，可用同名环境变量覆盖）。在源仓库中该宏由 pack 与 attention/FIA 两侧 CMake 联动，两侧必须一致；本包只含 pack 侧，若后续迁入 attention 侧需保持联动。

### 宿主侧（opapi/opsproto/optiling）选项

由 `csrc/cmake/intf_pub.cmake` 的 `intf_pub` INTERFACE 库提供：`-fPIC -O2` + 一长串 `-W...` 警告、`_GLIBCXX_USE_CXX11_ABI=0`、`VLLM_ASCEND_950_SOC_ENUM/CONFIG` 宏（来自 SoC 探测，要求 CANN 头里存在 `ASCEND950` 或 `ASCEND910_95` 枚举），链接 `-pie -Wl,-z,relro/now/noexecstack`，Release 下 `-s`（strip）。

`setup.py` 侧：`-std=c++17`，无 asan/msan 开关（host 侧 sanitizer 不在本仓库暴露，只在内核侧通过 `--cce-enable-sanitizer` 开）。
