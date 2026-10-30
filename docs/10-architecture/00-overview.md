| 字段 | 值 |
| --- | --- |
| 标题 | 整体架构与包分层 |
| 修改日期 | 2026-09-14 |
| 关联模块 | packages/openjiuwen_ascend_ops/ |
| 关联文档 | 00-index.md, 10-architecture/01-registration.md, 20-build/01-build-guide.md, 40-features/00-kv-cache-quantization.md, 30-specifications/00-operators.md |

# 整体架构

## 仓库定位

`agent-tools` 是 OpenJiuwenTools 的 monorepo（uv workspace，`packages/*` 为成员）。其中的 `packages/openjiuwen_ascend_ops` 是 Ascend 亲和（affinity）算子包，发布名 `openjiuwentools-openjiuwen-ascend-ops`。它在 Ascend NPU 上提供融合量化算子，用于把 KV cache 以低比特格式（K8V4 均匀量化）打包进 PagedAttention 的物理 cache。当前只含 pack 算子；配套的 attention 算子仍在源仓库 `jiuwen-ak-affinity` 维护。

算子通过 PyTorch 的 `torch.ops.openjiuwen.*` 命名空间暴露，底层走 CANN ACLNN 调度。详见 [注册机制](01-registration.md)。

## 算子包内部结构

算子包位于 `packages/openjiuwen_ascend_ops/`，发布名 `openjiuwentools-openjiuwen-ascend-ops`，模块路径 `openjiuwentools.openjiuwen_ascend_ops`。

| 子目录 | 角色 | 关键文件 |
| --- | --- | --- |
| `csrc/` | C++/AscendC 源码与 CANN 自定义算子工程 | `csrc/CMakeLists.txt`、`csrc/build.sh`、`csrc/cmake/`、`csrc/<op>/op_host/`、`csrc/<op>/op_kernel/` |
| `pysrc/openjiuwentools/openjiuwen_ascend_ops/` | Python 包源码（wheel 来源） | `__init__.py`、`ops/*.py`、构建产物 `_openjiuwen_ascend_ops.*.so` |
| `tests/e2e/singlecard/` | E2E 测试 | `xrx_rq_uniform_k8v4_batch_tiers.py` |
| `pyproject.toml` | PEP 517 构建后端声明 | 仅声明 `setuptools>=61.0 + wheel` |
| `setup.py` | wheel 构建入口 | 见 [构建指南](../20-build/01-build-guide.md) |

### `csrc/` 算子目录约定

每个算子遵循 CANN 自定义算子工程标准三段式：

```
csrc/<op_name>/
├── <op_name>_adapter.cpp   # Torch 扩展侧：torch.ops.openjiuwen.* 的 C++ 实现
├── <op_name>_adapter.h
├── op_host/
│   ├── CMakeLists.txt                # 注册到 op_host_aclnn / optiling / opsproto
│   ├── <op_name>_def.cpp             # OpDef 注册（输入/输出/属性/AICore 配置）
│   ├── <op_name>_tiling.cpp/.h        # host 侧 tiling 计算
│   └── aclnn_<op_name>.h             # aclnn 函数原型（autogen 的输入）
└── op_kernel/
    └── <op_name>.cpp (+ _aic.h/_aiv.h/_common.h)  # AscendC 设备内核
```

当前注册的算子见 `csrc/build.sh` 的 `CUSTOM_OPS` 与 `csrc/registration/all_ops.cpp`（快照：仅 `rq_uniform_pack_k8v4`，随版本演进可能增减）。逐个算子的接口契约见 [规格总览](../30-specifications/00-operators.md)，功能/用法/设计见 [量化方案](../40-features/00-kv-cache-quantization.md) 下各子文件。

### 公共代码

- `csrc/utils/`：CANN 风格公共头（`inc/kernel/*.h`、`inc/tiling/*.h`、`inc/log/`、`inc/error/`）与 `src/fallback_comm.cpp`，算子 tiling/kernel 可直接 include。
- `csrc/third_party/catlass/`：Huawei CANN 开源的 Catlass 模板库（类 CUTLASS 的 AscendC 矩阵/GEMM 抽象），主头 `catlass/catlass.hpp` 基于 `kernel_operator.h`，通过 `-I${CMAKE_SOURCE_DIR}/third_party/catlass/include` 注入到算子的 `op_host/CMakeLists.txt`。catlass 以 git submodule 形式引入（非本项目代码）：仓库根 `.gitmodules` 声明 `url = https://gitcode.com/cann/catlass.git`、`branch = catlass-v1-stable`，pin 到与 vllm-ascend 相同的 commit `716fd7b`；`csrc/build.sh` 检测到 `include/` 缺失时自动执行 `git submodule update --init --recursive`。
- `csrc/registration/`、`csrc/binding/`、`csrc/aclnn_torch_adapter/`：注册与 torch_npu 适配层，详见 [注册机制](01-registration.md)。

## 构建与运行时数据流

整体分两层构建，一层运行时分发：

1. **CANN 自定义算子工程**（`csrc/build.sh`）产出三个 `.so`（`libcust_opapi.so` / `libcust_opmaster_rt2.0.so` / `libcust_opsproto_rt2.0.so`）+ 内核二进制 + Python tiling adapter，默认安装到 `<repo>/ztmp/cann_ops_install/vendors/vllm-ascend/`（路径可由 `build.sh` 参数覆盖）。
2. **setuptools**（`setup.py`）把 C++ Torch 扩展编译为 `_openjiuwen_ascend_ops.*.so`，链接上一步的 `libcust_opapi.so`，并打 wheel。
3. **运行时**：import 包时自动注入 `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`，调用 `register_ops()` 注册到 `torch.ops.openjiuwen.*`。

详细步骤见 [构建指南](../20-build/01-build-guide.md)。

> pack 写出的 cache 物理布局与其配对的 attention 算子 1:1 对应（同方案 pack 写出的 cache 只能由同方案 attention 读）。本包当前只迁入了 pack 侧，cache 布局定义见 [特性设计](../40-features/00-kv-cache-quantization.md)。
