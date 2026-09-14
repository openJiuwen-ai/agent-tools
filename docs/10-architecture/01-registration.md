| 字段 | 值 |
| --- | --- |
| 标题 | 算子注册与分发机制 |
| 修改日期 | 2026-09-14 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/binding/, csrc/registration/, csrc/aclnn_torch_adapter/, csrc/<op>/*_adapter.cpp |
| 关联文档 | 00-index.md, 10-architecture/00-overview.md, 20-build/01-build-guide.md, 30-specifications/00-operators.md, 40-features/00-kv-cache-quantization.md |

# 算子注册与分发

算子从 Python 调用到 NPU 内核执行，经过三层结构：PyBind 入口 → Torch 注册 → aclnn 适配。本文解释每一层做什么、它们如何拼接。

## 1. PyBind 入口

`csrc/binding/pybind_binding.cpp` 定义 `PYBIND11_MODULE(_openjiuwen_ascend_ops, m)`，仅暴露一个函数 `register_ops()`，转发到 `openjiuwen_ascend_ops::register_all_ops`。

Python 侧在 `pysrc/openjiuwentools/openjiuwen_ascend_ops/__init__.py` 中：

```python
from . import _openjiuwen_ascend_ops
_openjiuwen_ascend_ops.register_ops()
```

因此 `import openjiuwentools.openjiuwen_ascend_ops` 即触发算子注册。

## 2. Torch 注册（`openjiuwen` 命名空间）

`csrc/registration/all_ops.cpp` 用 `c10::RegisterOperators` 把各算子注册到 schema 命名空间 `openjiuwen::`，dispatch key 为 `PrivateUse1`（即 torch_npu 占用的 Custom 后端键）。每个 schema 显式声明签名，kernel 函数指向 `openjiuwen_ascend_ops::<op>` 实现。

注册的 schema 汇总（随版本演进可能增减，以 `all_ops.cpp` 为准）：

| 算子 | Torch schema | 返回 |
| --- | --- | --- |
| `bit_residual_pack_k8v4` | `(Tensor key, Tensor value, Tensor slot_mapping, Tensor query_start_loc, Tensor rotation_t, Tensor key_cache, Tensor value_cache, int num_reqs, int block_size) -> ()` | in-place 写 `key_cache`/`value_cache` |

> 算子的完整张量 dtype/shape 约束与标量约束，见 [30-specifications](../30-specifications/00-operators.md)。

注册后即可通过 `torch.ops.openjiuwen.<op>(...)` 调用。

## 3. aclnn 适配层

每个算子目录下的 `<op>_adapter.cpp` 是算子的 C++ 实现，职责：

1. 校验输入 torch tensor 的 device / dtype / stride / shape。
2. 必要时做 `.contiguous()` 兜底（`def.cpp` 中标 `IgnoreContiguous()`，允许非连续 stride）。
3. 通过 `EXEC_NPU_CMD(aclnn<Op>, ...)` 宏把任务下发到 ACLNN。

例如 `bit_residual_pack_k8v4_adapter.cpp` 调用：

```cpp
EXEC_NPU_CMD(aclnnBitResidualPackK8v4, key_work, value_work, rotation_work,
             slot_work, query_start_work, n_vec, vec_per_core_i64, ...,
             key_cache, value_cache);
```

参数列表与 `op_host/bit_residual_pack_k8v4_def.cpp` 的 `Attr` 声明一一对应（`n_vec`、`vec_per_core`、`num_heads`、`block_size`、`num_blocks`、`num_reqs`、`key_stride_token/head`、`value_stride_token/head`、`key/value_storage_offset` 等）。

### 3.1 `EXEC_NPU_CMD` 工作原理

定义于 `csrc/aclnn_torch_adapter/op_api_common.h`。宏展开后：

- 通过 `GetOpApiFuncAddr("aclnn<Op>GetWorkspaceSize")` 与 `GetOpApiFuncAddr("aclnn<Op>")` 从 `libcust_opapi.so` 动态查表（dlsym）拿到函数指针。
- `ConvertTypes(...)` 把 torch tensor 转成 acl tensor 句柄，并打包参数。
- 调 `aclnn<Op>GetWorkspaceSize(...)` 计算 workspace 大小，用 `at::empty({size}, kByte)` 在 NPU 上分配 workspace。
- 在 `c10_npu::getCurrentNPUStream().stream(false)` 上异步 `aclnn<Op>(workspace, size, executor, stream)`。

因此 `libcust_opapi.so`（构建阶段 A 产出）必须在运行时可被 dlopen，且其符号表必须包含 `aclnn<Op>` 与 `aclnn<Op>GetWorkspaceSize`。`setup.py` 的 `-lcust_opapi` 与 `__init__.py` 对 `LD_LIBRARY_PATH` 的注入即为此服务。

### 3.2 NPU tensor 桥接

`csrc/aclnn_torch_adapter/NPUBridge.{h,cpp}` 与 `NPUStorageImpl.{h,cpp}` 提供 `vllm_ascend::NPUStorageImpl`（继承 `c10::StorageImpl`）与 `NPUBridge::GetNpuStorageImpl` 静态转换助手，用于从 torch tensor 拿到 CANN GE 友好的 storage 描述（`NPUStorageDesc` 含 base_sizes/strides/offset/aclFormat 等）。这部分代码来自 torch_npu 上游（BSD 风格 license 注释），与 `op_api_common.h` 一起构成"torch_npu 私有 API 的最小适配层"，让本仓库的 adapter 不直接依赖 torch_npu 内部头即可工作（但仍需 link `-ltorch_npu`）。

## 4. ACLNN autogen 闭环

`op_host/<op>_def.cpp` 用 `OpDef` DSL 声明输入/输出/属性/AICore 配置（如 `DynamicCompileStaticFlag`、`DynamicFormatFlag`、`ExtendCfgInfo("aclnnSupport.value", "support_aclnn")` 等），并 `OP_ADD(<Op>)` 注册到 `OpDefRegistry`。

CMake 阶段的 `opbuild_gen_default` 目标调用 CANN 的 `op_build` 工具，从该 `op_host_aclnn.so` 生成 `autogen/aclnn_<op>.cpp/.h` 与 `<op>_proto.cpp/.h`，再编译进 `ops_aclnn` / `opsproto`。这样 `aclnn<Op>` 符号最终落在 `libcust_opapi.so`，被 §3.1 的宏在运行时 dlsym。

几个对算子作者重要的 `OpDef` 配置点：

- **dtype 配对**：通过 `DataType({ge::DT_FLOAT16, ge::DT_BF16})` 把 fp16 与 bf16 注册为两个独立的"实例化"档位；kernel 内通过 `ORIG_DTYPE_KEY`/`DTYPE_KEY` 宏在编译期选择 `half` 或 `bfloat16_t`。
- **非连续输入**：`IgnoreContiguous()` 允许非连续 stride；adapter 在 stride 不满足 kernel 直读条件时 `.contiguous()` 兜底。
- **值依赖输入**：`actual_seq_len_q`/`actual_seq_len_kv` 标 `ValueDepend(REQUIRED)`，host 把 list 转成 `aclIntArray` 作为值依赖输入传入。
- **SoC 多档**：`def.cpp` 同时注册 `ascend910_93` 与 `ascend910b` 两档；910b 用 `jitCompile.flag=static_false`，910_93 走默认 JIT。

## 5. Python 侧分发

`pysrc/.../ops/<op>.py`（如 `bit_residual_pack_k8v4.py`）是 thin wrapper：转发到 `torch.ops.openjiuwen.<op>(...)`，并整理返回值（如 pack 算子返回 `(key_cache, value_cache)` 元组）便于 Python 调用。

`__init__.py` 在 import `_openjiuwen_ascend_ops` 后立即 `register_ops()`，再导出各算子函数，用户即可：

```python
from openjiuwentools.openjiuwen_ascend_ops import bit_residual_pack_k8v4
bit_residual_pack_k8v4(key, value, slot_mapping, query_start_loc, rotation_t,
                       key_cache, value_cache, num_reqs, block_size)
```

## 6. 运行时 OPP 分发

`ASCEND_CUSTOM_OPP_PATH` 是 CANN runtime 识别自定义算子包的环境变量。本包有两套 OPP 来源：

1. **生产 wheel 内置**：`pysrc/.../_cann_ops_custom/vendors/vllm-ascend`（若包发布者预先塞入），由 `__init__.py` 自动注入 `ASCEND_CUSTOM_OPP_PATH` 与 `LD_LIBRARY_PATH`，用户无需额外配置。
2. **仓库内独立安装**：`ztmp/cann_ops_install/vendors/vllm-ascend`，由 `csrc/build.sh` 写入，由用户脚本 `source $ASCEND_CUSTOM_OPP_PATH/bin/set_env.bash` 注入。

后者是调试/性能构建的默认路径，因为它需要带 `.debug_line` 的内核对象，MindStudio Insight 才能展示源码行（见 [Profiling](../60-profiling/00-profiling.md)）。
