| 字段 | 值 |
| --- | --- |
| 标题 | 算子接口规格总览 |
| 修改日期 | 2026-09-14 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/<op>/op_host/<op>_def.cpp, csrc/registration/all_ops.cpp |
| 关联文档 | 00-index.md, 10-architecture/01-registration.md, 40-features/00-kv-cache-quantization.md |

# 算子接口规格总览

本目录是算子的**正式接口契约**，面向调用方与测试。每个算子一份规格文件，只记录契约事实：Torch schema、张量 dtype/shape/约束、标量、常量、返回值。**不含**功能说明、用法示例、参数语义、tiling、cache 布局、代码位置——这些属于实现设计，见 [40-features](../40-features/00-kv-cache-quantization.md) 下对应算子文件。

## 通用约定

- 所有输入张量须在 NPU 上。
- `head_size` 恒为 128。
- 输入 dtype 为 fp16 或 bf16（key/value/query 同 dtype）；cache 为 uint8。fp16 与 bf16 是两个独立实例化档位，kernel 内编译期选择 `half` / `bfloat16_t`。
- SoC：同时注册 `ascend910b`（静态编译）与 `ascend910_93`（JIT）。
- 非连续输入：`IgnoreContiguous()` 允许非连续 stride；adapter 在不满足直读条件时 `.contiguous()` 兜底。
- `actual_seq_len_*` 标 `ValueDepend(REQUIRED)`，host 转 `aclIntArray` 作为值依赖输入传入。
- 命名空间：`torch.ops.openjiuwen.*`，dispatch key `PrivateUse1`。
- pack 算子 in-place 写 `key_cache`/`value_cache` 并返回两者元组。

## 算子与规格文件

| 算子 | 类别 | 量化方案 | 规格 |
| --- | --- | --- | --- |
| `bit_residual_pack_k8v4` | pack | BitResidual K8V4 | [01](01-bit-residual-pack-k8v4.md) |

## 配对关系

pack 算子写出的 cache 只能由同方案的 attention 算子读；两侧 cache 常量成对定义，须保持一致。

| 打包算子 | 对应注意力算子（源仓库维护） | cache 布局 |
| --- | --- | --- |
| `bit_residual_pack_k8v4` | `bit_residual_attention_paged_k8v4`（未迁入本包） | Key `(block_size/16)*2112`；Value `(block_size/16)*1088` |

## Python 调用

```python
from openjiuwentools.openjiuwen_ascend_ops import bit_residual_pack_k8v4
```

或直接走底层：

```python
torch.ops.openjiuwen.bit_residual_pack_k8v4(...)
```

## 相关

- 算子注册与分发机制：[注册与分发机制](../10-architecture/01-registration.md)
- 量化方案原理、cache 布局、tiling、用法与代码位置：[特性设计](../40-features/00-kv-cache-quantization.md)
