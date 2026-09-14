| 字段 | 值 |
| --- | --- |
| 标题 | bit_residual_pack_k8v4 设计 |
| 修改日期 | 2026-09-14 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/bit_residual_pack_k8v4/, pysrc/.../ops/bit_residual_pack_k8v4.py |
| 关联文档 | 00-kv-cache-quantization.md, 30-specifications/01-bit-residual-pack-k8v4.md, 10-architecture/01-registration.md |

# bit_residual_pack_k8v4 设计

BitResidual K8V4 KV cache 打包算子（Key 8-bit / Value 4-bit 均匀量化）的功能、用法、参数语义与实现设计。调用契约见 [接口规格](../30-specifications/01-bit-residual-pack-k8v4.md)。

## 功能

将 token-major fp16/bf16 Key/Value 写入 BitResidual KV cache：

- **Key** 用 8-bit 非对称均匀量化（Scheme A，默认），code 以 **signed int8** 字节存储。
- **Value** 用 4-bit 均匀量化（signed int4），每 16 行一组。

不保存 codebook / norm / Gram matrix，只保存旋转后的 base/step（Key）和 vmin/vstep（Value）元数据。in-place 修改 `key_cache` / `value_cache`。

参考 Python 等价（`op_kernel/bit_residual_pack_k8v4.cpp`）：

```python
y = x @ R^T                          # 旋转（FP32 累加，避免 bf16 精度损失）
key:   asymmetric uniform quantization (q in [0,255], stored as int8 = q-128)
value: 4-bit uniform quantization, sixteen rows at a time
```

## Python 接口

```python
from openjiuwentools.openjiuwen_ascend_ops import bit_residual_pack_k8v4

bit_residual_pack_k8v4(
    key, value, slot_mapping, query_start_loc,
    rotation_t, key_cache, value_cache,
    num_reqs, block_size,
)  # 返回 (key_cache, value_cache)
```

底层：`torch.ops.openjiuwen.bit_residual_pack_k8v4(...)`

## 关键参数含义

- **rotation_t**：128×128 旋转矩阵 R^T；AIC 做 `[M,128]@[128,128]` MMAD，结果以 FP32 保存（保精度）。
- **Key 量化（Scheme A，非对称均匀）**：`y` → `ReduceMax/Min` → `step=(max-min)/255` → `q=(y-min)/step`（0~255）；存储为 **signed int8 字节** `code_i8 = q - 128`，同时把中心偏移折进元数据 `base' = base + 128*step`（`bit_residual_pack_k8v4_aiv.h`）。cache 张量 dtype 在 Python 侧仍是 uint8，kernel 内部把底层字节按 int8 解释。
- **Value 量化（4-bit signed）**：`y` → `max/min` → `vstep=(max-min)/15` → `idx4=(y-min)/vstep`，编码为 **int4b_t**（`Adds(-8)` 后直接 Cast 到 int4），`+8*vstep` 的偏移在 pack 侧折进 `vmin`。每 byte 存同行相邻两维 nibble。
- **base/step/vmin/vstep**：16-bit 元数据（输入 dtype），保留在原始域；attention 反量化时 `K = base' + code_i8*step`、`V = vmin + idx4_s*vstep`（idx4_s 为 signed nibble）。
- **BR_KEY_UNIFORM_SCHEME**：Key 编码方案编译期开关（源仓库中 pack 与 attention/FIA 两侧 CMakeLists 一致联动，默认 1；本包只含 pack 侧，若迁入 attention 侧须保持一致）：1 = A 非对称（默认）、2 = B 对称（`s=max(|y|)/127.5`，`q=y/s+127.5`）、3 = C 遗留 `(q7<<1)|sign`。构建时可用同名环境变量覆盖。
- **block_size**：每物理 block 的 token 数，必须 16 的倍数（`TQ_BLOCK_ROWS=16`，sub-block 对齐）。
- **Partial tile 写回**：非完整 16-row tile 走 read-modify-write，避免覆盖未更新的 slot。

## Cache 物理布局

每个 head 内连续，跨 head 不共享（`op_kernel/bit_residual_pack_k8v4_common.h`）：

```
Key head region:
  + 0                            : key codes,  block_size * 128 bytes  (每行 128 个 uint8 code)
  + block_size * 128             : base,       block_size * 2 bytes   (uint16 per row)
  + block_size * (128 + 2)       : step,       block_size * 2 bytes   (uint16 per row)

Value head region:
  + 0                            : value codes, block_size * 64 bytes (每行 64 字节，存 128 个 4-bit nibble)
  + block_size * 64              : vmin,       block_size * 2 bytes
  + block_size * (64 + 2)        : vstep,      block_size * 2 bytes
```

## Tiling keys

host tiling 按 shape 特征二选一（`op_host/bit_residual_pack_k8v4_tiling.cpp`）：

| Key | 含义 |
| --- | --- |
| 0 | 默认路径（`BitResidualPackK8v4TilingData`） |
| 1 | Large-contiguous：key/value 内层 strideHead=128、strideToken=heads*128 且 tokenCount≥128、nVec≥dataCores*64、vec_per_core≥32 时启用（`BitResidualPackK8v4LargeContigTilingData`，注册名 `BitResidualPackK8v4_1`） |

任务类型为 `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC + 2 AIV）；dataCores 按 SoC 实际 AIC/AIV 数取 min，不再固定 16 组。

## 设计文档

- `csrc/bit_residual_pack_k8v4/IMPL_DESIGN.md`：完整实现设计（AIC/AIV 双缓冲、cache layout、partial tile 写回、attention decode 契约）。
- `csrc/bit_residual_pack_k8v4/design.md`：简要说明。

## C++ / aclnn 接口

Adapter 头：`csrc/bit_residual_pack_k8v4/bit_residual_pack_k8v4_adapter.h`

```cpp
void bit_residual_pack_k8v4(
    const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &slot_mapping, const at::Tensor &query_start_loc,
    const at::Tensor &rotation_t,
    at::Tensor &key_cache, at::Tensor &value_cache,
    int64_t num_reqs, int64_t block_size);
```

ACL NN：`aclnnBitResidualPackK8v4GetWorkspaceSize` / `aclnnBitResidualPackK8v4`。同样有 12 个派生 Attr。

## 配对算子

写出的 cache 只能由源仓库 `jiuwen-ak-affinity` 的 `bit_residual_attention_paged_k8v4` 读取（未迁入本包）。

## 关键代码位置

- Adapter：`csrc/bit_residual_pack_k8v4/bit_residual_pack_k8v4_adapter.cpp`
- OpDef：`csrc/bit_residual_pack_k8v4/op_host/bit_residual_pack_k8v4_def.cpp`
- Tiling：`csrc/bit_residual_pack_k8v4/op_host/bit_residual_pack_k8v4_tiling.cpp`
- Kernel：`csrc/bit_residual_pack_k8v4/op_kernel/bit_residual_pack_k8v4.cpp`
- 辅助头：`op_kernel/bit_residual_pack_k8v4_common.h`（cache 布局常量、`BR_KEY_UNIFORM_SCHEME` 默认值）、`bit_residual_pack_k8v4_aiv.h`（Key/Value 量化编码，含 Scheme A int8 code 与 base' 折叠）、`bit_residual_pack_k8v4_aic.h`（AIC MMAD）
- ACLNN header：`csrc/bit_residual_pack_k8v4/op_host/aclnn_bit_residual_pack_k8v4.h`
