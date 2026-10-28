| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_pack_k8v4 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_pack_k8v4/ |
| 关联文档 | 00-overview.md, 03-fia-paged-k8v4.md, 05-fia-paged-k8v4-msd.md |

# rq_uniform_pack_k8v4

将 token-major Key/Value（fp16/bf16）经 Haar 旋转后写入 **K8V4** uint8 paged cache：Key 为 Scheme A int8 均匀量化，Value 为 signed int4。

## Python

```python
import openjiuwentools.openjiuwen_ascend_ops  # noqa: F401
torch.ops.openjiuwen.rq_uniform_pack_k8v4(
    key, value, slot_mapping, query_start_loc,
    rotation_t, key_cache, value_cache, num_reqs, block_size)
```

Schema：见 `csrc/registration/all_ops.cpp`（`-> ()`，in-place 写 cache）。

## 约束

| 项 | 值 |
| --- | --- |
| `head_size` | 128 或 256（FIA adapter 与 pack 布局按 D 扩展） |
| `block_size` | `% 16 == 0` |
| dtype | key/value/rotation fp16 或 bf16；cache uint8 |
| 旋转 | `rotation_t = R^T`，AIC FP32 累加 |

## 量化

- **Key Scheme A**：`y=x@R^T` → `step=(max-min)/255` → `code_i8=q-128`，`base'=min+128*step`
- **Value int4**：`vstep=(max-min)/15`，signed nibble；`+8*vstep` 折进 `vmin`
- 宏 `BR_KEY_UNIFORM_SCHEME`（默认 1=A）；须与 FIA 侧一致

## Cache 布局（每 head / block）

16-row tile：Key **2112 B**（D=128），Value **1088 B**。

```text
Key:   [block_size * D codes][block_size * 2 base][block_size * 2 step]
Value: [block_size * (D/2) nibbles][block_size * 2 vmin][block_size * 2 vstep]
key_cache  第 3 维: (block_size/16) * 16*(D+4)
value_cache 第 3 维: (block_size/16) * 16*(D/2+4)
```

非完整 tile：read-modify-write。

## Tiling

| Key | 含义 |
| --- | --- |
| 0 | 默认 |
| 1 | Large-contiguous（长序列 Manual MMAD） |

`KERNEL_TYPE_MIX_AIC_1_2`。细节：`csrc/rq_uniform_pack_k8v4/IMPL_DESIGN.md`。

## 配对

- Prefill：`rq_uniform_fia_paged_k8v4`
- Decode：`rq_uniform_fia_paged_k8v4_msd`
- Serving：`rq_k8v4_uq`

## 代码位置

- Adapter：`rq_uniform_pack_k8v4_adapter.{h,cpp}`
- Host：`op_host/rq_uniform_pack_k8v4_{def,tiling}.*`、`aclnn_rq_uniform_pack_k8v4.h`
- Kernel：`op_kernel/rq_uniform_pack_k8v4.cpp`、`*_common.h`、`*_aic.h`、`*_aiv.h`

## 测试

`tests/ut/test_rq_uniform_pack_k8v4.py`；e2e layout：`tests/e2e/singlecard/rq_uniform_k8v4_batch_tiers.py`
