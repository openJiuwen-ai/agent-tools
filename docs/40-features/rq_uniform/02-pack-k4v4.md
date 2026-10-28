| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_pack_k4v4 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_pack_k4v4/ |
| 关联文档 | 00-overview.md, 04-fia-paged-k4v4.md, 06-fia-paged-k4v4-msd.md |

# rq_uniform_pack_k4v4

将 Key/Value 写入 **K4V4** cache：Key 与 Value **均为 int4** 均匀量化（与 Value 同路径 `EncodeValueBatch`），`head_size` **仅 128**。

> 布局以 `op_kernel/rq_uniform_pack_k4v4_common.h` 为准；实现细节见同目录 `IMPL_DESIGN.md`。

## Python

```python
torch.ops.openjiuwen.rq_uniform_pack_k4v4(
    key, value, slot_mapping, query_start_loc,
    rotation_t, key_cache, value_cache, num_reqs, block_size)
```

## 约束

| 项 | 值 |
| --- | --- |
| `head_size` | **128 only** |
| `block_size` | `% 16 == 0` |
| Key/Value code | 每行 `D/2 = 64` 字节 nibble |
| 16-row tile | Key/Value 均为 **1088 B** |

## Cache 布局

```text
# TQ_KEY_ROW_CODE_BYTES = TQ_VAL_ROW_CODE_BYTES = 64
# TQ_KEY_BYTES_PER_ROW = TQ_VAL_BYTES_PER_ROW = 68
Key/Value head region (identical structure):
  [block_size * 64 codes][block_size * 2 vmin][block_size * 2 vstep]
cache 第 3 维: (block_size/16) * 1088
```

反量化（读侧）：`y = vmin + (signed_nibble + 8) * vstep`。

## Tiling / 实现

与 k8v4 同构的 MIX AIC:AIV=1:2；Manual 旋转 tile M=32、每 AIV 16 行。详见 `IMPL_DESIGN.md`（实现细节）与 `*_common.h`（布局常量）。

## 配对

- `rq_uniform_fia_paged_k4v4` / `rq_uniform_fia_paged_k4v4_msd`
- Serving：`turboquant_k4v4_uq`

## 代码位置

- Adapter：`rq_uniform_pack_k4v4_adapter.{h,cpp}`
- Kernel：`op_kernel/rq_uniform_pack_k4v4.cpp`、`*_common.h`、`*_aic.h`、`*_aiv.h`

## 测试

`tests/ut/test_rq_uniform_pack_k4v4.py`
