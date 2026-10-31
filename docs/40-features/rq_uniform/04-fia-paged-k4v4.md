| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_fia_paged_k4v4 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_fia_paged_k4v4/ |
| 关联文档 | 00-overview.md, 02-pack-k4v4.md, 06-fia-paged-k4v4-msd.md |

# rq_uniform_fia_paged_k4v4

Paged FIA，读取 **K4V4** cache。Key/Value 均走 int4 dequant + fold；`head_size` **仅 128**；16-row tile **1088 B**。

Torch schema / 参数表与 `fia_paged_k8v4` 同构，仅布局与 dequant 路径不同。

## Python

```python
out = torch.ops.openjiuwen.rq_uniform_fia_paged_k4v4(
    query, key_cache, value_cache, block_table,
    actual_seq_len_q, actual_seq_len_kv, atten_mask,
    rotation_key, rotation_value,
    num_heads, num_kv_heads, head_size, block_size,
    scale_value, pre_tokens, next_tokens, sparse_mode, out=None)
```

## 与 k8v4 的差异（实现）

| 维度 | k8v4 | k4v4 |
| --- | --- | --- |
| Key dequant | int8 Scheme A | Value 系 int4 + fold |
| Tile | Key 2112 / Val 1088（D=128） | Key/Val 均为 1088 |
| `head_size` | 128/256 | 仅 128 |

## 代码位置

- Adapter：`rq_uniform_fia_paged_k4v4_adapter.{h,cpp}`
- Kernel + vendored：`op_kernel/`（含 `tq_pack_layout.h`、`tq_dequant_device.h`）

## 实现文档

`csrc/rq_uniform_fia_paged_k4v4/IMPL_DESIGN.md`

## 测试

`tests/ut/test_rq_uniform_fia_paged_k4v4.py`
