| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_fia_paged_k4v4_msd 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_fia_paged_k4v4_msd/ |
| 关联文档 | 00-overview.md, 02-pack-k4v4.md, 04-fia-paged-k4v4.md |

# rq_uniform_fia_paged_k4v4_msd

k4v4 的 MSD decode 算子。IO 与 `fia_paged_k4v4` 同构；`KV_T = int4b_t`，tiling key 族 `QF16_KVINT4_*`；`head_size` 仅 128；K/V meta 均 fold `+8`。

## Python

```python
out = torch.ops.openjiuwen.rq_uniform_fia_paged_k4v4_msd(
    query, key_cache, value_cache, block_table,
    actual_seq_len_q, actual_seq_len_kv, atten_mask,
    rotation_key, rotation_value,
    num_heads, num_kv_heads, head_size, block_size,
    scale_value, pre_tokens, next_tokens, sparse_mode, out=None)
```

## 与 k8v4_msd

| 维度 | k8v4_msd | k4v4_msd |
| --- | --- | --- |
| `KV_T` | `int8_t` | `int4b_t` |
| tiling key | `QF16_KVINT8_*` | `QF16_KVINT4_*` |
| head_size | 128/256 | 128 |
| cache tile | Key 2112 / Val 1088 | 1088 / 1088 |

## 代码位置

- Adapter：`rq_uniform_fia_paged_k4v4_msd_adapter.{h,cpp}`
- Us2 tiling / kernel：`op_host/*_us2_tiling.*`、`op_kernel/tq_msd_us2_*.h`

## 实现文档

`csrc/rq_uniform_fia_paged_k4v4_msd/IMPL_DESIGN.md`

## 测试

`tests/ut/test_rq_uniform_fia_paged_k4v4_msd.py`
