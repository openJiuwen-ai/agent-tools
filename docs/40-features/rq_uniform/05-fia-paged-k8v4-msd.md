| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_fia_paged_k8v4_msd 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_fia_paged_k8v4_msd/ |
| 关联文档 | 00-overview.md, 01-pack-k8v4.md, 03-fia-paged-k8v4.md |

# rq_uniform_fia_paged_k8v4_msd

**MSD（Multi-Step Decode）** 路径的 k8v4 paged attention。Torch IO 与 `fia_paged_k8v4` 相同，读同一 pack 布局；内部走 Us2 IncreFlashAttention 特化，`KV_T = int8_t`，tiling key 族 `QF16_KVINT8_*`。

典型用途：serving decode（`nq` 小、KV 长）。

## Python

```python
out = torch.ops.openjiuwen.rq_uniform_fia_paged_k8v4_msd(
    query, key_cache, value_cache, block_table,
    actual_seq_len_q, actual_seq_len_kv, atten_mask,
    rotation_key, rotation_value,
    num_heads, num_kv_heads, head_size, block_size,
    scale_value, pre_tokens, next_tokens, sparse_mode, out=None)
```

## 实现要点

- Host：`op_host/rq_uniform_fia_paged_k8v4_msd_us2_tiling.*`
- Device：`op_kernel/tq_msd_us2_{invoke,kernel_body,pack_gather}.h` + vendored Us2
- Meta fold：Value `+8`；Key 走 int8 Scheme A
- `head_size`：128/256

## 配对

写侧 `pack_k8v4`；正确性上与 `fia_paged_k8v4` 在相同 cache 上输出应接近（UT：MSD vs FIA cosine）。

## 实现文档

`csrc/rq_uniform_fia_paged_k8v4_msd/IMPL_DESIGN.md`

## 测试

`tests/ut/test_rq_uniform_fia_paged_k8v4_msd.py`
