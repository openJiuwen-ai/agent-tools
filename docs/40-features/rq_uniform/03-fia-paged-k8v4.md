| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_fia_paged_k8v4 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_fia_paged_k8v4/ |
| 关联文档 | 00-overview.md, 01-pack-k8v4.md, 05-fia-paged-k8v4-msd.md |

# rq_uniform_fia_paged_k8v4

Paged **Fused Infer Attention**，读取 `pack_k8v4` 写出的 cache：Key 按 Scheme A int8 dequant，Value 按 int4 dequant，在旋转域做 attention，再经 `rotation_value` 折回。

面向 **prefill / continuation**（非 MSD decode 热路径）。

## Python

```python
out = torch.ops.openjiuwen.rq_uniform_fia_paged_k8v4(
    query, key_cache, value_cache, block_table,
    actual_seq_len_q, actual_seq_len_kv, atten_mask,  # mask 可选
    rotation_key, rotation_value,
    num_heads, num_kv_heads, head_size, block_size,
    scale_value, pre_tokens, next_tokens, sparse_mode, out=None)
```

`actual_seq_len_*`：host `int[]`（ValueDepend，ACL-graph 友好）。

## 约束（adapter）

| 项 | 值 |
| --- | --- |
| `head_size` | 128 或 256 |
| Key block stride | `16 * (head_size + 4)` |
| Val block stride | `16 * (head_size/2 + 4)` |
| `rotation_key/value` | `[D, D]`，与 pack 的 `R^T` / `R` 配对 |

## 实现要点

- Vendored FIA arch32 路径 + `tq_pack_layout` / `tq_dequant_device`
- Key：`TqDecodeKeyTile`（int8 Scheme A）；Value：int4 decode + fold
- 支持 GQA、`sparse_mode`（含 compress causal）、FlashDecode 长 KV、bf16

## 配对

写侧：`rq_uniform_pack_k8v4`。Decode 等价路径：`rq_uniform_fia_paged_k8v4_msd`。

## 代码位置

- Adapter：`rq_uniform_fia_paged_k8v4_adapter.{h,cpp}`
- Host tiling：`op_host/rq_uniform_fia_paged_k8v4_tiling.*`、`turboquant_fia_tiling_data.h`
- Kernel：`op_kernel/rq_uniform_fia_paged_k8v4.cpp` + `op_kernel/vendored/`

## 实现文档

`csrc/rq_uniform_fia_paged_k8v4/IMPL_DESIGN.md`

## 测试

`tests/ut/test_rq_uniform_fia_paged_k8v4.py`
