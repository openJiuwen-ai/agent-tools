| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_kv_quant_sparse_flash_attention_v2 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_kv_quant_sparse_flash_attention_v2/ |
| 关联文档 | 00-overview.md, 07-pack-sfa.md |

# rq_uniform_kv_quant_sparse_flash_attention_v2

SFA Schema A **稀疏** Flash Attention：对 `pack_sfa` 的 388B PA slots 做 top-K gather + online softmax。V2 对 **signed int4** pack 布局做 dequant（与 pack V2 配对）。

共享 MLA tiling/infershape 已并入本目录（不再依赖独立 SFA v1 算子目录）。

## Python

```python
out = torch.ops.openjiuwen.rq_uniform_kv_quant_sparse_flash_attention_v2(
    query, key, value, sparse_indices,
    block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
    scale_value,
    key_quant_mode, value_quant_mode, sparse_block_size,
    layout_query, layout_kv, sparse_mode, attention_mode,
    quant_scale_repo_mode, tile_size, rope_head_dim, out=None)
```

典型 Schema A 调用（与源仓 golden / UT 一致）：

```text
key/value = cache.view(int8)
layouts: "TND" / "PA_BSND"
modes: key/value_quant=4, sparse_mode=3, attention_mode=2,
       quant_scale_repo_mode=1, tile_size=128, rope_head_dim=64
sparse_indices: [1, 1, topK] int32
```

## Query / 输出

- `query`：`[T, Nq, 576]`（nope 512 + rope 64）
- 输出有效比较维常取前 **512**（nope）；rope 参与 score

## 配对

写侧：`rq_uniform_pack_sfa`（默认 tilingKey=1）。勿与 legacy unsigned pack 混用。

## 代码位置

- Adapter：`rq_uniform_kv_quant_sparse_flash_attention_v2_adapter.{h,cpp}`
- Host：`op_host/*tiling*`、`*infershape*`、`*_v2_def.cpp`
- Kernel：`op_kernel/rq_uniform_kv_quant_sparse_flash_attention_v2.cpp` + MLA template headers

## 实现文档

`csrc/rq_uniform_kv_quant_sparse_flash_attention_v2/IMPL_DESIGN.md`

## 测试

`tests/ut/test_rq_uniform_kv_quant_sparse_flash_attention_v2.py`
