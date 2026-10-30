| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_dequant_mla_latent 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_dequant_mla_latent/ |
| 关联文档 | 00-overview.md, 09-pack-mla-latent.md |

# rq_uniform_dequant_mla_latent

从 `pack_mla_latent` 的 uint8 slab **gather + Scheme A dequant + `y @ R`**，恢复 fp latent `c`。不做 attention。

## Python

```python
c = torch.ops.openjiuwen.rq_uniform_dequant_mla_latent(
    latent_cache, rotation, block_table,
    context_lens, seq_starts, block_size, total_tokens)
# c: [total_tokens, L]，request 顺序
```

| 输入 | 含义 |
| --- | --- |
| `rotation` | `R [L,L]`（pack 用的是 `R^T`） |
| `context_lens` / `seq_starts` | 每请求长度与起始 |
| `total_tokens` | host 侧 `sum(context_lens)`，避免 device sync |

## 公式

```text
slot = seq_starts[r] + t
y = base + int8(code) * step
c = y @ R
```

## Tiling

MIX 1:2；输出按 32 token 一组，两 AIV 各 gather/dequant 16，再 cube ×R。

## 配对

写侧 `pack_mla_latent`；下游框架 `kv_b_proj` + 标准 FIA。Decode 融合路径见 `fia_mla_latent_paged`。

## 代码位置

- Adapter：`rq_uniform_dequant_mla_latent_adapter.{h,cpp}`
- 实现：`csrc/rq_uniform_dequant_mla_latent/IMPL_DESIGN.md`

## 测试

`tests/ut/test_rq_uniform_dequant_mla_latent.py`
