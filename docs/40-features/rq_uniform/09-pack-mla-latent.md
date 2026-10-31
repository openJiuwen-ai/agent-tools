| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_pack_mla_latent 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_pack_mla_latent/ |
| 关联文档 | 00-overview.md, 10-dequant-mla-latent.md, 11-fia-mla-latent.md |

# rq_uniform_pack_mla_latent

将 MLA latent `c [T, L]`（`L = kv_lora_rank`，≤512 且 `%64==0`）经 `y = c @ R^T` 后按 **Key Scheme A** 写入单一 uint8 slab。无 dummy V、无独立 codes/scales 张量。

## Python

```python
torch.ops.openjiuwen.rq_uniform_pack_mla_latent(
    c, rotation_t, slot_mapping, latent_cache, block_size)
```

## 量化

与 Full-Attn Key Scheme A 相同：`code_i8 = q-128`，`base = min+128*step`。Decode：`y = base + int8(code)*step`。

## Cache（SoA，`block_size % 16 == 0`）

```text
latent_cache[block]: uint8[block_size * (L + 4)]
  [0, block_size*L)                     codes
  [block_size*L, block_size*(L+2))      base (2B × block_size)
  [block_size*(L+2), block_size*(L+4))  step
```

负 `slot_mapping` 跳过不写。

## Tiling

对齐 pack_k8v4 Manual：cube 组 32 token，两 AIV 各 16；MIX AIC:AIV=1:2；K/N=128。

## 配对

- Chunked prefill：`rq_uniform_dequant_mla_latent` → 框架 `kv_b_proj` + FIA
- Decode absorb：`rq_uniform_fia_mla_latent_paged`

## 代码位置

- Adapter：`rq_uniform_pack_mla_latent_adapter.{h,cpp}`
- 实现：`csrc/rq_uniform_pack_mla_latent/IMPL_DESIGN.md`

## 测试

`tests/ut/test_rq_uniform_pack_mla_latent.py`
