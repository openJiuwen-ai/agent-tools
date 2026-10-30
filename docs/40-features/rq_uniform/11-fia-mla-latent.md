| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_fia_mla_latent 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_fia_mla_latent/ |
| 关联文档 | 00-overview.md, 09-pack-mla-latent.md, 10-dequant-mla-latent.md |

# rq_uniform_fia_mla_latent

MLA latent **absorb + FIA**（Scheme A，Path-B）。产品入口是 **`rq_uniform_fia_mla_latent_paged`**；稠密 `rq_uniform_fia_mla_latent(codes, scales, …)` 仅为 host ATen golden（UT）。

## Path-B 数据流（当前实现）

```text
adapter:  ql' = ql @ R^T                 # at::matmul；ql 已由框架做 W_UK
kernel:   Scheme A dequant page → y
          FIA MLA(ql', q_pe, y, k_pe) → o'
adapter:  o  = o' @ R                    # at::matmul；框架再做 W_UV
```

Haar 在 **Torch adapter**，不在 AscendC FIA 内核内。`w_uk_t` / `w_uv` 为 schema 占位（fuse 路径忽略）：`[1,512,512]` / `[1,512,V]`。

## Python（paged）

```python
out = torch.ops.openjiuwen.rq_uniform_fia_mla_latent_paged(
    q_nope, q_pe, latent_cache, k_pe_cache, block_table,
    actual_seq_q, actual_seq_kv,   # host int[]
    w_uk_t, w_uv, rotation,
    block_size, scale, num_tokens, total_kv, max_kv, out=None)
```

| 关键张量 | Shape |
| --- | --- |
| `q_nope` | `[Tq, H, 512]`（已 absorb） |
| `q_pe` | `[Tq, H, 64]` |
| `latent_cache` | uint8 `[Bn, block*(512+4)]` |
| `k_pe_cache` | `[Bn, Bs, 64]`（或可 reshape 的 PA） |
| `rotation` | `[512, 512]` = `R` |

Attrs：`max_kv > 0` → 融合 FIA，输出 latent context；`max_kv <= 0` → 仅 dequant `y`。

## 约束

- `L=512`，rope=64，`block_size % 16 == 0`，`num_kv_heads=1`
- fuse：`num_tokens/num_reqs * H ≤ 256`（单 S1 tile）

## 深层设计

`csrc/rq_uniform_fia_mla_latent/IMPL_DESIGN.md`（含 Path-B 产品说明与 NonQuant MLA + Scheme A 融合分析）

## 代码位置

- Adapter：`rq_uniform_fia_mla_latent_adapter.{h,cpp}`
- Kernel：`op_kernel/rq_uniform_fia_mla_latent.cpp` + vendored NonQuant MLA
- Host：`op_host/rq_uniform_fia_mla_latent_{def,tiling}.*`

## 测试

`tests/ut/test_rq_uniform_fia_mla_latent.py`
