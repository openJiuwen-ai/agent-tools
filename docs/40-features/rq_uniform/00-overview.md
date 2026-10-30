| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform 均匀量化算子族总览 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_*/ |
| 关联文档 | ../00-kv-cache-quantization.md, ../../00-index.md, ../../30-specifications/00-operators.md |

# rq_uniform 均匀量化算子族总览

本目录归档 `openjiuwen_ascend_ops` 中全部 **rq_uniform_*** 算子的说明书+设计（以当前 `csrc/` 实现为准）。
调用统一走 `torch.ops.openjiuwen.<op>`（硬切，无 `bit_residual_*` 旧名）。

## 算子清单与配对

| # | 算子 | 角色 | 配对 / 下游 | 文档 |
| --- | --- | --- | --- | --- |
| 1 | `rq_uniform_pack_k8v4` | Full-Attn / hybrid pack（K int8 + V int4） | `fia_paged_k8v4` / `_msd` | [01](01-pack-k8v4.md) |
| 2 | `rq_uniform_pack_k4v4` | Full-Attn pack（K/V 均为 int4） | `fia_paged_k4v4` / `_msd` | [02](02-pack-k4v4.md) |
| 3 | `rq_uniform_fia_paged_k8v4` | Prefill / continuation FIA | 读 pack_k8v4 cache | [03](03-fia-paged-k8v4.md) |
| 4 | `rq_uniform_fia_paged_k4v4` | Prefill / continuation FIA | 读 pack_k4v4 cache | [04](04-fia-paged-k4v4.md) |
| 5 | `rq_uniform_fia_paged_k8v4_msd` | Decode MSD（k8v4） | 同 pack_k8v4 cache | [05](05-fia-paged-k8v4-msd.md) |
| 6 | `rq_uniform_fia_paged_k4v4_msd` | Decode MSD（k4v4） | 同 pack_k4v4 cache | [06](06-fia-paged-k4v4-msd.md) |
| 7 | `rq_uniform_pack_sfa` | SFA Schema A 388B slot pack | attn V2 | [07](07-pack-sfa.md) |
| 8 | `rq_uniform_kv_quant_sparse_flash_attention_v2` | SFA sparse PA | 读 pack_sfa | [08](08-kv-quant-sparse-flash-attention-v2.md) |
| 9 | `rq_uniform_pack_mla_latent` | MLA latent Scheme A pack | dequant / fia_mla | [09](09-pack-mla-latent.md) |
| 10 | `rq_uniform_dequant_mla_latent` | MLA latent 反量化 | pack_mla_latent | [10](10-dequant-mla-latent.md) |
| 11 | `rq_uniform_fia_mla_latent` (+ `_paged`) | MLA absorb FIA（Path-B） | pack_mla_latent | [11](11-fia-mla-latent.md) |

## 与 vLLM Ascend dtype 对应（摘要）

| Serving dtype / 路径 | Pack | Attention |
| --- | --- | --- |
| `rq_k8v4_uq` | `pack_k8v4` | `fia_paged_k8v4` / `_msd` |
| `turboquant_k4v4_uq` | `pack_k4v4` | `fia_paged_k4v4` / `_msd` |
| SFA Schema A 388 | `pack_sfa` | `kv_quant_sparse_flash_attention_v2` |
| MLA TQ latent | `pack_mla_latent` | `fia_mla_latent_paged`（或 dequant + 框架 FIA） |

## 公共约定

- 命名空间：`torch.ops.openjiuwen`
- Full-Attn：`block_size % 16 == 0`；fp16/bf16 输入；cache `uint8`
- k8v4：`head_size ∈ {128, 256}`；k4v4：`head_size == 128`
- 无 codebook / Gram / invNorm；旋转后均匀量化
- UT：`packages/openjiuwen_ascend_ops/tests/ut/test_rq_uniform_*.py`

## 相关

- 量化方案动机与公式对照：[../00-kv-cache-quantization.md](../00-kv-cache-quantization.md)
- 接口契约总览：[../../30-specifications/00-operators.md](../../30-specifications/00-operators.md)
- 迁移计划（历史）：[../12-rq-uniform-ops-migration-plan.md](../12-rq-uniform-ops-migration-plan.md)
