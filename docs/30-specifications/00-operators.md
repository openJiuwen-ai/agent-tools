| 字段 | 值 |
| --- | --- |
| 标题 | 算子接口规格总览 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/registration/all_ops.cpp |
| 关联文档 | 00-index.md, 10-architecture/01-registration.md, 40-features/rq_uniform/00-overview.md |

# 算子接口规格总览

本目录是算子的**正式接口契约**（schema / dtype / shape / 约束）。功能说明、布局、tiling、代码位置见
[40-features/rq_uniform](../40-features/rq_uniform/00-overview.md)。

## 通用约定

- 输入须在 NPU；命名空间 `torch.ops.openjiuwen.*`，dispatch `PrivateUse1`
- Full-Attn：`block_size % 16 == 0`；fp16/bf16；cache uint8
- k8v4：`head_size ∈ {128,256}`；k4v4 / 多数路径：`head_size == 128`
- `actual_seq_len_*` / `actual_seq_*`：host `int[]`，ValueDepend
- pack：in-place 写 cache（schema `-> ()`）

## 算子与设计文档

| 算子 | 类别 | 设计（features） |
| --- | --- | --- |
| `rq_uniform_pack_k8v4` | pack | [01-pack-k8v4](../40-features/rq_uniform/01-pack-k8v4.md) |
| `rq_uniform_pack_k4v4` | pack | [02-pack-k4v4](../40-features/rq_uniform/02-pack-k4v4.md) |
| `rq_uniform_fia_paged_k8v4` | FIA | [03-fia-paged-k8v4](../40-features/rq_uniform/03-fia-paged-k8v4.md) |
| `rq_uniform_fia_paged_k4v4` | FIA | [04-fia-paged-k4v4](../40-features/rq_uniform/04-fia-paged-k4v4.md) |
| `rq_uniform_fia_paged_k8v4_msd` | MSD | [05-fia-paged-k8v4-msd](../40-features/rq_uniform/05-fia-paged-k8v4-msd.md) |
| `rq_uniform_fia_paged_k4v4_msd` | MSD | [06-fia-paged-k4v4-msd](../40-features/rq_uniform/06-fia-paged-k4v4-msd.md) |
| `rq_uniform_pack_sfa` | SFA pack | [07-pack-sfa](../40-features/rq_uniform/07-pack-sfa.md) |
| `rq_uniform_kv_quant_sparse_flash_attention_v2` | SFA attn | [08-attn-v2](../40-features/rq_uniform/08-kv-quant-sparse-flash-attention-v2.md) |
| `rq_uniform_pack_mla_latent` | MLA pack | [09-pack-mla](../40-features/rq_uniform/09-pack-mla-latent.md) |
| `rq_uniform_dequant_mla_latent` | MLA dequant | [10-dequant-mla](../40-features/rq_uniform/10-dequant-mla-latent.md) |
| `rq_uniform_fia_mla_latent` / `_paged` | MLA FIA | [11-fia-mla](../40-features/rq_uniform/11-fia-mla-latent.md) |

单算子纯契约文件可按需追加 `30-specifications/NN-rq-uniform-*.md`；当前以 `all_ops.cpp` schema + features 文档为准。

## 配对关系

| 写 | 读 |
| --- | --- |
| `pack_k8v4` | `fia_paged_k8v4` / `_msd` |
| `pack_k4v4` | `fia_paged_k4v4` / `_msd` |
| `pack_sfa` | `kv_quant_sparse_flash_attention_v2` |
| `pack_mla_latent` | `dequant_mla_latent` / `fia_mla_latent_paged` |

## Python 调用

```python
import openjiuwentools.openjiuwen_ascend_ops  # registers ops + OPP
torch.ops.openjiuwen.rq_uniform_pack_k8v4(...)
```

## 相关

- 注册机制：[注册与分发](../10-architecture/01-registration.md)
- 特性设计：[rq_uniform 总览](../40-features/rq_uniform/00-overview.md)
