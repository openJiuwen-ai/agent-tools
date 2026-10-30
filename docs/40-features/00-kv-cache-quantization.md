| 字段 | 值 |
| --- | --- |
| 标题 | KV cache 量化特性设计（rq_uniform 族） |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_*/ |
| 关联文档 | rq_uniform/00-overview.md, ../../30-specifications/00-operators.md, ../../00-index.md |

# KV cache 量化特性设计

本包的 KV / latent cache 低比特方案统一为 **rq_uniform 均匀量化族**（无 codebook、无 Gram/invNorm）。算子已全部迁入 `openjiuwen_ascend_ops`，对外命名空间 `torch.ops.openjiuwen`。

> 各算子说明书+设计（布局 / tiling / 代码位置 / 测试）统一归档在
> **[rq_uniform/](rq_uniform/00-overview.md)**。下文只保留方案级动机与对照。

## 一、设计动机

- **去掉 codebook**：均匀量化纯算术（min/max/step）。
- **按路径选比特**：Full-Attn Key 可用 8-bit（k8v4）或 Key/Value 均 4-bit（k4v4）；SFA/MLA 另有 388 / Scheme A 布局。
- **FP32 旋转累加**：`x @ R^T` 保精度，元数据留在原始域。

## 二、子方案速览

| 子方案 | Pack | Attention / 读侧 | Cache 要点 |
| --- | --- | --- | --- |
| Full K8V4 | `pack_k8v4` | `fia_paged_k8v4` / `_msd` | Key tile 16×(D+4)；Val 16×(D/2+4)；D∈{128,256} |
| Full K4V4 | `pack_k4v4` | `fia_paged_k4v4` / `_msd` | K/V 均为 16×68=1088；D=128 only |
| SFA Schema A | `pack_sfa` | `kv_quant_sparse_flash_attention_v2` | 388B slot；signed int4 V2 |
| MLA latent | `pack_mla_latent` | `dequant_mla_latent` 或 `fia_mla_latent_paged` | SoA `block*(L+4)` Scheme A |

详细公式与物理布局见各子文档；k8v4 Key Scheme A 示意：

```text
y = x @ R^T
step = (max-min)/255 ; q ∈ [0,255]
code_i8 = q - 128 ; base' = min + 128*step
decode: y = base' + int8(code)*step
```

## 三、与 TurboQuant 4-bit codebook（未迁入）对照

| 维度 | turboquant 4-bit | rq_uniform |
| --- | --- | --- |
| 量化 | codebook argmin | min/max 均匀量化 |
| 元数据 | 每行 norm | base/step 或 vmin/vstep 或 scale/offset |
| 归一化 | 量化前 /norm | 不归一化 |
| 本包状态 | 未迁入 | 11 算子已落地 |

## 四、Serving 映射（vLLM Ascend）

| dtype / 路径 | Pack + Attn |
| --- | --- |
| `rq_k8v4_uq` | pack_k8v4 + fia(_msd) |
| `turboquant_k4v4_uq` | pack_k4v4 + fia(_msd) |
| SFA 388 Schema A | pack_sfa + attn_v2 |
| MLA TQ | pack_mla_latent + fia_mla_latent_paged / dequant |

## 五、文档索引

- 族总览与清单：[rq_uniform/00-overview.md](rq_uniform/00-overview.md)
- 单算子设计：`rq_uniform/01` … `rq_uniform/11`
- 接口契约总览：[30-specifications/00-operators.md](../30-specifications/00-operators.md)
- UT：`packages/openjiuwen_ascend_ops/tests/ut/`
