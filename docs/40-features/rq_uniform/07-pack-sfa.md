| 字段 | 值 |
| --- | --- |
| 标题 | rq_uniform_pack_sfa 设计 |
| 修改日期 | 2026-10-28 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_pack_sfa/ |
| 关联文档 | 00-overview.md, 08-kv-quant-sparse-flash-attention-v2.md |

# rq_uniform_pack_sfa

将已旋转的 MLA **nope** `[T, 512]` + **rope** `[T, 64]` 打包进 SFA Schema A **388 字节** fused KV slot（`PA_BSND [Bn, Bs, 1, 388]`）。无 codebook。

## Python

```python
torch.ops.openjiuwen.rq_uniform_pack_sfa(nope, rope, slot_mapping, kv_cache, block_size)
# slot_mapping: int64；kv_cache: uint8 [Bn, Bs, 1, 388]
```

## Slot 布局（`rq_uniform_pack_sfa_common.h`）

| Offset | Bytes | 内容 |
| --- | --- | --- |
| `[0, 256)` | 256 | 4-bit nope codes（低 nibble = 偶维） |
| `[256, 384)` | 128 | rope × 64（输入 dtype，2B/维） |
| `[384, 388)` | 4 | FP16 `scale` + FP16 `offset` |

常量：`SLOT_BYTES=388`（禁止 386 TQ4 协议）。

## 量化 / 反量化

- Pack V2（默认）：signed int4 码 `[-8,7]` 直接存
- Attention V2 dequant：`q = (nibble + 8) & 0xF`，`y = offset + q * scale`
- Legacy（`JIUWEN_PACK_SFA_LEGACY=1`）：tilingKey=0，unsigned 语义；与 V2 **不可同进程切换**（aclnn 缓存）

## Tiling keys

| Key | 路径 |
| --- | --- |
| 0 | Legacy MIX scalar EncodeOne |
| 1 | **默认** V2：纯 AIV，`*_vector_v2.h` |

## 配对

`rq_uniform_kv_quant_sparse_flash_attention_v2`；vLLM SFA Schema A 388。

## 代码位置

- Adapter：`rq_uniform_pack_sfa_adapter.{h,cpp}`
- Kernel：`op_kernel/rq_uniform_pack_sfa.cpp`、`*_common.h`、`*_vector_v2.h`
- Host：`op_host/rq_uniform_pack_sfa_tiling.cpp`

## 实现文档

`csrc/rq_uniform_pack_sfa/IMPL_DESIGN.md`

## 测试

`tests/ut/test_rq_uniform_pack_sfa.py`
