| 字段 | 值 |
| --- | --- |
| 标题 | bit_residual_pack_k8v4 接口规格 |
| 修改日期 | 2026-09-14 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/bit_residual_pack_k8v4/op_host/ |
| 关联文档 | 00-operators.md, 40-features/01-bit-residual-pack-k8v4.md |

# bit_residual_pack_k8v4 接口规格

BitResidual K8V4 KV cache 打包算子的调用契约。功能说明、参数语义、cache 物理布局、tiling、代码位置见 [特性设计](../40-features/01-bit-residual-pack-k8v4.md)。

## Torch schema

```
openjiuwen::bit_residual_pack_k8v4(
    Tensor key, Tensor value, Tensor slot_mapping, Tensor query_start_loc,
    Tensor rotation_t, Tensor key_cache, Tensor value_cache,
    int num_reqs, int block_size) -> ()
```

## 张量规格

| 张量 | dtype | shape | 约束 |
| --- | --- | --- | --- |
| `key` / `value` | fp16 / bf16（同 dtype） | `[num_tokens, num_heads, 128]` | `num_heads` 须等于 cache dim1 |
| `slot_mapping` | int32 | 1D，长度 ≥ `num_tokens` | 每 token 的绝对 cache slot |
| `query_start_loc` | int32 | 1D，长度 ≥ `num_reqs+1` | CSR 风格请求边界 |
| `rotation_t` | 同 key | `[128, 128]` | 旋转矩阵 R^T |
| `key_cache` | uint8 | `[num_blocks, num_heads, (block_size/16)*2112]` | 3D、连续 |
| `value_cache` | uint8 | `[num_blocks, num_heads, (block_size/16)*1088]` | dim0/dim1 须与 key_cache 相同 |

## 标量

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| `num_reqs` | int | >0 |
| `block_size` | int | >0 且 % 16 == 0 |

## 常量

- `kHeadSize=128`
- `kBlockRows=16`
- `kKeyBlockStride=2112`
- `kValBlockStride=1088`

（每 16 行 tile：Key = `16*(128 + 2 + 2) = 2112` 字节，Value = `16*(64 + 2 + 2) = 1088` 字节。物理布局见特性设计。）

## 返回

`(key_cache, value_cache)` 元组（in-place 修改后回传）。
