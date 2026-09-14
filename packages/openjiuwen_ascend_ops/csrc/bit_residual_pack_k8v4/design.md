# BitResidualPackK8v4 简要说明

`bit_residual_pack_k8v4` 将 token-major 的 Key/Value 写入 BitResidual
KV cache，并与 `bit_residual_attention_paged_k8v4` 共用以下约定：

- `head_size = 128`，输入与旋转矩阵支持 `fp16`/`bf16`。
- AIC 直接读取原始 Key/Value，并以 FP32 保存旋转结果。
- 不保存 Gram matrix、norm 或 invNorm，也不执行 invNorm scaling。
- 旋转结果先取绝对值、ReduceMax、除以 max，再窄化到输入 dtype；
  随后的量化参数会把 max 折回最终 base/step。
- Key 使用 7-bit 幅值加 1-bit 符号；Value 使用独立的 4-bit 均匀量化。
- cache 以 16-row tile 对齐，`block_size` 必须是 16 的倍数。

每个 head 的物理布局为：

```text
Key:   [block_size * 128-byte codes]
       [block_size * 2-byte base]
       [block_size * 2-byte step]

Value: [block_size * 64-byte packed nibbles]
       [block_size * 2-byte vmin]
       [block_size * 2-byte vstep]
```

因此每 16 行 Key/Value 分别占 2112/1088 bytes。每一行独立量化，
不再存在跨行 group pack。非完整 16-row tile 使用 read-modify-write，避免覆盖
同一 cache block 中未更新的 slot。

实现与同步细节见 `IMPL_DESIGN.md`。
