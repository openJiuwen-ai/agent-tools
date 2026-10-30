# RqUniformFiaPagedK8v4 简要说明

> **权威设计文档**：`docs/40-features/rq_uniform/03-fia-paged-k8v4.md`

Paged Fused Infer Attention（prefill / continuation），读取 `rq_uniform_pack_k8v4` 写出的 K8V4 cache：

- Key：Scheme A int8 dequant；Value：signed int4 dequant + fold
- `head_size ∈ {128, 256}`；fp16/bf16；`block_size % 16 == 0`
- 16-row tile（D=128）：Key **2112 B** / Value **1088 B**
- `rotation_key` / `rotation_value` 为 `[D,D]`，与 pack 的 `R^T` / `R` 配对
- `actual_seq_len_*`：host `int[]`（ValueDepend）

Vendored FIA arch32 + `tq_pack_layout` / `tq_dequant_device`。  
Decode 热路径用 `rq_uniform_fia_paged_k8v4_msd`。

调用：`torch.ops.openjiuwen.rq_uniform_fia_paged_k8v4(...)`  
测试：`tests/ut/test_rq_uniform_fia_paged_k8v4.py`
