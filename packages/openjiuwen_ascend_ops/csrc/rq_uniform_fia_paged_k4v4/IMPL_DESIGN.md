# RqUniformFiaPagedK4v4 简要说明

> **权威设计文档**：`docs/40-features/rq_uniform/04-fia-paged-k4v4.md`

Paged FIA（prefill / continuation），读取 `rq_uniform_pack_k4v4` 写出的 K4V4 cache：

- Key / Value **均为** int4 dequant + fold（`+8*vstep`）
- `head_size` **仅 128**；fp16/bf16；`block_size % 16 == 0`
- 16-row tile：Key/Value 均为 **1088 B**
- Torch schema 与 `fia_paged_k8v4` 同构，仅布局与 dequant 路径不同

Decode 热路径用 `rq_uniform_fia_paged_k4v4_msd`。

调用：`torch.ops.openjiuwen.rq_uniform_fia_paged_k4v4(...)`  
测试：`tests/ut/test_rq_uniform_fia_paged_k4v4.py`
