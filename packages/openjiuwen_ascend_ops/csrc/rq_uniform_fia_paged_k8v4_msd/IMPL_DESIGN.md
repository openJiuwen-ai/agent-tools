# RqUniformFiaPagedK8v4Msd 简要说明

> **权威设计文档**：`docs/40-features/rq_uniform/05-fia-paged-k8v4-msd.md`

k8v4 的 **MSD（Multi-Step Decode）** 读侧；IO 与 `fia_paged_k8v4` 相同，读同一 pack 布局：

- 内部走 Us2 IncreFlashAttention；`KV_T = int8_t`；tiling key 族 `QF16_KVINT8_*`
- Key Scheme A int8；Value int4 + fold `+8`
- `head_size ∈ {128, 256}`；典型 serving decode（`nq` 小、KV 长）

调用：`torch.ops.openjiuwen.rq_uniform_fia_paged_k8v4_msd(...)`  
测试：`tests/ut/test_rq_uniform_fia_paged_k8v4_msd.py`
