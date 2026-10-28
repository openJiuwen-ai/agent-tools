# RqUniformFiaPagedK4v4Msd 简要说明

> **权威设计文档**：`docs/40-features/rq_uniform/06-fia-paged-k4v4-msd.md`

k4v4 的 **MSD decode** 读侧；IO 与 `fia_paged_k4v4` 同构：

- Us2 路径；`KV_T = int4b_t`；tiling key 族 `QF16_KVINT4_*`
- Key/Value 均为 int4 + fold `+8`；tile **1088/1088**
- `head_size` **仅 128**

调用：`torch.ops.openjiuwen.rq_uniform_fia_paged_k4v4_msd(...)`  
测试：`tests/ut/test_rq_uniform_fia_paged_k4v4_msd.py`
