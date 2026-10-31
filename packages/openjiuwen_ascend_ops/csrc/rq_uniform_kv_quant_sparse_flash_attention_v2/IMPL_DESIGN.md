# RqUniformKvQuantSparseFlashAttentionV2 简要说明

> **权威设计文档**：`docs/40-features/rq_uniform/08-kv-quant-sparse-flash-attention-v2.md`  
> 写侧 pack：`../rq_uniform_pack_sfa/IMPL_DESIGN.md` / `docs/40-features/rq_uniform/07-pack-sfa.md`

SFA Schema A **稀疏** Flash Attention：对 `pack_sfa` 的 **388B** PA slots 做 top-K gather + online softmax。

- V2 对 **signed int4** pack 做 dequant（与 pack 默认 tilingKey=1 配对）
- `query`：`[T, Nq, 576]`（nope 512 + rope 64）；输出有效比较维常取前 512
- 典型 mode：`key/value_quant=4`，`sparse_mode=3`，`attention_mode=2`，`quant_scale_repo_mode=1`，`tile_size=128`，`rope_head_dim=64`
- layouts：query `"TND"` / kv `"PA_BSND"`；`key`/`value` 常为 cache 的 `int8` view

勿与 legacy unsigned / 386B pack 混用。

调用：`torch.ops.openjiuwen.rq_uniform_kv_quant_sparse_flash_attention_v2(...)`  
测试：`tests/ut/test_rq_uniform_kv_quant_sparse_flash_attention_v2.py`
