# rq_uniform UT（契约 / golden / 性能）

每个迁入算子一份 `test_rq_uniform_*.py`，均覆盖：

| 类型 | marker | 是否需要 NPU | 内容 |
| --- | --- | --- | --- |
| 契约 | （无） | 否 | layout 字节 / Scheme A / Haar 折叠等 CPU 公式 |
| 注册 | （无） | 包已安装即可 | `torch.ops.openjiuwen.<op>` 可解析 |
| golden | `npu` + `golden` | 是 | 与 CPU 参考或对偶 op 的数值一致性 |
| perf | `npu` + `perf` | 是 | 固定 shape 计时（默认只采集，不设硬阈值） |

## 运行

```bash
cd packages/openjiuwen_ascend_ops
# 先完成 csrc/build.sh + pip install -e . --no-build-isolation

# 仅 CPU 契约（CI 友好）
pytest tests/ut -m 'not npu' -q

# NPU golden
pytest tests/ut -m 'npu and golden' -q

# NPU perf
pytest tests/ut -m 'npu and perf' -q

# 全量
pytest tests/ut -q
```

算子清单（11）：`pack_k{8,4}v4`、`fia_paged_k{8,4}v4`、`fia_paged_k{8,4}v4_msd`、
`pack_sfa`、`kv_quant_sparse_flash_attention_v2`、`pack_mla_latent`、
`dequant_mla_latent`、`fia_mla_latent`（含 `_paged`）。
