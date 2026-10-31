# rq_uniform_dequant_mla_latent

Restores MLA latent `c` from the uint8 slab written by `rq_uniform_pack_mla_latent`.
The op does not run attention. The caller passes `c` into the existing `kv_b_proj` and FIA path.

Inputs follow one history chunk:

- `latent_cache`: uint8 `[num_blocks, block_size*(L+4)]`
- `rotation`: `R [L, L]` (pack consumed `R^T`)
- `block_table`: int32 `[num_reqs, max_blocks]`
- `context_lens`: int32 `[num_reqs]`
- `seq_starts`: int32 `[num_reqs]`
- `total_tokens`: host-side `sum(context_lens)` (attr; no device sync)

Output `c` is `[total_tokens, L]`, fp16 or bf16, in request order.

Per token, `slot = seq_starts[r] + t`, then the pack SoA address:

```text
y = base + int8(code) * step
c = y @ R
```

Tiling is MIX `AIC:AIV = 1:2`. A group is 32 output tokens. The two vector
cores gather and dequantize 16 tokens each into a workspace, then the cube
multiplies by `R` with K=N=128. `L` is a multiple of 64 and at most 512.
`block_size` is a multiple of 16.

Correctness: `tests/ut/test_rq_uniform_dequant_mla_latent.py`。  
产品设计归档：`docs/40-features/rq_uniform/10-dequant-mla-latent.md`。
