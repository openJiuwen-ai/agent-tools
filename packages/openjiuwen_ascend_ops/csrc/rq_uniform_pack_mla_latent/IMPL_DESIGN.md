# rq_uniform_pack_mla_latent

Packs MLA latent `c [T, L]` (`L = kv_lora_rank`) into one uint8 KV-cache slab.
There is no dummy V and no separate codes/scales tensor.

Tiling follows `rq_uniform_pack_k8v4` manual mode:

- cube group: 32 tokens (`TQ_MANUAL_ROT_TILE_M`)
- rotation tile: K=128, N=128, with L0A/L0B ping-pong across K
- vector: two slices of 16 tokens (`TQ_MANUAL_AIV_SLICE_M`)

This is a fused MIX kernel (`AIC:AIV = 1:2`), same split as `pack_k8v4`:
the cube rotates up to 32 tokens and fixpipes fp16, then two vector cores
each encode 16 tokens.

Quant is Key Scheme A after `y = c @ R^T`. The matmul accumulates in fp32 and
is fixpipe-rounded to fp16 before the vector encode. Per token:

- `step = max((max(y) - min(y)) / 255, 1e-6)`
- `q = round((y - min) / step)` in `[0, 255]`
- stored code byte = `int8(q - 128)`
- stored base = `min + 128 * step`
- stored step = `step`

Decode is `y = base + int8(code) * step`. `base` and `step` use the input
dtype (fp16 or bf16, 2 bytes). `L` must be a multiple of 64 and at most 512.

Cache layout, SoA over the block (`block_size % 16 == 0`):

```text
latent_cache[block]: uint8[block_size * (L + 4)]
  [0, block_size*L)                    codes
  [block_size*L, block_size*(L+2))     base
  [block_size*(L+2), block_size*(L+4)) step
```

`slot_mapping` is int32. A negative slot is left untouched.

Correctness: `tests/ut/test_rq_uniform_pack_mla_latent.py`。  
产品设计归档：`docs/40-features/rq_uniform/09-pack-mla-latent.md`。
