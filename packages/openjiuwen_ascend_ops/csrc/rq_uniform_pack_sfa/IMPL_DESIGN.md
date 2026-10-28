# rq_uniform_pack_sfa (P0 write)

Packs rotated MLA nope `[T, 512]` (+ optional rope `[T, 64]`) into **388-byte**
fused SFA KV slots (`PA_BSND [Bn, Bs, 1, 388]`).

## Slot layout

| Offset | Bytes | Content |
| --- | --- | --- |
| `[0, 256)` | 256 | 4-bit nope codes (low nibble = even dim) |
| `[256, 384)` | 128 | rope × 64 as BF16/FP16 (**no** `rope/=scale` pre-divide) |
| `[384, 388)` | 4 | FP16 `scale` + FP16 `offset` |

Dequant: `y = offset + uint4(code) * scale` (no codebook).

**Forbidden:** `cann_ops_nn.turbo_quant`, 386B TQ pack, centroid LUT.

## Status

| Milestone | State |
| --- | --- |
| M1 / UT | Done — `tests/ut/test_rq_uniform_pack_sfa.py` |
| NPU legacy (tilingKey=0) | Kept — scalar EncodeOne, MIX, **int64** slots (`JIUWEN_PACK_SFA_LEGACY=1`) |
| NPU V2 (tilingKey=1) | Default — `*_vector_v2.h`, AIV-only, **int64** slots |
| Register in `all_ops.cpp` / CMake / setup.py | Done |

Constants: `op_kernel/rq_uniform_pack_sfa_common.h`。  
产品设计归档：`docs/40-features/rq_uniform/07-pack-sfa.md`。

## Reference API (Python)

```python
import openjiuwentools.openjiuwen_ascend_ops  # registers torch.ops.openjiuwen.*
import torch

# Slot layout: 388 bytes (256 codes + 128 rope + 4 scale/offset)
torch.ops.openjiuwen.rq_uniform_pack_sfa(nope, rope, slot_mapping, kv_cache, block_size)
```

