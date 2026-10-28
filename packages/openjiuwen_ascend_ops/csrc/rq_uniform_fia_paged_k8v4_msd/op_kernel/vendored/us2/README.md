# Vendored open-source Us2 antiquant MSD skeleton.

Source (ops-transformer):

- `attention/incre_flash_attention/op_kernel/arch32/incre_flash_attention_split_Bbn2s2_Us2.h`
- `attention/incre_flash_attention/op_kernel/ifa_public_define.h`

Include: Us2 uses `#include "ifa_public_define.h"` (same directory).

Entry: `../../rq_uniform_fia_paged_k8v4_msd.cpp` + `../../tq_msd_us2_kernel_body.h`.

`BR_MSD_US2_PACK_GATHER=1` (default): read BR K8V4 pack cache (product path).
`BR_MSD_US2_PACK_GATHER=0`: Us2-native PA int8 cache (skeleton gate).
`BR_MSD_US2_APPLY_PI=1` (default): Haar Q@Π / O@Π on product path.
P0: V unpack int4→int8 workspace; Cube `KV_T=int8`. Native int4 (`KT`/`VT` split) is later — see `DESIGN.md` §5.6.

## CopyND2NZ

Keep the **open-source one-shot**. `PACK_GATHER=0` vs stock IFA is 1.0 at `blockSize=128`.
Do not split ND2NZ into 16-row fractal loops.

BR-only: page-run meta gather (one copy per PA page) + V unpack tile 64 (UB bound).
BMM callback: open-source one-shot `CopyND2NZ` (`CopyND2NZPagedB` wrapper).

Identity vs BRFIA: **1.0** at bs=16/32/64/**128** (10/10). Haar **1.0** through bs=128 (10/10).
`kv=32 bs=16` (stepN=2) still ~0.5–0.6.

MIX 1:2 unpack 竞态（even 干活 / odd skip、开源对比、后续优化）：
`docs/mix12_unpack_race.md`。

```bash
python tools/rq_uniform_fia_paged_k8v4/diag_us2_native_vs_ifa.py --device 0   # PACK_GATHER=0
python tools/rq_uniform_fia_paged_k8v4/diag_msd_vs_brfia_decode.py --device 0 --identity --block-size 128 --kv 128
```
