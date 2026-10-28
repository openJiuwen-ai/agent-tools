# RqUniformPackK8v4 实现设计

> 产品设计归档：`docs/40-features/rq_uniform/01-pack-k8v4.md`  
> 布局常量：`op_kernel/rq_uniform_pack_k8v4_common.h`（以本文件+头文件为准）

## 计算分工

- `KERNEL_TYPE_MIX_AIC_1_2`：每个 AIC 配两个 AIV；Key/Value 各走独立 stream。
- AIC：从 GM 读原始 Key/Value + `rotation_t`，L0C FP32 MMAD；**Fixpipe `F322F16`** 写出到 GM bridge（`half`，非 FP32）。
- AIV：从 bridge 搬入 UB，在 half 域做均匀量化并写 uint8 cache。
- AIC/AIV 用固定事件同步 ping-pong workspace（`C_READY` / `C_FREE`）。

## Key / Value 量化（默认 Scheme A）

宏 `BR_KEY_UNIFORM_SCHEME`（默认 **1=A**，须与 FIA `TqDecodeKeyTile` 一致）：

| Scheme | Key 编码 |
| --- | --- |
| **A (=1)** | 有符号域非对称：`step=(max-min)/255`，`code_i8=q-128`，`base'=min+128*step` |
| B (=2) | 对称：`s=max(|y|)/127.5`，量化绕 127.5 |
| C (=3) | 遗留：`|y|` 上 7-bit + LSB sign |

Value（始终）：有符号 int4，`vstep=(max-min)/15`，nibble；`+8*vstep` 折进读侧/`vmin` 约定（与 FIA 一致）。  
无 Gram / norm / invNorm。编码走向量指令，metadata 组装不用标量 `GetValue/SetValue`。

`head_size ∈ {128,256}`：cube Haar 仍按 128×128 tile；D=256 为 2×2 mmad。

## AIC/AIV 双缓冲

GM bridge：两块 ping-pong，每块 `32 × PackD` 个 **half**（AIV 各处理不重叠的 `16 × PackD`）。

- buffer 下标 `stream_ordinal & 1`
- AIC 等两路 AIV `C_FREE` 后再覆盖；Fixpipe 后发 `C_READY`
- AIV 等 `C_READY`，MTE2 搬入 UB 后发 `C_FREE`；另有 UB ping-pong 预取下一批

## Cache layout

```text
key_head_base
  + 0                    : key codes, block_size * D bytes (int8 bit-pattern)
  + block_size * D       : base,  block_size * 2
  + block_size * (D + 2) : step,  block_size * 2

value_head_base
  + 0                        : value nibbles, block_size * (D/2)
  + block_size * (D/2)       : vmin,  block_size * 2
  + block_size * (D/2 + 2)   : vstep, block_size * 2
```

16-row tile（D=128）：Key **2112 B**，Value **1088 B**；D=256 按 `(D+4)` / `(D/2+4)` 线性扩展。  
`block_size % 16 == 0`。行独立量化，无跨行 group pack。

## Partial tile 写回

AIV 以 16-row tile 为单位对 metadata 做 read-modify-write；code 按有效行写回。  
ManualKey1 按 `slot_mapping[token]` 解析物理起点，在 16-row / 单 PA block 边界拆分（禁止假定跨 block 物理 slot 连续）。

## Tiling key

| Key | 含义 |
| --- | --- |
| 0 | 默认 |
| 1 | Large-contiguous（长序列 Manual MMAD） |

## 读侧配对

- Prefill：`rq_uniform_fia_paged_k8v4`
- Decode：`rq_uniform_fia_paged_k8v4_msd`

同 2112/1088（或 D 扩展）tile stride；读侧不读 norm、不做 invNorm。
