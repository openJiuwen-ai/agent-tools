# RqUniformPackK4v4 实现设计

> 产品设计归档：`docs/40-features/rq_uniform/02-pack-k4v4.md`  
> 布局常量：`op_kernel/rq_uniform_pack_k4v4_common.h`（**Key/Value 均为 int4，tile 均为 1088**）

## 计算分工

与 `pack_k8v4` 同构的 `KERNEL_TYPE_MIX_AIC_1_2`：

- AIC：原始 Key/Value × `rotation_t`，L0C FP32 MMAD；**Fixpipe `F322F16`** → GM bridge（`half`）。
- AIV：从 bridge 编码；**Key 与 Value 共用 `EncodeValueBatch`（int4）**（见 `EncodeStream`）。
- ping-pong / `C_READY`/`C_FREE` 与 k8v4 相同。

`head_size` **仅 128**（`TQ_PACK_D = 128`）。

## 量化（K 与 V 同路径）

对旋转后 `y`（signed half）逐行：

1. `WholeReduceMax/Min`（**不做 Abs**）
2. `vstep = (max-min)/15`（floor 于 eps）
3. `q ∈ [0,15]` → 平移到 `[-8,7]` → `Cast` 为 `int4b_t` nibble
4. metadata：`vmin` / `vstep`（输入 dtype 的 16-bit 表示）

读侧（FIA/MSD）折 `+8*vstep`：`y ≈ vmin + (signed_nibble + 8) * vstep`。  
无 Gram / norm / invNorm。  
（头文件中仍保留 Scheme A/C 的 `EncodeKeyBatch` 分支供对照；**运行路径不走 Key int8**。）

## AIC/AIV 双缓冲

与 k8v4：每块 bridge `32×128` half；每 AIV `16×128`；UB 双缓冲预取 + 单份 encode scratch。

## Cache layout

`TQ_KEY_ROW_CODE_BYTES = TQ_VAL_ROW_CODE_BYTES = 64`，`*_BYTES_PER_ROW = 68`：

```text
key_head_base / value_head_base（结构相同）
  + 0                    : codes, block_size * 64
  + block_size * 64      : vmin,  block_size * 2
  + block_size * (64+2)  : vstep, block_size * 2
```

16-row tile：**Key 与 Value 均为 `16 * 68 = 1088` bytes**（不是 2112/1088）。  
cache 第 3 维：`(block_size/16) * 1088`。

## Partial tile 写回

与 k8v4：16-row metadata RMW；`slot_mapping` 驱动 ManualKey1 边界拆分。

## 读侧配对

- Prefill：`rq_uniform_fia_paged_k4v4`
- Decode：`rq_uniform_fia_paged_k4v4_msd`

同 1088/1088 tile stride。
