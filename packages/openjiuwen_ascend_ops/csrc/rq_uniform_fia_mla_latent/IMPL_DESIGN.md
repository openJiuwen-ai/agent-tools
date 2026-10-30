# rq_uniform_fia_mla_latent

Decode-side MLA rq_uniform absorb (Scheme A, Path-B)。  
设计归档：`docs/40-features/rq_uniform/11-fia-mla-latent.md`。

Product entry: `torch.ops.openjiuwen.rq_uniform_fia_mla_latent_paged`.
Dense `rq_uniform_fia_mla_latent(codes, scales, …)` remains a host ATen
golden for UT only.

## Path-B (current product)

Pack stores rotated latent \(y \approx c R\) as a Scheme A uint8 page.
Attention runs in that space; Haar multiply stays in the **Torch adapter**,
not inside the AscendC FIA kernel.

```text
adapter:  ql' = ql @ R^T                    # at::matmul
kernel:   dequant page → y  +  FIA MLA(ql', q_pe, y, k_pe)
adapter:  o  = o' @ R                       # at::matmul
```

`W_UK` / `W_UV` are **not** applied by this op. Framework absorbs Q before
the call and applies `W_UV` after; the adapter still accepts `w_uk_t` /
`w_uv` as schema placeholders (ignored when `max_kv > 0`).

Upstream FIA MLA shape (what the kernel sees after the fold):

```text
query      = ql'       [T, H, L=512]
queryRope  = q_pe      [T, H, 64]
key/value  = y         [PA, L=512]   # Scheme A dequant inside kernel
keyRope    = k_pe      [PA, 64]
attentionOut           [T, H, L=512] # before adapter o@R and framework W_UV
```

## Product IO (`*_paged`)

| Tensor | Shape / type | Notes |
|--------|----------------|-------|
| `q_nope` | `[Tq, H, L]` fp16/bf16 | already absorbed (`ql`); adapter folds by `R^T` |
| `q_pe` | `[Tq, H, 64]` | rope |
| `latent_cache` | uint8 `[num_blocks, block_size*(L+4)]` | Scheme A SoA |
| `k_pe_cache` | paged fp rope | PA |
| `block_table` | int32 `[B, max_blocks]` | |
| `actual_seq_q` | host `int[]` `[B]` | TND accum-Q (ValueDepend, ACL-graph safe) |
| `actual_seq_kv` | host `int[]` `[B]` | per-request KV len (ValueDepend) |
| `w_uk_t` / `w_uv` | rank-3 | placeholders; ignored on fuse path |
| `rotation` | `[L, L]` | Haar `R` (used by adapter matmul; unused in kernel) |
| out | `[H, Tq, L]` | after `o@R`, before framework `W_UV` |

Attrs: `block_size`, `scale`, `num_tokens`, `total_kv`, `max_kv`
(`max_kv > 0` selects fused FIA; `max_kv == 0` is dequant-only).

## Layout constraints

- `L = kv_lora_rank = 512`, rope dim `64`
- `block_size` multiple of 16
- `num_kv_heads = 1`

## Design doc

FIA NonQuant MLA pipeline and Scheme A fusion：见同文件下文
「FiaKernelNonQuantMla 逻辑分析与 TQ Scheme A 融合设计」。

## Source map

- Device MLA core: vendored `op_kernel/vendored/arch32/fia_*_nonquant_mla.h`
- Host tiling: `op_host/rq_uniform_fia_mla_latent_tiling.cpp` (vendored `fia_tiling_nonquant_mla.*` removed; unused)
- Scheme A gather: same offsets as `rq_uniform_dequant_mla_latent`
- Path-B Haar: `at::matmul` in `rq_uniform_fia_mla_latent_adapter.cpp`

---

# FiaKernelNonQuantMla 逻辑分析与 TQ Scheme A 融合设计

> 源码基准：ops-transformer **9.2.0**  
> `attention/fused_infer_attention_score/op_kernel/arch22/`  
> - `fia_kernel_nonquant_mla.h`  
> - `fia_block_cube_nonquant_mla.h`  
> - `fia_block_vec_nonquant_mla.h`  
>
> 目标芯片：Atlas A2 / Ascend 910B（arch22）  
> 用途：作为 `rq_uniform_fia_mla_latent` 的 MLA 主算骨架，融合 TurboQuant **Scheme A** 反量化，以及 Scheme A + Haar 旋转矩阵融合方案。

---

## 1. 算子定位与数学形态

### 1.1 在 MLA 推理链路中的位置

```text
[mla_prolog / preprocess]          ← 前处理：W_DQ/UQ/UK、RoPE、写 cache
         ↓
[FiaKernelNonQuantMla]             ← 本算子：稠密 PA 上 Softmax(QK)V（absorb 形态）
         ↓
[可选 W_UV epilog]                 ← stock FIA 外；absorb 产品路径在外层融合
```

官方 README 将 DeepSeek-V2/V3.1 MLA 拆成：**`mla_prolog*` + `fused_infer_attention_score`**。  
本 kernel 只做注意力主算，不做 `W_UK` / `W_UV`。

### 1.2 Absorb MLA 公式（NonQuant）

固定维度（硬编码于 Cube/Vec）：

| 符号 | 值 | 含义 |
|------|----|------|
| `L` / `headDim` | 512 | latent / nope |
| `Dr` / `headDimRope` | 64 | RoPE |
| `D_qk` | 576 | MM1 的 K 维 = `L + Dr` |
| `D_v` | 512 | MM2 的 K 维 = `headDim`（V = K_nope） |

输入（rope-split 模式，产品默认）：

```text
query      [..., L=512]     # absorb 后的 q_nope（= q_c @ W_UK）
queryRope  [..., Dr=64]
key        [PA, L=512]      # latent c（与 value 同缓冲）
keyRope    [PA, Dr=64]
value      = key            # MLA：V 共享 K_nope
```

计算（形状按 **一批 / 一个 kv head**；`N2 = kv_heads = 1`）：

```text
符号:
  M   = S1 * G          # Q 侧有效行数；G = q_heads / kv_heads（MLA 常 G = H）
  S2  = act_seq_kv      # 历史长度（PA 逻辑长）
  L   = 512             # nope / latent
  Dr  = 64              # rope

Q_nope : [M, L ] = [M, 512]
Q_rope : [M, Dr] = [M, 64 ]
K_nope : [S2, L ] = [S2, 512]
K_rope : [S2, Dr] = [S2, 64 ]
V      : [S2, L ] = [S2, 512]     # V = K_nope，同缓冲

# MM1（两次 gemm 在 K 维拼成 576，数值等价于下面两项相加）:
S_nope = Q_nope @ K_nope^T      # [M, 512] @ [512, S2] -> [M, S2]
S_rope = Q_rope @ K_rope^T      # [M,  64] @ [ 64, S2] -> [M, S2]
S      = scale * (S_nope + S_rope)   # [M, S2]

P = OnlineSoftmax(S)            # [M, S2]  （沿 S2 维归一化）

# MM2:
O = P @ V                       # [M, S2] @ [S2, 512] -> [M, 512]
```

典型 decode：`S1 = 1`，`G = H`（如 32）→ `M = H`，输出 `attentionOut` 为 `[H, 512]`（再 `@ W_UV` → `[H, V]`）。

S2 在 kernel 内再切成 `s2Base`（常见 128）做 online softmax；单 tile 时把上式 `S2` 换成 `S2_tile`，`O` 在 Vec2 里跨 tile 归约到最终 `[M, 512]`。

### 1.3 与 TQ Scheme A 的关系（pack 侧）

Pack（`rq_uniform_pack_mla_latent`）在写入 cache 前：

```text
y = c @ R^T          # Haar 旋转到量化空间
code/base/step = SchemeA(y)
```

Slab SoA（每 block）：

```text
latent_cache[block]: uint8[block_size * (L + 4)]
  [0, BS*L)           codes   (int8 语义存于 uint8，decode 时 xor 0x80)
  [BS*L, BS*(L+2))    base    (fp16/bf16)
  [BS*(L+2), BS*(L+4)) step   (fp16/bf16)
```

解码恒等式：

```text
y = base + int8(code) * step
c = y @ R             # 回到原始 latent，再喂 FIA
```

**Stock `FiaKernelNonQuantMla` 假定 key/value 已是 fp latent `c`。**  
TQ 融合的本质：在 Cube 真正 MTE2 读 KV 之前，把 Scheme A（± Haar）插进同一条 S2 tile 流水。

---

## 2. 整体架构

```text
┌─────────────────────────────────────────────────────────────┐
│ FiaKernelNonQuantMla                                         │
│  Process()                                                   │
│    ├─ AIV: vectorService.InitBuffers / AllocEventID          │
│    ├─ AIC: matmulService.InitBuffers / AllocEventID          │
│    ├─ FlashAttention()   ← 主循环（三槽预取流水）              │
│    └─ [可选] FlashDecode() ← 多核切 S2 后再 FD merge          │
│                                                              │
│  matmulService = FiaBlockCubeNonQuantMla                     │
│  vectorService = FiaBlockVecNonQuantMla                      │
│  fdService     = FiaBlockVecFlashDecode…                     │
└─────────────────────────────────────────────────────────────┘
```

核比：MIX `AIC:AIV = 1:2`（与 jiuwen TQ pack/dequant 一致）。

---

## 3. 调度与流水设计

### 3.1 任务轴

外层按 tiling 分核后的三维游标推进：

```text
(bN2Cur, gS1Cur, s2Cur)
  bN2  = batch × kv_heads
  gS1  = Q 侧 M 维（gSize × qSeq），按 mBaseSize 切
  s2   = KV 历史，按 s2BaseSize 切（常见 128）
```

`FlashAttention()` 循环：

1. `ShouldDispatchTask` — 是否还有任务  
2. `GetTaskDealMode` — `CREATE_TASK` / `SKIP` / `DEAL_ZERO` …  
3. `CreateTask` — 填 `RunInfo` 入三槽环形缓存  
4. `ShouldExecuteTask` — 槽位就绪则 `ExecuteTask`

常量：

```text
PRELOAD_NUM = 2
FIA_PRELOAD_TASK_CACHE_SIZE = 3   # 环形 RunInfo[3]
```

### 3.2 三槽交叉流水（核心）

### 3.2 三槽交叉流水（核心）

`ExecuteTask(loop)` 对环形缓存（源码顺序：先本轮 MM1，再上轮 Vec1+MM2，再上两轮 Vec2）：

| 槽 | 含义 | AIC | AIV |
|----|------|-----|-----|
| `extraInfo0` = `loop % 3` | **本轮** tile `t` | **MM1(t)** | — |
| `extraInfo2` = `(loop+2)%3` | **上一轮** tile `t-1` | **MM2(t-1)** | **Vec1(t-1)** |
| `extraInfo1` = `(loop+1)%3` | **上两轮** tile `t-2` | — | **Vec2(t-2)** |

一次 `ExecuteTask(t)` 内实际在跑（**t 时刻一定有 MM1(t)**）：

```text
ExecuteTask(t):
  AIC:  MM1(t)  →  MM2(t-1)
  AIV:  Vec1(t-1) → Vec2(t-2)
```

按调用序号展开（每列 = 一次 ExecuteTask）：

```text
loop:          t                t+1               t+2
AIC:     MM1(t)+MM2(t-1)   MM1(t+1)+MM2(t)  MM1(t+2)+MM2(t+1)
AIV:     Vec1(t-1)+Vec2(t-2) Vec1(t)+Vec2(t-1) Vec1(t+1)+Vec2(t)
```

单 tile `i` 的生命周期（跨 3 次 loop）：

```text
loop i  : MM1(i)
loop i+1: Vec1(i), MM2(i)
loop i+2: Vec2(i)

依赖: MM1(i) → mm1ResGm → Vec1(i) → vec1ResGm(P) → MM2(i) → mm2ResGm → Vec2(i) → Out
```

#### 设计思想：为什么是 3 槽

单 S2 tile 的计算链是 **四段、两类核**：

```text
AIC: MM1(QK) ──► AIV: Vec1(Softmax→P) ──► AIC: MM2(PV) ──► AIV: Vec2(O归约)
```

严格串行要 4 拍。软件流水的目标是：**同一时刻让 AIC、AIV 都有活干**，且每个核上前后两拍尽量无假依赖。

观察依赖后，可合并成 **3 个同时存活的任务状态**：

| 在飞 tile | 本拍在干什么 | 为什么不能合并进别的槽 |
|-----------|--------------|------------------------|
| tile `t` | 只做 MM1 | 产出 `mm1Res`，供下一拍 Vec1 |
| tile `t-1` | Vec1 + MM2 | 同一 tile 的 P→O，共享一份 `RunInfo` |
| tile `t-2` | 只做 Vec2 | 消费 `mm2Res`，写回/归约，释放槽位 |

因此环形缓冲长度 = **同时在飞的 tile 数 = 3**，不是 2（盖不住 MM1∥Vec2 两头），也通常不必 4（Vec1 与 MM2 已捆在同一拍、同一 `RunInfo`）。

`PRELOAD_NUM = 2`（workspace 双缓冲）是另一件事：相邻 tile 的 `mm1Res`/`mm2Res` 不能原地覆盖，和「3 个 RunInfo 元数据槽」配套，但 GM 结果一般只 ping-pong 两份即可（最老 tile 的 Vec2 读完才复用）。

##### 反问：为何不是「MM1→Vec1」+「MM2→Vec2」两槽？

可以这么**在纸上**合并成两条同 tile 的超级阶段：

```text
α(i) = MM1(i) → Vec1(i)     # 算出 P
β(i) = MM2(i) → Vec2(i)     # 算出并写回 O
```

流水深度 2：`ExecuteTask(t)` 做 `α(t)` 和 `β(t-1)`。问题在于 **α/β 内部仍是 AIC→AIV 串行**，跨核重叠只能靠「别的 tile」填空，而 2 个在飞 tile **填不满两个半拍**：

```text
两槽调度（α(t)+β(t-1)）大致变成:
  半拍1: AIC MM1(t)     | AIV 无活可干     ← AIV 气泡
           （Vec1(t) 等 MM1(t)；Vec1(t-1) 已在上拍 α 做完；
             Vec2(t-1) 还要等 MM2(t-1)）
  半拍2: AIC MM2(t-1)   | AIV Vec1(t)      ← 只有这里双核都满
  半拍3: AIC 空         | AIV Vec2(t-1)    ← AIC 气泡
```

三槽把 **不同 tile** 的 AIC 活与 AIV 活对上：

```text
半拍1: AIC MM1(t)     ∥ AIV Vec1(t-1)    # 需要 tile t 与 t-1
半拍2: AIC MM2(t-1)   ∥ AIV Vec2(t-2)    # 还需要 tile t-2
```

要点：

1. **能重叠的是「AIC(tile A) ∥ AIV(tile B)」**，A≠B；同 tile 的 MM1→Vec1 **不能**重叠。  
2. 「MM1→Vec1」「MM2→Vec2」是沿 **同 tile 依赖链** 合并，只拉长阶段，**不减少在飞 tile 数**。  
3. 现设计真正做的合并是 **Vec1+MM2 同 tile、同一拍**（P 就绪立刻 MM2），外加再挂一个更老的 Vec2——这才需要第 3 个存活状态。  
4. 槽数口诀：**同时要喂满双核的半拍数，决定至少几个 tile 在飞**；这里两半拍都要满 → 至少 3 个 tile。

#### 一次 ExecuteTask(t) 里为什么是这个顺序

源码分支顺序（AIC / AIV 各走各的，靠 CrossCore flag 对齐）：

```text
1) AIC: MM1(t)          // AIV 跳过
2) AIV: Vec1(t-1)       // AIC 随后 MM2(t-1)，Wait Vec1 完成
3) AIV: Vec2(t-2)       // 与 AIC 的 MM2(t-1) 可跨核重叠
```

**窍门 1：先发「无跨核等待」的生产者。**  
`MM1(t)` 不依赖本拍 AIV，放最前；AIV 同拍立刻做 `Vec1(t-1)`（依赖的是**上拍** MM1 的 flag）。于是墙钟上：

```text
MM1(t)  ∥  Vec1(t-1)     ← Cube 算新 tile，Vector 消旧 tile 的 S
```

**窍门 2：同 tile 的 Vec1→MM2 捆在一拍。**  
Softmax 完立刻 MM2，中间只隔 `syncV1C2`，不必再占一个独立 loop 槽；`RunInfo` 仍描述同一 `(b,n,s2)`。

**窍门 3：最老的 Vec2 与当前 MM2 跨核重叠。**  
`Vec2(t-2)` 只碰更老的 `mm2Res`/归约状态，和 `MM2(t-1)` 的写端口错开（再加 db workspace），所以放在同一次 ExecuteTask 的后半段。

**窍门 4：环下标用 `(loop+k)%3` 表示「k 拍之前的任务」。**  
`(loop+2)%3` ≡ 上一拍，`(loop+1)%3` ≡ 上两拍——这是深度为 3 的标准 ring 映射，避免显式队列。

**窍门 5：流水深度 = 关键路径级数（合并后）。**  
`depth ≈ 3` ⇒ 填满需约 2 拍 prologue，排空需约 2 拍 epilogue（尾部空转 ExecuteTask / 无效 tile），launch 固定开销与 `S2/s2Base` 无关，长序列才赚回来。

**窍门 6：别和「硬件流水线级数」混为一谈。**  
L1 七缓冲、L0 ping-pong 是 **单次 MM 内部** 的 MTE2/MTE1/Cube 重叠；三槽是 **S2 tile 之间** 的 AIC↔AIV 宏观流水。两层正交：外层 3 槽调度，内层 7-buf 喂 Cube。

交叉核同步（Cube 侧）：

```text
MM1 结束: CrossCoreSetFlag(syncC1V1)     # 通知 Vec1 可读 mm1Res
MM2 开始: CrossCoreWaitFlag(syncV1C2)    # 等 Vec1 写完 P
MM2 结束: CrossCoreSetFlag(syncC2V2)     # 通知 Vec2
          CrossCoreSetFlag(syncC2V1)     # 与 Vec1 侧 N 缓冲握手
```

kernel 内 flag 编号（与 `ConstInfo` 对齐）：

```text
SYNC_V0_C1=6, SYNC_C1_V1=7, SYNC_V1_C2=8, SYNC_C2_V2=9,
SYNC_C2_V1=4, SYNC_V1_NUPDATE_C2=5
```

> 旧图把 `MM1(t)` 画在 `t-1` 列是错的；`MM1` 始终绑在**当前** `loop` 的 `extraInfo0` 上。

### 3.3 MM1 / MM2 片内流水（Cube L1 七缓冲）

注释原文：*use 7 buffer for matmul l1, better pipeline*。QK 和 PV 先后复用同一套 7 块槽，不会同时占着。

```text
L1 512KB
┌──────────────── QP：左矩阵，4 块常驻 ────────────────┬────── KV：右矩阵，3 块轮转 ──────┐
│ 块0          块1          块2          块3          │ 块4        块5        块6       │
│ 第1批行      第1批行      第2批行      第2批行        │ 上一段     当前段     再下一段    │
│ K 前半       K 后半       K 前半       K 后半        │                                  │
└─────────────────────────────────────────────────────┴──────────────────────────────────┘

一块 = 128 × 288 × 2 字节 = 72KB
4×72 + 3×72 = 504KB，放进 512KB
```

#### 为什么 Q 是 4 块，K 是 3 块

4 块不是再叠一层 double buffer。只有「2 批 Query 行」是 double buffer，另一倍是 K 对半之后两半都要留在 L1：

```text
4 = 2 批 Query 行 × 2 半 K
```

576 必须切半，是因为不切就放不下这 7 块。一整刀 K=576 时：

```text
128 × 576 × 2 = 144KB
2 批 Q = 288KB
3 块 K = 432KB
合计     720KB  >  512KB
```

切成 288 后一块变成 72KB，4+3 刚好 504KB。每半 288 的拼法是：

```text
前半 288 = nope[0:256]   + rope[0:32]
后半 288 = rope[32:64]   + nope[256:512]
```

Q 只在第一段 S2 时从 GM 装入这 4 块。S2 按 128 列切开，换下一组 token 时 Q 没变，只换右边的 K：

```text
S[M, S2] = Q[M, 576] × K[S2, 576]^T

列 0~127:    装入 Q，乘 token 0~127 的 K
列 128~255:  Q 不动，只换 token 128~255 的 K
列 256~383:  还是这一份 Q
```

K 的 3 块不是「两半 K 各占一块、再加一块」。两半 K 是先后装入的，同一时刻真正在乘的只有一块。3 块是给相邻三段 K 错开用的：上一段乘法已经发出，但它的 L1 还要往 L0 送，不能盖；当前段正在乘；再下一段要开始从 GM 搬。只有 2 块时，再下一段只能去盖还没读完的那块，预取就要停。

```text
时间 →     t0        t1        t2        t3        t4
块4        搬K0      乘K0      还占着      (空)      搬K3
块5        (空)      搬K1      乘K1      还占着      (空)
块6        (空)      (空)      搬K2      乘K2      还占着

t2 这一列:
  块4  上一段还在 L1→L0，不能盖（这就是上一段那次乘，不是另一次不乘的搬运）
  块5  当前段正在乘
  块6  再下一段正在从 GM 搬
```

两批 Query 行轮流乘同一块 K，不会把 3 块乘成 6 块。

event 环：`mte21QPIds[4] = {E2,E3,E4,E5}`，`mte21KVIds[3] = {E6,E7,E1}`。

#### MM1（QK）切分

```text
S[M, S2] = Q[M, 576] × K[S2, 576]^T
K 维 = 576（特征），N 维 = S2（token），右矩阵按 [N, K] 放
```

L1 一刀是 128×288。576 对半；每一半在 L0 上再按 96 切，因为 288/96=3。96 只属于 QK。

```text
        S2 token
        0~127     128~255    256~383
Q 行
0~127   小块      小块        小块      ← 同一份 Q，只换 K
128~255 小块      小块        小块

每个小块内部:
  K 前半 288 → L0 按 96 乘 3 次
  K 后半 288 → 再乘 3 次，累加进同一块 L0C
  后半结束才写出这块 S[128, 128]
```

```text
for 每 128 个 token:          // N
    for K 的两半:             // 288 + 288
        装这一半 K（3 块轮转里的一块）
        for 每 128 行 Q:      // M，两批行
            用对应那一半 Q（只在第一段 token 从 GM 装）
            L0 上按 96 做 3 次乘
        K 后半结束 → 写出这块 S
```

#### MM2（PV）切分

```text
O[M, 512] = P[M, S2] × V[S2, 512]
K 维 = S2（token），N 维 = 512（输出特征），右矩阵按 [K, N] 放
V = K 的 nope，没有 rope
```

和 QK 不是同一套切法。512 列按 128 切成 4 段；S2 在 K 方向按 256 切，L0 再按 128 切。P 只在第一组 V 列时装入，后面 3 组列复用。

```text
V 的 512 列
┌────────┬────────┬────────┬────────┐
│ 0~127  │128~255 │256~383 │384~511 │
└────────┴────────┴────────┴────────┘

        V 的列
        0~127   128~255  256~383  384~511
P 行
0~127   小块     小块      小块      小块     ← 同一份 P，只换 V 的列
128~255 小块     小块      小块      小块
```

L0 的 128 是一次 Cube 乘的 K，不是把 L1 那一刀一次乘完。L1 这一刀 K=256 时要做 2 次；S2 只剩 128 时做 1 次。

```text
QK  L1 的 K = 288  →  L0 每次 K=96   做 3 次
PV  L1 的 K = 256  →  L0 每次 K=128  做 2 次（尾块 128 则 1 次）
```

```text
for V 的每 128 列:            // N，共 4 次
    for S2 的每 256:          // K
        装这一段 V
        for 每 128 行 P:      // M
            用对应的 P（只在第 1 组 V 列时装入）
            L0 上按 128 做乘，结果累加
        这一段 S2 乘完 → 写出 O[128, 128]
        下一段 S2 用原子加累进同一块 O
```

#### 同一套 L1 槽，按 128 切放不放得下

放得下。QK 用完才轮到 PV，槽大小按 QK 的 128×288 开好。PV 一刀是 128×256：

```text
128 × 256 × 2 = 64KB  <  72KB
```

512 列切成 4 段、S2 按 256 换块，不会把整段 512 或整段 S2 塞进一块。PV 用这块槽时大约空 8KB。
### 3.4 Vector 流水

- **Vec1**：读 `mm1ResGm` → scale / mask → Online Softmax → 写 `vec1ResGm`（P，供 MM2-A）  
- **Vec2**：读 `mm2ResGm` → 与历史 O 做 flash 归约 → 写 `attentionOut`（或 FD 中间量）

UB 侧 input/output 均为 ping-pong（见 §4.2）。

### 3.5 Flash Decode

当单 head 的 S2 被多核切开时：

1. 各核只算自己的 S2 段，写 `accumOut` / `lseMax` / `lseSum`  
2. `SyncAll` 后 AIV 跑 `fdService.FlashDecode` 合并  

TQ 融合时 FD 路径仍可用；**反量化只需保证每个核自己的 S2 段 y_tile 正确**。

---

## 4. 缓存 / 缓冲布局

### 4.1 Cube：L1 / L0

| 资源 | 配置 | 字节 |
|------|------|------|
| L1 总预算 | `l1BufferManager.Init(524288)` | 512 KB |
| L1 QP | `L1_BLOCK × 4` | 288 KB |
| L1 KV | `L1_BLOCK × 3` | 216 KB |
| L0A | `32KB × 2` DB | 64 KB |
| L0B | `32KB × 2` DB | 64 KB |
| L0C | `64KB × 2` | 128 KB |

```text
L1_BLOCK_SIZE = M_L1_SPLIT_SIZE * MM1_K_L1_SPLIT_SIZE * sizeof(Q_T)
              = 128 * 288 * 2 = 73728 ≈ 72KB
```

**结论：FIA 跑起来时 L1/L0 几乎无空闲槽给 TQ 常驻。**

### 4.2 Vector：UB

`FiaBlockVecNonQuantMla::InitBuffers`（量级）：

| Buffer | 大小 |
|--------|------|
| inputBuff1 | 32K × 2 |
| inputBuff2 | 8K × 2 |
| outputBuff1 | 32K |
| outputBuff2 | 4K 或 8K |
| tmpBuff1 | 32K |
| attenMaskTmp | 8K |
| softmax / nValue / cof / aMlaSum … | 1K × `preLoadNum` 若干 |

合计约 **160–180 KB**，贴近单 AIV UB（~192 KB）上限。

**结论：不能与 Scheme A 反量化 UB（~25 KB）长期共驻；要么分相位重建 TPipe，要么在 Vec 空泡期短租一小段（需改 InitBuffers，风险高）。推荐分相位 / 独立小 TPipe 只在 DEQ 相位使用。**

### 4.3 GM Workspace（FIA 自用）

按核、按 `dbWorkspaceRatio = PRELOAD_NUM(=2)` 双缓冲：

```text
mm1ResGm   [core][db][mm1ResSize]     # S = QK^T
vec1ResGm  [core][db][mm1ResSize]     # P（KV_T）
mm2ResGm   [core][db][mm2ResSize]     # O 中间
vec2ResGm  [core][db][mm2ResSize]     # Vec2 暂存
[+ FD: accumOut, lseSum, lseMax]
```

### 4.4 为何空间「看起来不够」

| 需求 | 与 FIA 冲突？ |
|------|----------------|
| W_UK / Haar / W_UV 的 L1/L0 | **冲突** — 与 7-buf 抢 L1 |
| Scheme A 反量化 UB | **冲突** — 与 softmax UB 抢 |
| S2 级 `y_tile` GM | **不冲突** — 额外 GM bridge |
| 整段历史反量化到 GM `[ΣS,L]` | **不推荐** — 显存与带宽爆炸 |

正确策略：**分相位复用片上 + S2-tile GM 乒乓**，而不是扩大 L1。

---

## 5. Scheme A 反量化融合设计

### 5.1 目标语义

在 FIA 每个 S2 tile `i` 上，把：

```text
key/value GM 视图 = 已反量化的 y 或 c 的 [s2Base, L]
```

替换为：

```text
从 latent_cache 按 block_table 收集 → Scheme A decode → (可选 @R) → 供给 MM1-B / MM2-B
```

`keyRope` **不走 Scheme A**（仍为 fp PA cache）。

### 5.2 推荐挂点（相对 stock 流水）

参考 `kv_quant_sparse_flash_attention` 的模式：**Vec 反量化 → 写 L1/GM → Cube 消费**，但主循环仍用 FIA 的稠密 S2 三槽，而不是 sparse gather。

```text
                    ┌──────────────┐
  latent_cache ───► │ Vec0 / DEQ   │── y_tile[i%2] (GM 或直写 L1)
  block_table       │ Scheme A     │
                    └──────┬───────┘
                           │ READY_Y[i]
                    ┌──────▼───────┐
                    │ Cube MM1(i)  │  ← 原 CopyKv 改为读 y_tile
                    │ Cube MM2(i)  │  ← Value 同源 y_tile
                    └──────────────┘
```

两种深度：

| 级别 | 做法 | 正确性 | 性能 |
|------|------|--------|------|
| **L0 外壳** | 每 S2：Vec 写满 `y_tile` GM → 再调原 `ComputeMm1/2`（改 GM 指针） | 易 | tile 间可能空泡 |
| **L1 侵入** | 改 `copyKKropePAGmToL1`：MTE2 源改为 dequant 结果；或 Vec `CopyOutUb2L1` 后 CrossCore 给 Cube | 难 | 可与 MM 重叠 |

产品第一版建议 **L0 + 双缓冲 GM**；对齐后再做 L1 侵入。

### 5.3 S2-tile GM 桥布局

```text
S2 = s2BaseSize（与 FIA tiling 一致，典型 128）
L  = 512
elem = 2B (fp16/bf16)

y_tile[2][S2, L]     # ping-pong，每核一份
size_one = 128 * 512 * 2 = 128 KB
size_pp  = 256 KB / core

可选:
  若 MM1 需要「nope∥rope」在同一物理行：可只存 nope，rope 仍走原 keyRope PA
```

同步：

```text
READY_Y[i]  : Vec 写完 y_tile[i%2]
DONE_FIA[i] : Cube MM2 读完该 tile 的 V（或 MM1+MM2 都结束后释放）

Vec:  wait DONE_FIA[i-2]; dequant → y_tile[i%2]; set READY_Y[i]
AIC:  wait READY_Y[i]; MM1/MM2; set DONE_FIA[i]
```

与 FIA 三槽的关系：

```text
DEQ(i) 应在 MM1(i) 之前完成；
理想重叠：DEQ(i+1) ∥ MM1(i)/Vec1(i-1)/MM2(i-1)
```

Scheme A 单行开销远小于一次 576 维 MM1，**双缓冲通常能盖住**。

### 5.4 Vec 反量化微内核（复用 dequant op）

每 token（与 `rq_uniform_dequant_mla_latent` 相同）：

```text
1. slot → (block, offset) via block_table
2. MTE2: code[L], base[2], step[2]  (SoA 偏移 BS*L / BS*(L+2))
3. y[j] = base + int8(code[j]) * step     # int8 = uint8 ^ 0x80 或等价 cast
4. 若本方案不融 Haar：再 Cube y@R → c，写 y_tile 改为写 c_tile
5. 若融 Haar：直接把 y 当作 FIA 的 key/value（见 §6）
```

UB 临时（仅 DEQ 相位或独立小 pipe）：

```text
code 512B + meta 64B + fp32 row 2KB + out half 1KB + copy tile ≤16KB
≈ 20–25 KB
```

### 5.5 相位总览（与 FIA 分时）

```text
Phase ABSORB_Q   AIC: W_UK, (可选 ql@R^T)     → ql' GM
                 释放 L1，不与 FIA 共驻

Phase FIA+DEQ    重建 FIA InitBuffers
                 循环 S2: DEQ∥三槽 MM1/Vec1/MM2/Vec2
                 key/value → y_tile / c_tile

Phase ABSORB_O   释放 FIA 缓冲
                 AIC: (可选 o'@R), W_UV → out
```

片上「够不够」：**分相位够；与 FIA 共驻不够。**

### 5.6 对 `CopyKv` 的最小侵入接口

建议抽象一层：

```text
struct KvSource {
  enum { FP_PA, TQ_Y_TILE } kind;
  GlobalTensor<KV_T> yTileGm;   // [2, S2, L]
  uint32_t ping;
  // 仍保留 keyRope 的 PA tensor
};
```

在 `ProcessMm1` 的 `kL1` 拷贝处：

- `kind==FP_PA`：现有 `copyKKropePAGmToL1`
- `kind==TQ_Y_TILE`：nope 从 `yTileGm[ping]` 按 `s2BatchOffset+nL1*128` 做 ND→NZ；rope 仍 PA

MM2 的 Value 拷贝同样走 `yTileGm`（无 rope）。

---

## 6. Scheme A 反量化 + 旋转矩阵（Haar）融合设计

### 6.1 两种等价数学路径

Pack：`y = c @ R^T`，量化 `y`。  
注意力需要的是对 **原始** `c` 的点积，还是对 **旋转域** `y` 的点积？

正交 Haar（`R^T @ R = I`）：

```text
q^T @ c  =  q^T @ (y @ R)  =  (q @ R^T)^T @ y
```

因此有两条等价产品路径：

| 路径 | Query 侧 | KV 侧 | FIA 看到的 key/value |
|------|----------|-------|----------------------|
| **A. 先还原 c** | `ql`（原始 absorb 空间） | `y → c = y@R` | `c`（与 stock 相同） |
| **B. 旋转域注意力** | `ql' = ql @ R^T` | 只 Scheme A 出 `y`，**不做 @R** | `y` |

路径 B 把 **整段历史的 `y@R` 矩阵乘消掉**，只在 Q/O 两侧做一次小矩阵（`Tq x H x L`，decode 极小），是 TQ absorb 的优选。

路径 A 与现有 `rq_uniform_dequant_mla_latent` 一致，易做正确性对照，但每个 S2 token 都要一次 `L x L` 乘，贵。

### 6.2 路径 B：融合后的完整图（推荐）

```text
q_nope ──W_UK──► ql ──@R^T──► ql' ─┐
q_pe ─────────────────────────────►├── FiaKernelNonQuantMla
latent_cache ──SchemeA──► y ───────┤   query=ql', key=value=y, keyRope=k_pe
k_pe (PA) ─────────────────────────┘
                │
                ▼
              o' ──@R──► o ──W_UV──► out
```

注意：

- FIA 内部 **不感知** Haar；它只收到「像 fp latent 一样」的 `y`。  
- Softmax 对正交变换下的 score **数值不变**（`q'^T @ y = q^T @ c`）。  
- `keyRope` / `queryRope` **不乘 R**（RoPE 子空间与 latent 正交拼接）。

### 6.3 路径 B 的缓冲与流水

**片上：**

| 阶段 | 谁 | 缓冲 |
|------|----|------|
| `ql @ R^T` | AIC，ABSORB_Q | 小 L1/L0（同 pack：K=N=128 tile），结果写 `ql'_gm` |
| Scheme A | AIV，每 S2 | UB ~25KB → `y_tile` GM |
| FIA | AIC+AIV | 打满 L1/UB（独占） |
| `o' @ R` | AIC，ABSORB_O | 同 Haar tile |
| `W_UV` | AIC，ABSORB_O | 同 W_UK 量级 |

**流水（路径 B）：**

```text
时间 →
AIC: [W_UK][ql@R^T]     [==== FIA 三槽 × N_s2 ====]     [o'@R][W_UV]
AIV:            [DEQ0][DEQ1∥softmax…][DEQ2…] …
GM:  ql' ──────────────── y_tile 乒乓 ────────────── o' ──────── out
```

相对路径 A，**FIA 循环内没有 Cube 做 `y@R`**，DEQ 只做 gather+scale，更容易被 MM1 盖住。

### 6.4 路径 A：DEQ + `y@R` 融在 S2 内（对照 / 兜底）

若必须输出与 stock FIA 相同的 `c` 空间 query：

```text
每 S2 tile:
  Vec: SchemeA → y_ub / y_ws   (S2×L)
  AIC: y @ R → c_tile          (复用 dequant op 的 Cube 段，K=N=128)
  AIC: FIA MM1/MM2 读 c_tile
```

问题：

1. **`y@R` 与 FIA 抢 L1** — 必须在 tile 内串行：先 Haar Cube（自管 L1），Destroy/再 Init FIA 缓冲，或两套逻辑分时占满 L1（latency 差）。  
2. 每个 token `512×512` 乘，S2=128 时计算量与一次 MM1 同量级，**很难藏进空泡**。  
3. 实现复杂度高于路径 B，仅建议作 golden / 调试。

若强行同核流水，只能：

```text
L1 分时：Haar 用完立即还给 FIA 7-buf；
或把 Haar 结果只落 GM c_tile，接受无法与 MM1 重叠。
```

### 6.5 Q/O 侧 Haar 与 Scheme A 的「再融合」边界

可融合：

| 融合项 | 说明 |
|--------|------|
| `W_UK` 与 `R^T` | `W_UK_tq = W_UK @ R^T`，一次 matmul 得到 `ql'`（权重量换，预计算） |
| `R` 与 `W_UV` | `W_UV_tq = R @ W_UV`，一次 matmul 得到 `out` |
| Scheme A 与 FIA S2 | 路径 B：只融 DEQ，不融历史 Haar |

不建议硬融：

| 项 | 原因 |
|----|------|
| Scheme A 直接产出 NZ 进 L1 且同时跑 softmax | UB 打满 |
| 在 L0C 上边 dequant 边 MMAD | 数据类型/流水线模型不匹配 |
| 把 `R` 吸进 FIA 的 K 维切分 | 破坏 576=256+32 的 rope-split 硬编码 |

**预计算权重量换（强烈建议）：**

```text
Host / 初始化:
  W_UK_tq = W_UK @ R^T     # [H,P,L]
  W_UV_tq = R @ W_UV       # [H,L,V]

Decode kernel:
  ql' = q_nope @ W_UK_tq   # 省掉单独 Haar
  ... FIA(key=y) ...
  out = o' @ W_UV_tq       # 省掉单独 o'@R
```

这样 ABSORB_Q/O 各只剩 **一次** matmul，与 Scheme A+FIA 的衔接最干净。

### 6.6 数值与对齐注意点

1. Pack 用 `y=c@R^T` + fixpipe 到 fp16 再 quant；DEQ 用 fp16 `base/step` 还原 `y`，与 FIA 输入 dtype 一致。  
2. `int8(code)` 必须与 pack 的 `q-128` 存储一致（uint8 视图下注意符号）。  
3. 路径 B 下 query 必须乘同一 `R^T`；漏乘会导致 score 错位。  
4. `block_size % 16 == 0`，与 FIA PA / Scheme A SoA 一致。  
5. `kv_heads=1`，`L=512`，`Dr=64` — 不要在第一版做 D 泛化。

---

## 7. 推荐实现里程碑

| 阶段 | 内容 | 验收 |
|------|------|------|
| M0 | Vendor `FiaKernelNonQuantMla`，fp `c` 跑通 absorb I/O | vs 开源 FIA |
| M1 | 路径 A 外壳：独立 DEQ+`y@R` 写 `c_tile`，串行调 FIA | vs `dequant_mla_latent`+FIA |
| M2 | 路径 B：`ql@R^T` + SchemeA→`y_tile` 双缓冲 + FIA | vs M1（正交误差内） |
| M3 | 权重预乘 `W_UK@R^T`、`R@W_UV`，去掉运行时 Haar | 性能 + 对齐 M2 |
| M4 | （可选）L1 侵入：DEQ 写 L1，对齐 kv_quant 流水 | profiler 空泡下降 |

---

## 8. 小结

1. **`FiaKernelNonQuantMla` 是稠密 PA 上的 absorb MLA 主算**：三槽预取（MM1 / Vec1+MM2 / Vec2）+ Cube 7-buf L1 + rope-split K=576。  
2. **L1/UB 已被自身打满**；TQ 只能 **分相位 + GM `y_tile` 乒乓**，不能与 FIA 缓冲长期共驻。  
3. **Scheme A 融合挂在 S2 tile**：Vec gather/dequant → 供 MM1-B/MM2-B；rope 仍走原 PA。  
4. **Scheme A + Haar 优选路径 B（旋转域注意力）**：历史侧只 DEQ 出 `y`，Q/O 侧乘 `R^T`/`R`（或预吸入 `W_UK`/`W_UV`）；避免每 token `y@R`。  
5. 路径 A（还原 `c`）仅作对照；与 FIA 抢 L1 且算力重，不适合作为产品主路径。

---

## 9. 关键源码索引

| 主题 | 文件 |
|------|------|
| 调度 / 三槽 | `fia_kernel_nonquant_mla.h` → `ExecuteTask`, `FlashAttention` |
| Cube MM1/MM2 | `fia_block_cube_nonquant_mla.h` → `ProcessMm1`, `ComputeMm2`, `InitBuffers` |
| Vec Softmax | `fia_block_vec_nonquant_mla.h` → `InitBuffers`, `ProcessVec1L/2L` |
| Scheme A pack/dequant | `rq_uniform_pack_mla_latent`, `rq_uniform_dequant_mla_latent` |
| 产品 IO | 本文件上文 Product IO / Path-B |
| Vec→L1 参考 | `kv_quant_sparse_flash_attention` … `ProcessSparseKv` / `CopyOutKvUb2L1` |
