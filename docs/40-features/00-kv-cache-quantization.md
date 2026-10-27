| 字段 | 值 |
| --- | --- |
| 标题 | KV cache 量化特性设计 |
| 修改日期 | 2026-09-14 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/bit_residual_pack_k8v4/ |
| 关联文档 | 00-index.md, 30-specifications/00-operators.md, 10-architecture/01-registration.md, 01-bit-residual-pack-k8v4.md |

# KV cache 量化特性设计

本文解释本包 KV cache 量化方案的设计。当前包内只含 BitResidual K8V4 均匀量化方案的 pack 算子；源仓库 `jiuwen-ak-affinity` 中还有 TurboQuant 4-bit codebook 方案、OSCAR INT2 方案及各方案的 attention 侧算子，尚未迁入，此处不展开。

## 一、BitResidual K8V4（均匀量化方案）

### 1.1 设计动机

`bit_residual` 是 `turboquant` 的演进/替代方案（git log `098355a add bit_residual_pack_k8v4`）。设计文档 `IMPL_DESIGN.md` 明确："无 norm 量化"、"不再生成或读取 Gram matrix、norm、invNorm"。动机：

- **去掉 codebook 维护成本**：均匀量化纯算术（max/min/step），无需 16 个码字向量，也不需要 codebook 生成流程。
- **Key 提升到 8-bit 保留精度**：Key 对 attention 精度更敏感，给 Key 更多位宽（8-bit 均匀量化 code）。
- **FP32 累加保精度**：旋转 `x @ R^T` 用 FP32 累加（避免 bf16 精度损失），元数据保留在原始域。

### 1.2 K8V4 命名含义

- **K8**：Key 用 8-bit 非对称均匀量化（Scheme A，默认）：`base=min(y)`、`step=(max-min)/255`、`q=(y-base)/step`（0~255），以 **signed int8 字节** `code_i8 = q - 128` 存储，中心偏移折进 `base' = base + 128*step`。
- **V4**：Value 用 4-bit 均匀量化（signed int4 nibble）。

Key 编码方案由编译期宏 `BR_KEY_UNIFORM_SCHEME` 选择（源仓库中 pack 与 attention/FIA 两侧 CMake 联动，默认 1；本包只含 pack 侧）：1 = A 非对称（上述）、2 = B 对称、3 = C 遗留 LSB-sign。

### 1.3 量化公式

Key（`bit_residual_pack_k8v4_aiv.h`，Scheme A）：

```
y = x @ R^T                  # 旋转，FP32 累加
step = (max(y) - min(y)) / 255
q = (y - min) / step         # 0~255
code_i8 = q - 128            # 以 signed int8 字节写入 cache
base' = min + 128 * step     # 中心偏移折进元数据
```

Value：

```
y = x @ R^T
vstep = (max(y) - min(y)) / 15
idx4 = (y - min) / vstep     # 4-bit signed (int4b_t)
# +8*vstep 偏移折进 vmin；每 byte 存同行相邻两维 nibble
```

元数据：Key 每 16 行存 `base'`(uint16) + `step`(uint16)；Value 每 16 行存 `vmin`(uint16) + `vstep`(uint16)。均保留在原始域。

### 1.4 反量化（attention 侧，源仓库维护）

```
Key:   code_i8 = int8 字节重解释（Python 侧 cache dtype 是 uint8，底层字节语义为 int8）
       decoded_K = base' + code_i8 * step
Value: decoded_V = (vmin + 8*vstep) + idx4_s * vstep
```

attention 侧**不读 norm、不做 invNorm scaling**——这是与 turboquant 的关键差异。

### 1.5 Cache 布局

每 16 行一个 tile（`TQ_BLOCK_ROWS=16`）：

- **Key tile** = `16*(128 + 2 + 2) = 2112` 字节（128 code + 2 base + 2 step，每行）。
- **Value tile** = `16*(64 + 2 + 2) = 1088` 字节（64 nibble + 2 vmin + 2 vstep，每行）。

cache 第 3 维：Key `(block_size/16)*2112`，Value `(block_size/16)*1088`，`block_size` 必须 16 的倍数。

```
Key head region:
  +0                       : key codes,  block_size*128 bytes
  +block_size*128          : base,       block_size*2 bytes
  +block_size*(128+2)      : step,       block_size*2 bytes

Value head region:
  +0                       : value codes,block_size*64 bytes
  +block_size*64           : vmin,       block_size*2 bytes
  +block_size*(64+2)       : vstep,      block_size*2 bytes
```

### 1.6 Pack tiling

`bit_residual_pack_k8v4` 有两种 tiling key：0 = 默认路径，1 = Large-contiguous（key/value 内层 strideHead=128、tokenCount≥128、vec_per_core≥32 时启用，Manual MMAD + ping-pong workspace）。任务类型 `KERNEL_TYPE_MIX_AIC_1_2`，dataCores 按 SoC 实际核数自适应。（历史上的 batch-tiered tiling 16/32/48/64 档位已移除。）

### 1.7 Partial tile 写回

非完整 16-row tile 走 read-modify-write，避免覆盖未更新的 slot（典型场景：序列长度非 16 的倍数时的尾部）。

## 二、与 TurboQuant 4-bit 的对比（摘要）

| 维度 | turboquant 4-bit（未迁入） | bit_residual K8V4（本包） |
| --- | --- | --- |
| 量化方式 | codebook argmin（需 16 个码字向量） | 纯算术 min/max 均匀量化，无 codebook |
| 元数据 | 每行 2 字节 `norm` | Key base'/step（2+2），Value vmin/vstep（2+2） |
| Key 位宽 | 4-bit 索引 | 8-bit 均匀量化（Scheme A，signed int8 存储） |
| Value 位宽 | 4-bit 索引 | 4-bit signed idx |
| 归一化 | 量化前 `x / (norms+eps)` 旋转 | 不归一化，FP32 累加 |
| Cache 每行字节 | 66 | Key 132；Value 68 |
| Cache block stride | `block_size * 66` | Key `(block_size/16)*2112`；Value `(block_size/16)*1088` |
| `block_size` 约束 | % 4 == 0 | % 16 == 0 |
| 反量化 | `y_hat = codebook[idx] * norm`，再反旋转 | `K = base' + code_i8*step`、`V = vmin+idx4_s*vstep`，不读 norm |
| Python pack 参数 | `codebook`、`rotation_t` | 仅 `rotation_t` |

## 三、共同点

- 只支持 `head_size=128`、fp16/bf16 输入、uint8 cache。
- fp16 与 bf16 是两个独立的"实例化"档位（`DataType({ge::DT_FLOAT16, ge::DT_BF16})`），kernel 内通过 `ORIG_DTYPE_KEY`/`DTYPE_KEY` 宏在编译期选择 `half` 或 `bfloat16_t`。
- Pack 算子走 `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC + 2 AIV KFC 组）。
- Pack 与 attention 的 cache 布局 1:1 配对：本包 pack 写出的 cache 只能由源仓库同方案的 `bit_residual_attention_paged_k8v4` 读。两侧 cache 常量成对定义。

## 四、相关

各算子的功能、用法、参数语义、tiling、cache 物理布局、代码位置见本目录下对应文件：

- [01-bit-residual-pack-k8v4](01-bit-residual-pack-k8v4.md)

调用契约（schema/dtype/shape/约束）见 [30-specifications](../30-specifications/00-operators.md)。

其他：

- k8v4 编码/布局回归测试（Scheme A 量化语义 + cache 物理布局 ground truth 注释）：`packages/openjiuwen_ascend_ops/tests/e2e/singlecard/xrx_bit_residual_k8v4_batch_tiers.py`
- 设计文档：`packages/openjiuwen_ascend_ops/csrc/bit_residual_pack_k8v4/IMPL_DESIGN.md`、`design.md`
