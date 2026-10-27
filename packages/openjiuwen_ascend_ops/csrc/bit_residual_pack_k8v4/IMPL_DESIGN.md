# BitResidualPackK8v4 实现设计

## 计算分工

- AIC 直接从 GM 读取原始 Key/Value，加载旋转矩阵并执行 MMAD。
- Fixpipe 输出为 FP32，避免 BF16/FP16 在旋转阶段损失精度。
- 每个 AIC 配合两个 AIV；AIC/AIV 通过固定事件同步 ping-pong workspace。
- AIV 分别编码 Key 和 Value，并写入独立行布局的 KV cache。

## 无 norm 量化

实现不再生成或读取 Gram matrix、norm、invNorm。每个 128 维向量执行：

1. FP32 `abs` 和 `ReduceMax` 得到 `abs_max`。
2. 向量除以 `abs_max`，转换为输入 dtype，再转回 FP32 继续量化。
3. Key 对绝对值做 7-bit min/max 均匀量化，code 为
   `(q7 << 1) | sign`（sign 占 bit0，q7 占 bit1..7）；base/step 乘回 `abs_max`。
4. Value 对有符号值做 4-bit min/max 均匀量化，每个 byte 保存同一行相邻
   两个维度的 nibble；vmin/vstep 乘回 `abs_max`。
5. base、step、vmin、vstep 均以输入 dtype 的 16-bit 表示保存。

量化主体使用向量指令。Key 先将 7-bit quant code 左移 1 bit，再与 sign（bit0）
做 OR，得到 code = (q7<<1)|sign；再将后 64 维左移 8 bit 并与前 64 维做 OR，
得到连续的 64 个 uint16 packed code。Value 同样一次处理 16 行，批量完成 FP32
abs/max、FP16 归一化、min/max、Brcb 和 4-bit 量化。两种编码的 metadata 组装
均不使用标量 `GetValue/SetValue`。

## AIC/AIV 双缓冲

旋转结果的 GM bridge 使用两块 ping-pong buffer。每块保存完整的
`32 * 128` 个 FP32 元素（16 KiB），其中两个 AIV 分别处理不重叠的
`16 * 128` 切片；每个 AIC 对应的 bridge 总计占用 32 KiB。

- buffer 下标为 `stream_ordinal & 1`，相邻 stream 在两块区域间轮转；
- AIC 等待两个 AIV 的 `C_FREE` 后才覆盖对应 buffer；
- AIC 完成 Fixpipe 写出后发布 `C_READY`；
- AIV 等待 `C_READY`，把自己的切片从 GM 搬入 UB 后发布 `C_FREE`。

这样 AIC 可以计算下一批旋转，同时 AIV 编码上一批数据，且不会反复覆盖尚未
搬入 UB 的结果。AIV 另设两块 8 KiB 的 FP32 输入 UB：当前批次进入向量流水后，
MTE2 将下一批从 GM bridge 预取到另一块 UB，再执行当前批次的 cache
read-modify-write。编码 scratch 和 encoded output 保持单份，避免复制约 64 KiB
工作区；它们在写回完成后才由下一批复用。

## Cache layout

每个 head 内先连续保存所有行的 code，再连续保存所有行的两个 metadata 数组：

```text
key_head_base
  + 0                              : key codes, block_size * 128 bytes
  + block_size * 128               : base, block_size * 2 bytes
  + block_size * (128 + 2)         : step, block_size * 2 bytes

value_head_base
  + 0                              : value codes, block_size * 64 bytes
  + block_size * 64                : vmin, block_size * 2 bytes
  + block_size * (64 + 2)          : vstep, block_size * 2 bytes
```

16-row tile 的固定大小分别为：

- Key: `16 * (128 + 2 + 2) = 2112` bytes。
- Value: `16 * (64 + 2 + 2) = 1088` bytes。

Key 每行的 128 个 uint16 code 连续存放；Value 每行的 128 个 4-bit code
压缩为连续 64 bytes。行与行之间不再共享 packed word，也不要求任务按 2-row
或 4-row group 整除。

## Partial tile 写回

metadata 的 GM copy 需要物理块对齐。AIV 以 16-row tile 为单位：

- 读取当前 tile 的 metadata 到对齐 UB scratch；
- 只替换本次覆盖的行；
- 将完整 16-row metadata block 写回；
- code 按实际有效行连续写回。

该 read-modify-write 方案保证非对齐 slot、跨 block 和多请求场景不会破坏旧值。

ManualKey1 调度按 `slot_mapping[token]` 解析每个 chunk 的物理起点，并在
16-row tile / 单 PA block 边界拆分；禁止用 `firstSlot+rowOff` 假定跨 block
物理 slot 连续（reclaim 后 `block_table` 常为非递增 id）。

## Attention decode

`bit_residual_attention_paged_k8v4` 使用相同的 2112/1088 tile stride，按行读取
code 和 16-bit metadata，并将 metadata 转为 FP32 后反量化。Attention 侧不读取
任何 norm 字段，也不执行 invNorm scaling。
