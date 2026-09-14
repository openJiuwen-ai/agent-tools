| 字段 | 值 |
| --- | --- |
| 标题 | 性能 Profiling |
| 修改日期 | 2026-09-14 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/ |
| 关联文档 | 20-build/01-build-guide.md, 50-debugging/00-sanitizer.md, 40-features/00-kv-cache-quantization.md |

# 性能 Profiling

本文从源仓库 `jiuwen-ak-affinity` 的 profiling 文档裁剪而来，保留 `msprof op` 与 OPPROF 数据格式的通用知识。源仓库的 profile 入口脚本（`tools/op.profile.sh`、`tools/run_attention_512.sh`）与分析脚本（`tools/msi/`）未迁入本仓库；如需完整工作流，回源仓库执行。

## 1. `msprof op` 算子级 profile

`msprof op` 对单个算子做调度级 profile，输出 `OPPROF_*` 目录：

```bash
msprof op \
  --output=ztmp \
  --application=<host 程序，如 e2e 测试或 perf runner> \
  --aic-metrics=Source,PipeUtilization \
  --dump=on --launch-count=1 --warm-up=0 --kill=off
```

- `--aic-metrics`：可选 `Source`（源码行级，需内核带 `.debug_line`，即构建时设 `BIT_RESIDUAL_PACK_K8V4_DEBUG_LINE=ON`）、`PipeUtilization`、`TimelineDetail`。
- 前置：`ASCEND_CUSTOM_OPP_PATH` 指向 `ztmp/cann_ops_install/vendors/vllm-ascend`，并 source 其 `bin/set_env.bash`。

## 2. OPPROF 数据格式

`msprof op` 输出目录命名为 `OPPROF_{YYYYMMDDHHMMSS}_{16位随机大写字母}`，例如 `ztmp/OPPROF_20260707113223_IYMLORHUDLJUBCUU/`。标准产物：

```
OPPROF_xxx/
├── OpBasicInfo.csv
├── PipeUtilization.csv
├── visualize_data.bin
└── dump/
    ├── aicore_binary.o       # device kernel 二进制（含 .debug_line 时可反解）
    ├── DeviceProf1.bin ... DeviceProf7.bin
    ├── duration.bin
    ├── fdata                  # basic-block 执行计数（no_lbr bbcount 格式）
    ├── kernel0.extra.0
    ├── kernel0Stub.o.bbbmap.0 # basic-block id -> offset 映射
    ├── L2Cache.bin
    ├── op_basic_info.txt      # Op Name / Block Dim / SoC Version / Device Id / Pid / Freq / Is MC2 / Is Lccl
    └── pc_start_addr.txt      # kernel 起始 PC 地址
```

### 2.1 OpBasicInfo.csv

表头：`Op Name, Op Type, Task Duration(us), Block Dim, Mix Block Dim, Device Id, Pid, Current Freq, Rated Freq,`。

- `Op Name`：算子调度名（含 hash 和 mix 后缀，如 `BitResidualPackK8v4_..._1_mix_aic`）
- `Op Type`：`mix` / `aic` / `aiv`
- `Task Duration(us)`：算子任务总时长
- `Block Dim` / `Mix Block Dim`：核数与混合核数
- `dump/op_basic_info.txt`：同信息的人类可读 key=value 版本

### 2.2 PipeUtilization.csv

每行一个 `(block_id, sub_block_id)` 组合，`sub_block_id` 形如 `cube0`/`vector0`/`vector1`。列依次为：`aic_time / aic_total_cycles / aic_cube_time(ratio) / aic_scalar_time(ratio) / aic_mte1_time(ratio) / aic_mte2_time(ratio) / aic_mte3_time(ratio, active_bw) / aic_fixpipe_time(ratio, active_bw) / aic_icache_miss_rate`，以及对应的 `aiv_*` 系列，外加 `aic_mte1_active_bw / aic_mte2_active_bw`。`NA` 表示该 block 类型无此指标。

### 2.3 visualize_data.bin

MindStudio Insight 的二进制混合文件（可达数百 MB）。格式特征：

- 文件由二进制头部 + 多段嵌入式 JSON 对象组成
- 每个 JSON 对象以 `{"Cores":` 前缀起始，用括号深度计数解析到匹配的 `}`
- 含 `Files` 字段的 JSON 块即源码行视图：每行含 `source`、`line`、`Instructions Executed`、`process_bytes`、`l2_hit_rate`、`record_count`、`ranges` 等
- 以 `{"advice":` 前缀起始的 JSON 块含 `subblock_detail` 或 `table_detail` 字段

用途：作为 MindStudio Insight UI 的源数据；源仓库的 Python 脚本（`tools/msi/`）绕过 UI 直接抽取行/函数/子块/表级指标。

### 2.4 dump/fdata / aicore_binary.o / bbbmap

- `dump/fdata`：文本格式，首行 `no_lbr bbcount:`，后续行为 `<kernel_name> <offset> <count>` 的基本块执行计数。
- `dump/aicore_binary.o`：device kernel 二进制，若构建时开了 `*_DEBUG_LINE=ON` 则含 `.debug_line` 段。可用 `readelf -S` 验证，用 CANN 自带 `llvm-objdump` 反解地址到源码行。
- `kernel0Stub.o.bbbmap.0`：`BBB id:N Offset: 0x...` 形式的基本块 id 到代码偏移映射。

## 3. 相关

- debug line 构建开关：[构建指南](../20-build/01-build-guide.md)
- sanitizer：[Sanitizer](../50-debugging/00-sanitizer.md)
