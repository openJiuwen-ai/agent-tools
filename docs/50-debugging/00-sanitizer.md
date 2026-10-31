| 字段 | 值 |
| --- | --- |
| 标题 | msSanitizer 调试 |
| 修改日期 | 2026-09-14 |
| 关联模块 | packages/openjiuwen_ascend_ops/csrc/rq_uniform_pack_k8v4/op_host/CMakeLists.txt |
| 关联文档 | 20-build/01-build-guide.md, 60-profiling/00-profiling.md, 40-features/rq_uniform/01-pack-k8v4.md |

# msSanitizer 调试

本文从源仓库 `jiuwen-ak-affinity` 的 sanitizer 文档裁剪而来，去掉了未迁入算子的内容；perf runner（`tools/*_perf.cpp`）尚未迁入本仓库，相关命令需回源仓库执行。

## 1. msSanitizer 是什么

msSanitizer（MindStudio Sanitizer，命令 `mssanitizer`）是基于 Ascend AI 处理器的算子异常检测工具，包含四个子功能：

| 子工具 (`--tool=`) | 解决的问题 |
| --- | --- |
| `memcheck`（默认） | 内存越界、多核踩踏、非对齐访问、内存泄漏、非法释放、分配未使用 |
| `racecheck` | 数据竞争 WAW / WAR / RAW（核内/核间） |
| `initcheck` | 读取未初始化内存导致的脏数据 |
| `synccheck` | SetFlag / WaitFlag 未配对导致的同步失败 |

命令行格式：

```
mssanitizer [<options>] [--] <user_program> [<user_options>]
```

## 2. 典型参数

通用参数：

- `--tool` / `-t`：选择子工具，可多次指定以组合
- `--log-file`：报告输出文件，工具以覆盖方式写入
- `--log-level`：info / warn（默认）/ error
- `--block-id`：单 block 检测模式，0~200
- `--kernel-name`：模糊匹配算子名
- `--full-backtrace=yes`：显示 AscendC API 内调用栈
- `--max-debuglog-size`：单调试日志大小上限（MB，默认 1024）

memcheck 专属：

- `--leak-check=yes`：开启内存泄漏检测
- `--check-unused-memory=yes`：开启分配未使用检测
- `--check-device-heap` / `--check-cann-heap`：Device 侧 / CANN 软件栈内存检测（两者互斥，启用后跳过 Kernel 内检测）

## 3. 编译选项：快速定界 vs 全量检测

两种模式：

- **不修改编译选项（快速定界）**：仅检测 GM 相关搬运指令，仅支持非法读写与非对齐访问，无调用栈。要求算子 `-O2` 且链接阶段加 `-q` 保留符号重定位信息。
- **修改编译选项（全量检测）**：在算子 CMakeLists.txt 中追加 `-g --cce-enable-sanitizer`（或 `-sanitizer`），生成 `.debug_line` 段并插入 msSanitizer 检测指令，异常报告含文件名/行号/调用栈。`-g` 与 `-O0` 同开时需追加 `--cce-ignore-always-inline=false`。

### 3.1 本仓库实际接入

本仓库走"msOpGen 算子工程"场景。算子 `op_host/CMakeLists.txt` 暴露一个 `<OP>_DEBUG_LINE` 缓存变量（默认 OFF，可用同名环境变量覆盖）：

| 环境变量 / CMake 变量 | 追加选项 | 说明 |
| --- | --- | --- |
| `RQ_UNIFORM_PACK_K8V4_DEBUG_LINE` | `-g` | 给 RqUniformPackK8v4 内核生成 `.debug_line` |

`-g` 生成 `.debug_line` 支持源码行调用栈。若需 msSanitizer 全量插桩，可手动在 `add_ops_compile_options` 的 `OPTIONS` 里追加 `--cce-enable-sanitizer`（注意与 `-O0` 同开需 `--cce-ignore-always-inline=false`）。生产构建保持 unset 即可输出干净二进制。

## 4. 标准流程

1. 选择适用场景（Kernel 直调 `<<<>>>` / aclnn 单算子 API / PyTorch TorchAir / Triton-Ascend）。本仓库是 aclnn 单算子 API 场景（源仓库用 `tools/rq_uniform_pack_k8v4_perf` 作为 host runner，该工具未迁入）。
2. 可选：配置 `-g -sanitizer` 编译选项重新编译算子（设 `RQ_UNIFORM_PACK_K8V4_DEBUG_LINE=ON` 后重跑 `csrc/build.sh`）。
3. 运行 `mssanitizer` 命令。
4. 解读 `--log-file` 报告或 `mindstudio_sanitizer_log/` 下的详细日志。

定位要点：

- 报告中的 `serialNo` 可与 `mindstudio_sanitizer_log/mssanitizer_*.log` 中的 `RecordVersion:MEMORY_RECORD, serialNo:N` 行对照，定位是哪一次 `MALLOC/STORE/FREE` 引发的越界。
- `at 0x... on <space>` 的地址可与 `MEMORY_RECORD` 中的 `dstAddr` 比对，确认越界的具体 buffer。
- `in block aiv(N)` 指明在哪个 block 上发生，结合 `--block-id=N` 可单 block 调试。
- 若报告显示 `code in unknown:0`，说明 kernel 未带 `.debug_line`，需重新开启 `*_DEBUG_LINE=ON` 构建。
- 同步问题：先跑 memcheck + racecheck；若 racecheck 无异常但仍有竞争现象，再跑 synccheck 检查前序算子的多余 SetFlag。

## 5. 输出文件

msSanitizer 运行时产生三类产物：

| 文件 | 用途 |
| --- | --- |
| `mindstudio_sanitizer_log/mssanitizer_{TIMESTAMP}_{PID}.log` | 工具运行日志，记录所有 `MALLOC/STORE/FREE` 内存事件（`MEMORY_RECORD` 行）、检测器内部消息（`[I]` 前缀）以及 `ERROR/WARNING` 异常报告 |
| `kernel.{PID}.o` | 算子缓存，用于解析调用栈；正常退出自动清理，Ctrl+C 中止时保留 |
| `tmp_{PID}_{TIMESTAMP}/` | 临时文件夹，存放算子 kernel 二进制；同上自动清理策略 |

用户侧 `--log-file` 指定的文件只包含 `====== ERROR/WARNING:` 开头的异常报告；为空即表示未触发内存异常。

### 5.1 详细日志行格式

`mindstudio_sanitizer_log/mssanitizer_*.log` 的典型行：

- `[I] Init log at [日期时间]`：初始化行
- `[I] set mode param: memcheck`、`[I] set log-file: ...`、`[I] non-option argument: ...`：参数解析记录
- `[I] Use preload lib [.../libmssanitizer_injection.so]`、`[I] Use preload lib [.../libascend_san_stub.so]`：注入的检测库
- `[I] receive device summary. device:N, blockSize:N, blockNum:N, deviceId:N`：设备信息
- `RecordVersion:MEMORY_RECORD, serialNo:N, type:MALLOC|STORE|FREE, coreId:N, srcAddr:0x..., dstAddr:0x..., memInfoSrc:ACL, srcSpace:GM, dstSpace:GM, memSize:N, moduleId:N, lineNo:N, fileName:..., deviceId:N`：每条内存事件
- `[I] add heap block. memInfoSrc: 3, addr: 0x..., size: N` / `[I] free heap block.`：堆块管理
- `====== ERROR: ...` / `====== WARNING: ...`：异常报告（同用户 log）

## 6. 异常报告解读

每种异常报告都包含基本描述行 + 内存位置行 + block/device 行 + pc/serialNo 行 + `#0..#N` 调用栈（仅全量检测模式可见）：

- **非法读写**：`====== ERROR: illegal read of size 224` → `====== at 0x... on GM in <kernel>` → `====== in block aiv(0) on device 0` → `====== code in pc current 0x77c (serialNo:10)` → `#0 <file>:line:col`。`illegal read/write` + 字节数；`on <GM|UB|L0A|L1>`；`aiv(N)` 为 Vector 核 block 索引；`serialNo` 关联 `MEMORY_RECORD`。
- **多核踩踏**：`WARNING: out of bounds of size N`，并标 `when writing data`。
- **非对齐访问**：`ERROR: misaligned access of size N`。
- **内存泄漏**：`ERROR: LeakCheck: detected memory leaks` + 多个 `Direct leak of N byte(s)` + `SUMMARY: N byte(s) leaked in M allocation(s)`。
- **非法释放**：`ERROR: illegal free()`。
- **分配未使用**：`WARNING: Unused memory of N byte(s)` + `SUMMARY`。
- **竞争 RAW/WAR/WAW**：`ERROR: Potential RAW hazard detected at GM in kernel on device 0:` 后跟两个 PIPE 事件（如 `PIPE_MTE2 Write at RAW()+0x0 in block 0 (aiv) on device 0 at pc current 0xa98 (serialNo:14)`）及其调用栈。
- **未配对 SetFlag**：`WARNING: Unpaired set_flag instructions detected` + `from PIPE_S to PIPE_MTE3 in kernel` + block/device + serialNo。
- **寄存器告警**：`[mssanitizer]Warning:Register XXX was not reset to default in block aiv(XXX) on kernel XXX. Expected default value is (XXX), but current value is (XXX)`。

## 7. 带 debug line 的构建与验证

```shell
# 1. 设好 CANN 环境变量（见 20-build/00-environment.md），带 debug 信息构建算子 kernel
export RQ_UNIFORM_PACK_K8V4_DEBUG_LINE=ON
cd packages/openjiuwen_ascend_ops
bash csrc/build.sh 2>&1 | grep -iE "error|rq_uniform_pack_k8v4_tiling" | head -30
readelf -S ../../ztmp/cann_ops_install/vendors/vllm-ascend/op_impl/ai_core/tbe/kernel/ascend910b/rq_uniform_pack_k8v4/*.o | grep debug_line

# 2. 对任意 host 程序（如 e2e 测试）跑 msSanitizer memcheck
source ../../ztmp/cann_ops_install/vendors/vllm-ascend/bin/set_env.bash
export ASCEND_CUSTOM_OPP_PATH=<repo>/ztmp/cann_ops_install/vendors/vllm-ascend
mssanitizer --tool=memcheck --log-file=ztmp/msanitizer.log -- \
    python packages/openjiuwen_ascend_ops/tests/e2e/singlecard/xrx_rq_uniform_k8v4_batch_tiers.py

# 3. 分析报告
grep -c "ERROR:" ztmp/msanitizer.log
```

## 8. 已知坑位

- `No sink op need to be detected`：未设 `ASCEND_CUSTOM_OPP_PATH` 或未 source `set_env.bash` 或未 `aclInit(acl.json)`。
- `Kernel missed debug_line information`：未开 `RQ_UNIFORM_PACK_K8V4_DEBUG_LINE=ON`。
- 检测过慢：加 `--block-id=0` 或 `--kernel-name=RqUniformPackK8v4` 缩小范围。
- msSanitizer 通常使算子执行慢 10~50 倍，宜单点定位而非全量 bench。

## 9. 相关

- 编译开关细节：[构建指南](../20-build/01-build-guide.md)
- rq_uniform pack 算子：[rq_uniform_pack_k8v4](../40-features/rq_uniform/01-pack-k8v4.md)（设计）/ [接口规格](../30-specifications/01-rq-uniform-pack-k8v4.md)
