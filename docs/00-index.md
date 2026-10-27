| 字段 | 值 |
| --- | --- |
| 标题 | 文档中心综述 |
| 修改日期 | 2026-09-14 |
| 关联模块 | 全仓库 |
| 关联文档 | README.md, 10-architecture/00-overview.md, 20-build/00-environment.md, 40-features/00-kv-cache-quantization.md, 30-specifications/00-operators.md |

# 文档中心

本目录是 `openjiuwen_ascend_ops` 相关文档的来源（另有 `openjiuwen_package_design.md` 记录仓库整体包设计）。`openjiuwen_ascend_ops` 在 Ascend NPU 上提供融合量化算子，把 KV cache 以低比特格式（K8V4 均匀量化）打包进 PagedAttention 的物理 cache。当前包内只含打包（pack）算子 `bit_residual_pack_k8v4`；配套的 attention 侧算子仍在源仓库 `jiuwen-ak-affinity` 维护，尚未迁入。目标平台 Ascend 910B（CANN 8.5.1，PyTorch 2.9.0，torch_npu 2.9.0，Python 3.11，aarch64）。

> 本文档树从源仓库 `jiuwen-ak-affinity` 的 docs 裁剪而来：只保留 `bit_residual_pack_k8v4` 相关内容，综述类文档按源仓库约定改写为职责描述，不写死算子名与数量。具体算子随版本演进可能增减，请以 [30-specifications](30-specifications/00-operators.md) 与 [40-features](40-features/00-kv-cache-quantization.md) 下的实际文件为准。

## 结构说明

文档采用**总分结构**，按编号目录组织，编号即推荐阅读顺序（`find docs | sort` 即得）。每个模块下 `00-*.md` 是该模块综述，后续编号文件是该模块的具体主题。模块间**解耦**：综述只讲职责与组织方式，不写死具体算子名与数量；新增/调整算子时，只动 `40-features`（说明书+设计）与 `30-specifications`（契约）下的对应文件，不必回改各模块综述。

| 编号目录 | 模块 | 综述职责 |
| --- | --- | --- |
| [10-architecture](10-architecture/00-overview.md) | 架构 | 包分层、目录约定、构建与运行时数据流、算子注册与分发 |
| [20-build](20-build/00-environment.md) | 构建与环境 | 硬件软件依赖、构建流水线、编译选项 |
| [30-specifications](30-specifications/00-operators.md) | 规格 | 纯接口契约（schema/dtype/shape/约束/标量/常量/返回），面向调用方与测试 |
| [40-features](40-features/00-kv-cache-quantization.md) | 特性 | 量化方案设计综述 + 各算子的功能/用法/参数语义/tiling/cache 布局/代码位置（说明书+实现设计） |
| [50-debugging](50-debugging/00-sanitizer.md) | 调试 | msSanitizer 工作流与 crash 复现 |
| [60-profiling](60-profiling/00-profiling.md) | 性能 | msprof op / OPPROF / 分析脚本 |

> **features 与 specifications 的区分**：specifications 是「接口契约」（一文件一算子，只记录调用事实，纯、稳定，面向用户和测试）；features 是「方案设计 + 算子说明书」（综述讲量化方案思想/动机/对比，子文件讲各算子的功能/用法/参数语义/tiling/cache 布局/代码位置，随实现演进）。

## 全部文件

| 文件 | 内容 |
| --- | --- |
| [10-architecture/00-overview.md](10-architecture/00-overview.md) | 整体架构、包分层、目录约定、数据流 |
| [10-architecture/01-registration.md](10-architecture/01-registration.md) | 算子注册与分发（openjiuwen 命名空间 / aclnn / EXEC_NPU_CMD） |
| [20-build/00-environment.md](20-build/00-environment.md) | 硬件软件依赖、CANN / torch_npu、环境变量 |
| [20-build/01-build-guide.md](20-build/01-build-guide.md) | 构建流水线与编译选项 |
| [30-specifications/00-operators.md](30-specifications/00-operators.md) | 算子接口规格总览 + 通用约定 |
| [30-specifications/01-bit-residual-pack-k8v4.md](30-specifications/01-bit-residual-pack-k8v4.md) | bit_residual_pack_k8v4 接口规格 |
| [40-features/00-kv-cache-quantization.md](40-features/00-kv-cache-quantization.md) | 量化方案综述：K8V4 均匀量化设计 |
| [40-features/01-bit-residual-pack-k8v4.md](40-features/01-bit-residual-pack-k8v4.md) | BitResidual K8V4 打包算子（说明书+设计） |
| [50-debugging/00-sanitizer.md](50-debugging/00-sanitizer.md) | msSanitizer 工作流、crash 复现、报告解读 |
| [60-profiling/00-profiling.md](60-profiling/00-profiling.md) | msprof op / OPPROF 数据格式 / 分析脚本 |

## 阅读建议

- **首次上手**：[环境](20-build/00-environment.md) → [构建指南](20-build/01-build-guide.md) → [量化方案综述](40-features/00-kv-cache-quantization.md) → [算子规格](30-specifications/00-operators.md)。
- **接入新算子**：[架构](10-architecture/00-overview.md) → [注册机制](10-architecture/01-registration.md) → [规格总览](30-specifications/00-operators.md)。
- **查接口契约**：[算子规格](30-specifications/00-operators.md)（schema/约束，面向调用与测试）。
- **理解实现/调优**：[特性设计](40-features/00-kv-cache-quantization.md) → 各算子 features 子文件（tiling/cache 布局/代码位置）。
- **排障内存问题**：[Sanitizer](50-debugging/00-sanitizer.md)。
- **性能调优**：[Profiling](60-profiling/00-profiling.md)。

## 约定

- 路径引用以仓库根 `agent-tools/` 为基准；涉及算子包时相对 `packages/openjiuwen_ascend_ops/`（正文中 `csrc/...`、`pysrc/...` 均相对该包根）。
- 涉及源码位置时，以 `文件相对路径:行号` 标注，便于跳转。
- 每个文档顶部有表格文件头，记录 `标题 / 修改日期 / 关联模块 / 关联文档`。
- 文档与代码同步演进：算子签名、编译开关、脚本参数若发生变化，请一并更新对应算子的 features（说明书+设计）与 specifications（契约）文件与文件头日期；**综述类文档尽量只改职责描述，不写死具体算子名与数量**，避免算子增减时连带修改。
