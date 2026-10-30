| 字段 | 值 |
| --- | --- |
| 标题 | 文档中心综述 |
| 修改日期 | 2026-10-28 |
| 关联模块 | 全仓库 |
| 关联文档 | README.md, 10-architecture/00-overview.md, 20-build/00-environment.md, 40-features/rq_uniform/00-overview.md |

# 文档中心

本目录是 `openjiuwen_ascend_ops` 相关文档来源。包在 Ascend NPU 上提供 **rq_uniform** 均匀量化算子族（Full-Attn k8v4/k4v4、SFA 388、MLA latent），对外 `torch.ops.openjiuwen.*`。

## 结构说明

| 编号目录 | 模块 | 综述职责 |
| --- | --- | --- |
| [10-architecture](10-architecture/00-overview.md) | 架构 | 包分层、构建与运行时、注册分发 |
| [20-build](20-build/00-environment.md) | 构建与环境 | 依赖、流水线、编译选项 |
| [30-specifications](30-specifications/00-operators.md) | 规格 | 接口契约总览 |
| [40-features](40-features/00-kv-cache-quantization.md) | 特性 | 量化方案 + **[rq_uniform 算子设计归档](40-features/rq_uniform/00-overview.md)** |
| [50-debugging](50-debugging/00-sanitizer.md) | 调试 | msSanitizer |
| [60-profiling](60-profiling/00-profiling.md) | 性能 | msprof / OPPROF |

## 全部文件（要点）

| 文件 | 内容 |
| --- | --- |
| [40-features/00-kv-cache-quantization.md](40-features/00-kv-cache-quantization.md) | 量化方案综述 |
| [40-features/rq_uniform/00-overview.md](40-features/rq_uniform/00-overview.md) | **rq_uniform 族总览 + 11 算子索引** |
| [40-features/rq_uniform/01…11-*.md](40-features/rq_uniform/00-overview.md) | 各算子说明书+设计（以最新实现为准） |
| [40-features/12-rq-uniform-ops-migration-plan.md](40-features/12-rq-uniform-ops-migration-plan.md) | 迁移计划（历史） |
| [30-specifications/00-operators.md](30-specifications/00-operators.md) | 算子契约总览 |
| [20-build/01-build-guide.md](20-build/01-build-guide.md) | 构建指南 |

## 阅读建议

- **上手**：环境 → 构建 → [量化综述](40-features/00-kv-cache-quantization.md) → [rq_uniform 总览](40-features/rq_uniform/00-overview.md)
- **查实现**：`40-features/rq_uniform/<op>.md` → `csrc/rq_uniform_*/`
- **查契约**：[规格总览](30-specifications/00-operators.md) + `csrc/registration/all_ops.cpp`
- **跑测**：`packages/openjiuwen_ascend_ops/tests/ut/`

## 约定

- 路径以仓库根 `agent-tools/` 为基准；算子包相对 `packages/openjiuwen_ascend_ops/`
- 文档头表格：`标题 / 修改日期 / 关联模块 / 关联文档`
- 算子签名或布局变更时，同步更新 `rq_uniform/` 下对应文件与日期
