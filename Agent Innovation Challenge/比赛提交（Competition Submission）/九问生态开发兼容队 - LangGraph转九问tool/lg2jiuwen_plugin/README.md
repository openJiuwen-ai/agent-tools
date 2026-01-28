# LG2Jiuwen

> 版本：V2.0.0
> 更新日期：2026-01-24

## 1. 工具概述

LangGraph to openJiuwen Migration Tool - 是一款自动化迁移工具，用于将基于 LangGraph 框架开发的 Agent 代码迁移至 openJiuwen 框架。

**核心设计原则：规则优先，AI 兜底**
- 规则能处理的用规则（快速、确定、低成本）
- 规则无法处理的用 AI（语义理解、灵活）

## 2. 功能特性

### 2.1 核心功能

- 自动解析 LangGraph 源代码结构（状态、节点、边、工具等）
- 基于规则的代码转换（状态访问、LLM 调用、工具调用）
- 生成符合 openJiuwen 规范的多文件项目结构
- 自动提取源代码中的示例输入
- 支持单文件和多文件项目迁移
- 生成详细的迁移报告

### 2.2 典型场景

| 场景 | 说明 |
|------|------|
| 企业框架迁移 | 原有 LangGraph Agent 迁移至 openJiuwen 平台 |
| 多 Agent 批量迁移 | 命令行批量处理，提高迁移效率 |
| 生态工具转换 | 扩展 openJiuwen 生态工具的丰富性 |

## 3. 技术架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LG2Jiuwen 迁移工具                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  源代码 → AST → RuleExtractor → [AISemantic] → IR → CodeGenerator      │
│                     │                │           │           │          │
│                     ▼                ▼           ▼           ▼          │
│              ┌───────────┐    ┌───────────┐  ┌──────┐  ┌──────────┐    │
│              │ 转换规则   │    │ AI 处理   │  │ 中间 │  │ 代码生成 │    │
│              │ - 状态访问 │    │ (可选)    │  │ 表示 │  │ - 组件   │    │
│              │ - LLM调用  │    │           │  │      │  │ - 路由   │    │
│              │ - 工具调用 │    │           │  │      │  │ - 工作流 │    │
│              └───────────┘    └───────────┘  └──────┘  └──────────┘    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                    openJiuwen Workflow 实现                              │
│                    CLI / Service API                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 项目结构

```
src/lg2jiuwen_tool/
├── __init__.py              # 模块入口
├── __main__.py              # 命令行入口
├── cli.py                   # CLI 实现
├── service.py               # 服务接口（主入口）
├── workflow/                # openJiuwen 工作流定义
│   ├── migration_workflow.py   # 主工作流
│   └── state.py                # 状态和数据模型
├── components/              # 工作流组件
│   ├── project_detector.py     # 项目检测（单/多文件）
│   ├── file_loader.py          # 文件加载
│   ├── ast_parser.py           # AST 解析
│   ├── rule_extractor.py       # 规则提取+转换（核心）
│   ├── pending_check.py        # 待处理检查
│   ├── ai_semantic.py          # AI 语义理解
│   ├── ir_builder.py           # IR 构建
│   ├── code_generator.py       # 代码生成（核心）
│   └── report.py               # 报告生成
├── rules/                   # 转换规则
│   ├── base.py                 # 规则基类
│   ├── state_rules.py          # 状态访问规则
│   ├── llm_rules.py            # LLM 调用规则
│   ├── tool_rules.py           # 工具调用规则
│   └── edge_rules.py           # 边/路由规则
└── ir/                      # 中间表示
    └── models.py               # IR 数据模型
```

### 3.3 核心模块

| 模块 | 目录/文件 | 职责 |
|------|----------|------|
| **Workflow** | `workflow/` | 定义迁移工作流和状态模型 |
| **Components** | `components/` | 各阶段处理组件 |
| **Rules** | `rules/` | 代码转换规则（状态、LLM、工具） |
| **IR** | `ir/` | 中间表示数据模型 |
| **Service** | `service.py` | 对外服务接口 |

## 4. 迁移映射关系

### 4.1 结构映射

| LangGraph | openJiuwen |
|-----------|------------|
| `StateGraph` | `Workflow` |
| `TypedDict` State | `inputs_schema` 传递 |
| Node Function | `WorkflowComponent` |
| `add_edge()` | `add_connection()` |
| `add_conditional_edges()` | `add_conditional_connection()` + Router |
| `@tool` | `@tool()` + `Param` |
| `END` | `"end"` |

### 4.2 代码转换映射

| LangGraph | openJiuwen |
|-----------|------------|
| `state["key"]` | `inputs["key"]` 或 `runtime.get_global_state("key")` |
| `state.get("key", default)` | `inputs.get("key", default)` |
| `state["key"] = val` | 收集到 `return {"key": val}` |
| `llm.invoke(msgs)` | `await self._llm.ainvoke(model_name, msgs)` |
| `tool.invoke({"arg": val})` | `tool.invoke(inputs={"arg": val})` |
| `tool_map[key].run(arg)` | `invoke_tool(key, arg)` |
| `return state` | `return {"key1": val1, ...}` |
| `return END` | `return "end"` |

### 4.3 路由函数转换

```python
# LangGraph
def should_continue(state):
    if state.get("is_end"):
        return END
    if state.get("loop_count", 0) >= 3:
        return END
    return "think"

# openJiuwen（自动生成）
def judge_router(runtime) -> str:
    # 上游组件输出 → 带节点前缀
    if runtime.get_global_state("judge.is_end"):
        return "end"
    # 全局状态 → 不带前缀
    if (runtime.get_global_state("loop_count") or 0) >= 3:
        return "end"
    return "think"
```

## 5. 安装和使用

### 5.1 环境要求

- Python 3.9+
- openJiuwen 框架（用于运行迁移工作流）

### 5.2 安装

```bash
# 克隆项目
git clone https://github.com/openjiuwen/lg2jiuwen.git

# 安装依赖
pip install -e .
```

### 5.3 命令行使用

```bash
# 迁移单个文件
python -m lg2jiuwen_tool my_agent.py -o ./output

# 迁移项目目录
python -m lg2jiuwen_tool ./my_project/ -o ./output

# 启用 AI 处理
python -m lg2jiuwen_tool my_agent.py -o ./output --use-ai

# 显示详细输出
python -m lg2jiuwen_tool my_agent.py -o ./output -v
```

### 5.4 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `source` | - | 源文件或目录路径 | (必需) |
| `--output` | `-o` | 输出目录 | `./output` |
| `--use-ai` | - | 启用 AI 处理 | `False` |
| `--no-report` | - | 不生成迁移报告 | `False` |
| `--no-comments` | - | 不保留原始注释 | `False` |
| `--verbose` | `-v` | 显示详细输出 | `False` |

### 5.5 编程接口

```python
from lg2jiuwen_tool import migrate_new, MigrationOptions

# 基本迁移
result = migrate_new(
    source_path="my_agent.py",
    output_dir="./output"
)

# 自定义选项
options = MigrationOptions(
    use_ai=True,
    preserve_comments=True,
    include_report=True
)
result = migrate_new("my_agent.py", "./output", options)

# 检查结果
if result.success:
    print("迁移成功！")
    print(f"规则处理: {result.rule_count} 项")
    print(f"AI 处理: {result.ai_count} 项")
    print("生成文件:", result.generated_files)
else:
    print("迁移失败:", result.errors)
```

## 6. 输出文件结构

迁移后生成的项目结构：

```
{agent_name}/
├── __init__.py           # 模块入口
├── config.py             # 配置（LLM、全局变量）
├── tools.py              # 工具函数 + invoke_tool 辅助函数
├── components/           # 组件目录
│   ├── __init__.py
│   └── {node}_comp.py    # 每个节点一个组件
├── routers.py            # 路由函数
├── workflow.py           # 工作流构建
└── main.py               # 主入口（含示例输入）
```

## 7. 支持的 LangGraph 特性

### 7.1 已支持

- [x] `StateGraph` 状态图定义
- [x] `TypedDict` 状态类
- [x] `add_node()` 节点添加
- [x] `add_edge()` 普通边
- [x] `add_conditional_edges()` 条件边
- [x] `set_entry_point()` 入口点
- [x] `@tool` 工具装饰器
- [x] `ChatOpenAI` 等 LLM 初始化
- [x] `tool_map[key].run()` 工具映射调用
- [x] 示例输入自动提取

### 7.2 待支持

- [ ] 子图 (Subgraph)
- [ ] 并行节点
- [ ] Checkpointer 持久化
- [ ] 动态节点添加
- [ ] 流式输出 (Streaming)

## 8. 常见问题

### Q: 迁移后代码报错 `NameError`

检查变量是否在所有代码路径中都有定义。工具会自动初始化返回变量为 `None`，但复杂逻辑可能需要手动调整。

### Q: LLM 调用失败

确认：
1. API Key 和 API Base 配置正确
2. 网络可访问 LLM 服务
3. 模型名称正确

### Q: 条件路由不生效

检查路由函数中的状态访问路径：
- 上游组件输出：`runtime.get_global_state("node.field")` 带前缀
- 全局状态：`runtime.get_global_state("field")` 不带前缀

### Q: 工具映射变量名不对

工具会自动从源代码提取工具映射变量名（如 `tool_map`、`tools` 等），无需手动修改。

## 9. 相关链接

- [openJiuwen 文档](https://docs.openjiuwen.com)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [问题反馈](https://github.com/openjiuwen/lg2jiuwen/issues)
