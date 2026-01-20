# lg2jiuwentool

## 1 工具概述 

LangGraph to openJiuwen Migration Tool - 是一款自动化迁移工具，用于将基于 LangGraph 框架开发的 Agent 代码迁移至 openJiuwen 框架。该工具通过静态代码分析和中间表示（IR）转换，实现跨框架的代码自动迁移。

## 2 应用场景和功能特性

### 2.1 功能特性

- 自动解析 LangGraph 源代码结构（状态、节点、边、工具等）
- 生成符合 openJiuwen 规范的组件和工作流代码
- 保留原有静态函数逻辑
- 生成详细的迁移报告和中间表示（IR）文件

### 2.2 典型场景

#### 场景一：企业框架迁移
企业原有基于 LangGraph 开发的智能客服、数据处理等 Agent 应用，因业务需求需迁移至 openJiuwen 平台。使用本工具可快速完成代码迁移，减少人工改写成本。

#### 场景二：多 Agent 批量迁移
拥有多个 LangGraph Agent 的项目，需要批量迁移至 openJiuwen。工具支持命令行批量处理，提高迁移效率。

#### 场景三：LangGraph生态工具转换成openJiuwen生态工具
扩展和提高openJiuwen生态工具的丰富性和兼容性

## 3. 技术架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        lg2jiuwentool                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Parser    │───▶│  IR Models  │───▶│     Generator       │  │
│  │ (解析器)    │    │ (中间表示)  │    │    (代码生成器)     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                  │                      │             │
│         ▼                  ▼                      ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ AST 分析    │    │  AgentIR    │    │  openJiuwen 代码    │  │
│  │ 状态提取    │    │ WorkflowIR  │    │  组件/工作流生成    │  │
│  │ 节点识别    │    │   ToolIR    │    │  路由函数生成       │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                         Migrator (迁移协调器)                    │
│              CLI Interface / Programmatic API                    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **Parser** | `parser.py` | 解析 LangGraph 源代码，提取状态、节点、边、工具等信息 |
| **IR Models** | `ir_models.py` | 定义平台无关的中间表示数据结构 |
| **Generator** | `generator.py` | 根据 IR 生成 openJiuwen 框架代码 |
| **Migrator** | `migrator.py` | 协调整个迁移流程，生成报告 |


## 4. 核心技术方案

### 4.1 基于 AST 的代码解析

使用 Python 标准库 `ast` 模块进行源代码解析，实现：

```python
# 解析流程
source_code → AST → 结构化信息提取

# 提取内容
- TypedDict 状态类定义
- @tool 装饰的工具函数
- 节点函数（通过 add_node() 调用识别）
- 条件路由函数
- LLM 配置信息
- 图构建调用（add_edge, add_conditional_edges 等）
```

### 4.2 中间表示（IR）设计

采用三层 IR 结构：

```
ParseResult (解析结果)
    ├── state_fields: 状态字段定义
    ├── node_functions: 节点函数信息
    ├── conditional_functions: 条件函数信息
    ├── edges: 边定义
    └── tools: 工具定义

AgentIR (Agent 中间表示)
    ├── name: Agent 名称
    ├── llm_config: LLM 配置
    ├── tools: 工具列表
    ├── workflow: 工作流 IR
    └── state_fields: 状态字段

WorkflowIR (工作流中间表示)
    ├── nodes: 节点列表
    ├── edges: 边列表
    └── entry_node: 入口节点
```

### 4.3 代码生成策略

#### 组件生成
每个 LangGraph 节点函数转换为 openJiuwen 的 `WorkflowComponent`：

```python
# LangGraph
def call_llm(state: AgentState) -> AgentState:
    state["answer"] = llm.invoke(...)
    return state

# 生成的 openJiuwen
class CallLlmComponent(WorkflowComponent, ComponentExecutable):
    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        answer = None
        answer = await self._llm.ainvoke(...)
        return {"answer": answer}
```

#### 条件边转换路由函数
LangGraph 的条件边转换为 openJiuwen 的路由函数：

```python
# LangGraph
def should_continue(state):
    return END if state.get("error") else "next_node"

# 生成的 openJiuwen
def node_router(runtime: Runtime) -> str:
    return 'end' if runtime.get_global_state('node.error') else 'next_node'
```

#### 工具函数转换

保留原有工具逻辑，适配 openJiuwen 的 `@tool` 装饰器：

```python
# LangGraph
@tool
def get_weather(city: str) -> str:
    """获取天气"""
    ...

# 生成的 openJiuwen
@tool(
    name="get_weather",
    description="获取天气",
    params=[Param(name="city", type="string", required=True)]
)
def get_weather(city: str) -> str:
    ...
```


## 5. 迁移映射关系

### 5.1 概念映射

| LangGraph | openJiuwen |
|-----------|------------|
| `StateGraph` | `Workflow` |
| `TypedDict` State | 状态通过 `inputs_schema` 传递 |
| Node Function | `WorkflowComponent` |
| `add_edge()` | `add_connection()` |
| `add_conditional_edges()` | `add_conditional_connection()` + Router |
| `@tool` 装饰器 | `@tool()` 装饰器 + `Param` |

### 5.2 LLM 调用映射

| LangGraph | openJiuwen |
|-----------|------------|
| `ChatOpenAI(...)` | `OpenAIChatModel(...)` |
| `llm.invoke(messages)` | `await self._llm.ainvoke(model_name, messages, ...)` |
| 同步调用 | 异步调用 (`async/await`) |

## 6. 安装和使用指导

### 6.1 项目结构

```

lg2jiuwentool/
  ├── __init__.py   # 包入口，导出公共API
  ├── __main__.py   # 命令行入口，支持python -m运行
  ├── cli.py        # 命令行接口实现
  ├── generator.py  # openJiuwen代码生成器
  ├── ir_models.py  # 中间表示数据模型
  ├── migrator.py   # 迁移流程协调器
  ├── parser.py     # LangGraph代码解析器

```

### 6.2 环境要求

- Python 3.9+
- 无第三方依赖（仅使用 Python 标准库）

### 6.3 安装
将 `lg2jiuwentool` 目录放置在项目中，即可直接使用：

```bash
# 确保在项目根目录下
python -m lg2jiuwentool <source_file> [options]
```


## 7 快速开始

### 7.1 基本用法

```bash
# 迁移单个文件到当前目录
python -m lg2jiuwentool my_agent.py

# 指定输出目录
python -m lg2jiuwentool my_agent.py -o ./output

# 指定输出文件名
python -m lg2jiuwentool my_agent.py -o ./output -n my_migrated_agent

# 显示详细输出
python -m lg2jiuwentool my_agent.py --verbose
```

### 7.2 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `source` | - | LangGraph 源文件路径 | (必需) |
| `--output` | `-o` | 输出目录 | `.` (当前目录) |
| `--name` | `-n` | 输出文件名（不含扩展名） | `{source}_openjiuwen` |
| `--no-report` | - | 不生成迁移报告 | `False` |
| `--no-comments` | - | 不保留原始注释 | `False` |
| `--verbose` | `-v` | 显示详细输出 | `False` |
| `--version` | - | 显示版本号 | - |

### 7.3 示例

```bash
# 迁移天气查询 Agent
python -m lg2jiuwentool langgraph_test/weather_agent_v2.py -o output/

# 输出文件：
#   output/weather_agent_v2_openjiuwen.py      - 迁移后的代码
#   output/weather_agent_v2_openjiuwen_report.txt  - 迁移报告
#   output/weather_agent_v2_openjiuwen_ir.json     - 中间表示
```

### 7.4 编程接口

除命令行外，也可在 Python 代码中调用：

```python
from lg2jiuwentool import migrate, MigrationOptions

# 基本迁移
result = migrate(
    source_path="my_agent.py",
    output_dir="./output"
)

# 自定义选项
options = MigrationOptions(
    preserve_comments=True,
    include_report=True,
    include_ir=True,
    output_name="custom_name"
)
result = migrate("my_agent.py", "./output", options)

# 检查结果
if result.success:
    print("迁移成功！")
    print("生成文件:", result.generated_files)
else:
    print("迁移失败:", result.error)
```

### 7.5 迁移后的手动工作

迁移完成后，请检查以下内容：

1. **LLM 配置**：确认 API Key、API Base 等配置正确
2. **异步调用**：openJiuwen 使用异步模式，确保运行环境支持
3. **组件调用**：检查各组件输入输出是否符合预期
4. **路由逻辑**：验证条件路由的状态访问路径
5. **测试运行**：使用示例输入测试迁移后的代码

### 7.6 输出文件说明

#### 迁移代码 (`*_openjiuwen.py`)

包含：
- 导入语句
- 全局变量
- 工具函数定义
- 组件类定义
- 路由函数
- 工作流构建函数
- 主函数入口

#### 迁移报告 (`*_report.txt`)

包含：
- 源文件和输出文件路径
- 转换统计（节点数、边数、工具数）
- 警告信息
- 待手动处理的任务清单

#### 中间表示 (`*_ir.json`)

包含：
- 解析结果 (parse_result)
- Agent IR (agent_ir)
- 工作流 IR (workflow_ir)

用于调试和分析迁移过程。


## 8 支持的 LangGraph 特性

### 8.1 已支持

- [x] `StateGraph` 状态图定义
- [x] `TypedDict` 状态类
- [x] `add_node()` 节点添加
- [x] `add_edge()` 普通边
- [x] `add_conditional_edges()` 条件边
- [x] `set_entry_point()` 入口点
- [x] `@tool` 工具装饰器
- [x] `ChatOpenAI` 等 LLM 初始化

### 8.2 待支持

- [ ] 子图 (Subgraph)
- [ ] 并行节点
- [ ] Checkpointer 持久化
- [ ] 动态节点添加
- [ ] 流式输出 (Streaming)


## 9 常见问题

### Q: 迁移后代码报错 `NameError: name 'xxx' is not defined`

检查变量是否在所有代码路径中都有定义。工具会自动初始化返回变量为 `None`，但复杂逻辑可能需要手动调整。

### Q: LLM 调用失败

确认：
1. API Key 和 API Base 配置正确
2. 网络可访问 LLM 服务
3. 模型名称正确

### Q: 条件路由不生效

检查路由函数中的状态访问路径是否正确，格式为 `runtime.get_global_state('node_name.field_name')`。


## 相关链接

- [openJiuwen 文档](https://docs.openjiuwen.com)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
