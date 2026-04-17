# Agent Tools Repository

这是一个用于构建和管理智能代理工具的代码仓库，包含了多个与AI代理相关的项目和模块。

## 顶层目录结构

| 目录名称 | 描述 |
|---------|------|
| `packages/` | Python包目录，包含所有可安装的子包 |
| `packages/openjiuwentools_core/` | 核心包，提供基础功能 |
| `packages/openjiuwentools_infer_router/` | 推理路由包，提供KV Cache感知路由系统 |
| `openJiuwen-vllm-affinity/` | OpenJiuwen与vLLM的集成项目，支持亲和性调度 |
| `Agent Innovation Challenge/` | 代理创新挑战赛相关代码和资源 |

## 包管理机制

本项目采用模块化包管理机制，每个子包独立构建为wheel包：

| 源码目录 | wheel包名 |
|---------|----------|
| `packages/openjiuwentools_core` | `openjiuwentools-core-{version}-py3-none-any.whl` |
| `packages/openjiuwentools_infer_router` | `openjiuwentools-infer-router-{version}-py3-none-any.whl` |

### 安装方式

#### 方式一：安装元包（推荐）

```bash
# 安装核心包（最小依赖）
pip install openjiuwentools

# 安装特定功能的子包
pip install "openjiuwentools[infer-router]"

# 安装所有常用子包
pip install "openjiuwentools[top]"
```

#### 方式二：直接安装子包

```bash
# 单独安装某个子包
pip install openjiuwentools-infer-router
pip install openjiuwentools-core
```

#### 方式三：开发模式安装（从源码）

```bash
git clone https://gitcode.com/openJiuwen/agent-tools.git
cd agent-tools

# 使用uv安装（推荐）
uv sync --extra top

# 或使用pip开发模式安装
pip install -e ".[top]"
```

## 开发环境配置

### 1. 系统要求

- Python 3.10 或更高版本
- Git
- 推荐使用Linux或macOS操作系统

### 2. 克隆代码仓库

```bash
git clone https://gitcode.com/openJiuwen/agent-tools.git
cd agent-tools
```

### 3. 安装依赖管理工具

推荐使用pip或uv管理依赖：

```bash
# 使用pip安装uv（可选）
pip install uv
```

### 4. 虚拟环境配置

为整个代码仓库创建一个统一的虚拟环境：

```bash
# 在代码仓库根目录创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装项目依赖
pip install -e ".[top]"
# 或使用uv
uv sync --extra top
```

## pre-commit 配置

### 1. 安装pre-commit

```bash
# 全局安装（推荐）
pip install pre-commit

# 或在虚拟环境中安装
pip install -e ".[dev]"
```

### 2. 安装钩子

在仓库根目录执行：

```bash
pre-commit install
```

### 3. pre-commit配置说明

仓库使用以下pre-commit钩子：

- **ruff**：快速的Python代码检查和格式化工具
- **codespell**：源代码拼写检查器
- **typos**：另一个源代码拼写检查工具
- **markdownlint-cli**：Markdown文件格式检查工具

钩子配置位于根目录的`.pre-commit-config.yaml`文件中。

### 4. 手动运行pre-commit

```bash
# 检查所有文件
pre-commit run --all-files

# 检查特定文件
pre-commit run --files file1.py file2.md
```

## 开发注意事项

### 1. 代码风格

- 遵循PEP 8代码风格规范
- 使用ruff进行代码格式化和检查
- 保持代码简洁、清晰，并添加适当的注释

### 2. 提交规范

- 提交前确保代码通过所有pre-commit检查
- 提交信息要清晰、准确，描述具体的更改内容
- 避免在同一个提交中包含不相关的更改

### 3. 分支管理

- 使用`main`分支作为稳定分支
- 功能开发使用`feature/feature-name`分支
- 修复bug使用`fix/bug-name`分支

### 4. 测试

- 为新功能编写单元测试
- 确保所有测试通过后再提交代码
- 使用pytest运行测试：

  ```bash
  pytest
  ```

### 5. 文档

- 更新相关文档以反映代码更改
- 保持API文档的准确性和完整性
- 使用Markdown格式编写文档

## 项目特定说明

### openjiuwentools-infer-router 包

- 提供LLM推理路由和调度功能
- 支持多种负载均衡算法和调度策略
- 包含完整的API服务层和监控系统
- KV Cache感知路由，支持vLLM和SGLang

### openJiuwen-vllm-affinity 项目

- 实现OpenJiuwen与vLLM的集成
- 支持基于亲和性的调度策略
- 优化大模型推理性能

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目主页：<https://gitcode.com/openJiuwen/agent-tools>
- 问题反馈：<https://gitcode.com/openJiuwen/agent-tools/issues>
- 讨论区：<https://gitcode.com/openJiuwen/agent-tools/discussions>
