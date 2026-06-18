## agent-tools 包构建和安装机制
根目录增加packages目录，每个子包在packages下层目录放一个目录
### 1. 构建方式：每个文件夹一个独立的wheel

**是的，packages目录下的每个文件夹都会构建一个独立的wheel包**。具体对应关系如下：

| 源码目录 | wheel包名 |
|---------|----------|
| `packages/openjiuwentools_core` | `openjiuwentools-core-{version}-py3-none-any.whl` |
| `packages/openjiuwentools_infer_router` | `openjiuwentools-infer-router-{version}-py3-none-any.whl` |
| `packages/openjiuwentools_langchain` | `openjiuwentools-langchain-{version}-py3-none-any.whl` |
| ... | ... |

根目录的 `pyproject.toml` 定义一个**元包（meta-package）** `openjiuwentools`，它本身不包含代码，而是通过动态依赖引用所有子包：

```toml
[project]
name = "openjiuwentools"
dynamic = ["version", "dependencies", "optional-dependencies"]

[tool.setuptools_dynamic_dependencies]
dependencies = ["openjiuwentools-core == {version}"]

[tool.setuptools_dynamic_dependencies.optional-dependencies]
infer-router = ["openjiuwentools-infer-router == {version}"]
langchain = ["openjiuwentools-langchain == {version}"]
top = [
  "openjiuwentools-infer-router == {version}",
  "openjiuwentools-langchain == {version}",
  # ... 所有子包
]
```

这种架构设计有以下优势：

| 优势 | 说明 |
|------|------|
| **按需安装** | 用户只安装需要的子包，减少依赖冲突 |
| **模块化开发** | 各子包独立开发、测试、发布 |
| **统一接口** | 所有子包通过 `openjiuwentools` 命名空间提供统一API |
| **插件机制** | 通过 entry points 动态发现和加载组件 |
| **版本一致** | 所有子包共享同一版本号，避免兼容性问题 |

### 2. 用户安装方式

#### 方式一：安装元包（推荐）

```bash
# 安装核心包（最小依赖）
pip install openjiuwentools

# 安装特定功能的子包
pip install "openjiuwentools[infer-router]"
pip install "openjiuwentools[langchain]"

# 安装所有常用子包
pip install "openjiuwentools[top]"
```

#### 方式二：直接安装子包

```bash
# 单独安装某个子包
pip install openjiuwentools-infer-router
pip install openjiuwentools-langchain
pip install openjiuwentools-core
```

#### 方式三：开发模式安装（从源码）

```bash
cd agent-tools

# 使用uv安装（推荐）
uv sync --extra top

# 或使用pip开发模式安装
pip install -e ".[top]"
```

### 3. 安装后的包目录结构

所有子包安装后，代码都合并到统一的 `openjiuwentools` 命名空间下：

```text
site-packages/
├── openjiuwentools/                          # 统一命名空间
│   ├── __init__.py               # 来自 openjiuwentools-core
│   │
│   ├── plugins/                  # 插件目录（各子包贡献）
│   │   ├── router/                  # 来自 openjiuwentools-infer-router
│   │   │   ├── api/
│   │   │   ├── core/
│   │   │   └── schemas/
│   │   ├── langchain/            # 来自 openjiuwentools-langchain
│   │   └── ...
│   │
│   └── meta/                     # 元数据（各子包都有）
│       └── pypi.md
│
├── openjiuwentools_core-{version}.dist-info/
├── openjiuwentools_infer_router-{version}.dist-info/
├── openjiuwentools_langchain-{version}.dist-info/
└── ...
```

### 4. 入口点机制

每个子包通过 Python entry points 注册自己的组件：

```toml
# openjiuwentools_core/pyproject.toml
[project.entry-points.'openjiuwentools.components']
openjiuwentools_llms = "openjiuwentools.llm.register"
openjiuwentools_embedders = "openjiuwentools.embedder.register"

[project.entry-points.'openjiuwentools.cli']
start = "openjiuwentools.cli.commands.start:start_command"

```

```toml
# openjiuwentools_langchain/pyproject.toml
[project.entry-points.'openjiuwentools.components']
openjiuwentools_langchain_client = "openjiuwentools.plugins.langchain.client.client_impl"

[project.entry-points.'openjiuwentools.front_ends']
openjiuwentools_langchain_server = "openjiuwentools.plugins.langchain.server.register_frontend"
```

### 5. 版本管理

所有子包使用统一的版本号，通过 `setuptools_scm` 从 Git 标签自动生成：

```toml
[tool.setuptools_scm]
git_describe_command = "git describe --long --first-parent"
root = "../.."  # 子包指向仓库根目录
```
