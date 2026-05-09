# Jiuwen Agent Router 开发指南

## 开发环境设置

### 1. 克隆代码仓库

```bash
git clone https://gitcode.com/openJiuwen/agent-tools.git
cd agent-tools
```

### 2. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

### 3. 安装开发依赖

```bash
pip install -e .[dev]
```

### 4. 安装 pre-commit 钩子

```bash
pre-commit install
```

## 项目结构

```text
llm_tools/
├── main.py                    # 主入口文件
├── requirements.txt           # 生产依赖列表
├── pyproject.toml             # 项目配置文件
├── README.md                  # 项目说明文档
├── docs/                      # 文档目录
├── src/                       # 源代码目录
│   └── openjiuwentools/
│       └── router/
│           ├── api/           # API服务层
│           ├── core/          # 核心功能层
│           └── schemas/       # 数据模型
└── tests/                     # 测试目录
```

## 核心模块说明

### API 服务层 (`src/openjiuwentools/router/api/`)

- `server.py`：FastAPI 服务器实现
- `auth.py`：认证机制实现

### 核心功能层 (`src/openjiuwentools/router/core/`)

- `config.py`：配置管理
- `preprocessor.py`：请求预处理
- `router.py`：路由决策
- `scheduler.py`：调度系统
- `kv_cache.py`：KV缓存管理
- `worker_manager.py`：工作器管理
- `monitoring.py`：监控系统
- `fault_tolerance.py`：容错机制

### 数据模型 (`src/openjiuwentools/router/schemas/`)

- `agent_hints.py`：Agent Hints 数据模型

## 开发流程

### 1. 创建分支

```bash
git checkout -b feature/your-feature-name
```

### 2. 编写代码

遵循项目的代码风格和最佳实践：

- 使用 PEP 8 代码风格
- 编写类型注解
- 添加适当的注释和文档字符串
- 保持代码简洁和可读性

### 3. 运行测试

```bash
pytest
```

### 4. 运行代码检查

```bash
ruff check .
ruff format .
codespell .
typos .
```

### 5. 提交代码

```bash
git add .
git commit -m "Add your feature description"
```

### 6. 创建拉取请求

将分支推送到远程仓库并创建拉取请求。

## 测试

### 单元测试

单元测试位于 `tests/` 目录，使用 pytest 框架。

```bash
pytest tests/unit/
```

### 集成测试

```bash
pytest tests/integration/
```

### 端到端测试

```bash
pytest tests/e2e/
```

## 代码风格

项目使用以下工具确保代码风格一致性：

- **ruff**：代码检查和格式化
- **codespell**：拼写检查
- **typos**：拼写错误修复

### 运行代码格式化

```bash
ruff format .
```

### 运行代码检查

```bash
ruff check .
```

## 文档

### 更新 API 文档

API 文档使用 FastAPI 的自动文档生成功能，可以通过以下地址访问：

```text
http://localhost:8000/docs
http://localhost:8000/redoc
```

### 更新项目文档

项目文档位于 `docs/` 目录，使用 Markdown 格式编写。

## 调试

### 启用调试模式

```bash
python main.py --debug
```

或在配置文件中设置：

```yaml
server:
  debug: true
```

### 使用日志

项目使用 loguru 库记录日志，可以通过配置文件调整日志级别：

```yaml
logging:
  level: DEBUG
```

## 扩展开发

### 添加新的调度策略

1. 在 `src/openjiuwentools/router/core/scheduler.py` 中创建新的调度策略类，继承自 `SchedulingStrategy`
2. 实现 `select_next` 方法
3. 在 `Scheduler` 类中注册新的策略

### 添加新的负载均衡算法

1. 在 `src/openjiuwentools/router/core/router.py` 中创建新的负载均衡算法类，继承自 `LoadBalancingAlgorithm`
2. 实现 `select_worker` 方法
3. 在 `Router` 类中注册新的算法

### 添加新的 API 接口

1. 在 `src/openjiuwentools/router/api/server.py` 中添加新的路由
2. 定义请求和响应模型
3. 实现处理函数

## 贡献指南

### 问题报告

如果发现 bug 或有功能请求，请在 GitHub Issues 中报告：

1. 描述问题的详细信息
2. 提供重现步骤
3. 包含相关日志和错误信息
4. 说明预期行为和实际行为

### 代码贡献

1. Fork 项目仓库
2. 创建特性分支
3. 编写代码和测试
4. 运行所有测试确保通过
5. 创建拉取请求

### 文档贡献

1. 改进现有文档
2. 添加新的文档内容
3. 修复文档中的错误

## 版本管理

项目使用语义化版本控制：

- MAJOR：不兼容的 API 更改
- MINOR：向后兼容的功能添加
- PATCH：向后兼容的错误修复

## 发布流程

1. 更新版本号
2. 运行所有测试
3. 更新 CHANGELOG.md
4. 创建标签
5. 推送到远程仓库

## 联系方式

- 项目主页：<https://gitcode.com/openJiuwen/agent-tools>
- 问题反馈：<https://gitcode.com/openJiuwen/agent-tools/issues>
- 讨论区：<https://gitcode.com/openJiuwen/agent-tools/discussions>
