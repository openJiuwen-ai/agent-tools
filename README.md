# Agent Tools

openJiuwen 平台通用市场。

## 项目结构

```
agent-tools/
├── cli/                          # CLI 命令行工具
├── marketplace/                  # 通用市场服务
│   ├── plugins_market/           # 插件市场模块
│   │   ├── core/                 # 核心配置
│   │   ├── models/               # ORM 模型
│   │   ├── repositories/         # 数据访问层
│   │   ├── routers/              # API 接口层
│   │   ├── schemas/              # 请求/响应模型
│   │   └── services/             # 业务逻辑层
│   ├── main.py
│   └── pyproject.toml
├── plugins/                      # 插件目录
└── .env.example
```

## 模块说明

### marketplace

插件市场服务，提供插件发布、管理等功能。

**三层架构：**

```
routers (接口层) → services (业务层) → repositories (数据层) → models (模型层)
```

| 层级 | 职责 |
|------|------|
| routers | HTTP 请求处理、参数校验、响应封装 |
| services | 业务逻辑、校验、流程编排 |
| repositories | 数据持久化、查询封装 |
| models | ORM 模型定义 |

**快速启动：**

```bash
cd marketplace
uv sync
.venv\Scripts\activate
python main.py
```

## 环境配置

在项目根目录创建 `.env` 文件：

```env
# 数据库配置（可选，默认使用 SQLite）
DB_TYPE=mysql
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=password
STORE_DB_NAME=openjiuwen_market

# 或直接指定数据库 URL
STORE_DB_URL=sqlite:///./data/store.db

# 服务配置
STORE_HOST=0.0.0.0
STORE_PORT=8100
```