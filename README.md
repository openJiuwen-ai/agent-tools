# Agent Tools

openJiuwen 平台通用市场：提供 **插件市场服务（marketplace）** 与 **CLI 工具（cli）** 等能力。

## 环境要求

- **Python**：建议 **3.11+**（`marketplace/pyproject.toml` 中 `requires-python >= 3.11.4`）
- **包管理**：推荐使用 [uv](https://github.com/astral-sh/uv)（与下文快速启动一致）；亦可用 `pip` + 虚拟环境

## 项目结构

```text
agent-tools/
├── cli/                          # CLI 命令行工具
├── marketplace/                  # 通用市场服务（FastAPI）
│   ├── plugins_market/           # 插件市场模块
│   │   ├── core/                 # 配置、数据库、鉴权、对象存储
│   │   ├── models/               # ORM 模型
│   │   ├── repositories/         # 数据访问层
│   │   ├── routers/              # HTTP 路由
│   │   ├── schemas/              # 请求/响应模型
│   │   └── services/             # 业务逻辑
│   ├── main.py                   # 服务入口
│   └── pyproject.toml
├── plugins/                      # 插件示例/本地插件目录
├── .env.example                  # 环境变量示例（复制为根目录 .env）
└── README.md
```

## 模块说明

### marketplace

插件市场服务：插件列表、发布（上传 zip）、按版本删除等。

**分层：**

```text
routers (接口层) → services (业务层) → repositories (数据层)
    ↑                                      ↑
 schemas                                models
(请求/响应 DTO)                        (ORM 实体)
```


| 层级           | 职责                              |
| ------------ | ------------------------------- |
| routers      | HTTP 处理、参数与请求头校验、响应封装           |
| services     | 业务规则、zip/plugin.yaml 校验、与对象存储交互 |
| repositories | 数据库 CRUD、查询封装                   |


## 快速启动（marketplace）

在仓库根目录准备好 `**.env**`（见下一节），然后：

```bash
cd marketplace
uv sync
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

python main.py
```

默认由 `main.py` 读取环境变量中的 `**STORE_HOST` / `STORE_PORT**`（未设置则使用内置默认）启动 Uvicorn。

## 环境配置

1. 复制 `**/.env.example**` 为仓库根目录 `**.env**`，按部署环境修改。
2. **marketplace 会从仓库根目录加载 `.env`**（与 `marketplace/plugins_market/core/database.py` 中 `load_dotenv` 一致），请勿只改子目录里的孤立配置文件而忽略根目录 `.env`。

### 常用变量（节选）


| 变量                                                                 | 说明                                                                         |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| `STORE_HOST` / `STORE_PORT`                                        | HTTP 监听地址与端口                                                               |
| `DB_TYPE`                                                          | 设为 `mysql` 时使用下方 `DB_*` 拼装连接串；不设置则走默认 SQLite 逻辑                            |
| `DB_HOST` / `DB_PORT` / `DB_USER` / `DB_PASSWORD`                  | MySQL 连接参数                                                                 |
| `STORE_DB_NAME`                                                    | MySQL 库名（与 `DB_TYPE=mysql` 配合）                                             |
| `MARKET_DB_URL`                                                    | Pydantic Settings 中的 `db_url`，可显式指定 SQLite/MySQL 等连接串（与代码中 `MARKET_` 前缀一致） |
| `AUTH_SERVICE_HOST` / `AUTH_SERVICE_PORT`                          | 账号/Studio 鉴权服务地址（Bearer 发布时会请求用户校验接口）                                      |
| `SYSTEM_ADMIN_TOKEN`                                               | 与请求头 `X-System-Token` 比对，用于运维/系统调用（与 Bearer 二选一，见路由实现）                     |
| `STORAGE_TYPE`                                                     | `MinIO` 或 `OBS`                                                            |
| `MARKET_S3_*` / `MARKET_BUCKET_NAME` / `MARKET_STORAGE_PUBLIC_URL` | 对象存储与公开访问 URL（图标等）                                                         |


完整示例以 `**.env.example**` 为准。

### 对象存储

- 本地/开发常用 **MinIO**（S3 兼容 API）。
- 云端可使用 **华为云 OBS** 等（代码侧按 `STORAGE_TYPE` 与 `MARKET_S3_`* 配置）。

