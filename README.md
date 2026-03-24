# Agent Tools

openJiuwen 平台通用市场：提供 **插件市场服务（marketplace）** 、 **CLI 工具（cli）** 和 **预置插件（plugins）** 等能力。

## 快速开始

本地安装、依赖服务（MySQL / MinIO / OBS）、环境变量与启动命令等，请参考中文安装指导：

**[→ 本地安装指导](docs/zh/安装指导/本地安装/安装指导.md)**

完成安装并启动 **marketplace** 后，可参考 **[中文接口文档](docs/zh/接口文档/v1/插件市场.md)** 进行调试与运行。

---

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
├── docs/                         # 文档
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


更细的 HTTP 接口说明见 **[中文接口文档](docs/zh/接口文档/v1/插件市场.md)**。