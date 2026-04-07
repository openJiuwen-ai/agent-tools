# Windows 系统安装指导（Docker 方式）

本文用于说明在 **Windows** 上通过 Docker 方式启动 `marketplace-tools`（插件市场服务）。

## 1. 环境准备

- 操作系统：Windows 10 及以上
- Docker：建议使用 **Docker Desktop**（推荐 WSL 2 后端）
- 需要确认你已能正常运行 Docker（例如执行 `docker info` 能看到输出）

### MySQL 数据库要求（必选）

- 当前 `marketplace-tools` 默认以 MySQL 作为主数据存储（`DB_TYPE=mysql`）。
- 请确保你已启动可访问的 MySQL，并且 **容器内可以连通**该 MySQL。
- 请在 MySQL 中预先创建数据库与账号，并确保 `STORE_DB_NAME`、`DB_USER`、`DB_PASSWORD` 与实际配置一致。

## 2. 准备环境变量文件

在仓库根目录（`agent-tools`）下创建 `.env.docker`，做法是将代码仓里的 `.env.example` 复制一份并按需修改：

```powershell
Copy-Item ".env.example" ".env.docker"
```

然后编辑 `.env.docker` 填写你的 MySQL / 对象存储 / 鉴权服务等配置（其中鉴权 token 需要与服务端一致）。

### MySQL

`marketplace-tools` 在容器内连接 **宿主机** 上的 MySQL 时，可将 `DB_HOST` 设为 `host.docker.internal`（Docker Desktop on Windows 常见写法，与 MinIO 小节一致）。

#### 1) 预先在 MySQL 中建库与授权

若 `DB_TYPE=mysql`，请先在 MySQL 中创建与 `.env.docker` **一致**的数据库名、用户名与密码，否则服务启动后会连接失败。

示例 SQL（请与你的 `STORE_DB_NAME`、`DB_USER`、`DB_PASSWORD` 对齐）：

```sql
CREATE DATABASE IF NOT EXISTS openjiuwen_market
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;

CREATE USER IF NOT EXISTS 'your_user'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON openjiuwen_market.* TO 'your_user'@'%';
FLUSH PRIVILEGES;
```

#### 2) 在 `.env.docker` 配置 MySQL（示例）

将 `your_user` / `your_password` 换成你在 MySQL 中创建的实际账号；`STORE_DB_NAME` 须与已建库名一致。

```env
DB_TYPE=mysql
DB_HOST=host.docker.internal
DB_PORT=3306
DB_USER=your_user
DB_PASSWORD=your_password
STORE_DB_NAME=openjiuwen_market
```

若 MySQL 部署在其它机器或容器内，请将 `DB_HOST`（及必要时 `DB_PORT`）改为可达地址。

### 对象存储 / 向量存储：本地 MinIO 对接（建议先跑起来）

`marketplace-tools` 需要通过 S3 兼容接口访问对象存储（本地默认 MinIO）。因此你需要在本机先启动一个 `minio` 容器，并创建好对应的 Bucket。

#### 1) 启动本地 MinIO

在宿主机执行（PowerShell）示例（直接使用 MinIO 默认账号密码；若你在 `docker run` 里不显式传 `MINIO_ROOT_USER/MINIO_ROOT_PASSWORD`，MinIO 会使用镜像默认值 `minioadmin/minioadmin`）：

```powershell
docker run -d --name minio `
  -p 9000:9000 `
  -p 9001:9001 `
  -v "minio-data:/data" `
  minio/minio server /data --console-address ":9001"
```

然后打开 MinIO Console：`http://localhost:9001`，登录后创建 Bucket（桶名需要与你的 `.env.docker` 里 `MARKET_BUCKET_NAME` 一致）。

桶应保持 **私有**（禁止匿名/Public 读）。插件包与图标由服务端生成 **预签名临时 URL**，客户端不拼接桶直链。

#### 2) 在 `.env.docker` 配置 MinIO（示例）

`marketplace-tools` 跑在容器内时，S3 API 需连宿主机上的 MinIO，故 `MARKET_S3_ENDPOINT` 使用 `host.docker.internal`。

```env
STORAGE_TYPE=MinIO
MARKET_BUCKET_NAME=openjiuwen-market
MARKET_S3_ENDPOINT=http://host.docker.internal:9000
MARKET_S3_ACCESS_KEY=minioadmin
MARKET_S3_SECRET_KEY=minioadmin
# 可选：预签名有效期（秒），默认 1800
# MARKET_S3_PRESIGNED_EXPIRES=1800
```

若你在 MinIO 启动命令里改过 `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`，请把上述 `MARKET_S3_ACCESS_KEY`、`MARKET_S3_SECRET_KEY` 改成相同值。可选：`MARKET_S3_REGION=us-east-1`（与代码默认一致时可省略）。

### 对象存储 / 向量存储：华为云 OBS 对接（示例）

若你希望对接[华为云 OBS](https://support.huaweicloud.com/obs/index.html)（S3 兼容接口），可按以下方式配置（桶建议保持 **私有**；访问对象仅通过接口返回的 **预签名 URL**）：

```env
STORAGE_TYPE=OBS
MARKET_S3_ENDPOINT=https://obs.<区域>.myhuaweicloud.com
MARKET_S3_ACCESS_KEY=你的_ACCESS_KEY
MARKET_S3_SECRET_KEY=你的_SECRET_KEY
# 与桶所在区域一致
MARKET_S3_REGION=<区域>
# 与 OBS 桶名一致
MARKET_BUCKET_NAME=<你的桶名>
```

## 3. 拉取镜像并启动

以下分为 **后端（marketplace-tools）** 与 **前端（Web 静态页 + Nginx 反代）**。请按需执行；若只需 API，可只启动后端；若要在浏览器访问插件市场页面，需启动前端，并正确配置 **`BACKEND_URL`** 与 **`BACKEND_PORT`**，使 Nginx 能将 `/api/` 转发到已运行的后端。

### 3.1 后端服务（marketplace-tools）

在 PowerShell 中执行：

```powershell
docker pull swr.cn-north-4.myhuaweicloud.com/openjiuwen/marketplace-tools-server-amd64

docker run --rm --name marketplace-store `
  -p 8100:8100 `
  --env-file "D:\Workspace\agent-tools\.env.docker" `
  swr.cn-north-4.myhuaweicloud.com/openjiuwen/marketplace-tools-server-amd64
```

> 如果你的仓库路径不是 `D:\Workspace\agent-tools`，请把 `--env-file` 后面的路径替换成你实际的绝对路径。\
> 如果你的主机是 arm64 架构，请将路径中的 `amd64` 替换为 `arm64`。

### 3.2 前端（插件市场 Web）

前端镜像（华为云 SWR，请以仓库实际 **Web/前端** 镜像名为准）：

`swr.cn-north-4.myhuaweicloud.com/openjiuwen/marketplace-tools-web-amd64:latest`

拉取并启动示例（将宿主机 **8100** 上的后端作为 API 上游；与上一节中 `-p 8100:8100` 的后端对应）：

```powershell
docker pull swr.cn-north-4.myhuaweicloud.com/openjiuwen/marketplace-tools-web-amd64:latest

docker run -d --rm --name marketplace-web `
  -p 9002:9002 `
  -e BACKEND_URL=host.docker.internal `
  -e BACKEND_PORT=8100 `
  swr.cn-north-4.myhuaweicloud.com/openjiuwen/marketplace-tools-web-amd64:latest
```

说明：

- **`-p 9002:9002`**：容器内 Nginx 监听 **9002**，映射到宿主机 **9002**，浏览器访问 **`http://localhost:9002`**。如需其它宿主机端口，请改映射左侧。
- **`BACKEND_URL` / `BACKEND_PORT`**：镜像内 Nginx 将 `/api/` 转发到 **`http://BACKEND_URL:BACKEND_PORT`**。后端监听在宿主机 `8100` 时，Docker Desktop 下通常使用 **`BACKEND_URL=host.docker.internal`**、**`BACKEND_PORT=8100`**。
- 若后端与前端在同一 Docker **自定义网络** 中，且后端容器名为 `marketplace-store`，可改为 **`BACKEND_URL=marketplace-store`**、**`BACKEND_PORT=8100`**。
- 请先启动 **3.1 后端**，再启动前端，否则页面无法拉到接口数据。
- **跨域（CORS）**：`marketplace-tools` 后端 **未对浏览器配置 CORS**。浏览器访问插件市场页面时，请使用 **Web 容器的入口（如 `http://localhost:9002`）**，由 Nginx **同域转发** `/api/`；不要指望从**与 `:8100` 不同源**的页面上直接请求 `http://...:8100/api/...`（会被浏览器拦截）。

## 4. 访问接口

- **命令行 / 服务端联调**：可直接访问后端，例如 `http://localhost:8100/`（无浏览器跨域限制）。
- **浏览器中的插件市场页面**：请优先访问 **前端 Nginx 入口**（已按 **3.2** 启动时一般为 `http://localhost:9002`），通过 **同源 `/api/`** 调用后端；不要仅把静态页放在与 `:8100` 不同源的地址却直连后端 API（见 **3.2** 跨域说明）。

插件列表接口示例（curl）：

```bash
curl --location 'http://localhost:8100/api/v1/plugins'
```

如果你使用 `X-System-Token` 鉴权，请确保 `.env.docker` 中的系统 token 与请求头一致。\
如果你使用 `Authorization` 鉴权，请先[注册 GitCode 账号](https://gitcode.com/)并[申请访问令牌](https://docs.gitcode.com/docs/help/home/user_center/security_management/user_pat)作为 Bearer Token。

