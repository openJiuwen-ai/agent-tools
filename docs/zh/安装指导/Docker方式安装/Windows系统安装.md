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

请将对应 Bucket 的访问策略配置为 **公开（PUBLIC / public 读）**（例如在 MinIO Console 中进入该桶的 **Access Policy**，选择允许匿名读取或自定义策略放行 `GetObject`），以便浏览器通过 `MARKET_STORAGE_PUBLIC_URL` 访问插件图标等对象；若桶为私有，页面可能无法直链加载资源。

#### 2) 在 `.env.docker` 配置 MinIO（示例）

`marketplace-tools` 跑在容器内时，S3 API 需连宿主机上的 MinIO，故 `MARKET_S3_ENDPOINT` 使用 `host.docker.internal`；浏览器访问公开对象地址仍可用本机 `localhost`。

```env
STORAGE_TYPE=MinIO
MARKET_BUCKET_NAME=openjiuwen-market
MARKET_S3_ENDPOINT=http://host.docker.internal:9000
MARKET_S3_ACCESS_KEY=minioadmin
MARKET_S3_SECRET_KEY=minioadmin
MARKET_STORAGE_PUBLIC_URL=http://localhost:9000/openjiuwen-market
```

若你在 MinIO 启动命令里改过 `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`，请把上述 `MARKET_S3_ACCESS_KEY`、`MARKET_S3_SECRET_KEY` 改成相同值。可选：`MARKET_S3_REGION=us-east-1`（与代码默认一致时可省略）。

> 鉴权服务等若也部署在**宿主机**上：`AUTH_SERVICE_HOST` 等同样可设为 `host.docker.internal`（与上文 MySQL、MinIO 一致）。若该主机名不可用，请改用宿主机实际 IP 或 `--add-host`。

## 3. 拉取镜像并启动

在 PowerShell 中执行以下命令启动：

```powershell
docker pull swr.ap-southeast-1.myhuaweicloud.com/openjiuwen-online-test/marketplace-tools:0.1.1

docker run --rm --name marketplace-store `
  -p 8100:8100 `
  --env-file "D:\Workspace\agent-tools\.env.docker" `
  swr.ap-southeast-1.myhuaweicloud.com/openjiuwen-online-test/marketplace-tools:0.1.1
```

> 如果你的仓库路径不是 `D:\Workspace\agent-tools`，请把 `--env-file` 后面的路径替换成你实际的绝对路径。

## 4. 访问接口

- 健康检查：`http://localhost:8100/api/health`
- Swagger 文档：`http://localhost:8100/api/docs`

如果你使用 `X-System-Token` 鉴权，请确保 `.env.docker` 中的系统 token 与请求头一致。

