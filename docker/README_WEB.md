# Docker 镜像（插件市场前端）

与 `agent-studio` 一致：Node 多阶段构建静态资源，**Nginx 官方镜像**通过 `/etc/nginx/templates/*.template` 在启动时做 **envsubst**，用环境变量配置 **`proxy_pass` 上游**（`BACKEND_URL` + `BACKEND_PORT`）。

## 构建

在**仓库根目录**执行（上下文为当前目录）：

```bash
docker build -f docker/Dockerfile.web-tools -t agent-tools-frontend:latest .
```

可选：构建时指定前端访问后端的 API 基础路径（写入 Vite 产物；**推荐保持默认** `/api/v1`，由同源 Nginx 反代，避免浏览器跨域）：

```bash
docker build -f docker/Dockerfile.web-tools \
  --build-arg VITE_API_BASE_URL=/api/v1 \
  -t agent-tools-frontend:latest .
```

## 运行

模板文件：`frontend/nginx/default.conf.template` → 容器启动时生成 `conf.d/default.conf`。

| 环境变量 | 说明 | 镜像内默认值 |
|----------|------|----------------|
| `BACKEND_URL` | 上游主机名或 IP（**不要**带 `http://`） | `localhost` |
| `BACKEND_PORT` | 上游端口 | `8100` |

示例：

```bash
docker run --rm -p 9002:9002 \
  -e BACKEND_URL=host.docker.internal \
  -e BACKEND_PORT=8100 \
  agent-tools-frontend:latest
```

与后端在同一 Docker 网络、后端 Service/容器名为 `marketplace-store` 时：

```bash
docker run --rm -p 9002:9002 \
  -e BACKEND_URL=marketplace-store \
  -e BACKEND_PORT=8100 \
  agent-tools-frontend:latest
```

浏览器访问：`http://localhost:9002`

**说明**：对外只需配置 **`BACKEND_URL`**（主机名或 IP，无 `http://`）与 **`BACKEND_PORT`**。仓库根目录 `.env.example` 中的同名变量与本地 Vite 开发代理一致。

## 健康检查

容器内 Nginx 提供 **`GET /health`**（不经过后端）。镜像 `HEALTHCHECK` 使用此路径。

后端 Store 自身的健康检查仍为 **`GET /api/health`**（经反代访问时为 `http://<前端>/api/health`）。
