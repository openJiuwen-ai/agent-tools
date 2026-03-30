# Docker 镜像（插件市场前端）

参考 `agent-studio/docker` 中的多阶段构建思路：Node 构建静态资源，Nginx Alpine 提供页面与 `/api` 反代。

## 构建

在**仓库根目录**执行（上下文为当前目录）：

```bash
docker build -f docker/Dockerfile.web-tools -t agent-tools-frontend:latest .
```

可选：构建时指定前端访问后端的 API 基础路径（写入 Vite 产物，默认未设则为 `/api/v1`）：

```bash
docker build -f docker/Dockerfile.web-tools \
  --build-arg VITE_API_BASE_URL=/api/v1 \
  -t agent-tools-frontend:latest .
```

## 运行

`frontend/nginx.conf` 中占位为 `jiuwen-backend:8100`，容器启动脚本会替换为环境变量 **`BACKEND_UPSTREAM`** 的值。

- **未设置** `BACKEND_UPSTREAM` 时，默认为 **`localhost:8100`**（本机可解析，便于无独立后端服务名时先启动 Nginx；纯前端容器内 `localhost` 指向容器自身，访问宿主机上的 API 请自行设为 `host.docker.internal:端口` 等）。
- **有后端**（Docker 网络服务名、K8s Service 等）时显式设置，例如：

```bash
docker run --rm -p 8080:80 \
  -e BACKEND_UPSTREAM=marketplace:8100 \
  agent-tools-frontend:latest
```

浏览器访问：`http://localhost:8080`

健康检查路径：`GET /api/health`（由 `frontend/nginx.conf` 提供）。
