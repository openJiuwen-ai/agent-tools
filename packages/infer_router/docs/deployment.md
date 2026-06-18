# Jiuwen Agent Router 部署指南

## 系统要求

- Python 3.10+
- pip 20.0+
- 至少 2GB RAM
- 网络连接（用于安装依赖和与推理引擎通信）

## 安装步骤

### 1. 克隆代码仓库

```bash
git clone https://gitcode.com/openJiuwen/agent-tools.git
cd agent-tools
```

### 2. 创建虚拟环境（可选但推荐）

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
# 安装项目
pip install -e packages/openjiuwentools_infer_router

# 或使用 uv
uv pip install -e packages/openjiuwentools_infer_router
```

### 4. 配置服务

创建配置文件 `config.yaml`：

```yaml
# 服务器配置
host: 0.0.0.0
port: 8000
log_level: info

# 认证配置
enable_auth: false
api_key: your-api-key

# 调度配置
default_scheduling_strategy: FCFS  # FCFS, LCFS, WSPT, PRIORITY

# 负载均衡配置
load_balancing_algorithm: weighted  # weighted, least_connection, kv_cache_aware

# KV缓存配置
kv_cache_max_blocks: 1000
kv_cache_aging_block_factor: 0.3
kv_cache_decay_factor: 0.9
kv_cache_block_size: 16
kv_cache_enable_session_affinity: true

# 工作器发现配置
worker_discovery_type: config  # config, etcd
worker_config_path: workers.json
worker_discovery_interval: 30
worker_health_check_interval: 10
worker_health_check_timeout: 5

# etcd配置（当 worker_discovery_type 为 etcd 时使用）
etcd_hosts:
  - localhost
etcd_port: 2379
etcd_prefix: /jiuwen/workers
etcd_user: null
etcd_password: null
etcd_enable_watch: false

# 监控配置
enable_metrics: true
metrics_port: 8001

# 容错配置
retry_attempts: 3
retry_delay: 0.5

# 性能优化配置
http_pool_connections: 100
http_pool_max_keepalive: 20
http_keepalive_expiry: 5.0
request_rate_limit: 1000
request_burst_limit: 100
enable_response_cache: true
response_cache_ttl: 300
response_cache_max_size: 1000
max_concurrent_requests: 100
```

#### 工作器配置文件

当 `worker_discovery_type` 设置为 `config` 时，需要创建工作器配置文件 `workers.json`（或 `workers.yaml`）：

**JSON 格式示例 (workers.json)：**

```json
{
  "workers": [
    {
      "worker_id": "worker-1",
      "model": "qwen-7b-chat",
      "url": "http://localhost:8001/v1",
      "available_memory": 16000000000,
      "current_load": 0,
      "cached_prefixes": [],
      "engine_type": "vllm"
    },
    {
      "worker_id": "worker-2",
      "model": "qwen-7b-chat",
      "url": "http://localhost:8002/v1",
      "available_memory": 16000000000,
      "current_load": 0,
      "cached_prefixes": [],
      "engine_type": "vllm"
    }
  ]
}
```

**YAML 格式示例 (workers.yaml)：**

```yaml
workers:
  - worker_id: worker-1
    model: qwen-7b-chat
    url: http://localhost:8001/v1
    available_memory: 16000000000
    current_load: 0
    cached_prefixes: []
    engine_type: vllm
  - worker_id: worker-2
    model: qwen-7b-chat
    url: http://localhost:8002/v1
    available_memory: 16000000000
    current_load: 0
    cached_prefixes: []
    engine_type: vllm
```

**工作器配置字段说明：**

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `worker_id` | string | 是 | 工作器唯一标识符 |
| `model` | string | 是 | 模型名称 |
| `url` | string | 是 | 工作器 API 地址 |
| `available_memory` | integer | 否 | 可用内存（字节），默认 0 |
| `current_load` | integer | 否 | 当前负载（0-100），默认 0 |
| `cached_prefixes` | array | 否 | 已缓存的前缀列表，默认 [] |
| `engine_type` | string | 否 | 引擎类型（vllm/sglang），默认 vllm |
| `api_key` | string | 否 | 工作器 API 密钥（用于认证），默认 None |

#### API Key 配置说明

如果工作器需要 API Key 认证，可以在工作器配置中添加 `api_key` 字段：

```yaml
workers:
  - worker_id: worker-1
    model: qwen-7b-chat
    url: http://localhost:8001/v1
    api_key: sk-your-api-key-here
    available_memory: 16000000000
    current_load: 0
    cached_prefixes: []
    engine_type: vllm
```

路由器会在健康检查和请求转发时自动添加 `Authorization: Bearer <api_key>` 请求头。

### 5. 运行服务

```bash
# 使用默认配置
jiuwen-infer-router

# 使用指定配置文件
jiuwen-infer-router --config config.yaml

# 使用环境变量
CONFIG_PATH=config.yaml jiuwen-infer-router

# 使用命令行参数覆盖配置
jiuwen-infer-router --host 0.0.0.0 --port 8080 --log-level debug

# 使用 Python 模块方式运行
python -m openjiuwentools.infer_router.api.server --config config.yaml
```

服务将在 `http://localhost:8000` 启动。

## 配置优先级

配置加载优先级（从高到低）：

1. 命令行参数（`--host`, `--port`, `--log-level`）
2. 环境变量
3. YAML 配置文件
4. 默认值

## 环境变量

服务支持以下环境变量：

| 环境变量 | 描述 | 默认值 |
|---------|------|-------|
| `CONFIG_PATH` | 配置文件路径 | `config.yaml` |
| `HOST` | 服务器主机 | `0.0.0.0` |
| `PORT` | 服务器端口 | `8000` |
| `LOG_LEVEL` | 日志级别 | `info` |
| `ENABLE_AUTH` | 是否启用认证 | `false` |
| `API_KEY` | API 密钥 | `None` |
| `DEFAULT_SCHEDULING_STRATEGY` | 调度策略 | `FCFS` |
| `LOAD_BALANCING_ALGORITHM` | 负载均衡算法 | `weighted` |
| `KV_CACHE_MAX_BLOCKS` | KV缓存最大块数 | `1000` |
| `WORKER_DISCOVERY_TYPE` | 工作器发现类型 | `config` |
| `WORKER_CONFIG_PATH` | 工作器配置文件路径 | `workers.json` |
| `ENABLE_METRICS` | 是否启用指标收集 | `true` |
| `METRICS_PORT` | 指标端口 | `8001` |
| `RETRY_ATTEMPTS` | 重试次数 | `3` |
| `RETRY_DELAY` | 重试延迟（秒） | `0.5` |

## Docker 部署

### 1. 构建 Docker 镜像

```bash
docker build -t jiuwen-agent-router .
```

### 2. 运行 Docker 容器

```bash
docker run -d \
  --name jiuwen-agent-router \
  -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e CONFIG_PATH=/app/config.yaml \
  jiuwen-agent-router
```

## Kubernetes 部署

### 1. 创建 Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jiuwen-agent-router
spec:
  replicas: 3
  selector:
    matchLabels:
      app: jiuwen-agent-router
  template:
    metadata:
      labels:
        app: jiuwen-agent-router
    spec:
      containers:
      - name: jiuwen-agent-router
        image: jiuwen-agent-router:latest
        ports:
        - containerPort: 8000
        env:
        - name: HOST
          value: "0.0.0.0"
        - name: PORT
          value: "8000"
        - name: ENABLE_AUTH
          value: "true"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: jiuwen-agent-router-secrets
              key: api-key
```

### 2. 创建 Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: jiuwen-agent-router
spec:
  selector:
    app: jiuwen-agent-router
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 3. 创建 Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: jiuwen-agent-router-secrets
type: Opaque
data:
  api-key: <base64-encoded-api-key>
```

## 健康检查

服务提供健康检查接口：

```bash
curl http://localhost:8000/health
```

正常响应：

```json
{
  "status": "healthy",
  "components": {
    "preprocessor": "ok",
    "kvcache_manager": "ok",
    "scheduler": "ok",
    "router": "ok",
    "worker_manager": "ok",
    "performance_optimizer": "ok"
  },
  "workers": {
    "total": 2,
    "healthy": 2
  }
}
```

## 监控

如果启用了监控，可以通过以下地址访问指标：

```text
http://localhost:8000/metrics
```

支持的指标包括：

- `requests_total`：总请求数
- `requests_success`：成功请求数
- `requests_failed`：失败请求数
- `queue_time_seconds`：请求排队时间
- `processing_time_seconds`：请求处理时间

## 日志

服务使用 loguru 库记录日志，日志级别可以通过配置文件设置。默认日志格式：

```text
2026-04-07 12:00:00.123 | INFO | module:function:45 - Server started
```

## 常见问题

### 1. 服务无法启动

检查端口是否被占用：

```bash
lsof -i :8000
```

### 2. 认证失败

确保 API 密钥正确，并且认证已启用或禁用（根据配置）。

### 3. 推理引擎连接失败

检查工作器配置中的 URL 是否正确，以及推理引擎是否正在运行。

### 4. 性能问题

- 调整调度策略和负载均衡算法
- 增加工作器数量
- 调整 KV 缓存大小

## 升级指南

### 1. 备份配置文件

```bash
cp config.yaml config.yaml.backup
```

### 2. 拉取最新代码

```bash
git pull origin main
```

### 3. 安装新依赖

```bash
pip install -e packages/openjiuwentools_infer_router
```

### 4. 重启服务

```bash
# 停止当前服务
# 运行新服务
jiuwen-infer-router --config config.yaml
```
