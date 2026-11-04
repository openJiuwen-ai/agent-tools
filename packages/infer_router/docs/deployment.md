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

Worker 默认仅监听 `127.0.0.1`。跨节点部署、Bearer 认证、请求限制和 Rust 扩展
重新编译要求见[安全边界说明](./security_boundaries.md)。下方 Router 服务配置与
Worker 配置分开管理。

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

# prefix_aware 插件评分配置；location scorer 不配置时不会改变原有调度行为
load_balancing_algorithm: prefix_aware
# 可选：PD 分离时分别指定算法；未配置时两阶段沿用上面的全局算法。
prefill_load_balancing_algorithm: prefix_aware
decode_load_balancing_algorithm: round_robin
scoring_plugins:
  - type: prefix-cache-scorer
    weight: 2.0
  - type: cache-location-scorer
    weight: 3.0
    parameters:
      npu_weight: 1.0
      ram_weight: 0.6
      disk_weight: 0.2
  - type: queue-scorer
    weight: 2.0

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

# 请求兼容配置：null 表示不改写客户端的生成上限（推荐用于 Agent 客户端）
default_max_tokens: null
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

#### Router 自动跨 Worker KV 传输

启用后，Router 会在向目标 Worker 转发推理请求前，从同模型的其他 Worker
选择目标端缺失、源端位于 NPU 的连续前缀 block，并调用
LLM-DataDist/HIXL 事务接口完成拉取和 prefix-cache 发布：

```yaml
kv_cache:
  enable_radix_tree: true
kv_event_mode: worker_event
kvbm:
  router_transfer_enabled: true
  router_transfer_max_blocks: 8
  router_transfer_timeout_ms: 3000
  router_link_timeout_ms: 3000
  router_datadist_status_ttl_seconds: 10
  router_source_lease_ttl_ms: 30000
  router_reservation_ttl_ms: 120000
```

每个 Worker 必须启用 `llm_datadist` 数据后端、配置全局唯一的 cluster ID，
并通过 KV event publisher 上报事件。若 `worker.url` 不能被其他 Worker
直接访问，在发现配置中增加 `kvbm_control_url`：

```json
{
  "worker_id": "worker-1",
  "model": "qwen-7b-chat",
  "url": "http://router-visible-worker-1:8011/v1",
  "kvbm_control_url": "http://worker-visible-worker-1:8011",
  "publisher_endpoint": "tcp://worker-visible-worker-1:5557"
}
```

`kvbm_control_url` 是目标 Worker 获取源 lease 时访问的地址；未配置时由
`worker.url` 去除末尾 `/v1` 得到。建链或传输失败不会阻断主请求，Router
会继续转发并由 vLLM 重算；传输超时会请求取消，目标端在 publish 前回滚预留块。

#### Router 自动 P/D KVBM handoff

P/D 分离部署可让 Rust data-plane 自动执行 request-level handoff：Router 先建立
D gate 并立即派发 D 请求，再异步发送 P 请求。prompt 尾部不足一个 block 时，
P scheduler 会在释放请求 block table 前同时 pin partial block，并通过 engine-core
capture-ready 事件唤醒正在等待的 export；Router 不需要等待 P HTTP 响应完成才开始
HIXL pull。D import 最后在 scheduler 线程原子 attach。开启方式：

```yaml
kvbm:
  enabled: true
  router_pd_enabled: true
  router_pd_timeout_ms: 10000
  # 默认false：严格PD，Prompt尾部仍由P计算并通过HIXL交给D。
  # 只有接受D补算少量Prompt尾部时才显式改为true。
  router_pd_decode_direct_enabled: false
  router_pd_decode_direct_min_npu_tokens: 1024
  router_pd_decode_direct_min_ratio: 0.8
router_pd_prefill_admission_mode: auto
router_pd_prefill_max_inflight: 0
router_pd_prefill_block_admission_enabled: true
router_pd_prefill_block_headroom_ratio: 0.10
router_pd_decode_admission_mode: auto
router_pd_decode_max_inflight: 0
router_pd_decode_block_admission_enabled: true
router_pd_decode_block_headroom_ratio: 0.10
router_pd_decode_initial_window_blocks: 2
# 可选：开启HIXL D热点分流；关闭时沿用原有路由逻辑。
router_pd_decode_hotspot_spill_enabled: false
router_pd_decode_hotspot_spill_tpot_ratio: 1.25
# Running条件二选一：只配差值时仅判断差值。
# router_pd_decode_hotspot_spill_running_gap: 8
# 配任一具体Running门槛时，只判断已配置的门槛，忽略差值。
# router_pd_decode_hotspot_spill_hot_min_running: 16
# router_pd_decode_hotspot_spill_cold_max_running: 8
# 三项都不配时沿用旧默认：热点>=16、冷点<=8。
```

PD admission 在派发 P 请求和创建 D gate 之前同时检查 P/D 容量。默认的 `auto`
模式不使用固定并发数：P 按前缀命中后仍需计算的 prompt token/block 预算准入，
并在 export-ready 或原生 Connector 的 P 响应完成时立即释放；D 按缺失 prompt block
和完整预计 Decode 输出准入。attach 后移除 Prompt 和已开始分配的初始窗口，只保留
尚未落地的 Decode 增长预算；该预算与 Worker 实际占用相加，防止大量运行请求后续
增长突破安全水位。`router_pd_decode_initial_window_blocks` 表示 attach 后视为已开始落地
的窗口，短输出仍只按实际预计 block 数准入；例如 block size 与 output length 都是
128 时只计算 1 个 Decode block，attach 后剩余增长预算为 0。`max_inflight: 0`
表示不设置硬并发上限；配置正整数时仍可作为异常保护。所有 D 都不足时，请求保留
在 Router 队列并在容量释放后重新路由，不会先执行 P prefill。默认保留 10% KV
block，避免指标更新延迟和突发分配耗尽 BlockPool。

P Worker 会自动上报 vLLM `max_num_batched_tokens`。Router 将尚未完成请求的缺失
prompt token 累加为计算预算，因此完整 6300-token no-hit 请求与只需补算 412 token
的前缀命中请求具有不同准入成本。Worker 新启动、指标尚未到达时，Router 临时使用
旧配置 `kvbm_router_pd_prefill_max_inflight`（默认 8）作为启动保护；收到指标后自动
切换为 token 预算，不需要人工按 Worker 并发能力配置固定值。

当 D Worker 从第一个 block 开始连续位于 NPU 的前缀同时达到最少 token 数和
prompt 比例阈值时，Router 会释放本次为 P 预占的并发额度，并让 Rust data-plane
跳过 P 请求、handoff、export 和 HIXL，直接把原请求发送给 D。默认的
`1024 + 0.8` 表示至少命中 1024 个 NPU token，且命中量不少于 prompt 的 80%。
该优化只使用 Router 已维护的 location 索引；中间出现 RAM/DISK block 时，后续
NPU block 不计入连续命中。

P、D Worker 的发现信息都必须提供可由 Router 访问的 `kvbm_control_url`。Worker
仍需分别配置全局唯一的 `kvbm_llm_datadist_cluster_id`，并确保
`kvbm_llm_datadist_listen_ip` 是对端 NPU 网络可达地址。handoff 任一阶段失败时，
Router 会释放 P pin 和 D gate，使 D 请求回退到本地 prefill。

partial block 只绑定当前 `handoff_id/request_id`：D 会把其中的有效 KV 挂接到
当前请求，但不会为该物理 block 写入通用 prefix-cache hash，也不会通过 Router
location index 对其他请求发布。以 block size 128、prompt 6300 token 为例，
49 个完整 block 和最后 28-token partial block 都会通过 HIXL 传输；受 vLLM
最后一个 token logits 语义影响，D 诊断通常为 `computed_prompt_tokens=1`，而不是
旧路径的 28。该能力自动启用，不需要新增 Worker 配置。

D import 在 HIXL pull 前会由 scheduler 生成原子 delta plan：完整 packed hash
已经存在于 D BlockPool 时，计划会立即 pin 该物理 block；只有缺失完整 block
和 request-scoped partial block 才从 free queue 预留目标并加入 HIXL mapping。
查询、pin 和预留使用同一个 BlockPool 锁，因此本地命中不会在传输期间被淘汰
或复用。attach 成功后释放计划引用并转交给请求；失败、超时或 rollback 会同时
归还目标 reservation 和本地命中 pin。该优化自动启用，无新增配置。

成功链路的 Router 日志顺序为：

```text
KVBM PD handoff initialized
KVBM PD source exported and early D gate prepared
KVBM PD import attached to D request
```

存在 partial block 时，P Worker 还会打印：

```text
Jiuwen KVBM captured request-scoped P partial block before free
Jiuwen KVBM D delta import planned
Jiuwen KVBM async PD reconciliation completed
```

delta plan 和 import 日志包含 `total_blocks`、`reused_blocks`、
`transfer_blocks`/`transferred_blocks` 及 `transfer_positions`。例如 D 已有前 49 个
完整 blocks、仅缺少最后一个 partial block 时，应看到
`total_blocks=50 reused_blocks=49 transfer_blocks=1`，HIXL 只传输该位置。

D 直达时会打印：

```text
[KVBMPDDecodeDirectDecision] ... enabled=True reason=target_npu_prefix_sufficient
[KVBMPDTiming] decode-direct; skipped P request and PD handoff
```

普通 PD 请求的 `[KVBMPDTiming]` 会拆分 `route_ms`、`pd_begin_ms`、
`p_request_ms`、`pd_export_ms`、`pd_link_ms`、`pd_prepare_ms`、
`d_gate_wait_ms`、`pd_import_ms` 和 `pre_decode_total_ms`。其中
`p_request_ms` 表示 P 请求发出到 engine-core KV capture-ready，不包含 P HTTP
响应尾部；`pd_export_ms` 只保留 capture 等待之外的 export 控制开销。
`pre_decode_total_ms` 是 Rust data-plane 收到请求到具备 import 条件的累计时间；
D 请求已在 P 请求之前派发，因此正常情况下 `d_gate_wait_ms` 只剩一次状态确认。
`pd_import_ms` 的同步关键路径只保留 gate renew、delta plan、HIXL pull、request
attach 和后台 reconciliation 入队。Worker `phase_ms` 中原来的 `reserve` 对应项
改为 `delta_plan`。P 镜像状态推进及 source release 异步执行，
失败时由 source lease TTL 兜底释放。

启用专用 KVBM PD handoff 后，Router 不再在返回 Rust 路由决策前对 P、D 分别
执行通用跨 Worker prefetch；P export 到 D delta import 已覆盖对应传输语义。
P/D handoff begin 与 D gate prepare 也会并行执行，因此日志中的 `pd_begin_ms`
和 `pd_prepare_ms` 是同一时间窗口内的独立耗时，不能相加。Router 会打印
`KVBM generic prefetch skipped for dedicated PD handoff` 作为生效诊断。

data-plane 的首次 KV 匹配会同时生成每个候选 Worker 的请求级 prefix snapshot。
`PrefixAware` worker-pair、`cache-location-scorer`、PD decode-direct 和后续 prefetch
会复用同一快照，不再重复遍历 6000 token 等长公共前缀。Router 还会随 store、
remove、evict、clear 事件增量维护活跃 Radix 节点引用计数，KV 匹配不再为每个
请求扫描完整 `block_node_map`。生效后同一批公共前缀请求的
`RouteDecisionTiming.routing` 不应再随已缓存 Worker 数量线性增长。

Router 完成日志还会记录 `router_to_early_d_dispatch_ms`、
`d_request_seen_to_attach_start_ms` 和 `d_attach_complete_to_first_byte_ms`。
前者表示 Router 收到请求到提前派发 D 请求；中间项表示 D scheduler 观察请求到
开始原子 attach；最后一项表示 Router 收到 attach 成功响应到 D 首字节。
D Worker 的 `Jiuwen KVBM D attach fast-path diagnostics` 会同时打印
`attach_cycle_id`、`first_scheduled_cycle_id`、`scheduled_same_cycle`、
`attach_task_queue_ms` 和 `attach_execution_ms`。attach 成功后请求会被提升到
waiting 队首，并在当前 scheduler 周期参与 token budget；若受容量约束未能入选，
后续首次入选时会打印 `D attach fast-path scheduled later`。

Rust data-plane 的 tracing target 是 `_data_plane`。需要显式控制日志级别时，
应在启动 Router 前配置：

```bash
export RUST_LOG=_data_plane=info
```

环境变量只在 Rust 代理启动时读取，修改后需要重启 Router。

`router_pd_timeout_ms` 是 D scheduler gate/attach 的内部阶段预算。Router 调用
`/datadist/pd/import` 时会自动使用更长的 HTTP 包络超时，以覆盖 HIXL pull、
reservation 和 attach，避免外层连接先于内部事务超时。

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
