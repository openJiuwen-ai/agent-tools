# jiuwen-kvbm

Independent KV Block Manager core package for Jiuwen.

当前 KVBM、vLLM-Ascend、LLM-DataDist、HIXL 与 Router 的完整设计见
[KVBM 与 HIXL 设计方案](../infer_router/docs/kvbm-hixl-design.md)。

缓存格式迁移、访问控制和清理失败恢复见
[安全边界说明](../infer_router/docs/security_boundaries.md)。

This package intentionally does not depend on existing vLLM affinity code. It
provides a neutral management layer for KV block metadata, tier state, local
RAM/disk storage, and future NPU/HIXL transfer backends.

Initial scope:

- metadata and lifecycle state machine
- fake runtime accessor for tests
- local RAM and disk backends
- HIXL backend boundary for future native implementation
- LLM-DataDist backend lifecycle boundary for engine-core worker integration

The core package avoids CUDA/NIXL-specific terms. Runtime-specific integration
should be added through accessors. Local RAM/disk tiering is handled by the
Python management plane. High-performance remote KV transfer should be added
through the engine-core embedded LLM-DataDist backend.

## LLM-DataDist backend

The embedded LLM-DataDist integration:

- defines neutral cache/block semantic types under `jiuwen_kvbm.transfer.llm_datadist`
- provides `FakeLLMDataDistBackend` for tests and dry-run wiring
- provides `PythonLLMDataDistBackend` that imports CANN `llm_datadist` and initializes
  the object-style API used by CANN examples: `LLMDataDist(role, cluster_id).init(options)`
- registers the vLLM-Ascend 0.18 KV tensors with `register_blocks_cache`
- exposes cluster link, unlink, status, and physical block pull operations through the
  existing engine-core control service

Enable it in the engine-core worker environment:

```bash
export JIUWEN_KVBM_DATA_BACKEND=llm_datadist
export JIUWEN_KVBM_LLM_DATADIST_MODE=python
export JIUWEN_KVBM_LLM_DATADIST_ROLE=mix
export JIUWEN_KVBM_LLM_DATADIST_CLUSTER_ID=0
export JIUWEN_KVBM_LLM_DATADIST_LISTEN_IP=127.0.0.1
export JIUWEN_KVBM_LLM_DATADIST_LISTEN_PORT_BASE=19000
export JIUWEN_KVBM_LLM_DATADIST_TRANSFER_BACKEND=hixl
```

For local tests without CANN:

```bash
export JIUWEN_KVBM_DATA_BACKEND=llm_datadist
export JIUWEN_KVBM_LLM_DATADIST_MODE=fake
```

For a worker started by `infer_router.vllm`, use top-level worker YAML keys such as
`kvbm_data_backend: llm_datadist` and `kvbm_llm_datadist_cluster_id: 0`. The nested
`kvbm:` form is not accepted by the current worker config loader.

After two workers have initialized and registered their caches, send the following
requests to the destination worker's engine-core control port. Linking must happen
before pulling blocks, and all caches must be registered before linking.

```bash
curl -X POST http://127.0.0.1:18001/datadist/link \
  -H 'Content-Type: application/json' \
  -d '{"remote_cluster_id":0,"remote_ip":"10.0.0.1","remote_port":26000,"local_ip":"10.0.0.2","local_port":26001,"timeout_ms":5000}'

curl -X POST http://127.0.0.1:18001/datadist/pull \
  -H 'Content-Type: application/json' \
  -d '{"remote_cluster_id":0,"source_blocks":[1,2],"destination_blocks":[11,12]}'
```

Here `18001` is the destination worker's KVBM prefetch/control port. The
`source_blocks` are physical block IDs in the remote worker cache, while
`destination_blocks` are reserved physical block IDs in the destination worker cache.

For debug-only content validation, calculate SHA-256 digests before and after the pull:

```bash
curl -X POST http://127.0.0.1:18011/datadist/digest \
  -H 'Content-Type: application/json' \
  -d '{"block_ids":[1,2]}'

curl -X POST http://127.0.0.1:18020/datadist/digest \
  -H 'Content-Type: application/json' \
  -d '{"block_ids":[11,12]}'
```

Source block 1 must match destination block 11, and source block 2 must match
destination block 12. Set `include_layers` to `true` to return per-layer key/value
digests when the whole-block digest differs. Digest calculation copies complete KV
blocks from NPU to CPU and must not be enabled on the normal inference hot path.

To make transferred blocks directly matchable by vLLM, call the destination worker's
transactional pull endpoint. `source_control_url` must be the source worker's KVBM
control address as seen from the destination worker:

```bash
curl -X POST http://127.0.0.1:18020/datadist/transfers/pull \
  -H 'Content-Type: application/json' \
  -d '{
    "transfer_id": "request-123-prefetch-0",
    "remote_cluster_id": 0,
    "source_control_url": "http://10.0.0.1:18011",
    "source_blocks": [1, 2],
    "source_lease_ttl_ms": 30000,
    "reservation_ttl_ms": 120000
  }'
```

For workers in different containers or hosts, configure the source worker with
`kvbm_prefetch_host: 0.0.0.0` (or another reachable bind address), expose its
`kvbm_prefetch_port`, and use that reachable address in `source_control_url`. Keep this
control port on the trusted cluster network.

This endpoint first acquires a source lease that atomically reads the exact packed hashes
and pins the source blocks through vLLM `BlockPool.touch()`. It renews the lease while
LLM-DataDist is pulling, publishes the hashes into the destination prefix cache, and
releases the lease in all completion paths. A failed pull, lost lease, or failed publish
automatically rolls back the destination reservation. The source scheduler also expires
abandoned leases, so a crashed destination cannot hold source blocks permanently.

Production callers should provide a stable `transfer_id`. Repeating the same request with
that ID returns the existing in-flight snapshot or terminal result instead of pulling the
blocks again. A timed-out caller can query or cancel the transaction:

```bash
curl -X POST http://127.0.0.1:18020/datadist/transfers/status \
  -H 'Content-Type: application/json' \
  -d '{"transfer_id":"request-123-prefetch-0"}'

curl -X POST http://127.0.0.1:18020/datadist/transfers/cancel \
  -H 'Content-Type: application/json' \
  -d '{"transfer_id":"request-123-prefetch-0"}'
```

Cancellation is cooperative. If HIXL is currently blocking, KVBM waits for that call to
return and rolls back before prefix-cache publish. Once publish begins, cancellation is
rejected because the destination block may already be visible to vLLM.

### Prefill/Decode handoff protocol

The PD-separation protocol provides a request-level handoff state machine, real P-side
block export/pin, and atomic D-side request block-table attachment. Router deployments
can enable `kvbm.router_pd_enabled` to let the Rust data plane orchestrate this protocol;
the following requests remain useful for direct Worker diagnostics. Create a handoff on
the KVBM worker control port:

```bash
curl -sS -X POST http://127.0.0.1:18020/datadist/pd/handoffs/begin \
  -H 'Content-Type: application/json' \
  -d '{
    "handoff_id":"pd-request-1",
    "request_id":"request-1",
    "source_worker_id":"prefill-0",
    "target_worker_id":"decode-0",
    "model_id":"model-a",
    "prompt_token_count":1024
  }'

curl -sS -X POST http://127.0.0.1:18020/datadist/pd/handoffs/status \
  -H 'Content-Type: application/json' \
  -d '{"handoff_id":"pd-request-1"}'
```

After vLLM has completed prefill, export by request ID. The caller does not provide
physical block IDs: the scheduler resolves every full prompt block from its BlockPool
prefix-cache index and calls `BlockPool.touch()` while holding the shared lock. The
returned source lease prevents eviction or physical-ID reuse during HIXL transfer:

```bash
curl -sS -X POST http://127.0.0.1:18020/datadist/pd/export \
  -H 'Content-Type: application/json' \
  -d '{
    "handoff_id":"pd-request-1",
    "request_id":"request-1",
    "ttl_ms":30000
  }'
```

The result moves the handoff through `prefill_ready` to `source_pinned` and returns
`source_blocks`, packed `block_hashes`, per-group mappings, `source_lease_id`, and the
local LLM-DataDist cache metadata. Retrying the same export is idempotent. Release the P
reference after decode accepts the handoff, or release it early to roll the handoff back:

```bash
curl -sS -X POST http://127.0.0.1:18020/datadist/pd/export/release \
  -H 'Content-Type: application/json' \
  -d '{"handoff_id":"pd-request-1","reason":"decode did not accept"}'
```

On the D Worker, prepare the request gate **before** dispatching the decode request. The
gate keeps this request in the scheduler waiting queue but excludes it from local prefill:

```bash
curl -sS -X POST http://127.0.0.1:18021/datadist/pd/import/prepare \
  -H 'Content-Type: application/json' \
  -d '{"handoff_id":"pd-request-1","request_id":"request-1"}'
```

Submit the decode request with the same `request_id`, then wait until
`/datadist/pd/import/status` reports `request_seen=true`. Start the atomic D import with
the `export` object returned by P `/datadist/pd/export`:

```bash
curl -sS -X POST http://127.0.0.1:18021/datadist/pd/import \
  -H 'Content-Type: application/json' \
  -d '{
    "handoff_id":"pd-request-1",
    "request_id":"request-1",
    "source_control_url":"http://10.0.0.1:18020",
    "remote_cluster_id":0,
    "source_export":{
      "request_id":"request-1",
      "source_blocks":[7,8],
      "source_lease_id":"<P export lease id>",
      "block_hashes":["<P export descriptors>"],
      "block_groups":["<P export group mappings>"]
    }
  }'
```

The recommended D-side ownership mode is configured on every PD Worker:

```yaml
kvbm_pd_block_ownership: scheduler
```

In `scheduler` mode, vLLM performs local prefix lookup and calls its own
`KVCacheManager.allocate_slots()` with external-computed-token semantics. The returned
block table belongs to the D request before HIXL starts. KVBM only transfers the missing
positions, then commits them with `cache_blocks()` and advances `num_computed_tokens`.
Cancellation or transfer failure calls `KVCacheManager.free(request)`, so allocation,
waiting and request ownership remain governed by the vLLM Scheduler. This mode does not
use the independent destination reservation manager or its renewal thread.

When Scheduler capacity is temporarily unavailable, the accepted import remains attached
to the request gate and is retried from later Scheduler cycles. It is submitted to the
HIXL import executor only after request-owned destination blocks are available, so
capacity waiters do not occupy the fixed import thread pool or delay ready transactions.

`scheduler` is the default when this option is omitted. Setting
`kvbm_pd_block_ownership: legacy` explicitly keeps the previous KVBM
reservation-and-attach path for rollback compatibility. P and D Workers in one
deployment should use the same value. `/datadist/status` reports the active mode under
`scheduler_kv_bridge`, and `/datadist/pd/import/status` reports per-request native
allocation state and current free blocks.

Both modes renew the P lease, pull KV through LLM-DataDist/HIXL, and submit the final
commit to the scheduler thread. A successful decode ACK releases the P source lease.

Repeating `begin` or the same `advance` is idempotent. Invalid stage skips and resource
replacement are rejected. A caller can request cooperative rollback as follows. If a P
source lease was acquired, call `/datadist/pd/export/release` to unpin it and advance the
handoff to `rolled_back`:

```bash
curl -sS -X POST http://127.0.0.1:18020/datadist/pd/handoffs/rollback \
  -H 'Content-Type: application/json' \
  -d '{"handoff_id":"pd-request-1","reason":"decode worker unavailable"}'
```

`JIUWEN_KVBM_PD_HANDOFF_MAX_INFLIGHT` defaults to `1024`, and
`JIUWEN_KVBM_PD_HANDOFF_RETENTION_SECONDS` defaults to `300`.

`kvbm_transfer_max_inflight` defaults to `1`, matching the current serialized
LLM-DataDist/HIXL call path. `kvbm_transfer_retention_seconds` defaults to `300`, which
controls how long terminal results remain queryable. Destination reservations default to
a 120-second TTL and are returned to the vLLM free queue if the caller disappears.

In vLLM's multiprocess architecture, KV tensors and HIXL live in the NPU worker process,
while `KVCacheManager` and `BlockPool` live in the engine-core scheduler process. Jiuwen
therefore starts an internal reservation service in the scheduler and proxies the public
worker control endpoint to it. By default its port is the KVBM control port plus 10000
(for example, public `18020`, internal `28020`). Override it with the top-level worker
YAML key `kvbm_reservation_port` when that derived port conflicts with another service.

## HIXL 与 TCP staging 隔离对比

基准入口默认关闭，只应在没有推理请求的专用 worker 上使用。它会直接覆盖目标
物理 block，但不会执行 BlockPool reserve/publish，因此不要把正在被 vLLM 使用的
block ID 作为 `destination_blocks`。

源、目标 worker 都开启基准；每个 worker 使用不同的 TCP staging 端口。监听地址
必须是对端可达的容器或节点 IP：

```yaml
kvbm_benchmark_enabled: true
kvbm_benchmark_tcp_host: 10.0.0.1
kvbm_benchmark_tcp_port: 29011
kvbm_benchmark_max_blocks: 8
```

目标 worker 可使用自己的端口，例如 `29020`。重启后通过
`/datadist/status` 的 `benchmark.tcp_server_ready` 确认服务已启动。HIXL 对比前
仍需先完成 `/datadist/link`。

安装 editable 包后，在目标 worker 容器内执行：

```bash
jiuwen-kvbm-benchmark \
  --target-url http://127.0.0.1:18020 \
  --backend both \
  --source-blocks 1,2,3,4 \
  --destination-blocks 101,102,103,104 \
  --remote-cluster-id 0 \
  --source-host 10.0.0.1 \
  --source-port 29011 \
  --warmup 5 \
  --iterations 50
```

`hixl` 结果表示 LLM-DataDist `pull_blocks` 的 NPU 到 NPU 同步路径；
`tcp_staging` 结果表示源 NPU 到 CPU RAM、TCP、目标 CPU RAM 再到目标 NPU 的
完整路径。两者统一以完整 KV block 字节数计算 `effective_gbps`，并输出
`latency_ms` 的 p50/p95/p99。TCP 结果还包含源端 D2H、序列化、目标解码和 H2D
阶段耗时；这些阶段可能重叠或被 socket 等待包含，不应相加后当作总耗时。

### 不依赖 vLLM Worker 的独立对比

`jiuwen-kvbm-standalone-benchmark` 会自行在两张 NPU 上分配与 vLLM-Ascend
相同布局的 KV tensor，并直接初始化 LLM-DataDist。它不启动 Router、vLLM、
BlockPool 或 Worker HTTP 服务，适合隔离测量 HIXL 与 TCP staging 的数据面差异。

先在源 NPU 进程启动测试 cache 和 TCP staging 服务：

```bash
jiuwen-kvbm-standalone-benchmark source \
  --device-id 0 \
  --cluster-id 0 \
  --listen-ip 10.41.38.238 \
  --listen-port 19000 \
  --tcp-host 0.0.0.0 \
  --tcp-port 29011 \
  --num-blocks 64 \
  --block-size 128 \
  --num-layers 36 \
  --num-kv-heads 8 \
  --head-dim 128
```

保持源进程运行，在目标 NPU 的另一个进程执行：

```bash
jiuwen-kvbm-standalone-benchmark target \
  --device-id 1 \
  --cluster-id 1 \
  --listen-ip 10.41.38.238 \
  --listen-port 19001 \
  --source-cluster-id 0 \
  --source-ip 10.41.38.238 \
  --source-hixl-port 19000 \
  --source-tcp-host 10.41.38.238 \
  --source-tcp-port 29011 \
  --blocks 4 \
  --warmup 5 \
  --iterations 50
```

同机双卡时，两端 `listen-ip` 可以使用同一个 HCCN 可达 IP，但 cluster ID、
HIXL 监听端口和 NPU device ID 必须不同。跨节点时分别使用各节点可达的 HCCN
地址。两端 tensor 布局参数必须完全一致。

目标端会输出两种路径的 p50/p95/p99、有效带宽和 p50 加速比，并在计时结束后
比较 HIXL 与 TCP 写入结果的 SHA-256。`validation.digests_match=true` 才表示
性能结果通过了基本正确性校验。默认布局每张卡约占用 1.2 GiB NPU 内存，可通过
`--num-blocks` 或其他 tensor 参数缩小。

## 自动识别 KV block size

vLLM-Ascend engine-core 在真实 `kv_caches` 初始化后，会从首层 key cache 的
`(num_blocks, block_size, num_kv_heads, head_dim)` shape 自动识别
`block_size`。KVBM store/location relay 会把识别结果连同 `token_ids` 和
`parent_block_hash` 上报，Router 按事件中的尺寸构建或修正 RadixTree block
边界。

`kvbm_block_size` 和 Router 的 `kv_cache_block_size` 现在只用于真实 tensor
或旧事件尚未提供尺寸时的兼容兜底，正常 vLLM-Ascend 0.18 路径不需要手工将
两者配置为相同值。若配置与 tensor shape 不一致，Worker 日志会打印修正前后的
值，并以 tensor shape 为准。
