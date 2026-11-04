# Serving Security Boundaries

## Worker Exposure and Authentication

Workers now bind to `127.0.0.1` by default in both CLI and YAML defaults. For
cross-machine P/D serving, explicitly set `host` to the worker's private address
and configure firewall/network isolation. Existing explicit `host` settings are
not overwritten.

Set `JIUWEN_WORKER_API_KEY` in the worker environment (preferred), or use
`api_key` in worker YAML / `--api-key`. HTTP endpoints require
`Authorization: Bearer <key>` when configured; `/health` remains public for
liveness probes. TCP messages carry the key in a top-level `api_key` field,
which is removed before inference. Raw TCP deployments should use a private
network or an authenticated encrypted tunnel.

For the Rust proxy, set `OPENAI_API_KEY` to the same worker key; its existing
route headers forward that credential to inference workers.
For Python WorkerManager requests (including prefetch), configure each discovered
worker's `api_key` in the Router worker configuration as well. Secrets are not
automatically published into etcd discovery records.
Do not publish engine-core control ports externally: they remain an internal trusted control
plane, normally bound to localhost. Bearer authentication is not encryption;
terminate TLS at a trusted gateway when traffic crosses a trust boundary.

HTTP request bodies default to a 64 MiB limit, configurable through worker
`max_request_body_bytes` / `--max-request-body-bytes`. Both declared lengths and
actual chunked bytes are checked before JSON parsing. Raw TCP retains its
existing 64 MiB frame limit.

## Rust Data Plane Limits

The native extension must be rebuilt after updating Rust source. Python-only
package updates do not update an already compiled `_data_plane` extension.

| Environment Variable | Default | Scope |
| --- | --- | --- |
| `JIUWEN_PROXY_MAX_RESPONSE_BYTES` | 67108864 | Buffered inference, control and upstream error bodies |
| `JIUWEN_PROXY_READ_TIMEOUT_SECS` | 120 | Maximum idle wait for an upstream read |
| `JIUWEN_PROXY_REQUEST_TIMEOUT_SECS` | 600 | Overall upstream HTTP request budget |

All values must be positive integers. Request-specific control-plane deadlines
continue to override the client request timeout. Long output generation may
require increasing the overall budget. SSE inference bodies are forwarded
incrementally rather than buffered under the response-size budget.

## Disk Cache Format

New `.kv` files contain data-only JSON with explicit block/layer tags and raw
CPU tensor bytes encoded as base64. Loading never resolves a Python class or
executes a callable named by a file. Unknown types/tags are rejected. Files are
written through an atomic replace; disk capacity counts serialized bytes.

This is a cache format change: old pickle files are not loaded. The existing
startup cleanup removes managed `.pkl` and `.kv` payloads; operators disabling
startup cleanup must discard old cache data before reuse. `store_ref` must be a
single managed filename; absolute paths, traversal and symbolic links are
rejected. The cache directory must remain private to the worker account.

## Cleanup Failures

`release_failed` leases and `cleanup_failed` reservations retain ownership and
error records. They are not reported as successfully freed and are not blindly
retried after a possibly partial `free_blocks()` operation. Unrelated expired
leases/reservations continue to be processed. Index rollback attempts every
block; blocks with an unresolved index remain pinned instead of being recycled.
An affected pool requires operator recovery or worker restart; these changes do
not claim automatic repair of an arbitrarily corrupted engine BlockPool.

Expired PD rollbacks without acquired resources become failed and release their
logical slot. Rollbacks with unresolved physical resources remain capacity
accounted and appear in `cleanup_required`; changing them to terminal merely to
increase capacity would hide ownership and risk unsafe reuse.
