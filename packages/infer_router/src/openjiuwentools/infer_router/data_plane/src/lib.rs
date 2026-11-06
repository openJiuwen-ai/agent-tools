//! Rust 数据面代理 — 高性能 HTTP 反向代理层。
//!
//! 架构：Axum HTTP 服务接收客户端请求，通过 PyO3 + GIL 调用 Python 控制面
//! 获取路由决策（目标 worker URL、请求体、header 等），然后通过 reqwest
//! 连接池将请求转发到目标 worker，支持流式 SSE 和非流式两种响应模式。
//!
//! 请求生命周期（组合模式）：
//!   客户端 → Axum → spawn_blocking(Python route_fn) → reqwest → worker → 客户端
//!
//! 请求生命周期（PD 分离模式，disagg=true）：
//!   客户端 → Axum → route_fn → reqwest → prefill worker（等待完成）
//!   → reqwest → decode worker → 流式/缓冲转发 → 客户端
//!
//!   请求完成 → spawn_blocking(Python complete_fn) → 释放 inflight 计数

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, Response, StatusCode};
use axum::routing::{any, get};
use axum::Router;
use bytes::Bytes;
use futures::StreamExt;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};

// ---------------------------------------------------------------------------
// Python 路由回调返回的路由决策
// ---------------------------------------------------------------------------

/// Python route_fn 返回的路由决策，包含转发所需的全部信息。
///
/// PD 分离模式下 `disagg = true`，Rust 代理先发 prefill 请求（target_url / body_json），
/// 等待完成后再发 decode 请求（decode_target_url / decode_body_json）。
struct RouteDecision {
    target_url: String,
    worker_id: String,
    headers: HashMap<String, String>,
    body_json: String,
    request_id: String,
    stream: bool,
    disagg: bool,
    decode_target_url: String,
    decode_worker_id: String,
    decode_body_json: String,
    pd_kvbm_enabled: bool,
    pd_decode_direct: bool,
    pd_handoff_id: String,
    pd_decode_admission_ticket_id: String,
    pd_request_id: String,
    pd_prefill_control_url: String,
    pd_decode_control_url: String,
    pd_prompt_token_count: i64,
    pd_timeout_ms: u64,
    pd_export_timeout_ms: u64,
}

/// 从 Python dict 中提取路由决策字段。
fn extract_route_decision(obj: &Bound<'_, PyAny>) -> PyResult<RouteDecision> {
    let dict = obj.downcast::<PyDict>()?;

    let get_str = |key: &str| -> PyResult<String> {
        dict.get_item(key)?
            .map(|v| v.extract::<String>())
            .unwrap_or(Ok(String::new()))
    };

    let headers: HashMap<String, String> = dict
        .get_item("headers")?
        .map(|v| v.extract())
        .unwrap_or(Ok(HashMap::new()))?;

    let stream = dict
        .get_item("stream")?
        .map(|v| v.extract::<bool>())
        .unwrap_or(Ok(false))?;

    let disagg = dict
        .get_item("disagg")?
        .map(|v| v.extract::<bool>())
        .unwrap_or(Ok(false))?;

    let pd_kvbm_enabled = dict
        .get_item("pd_kvbm_enabled")?
        .map(|v| v.extract::<bool>())
        .unwrap_or(Ok(false))?;
    let pd_decode_direct = dict
        .get_item("pd_decode_direct")?
        .map(|v| v.extract::<bool>())
        .unwrap_or(Ok(false))?;
    let pd_prompt_token_count = dict
        .get_item("pd_prompt_token_count")?
        .map(|v| v.extract::<i64>())
        .unwrap_or(Ok(0))?;
    let pd_timeout_ms = dict
        .get_item("pd_timeout_ms")?
        .map(|v| v.extract::<u64>())
        .unwrap_or(Ok(10_000))?;
    let pd_export_timeout_ms = dict
        .get_item("pd_export_timeout_ms")?
        .map(|v| v.extract::<u64>())
        .unwrap_or(Ok(120_000))?;

    Ok(RouteDecision {
        target_url: get_str("target_url")?,
        worker_id: get_str("worker_id")?,
        headers,
        body_json: get_str("body_json")?,
        request_id: get_str("request_id")?,
        stream,
        disagg,
        decode_target_url: get_str("decode_target_url")?,
        decode_worker_id: get_str("decode_worker_id")?,
        decode_body_json: get_str("decode_body_json")?,
        pd_kvbm_enabled,
        pd_decode_direct,
        pd_handoff_id: get_str("pd_handoff_id")?,
        pd_decode_admission_ticket_id: get_str("pd_decode_admission_ticket_id")?,
        pd_request_id: get_str("pd_request_id")?,
        pd_prefill_control_url: get_str("pd_prefill_control_url")?,
        pd_decode_control_url: get_str("pd_decode_control_url")?,
        pd_prompt_token_count,
        pd_timeout_ms,
        pd_export_timeout_ms,
    })
}

// ---------------------------------------------------------------------------
// Python 回调函数，使用 Arc 实现线程安全共享
// ---------------------------------------------------------------------------

/// 封装 Python 侧的两个回调：route_fn（路由决策）和 complete_fn（请求完成通知）。
struct PyCallbacks {
    route_fn: PyObject,
    complete_fn: PyObject,
    models_fn: PyObject,
    pd_prefill_complete_fn: Option<PyObject>,
    pd_dispatch_snapshot_fn: Option<PyObject>,
    pd_decode_budget_fn: Option<PyObject>,
}

// Safety: PyObject is Send. We acquire the GIL before any access.
unsafe impl Send for PyCallbacks {}
unsafe impl Sync for PyCallbacks {}

// ---------------------------------------------------------------------------
// 共享应用状态
// ---------------------------------------------------------------------------

/// Axum 路由共享状态：Python 回调、HTTP 连接池客户端、请求体大小限制。
#[derive(Clone)]
struct AppState {
    callbacks: Arc<PyCallbacks>,
    http_client: Client,
    max_body_size: usize,
    pd_link_cache: Arc<RwLock<HashMap<(String, String), i64>>>,
    pd_prefill_admission: Arc<Mutex<HashMap<String, Arc<Semaphore>>>>,
    pd_prefill_max_inflight: usize,
}

// ---------------------------------------------------------------------------
// 错误类型
// ---------------------------------------------------------------------------

/// 代理错误分类：客户端请求错误、控制面错误、上游 worker 错误、内部错误。
#[derive(Debug)]
enum ProxyError {
    BadRequest(String),
    NoCapacity(String),
    ControlPlane(String),
    Upstream(String),
    Internal(String),
}

impl std::fmt::Display for ProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadRequest(m) => write!(f, "Bad request: {m}"),
            Self::NoCapacity(m) => write!(f, "No worker capacity: {m}"),
            Self::ControlPlane(m) => write!(f, "Control plane: {m}"),
            Self::Upstream(m) => write!(f, "Upstream: {m}"),
            Self::Internal(m) => write!(f, "Internal: {m}"),
        }
    }
}

impl std::error::Error for ProxyError {}

impl ProxyError {
    fn response_body(&self) -> String {
        json!({"error": {"message": self.to_string(), "type": "proxy_error"}}).to_string()
    }
    fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::NoCapacity(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::BAD_GATEWAY,
        }
    }
}

fn positive_env(name: &str, fallback: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(fallback)
}

async fn read_response_limited(response: reqwest::Response) -> Result<Bytes, String> {
    let limit = positive_env("JIUWEN_PROXY_MAX_RESPONSE_BYTES", 64 * 1024 * 1024) as usize;
    read_response_with_limit(response, limit).await
}

async fn read_response_with_limit(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<Bytes, String> {
    if response
        .content_length()
        .is_some_and(|size| size > limit as u64)
    {
        return Err("Upstream response exceeds configured limit".into());
    }
    let mut data = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
        if chunk.len() > limit.saturating_sub(data.len()) {
            return Err("Upstream response exceeds configured limit".into());
        }
        data.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(data))
}

async fn read_text_limited(response: reqwest::Response) -> Result<String, String> {
    read_response_limited(response)
        .await
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

// ---------------------------------------------------------------------------
// 请求处理
// ---------------------------------------------------------------------------

/// 健康检查端点，返回 {"status":"ok"}。
async fn health_handler() -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"status":"ok"}"#))
        .unwrap()
}

/// 返回当前 Router 已发现的 OpenAI 兼容模型列表。
async fn models_handler(State(state): State<AppState>) -> Response<Body> {
    let callbacks = state.callbacks.clone();
    let result = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| callbacks.models_fn.call0(py)?.extract::<String>(py))
    })
    .await;

    match result {
        Ok(Ok(body)) => Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(body))
            .unwrap(),
        Ok(Err(error)) => {
            error!("Models callback failed: {error}");
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    json!({
                        "error": {
                            "message": format!("Models callback: {error}"),
                            "type": "internal_error",
                        }
                    })
                    .to_string(),
                ))
                .unwrap()
        }
        Err(error) => {
            error!("Models callback task failed: {error}");
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    json!({
                        "error": {
                            "message": format!("Models callback task: {error}"),
                            "type": "internal_error",
                        }
                    })
                    .to_string(),
                ))
                .unwrap()
        }
    }
}

/// 代理请求处理入口，捕获错误并转为 JSON 错误响应。
async fn proxy_handler(State(state): State<AppState>, req: Request<Body>) -> Response<Body> {
    match handle_proxy(&state, req).await {
        Ok(resp) => resp,
        Err(e) => {
            error!("Proxy error: {e}");
            let status = e.status();
            Response::builder()
                .status(status)
                .header("Content-Type", "application/json")
                .body(Body::from(e.response_body()))
                .unwrap()
        }
    }
}

/// 代理核心逻辑：读取请求体 → 调用 Python 路由决策 → 转发到 worker → 返回响应。
///
/// PD 分离模式（disagg=true）下执行两步转发：
///   1. 发送 prefill 请求到 prefill worker，等待完成（max_tokens=1, stream=false）
///   2. 发送 decode 请求到 decode worker，将响应返回客户端
async fn handle_proxy(state: &AppState, req: Request<Body>) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    let path = req.uri().path().to_string();
    let original_headers = req.headers().clone();

    // 1. Read body
    let body_bytes = axum::body::to_bytes(req.into_body(), state.max_body_size)
        .await
        .map_err(|e| ProxyError::BadRequest(format!("Failed to read body: {e}")))?;

    let body_json = String::from_utf8(body_bytes.to_vec())
        .map_err(|e| ProxyError::BadRequest(format!("Invalid UTF-8 body: {e}")))?;

    // 2. Call Python route_fn via spawn_blocking + GIL
    let cb = state.callbacks.clone();
    let path_clone = path.clone();
    let body_clone = body_json.clone();

    let decision = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            let result = cb
                .route_fn
                .call1(py, (&body_clone, &path_clone))
                .map_err(|e| {
                    let code = e
                        .value(py)
                        .getattr("status_code")
                        .and_then(|v| v.extract::<u16>())
                        .ok();
                    if code == Some(503) {
                        ProxyError::NoCapacity(e.to_string())
                    } else {
                        ProxyError::ControlPlane(e.to_string())
                    }
                })?;
            let bound = result.bind(py);
            extract_route_decision(bound)
                .map_err(|e| ProxyError::ControlPlane(format!("Bad route response: {e}")))
        })
    })
    .await
    .map_err(|e| ProxyError::Internal(format!("spawn_blocking join: {e}")))??;

    if decision.disagg {
        handle_disagg_proxy(state, decision, &path, &original_headers, start).await
    } else {
        handle_combined_proxy(state, decision, &path, &original_headers, start).await
    }
}

/// 组合模式：单跳转发到一个 worker。
async fn handle_combined_proxy(
    state: &AppState,
    decision: RouteDecision,
    path: &str,
    original_headers: &axum::http::HeaderMap,
    start: Instant,
) -> Result<Response<Body>, ProxyError> {
    let request_id = decision.request_id.clone();
    let worker_id = decision.worker_id.clone();
    let is_stream = decision.stream;

    info!(
        request_id = %request_id,
        worker = %worker_id,
        stream = is_stream,
        route_ms = start.elapsed().as_secs_f64() * 1000.0,
        "Routed request (combined)"
    );

    let upstream_url = format!("{}{}", decision.target_url, path);
    let mut req_builder = state
        .http_client
        .post(&upstream_url)
        .header("Content-Type", "application/json");

    if let Some(auth) = original_headers.get("authorization") {
        if let Ok(v) = auth.to_str() {
            req_builder = req_builder.header("Authorization", v);
        }
    }

    for (k, v) in &decision.headers {
        req_builder = req_builder.header(k.as_str(), v.as_str());
    }

    req_builder = req_builder.body(decision.body_json);

    let worker_dispatch_started = Instant::now();
    let upstream_resp = match req_builder.send().await {
        Ok(resp) => resp,
        Err(e) => {
            notify_complete_background(state.callbacks.clone(), &request_id, &worker_id, start, 0);
            return Err(ProxyError::Upstream(format!(
                "Failed to connect to worker: {e}"
            )));
        }
    };

    let status = upstream_resp.status();
    if !status.is_success() {
        let err_body = read_text_limited(upstream_resp)
            .await
            .unwrap_or_else(|_| "unknown error".into());
        warn!(request_id = %request_id, worker = %worker_id, status = %status, "Upstream error");
        notify_complete_background(state.callbacks.clone(), &request_id, &worker_id, start, 0);
        return Ok(Response::builder()
            .status(status.as_u16())
            .header("Content-Type", "application/json")
            .body(Body::from(err_body))
            .unwrap());
    }

    if is_stream {
        build_streaming_response(
            state,
            upstream_resp,
            request_id,
            worker_id,
            start,
            worker_dispatch_started,
            0.0,
            0.0,
            None,
            json!({}),
        )
    } else {
        build_buffered_response(state, upstream_resp, request_id, worker_id, start).await
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct KvbmControlHttpTiming {
    send_to_headers_ms: f64,
    headers_to_body_ms: f64,
    json_decode_ms: f64,
    total_ms: f64,
}

/// 调用 Worker KVBM 控制接口，并保留非 2xx 响应中的结构化错误信息。
///
/// 参数：
/// - `client`: Router 共享的 HTTP 连接池客户端。
/// - `base_url`: Worker KVBM 控制服务根地址。
/// - `path`: 控制接口路径。
/// - `payload`: 待发送的 JSON 请求体。
/// - `timeout_ms`: 单次 HTTP 请求超时，单位毫秒。
///
/// 返回：成功时返回 Worker JSON；网络、状态码或协议异常时返回错误文本。
async fn post_kvbm_control(
    client: &Client,
    base_url: &str,
    path: &str,
    payload: Value,
    timeout_ms: u64,
) -> Result<Value, String> {
    post_kvbm_control_timed(client, base_url, path, payload, timeout_ms)
        .await
        .0
}

/// 调用 Worker KVBM 控制接口，同时返回 Router 侧 HTTP 分阶段耗时。
async fn post_kvbm_control_timed(
    client: &Client,
    base_url: &str,
    path: &str,
    payload: Value,
    timeout_ms: u64,
) -> (Result<Value, String>, KvbmControlHttpTiming) {
    let total_started = Instant::now();
    let mut timing = KvbmControlHttpTiming::default();
    let url = format!("{}{}", base_url.trim_end_matches('/'), path);
    let response = match client
        .post(&url)
        .header("Content-Type", "application/json")
        .timeout(Duration::from_millis(timeout_ms.max(1)))
        .body(payload.to_string())
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            timing.total_ms = total_started.elapsed().as_secs_f64() * 1000.0;
            return (
                Err(format!(
                    "{url}: {error}; is_timeout={} is_connect={} detail={error:?}",
                    error.is_timeout(),
                    error.is_connect(),
                )),
                timing,
            );
        }
    };
    timing.send_to_headers_ms = total_started.elapsed().as_secs_f64() * 1000.0;
    let status = response.status();
    let body_started = Instant::now();
    let body = match read_response_limited(response).await {
        Ok(body) => body,
        Err(error) => {
            timing.headers_to_body_ms = body_started.elapsed().as_secs_f64() * 1000.0;
            timing.total_ms = total_started.elapsed().as_secs_f64() * 1000.0;
            return (
                Err(format!("{url}: failed to read response: {error}")),
                timing,
            );
        }
    };
    timing.headers_to_body_ms = body_started.elapsed().as_secs_f64() * 1000.0;
    let decode_started = Instant::now();
    let value = match serde_json::from_slice::<Value>(&body) {
        Ok(value) => value,
        Err(error) => {
            timing.json_decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
            timing.total_ms = total_started.elapsed().as_secs_f64() * 1000.0;
            return (
                Err(format!(
                    "{url}: invalid JSON response status={status}: {error}; body={}",
                    String::from_utf8_lossy(&body)
                )),
                timing,
            );
        }
    };
    timing.json_decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    timing.total_ms = total_started.elapsed().as_secs_f64() * 1000.0;
    let result = if status.is_success() {
        Ok(value)
    } else {
        Err(format!("{url}: status={status} response={value}"))
    };
    (result, timing)
}

/// 将Worker推理请求发送错误展开为可诊断文本。
///
/// 参数：
/// - `worker_kind`: Worker角色，例如Decode或Prefill。
/// - `url`: 实际请求地址。
/// - `error`: reqwest返回的发送错误。
/// - `elapsed_ms`: 从请求派发到观察到错误的耗时。
///
/// 返回：包含超时、连接错误分类和Debug错误链的文本。
fn format_worker_send_error(
    worker_kind: &str,
    url: &str,
    error: &reqwest::Error,
    elapsed_ms: f64,
) -> String {
    format!(
        "{worker_kind} worker request failed: url={url} elapsed_ms={elapsed_ms:.3} \
         is_timeout={} is_connect={} is_request={} detail={error:?}",
        error.is_timeout(),
        error.is_connect(),
        error.is_request(),
    )
}

/// Merge native KVConnector metadata returned by P into the pending D request.
fn merge_native_kv_transfer_params(
    decode_body_json: &str,
    prefill_response_body: &[u8],
) -> Result<(String, usize), String> {
    let prefill_response: Value = serde_json::from_slice(prefill_response_body)
        .map_err(|error| format!("invalid P response JSON: {error}"))?;
    let mut transfer_params = prefill_response
        .get("kv_transfer_params")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| "P response did not contain kv_transfer_params".to_string())?;
    transfer_params.insert("do_remote_prefill".to_string(), Value::Bool(true));
    transfer_params.insert("do_remote_decode".to_string(), Value::Bool(false));
    let remote_block_count = transfer_params
        .get("remote_block_ids")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);

    let mut decode_body: Value = serde_json::from_str(decode_body_json)
        .map_err(|error| format!("invalid D request JSON: {error}"))?;
    decode_body
        .as_object_mut()
        .ok_or_else(|| "D request JSON must be an object".to_string())?
        .insert(
            "kv_transfer_params".to_string(),
            Value::Object(transfer_params),
        );
    serde_json::to_string(&decode_body)
        .map(|body| (body, remote_block_count))
        .map_err(|error| format!("failed to serialize D request JSON: {error}"))
}

/// 判断 KVBM 控制响应是否明确返回 `ok=true`。
///
/// 参数：`value` 为 Worker 返回的 JSON 对象。
/// 返回：响应成功时为 `true`，否则为 `false`。
fn control_succeeded(value: &Value) -> bool {
    value.get("ok").and_then(Value::as_bool).unwrap_or(false)
}

/// 一次P到D链路解析结果及其控制面分阶段耗时。
struct PdLinkResolution {
    cluster_id: i64,
    cache_hit: bool,
    source_status_ms: f64,
    target_status_ms: f64,
    link_call_ms: f64,
}

/// 从两侧 endpoint 状态建立 D 到 P 的单向 LLM-DataDist 链路。
///
/// 参数：
/// - `client`: Router 共享的 HTTP 连接池客户端。
/// - `prefill_control_url`: P Worker KVBM 控制地址。
/// - `decode_control_url`: D Worker KVBM 控制地址。
/// - `timeout_ms`: status 和 link 的单次调用超时。
///
/// 返回：成功时返回 P Worker cluster ID，失败时返回诊断文本。
async fn ensure_pd_datadist_link(
    client: &Client,
    link_cache: &RwLock<HashMap<(String, String), i64>>,
    prefill_control_url: &str,
    decode_control_url: &str,
    timeout_ms: u64,
) -> Result<PdLinkResolution, String> {
    let cache_key = (
        prefill_control_url.trim_end_matches('/').to_string(),
        decode_control_url.trim_end_matches('/').to_string(),
    );
    if let Some(cluster_id) = link_cache
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&cache_key)
        .copied()
    {
        return Ok(PdLinkResolution {
            cluster_id,
            cache_hit: true,
            source_status_ms: 0.0,
            target_status_ms: 0.0,
            link_call_ms: 0.0,
        });
    }
    let source_status_future = async {
        let started = Instant::now();
        let result = post_kvbm_control(
            client,
            prefill_control_url,
            "/datadist/status",
            json!({}),
            timeout_ms,
        )
        .await;
        (result, started.elapsed().as_secs_f64() * 1000.0)
    };
    let target_status_future = async {
        let started = Instant::now();
        let result = post_kvbm_control(
            client,
            decode_control_url,
            "/datadist/status",
            json!({}),
            timeout_ms,
        )
        .await;
        (result, started.elapsed().as_secs_f64() * 1000.0)
    };
    let ((source_status, source_status_ms), (target_status, target_status_ms)) =
        tokio::join!(source_status_future, target_status_future);
    let source_status = source_status?;
    let target_status = target_status?;
    if !control_succeeded(&source_status) || !control_succeeded(&target_status) {
        return Err(format!(
            "source or target LLM-DataDist backend is not ready: source={source_status} target={target_status}"
        ));
    }
    let source_endpoint = &source_status["health"]["endpoint"];
    let target_endpoint = &target_status["health"]["endpoint"];
    let source_cluster_id = source_endpoint["cluster_id"]
        .as_i64()
        .ok_or_else(|| "source status is missing health.endpoint.cluster_id".to_string())?;
    let already_linked = target_status["health"]["linked_cluster_ids"]
        .as_array()
        .is_some_and(|ids| {
            ids.iter()
                .any(|value| value.as_i64() == Some(source_cluster_id))
        });
    if already_linked {
        link_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(cache_key, source_cluster_id);
        return Ok(PdLinkResolution {
            cluster_id: source_cluster_id,
            cache_hit: false,
            source_status_ms,
            target_status_ms,
            link_call_ms: 0.0,
        });
    }
    let link_payload = json!({
        "remote_cluster_id": source_cluster_id,
        "remote_ip": source_endpoint["listen_ip"],
        "remote_port": source_endpoint["listen_port"],
        "local_ip": target_endpoint["listen_ip"],
        "local_port": target_endpoint["listen_port"],
        "timeout_ms": timeout_ms,
    });
    let link_started = Instant::now();
    let link_result = post_kvbm_control(
        client,
        decode_control_url,
        "/datadist/link",
        link_payload,
        timeout_ms.saturating_add(200),
    )
    .await;
    let link_call_ms = link_started.elapsed().as_secs_f64() * 1000.0;
    match link_result {
        Ok(value) if control_succeeded(&value) => {
            link_cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(cache_key, source_cluster_id);
            Ok(PdLinkResolution {
                cluster_id: source_cluster_id,
                cache_hit: false,
                source_status_ms,
                target_status_ms,
                link_call_ms,
            })
        }
        Ok(value) => Err(format!("LLM-DataDist link failed: {value}")),
        Err(error) if error.contains("LLM_ALREADY_LINK") => {
            link_cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(cache_key, source_cluster_id);
            Ok(PdLinkResolution {
                cluster_id: source_cluster_id,
                cache_hit: false,
                source_status_ms,
                target_status_ms,
                link_call_ms,
            })
        }
        Err(error) => Err(error),
    }
}

/// 判断import失败是否表示缓存中的Worker链路已经失效。
fn pd_import_reports_not_linked(result: &Result<Value, String>) -> bool {
    match result {
        Ok(value) => value.to_string().contains("LLM_NOT_YET_LINK"),
        Err(error) => error.contains("LLM_NOT_YET_LINK"),
    }
}

/// 解除 D 请求 gate 并释放 P 侧 pin；相关接口均为幂等清理操作。
///
/// 参数：
/// - `client`: Router 共享的 HTTP 连接池客户端。
/// - `decision`: 当前 PD 路由决策及控制地址。
/// - `reason`: 写入 handoff rollback 的失败原因。
///
/// 返回：无；清理接口失败不会覆盖原始推理错误。
async fn release_pd_handoff(client: &Client, decision: &RouteDecision, reason: &str) {
    let payload = json!({
        "handoff_id": decision.pd_handoff_id,
        "request_id": decision.pd_request_id,
        "admission_ticket_id": decision.pd_decode_admission_ticket_id,
        "reason": reason,
    });
    let rollback_payload = json!({
        "handoff_id": decision.pd_handoff_id,
        "reason": reason,
    });
    let _ = tokio::join!(
        post_kvbm_control(
            client,
            &decision.pd_decode_control_url,
            "/datadist/pd/import/release",
            payload.clone(),
            decision.pd_timeout_ms,
        ),
        post_kvbm_control(
            client,
            &decision.pd_prefill_control_url,
            "/datadist/pd/export/release",
            payload,
            decision.pd_timeout_ms,
        ),
    );
    let _ = tokio::join!(
        post_kvbm_control(
            client,
            &decision.pd_decode_control_url,
            "/datadist/pd/handoffs/rollback",
            rollback_payload.clone(),
            decision.pd_timeout_ms,
        ),
        post_kvbm_control(
            client,
            &decision.pd_prefill_control_url,
            "/datadist/pd/handoffs/rollback",
            rollback_payload,
            decision.pd_timeout_ms,
        ),
    );
}

/// 等待 D scheduler 观察到 gated request，防止 attach 早于 add_request。
///
/// 参数：
/// - `client`: Router 共享的 HTTP 连接池客户端。
/// - `decision`: 包含 handoff ID、engine request ID 和 D 控制地址的路由决策。
///
/// 返回：scheduler 已观察请求时返回Worker状态；控制请求失败或超时时返回错误文本。
async fn wait_for_pd_request(
    client: &Client,
    decision: &RouteDecision,
    deadline: Instant,
) -> Result<Value, String> {
    let payload = json!({
        "handoff_id": decision.pd_handoff_id,
        "request_id": decision.pd_request_id,
    });
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err("timed out waiting for D scheduler to observe request".to_string());
        }
        let remaining_ms = deadline
            .saturating_duration_since(now)
            .as_millis()
            .clamp(1, u64::MAX as u128) as u64;
        let status = post_kvbm_control(
            client,
            &decision.pd_decode_control_url,
            "/datadist/pd/import/status",
            payload.clone(),
            remaining_ms,
        )
        .await?;
        if status["gate"]["request_seen"].as_bool().unwrap_or(false) {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            return Err("timed out waiting for D scheduler to observe request".to_string());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Router持有的单次PD事务事件长连接，可订阅P或D Worker。
struct PdControlEventStream {
    response: reqwest::Response,
    buffer: Vec<u8>,
}

/// 判断P export响应是否表示捕获仍在进行、可以继续等待。
fn pd_export_is_pending(result: &Result<Value, String>) -> bool {
    matches!(
        result,
        Ok(value)
            if value["pending"].as_bool().unwrap_or(false)
                && value["retryable"].as_bool().unwrap_or(false)
    )
}

/// 通过P Worker事件流等待export捕获完成。
async fn wait_for_pd_source_export_ready(
    client: &Client,
    decision: &RouteDecision,
) -> Result<Value, String> {
    let connect = PdControlEventStream::connect(
        client,
        &decision.pd_prefill_control_url,
        &decision.pd_handoff_id,
        &decision.pd_request_id,
    );
    let mut stream = tokio::time::timeout(Duration::from_millis(decision.pd_timeout_ms), connect)
        .await
        .map_err(|_| "timed out connecting P export event stream".to_string())??;
    let first = stream
        .wait_for(
            &decision.pd_handoff_id,
            &["P_REQUEST_SEEN", "EXPORT_READY", "EXPORT_FAILED"],
            decision.pd_timeout_ms,
        )
        .await
        .map_err(|error| format!("P request did not reach scheduler: {error}"))?;
    let event = if matches!(
        first["event_type"].as_str(),
        Some("EXPORT_READY" | "EXPORT_FAILED")
    ) {
        first
    } else {
        stream
            .wait_for(
                &decision.pd_handoff_id,
                &["EXPORT_READY", "EXPORT_FAILED"],
                decision.pd_export_timeout_ms,
            )
            .await?
    };
    if event["event_type"].as_str() == Some("EXPORT_FAILED") {
        return Err(event["result"]["error"]
            .as_str()
            .unwrap_or("P request block capture failed")
            .to_string());
    }
    Ok(event)
}

/// 计算覆盖完整PD控制事务的Worker资源TTL。
///
/// 参数：
/// - `export_timeout_ms`: P节点完成prefill和export捕获的最长等待时间。
/// - `control_timeout_ms`: 单个控制阶段的超时时间。
///
/// 返回：同时覆盖P capture lease、D gate和后续import/reconcile的TTL，单位毫秒。
fn pd_transaction_ttl_ms(export_timeout_ms: u64, control_timeout_ms: u64) -> u64 {
    export_timeout_ms
        .saturating_add(pd_import_completion_timeout_ms(control_timeout_ms))
        .clamp(120_000, 600_000)
}

/// 计算P export完成后等待D Scheduler容量和HIXL import的总预算。
///
/// 参数：
/// - `control_timeout_ms`: 单个控制阶段的超时时间。
///
/// 返回：import事件流和兜底结果查询共享的总等待时间，单位毫秒。
fn pd_import_completion_timeout_ms(control_timeout_ms: u64) -> u64 {
    control_timeout_ms
        .saturating_mul(4)
        .saturating_add(30_000)
        .clamp(60_000, 300_000)
}

/// 获取指定P Worker的通用PD Prefill阶段并发许可。
///
/// 参数：
/// - `state`: Rust数据面的共享状态，保存按Worker划分的Semaphore。
/// - `worker_id`: 当前请求选中的P Worker ID。
/// - `timeout_ms`: 等待并发许可的最长时间。
///
/// 返回：成功时返回持有的permit和排队耗时；超时或Semaphore关闭时返回错误。
async fn acquire_pd_prefill_permit(
    state: &AppState,
    worker_id: &str,
    timeout_ms: u64,
) -> Result<(Option<OwnedSemaphorePermit>, f64), String> {
    // max_inflight=0 表示 Python Router 已启用动态 token/block 预算，Rust
    // 数据面不再叠加固定 Semaphore，以免限制 Worker 的真实执行能力。
    if state.pd_prefill_max_inflight == 0 {
        return Ok((None, 0.0));
    }
    let semaphore = {
        let mut admission = state
            .pd_prefill_admission
            .lock()
            .map_err(|_| "PD prefill admission lock is poisoned".to_string())?;
        admission
            .entry(worker_id.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(state.pd_prefill_max_inflight)))
            .clone()
    };
    let started = Instant::now();
    let permit = tokio::time::timeout(Duration::from_millis(timeout_ms), semaphore.acquire_owned())
        .await
        .map_err(|_| {
            format!("timed out waiting for P Worker {worker_id} prefill admission permit")
        })?
        .map_err(|_| format!("P Worker {worker_id} prefill admission is closed"))?;
    Ok((Some(permit), started.elapsed().as_secs_f64() * 1000.0))
}

impl PdControlEventStream {
    /// 订阅D Worker针对一个handoff发布的NDJSON控制事件。
    async fn connect(
        client: &Client,
        decode_control_url: &str,
        handoff_id: &str,
        request_id: &str,
    ) -> Result<Self, String> {
        let url = format!(
            "{}/datadist/pd/events",
            decode_control_url.trim_end_matches('/')
        );
        let response = client
            .get(url)
            .query(&[
                ("handoff_id", handoff_id),
                ("request_id", request_id),
                ("after_sequence", "0"),
            ])
            .send()
            .await
            .map_err(|error| format!("failed to connect PD event stream: {error}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = read_text_limited(response).await.unwrap_or_default();
            return Err(format!("PD event stream returned HTTP {status}: {body}"));
        }
        Ok(Self {
            response,
            buffer: Vec::new(),
        })
    }

    /// 读取下一个完整NDJSON事件。
    async fn next_event(&mut self) -> Result<Value, String> {
        loop {
            if let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
                let line: Vec<u8> = self.buffer.drain(..=position).collect();
                let line = &line[..line.len().saturating_sub(1)];
                if line.is_empty() {
                    continue;
                }
                return serde_json::from_slice(line)
                    .map_err(|error| format!("invalid PD event JSON: {error}"));
            }
            let chunk = self
                .response
                .chunk()
                .await
                .map_err(|error| format!("failed to read PD event stream: {error}"))?
                .ok_or_else(|| "PD event stream closed before terminal event".to_string())?;
            self.buffer.extend_from_slice(&chunk);
        }
    }

    /// 等待指定事件类型；心跳和其他事务事件会被忽略。
    async fn wait_for(
        &mut self,
        handoff_id: &str,
        event_types: &[&str],
        timeout_ms: u64,
    ) -> Result<Value, String> {
        tokio::time::timeout(Duration::from_millis(timeout_ms), async {
            loop {
                let event = self.next_event().await?;
                if event["handoff_id"].as_str() != Some(handoff_id) {
                    continue;
                }
                let event_type = event["event_type"].as_str().unwrap_or_default();
                if event_types.contains(&event_type) {
                    return Ok(event);
                }
            }
        })
        .await
        .map_err(|_| format!("timed out waiting for PD events {event_types:?}"))?
    }
}

/// 优先通过Worker推送等待D请求进入scheduler，长连接异常时退回旧轮询接口。
async fn wait_for_pd_request_ready(
    stream: &mut Option<PdControlEventStream>,
    client: &Client,
    decision: &RouteDecision,
) -> Result<Value, String> {
    let deadline = Instant::now() + Duration::from_millis(decision.pd_timeout_ms);
    if let Some(events) = stream.as_mut() {
        let remaining_ms = deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .clamp(1, u64::MAX as u128) as u64;
        match events
            .wait_for(&decision.pd_handoff_id, &["REQUEST_SEEN"], remaining_ms)
            .await
        {
            Ok(event) => return Ok(event),
            Err(error) => {
                warn!(
                    request_id = %decision.request_id,
                    error = %error,
                    "KVBM PD event stream failed while waiting for request; falling back to status polling"
                );
                *stream = None;
            }
        }
    }
    wait_for_pd_request(client, decision, deadline).await
}

/// 判断D Worker是否已经明确拒绝本次delta import容量准入。
///
/// 参数：
/// - `status`: REQUEST_SEEN事件或兼容轮询返回的gate状态。
///
/// 返回：容量不足时返回Worker错误文本，否则返回None。
fn pd_preplan_capacity_error(status: &Value) -> Option<String> {
    let event_result = status.get("result");
    if event_result
        .and_then(|result| result["capacity_unavailable"].as_bool())
        .unwrap_or(false)
    {
        return Some(
            event_result
                .and_then(|result| result["preplan_error"].as_str())
                .unwrap_or("D Worker has insufficient free blocks for delta import")
                .to_string(),
        );
    }
    if status["gate"]["preplan_capacity_waiting"]
        .as_bool()
        .unwrap_or(false)
    {
        return None;
    }
    let gate_error = status["gate"]["preplan_error"].as_str()?;
    gate_error
        .contains("insufficient free vLLM blocks for delta import")
        .then(|| gate_error.to_string())
}

/// 从REQUEST_SEEN、状态快照或目标分配事件中提取Scheduler分配元数据。
///
/// 参数：
/// - `status`: D Worker返回的PD事件或gate状态快照。
///
/// 返回：Scheduler已经分配目标block时返回回调元数据，否则返回None。
fn pd_target_allocation_metadata(status: &Value) -> Option<Value> {
    let event_result = status.get("result");
    if event_result
        .and_then(|result| result["scheduler_allocation_ready"].as_bool())
        .unwrap_or(false)
    {
        return Some(json!({
            "allocated_blocks": event_result
                .and_then(|result| result["allocated_blocks"].as_u64())
                .unwrap_or(0),
            "capacity_generation": event_result
                .and_then(|result| result["capacity_generation"].as_u64()),
            "source": status["event_type"].as_str().unwrap_or("request_seen"),
        }));
    }
    if status["gate"]["scheduler_allocation_ready"]
        .as_bool()
        .unwrap_or(false)
    {
        return Some(json!({
            "allocated_blocks": status["gate"]["scheduler_allocation"]["total_blocks"]
                .as_u64()
                .unwrap_or(0),
            "capacity_generation": status["gate"]["scheduler_allocation"]
                ["capacity_generation"]
                .as_u64(),
            "source": "status",
        }));
    }
    None
}

/// 提交异步import并等待Worker推送结果；流断开时只做一次阻塞结果查询。
async fn submit_and_wait_pd_import(
    stream: &mut Option<PdControlEventStream>,
    client: &Client,
    decision: &RouteDecision,
    payload: Value,
    callbacks: Arc<PyCallbacks>,
    request_id: &str,
    decode_worker_id: &str,
) -> Result<Value, String> {
    let total_timeout_ms = pd_import_completion_timeout_ms(decision.pd_timeout_ms);
    if stream.is_none() {
        return post_kvbm_control(
            client,
            &decision.pd_decode_control_url,
            "/datadist/pd/import",
            payload,
            total_timeout_ms,
        )
        .await;
    }
    let accepted = post_kvbm_control(
        client,
        &decision.pd_decode_control_url,
        "/datadist/pd/import/submit",
        payload,
        decision.pd_timeout_ms,
    )
    .await?;
    if !control_succeeded(&accepted) {
        return Ok(accepted);
    }
    let import_deadline = Instant::now() + Duration::from_millis(total_timeout_ms);
    let stream_result = loop {
        let remaining_ms = import_deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .clamp(1, u128::from(u64::MAX)) as u64;
        let event = stream
            .as_mut()
            .expect("PD event stream checked above")
            .wait_for(
                &decision.pd_handoff_id,
                &[
                    "TARGET_BLOCKS_ALLOCATED",
                    "IMPORT_COMPLETE",
                    "IMPORT_FAILED",
                ],
                remaining_ms,
            )
            .await;
        match event {
            Ok(event) if event["event_type"].as_str() == Some("TARGET_BLOCKS_ALLOCATED") => {
                if let Some(metadata) = pd_target_allocation_metadata(&event) {
                    notify_pd_decode_budget_background(
                        callbacks.clone(),
                        request_id,
                        decode_worker_id,
                        "allocated",
                        &metadata,
                    );
                }
            }
            other => break other,
        }
    };
    match stream_result {
        Ok(event) => event
            .get("result")
            .cloned()
            .ok_or_else(|| "PD import event is missing result".to_string()),
        Err(error) => {
            warn!(
                request_id = %decision.request_id,
                error = %error,
                "KVBM PD event stream failed while waiting for import; querying result once"
            );
            *stream = None;
            let remaining_timeout_ms = import_deadline
                .saturating_duration_since(Instant::now())
                .as_millis()
                .min(u128::from(u64::MAX)) as u64;
            post_kvbm_control(
                client,
                &decision.pd_decode_control_url,
                "/datadist/pd/import/result",
                json!({
                    "handoff_id": &decision.pd_handoff_id,
                    "wait_timeout_ms": remaining_timeout_ms,
                }),
                remaining_timeout_ms.saturating_add(500),
            )
            .await
        }
    }
}

/// PD 分离模式：先发 prefill 请求等待 KV 传输完成，再发 decode 请求返回给客户端。
async fn handle_disagg_proxy(
    state: &AppState,
    mut decision: RouteDecision,
    path: &str,
    original_headers: &axum::http::HeaderMap,
    start: Instant,
) -> Result<Response<Body>, ProxyError> {
    if decision.pd_decode_direct {
        let pre_decode_total_ms = start.elapsed().as_secs_f64() * 1000.0;
        info!(
            request_id = %decision.request_id,
            decode = %decision.decode_worker_id,
            route_ms = pre_decode_total_ms,
            pd_begin_ms = 0.0,
            p_request_ms = 0.0,
            pd_export_ms = 0.0,
            pd_link_ms = 0.0,
            pd_prepare_ms = 0.0,
            d_gate_wait_ms = 0.0,
            pd_import_ms = 0.0,
            pre_decode_total_ms = pre_decode_total_ms,
            "[KVBMPDTiming] decode-direct; skipped P request and PD handoff"
        );
        decision.target_url = decision.decode_target_url.clone();
        decision.worker_id = decision.decode_worker_id.clone();
        decision.body_json = decision.decode_body_json.clone();
        decision.disagg = false;
        return handle_combined_proxy(state, decision, path, original_headers, start).await;
    }

    let request_id = decision.request_id.clone();
    let prefill_worker_id = decision.worker_id.clone();
    let decode_worker_id = decision.decode_worker_id.clone();
    let is_stream = decision.stream;
    let route_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut pd_begin_ms = 0.0;
    let mut pd_source_begin_ms = 0.0;
    let mut pd_target_begin_ms = 0.0;
    let mut p_request_ms = 0.0;
    let mut p_completed_at: Option<Instant> = None;
    let mut pd_export_ms = 0.0;
    let mut pd_link_ms = 0.0;
    let mut pd_link_cache_hit = false;
    let mut pd_source_status_ms = 0.0;
    let mut pd_target_status_ms = 0.0;
    let mut pd_link_call_ms = 0.0;
    let mut pd_prepare_ms = 0.0;
    let mut pd_import_ms = 0.0;
    let mut event_stream_connect_ms = 0.0;
    let p_http_dispatch_ms;
    let mut p_prepare_to_http_dispatch_ms = 0.0;
    let mut p_dispatch_to_scheduler_seen_ms = 0.0;
    let mut p_worker_request_received_to_scheduler_seen_ms = 0.0;
    let mut p_worker_request_received_timing_source = "unavailable".to_string();
    let mut p_scheduler_queue_cycles = 0_i64;
    let mut p_scheduler_seen_cycle_id = 0_i64;
    let mut p_first_scheduled_cycle_id = 0_i64;
    let mut p_kv_ready_cycle_id = 0_i64;
    let mut p_scheduler_compute_cycles = 0_i64;
    let mut p_first_scheduled_token_count = 0_i64;
    let mut p_scheduler_running_request_count = 0_i64;
    let mut p_scheduler_waiting_request_count = 0_i64;
    let mut p_first_scheduled_running_request_count = 0_i64;
    let mut p_first_scheduled_waiting_request_count = 0_i64;
    let mut p_first_scheduled_cycle_ms = 0.0;
    let mut p_scheduler_seen_to_kv_ready_ms = 0.0;
    let mut p_kv_ready_to_export_ready_ms = 0.0;
    let mut p_scheduler_seen_to_export_ready_ms = 0.0;
    let mut p_export_ready_to_http_response_ms = 0.0;
    let mut p_export_event_wait_ms = 0.0;
    let mut p_response_network_ms = 0.0;
    let mut router_to_early_d_dispatch_ms = 0.0;
    let mut d_request_seen_to_attach_start_ms = 0.0;
    let mut pd_request_seen_early = false;
    let mut pd_early_local_fallback = false;
    let mut pd_decode_dispatch_failed = false;
    let mut pd_import_completed_at: Option<Instant> = None;
    let mut pd_preparation_completed_at: Option<Instant> = None;
    let mut pd_request_diagnostics = json!({});

    info!(
        request_id = %request_id,
        prefill = %prefill_worker_id,
        decode = %decode_worker_id,
        stream = is_stream,
        route_ms = start.elapsed().as_secs_f64() * 1000.0,
        "Routed request (disagg)"
    );

    let mut pd_active = decision.pd_kvbm_enabled
        && !decision.pd_handoff_id.is_empty()
        && !decision.pd_request_id.is_empty()
        && !decision.pd_prefill_control_url.is_empty()
        && !decision.pd_decode_control_url.is_empty();
    if decision.pd_kvbm_enabled && !pd_active {
        warn!(
            request_id = %request_id,
            handoff_id_empty = decision.pd_handoff_id.is_empty(),
            engine_request_id_empty = decision.pd_request_id.is_empty(),
            prefill_control_url_empty = decision.pd_prefill_control_url.is_empty(),
            decode_control_url_empty = decision.pd_decode_control_url.is_empty(),
            "KVBM PD orchestration metadata is incomplete; using local decode recompute"
        );
    }
    // Router-managed HIXL uses its own P-stage semaphore. Native connectors
    // retain their original lifecycle and are governed by generic Router
    // admission only.
    let (permit, admission_wait_ms) = if decision.pd_kvbm_enabled {
        match acquire_pd_prefill_permit(state, &prefill_worker_id, decision.pd_export_timeout_ms)
            .await
        {
            Ok(result) => result,
            Err(error) => {
                notify_complete_background(
                    state.callbacks.clone(),
                    &request_id,
                    &prefill_worker_id,
                    start,
                    0,
                );
                return Err(ProxyError::Upstream(error));
            }
        }
    } else {
        (None, 0.0)
    };
    let mut pd_prefill_permit: Option<OwnedSemaphorePermit> = permit;
    let p_prefill_admission_wait_ms = admission_wait_ms;
    info!(
        request_id = %request_id,
        prefill = %prefill_worker_id,
        backend = if pd_active { "kvbm_hixl" } else { "native_connector" },
        admission_wait_ms = admission_wait_ms,
        max_inflight = if decision.pd_kvbm_enabled {
            state.pd_prefill_max_inflight
        } else {
            0
        },
        "PD P prefill admission acquired"
    );
    if pd_active {
        let transaction_ttl_ms =
            pd_transaction_ttl_ms(decision.pd_export_timeout_ms, decision.pd_timeout_ms);
        let source_begin_payload = json!({
            "handoff_id": &decision.pd_handoff_id,
            "request_id": &decision.pd_request_id,
            "source_worker_id": &prefill_worker_id,
            "target_worker_id": &decode_worker_id,
            "model_id": "0",
            "prompt_token_count": decision.pd_prompt_token_count,
            "local_role": "source",
            "source_capture_ttl_ms": transaction_ttl_ms,
        });
        let target_begin_payload = json!({
            "handoff_id": &decision.pd_handoff_id,
            "request_id": &decision.pd_request_id,
            "source_worker_id": &prefill_worker_id,
            "target_worker_id": &decode_worker_id,
            "model_id": "0",
            "prompt_token_count": decision.pd_prompt_token_count,
            "local_role": "target",
        });
        let prepare_payload = json!({
            "handoff_id": &decision.pd_handoff_id,
            "request_id": &decision.pd_request_id,
            "admission_ticket_id": &decision.pd_decode_admission_ticket_id,
            "ttl_ms": transaction_ttl_ms,
        });
        let source_begin_future = async {
            let started = Instant::now();
            let result = post_kvbm_control(
                &state.http_client,
                &decision.pd_prefill_control_url,
                "/datadist/pd/handoffs/begin",
                source_begin_payload,
                decision.pd_timeout_ms,
            )
            .await;
            (result, started.elapsed().as_secs_f64() * 1000.0)
        };
        let target_begin_future = async {
            let started = Instant::now();
            let result = post_kvbm_control(
                &state.http_client,
                &decision.pd_decode_control_url,
                "/datadist/pd/handoffs/begin",
                target_begin_payload,
                decision.pd_timeout_ms,
            )
            .await;
            (result, started.elapsed().as_secs_f64() * 1000.0)
        };
        let prepare_future = async {
            let started = Instant::now();
            let result = post_kvbm_control(
                &state.http_client,
                &decision.pd_decode_control_url,
                "/datadist/pd/import/prepare",
                prepare_payload,
                decision.pd_timeout_ms,
            )
            .await;
            (result, started.elapsed().as_secs_f64() * 1000.0)
        };
        let (
            (source_begin, source_begin_elapsed_ms),
            (target_begin, target_begin_elapsed_ms),
            (prepare, prepare_elapsed_ms),
        ) = tokio::join!(source_begin_future, target_begin_future, prepare_future);
        pd_preparation_completed_at = Some(Instant::now());
        pd_source_begin_ms = source_begin_elapsed_ms;
        pd_target_begin_ms = target_begin_elapsed_ms;
        pd_begin_ms = pd_source_begin_ms.max(pd_target_begin_ms);
        pd_prepare_ms = prepare_elapsed_ms;
        let begin_ready = matches!(&source_begin, Ok(value) if control_succeeded(value))
            && matches!(&target_begin, Ok(value) if control_succeeded(value));
        let gate_ready = matches!(&prepare, Ok(value) if control_succeeded(value));
        pd_active = begin_ready && gate_ready;
        if begin_ready {
            info!(
                request_id = %request_id,
                engine_request_id = %decision.pd_request_id,
                handoff_id = %decision.pd_handoff_id,
                phase_ms = pd_begin_ms,
                "KVBM PD handoff initialized"
            );
        } else {
            warn!(
                request_id = %request_id,
                source = ?source_begin,
                target = ?target_begin,
                "KVBM PD handoff initialization failed; using local decode recompute"
            );
        }
        if !gate_ready {
            warn!(
                request_id = %request_id,
                result = ?prepare,
                "KVBM PD early D gate prepare failed; using local decode recompute"
            );
        }
        if !pd_active {
            release_pd_handoff(
                &state.http_client,
                &decision,
                "router concurrent PD preparation failed",
            )
            .await;
        }
    }

    let mut pd_event_stream = None;
    let mut pd_event_stream_task = if pd_active {
        let client = state.http_client.clone();
        let decode_control_url = decision.pd_decode_control_url.clone();
        let handoff_id = decision.pd_handoff_id.clone();
        let engine_request_id = decision.pd_request_id.clone();
        Some(tokio::spawn(async move {
            let started = Instant::now();
            let result = PdControlEventStream::connect(
                &client,
                &decode_control_url,
                &handoff_id,
                &engine_request_id,
            )
            .await;
            (result, started.elapsed().as_secs_f64() * 1000.0)
        }))
    } else {
        None
    };

    let decode_url = format!("{}{}", decision.decode_target_url, path);
    let mut decode_builder = state
        .http_client
        .post(&decode_url)
        .header("Content-Type", "application/json");
    if let Some(auth) = original_headers.get("authorization") {
        if let Ok(v) = auth.to_str() {
            decode_builder = decode_builder.header("Authorization", v);
        }
    }
    for (k, v) in &decision.headers {
        decode_builder = decode_builder.header(k.as_str(), v.as_str());
    }
    decode_builder = decode_builder.body(decision.decode_body_json.clone());
    let mut decode_builder = Some(decode_builder);
    let mut worker_dispatch_started: Option<Instant> = None;
    let mut early_decode_task = if pd_active {
        worker_dispatch_started = Some(Instant::now());
        router_to_early_d_dispatch_ms = start.elapsed().as_secs_f64() * 1000.0;
        let builder = decode_builder
            .take()
            .expect("decode request builder must exist before early dispatch");
        Some(tokio::spawn(async move { builder.send().await }))
    } else {
        None
    };
    let mut early_decode_result: Option<Result<reqwest::Response, String>> = None;

    // Phase 1: prefill（max_tokens=1, stream=false），等待完成触发 KV 传输
    let prefill_url = format!("{}{}", decision.target_url, path);
    let mut prefill_builder = state
        .http_client
        .post(&prefill_url)
        .header("Content-Type", "application/json");

    if let Some(auth) = original_headers.get("authorization") {
        if let Ok(v) = auth.to_str() {
            prefill_builder = prefill_builder.header("Authorization", v);
        }
    }
    for (k, v) in &decision.headers {
        prefill_builder = prefill_builder.header(k.as_str(), v.as_str());
    }
    prefill_builder = prefill_builder.body(decision.body_json.clone());

    let p_request_started = Instant::now();
    if let Some(preparation_completed_at) = pd_preparation_completed_at {
        p_prepare_to_http_dispatch_ms = preparation_completed_at.elapsed().as_secs_f64() * 1000.0;
    }
    let (p_dispatch_tx, p_dispatch_rx) = oneshot::channel();
    let mut prefill_task = Some(tokio::spawn(async move {
        let _ = p_dispatch_tx.send(p_request_started.elapsed().as_secs_f64() * 1000.0);
        let response = prefill_builder
            .send()
            .await
            .map_err(|error| (None, format!("Prefill worker connection failed: {error}")))?;
        let status = response.status();
        let body = read_response_limited(response)
            .await
            .map_err(|error| (None, format!("failed to read P response: {error}")))?;
        if !status.is_success() {
            return Err((Some(status), String::from_utf8_lossy(&body).into_owned()));
        }
        Ok::<Bytes, (Option<StatusCode>, String)>(body)
    }));
    p_http_dispatch_ms = p_dispatch_rx.await.unwrap_or(0.0);

    // D请求进入Scheduler时已经完成delta preplan。先消费REQUEST_SEEN事件，
    // 容量不足时立即释放gate并回退本地计算，避免继续等待P export后再失败。
    if pd_active {
        if let Some(event_stream_task) = pd_event_stream_task.take() {
            match event_stream_task.await {
                Ok((Ok(stream), elapsed_ms)) => {
                    event_stream_connect_ms = elapsed_ms;
                    pd_event_stream = Some(stream);
                    info!(
                        request_id = %request_id,
                        handoff_id = %decision.pd_handoff_id,
                        event_stream_connect_ms = event_stream_connect_ms,
                        "KVBM PD Worker event stream connected before P export"
                    );
                }
                Ok((Err(error), elapsed_ms)) => {
                    event_stream_connect_ms = elapsed_ms;
                    warn!(
                        request_id = %request_id,
                        error = %error,
                        event_stream_connect_ms = event_stream_connect_ms,
                        "KVBM PD Worker event stream unavailable; checking D request by status"
                    );
                }
                Err(error) => warn!(
                    request_id = %request_id,
                    error = %error,
                    "KVBM PD Worker event stream task failed; checking D request by status"
                ),
            }
        }
        let request_ready_result = if let Some(mut decode_task) = early_decode_task.take() {
            let request_seen_future =
                wait_for_pd_request_ready(&mut pd_event_stream, &state.http_client, &decision);
            tokio::pin!(request_seen_future);
            tokio::select! {
                result = request_seen_future.as_mut() => {
                    early_decode_task = Some(decode_task);
                    result
                }
                result = &mut decode_task => {
                    match result {
                        Ok(Ok(response)) if response.status().is_success() => {
                            // 流式响应可能先返回2xx响应头，再进入engine-core Scheduler。
                            // 保存响应对象并继续等待REQUEST_SEEN，不能误判为请求完成。
                            early_decode_result = Some(Ok(response));
                            request_seen_future.as_mut().await
                        }
                        Ok(Ok(response)) => {
                            let status = response.status();
                            early_decode_result = Some(Ok(response));
                            pd_decode_dispatch_failed = true;
                            Err(format!(
                                "D worker returned HTTP {status} before the scheduler observed request"
                            ))
                        }
                        Ok(Err(error)) => {
                            let elapsed_ms = worker_dispatch_started
                                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                                .unwrap_or(0.0);
                            let message = format_worker_send_error(
                                "Decode",
                                &decode_url,
                                &error,
                                elapsed_ms,
                            );
                            early_decode_result = Some(Err(message.clone()));
                            pd_decode_dispatch_failed = true;
                            Err(message)
                        }
                        Err(error) => {
                            let message = format!("decode dispatch task failed: {error}");
                            early_decode_result = Some(Err(message.clone()));
                            pd_decode_dispatch_failed = true;
                            Err(message)
                        }
                    }
                }
            }
        } else {
            wait_for_pd_request_ready(&mut pd_event_stream, &state.http_client, &decision).await
        };
        match request_ready_result {
            Ok(status) => {
                if let Some(error) = pd_preplan_capacity_error(&status) {
                    pd_active = false;
                    pd_early_local_fallback = true;
                    warn!(
                        request_id = %request_id,
                        decode = %decode_worker_id,
                        error = %error,
                        "KVBM PD D capacity admission rejected; skipping P export and using local decode recompute"
                    );
                    release_pd_handoff(
                        &state.http_client,
                        &decision,
                        "D delta preplan capacity admission rejected",
                    )
                    .await;
                } else {
                    if let Some(metadata) = pd_target_allocation_metadata(&status) {
                        notify_pd_decode_budget_background(
                            state.callbacks.clone(),
                            &request_id,
                            &decode_worker_id,
                            "allocated",
                            &metadata,
                        );
                    }
                    pd_request_seen_early = true;
                }
            }
            Err(error) => {
                pd_active = false;
                pd_early_local_fallback = true;
                let decode_http_task_finished = early_decode_task
                    .as_ref()
                    .map(|task| task.is_finished())
                    .unwrap_or(false);
                if pd_decode_dispatch_failed {
                    warn!(
                        request_id = %request_id,
                        error = %error,
                        decode_http_task_finished = decode_http_task_finished,
                        decode_target_url = %decode_url,
                        "KVBM PD D request dispatch failed before Scheduler admission; aborting handoff"
                    );
                } else {
                    warn!(
                        request_id = %request_id,
                        error = %error,
                        decode_http_task_finished = decode_http_task_finished,
                        decode_target_url = %decode_url,
                        "KVBM PD D request admission check failed; using local decode recompute"
                    );
                }
                release_pd_handoff(
                    &state.http_client,
                    &decision,
                    "D request admission check failed",
                )
                .await;
            }
        }
    }

    if pd_early_local_fallback {
        if let Some(prefill_task) = prefill_task.take() {
            prefill_task.abort();
        }
    } else if !pd_active {
        let result = prefill_task
            .take()
            .expect("P request task must exist")
            .await
            .map_err(|error| ProxyError::Upstream(format!("P request task failed: {error}")))?;
        let prefill_response_body = match result {
            Ok(body) => body,
            Err((status, error)) => {
                notify_complete_background(
                    state.callbacks.clone(),
                    &request_id,
                    &prefill_worker_id,
                    start,
                    0,
                );
                if let Some(status) = status {
                    return Ok(Response::builder()
                        .status(status.as_u16())
                        .header("Content-Type", "application/json")
                        .body(Body::from(error))
                        .unwrap());
                }
                return Err(ProxyError::Upstream(error));
            }
        };
        let (decode_body_json, remote_block_count) =
            merge_native_kv_transfer_params(&decision.decode_body_json, &prefill_response_body)
                .map_err(|error| {
                    ProxyError::Upstream(format!(
                        "native PD transfer metadata unavailable: {error}"
                    ))
                })?;
        decision.decode_body_json = decode_body_json;
        let mut refreshed_decode_builder = state
            .http_client
            .post(&decode_url)
            .header("Content-Type", "application/json");
        if let Some(auth) = original_headers.get("authorization") {
            if let Ok(value) = auth.to_str() {
                refreshed_decode_builder = refreshed_decode_builder.header("Authorization", value);
            }
        }
        for (key, value) in &decision.headers {
            refreshed_decode_builder =
                refreshed_decode_builder.header(key.as_str(), value.as_str());
        }
        decode_builder = Some(refreshed_decode_builder.body(decision.decode_body_json.clone()));
        info!(
            request_id = %request_id,
            prefill = %prefill_worker_id,
            decode = %decode_worker_id,
            remote_block_count = remote_block_count,
            "Mooncake PD transfer metadata forwarded from P response to D request"
        );
        p_request_ms = p_request_started.elapsed().as_secs_f64() * 1000.0;
        p_completed_at = Some(Instant::now());
    }

    let mut pd_export: Option<Value> = None;
    let mut pd_remote_cluster_id: Option<i64> = None;
    if pd_active {
        let export_started = Instant::now();
        let export_payload = json!({
            "handoff_id": &decision.pd_handoff_id,
            "request_id": &decision.pd_request_id,
            "ttl_ms": pd_transaction_ttl_ms(
                decision.pd_export_timeout_ms,
                decision.pd_timeout_ms,
            ),
            "wait_timeout_ms": 0,
        });
        let (mut export, mut export_http_timing) = post_kvbm_control_timed(
            &state.http_client,
            &decision.pd_prefill_control_url,
            "/datadist/pd/export",
            export_payload.clone(),
            decision.pd_timeout_ms.saturating_add(1_000),
        )
        .await;
        if pd_export_is_pending(&export) {
            let event_wait_started = Instant::now();
            let mut prefill_task_consumed = false;
            let event_result = if let Some(task) = prefill_task.as_mut() {
                tokio::select! {
                    event = wait_for_pd_source_export_ready(&state.http_client, &decision) => event,
                    response = task => {
                        prefill_task_consumed = true;
                        match response {
                            Ok(Ok(_)) => {
                                // 正常P响应应晚于engine-core捕获事件；保留短暂补发窗口
                                // 覆盖事件流调度抖动，但不再盲等完整控制超时。
                                match tokio::time::timeout(
                                    Duration::from_millis(1_000),
                                    wait_for_pd_source_export_ready(
                                        &state.http_client,
                                        &decision,
                                    ),
                                )
                                .await
                                {
                                    Ok(result) => result,
                                    Err(_) => Err("P HTTP response completed successfully but no \
                                        P_REQUEST_SEEN or export event was observed".to_string()),
                                }
                            }
                            Ok(Err((status, error))) => Err(format!(
                                "P HTTP request failed before scheduler event: \
                                status={status:?} response={error}"
                            )),
                            Err(error) => Err(format!(
                                "P HTTP task failed before scheduler event: {error}"
                            )),
                        }
                    }
                }
            } else {
                wait_for_pd_source_export_ready(&state.http_client, &decision).await
            };
            if prefill_task_consumed {
                prefill_task.take();
            }
            match event_result {
                Ok(event) => {
                    p_export_event_wait_ms = event_wait_started.elapsed().as_secs_f64() * 1000.0;
                    info!(
                        request_id = %request_id,
                        handoff_id = %decision.pd_handoff_id,
                        event_wait_ms = p_export_event_wait_ms,
                        event_sequence = event["sequence"].as_i64().unwrap_or(0),
                        "KVBM PD P export became ready through Worker event stream"
                    );
                    let (retried, retry_timing) = post_kvbm_control_timed(
                        &state.http_client,
                        &decision.pd_prefill_control_url,
                        "/datadist/pd/export",
                        export_payload,
                        decision.pd_timeout_ms.saturating_add(1_000),
                    )
                    .await;
                    export_http_timing.send_to_headers_ms += retry_timing.send_to_headers_ms;
                    export_http_timing.headers_to_body_ms += retry_timing.headers_to_body_ms;
                    export_http_timing.json_decode_ms += retry_timing.json_decode_ms;
                    export_http_timing.total_ms += retry_timing.total_ms;
                    export = retried;
                }
                Err(error) => {
                    p_export_event_wait_ms = event_wait_started.elapsed().as_secs_f64() * 1000.0;
                    export = Err(format!(
                        "P export did not become ready within request deadline: {error}"
                    ));
                }
            }
        }
        p_request_ms = p_request_started.elapsed().as_secs_f64() * 1000.0;
        pd_export_ms = export_started.elapsed().as_secs_f64() * 1000.0;
        match export {
            Ok(value) if control_succeeded(&value) && value.get("export").is_some() => {
                let capture_wait_ms = p_export_event_wait_ms
                    + value["export"]["capture_wait_ms"].as_f64().unwrap_or(0.0);
                p_dispatch_to_scheduler_seen_ms = value["export"]
                    ["p_dispatch_to_scheduler_seen_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                p_worker_request_received_to_scheduler_seen_ms = value["export"]
                    ["p_worker_request_received_to_scheduler_seen_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                p_worker_request_received_timing_source = value["export"]
                    ["p_worker_request_received_timing_source"]
                    .as_str()
                    .unwrap_or("unavailable")
                    .to_string();
                let p_prepare_to_worker_receive_ms = (p_dispatch_to_scheduler_seen_ms
                    - p_worker_request_received_to_scheduler_seen_ms)
                    .max(0.0);
                let p_export_handler_request_read_ms = value["export"]["control_timing"]
                    ["handler_request_read_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                let p_export_handler_execution_ms = value["export"]["control_timing"]
                    ["handler_execution_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                let p_export_engine_core_proxy_ms = value["export"]["control_timing"]
                    ["engine_core_proxy_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                let p_export_engine_core_queue_ms = value["export"]["control_timing"]
                    ["engine_core_queue_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                let p_export_engine_core_execution_ms = value["export"]["control_timing"]
                    ["engine_core_execution_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                let p_export_response_build_ms = value["export"]["control_timing"]
                    ["response_build_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                p_scheduler_queue_cycles = value["export"]["p_scheduler_queue_cycles"]
                    .as_i64()
                    .unwrap_or(0);
                p_scheduler_seen_cycle_id = value["export"]["p_scheduler_seen_cycle_id"]
                    .as_i64()
                    .unwrap_or(0);
                p_first_scheduled_cycle_id = value["export"]["p_first_scheduled_cycle_id"]
                    .as_i64()
                    .unwrap_or(0);
                p_kv_ready_cycle_id = value["export"]["p_kv_ready_cycle_id"].as_i64().unwrap_or(0);
                p_scheduler_compute_cycles = value["export"]["p_scheduler_compute_cycles"]
                    .as_i64()
                    .unwrap_or(0);
                p_first_scheduled_token_count = value["export"]["p_first_scheduled_token_count"]
                    .as_i64()
                    .unwrap_or(0);
                p_scheduler_running_request_count = value["export"]
                    ["p_scheduler_running_request_count"]
                    .as_i64()
                    .unwrap_or(0);
                p_scheduler_waiting_request_count = value["export"]
                    ["p_scheduler_waiting_request_count"]
                    .as_i64()
                    .unwrap_or(0);
                p_first_scheduled_running_request_count = value["export"]
                    ["p_first_scheduled_running_request_count"]
                    .as_i64()
                    .unwrap_or(0);
                p_first_scheduled_waiting_request_count = value["export"]
                    ["p_first_scheduled_waiting_request_count"]
                    .as_i64()
                    .unwrap_or(0);
                p_first_scheduled_cycle_ms = value["export"]["p_first_scheduled_cycle_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                p_scheduler_seen_to_kv_ready_ms = value["export"]
                    ["p_scheduler_seen_to_kv_ready_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                p_kv_ready_to_export_ready_ms = value["export"]["p_kv_ready_to_export_ready_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                p_scheduler_seen_to_export_ready_ms = value["export"]
                    ["p_scheduler_seen_to_export_ready_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                p_export_ready_to_http_response_ms = value["export"]
                    ["p_export_ready_to_http_response_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                p_response_network_ms =
                    (pd_export_ms - capture_wait_ms - p_export_ready_to_http_response_ms).max(0.0);
                info!(
                    request_id = %request_id,
                    source_worker = %prefill_worker_id,
                    target_worker = %decode_worker_id,
                    pd_source_begin_ms = pd_source_begin_ms,
                    pd_target_begin_ms = pd_target_begin_ms,
                    pd_gate_prepare_ms = pd_prepare_ms,
                    p_prepare_to_worker_receive_ms = p_prepare_to_worker_receive_ms,
                    p_worker_receive_to_scheduler_ms = p_worker_request_received_to_scheduler_seen_ms,
                    export_send_to_headers_ms = export_http_timing.send_to_headers_ms,
                    export_headers_to_body_ms = export_http_timing.headers_to_body_ms,
                    export_json_decode_ms = export_http_timing.json_decode_ms,
                    export_engine_core_proxy_ms = p_export_engine_core_proxy_ms,
                    export_engine_core_queue_ms = p_export_engine_core_queue_ms,
                    export_engine_core_execution_ms = p_export_engine_core_execution_ms,
                    export_response_build_ms = p_export_response_build_ms,
                    export_handler_request_read_ms = p_export_handler_request_read_ms,
                    export_handler_execution_ms = p_export_handler_execution_ms,
                    export_http_total_ms = export_http_timing.total_ms,
                    export_event_wait_ms = p_export_event_wait_ms,
                    "[KVBMPDControlTiming] P control-path phases"
                );
                pd_export_ms = (pd_export_ms - capture_wait_ms).max(0.0);
                p_request_ms = (p_request_ms - pd_export_ms).max(0.0);
                let link_started = Instant::now();
                let link_result = ensure_pd_datadist_link(
                    &state.http_client,
                    &state.pd_link_cache,
                    &decision.pd_prefill_control_url,
                    &decision.pd_decode_control_url,
                    decision.pd_timeout_ms,
                )
                .await;
                pd_link_ms = link_started.elapsed().as_secs_f64() * 1000.0;
                match link_result {
                    Ok(resolution) => {
                        pd_link_cache_hit = resolution.cache_hit;
                        pd_source_status_ms = resolution.source_status_ms;
                        pd_target_status_ms = resolution.target_status_ms;
                        pd_link_call_ms = resolution.link_call_ms;
                        pd_export = value.get("export").cloned();
                        pd_remote_cluster_id = Some(resolution.cluster_id);
                        info!(
                            request_id = %request_id,
                            blocks = pd_export.as_ref()
                                .and_then(|item| item["source_blocks"].as_array())
                                .map_or(0, Vec::len),
                            link_cache_hit = pd_link_cache_hit,
                            source_status_ms = pd_source_status_ms,
                            target_status_ms = pd_target_status_ms,
                            link_call_ms = pd_link_call_ms,
                            pd_link_ms = pd_link_ms,
                            phase_ms = pd_export_ms + pd_link_ms + pd_prepare_ms,
                            "KVBM PD source exported and early D gate prepared"
                        );
                    }
                    Err(error) => {
                        pd_active = false;
                        warn!(request_id = %request_id, error = %error, "KVBM PD link failed");
                    }
                }
            }
            Ok(value) => {
                pd_active = false;
                let prefill_http_task_finished = prefill_task
                    .as_ref()
                    .map(|task| task.is_finished())
                    .unwrap_or(false);
                warn!(
                    request_id = %request_id,
                    result = %value,
                    prefill_http_task_finished = prefill_http_task_finished,
                    prefill_target_url = %prefill_url,
                    "KVBM PD P export failed"
                );
            }
            Err(error) => {
                pd_active = false;
                let prefill_http_task_finished = prefill_task
                    .as_ref()
                    .map(|task| task.is_finished())
                    .unwrap_or(false);
                warn!(
                    request_id = %request_id,
                    error = %error,
                    prefill_http_task_finished = prefill_http_task_finished,
                    prefill_target_url = %prefill_url,
                    "KVBM PD P export failed"
                );
            }
        }
        if !pd_active {
            release_pd_handoff(
                &state.http_client,
                &decision,
                "router PD preparation failed",
            )
            .await;
        }
    }

    // Both backends have finished their P stage here. Release the Router's P
    // inflight budget now so native-connector admission does not count requests
    // already decoding as undispatched P work. The completion callback keeps D
    // in request_workers for the final response cleanup.
    drop(pd_prefill_permit.take());
    notify_pd_prefill_complete_background(state.callbacks.clone(), &request_id, &prefill_worker_id);
    if let Some(prefill_task) = prefill_task.take() {
        if !pd_active {
            prefill_task.abort();
            info!(
                request_id = %request_id,
                prefill = %prefill_worker_id,
                "KVBM PD cancelled P request after export preparation failed"
            );
        } else {
            let background_request_id = request_id.clone();
            let background_prefill_worker_id = prefill_worker_id.clone();
            tokio::spawn(async move {
                match prefill_task.await {
                    Ok(Ok(_)) => info!(
                        request_id = %background_request_id,
                        prefill = %background_prefill_worker_id,
                        "KVBM PD P response completed after engine-core export-ready"
                    ),
                    Ok(Err((status, error))) => warn!(
                        request_id = %background_request_id,
                        prefill = %background_prefill_worker_id,
                        status = ?status,
                        error = %error,
                        "KVBM PD P response failed after export-ready"
                    ),
                    Err(error) => warn!(
                        request_id = %background_request_id,
                        prefill = %background_prefill_worker_id,
                        error = %error,
                        "KVBM PD P response task failed after export-ready"
                    ),
                }
            });
        }
    }

    // Phase 2: decode（原始 max_tokens / stream），响应返回客户端。PD 激活时
    // 请求已经在 P 请求之前发出，此处只等待 gate 并执行 KV import。
    let pre_decode_total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let decode_resp_result: Result<reqwest::Response, String> = if pd_active {
        let mut decode_task = early_decode_task.take();
        let gate_wait_started = Instant::now();
        let request_seen = if pd_request_seen_early {
            Ok(json!({"request_seen": true, "source": "early_admission"}))
        } else {
            let request_seen_future =
                wait_for_pd_request_ready(&mut pd_event_stream, &state.http_client, &decision);
            tokio::pin!(request_seen_future);
            if let Some(decode_task) = decode_task.as_mut() {
                tokio::select! {
                    result = request_seen_future.as_mut() => result,
                    result = decode_task => {
                    match result {
                        Ok(Ok(response)) if response.status().is_success() => {
                            // StreamingResponse 会先返回 2xx 响应头，随后才开始迭代
                            // vLLM generator 并进入 scheduler。保留响应对象，继续等
                            // request gate，不能把响应头误判为请求已完成。
                            early_decode_result = Some(Ok(response));
                            request_seen_future.as_mut().await
                        }
                        Ok(Ok(response)) => {
                            let status = response.status();
                            early_decode_result = Some(Ok(response));
                            Err(format!(
                                "D worker returned HTTP {status} before the scheduler gate observed the request"
                            ))
                        }
                        Ok(Err(error)) => {
                            let elapsed_ms = worker_dispatch_started
                                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                                .unwrap_or(0.0);
                            let message = format_worker_send_error(
                                "Decode",
                                &decode_url,
                                &error,
                                elapsed_ms,
                            );
                            early_decode_result = Some(Err(message.clone()));
                            Err(message)
                        }
                        Err(error) => {
                            let message = format!("decode dispatch task failed: {error}");
                            early_decode_result = Some(Err(message.clone()));
                            Err(message)
                        }
                    }
                }
                }
            } else {
                request_seen_future.as_mut().await
            }
        };
        let d_gate_wait_ms = gate_wait_started.elapsed().as_secs_f64() * 1000.0;
        let import_result = match request_seen {
            Ok(_) => {
                let import_started = Instant::now();
                let result = submit_and_wait_pd_import(
                    &mut pd_event_stream,
                    &state.http_client,
                    &decision,
                    json!({
                        "handoff_id": &decision.pd_handoff_id,
                        "request_id": &decision.pd_request_id,
                        "source_control_url": &decision.pd_prefill_control_url,
                        "remote_cluster_id": pd_remote_cluster_id,
                        "source_export": pd_export.clone(),
                        "attach_timeout_ms": decision.pd_timeout_ms,
                        // Router已在派发前按真实BlockPool容量准入；这里仅覆盖
                        // 指标发布到实际分配之间的竞态，不能再次等待完整请求周期。
                        "capacity_wait_timeout_ms": decision.pd_timeout_ms.clamp(500, 5_000),
                    }),
                    state.callbacks.clone(),
                    &request_id,
                    &decode_worker_id,
                )
                .await;
                pd_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                if let Ok(value) = &result {
                    if control_succeeded(value) {
                        pd_import_completed_at = Some(Instant::now());
                        notify_pd_decode_budget_background(
                            state.callbacks.clone(),
                            &request_id,
                            &decode_worker_id,
                            "attached",
                            &pd_router_cache_publish_metadata(
                                &prefill_worker_id,
                                &decode_worker_id,
                                pd_export.as_ref(),
                                value,
                            ),
                        );
                        d_request_seen_to_attach_start_ms = value["scheduler_timing"]
                            ["d_request_seen_to_attach_start_ms"]
                            .as_f64()
                            .unwrap_or(0.0);
                        pd_request_diagnostics = json!({
                            "p_request_ms": p_request_ms,
                            "p_http_dispatch_ms": p_http_dispatch_ms,
                            "p_prepare_to_http_dispatch_ms": p_prepare_to_http_dispatch_ms,
                            "p_dispatch_to_scheduler_seen_ms": p_dispatch_to_scheduler_seen_ms,
                            "p_worker_request_received_to_scheduler_seen_ms": p_worker_request_received_to_scheduler_seen_ms,
                            "p_worker_request_received_timing_source": p_worker_request_received_timing_source.clone(),
                            "p_scheduler_queue_cycles": p_scheduler_queue_cycles,
                            "p_scheduler_seen_cycle_id": p_scheduler_seen_cycle_id,
                            "p_first_scheduled_cycle_id": p_first_scheduled_cycle_id,
                            "p_kv_ready_cycle_id": p_kv_ready_cycle_id,
                            "p_scheduler_compute_cycles": p_scheduler_compute_cycles,
                            "p_first_scheduled_token_count": p_first_scheduled_token_count,
                            "p_scheduler_running_request_count": p_scheduler_running_request_count,
                            "p_scheduler_waiting_request_count": p_scheduler_waiting_request_count,
                            "p_first_scheduled_running_request_count": p_first_scheduled_running_request_count,
                            "p_first_scheduled_waiting_request_count": p_first_scheduled_waiting_request_count,
                            "p_first_scheduled_cycle_ms": p_first_scheduled_cycle_ms,
                            "p_scheduler_seen_to_kv_ready_ms": p_scheduler_seen_to_kv_ready_ms,
                            "p_kv_ready_to_export_ready_ms": p_kv_ready_to_export_ready_ms,
                            "p_export_ready_to_http_response_ms": p_export_ready_to_http_response_ms,
                            "p_export_event_wait_ms": p_export_event_wait_ms,
                            "p_response_network_ms": p_response_network_ms,
                            "delta_plan_ms": value["phase_ms"]["delta_plan"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "delta_preplanned": value["phase_ms"]["delta_preplanned"]
                                .as_f64()
                                .unwrap_or(0.0) > 0.0,
                            "delta_plan_task_queue_ms": value["phase_ms"]
                                ["delta_plan_task_queue_ms"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "delta_hash_validate_ms": value["phase_ms"]
                                ["delta_hash_validate_ms"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "delta_reserve_ms": value["phase_ms"]["delta_reserve_ms"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "delta_plan_execution_ms": value["phase_ms"]
                                ["delta_plan_execution_ms"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "delta_preplan_age_ms": value["phase_ms"]
                                ["delta_preplan_age_ms"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "delta_inflight_wait_ms": value["phase_ms"]
                                ["delta_inflight_wait_ms"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "delta_inflight_retry_count": value["phase_ms"]
                                ["delta_inflight_retry_count"]
                                .as_f64()
                                .unwrap_or(0.0) as u64,
                            "delta_capacity_wait_ms": value["phase_ms"]
                                ["delta_capacity_wait_ms"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "delta_capacity_retry_count": value["phase_ms"]
                                ["delta_capacity_retry_count"]
                                .as_f64()
                                .unwrap_or(0.0) as u64,
                            "import_executor_queue_ms": value["phase_ms"]
                                ["import_executor_queue_ms"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "reserve_ms": value["phase_ms"]["reserve"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "hixl_pull_ms": value["phase_ms"]["hixl_pull"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "attach_task_queue_ms": value["phase_ms"]
                                ["attach_task_queue_ms"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "attach_execution_ms": value["phase_ms"]
                                ["attach_execution_ms"]
                                .as_f64()
                                .unwrap_or(0.0),
                            "free_blocks": value["free_blocks"].as_i64(),
                            "transferred_blocks": value["transferred_block_count"]
                                .as_i64()
                                .unwrap_or(0),
                            "reused_blocks": value["reused_block_count"]
                                .as_i64()
                                .unwrap_or(0),
                        });
                        pd_request_diagnostics["p_prefill_admission_wait_ms"] =
                            json!(p_prefill_admission_wait_ms);
                    }
                }
                match &result {
                    Ok(value) if control_succeeded(value) => info!(
                        request_id = %request_id,
                        source_blocks = value["source_blocks"].as_array().map_or(0, Vec::len),
                        destination_blocks = value["destination_blocks"].as_array().map_or(0, Vec::len),
                        phase_ms = import_started.elapsed().as_secs_f64() * 1000.0,
                        worker_phase_ms = %value["phase_ms"],
                        "KVBM PD import attached to D request"
                    ),
                    _ => {
                        warn!(request_id = %request_id, result = ?result, "KVBM PD import failed; D request will recompute locally")
                    }
                }
                result
            }
            Err(error) => {
                warn!(request_id = %request_id, error = %error, "KVBM PD request gate wait failed");
                Err(error)
            }
        };
        if pd_import_reports_not_linked(&import_result) {
            let cache_key = (
                decision
                    .pd_prefill_control_url
                    .trim_end_matches('/')
                    .to_string(),
                decision
                    .pd_decode_control_url
                    .trim_end_matches('/')
                    .to_string(),
            );
            state
                .pd_link_cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&cache_key);
            warn!(
                request_id = %request_id,
                prefill_control_url = %decision.pd_prefill_control_url,
                decode_control_url = %decision.pd_decode_control_url,
                "KVBM PD cached LLM-DataDist link invalidated after LLM_NOT_YET_LINK"
            );
            let http_client = state.http_client.clone();
            let link_cache = state.pd_link_cache.clone();
            let prefill_control_url = decision.pd_prefill_control_url.clone();
            let decode_control_url = decision.pd_decode_control_url.clone();
            let timeout_ms = decision.pd_timeout_ms;
            tokio::spawn(async move {
                match ensure_pd_datadist_link(
                    &http_client,
                    &link_cache,
                    &prefill_control_url,
                    &decode_control_url,
                    timeout_ms,
                )
                .await
                {
                    Ok(resolution) => info!(
                        cluster_id = resolution.cluster_id,
                        source_status_ms = resolution.source_status_ms,
                        target_status_ms = resolution.target_status_ms,
                        link_call_ms = resolution.link_call_ms,
                        "KVBM PD LLM-DataDist link rebuilt after cache invalidation"
                    ),
                    Err(error) => warn!(
                        error = %error,
                        "KVBM PD LLM-DataDist background relink failed"
                    ),
                }
            });
        }
        if !matches!(&import_result, Ok(value) if control_succeeded(value)) {
            release_pd_handoff(&state.http_client, &decision, "router PD import failed").await;
            notify_pd_decode_budget_background(
                state.callbacks.clone(),
                &request_id,
                &decode_worker_id,
                "fallback",
                &json!({"reason": "pd_import_failed"}),
            );
        }
        info!(
            request_id = %request_id,
            prefill = %prefill_worker_id,
            decode = %decode_worker_id,
            route_ms = route_ms,
            pd_begin_ms = pd_begin_ms,
            p_request_ms = p_request_ms,
            pd_export_ms = pd_export_ms,
            pd_link_ms = pd_link_ms,
            pd_link_cache_hit = pd_link_cache_hit,
            pd_source_status_ms = pd_source_status_ms,
            pd_target_status_ms = pd_target_status_ms,
            pd_link_call_ms = pd_link_call_ms,
            pd_prepare_ms = pd_prepare_ms,
            d_gate_wait_ms = d_gate_wait_ms,
            pd_import_ms = pd_import_ms,
            event_stream_connect_ms = event_stream_connect_ms,
            p_http_dispatch_ms = p_http_dispatch_ms,
            p_prepare_to_http_dispatch_ms = p_prepare_to_http_dispatch_ms,
            p_dispatch_to_scheduler_seen_ms = p_dispatch_to_scheduler_seen_ms,
            p_worker_request_received_to_scheduler_seen_ms = p_worker_request_received_to_scheduler_seen_ms,
            p_worker_request_received_timing_source = %p_worker_request_received_timing_source,
            p_scheduler_queue_cycles = p_scheduler_queue_cycles,
            p_scheduler_seen_cycle_id = p_scheduler_seen_cycle_id,
            p_first_scheduled_cycle_id = p_first_scheduled_cycle_id,
            p_kv_ready_cycle_id = p_kv_ready_cycle_id,
            p_scheduler_compute_cycles = p_scheduler_compute_cycles,
            p_first_scheduled_token_count = p_first_scheduled_token_count,
            p_scheduler_running_request_count = p_scheduler_running_request_count,
            p_scheduler_waiting_request_count = p_scheduler_waiting_request_count,
            p_first_scheduled_running_request_count = p_first_scheduled_running_request_count,
            p_first_scheduled_waiting_request_count = p_first_scheduled_waiting_request_count,
            p_first_scheduled_cycle_ms = p_first_scheduled_cycle_ms,
            p_scheduler_seen_to_kv_ready_ms = p_scheduler_seen_to_kv_ready_ms,
            p_kv_ready_to_export_ready_ms = p_kv_ready_to_export_ready_ms,
            p_scheduler_seen_to_export_ready_ms = p_scheduler_seen_to_export_ready_ms,
            p_export_ready_to_http_response_ms = p_export_ready_to_http_response_ms,
            p_export_event_wait_ms = p_export_event_wait_ms,
            p_response_network_ms = p_response_network_ms,
            p_prefill_admission_wait_ms = p_prefill_admission_wait_ms,
            router_to_early_d_dispatch_ms = router_to_early_d_dispatch_ms,
            d_request_seen_to_attach_start_ms = d_request_seen_to_attach_start_ms,
            pre_decode_total_ms = pre_decode_total_ms,
            "[KVBMPDTiming] PD orchestration phases"
        );
        match early_decode_result {
            Some(result) => result,
            None => match decode_task {
                Some(decode_task) => match decode_task.await {
                    Ok(result) => result.map_err(|error| {
                        let elapsed_ms = worker_dispatch_started
                            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                            .unwrap_or(0.0);
                        format_worker_send_error("Decode", &decode_url, &error, elapsed_ms)
                    }),
                    Err(error) => Err(format!("decode dispatch task failed: {error}")),
                },
                None => Err("decode dispatch result is unavailable".to_string()),
            },
        }
    } else {
        notify_pd_decode_budget_background(
            state.callbacks.clone(),
            &request_id,
            &decode_worker_id,
            "fallback",
            &json!({"reason": "pd_handoff_inactive"}),
        );
        if decision.pd_kvbm_enabled {
            info!(
                request_id = %request_id,
                prefill = %prefill_worker_id,
                decode = %decode_worker_id,
                fallback_mode = if pd_decode_dispatch_failed {
                    "dispatch_failed"
                } else {
                    "local_recompute"
                },
                route_ms = route_ms,
                pd_begin_ms = pd_begin_ms,
                p_request_ms = p_request_ms,
                pd_export_ms = pd_export_ms,
                pd_link_ms = pd_link_ms,
                pd_link_cache_hit = pd_link_cache_hit,
                pd_source_status_ms = pd_source_status_ms,
                pd_target_status_ms = pd_target_status_ms,
                pd_link_call_ms = pd_link_call_ms,
                pd_prepare_ms = pd_prepare_ms,
                d_gate_wait_ms = 0.0,
                pd_import_ms = 0.0,
                event_stream_connect_ms = event_stream_connect_ms,
                p_http_dispatch_ms = p_http_dispatch_ms,
                p_prepare_to_http_dispatch_ms = p_prepare_to_http_dispatch_ms,
                p_dispatch_to_scheduler_seen_ms = p_dispatch_to_scheduler_seen_ms,
                p_worker_request_received_to_scheduler_seen_ms = p_worker_request_received_to_scheduler_seen_ms,
                p_worker_request_received_timing_source = %p_worker_request_received_timing_source,
                p_scheduler_queue_cycles = p_scheduler_queue_cycles,
                p_scheduler_seen_cycle_id = p_scheduler_seen_cycle_id,
                p_first_scheduled_cycle_id = p_first_scheduled_cycle_id,
                p_kv_ready_cycle_id = p_kv_ready_cycle_id,
                p_scheduler_compute_cycles = p_scheduler_compute_cycles,
                p_first_scheduled_token_count = p_first_scheduled_token_count,
                p_scheduler_running_request_count = p_scheduler_running_request_count,
                p_scheduler_waiting_request_count = p_scheduler_waiting_request_count,
                p_first_scheduled_running_request_count = p_first_scheduled_running_request_count,
                p_first_scheduled_waiting_request_count = p_first_scheduled_waiting_request_count,
                p_first_scheduled_cycle_ms = p_first_scheduled_cycle_ms,
                p_scheduler_seen_to_kv_ready_ms = p_scheduler_seen_to_kv_ready_ms,
                p_kv_ready_to_export_ready_ms = p_kv_ready_to_export_ready_ms,
                p_scheduler_seen_to_export_ready_ms = p_scheduler_seen_to_export_ready_ms,
                p_export_ready_to_http_response_ms = p_export_ready_to_http_response_ms,
                p_export_event_wait_ms = p_export_event_wait_ms,
                p_response_network_ms = p_response_network_ms,
                p_prefill_admission_wait_ms = p_prefill_admission_wait_ms,
                router_to_early_d_dispatch_ms = router_to_early_d_dispatch_ms,
                d_request_seen_to_attach_start_ms = 0.0,
                pre_decode_total_ms = pre_decode_total_ms,
                "[KVBMPDTiming] PD handoff inactive; resolving D fallback"
            );
        } else {
            info!(
                request_id = %request_id,
                prefill = %prefill_worker_id,
                decode = %decode_worker_id,
                route_ms = route_ms,
                p_request_ms = p_request_ms,
                pre_decode_total_ms = pre_decode_total_ms,
                "[MooncakePDTiming] native connector P completed; dispatching D request"
            );
        }
        if let Some(result) = early_decode_result {
            result
        } else if let Some(decode_task) = early_decode_task.take() {
            match decode_task.await {
                Ok(result) => result.map_err(|error| {
                    let elapsed_ms = worker_dispatch_started
                        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);
                    format_worker_send_error("Decode", &decode_url, &error, elapsed_ms)
                }),
                Err(error) => Err(format!("decode dispatch task failed: {error}")),
            }
        } else {
            let snapshot_requested_at = Instant::now();
            let dispatch_snapshot_task = if !decision.pd_kvbm_enabled {
                let callbacks = state.callbacks.clone();
                let snapshot_request_id = request_id.clone();
                let snapshot_worker_id = decode_worker_id.clone();
                Some(tokio::spawn(async move {
                    query_pd_dispatch_snapshot(
                        callbacks,
                        &snapshot_request_id,
                        &snapshot_worker_id,
                        snapshot_requested_at,
                    )
                    .await
                }))
            } else {
                None
            };
            let dispatch_started = Instant::now();
            let p_complete_to_d_dispatch_ms = p_completed_at
                .map(|completed| dispatch_started.duration_since(completed).as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            worker_dispatch_started = Some(dispatch_started);
            let result = decode_builder
                .take()
                .expect("decode request builder must exist before dispatch")
                .send()
                .await
                .map_err(|error| error.to_string());
            if let Some(snapshot_task) = dispatch_snapshot_task {
                let log_request_id = request_id.clone();
                let log_engine_request_id = decision.pd_request_id.clone();
                let log_prefill_worker_id = prefill_worker_id.clone();
                let log_decode_worker_id = decode_worker_id.clone();
                let d_dispatch_to_headers_ms = dispatch_started.elapsed().as_secs_f64() * 1000.0;
                let dispatch_ok = result.is_ok();
                tokio::spawn(async move {
                    let (dispatch_snapshot, snapshot_callback_ms) = match snapshot_task.await {
                        Ok(snapshot) => snapshot,
                        Err(error) => {
                            warn!(
                                request_id = %log_request_id,
                                worker_id = %log_decode_worker_id,
                                error = %error,
                                "Mooncake PD dispatch snapshot join failed"
                            );
                            (json!({}), 0.0)
                        }
                    };
                    let selection_to_snapshot_ms = dispatch_snapshot["selection_to_dispatch_ms"]
                        .as_f64()
                        .unwrap_or(0.0);
                    info!(
                        request_id = %log_request_id,
                        engine_request_id = %log_engine_request_id,
                        prefill = %log_prefill_worker_id,
                        decode = %log_decode_worker_id,
                        p_request_ms = p_request_ms,
                        d_selection_to_dispatch_ms =
                            (selection_to_snapshot_ms - snapshot_callback_ms).max(0.0),
                        p_complete_to_d_dispatch_ms = p_complete_to_d_dispatch_ms,
                        d_selection_inflight = dispatch_snapshot["selection_inflight"]
                            .as_i64()
                            .unwrap_or(0),
                        d_selection_running = dispatch_snapshot["selection_running"]
                            .as_i64()
                            .unwrap_or(0),
                        d_selection_waiting = dispatch_snapshot["selection_waiting"]
                            .as_i64()
                            .unwrap_or(0),
                        d_dispatch_inflight = dispatch_snapshot["dispatch_inflight"]
                            .as_i64()
                            .unwrap_or(0),
                        d_dispatch_running = dispatch_snapshot["dispatch_running"]
                            .as_i64()
                            .unwrap_or(0),
                        d_dispatch_waiting = dispatch_snapshot["dispatch_waiting"]
                            .as_i64()
                            .unwrap_or(0),
                        snapshot_callback_ms = snapshot_callback_ms,
                        snapshot_stale = snapshot_callback_ms > 50.0,
                        d_dispatch_to_headers_ms = d_dispatch_to_headers_ms,
                        dispatch_ok = dispatch_ok,
                        "[MooncakePDDispatchTiming] Native connector D dispatch phases"
                    );
                });
            }
            result
        }
    };
    let decode_resp = match decode_resp_result {
        Ok(resp) => resp,
        Err(e) => {
            notify_complete_background(
                state.callbacks.clone(),
                &request_id,
                &decode_worker_id,
                start,
                0,
            );
            return Err(ProxyError::Upstream(format!(
                "Decode worker connection failed: {e}"
            )));
        }
    };

    let decode_status = decode_resp.status();
    if !decode_status.is_success() {
        let err_body = read_text_limited(decode_resp)
            .await
            .unwrap_or_else(|_| "unknown error".into());
        warn!(
            request_id = %request_id,
            worker = %decode_worker_id,
            status = %decode_status,
            "Decode upstream error"
        );
        notify_complete_background(
            state.callbacks.clone(),
            &request_id,
            &decode_worker_id,
            start,
            0,
        );
        return Ok(Response::builder()
            .status(decode_status.as_u16())
            .header("Content-Type", "application/json")
            .body(Body::from(err_body))
            .unwrap());
    }

    if is_stream {
        build_streaming_response(
            state,
            decode_resp,
            request_id,
            decode_worker_id,
            start,
            worker_dispatch_started.unwrap_or(start),
            router_to_early_d_dispatch_ms,
            d_request_seen_to_attach_start_ms,
            pd_import_completed_at,
            pd_request_diagnostics,
        )
    } else {
        build_buffered_response(state, decode_resp, request_id, decode_worker_id, start).await
    }
}

/// 构建流式响应：通过 mpsc channel 逐块转发 worker 的 SSE 流，记录 TTFT。
fn build_streaming_response(
    state: &AppState,
    upstream_resp: reqwest::Response,
    request_id: String,
    worker_id: String,
    start: Instant,
    worker_dispatch_started: Instant,
    router_to_early_d_dispatch_ms: f64,
    d_request_seen_to_attach_start_ms: f64,
    pd_import_completed_at: Option<Instant>,
    pd_request_diagnostics: Value,
) -> Result<Response<Body>, ProxyError> {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let cb = state.callbacks.clone();

    tokio::spawn(async move {
        let mut byte_stream = upstream_resp.bytes_stream();
        let mut first_chunk = true;
        let mut ttft_s: f64 = 0.0;
        let mut worker_dispatch_to_first_byte_s: f64 = 0.0;
        let mut d_attach_complete_to_first_byte_s: f64 = 0.0;
        let mut last_data_line = String::new();

        while let Some(chunk) = byte_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if first_chunk {
                        ttft_s = start.elapsed().as_secs_f64();
                        worker_dispatch_to_first_byte_s =
                            worker_dispatch_started.elapsed().as_secs_f64();
                        d_attach_complete_to_first_byte_s = pd_import_completed_at
                            .map(|completed_at| completed_at.elapsed().as_secs_f64())
                            .unwrap_or(0.0);
                        first_chunk = false;
                        notify_pd_decode_budget_background(
                            cb.clone(),
                            &request_id,
                            &worker_id,
                            "first_byte",
                            &json!({}),
                        );
                        info!(
                            request_id = %request_id,
                            ttft_ms = ttft_s * 1000.0,
                            router_to_early_d_dispatch_ms = router_to_early_d_dispatch_ms,
                            d_request_seen_to_attach_start_ms = d_request_seen_to_attach_start_ms,
                            d_attach_complete_to_first_byte_ms =
                                d_attach_complete_to_first_byte_s * 1000.0,
                            "First byte received"
                        );
                    }
                    let text = String::from_utf8_lossy(&bytes);
                    for line in text.lines() {
                        let trimmed = line.trim();
                        if trimmed.starts_with("data: {") {
                            last_data_line = trimmed[6..].to_string();
                        }
                    }
                    if tx.send(Ok(bytes)).await.is_err() {
                        warn!(request_id = %request_id, "Client disconnected during stream");
                        break;
                    }
                }
                Err(e) => {
                    error!(request_id = %request_id, "Upstream stream error: {e}");
                    let _ = tx
                        .send(Err(std::io::Error::new(std::io::ErrorKind::Other, e)))
                        .await;
                    break;
                }
            }
        }

        let usage_val = serde_json::from_str::<serde_json::Value>(&last_data_line)
            .ok()
            .and_then(|v| v.get("usage").cloned());
        let completion_tokens: i64 = usage_val
            .as_ref()
            .and_then(|u| u.get("completion_tokens")?.as_i64())
            .unwrap_or(0);
        let prompt_tokens: i64 = usage_val
            .as_ref()
            .and_then(|u| u.get("prompt_tokens")?.as_i64())
            .unwrap_or(0);

        let total_s = start.elapsed().as_secs_f64();
        info!(
            request_id = %request_id,
            worker = %worker_id,
            total_ms = total_s * 1000.0,
            prompt_tokens = prompt_tokens,
            completion_tokens = completion_tokens,
            "Stream completed"
        );

        let req_id = request_id.clone();
        let w_id = worker_id.clone();
        let _ = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                if let Err(e) = cb.complete_fn.call1(
                    py,
                    (
                        &req_id,
                        &w_id,
                        ttft_s,
                        total_s,
                        completion_tokens,
                        prompt_tokens,
                        worker_dispatch_to_first_byte_s,
                        router_to_early_d_dispatch_ms / 1000.0,
                        d_request_seen_to_attach_start_ms / 1000.0,
                        d_attach_complete_to_first_byte_s,
                        pd_request_diagnostics.to_string(),
                    ),
                ) {
                    warn!("complete_fn failed: {e}");
                }
            });
        })
        .await;
    });

    let body_stream = ReceiverStream::new(rx);
    let body = Body::from_stream(body_stream);

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(body)
        .map_err(|e| ProxyError::Internal(e.to_string()))
}

/// 构建缓冲响应：等待 worker 完整响应后一次性返回给客户端。
///
/// 从响应 JSON 的 `usage.completion_tokens` 中提取 token 数传给 complete_fn。
async fn build_buffered_response(
    state: &AppState,
    upstream_resp: reqwest::Response,
    request_id: String,
    worker_id: String,
    start: Instant,
) -> Result<Response<Body>, ProxyError> {
    let body_bytes = match read_response_limited(upstream_resp).await {
        Ok(bytes) => bytes,
        Err(e) => {
            notify_complete_background(state.callbacks.clone(), &request_id, &worker_id, start, 0);
            return Err(ProxyError::Upstream(format!(
                "Failed to read upstream body: {e}"
            )));
        }
    };

    let usage_val = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|v| v.get("usage").cloned());
    let completion_tokens: i64 = usage_val
        .as_ref()
        .and_then(|u| u.get("completion_tokens")?.as_i64())
        .unwrap_or(0);
    let prompt_tokens: i64 = usage_val
        .as_ref()
        .and_then(|u| u.get("prompt_tokens")?.as_i64())
        .unwrap_or(0);

    let total_s = start.elapsed().as_secs_f64();
    info!(
        request_id = %request_id,
        worker = %worker_id,
        total_ms = total_s * 1000.0,
        prompt_tokens = prompt_tokens,
        completion_tokens = completion_tokens,
        "Buffered response completed"
    );

    // 非流式：ttft 设为 0，让 Python 侧用 total_s / completion_tokens 近似 TPOT
    let cb = state.callbacks.clone();
    let req_id = request_id.clone();
    let w_id = worker_id.clone();
    tokio::spawn(async move {
        let _ = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let ttft_s: f64 = 0.0;
                if let Err(e) = cb.complete_fn.call1(
                    py,
                    (
                        &req_id,
                        &w_id,
                        ttft_s,
                        total_s,
                        completion_tokens,
                        prompt_tokens,
                    ),
                ) {
                    warn!("complete_fn failed: {e}");
                }
            });
        })
        .await;
    });

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(body_bytes))
        .unwrap())
}

/// 异步通知 Python 控制面请求已完成（错误/超时路径），释放 worker 的 inflight 计数。
fn notify_complete_background(
    cb: Arc<PyCallbacks>,
    request_id: &str,
    worker_id: &str,
    start: Instant,
    completion_tokens: i64,
) {
    let req_id = request_id.to_string();
    let w_id = worker_id.to_string();
    tokio::spawn(async move {
        let total_s = start.elapsed().as_secs_f64();
        let prompt_tokens: i64 = 0;
        let _ = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                if let Err(e) = cb.complete_fn.call1(
                    py,
                    (
                        &req_id,
                        &w_id,
                        total_s,
                        total_s,
                        completion_tokens,
                        prompt_tokens,
                    ),
                ) {
                    warn!("complete_fn failed: {e}");
                }
            });
        })
        .await;
    });
}

/// 异步通知 Python Router 的 P 阶段已经结束，提前释放动态 Prefill 预算。
fn notify_pd_prefill_complete_background(cb: Arc<PyCallbacks>, request_id: &str, worker_id: &str) {
    if cb.pd_prefill_complete_fn.is_none() {
        return;
    }
    let req_id = request_id.to_string();
    let worker_id = worker_id.to_string();
    tokio::spawn(async move {
        let _ = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                if let Some(callback) = cb.pd_prefill_complete_fn.as_ref() {
                    if let Err(error) = callback.call1(py, (&req_id, &worker_id)) {
                        warn!("pd_prefill_complete_fn failed: {error}");
                    }
                }
            });
        })
        .await;
    });
}

/// 在原生Connector真正派发D请求前读取Router最新负载，仅用于诊断。
async fn query_pd_dispatch_snapshot(
    cb: Arc<PyCallbacks>,
    request_id: &str,
    worker_id: &str,
    requested_at: Instant,
) -> (Value, f64) {
    if cb.pd_dispatch_snapshot_fn.is_none() {
        return (json!({}), 0.0);
    }
    let request_id_for_log = request_id.to_string();
    let worker_id_for_log = worker_id.to_string();
    let request_id = request_id_for_log.clone();
    let worker_id = worker_id_for_log.clone();
    let result = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            let Some(callback) = cb.pd_dispatch_snapshot_fn.as_ref() else {
                return Ok(json!({}));
            };
            let value = callback
                .call1(py, (&request_id, &worker_id))
                .map_err(|error| error.to_string())?;
            let payload = value
                .extract::<String>(py)
                .map_err(|error| error.to_string())?;
            serde_json::from_str::<Value>(&payload).map_err(|error| error.to_string())
        })
    })
    .await;
    let elapsed_ms = requested_at.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(Ok(snapshot)) => (snapshot, elapsed_ms),
        Ok(Err(error)) => {
            warn!(
                request_id = %request_id_for_log,
                worker_id = %worker_id_for_log,
                error = %error,
                "Mooncake PD dispatch snapshot callback failed"
            );
            (json!({}), elapsed_ms)
        }
        Err(error) => {
            warn!(
                request_id = %request_id_for_log,
                worker_id = %worker_id_for_log,
                error = %error,
                "Mooncake PD dispatch snapshot task failed"
            );
            (json!({}), elapsed_ms)
        }
    }
}

/// 通知 Python Router：D 请求已完成 attach 或开始输出，可收缩 Prompt 临时预算。
fn notify_pd_decode_budget_background(
    cb: Arc<PyCallbacks>,
    request_id: &str,
    worker_id: &str,
    phase: &str,
    metadata: &Value,
) {
    if cb.pd_decode_budget_fn.is_none() {
        return;
    }
    let request_id = request_id.to_string();
    let worker_id = worker_id.to_string();
    let phase = phase.to_string();
    let metadata_json = metadata.to_string();
    tokio::spawn(async move {
        let _ = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                if let Some(callback) = cb.pd_decode_budget_fn.as_ref() {
                    if let Err(error) =
                        callback.call1(py, (&request_id, &worker_id, &phase, &metadata_json))
                    {
                        warn!("pd_decode_budget_fn failed: {error}");
                    }
                }
            });
        })
        .await;
    });
}

/// 从成功的 D import 结果中提取可写入 Router 前缀索引的完整 block 元数据。
fn pd_router_cache_publish_metadata(
    source_worker_id: &str,
    target_worker_id: &str,
    source_export: Option<&Value>,
    import_result: &Value,
) -> Value {
    let Some(source_export) = source_export else {
        return json!({});
    };
    let Some(descriptors) = source_export["block_hashes"].as_array() else {
        return json!({});
    };
    let Some(destination_blocks) = import_result["destination_blocks"].as_array() else {
        return json!({});
    };

    let mut block_hashes = Vec::new();
    let mut destination_block_ids = Vec::new();
    for (descriptor, destination_block) in descriptors.iter().zip(destination_blocks) {
        if descriptor["request_scoped_partial"]
            .as_bool()
            .unwrap_or(false)
        {
            continue;
        }
        let Some(block_hash) = descriptor["external_hash"].as_str() else {
            continue;
        };
        let Some(destination_block_id) = destination_block.as_i64() else {
            continue;
        };
        if block_hash.is_empty() {
            continue;
        }
        block_hashes.push(block_hash.to_string());
        destination_block_ids.push(destination_block_id);
    }

    json!({
        "source_worker_id": source_worker_id,
        "target_worker_id": target_worker_id,
        "block_hashes": block_hashes,
        "destination_block_ids": destination_block_ids,
        "block_size": source_export["block_size"].as_u64(),
        "partial_valid_tokens": source_export["partial_valid_tokens"]
            .as_u64()
            .unwrap_or(0),
        "capacity_generation": import_result["capacity_generation"].as_u64(),
    })
}

// ---------------------------------------------------------------------------
// PyO3 模块入口
// ---------------------------------------------------------------------------

/// 启动 Rust 数据面代理。
///
/// 创建 tokio 多线程运行时，初始化 reqwest 连接池和 Axum 路由，
/// 监听指定地址并开始接受请求。此函数会阻塞直到服务关闭。
///
/// 参数:
///   - listen_addr: 监听地址（如 "0.0.0.0:8080"）
///   - route_fn: Python 路由回调，签名 (body_json, path) -> dict
///   - complete_fn: Python 完成回调，签名
///     (request_id, worker_id, ttft_s, total_s, completion_tokens,
///      prompt_tokens, worker_dispatch_to_first_byte_s,
///      router_to_early_d_dispatch_s, d_request_seen_to_attach_start_s,
///      d_attach_complete_to_first_byte_s, pd_diagnostics_json) -> None
///   - models_fn: Python 模型列表回调，签名 () -> JSON string
///   - max_body_size: 请求体大小限制（默认 10MB）
///   - pool_max_idle: 每个 host 的最大空闲连接数（默认 200）
///   - pool_idle_timeout: 空闲连接超时秒数（默认 30）
///   - pd_dispatch_snapshot_fn: 原生Connector派发D前读取Router负载的诊断回调
#[pyfunction]
#[pyo3(signature = (listen_addr, route_fn, complete_fn, models_fn, max_body_size=None, pool_max_idle=None, pool_idle_timeout=None, pd_prefill_max_inflight=None, pd_prefill_complete_fn=None, pd_dispatch_snapshot_fn=None, pd_decode_budget_fn=None))]
fn start_proxy(
    py: Python<'_>,
    listen_addr: String,
    route_fn: PyObject,
    complete_fn: PyObject,
    models_fn: PyObject,
    max_body_size: Option<usize>,
    pool_max_idle: Option<usize>,
    pool_idle_timeout: Option<u64>,
    pd_prefill_max_inflight: Option<usize>,
    pd_prefill_complete_fn: Option<PyObject>,
    pd_dispatch_snapshot_fn: Option<PyObject>,
    pd_decode_budget_fn: Option<PyObject>,
) -> PyResult<()> {
    let callbacks = Arc::new(PyCallbacks {
        route_fn,
        complete_fn,
        models_fn,
        pd_prefill_complete_fn,
        pd_dispatch_snapshot_fn,
        pd_decode_budget_fn,
    });

    py.allow_threads(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        rt.block_on(async {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        // Rust 日志 target 使用 lib crate 名 `_data_plane`，不是
                        // Cargo package 名 `jiuwen-data-plane`。
                        .unwrap_or_else(|_| "_data_plane=info".into()),
                )
                .try_init()
                .ok();

            let http_client = Client::builder()
                .pool_max_idle_per_host(pool_max_idle.unwrap_or(200))
                .pool_idle_timeout(std::time::Duration::from_secs(
                    pool_idle_timeout.unwrap_or(30),
                ))
                .connect_timeout(std::time::Duration::from_secs(10))
                .read_timeout(Duration::from_secs(positive_env(
                    "JIUWEN_PROXY_READ_TIMEOUT_SECS",
                    120,
                )))
                .timeout(Duration::from_secs(positive_env(
                    "JIUWEN_PROXY_REQUEST_TIMEOUT_SECS",
                    600,
                )))
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            let state = AppState {
                callbacks,
                http_client,
                max_body_size: max_body_size.unwrap_or(10 * 1024 * 1024),
                pd_link_cache: Arc::new(RwLock::new(HashMap::new())),
                pd_prefill_admission: Arc::new(Mutex::new(HashMap::new())),
                pd_prefill_max_inflight: pd_prefill_max_inflight.unwrap_or(0),
            };

            let app = Router::new()
                .route("/health", any(health_handler))
                .route("/v1/models", get(models_handler))
                .route("/{*path}", any(proxy_handler))
                .with_state(state);

            let addr: SocketAddr = listen_addr.parse().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Bad listen addr: {e}"))
            })?;

            info!("Data plane proxy listening on {addr}");

            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            axum::serve(listener, app)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            Ok(())
        })
    })
}

#[pymodule]
fn _data_plane(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_proxy, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_json_escapes_untrusted_text_and_status_is_typed() {
        let message = "quote\" slash\\ newline\n control\u{0001}";
        let error = ProxyError::Upstream(message.into());
        let parsed: Value = serde_json::from_str(&error.response_body()).unwrap();
        assert_eq!(parsed["error"]["message"], error.to_string());
        assert_eq!(
            ProxyError::NoCapacity("busy".into()).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            ProxyError::Upstream("No available workers".into()).status(),
            StatusCode::BAD_GATEWAY
        );
    }

    #[tokio::test]
    async fn upstream_body_limit_checks_actual_chunks() {
        let app = Router::new().route(
            "/large",
            get(|| async {
                let chunks = futures::stream::iter(vec![
                    Ok::<_, std::io::Error>(Bytes::from_static(b"12345678")),
                    Ok(Bytes::from_static(b"abcdefgh")),
                ]);
                Body::from_stream(chunks)
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let response = Client::new()
            .get(format!("http://{address}/large"))
            .send()
            .await
            .unwrap();
        let result = read_response_with_limit(response, 10).await;
        assert!(result.unwrap_err().contains("configured limit"));
        server.abort();
    }

    #[tokio::test]
    async fn stalled_upstream_body_hits_read_timeout() {
        let app = Router::new().route(
            "/stall",
            get(|| async {
                Body::from_stream(futures::stream::pending::<Result<Bytes, std::io::Error>>())
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = Client::builder()
            .read_timeout(Duration::from_millis(50))
            .build()
            .unwrap();
        let response = client
            .get(format!("http://{address}/stall"))
            .send()
            .await
            .unwrap();
        assert!(read_response_with_limit(response, 100).await.is_err());
        server.abort();
    }

    #[tokio::test]
    async fn kvbm_control_http_timing_splits_headers_body_and_decode() {
        let app = Router::new().route(
            "/timed",
            axum::routing::post(|| async { axum::Json(json!({"ok": true})) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener must bind");
        let address = listener.local_addr().expect("test address must exist");
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test server must run");
        });

        let (result, timing) = post_kvbm_control_timed(
            &Client::new(),
            &format!("http://{address}"),
            "/timed",
            json!({}),
            1_000,
        )
        .await;
        server.abort();

        assert_eq!(result.expect("timed request must succeed")["ok"], true);
        assert!(timing.send_to_headers_ms >= 0.0);
        assert!(timing.headers_to_body_ms >= 0.0);
        assert!(timing.json_decode_ms >= 0.0);
        assert!(timing.total_ms >= timing.send_to_headers_ms);
    }

    #[tokio::test]
    async fn pd_link_cache_hit_skips_worker_status_requests() {
        let cache = RwLock::new(HashMap::from([(
            (
                "http://127.0.0.1:1".to_string(),
                "http://127.0.0.1:2".to_string(),
            ),
            7,
        )]));
        let result = ensure_pd_datadist_link(
            &Client::new(),
            &cache,
            "http://127.0.0.1:1/",
            "http://127.0.0.1:2/",
            10,
        )
        .await
        .expect("cache hit must not access unreachable control URLs");

        assert_eq!(result.cluster_id, 7);
        assert!(result.cache_hit);
        assert_eq!(result.source_status_ms, 0.0);
        assert_eq!(result.target_status_ms, 0.0);
        assert_eq!(result.link_call_ms, 0.0);
    }

    #[test]
    fn pd_import_not_linked_error_is_detected_from_worker_event() {
        let result = Ok(json!({
            "ok": false,
            "error": "pull failed: LLM_NOT_YET_LINK",
        }));

        assert!(pd_import_reports_not_linked(&result));
        assert!(!pd_import_reports_not_linked(&Ok(json!({"ok": true}))));
    }

    #[test]
    fn pd_export_pending_requires_explicit_retryable_state() {
        assert!(pd_export_is_pending(&Ok(json!({
            "ok": false,
            "pending": true,
            "retryable": true,
        }))));
        assert!(!pd_export_is_pending(&Ok(json!({
            "ok": false,
            "pending": true,
            "retryable": false,
        }))));
        assert!(!pd_export_is_pending(&Err("network failed".to_string())));
    }

    #[test]
    fn pd_transaction_ttl_covers_export_and_control_tail() {
        assert_eq!(pd_import_completion_timeout_ms(10_000), 70_000);
        assert_eq!(pd_import_completion_timeout_ms(1_000), 60_000);
        assert_eq!(pd_import_completion_timeout_ms(100_000), 300_000);
        assert_eq!(pd_transaction_ttl_ms(120_000, 10_000), 190_000);
        assert_eq!(pd_transaction_ttl_ms(1_000, 1_000), 120_000);
        assert_eq!(pd_transaction_ttl_ms(590_000, 30_000), 600_000);
    }

    #[test]
    fn pd_capacity_admission_error_is_detected_from_event_and_status() {
        let error =
            "insufficient free vLLM blocks for delta import: requested=50, available=19, reused=0";
        let event = json!({
            "event_type": "REQUEST_SEEN",
            "result": {
                "preplan_ready": false,
                "preplan_error": error,
                "capacity_unavailable": true,
            }
        });
        let status = json!({
            "gate": {
                "request_seen": true,
                "preplan_error": error,
            }
        });
        let waiting = json!({
            "gate": {
                "request_seen": true,
                "preplan_error": error,
                "preplan_capacity_waiting": true,
            }
        });

        assert_eq!(pd_preplan_capacity_error(&event).as_deref(), Some(error));
        assert_eq!(pd_preplan_capacity_error(&status).as_deref(), Some(error));
        assert!(pd_preplan_capacity_error(&waiting).is_none());
        assert!(pd_preplan_capacity_error(&json!({
            "event_type": "REQUEST_SEEN",
            "result": {"preplan_ready": true, "capacity_unavailable": false}
        }))
        .is_none());
    }

    #[test]
    fn pd_target_allocation_metadata_supports_event_and_status() {
        let event = json!({
            "event_type": "REQUEST_SEEN",
            "result": {
                "scheduler_allocation_ready": true,
                "allocated_blocks": 50,
                "capacity_generation": 101,
            }
        });
        let status = json!({
            "gate": {
                "scheduler_allocation_ready": true,
                "scheduler_allocation": {"total_blocks": 51},
            }
        });

        assert_eq!(
            pd_target_allocation_metadata(&event).unwrap()["allocated_blocks"],
            50
        );
        assert_eq!(
            pd_target_allocation_metadata(&event).unwrap()["capacity_generation"],
            101
        );
        assert_eq!(
            pd_target_allocation_metadata(&status).unwrap()["allocated_blocks"],
            51
        );
        assert!(pd_target_allocation_metadata(&json!({
            "result": {"scheduler_allocation_ready": false}
        }))
        .is_none());
    }

    #[test]
    fn pd_router_publish_metadata_excludes_request_scoped_partial_block() {
        let source_export = json!({
            "block_hashes": [
                {"external_hash": "hash-0", "request_scoped_partial": false},
                {"external_hash": "hash-1", "request_scoped_partial": false},
                {"request_scoped_partial": true, "valid_token_count": 28}
            ],
            "partial_valid_tokens": 28
        });
        let import_result = json!({"destination_blocks": [70, 71, 72]});

        let metadata = pd_router_cache_publish_metadata(
            "prefill-0",
            "decode-0",
            Some(&source_export),
            &import_result,
        );

        assert_eq!(metadata["source_worker_id"], "prefill-0");
        assert_eq!(metadata["target_worker_id"], "decode-0");
        assert_eq!(metadata["block_hashes"], json!(["hash-0", "hash-1"]));
        assert_eq!(metadata["destination_block_ids"], json!([70, 71]));
        assert_eq!(metadata["partial_valid_tokens"], 28);
    }

    #[test]
    fn native_pd_merges_prefill_transfer_metadata_into_decode_request() {
        let prefill_response = json!({
            "id": "prefill-1",
            "kv_transfer_params": {
                "do_remote_decode": true,
                "do_remote_prefill": false,
                "remote_engine_id": "engine-1",
                "remote_block_ids": [10, 11, 12]
            }
        });
        let (body, remote_block_count) = merge_native_kv_transfer_params(
            r#"{"model":"test","prompt":"hello"}"#,
            prefill_response.to_string().as_bytes(),
        )
        .expect("native metadata must merge");
        let value: Value = serde_json::from_str(&body).expect("merged body must be JSON");

        assert_eq!(remote_block_count, 3);
        assert_eq!(value["kv_transfer_params"]["do_remote_prefill"], true);
        assert_eq!(value["kv_transfer_params"]["do_remote_decode"], false);
        assert_eq!(value["kv_transfer_params"]["remote_engine_id"], "engine-1");
    }

    #[test]
    fn native_pd_rejects_prefill_response_without_transfer_metadata() {
        let error = merge_native_kv_transfer_params(
            r#"{"model":"test","prompt":"hello"}"#,
            br#"{"id":"prefill-1"}"#,
        )
        .expect_err("missing native metadata must not silently recompute on D");

        assert!(error.contains("did not contain kv_transfer_params"));
    }
}
