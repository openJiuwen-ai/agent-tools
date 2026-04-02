"""Market API client: search, delete, upload. Depends on market providing corresponding APIs."""
from __future__ import annotations

import logging
import hashlib
import time
import urllib.parse
from pathlib import Path
from typing import Any, TypeVar, Callable

import requests
from requests import Response
from pydantic import TypeAdapter, ValidationError
from openjiuwen_plugin.schemas import (
    DownloadArtifactResult,
    PluginDownloadData,
    PluginListQuery,
    PluginListResponse,
    PublishRequest,
    PluginPublishResult,
    ResponseModel,
    PluginVersionDeleteData,
    PluginVersionDetail,
)

ModelT = TypeVar("ModelT")


class PublishError(Exception):
    """Publish/upload failed: network or market returns error."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code  # 0 for local/network error
        self.detail = detail
        super().__init__(f"{detail} (status={status_code})")


def _upload_error_detail(resp: Response) -> str:
    """Extract readable information from market error response."""
    try:
        j = resp.json()
        if not isinstance(j, dict):
            return resp.text or f"HTTP {resp.status_code}"
        for key in ("message", "detail"):
            val = j.get(key)
            if val is None:
                continue
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, list) and val:
                return "; ".join(str(x) for x in val)
        if j.get("error"):
            return str(j["error"])
    except Exception as exc:
        return resp.text or f"HTTP {resp.status_code} (failed to parse error body: {exc})"
    return resp.text or f"HTTP {resp.status_code}"


def _parse_json_body(resp: Response, *, err_prefix: str = "response") -> dict[str, Any]:
    content_type = str(resp.headers.get("content-type") or "")
    if not content_type.startswith("application/json"):
        raise RuntimeError(f"{err_prefix} is not a valid JSON object")
    try:
        payload = resp.json()
    except Exception as exc:
        raise RuntimeError(f"{err_prefix} is not a valid JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{err_prefix} is not a valid JSON object")
    return payload


def _extract_data_dict(payload: dict[str, Any], *, err_prefix: str = "response") -> dict[str, Any]:
    data = ResponseModel[dict].model_validate(payload).data
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid {err_prefix}: missing object field 'data'")
    return data


def _parse_data_as(payload: dict[str, Any], model_type: type[ModelT], *, err_prefix: str = "response") -> ModelT:
    data = _extract_data_dict(payload, err_prefix=err_prefix)
    try:
        return TypeAdapter(model_type).validate_python(data)
    except ValidationError as exc:
        raise RuntimeError(f"invalid {err_prefix}: {exc}") from exc


def _should_retry_response(resp: Response) -> bool:
    # Retry typical transient / overload responses. Omit 409 (conflict): usually not safe to blindly retry.
    return resp.status_code in {408, 425, 429, 500, 502, 503, 504}


def _brief_http_error_message(resp: Response) -> str:
    """Best-effort message for logging / final error; safe before resp.close()."""
    try:
        return _upload_error_detail(resp)
    except Exception:
        try:
            text = (resp.text or "")[:800]
            return text.strip() or f"HTTP {resp.status_code}"
        except Exception:
            return f"HTTP {resp.status_code}"


def _release_response(resp: Response) -> None:
    try:
        resp.close()
    except Exception as exc:
        logging.getLogger(__name__).debug("response.close() failed: %s", exc, exc_info=True)


def _request_with_retry(
    method: Callable[..., Response],
    *args: Any,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> Response:
    """
    Wrapper around requests.* with retries on network errors and selected transient HTTP statuses.

    - Retries: RequestException, and HTTP 408/425/429/5xx (not 409)
    - Before retrying on HTTP: response body summarized, connection released via close()
    - Backoff: exponential, capped

    Do not use for mutating market calls without idempotency (e.g. ``upload_plugin`` POST, ``delete_plugin`` DELETE);
    those use a single ``requests.*`` call.
    """
    log = logger or logging.getLogger(__name__)

    max_attempts = 3
    delay = 0.5
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = method(*args, **kwargs)
            if _should_retry_response(resp):
                detail = _brief_http_error_message(resp)
                _release_response(resp)
                msg = f"HTTP {resp.status_code}: {detail}"
                if attempt >= max_attempts:
                    raise requests.RequestException(msg)
                log.warning(
                    "transient HTTP (attempt %s/%s): %s; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    msg[:500],
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 6.0)
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            log.warning("request failed (attempt %s/%s): %s; retrying in %.1fs", attempt, max_attempts, exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 6.0)

    if last_exc is None:
        raise RuntimeError("_request_with_retry: loop exited without result or exception")
    raise last_exc


def upload_plugin(
    market_url: str,
    user_token: str | None,
    system_token: str | None,
    req: PublishRequest,
) -> PluginPublishResult:
    """
    Upload plugin zip to market.

    Auth: provide exactly one of Bearer user_token or X-System-Token system_token.

    Single HTTP attempt (no automatic retry): multipart POST is not safely idempotent if the server
    accepted the first request but the client did not receive the response.
    """
    base = market_url.rstrip("/")
    url = f"{base}/api/v1/plugins"
    has_user = bool(user_token and user_token.strip())
    has_sys = bool(system_token and system_token.strip())
    if has_user == has_sys:
        raise PublishError(0, "provide exactly one auth method: user_token or system_token")

    headers: dict[str, str] = {"X-Checksum-SHA256": req.checksum_sha256}
    if has_sys:
        headers["X-System-Token"] = system_token.strip()
    else:
        headers["Authorization"] = f"Bearer {user_token.strip()}"
    data = {"force": "true" if req.force else "false"}
    if req.plugin_id is not None:
        data["plugin_id"] = req.plugin_id
    if req.plugin_version is not None:
        data["plugin_version"] = req.plugin_version
    if req.version_desc is not None:
        data["version_desc"] = req.version_desc

    with open(req.zip_path, "rb") as f:
        files = {"file": (req.zip_path.name, f, "application/zip")}
        try:
            resp = requests.post(url, data=data, files=files, headers=headers, timeout=60)
        except requests.RequestException as e:
            raise PublishError(0, f"network error: {e}") from e

    if not resp.ok:
        detail = _upload_error_detail(resp)
        raise PublishError(resp.status_code, detail)

    try:
        payload = _parse_json_body(resp)
        return _parse_data_as(payload, PluginPublishResult)
    except RuntimeError as exc:
        raise PublishError(resp.status_code, str(exc)) from exc


def get_plugin_version_detail(
    market_url: str,
    asset_id: str,
    version: str,
) -> PluginVersionDetail:
    """Get plugin version details from market: GET /api/v1/plugins/{asset_id}/versions/{version}."""
    base = market_url.rstrip("/")
    aid_seg = urllib.parse.quote(asset_id.strip(), safe="")
    ver_seg = urllib.parse.quote(version.strip(), safe="")
    url = f"{base}/api/v1/plugins/{aid_seg}/versions/{ver_seg}"
    try:
        resp = _request_with_retry(requests.get, url, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"request failed: {e}") from e
    if resp.status_code == 404:
        raise FileNotFoundError(f"plugin version not found: asset_id={asset_id!r} version={version!r}")
    resp.raise_for_status()
    payload = _parse_json_body(resp)
    return _parse_data_as(payload, PluginVersionDetail)


PLUGIN_LIST_ORDER_BY = ("install_count", "like_count", "create_time", "update_time", "review_count")


def _plugin_list_order_by_desc(order_by: str | None, desc: bool) -> tuple[str, bool]:
    """Consistent with PluginListQuery.order_by/desc; CLI directly uses desc semantics."""
    key = (order_by or "install_count").strip()
    if key not in PLUGIN_LIST_ORDER_BY:
        key = "install_count"
    return key, bool(desc)


def search_plugins(
    market_url: str,
    query: PluginListQuery,
) -> PluginListResponse:
    """
    Call GET /api/v1/plugins, query parameters correspond one-to-one with PluginListQuery fields.

    Public interface: does not send Authorization.

    Query should be PluginListQuery.
    """
    base = market_url.rstrip("/")
    url = f"{base}/api/v1/plugins"
    ob, desc_value = _plugin_list_order_by_desc(query.order_by, query.desc)
    q = PluginListQuery(
        search_keyword=query.search_keyword or "",
        plugin_type=query.plugin_type,
        publisher_name=query.publisher_name,
        asset_id=query.asset_id,
        asset_type=query.asset_type,
        publisher_id=query.publisher_id,
        page=max(1, int(query.page)),
        page_size=max(1, min(int(query.page_size), 100)),
        order_by=ob,
        desc=desc_value,
    )
    params: dict[str, str | int | bool] = {
        "page": q.page,
        "page_size": q.page_size,
        "order_by": q.order_by,
        "desc": q.desc,
    }
    if q.search_keyword:
        params["search_keyword"] = q.search_keyword
    if q.plugin_type:
        params["plugin_type"] = q.plugin_type
    if q.publisher_name:
        params["publisher_name"] = q.publisher_name
    if q.asset_id:
        params["asset_id"] = q.asset_id
    if q.asset_type:
        params["asset_type"] = q.asset_type
    if q.publisher_id:
        params["publisher_id"] = q.publisher_id

    try:
        resp = _request_with_retry(requests.get, url, params=params, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"request failed: {e}") from e
    if resp.status_code == 404:
        return PluginListResponse(page=q.page, page_size=q.page_size, total=0, items=[])
    resp.raise_for_status()
    payload = _parse_json_body(resp)
    return _parse_data_as(payload, PluginListResponse)


def delete_plugin(
    market_url: str,
    asset_id: str,
    logger: logging.Logger,
    *,
    version: str | None = None,
    user_token: str | None = None,
    system_token: str | None = None,
) -> PluginVersionDeleteData:
    """
    DELETE /api/v1/plugins/{asset_id}/versions/{version}

    version defaults to ``all`` (delete all versions and the asset).

    Auth: provide exactly one of Bearer user_token or X-System-Token system_token.

    Single HTTP attempt (no automatic retry): same rationale as ``upload_plugin`` — the server may
    have applied the delete while the client sees a timeout or connection error; retries add
    ambiguity without an idempotency contract.
    """
    has_user = bool(user_token and user_token.strip())
    has_sys = bool(system_token and system_token.strip())
    if has_user == has_sys:
        raise RuntimeError("provide exactly one of user_token or system_token")

    base = market_url.rstrip("/")
    ver_seg = (version or "all").strip()
    path_id = urllib.parse.quote(asset_id.strip(), safe="")
    path_ver = urllib.parse.quote(ver_seg, safe="")
    url = f"{base}/api/v1/plugins/{path_id}/versions/{path_ver}"
    headers: dict[str, str] = {}
    if has_sys:
        headers["X-System-Token"] = system_token.strip()
    else:
        headers["Authorization"] = f"Bearer {user_token.strip()}"

    try:
        resp = requests.delete(url, headers=headers, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"request failed: {e}") from e
    if not resp.ok:
        raise RuntimeError(_upload_error_detail(resp))
    logger.info("deleted: asset_id=%s version=%s", asset_id, ver_seg)
    if str(resp.headers.get("content-type") or "").startswith("application/json"):
        payload = _parse_json_body(resp)
        try:
            return _parse_data_as(payload, PluginVersionDeleteData)
        except RuntimeError:
            pass
    return PluginVersionDeleteData(asset_id=asset_id, version=ver_seg)


def download_artifact_zip(
    market_url: str,
    asset_id: str,
    dest_path: Path,
) -> DownloadArtifactResult:
    """
    First call GET /api/v1/artifacts/{asset_id} to get download information (download_url/checksum),
    then download zip to ``dest_path``, and perform integrity check when the server provides checksum_sha256.

    Return value example:
    {
      "download_url": "...",
      "expected_checksum_sha256": "...",  # May be empty
      "actual_checksum_sha256": "...",
      "verified": True/False,
    }
    """
    base = market_url.rstrip("/")
    aid_seg = urllib.parse.quote(asset_id.strip(), safe="")
    metadata_url = f"{base}/api/v1/artifacts/{aid_seg}"

    try:
        meta_resp = _request_with_retry(requests.get, metadata_url, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"request failed: {e}") from e

    if not meta_resp.ok:
        raise RuntimeError(_upload_error_detail(meta_resp))

    payload = _parse_json_body(meta_resp, err_prefix="artifact metadata response")
    data = _parse_data_as(payload, PluginDownloadData, err_prefix="artifact metadata response")

    download_url = data.download_url.strip()
    expected_checksum = data.checksum_sha256.strip().lower()
    if not download_url:
        raise RuntimeError("artifact metadata missing download_url")
    if expected_checksum and (
        len(expected_checksum) != 64
        or any(c not in "0123456789abcdef" for c in expected_checksum)
    ):
        raise RuntimeError("artifact metadata checksum_sha256 is invalid")

    try:
        dl_resp = _request_with_retry(requests.get, download_url, stream=True, timeout=300)
    except requests.RequestException as e:
        raise RuntimeError(f"request failed: {e}") from e
    if not dl_resp.ok:
        raise RuntimeError(_upload_error_detail(dl_resp))

    dest_path = dest_path.resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    first_bytes = b""
    with open(dest_path, "wb") as f:
        for chunk in dl_resp.iter_content(chunk_size=1 << 20):
            if chunk:
                if len(first_bytes) < 4:
                    need = 4 - len(first_bytes)
                    first_bytes += chunk[:need]
                f.write(chunk)
                hasher.update(chunk)

    if len(first_bytes) < 2 or first_bytes[:2] != b"PK":
        try:
            dest_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            "response is not a zip file (missing ZIP magic); check URL or server error body"
        )
    actual_checksum = hasher.hexdigest()
    verified = bool(expected_checksum)
    if expected_checksum and actual_checksum != expected_checksum:
        try:
            dest_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            "downloaded zip checksum mismatch: "
            f"expected={expected_checksum} actual={actual_checksum}"
        )

    return DownloadArtifactResult(
        download_url=download_url,
        expected_checksum_sha256=expected_checksum,
        actual_checksum_sha256=actual_checksum,
        verified=verified,
    )