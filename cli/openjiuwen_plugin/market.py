"""Market API client: search, delete, upload. Depends on market providing corresponding APIs."""
from __future__ import annotations

import json
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
    SkillImportResponse,
    PluginVersionDeleteData,
    PluginVersionDetail,
)

ModelT = TypeVar("ModelT")

logger = logging.getLogger(__name__)

MARKET_HTTP_DEFAULT_TIMEOUT_SEC = 60
MARKET_HTTP_LONG_TRANSFER_TIMEOUT_SEC = 600


class PublishError(Exception):
    """Publish/upload failed: network or market returns error."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{detail} (status={status_code})")


def _market_humanize_error_body(j: Any) -> str | None:
    """Turn market JSON error body into one line for CLI (handles nested ``detail`` dict)."""
    if isinstance(j, str) and j.strip():
        s = j.strip()
        if s.startswith("{") and '"message"' in s:
            try:
                return _market_humanize_error_body(json.loads(s))
            except Exception:
                return s
        return s
    if not isinstance(j, dict):
        return None
    m = j.get("message")
    if isinstance(m, str) and m.strip():
        return m.strip()
    d = j.get("detail")
    if isinstance(d, str) and d.strip():
        return _market_humanize_error_body(d.strip())
    if isinstance(d, list) and d:
        parts: list[str] = []
        for item in d:
            if isinstance(item, dict) and isinstance(item.get("msg"), str):
                parts.append(str(item["msg"]).strip())
        if parts:
            return "; ".join(parts)
    if isinstance(d, dict):
        msg = d.get("message")
        err = d.get("error")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
        if isinstance(err, str) and err.strip():
            return err.strip()
    e = j.get("error")
    if isinstance(e, str) and e.strip():
        return e.strip()
    return None


def _market_request_error_message(market_base: str, exc: BaseException) -> str:
    """Wording for ``requests.RequestException`` (DNS/connect/timeout/TLS, etc.)."""
    return (
        f"Cannot reach marketplace {market_base!r}: {exc}. "
        f"Check URL and network; base URL is from --market-url or OPENJIUWEN_MARKET_URL."
    )


def _redact_url_for_cli_error(url: str) -> str:
    """Strip query and fragment so presigned tokens are not echoed in CLI errors/logs."""
    raw = (url or "").strip()
    if not raw:
        return "<empty download url>"
    try:
        parts = urllib.parse.urlsplit(raw)
    except ValueError:
        return "<invalid download url>"
    path = parts.path or "/"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _market_format_http_error(resp: Response) -> str:
    """Build a user-facing error line from a failed HTTP response (shared by market client calls)."""
    status = resp.status_code
    ct = (resp.headers.get("content-type") or "").lower()
    if status == 404 and "application/json" not in ct:
        prev = (resp.text or "")[:240].replace("\n", " ").strip()
        return (
            f"HTTP 404 from market URL (wrong host/port/path or not the marketplace API). "
            f"Check URL and path; base URL is from --market-url or OPENJIUWEN_MARKET_URL. Preview: {prev!r}"
        )
    try:
        j = resp.json()
    except Exception as exc:
        body = (resp.text or "").strip()
        if body:
            msg = body if len(body) < 800 else f"HTTP {status} (long non-JSON body)"
        else:
            msg = f"HTTP {status} (failed to parse error body: {exc})"
        return msg

    if isinstance(j, dict):
        human = _market_humanize_error_body(j)
        if human:
            return human
    if not isinstance(j, dict):
        msg = (resp.text or "").strip() or f"HTTP {status}"
        return msg
    msg = (resp.text or "").strip() or f"HTTP {status}"
    return msg


def _market_read_json_response(resp: Response, *, err_prefix: str = "response") -> dict[str, Any]:
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


def _market_coerce_envelope_model(
    payload: dict[str, Any],
    model_type: type[ModelT],
    *,
    err_prefix: str = "response",
) -> ModelT:
    data = ResponseModel[dict].model_validate(payload).data
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid {err_prefix}: missing object field 'data'")
    try:
        return TypeAdapter(model_type).validate_python(data)
    except ValidationError as exc:
        raise RuntimeError(f"invalid {err_prefix}: {exc}") from exc


def _market_get_json_envelope(
    market_base: str,
    url: str,
    model_type: type[ModelT],
    *,
    params: dict[str, Any] | None = None,
    timeout: int = MARKET_HTTP_DEFAULT_TIMEOUT_SEC,
    err_prefix: str = "response",
) -> ModelT:
    req_kw: dict[str, Any] = {"timeout": timeout}
    if params is not None:
        req_kw["params"] = params
    try:
        resp = _market_http_request_with_retry(requests.get, url, **req_kw)
    except requests.RequestException as e:
        raise RuntimeError(_market_request_error_message(market_base, e)) from e
    if not resp.ok:
        raise RuntimeError(_market_format_http_error(resp))
    payload = _market_read_json_response(resp, err_prefix=err_prefix)
    return _market_coerce_envelope_model(payload, model_type, err_prefix=err_prefix)


def _market_should_retry_status(resp: Response) -> bool:
    return resp.status_code in {408, 425, 429, 500, 502, 503, 504}


def _market_http_error_brief(resp: Response) -> str:
    """Best-effort message for logging / final error; safe before resp.close()."""
    try:
        return _market_format_http_error(resp)
    except Exception:
        try:
            text = (resp.text or "")[:800]
            return text.strip() or f"HTTP {resp.status_code}"
        except Exception:
            return f"HTTP {resp.status_code}"


def _market_release_response(resp: Response) -> None:
    try:
        resp.close()
    except Exception:
        logger.debug("failed to close HTTP response", exc_info=True)


def _market_http_request_with_retry(
    method: Callable[..., Response],
    *args: Any,
    **kwargs: Any,
) -> Response:
    """Idempotent GET-style calls: retry with backoff on network errors and some HTTP statuses. Not for writes."""
    max_attempts = 3
    delay = 0.5
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = method(*args, **kwargs)
            if _market_should_retry_status(resp):
                detail = _market_http_error_brief(resp)
                _market_release_response(resp)
                msg = f"HTTP {resp.status_code}: {detail}"
                if attempt >= max_attempts:
                    raise requests.RequestException(msg)
                logger.warning(
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
            logger.warning("request failed (attempt %s/%s): %s; retrying in %.1fs", attempt, max_attempts, exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 6.0)

    if last_exc is None:
        raise RuntimeError("_market_http_request_with_retry: loop exited without result or exception")
    raise last_exc


def plugin_upload(
    market_url: str,
    user_token: str | None,
    system_token: str | None,
    req: PublishRequest,
) -> PluginPublishResult:
    """Publish: multipart zip upload; exactly one of Bearer or X-System-Token; no retries."""
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
    data: dict[str, str] = {
        "force": "true" if req.force else "false",
        "version_desc": req.version_desc if req.version_desc is not None else "",
    }
    if req.plugin_id is not None:
        data["plugin_id"] = str(req.plugin_id)
    if req.plugin_version is not None:
        data["plugin_version"] = str(req.plugin_version)

    logger.info("正在上传插件包，请稍候；包体较大时耗时更长。")
    with open(req.zip_path, "rb") as f:
        files = {"file": (req.zip_path.name, f, "application/zip")}
        try:
            resp = requests.post(
                url,
                data=data,
                files=files,
                headers=headers,
                timeout=MARKET_HTTP_LONG_TRANSFER_TIMEOUT_SEC,
            )
        except requests.RequestException as e:
            raise PublishError(0, _market_request_error_message(base, e)) from e

    if not resp.ok:
        detail = _market_format_http_error(resp)
        raise PublishError(resp.status_code, detail)

    try:
        payload = _market_read_json_response(resp)
        return _market_coerce_envelope_model(payload, PluginPublishResult)
    except RuntimeError as exc:
        raise PublishError(resp.status_code, str(exc)) from exc


def skill_import(
    market_url: str,
    system_token: str,
    *,
    zip_path: Path,
    checksum_sha256: str,
    force: bool = False,
    fail_fast: bool = False,
    timeout_sec: int = MARKET_HTTP_LONG_TRANSFER_TIMEOUT_SEC,
) -> SkillImportResponse:
    """skill-import: POST /skill-import; X-System-Token only."""
    base = market_url.rstrip("/")
    url = f"{base}/api/v1/plugins/skill-import"
    tok = str(system_token).strip()
    if not tok:
        raise PublishError(0, "system_token is required for skill-import")

    headers: dict[str, str] = {
        "X-System-Token": tok,
        "X-Checksum-SHA256": str(checksum_sha256).strip().lower(),
    }
    data: dict[str, str] = {
        "force": "true" if force else "false",
        "fail_fast": "true" if fail_fast else "false",
    }

    bundle = Path(zip_path).resolve()
    if not bundle.is_file():
        raise PublishError(0, f"zip file not found: {bundle}")

    logger.info("正在上传 skills 包，请稍候；包体较大时耗时更长。")
    with open(bundle, "rb") as f:
        files = {"file": (bundle.name, f, "application/zip")}
        try:
            resp = requests.post(url, data=data, files=files, headers=headers, timeout=timeout_sec)
        except requests.RequestException as e:
            raise PublishError(0, _market_request_error_message(base, e)) from e

    if not resp.ok:
        detail = _market_format_http_error(resp)
        raise PublishError(resp.status_code, detail)

    try:
        payload = _market_read_json_response(resp)
        return _market_coerce_envelope_model(payload, SkillImportResponse)
    except RuntimeError as exc:
        raise PublishError(resp.status_code, str(exc)) from exc


def plugin_info(
    market_url: str,
    asset_id: str,
    version: str,
) -> PluginVersionDetail:
    """plugin info: GET one plugin version detail."""
    base = market_url.rstrip("/")
    aid_seg = urllib.parse.quote(asset_id.strip(), safe="")
    ver_seg = urllib.parse.quote(version.strip(), safe="")
    url = f"{base}/api/v1/plugins/{aid_seg}/versions/{ver_seg}"
    return _market_get_json_envelope(base, url, PluginVersionDetail)


def plugin_search(
    market_url: str,
    query: PluginListQuery,
) -> PluginListResponse:
    """plugin search: GET list; params match ``PluginListQuery``; no Authorization header."""
    base = market_url.rstrip("/")
    url = f"{base}/api/v1/plugins"
    q = PluginListQuery(
        search_keyword=query.search_keyword or "",
        plugin_type=query.plugin_type,
        publisher_name=query.publisher_name,
        asset_id=query.asset_id,
        asset_type=query.asset_type,
        publisher_id=query.publisher_id,
        page=max(1, int(query.page)),
        page_size=max(1, min(int(query.page_size), 100)),
        order_by=query.order_by,
        desc=bool(query.desc),
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

    return _market_get_json_envelope(base, url, PluginListResponse, params=params)


def plugin_delete(
    market_url: str,
    asset_id: str,
    log: logging.Logger,
    *,
    version: str | None = None,
    user_token: str | None = None,
    system_token: str | None = None,
) -> PluginVersionDeleteData:
    """plugin delete: DELETE version (default version=all); exactly one auth method; no retries."""
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
        resp = requests.delete(url, headers=headers, timeout=MARKET_HTTP_DEFAULT_TIMEOUT_SEC)
    except requests.RequestException as e:
        raise RuntimeError(_market_request_error_message(base, e)) from e
    if not resp.ok:
        raise RuntimeError(_market_format_http_error(resp))
    log.info("deleted: asset_id=%s version=%s", asset_id, ver_seg)
    if str(resp.headers.get("content-type") or "").startswith("application/json"):
        payload = _market_read_json_response(resp)
        try:
            return _market_coerce_envelope_model(payload, PluginVersionDeleteData)
        except RuntimeError:
            pass
    return PluginVersionDeleteData(asset_id=asset_id, version=ver_seg)


def plugin_install_download(
    market_url: str,
    asset_id: str,
    dest_path: Path,
    *,
    version: str | None = None,
) -> DownloadArtifactResult:
    """Install phase 1: fetch artifact metadata, download zip to ``dest_path``, verify checksum if present."""
    base = market_url.rstrip("/")
    aid_seg = urllib.parse.quote(asset_id.strip(), safe="")
    metadata_url = f"{base}/api/v1/artifacts/{aid_seg}"
    ver = (version or "").strip()
    if ver:
        metadata_url = f"{metadata_url}?{urllib.parse.urlencode({'version': ver})}"

    data = _market_get_json_envelope(
        base,
        metadata_url,
        PluginDownloadData,
        err_prefix="artifact metadata response",
    )

    download_url = data.download_url.strip()
    expected_checksum = data.checksum_sha256.strip().lower()
    if not download_url:
        raise RuntimeError("artifact metadata missing download_url")
    if expected_checksum and (
        len(expected_checksum) != 64
        or any(c not in "0123456789abcdef" for c in expected_checksum)
    ):
        raise RuntimeError("artifact metadata checksum_sha256 is invalid")

    dl_loc = _redact_url_for_cli_error(download_url)
    try:
        dl_resp = _market_http_request_with_retry(
            requests.get, download_url, stream=True, timeout=MARKET_HTTP_LONG_TRANSFER_TIMEOUT_SEC
        )
    except requests.RequestException as e:
        raise RuntimeError(
            f"{_market_request_error_message(base, e)} "
            f"(artifact zip GET; location={dl_loc!r})"
        ) from e
    if not dl_resp.ok:
        raise RuntimeError(
            f"{_market_format_http_error(dl_resp)} "
            f"(artifact zip GET; location={dl_loc!r})"
        )

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