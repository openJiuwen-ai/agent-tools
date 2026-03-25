"""Market API client: search, delete, upload. 依赖市场提供对应接口。"""
from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path

import requests


class PublishError(Exception):
    """发布/上传失败：网络或市场返回错误。"""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code  # 0 表示本地/网络错误
        self.detail = detail
        super().__init__(f"{detail} (status={status_code})")


def _upload_error_detail(resp) -> str:
    """从市场错误响应中取可读信息。"""
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


def upload_plugin(
    market_url: str,
    user_token: str | None,
    system_token: str | None,
    user_id: str,
    zip_path: Path,
    *,
    checksum_sha256: str,
    plugin_id: str | None = None,
    plugin_version: str | None = None,
    version_desc: str | None = None,
    force: bool = False,
) -> dict:
    """
    将插件 zip 上传到市场。成功返回市场响应的 data 字典；失败抛出 PublishError。
    """
    base = market_url.rstrip("/")
    url = f"{base}/api/v1/plugins"
    has_user = bool(user_token and user_token.strip())
    has_sys = bool(system_token and system_token.strip())
    if has_user == has_sys:
        raise PublishError(0, "provide exactly one auth method: user_token or system_token")

    headers: dict[str, str] = {"X-Checksum-SHA256": checksum_sha256}
    if has_sys:
        headers["X-System-Token"] = system_token.strip()
    else:
        headers["Authorization"] = f"Bearer {user_token.strip()}"
    data = {
        "user_id": user_id,
        "force": "true" if force else "false",
    }
    if plugin_id is not None:
        data["plugin_id"] = plugin_id
    if plugin_version is not None:
        data["plugin_version"] = plugin_version
    if version_desc is not None:
        data["version_desc"] = version_desc

    with open(zip_path, "rb") as f:
        files = {"file": (zip_path.name, f, "application/zip")}
        try:
            resp = requests.post(url, data=data, files=files, headers=headers, timeout=60)
        except requests.RequestException as e:
            raise PublishError(0, f"network error: {e}") from e

    if not resp.ok:
        detail = _upload_error_detail(resp)
        raise PublishError(resp.status_code, detail)

    try:
        body = resp.json()
    except Exception as e:
        raise PublishError(resp.status_code, f"response is not valid JSON: {e}") from e
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body if isinstance(body, dict) else {}


def get_plugin_version_detail(
    market_url: str,
    asset_id: str,
    version: str,
    token: str | None = None,
) -> dict:
    """从市场获取插件版本详情：GET /api/v1/plugins/{asset_id}/version/{version}。"""
    base = market_url.rstrip("/")
    aid_seg = urllib.parse.quote(asset_id.strip(), safe="")
    ver_seg = urllib.parse.quote(version.strip(), safe="")
    url = f"{base}/api/v1/plugins/{aid_seg}/version/{ver_seg}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(url, headers=headers or None, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"request failed: {e}") from e
    if resp.status_code == 404:
        raise FileNotFoundError(f"plugin version not found: asset_id={asset_id!r} version={version!r}")
    resp.raise_for_status()
    payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    if not isinstance(payload, dict):
        return {}
    return payload


PLUGIN_LIST_ORDER_BY = ("install_count", "like_count", "create_time", "update_time", "review_count")


def _plugin_list_order_by_desc(order_by: str | None, asc: bool) -> tuple[str, bool]:
    """与 PluginListQuery.order_by / desc 一致；asc=True 对应 API desc=false。"""
    key = (order_by or "install_count").strip()
    if key not in PLUGIN_LIST_ORDER_BY:
        key = "install_count"
    return key, not asc


def _extract_plugin_list_items(payload: object) -> list:
    """Normalize GET /api/v1/plugins JSON body to a list of item dicts (always returns a list)."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict) or "data" not in payload:
        return []
    inner = payload["data"]
    if isinstance(inner, dict):
        raw = inner.get("items")
        return raw if isinstance(raw, list) else []
    if isinstance(inner, list):
        return inner
    return []


def search_plugins(
    market_url: str,
    query: str,
    logger: logging.Logger,
    *,
    plugin_type: str | None = None,
    publisher_name: str | None = None,
    asset_id: str | None = None,
    asset_type: str | None = None,
    publisher_id: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
    order_by: str | None = None,
    asc: bool = False,
) -> None:
    """
    调用 GET /api/v1/plugins，查询参数与 PluginListQuery 字段一一对应。

    公开接口：不发送 Authorization。

    - 位置参数 query → search_keyword
    - --type → plugin_type
    - --author → publisher_name（CLI 沿用 --author 名称，与 API publisher_name 对应）
    - --asset-id / --asset-type / --publisher-id → 同名 query
    - --page → page；--page-size → page_size（默认与 API 一致：1 / 20）
    - --order-by / --asc → order_by / desc
    """
    base = market_url.rstrip("/")
    url = f"{base}/api/v1/plugins"
    params: dict[str, str | int | bool] = {}
    if query:
        params["search_keyword"] = query
    if plugin_type and str(plugin_type).strip():
        params["plugin_type"] = str(plugin_type).strip()
    if publisher_name and str(publisher_name).strip():
        params["publisher_name"] = str(publisher_name).strip()
    if asset_id and str(asset_id).strip():
        params["asset_id"] = str(asset_id).strip()
    if asset_type and str(asset_type).strip():
        params["asset_type"] = str(asset_type).strip()
    if publisher_id and str(publisher_id).strip():
        params["publisher_id"] = str(publisher_id).strip()

    ps = 20 if page_size is None else max(1, min(int(page_size), 100))
    pg = 1 if page is None else max(1, int(page))
    params["page"] = pg
    params["page_size"] = ps

    ob, desc = _plugin_list_order_by_desc(order_by, asc)
    params["order_by"] = ob
    params["desc"] = desc

    try:
        resp = requests.get(url, params=params, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"request failed: {e}") from e
    if resp.status_code == 404:
        logger.info("search API not implemented (404); market may not expose GET /api/v1/plugins yet.")
        return
    resp.raise_for_status()
    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    items = _extract_plugin_list_items(data)
    if not items:
        logger.info("no results.")
        return
    for item in items:
        if isinstance(item, dict):
            aid = item.get("asset_id", "")
            name = item.get("name", "")
            ver = item.get("latest_version", "")
            logger.info("  %s  %s  %s", aid, name, ver)
        else:
            logger.info("  %s", item)


def delete_plugin(
    market_url: str,
    asset_id: str,
    logger: logging.Logger,
    *,
    version: str | None = None,
    user_id: str | None = None,
    user_token: str | None = None,
    system_token: str | None = None,
) -> None:
    """
    DELETE /api/v1/plugins/{asset_id}/versions/{version}

    与 Store 一致：version 缺省为 ``all``（删光版本并删资产）；Bearer 须带 query user_id；
    系统管理员使用 X-System-Token，勿与 Bearer 同时传。
    """
    has_user = bool(user_token and user_token.strip())
    has_sys = bool(system_token and system_token.strip())
    if has_user == has_sys:
        raise RuntimeError("provide exactly one of user_token or system_token")
    if has_user and not (user_id and user_id.strip()):
        raise RuntimeError("user_id is required when using Bearer (see marketplace require_auth_with_user_id)")

    base = market_url.rstrip("/")
    ver_seg = (version or "all").strip()
    path_id = urllib.parse.quote(asset_id.strip(), safe="")
    path_ver = urllib.parse.quote(ver_seg, safe="")
    url = f"{base}/api/v1/plugins/{path_id}/versions/{path_ver}"
    params: dict[str, str] | None = None
    headers: dict[str, str] = {}
    if has_sys:
        headers["X-System-Token"] = system_token.strip()
    else:
        headers["Authorization"] = f"Bearer {user_token.strip()}"
        params = {"user_id": user_id.strip()}

    try:
        resp = requests.delete(url, headers=headers, params=params, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"request failed: {e}") from e
    if not resp.ok:
        raise RuntimeError(_upload_error_detail(resp))
    logger.info("deleted: asset_id=%s version=%s", asset_id, ver_seg)


def download_artifact_zip(
    market_url: str,
    asset_id: str,
    dest_path: Path,
) -> None:
    """
    GET /api/v1/artifacts/{asset_id}

    将响应体流式写入 ``dest_path``（应为 zip）。公开接口，不携带鉴权头。
    """
    base = market_url.rstrip("/")
    aid_seg = urllib.parse.quote(asset_id.strip(), safe="")
    url = f"{base}/api/v1/artifacts/{aid_seg}"

    try:
        resp = requests.get(url, stream=True, timeout=300)
    except requests.RequestException as e:
        raise RuntimeError(f"request failed: {e}") from e

    if not resp.ok:
        raise RuntimeError(_upload_error_detail(resp))

    dest_path = dest_path.resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)

    if not dest_path.read_bytes().startswith(b"PK"):
        try:
            dest_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            "response is not a zip file (missing ZIP magic); check URL or server error body"
        )
