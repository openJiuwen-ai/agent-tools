"""Plugin publish, validation, and conflict handling."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
import re
import uuid
from urllib.parse import urlparse
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from plugins_market.core.auth import AuthContext
from plugins_market.core.errors import PublishError
from plugins_market.core.s3_storage_client import S3StorageClient
from plugins_market.models.market_assets import MarketAssetDB, MarketAssetVersionDB
from plugins_market.repositories import (
    MarketAssetRepository,
    MarketAssetVersionRepository,
    PluginFetchRecordRepository,
)
from plugins_market.schemas.plugin import (
    PluginDownloadData,
    PluginListItem,
    PluginListQuery,
    PluginListResponse,
    PluginPublishResult,
    PluginVersionDeleteData,
    PluginVersionDetail,
)
from plugins_market.validation import extract_plugin_metadata
from plugins_market.validation.constants import (
    MAX_FILE_SIZE,
    MARKET_ASSET_SHORT_DESC_MAX_LEN,
    VERSION_PATTERN,
)

logger = logging.getLogger(__name__)


def _normalize_version(version: str) -> str:
    """Normalize surrounding whitespace only; do not rewrite semantic content."""
    return version.strip()


def _validate_version(version: str) -> None:
    """Ensure version matches <major>.<minor>.<patch> (no v prefix)."""
    if not VERSION_PATTERN.match(version):
        raise PublishError(
            code=422,
            error="manifest_validation_failed",
            message=("版本号格式错误，必须为 <主版本号>.<次版本号>.<修订号>，" "例如 1.0.0、1.0.1（不应有 v 前缀）"),
        )


def _storage_root(plugin_type: str | None) -> str:
    """Top-level OBS prefix: skills for skill type, plugins for everything else."""
    return "skills" if (plugin_type or "").lower() == "skill" else "plugins"


def _version_dir_prefix(publisher_id: str, asset_id: str, version: str, plugin_type: str | None = None) -> str:
    """Version directory key prefix: {root}/{publisher_id}/{asset_id}/{version}/"""
    root = _storage_root(plugin_type)
    return f"{root}/{publisher_id}/{asset_id}/{version}/"


def _build_storage_path(
    *,
    publisher_id: str,
    asset_id: str,
    version: str,
    asset_name: str,
    plugin_type: str | None = None,
) -> str:
    """Build object-key for zip: {root}/{publisher_id}/{asset_id}/{version}/{name}_{version}.zip"""
    prefix = _version_dir_prefix(publisher_id, asset_id, version, plugin_type)
    safe_name = asset_name.strip().replace(" ", "-")
    return f"{prefix}{safe_name}_{version}.zip"


def _compute_checksum(content: bytes) -> str:
    """SHA256 of content (for future client checksum comparison)."""
    return hashlib.sha256(content).hexdigest()


def _publish_idempotent_same_artifact(
    existing_version: MarketAssetVersionDB | None,
    computed_sha256: str,
) -> bool:
    """同一 asset + version 且库内已记录相同子包 SHA-256 时跳过写存储（幂等重试）。"""
    if existing_version is None:
        return False
    if (existing_version.status or "").upper() != "ACTIVE":
        return False
    stored = (existing_version.artifact_sha256 or "").strip()
    if not stored:
        return False
    return stored.lower() == computed_sha256.lower()


def _make_publish_result(
    asset: MarketAssetDB,
    version_row: MarketAssetVersionDB,
    zip_key: str,
) -> PluginPublishResult:
    ts_ms = version_row.create_time or asset.create_time or 0
    published_at = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
    return PluginPublishResult(
        plugin_id=asset.asset_id,
        name=asset.name,
        version=version_row.version,
        status=version_row.status or "ACTIVE",
        published_at=published_at,
        storage_url=zip_key,
    )


def _semver_sort_key(version: str | None) -> tuple[int, int, int]:
    """Parse x.y.z for ordering; invalid/missing sorts last."""
    v = (version or "").strip()
    if not v or not VERSION_PATTERN.match(v):
        return (-1, -1, -1)
    a, b, c = v.split(".", 2)
    return (int(a), int(b), int(c))


def _render_cumulative_changelog_file(versions: list[MarketAssetVersionDB]) -> str:
    """
    Build UTF-8 text for changelog.log: all historical version rows.
    Order by semver descending (largest version first).
    Plain style: [version] then blank line then changelog (same as API version_desc).
    """
    if not versions:
        return "（暂无版本记录）\n"

    ordered = sorted(
        versions,
        key=lambda r: _semver_sort_key(r.version),
        reverse=True,
    )
    blocks: list[str] = []
    for row in ordered:
        ver = (row.version or "").strip() or "未知"
        body = (row.changelog or "").strip() or "（无变更说明）"
        blocks.append(f"[{ver}]\n\n{body}")

    return "\n\n".join(blocks) + "\n"


def _is_uk_publisher_name_error(exc: IntegrityError) -> bool:
    msg = str(getattr(exc, "orig", None) or exc)
    low = msg.lower()
    return "uk_publisher_name" in low or ("unique" in low and "publisher_id" in low and "name" in low)


def _is_uk_asset_version_error(exc: IntegrityError) -> bool:
    msg = str(getattr(exc, "orig", None) or exc)
    low = msg.lower()
    return "uk_asset_version" in low or ("unique" in low and "asset_id" in low and "version" in low)


def publish(
    *,
    user_id: str,
    content: bytes,
    filename: str | None,
    expected_checksum: str,
    plugin_id: str | None,
    plugin_version: str | None,
    version_desc: str | None,
    force: bool,
    db: Session,
    storage: S3StorageClient,
) -> PluginPublishResult:
    """Validate, resolve conflicts, upload to S3, write asset/version, return result. Raises PublishError on failure."""
    if not filename or not filename.lower().endswith(".zip"):
        raise PublishError(
            code=400,
            error="invalid_file_format",
            message="仅支持 .zip 格式的插件包文件",
        )

    if len(content) > MAX_FILE_SIZE:
        raise PublishError(
            code=413,
            error="file_too_large",
            message="文件大小超过限制（最大512MB）",
        )

    computed = _compute_checksum(content)
    if computed != expected_checksum.lower():
        raise PublishError(
            code=400,
            error="checksum_mismatch",
            message="文件校验和不匹配，文件可能在传输过程中损坏",
        )

    if len(content) < 2 or content[:2] != b"PK":
        raise PublishError(
            code=400,
            error="invalid_file_format",
            message="仅支持 .zip 格式的插件包文件",
        )

    meta = extract_plugin_metadata(content)
    content_size = len(content)
    name = (meta["name"] or "").strip()
    display_name = (meta.get("display_name") or "").strip()
    manifest_version = (meta["version"] or "").strip()

    if not name:
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="plugin.yaml 配置文件格式错误或缺失：缺少必需的 name 字段",
        )

    if plugin_version is None:
        if not manifest_version:
            raise PublishError(
                code=400,
                error="invalid_plugin_config",
                message="plugin.yaml 配置文件格式错误或缺失：缺少必需的 version 字段",
            )
        version = _normalize_version(manifest_version)
    else:
        version = _normalize_version(plugin_version)

    _validate_version(version)

    short_desc = meta.get("short_desc")
    if isinstance(short_desc, str) and len(short_desc) > MARKET_ASSET_SHORT_DESC_MAX_LEN:
        short_desc = short_desc[:MARKET_ASSET_SHORT_DESC_MAX_LEN]
    detail_desc = meta.get("detail_desc")
    tags = meta.get("tags") or []
    publisher_name = meta.get("publisher_name") or ""
    plugin_type = meta.get("plugin_type")
    icon_bytes = meta.get("icon_bytes")

    asset_repo = MarketAssetRepository(db)
    version_repo = MarketAssetVersionRepository(db)

    pid = (plugin_id or "").strip()
    if pid:
        existing_asset = asset_repo.get_by_asset_id(pid)
        if not existing_asset:
            raise PublishError(
                code=404,
                error="plugin_not_found",
                message=f"插件 '{pid}' 不存在，无法添加新版本",
            )
        if existing_asset.publisher_id != user_id:
            raise PublishError(
                code=403,
                error="permission_denied",
                message="您无权限操作该插件",
            )
        by_name = asset_repo.list_by_publisher_name_and_type(user_id, name, "plugin")
        if len(by_name) == 1 and by_name[0].asset_id != pid:
            raise PublishError(
                code=422,
                error="plugin_id_mismatch",
                message=f"plugin_id 与插件包不匹配：您填写的 plugin_id='{pid}' 与插件名称 '{name}' 对应的插件id不一致",
                data={"expected_plugin_id": by_name[0].asset_id},
            )
        if len(by_name) > 1 and pid not in {m.asset_id for m in by_name}:
            raise PublishError(
                code=422,
                error="plugin_id_mismatch",
                message=f"plugin_id 与插件包不匹配：您填写的 plugin_id='{pid}' 与插件名称 '{name}' 对应的插件id不一致，请从同名候选中选择正确的 plugin_id",
                data={"ambiguous_plugin_ids": [m.asset_id for m in by_name]},
            )
        asset_id = pid
    else:
        matches = asset_repo.list_by_publisher_name_and_type(user_id, name, "plugin")
        if len(matches) > 1:
            raise PublishError(
                code=422,
                error="manifest_validation_failed",
                message=f"存在多个同名插件 '{name}'，请通过 plugin_id 指定要发布版本的插件",
                data={"ambiguous_plugin_ids": [m.asset_id for m in matches]},
            )
        if len(matches) == 1:
            # 同发布者 + 包内 name 唯一定位一条插件：不传 plugin_id 也可发新版 / 幂等重试
            existing_asset = matches[0]
            asset_id = existing_asset.asset_id
        else:
            asset_id = uuid.uuid4().hex
            existing_asset = None
    existing_version = version_repo.get_version(asset_id=asset_id, version=version)

    version_dir = _version_dir_prefix(user_id, asset_id, version, plugin_type)
    zip_key = _build_storage_path(
        publisher_id=user_id,
        asset_id=asset_id,
        version=version,
        asset_name=name,
        plugin_type=plugin_type,
    )
    file_path = version_dir

    if existing_version and _publish_idempotent_same_artifact(existing_version, computed):
        asset_for_result = existing_asset if existing_asset is not None else asset_repo.get_by_asset_id(asset_id)
        if asset_for_result is None:
            raise PublishError(
                code=500,
                error="internal_error",
                message="发布幂等校验失败：缺少插件主记录",
            )
        logger.info(
            "publish idempotent skip (same version + artifact_sha256): asset_id=%s version=%s",
            asset_id,
            version,
        )
        return _make_publish_result(asset_for_result, existing_version, zip_key)

    if existing_version and not force:
        raise PublishError(
            code=409,
            error="version_conflict",
            message=f"插件 '{name}' 版本 '{version}' 已存在，如需覆盖请设置 force=true",
            data={
                "existing_plugin": {
                    "plugin_id": existing_asset.asset_id if existing_asset else asset_id,
                    "version": existing_version.version,
                }
            },
        )

    # 写入校验和/大小到对象 metadata，避免下载时读全量对象重复计算
    upload_result = storage.upload_bytes(
        content,
        zip_key,
        metadata={"sha256": computed, "size": str(content_size)},
    )
    if not upload_result.get("success"):
        raise PublishError(
            code=500,
            error="storage_error",
            message=upload_result.get("error", "插件包上传失败"),
        )

    icon_key = f"{version_dir}icon.png"
    r = storage.upload_bytes(icon_bytes, icon_key)
    if not r.get("success"):
        raise PublishError(
            code=500,
            error="storage_error",
            message=r.get("error", "插件图标上传失败"),
        )

    if detail_desc is not None:
        readme_key = f"{version_dir}readme.md"
        r = storage.upload_bytes(detail_desc.encode("utf-8"), readme_key)
        if not r.get("success"):
            raise PublishError(
                code=500,
                error="storage_error",
                message=r.get("error", "插件 README 上传失败"),
            )

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    try:
        if not existing_asset:
            # 新建插件：插入主表 + 版本表（同一事务）
            asset_obj = MarketAssetDB(
                asset_id=asset_id,
                asset_type="plugin",
                name=name,
                display_name=display_name,
                short_desc=short_desc,
                detail_desc=detail_desc,
                publisher_id=user_id,
                publisher_name=publisher_name,
                tags=tags if tags else None,
                status="PUBLISHED",
                plugin_type=plugin_type,
                latest_version=version,
                create_time=now_ms,
                update_time=now_ms,
            )
            version_obj = MarketAssetVersionDB(
                version_id=uuid.uuid4().hex,
                asset_id=asset_id,
                version=version,
                changelog=version_desc,
                status="ACTIVE",
                create_time=now_ms,
                file_path=file_path,
                artifact_sha256=computed,
            )
            db.add(asset_obj)
            db.add(version_obj)
            db.commit()
            db.refresh(asset_obj)
            db.refresh(version_obj)
            asset = asset_obj
            version_row = version_obj
        else:
            # 已有插件：更新主表 + 新增或覆盖版本
            existing_asset.name = name
            existing_asset.display_name = display_name
            existing_asset.latest_version = version
            existing_asset.update_time = now_ms
            existing_asset.short_desc = short_desc
            existing_asset.detail_desc = detail_desc
            existing_asset.tags = tags if tags else None
            existing_asset.publisher_name = publisher_name
            existing_asset.plugin_type = plugin_type

            if existing_version and force:
                existing_version.changelog = version_desc
                existing_version.status = "ACTIVE"
                existing_version.file_path = file_path
                existing_version.artifact_sha256 = computed
                version_row = existing_version
            else:
                version_row = MarketAssetVersionDB(
                    version_id=uuid.uuid4().hex,
                    asset_id=asset_id,
                    version=version,
                    changelog=version_desc,
                    status="ACTIVE",
                    create_time=now_ms,
                    file_path=file_path,
                    artifact_sha256=computed,
                )
                db.add(version_row)
            db.add(existing_asset)
            db.commit()
            db.refresh(existing_asset)
            db.refresh(version_row)
            asset = existing_asset

    except IntegrityError as e:
        db.rollback()
        if _is_uk_publisher_name_error(e):
            raise PublishError(
                code=409,
                error="plugin_name_exists",
                message=f"您已发布过同名插件 '{name}'，请使用其他名称或为现有插件添加新版本",
            ) from e
        if _is_uk_asset_version_error(e):
            raise PublishError(
                code=409,
                error="version_exists",
                message=f"插件版本 '{version}' 已存在，如需覆盖请设置 force=true",
                data={"existing_version": version},
            ) from e
        raise

    # Cumulative changelog for this release dir
    all_versions = version_repo.list_versions_chronological(asset_id)
    changelog_text = _render_cumulative_changelog_file(all_versions)
    cl_key = f"{version_dir}changelog.log"
    r = storage.upload_bytes(changelog_text.encode("utf-8"), cl_key)
    if not r.get("success"):
        raise PublishError(
            code=500,
            error="storage_error",
            message=r.get("error", "插件 changelog.log 上传失败"),
        )

    storage_url = zip_key
    published_at = datetime.fromtimestamp(
        (version_row.create_time or asset.create_time) / 1000, tz=timezone.utc
    ).isoformat()

    return PluginPublishResult(
        plugin_id=asset.asset_id,
        name=asset.name,
        version=version_row.version,
        status=version_row.status or "ACTIVE",
        published_at=published_at,
        storage_url=storage_url,
    )


def _icon_presigned_url_from_file_path(
    storage: S3StorageClient,
    file_path: str | None,
) -> str | None:
    """图标固定为版本目录下 icon.png，与 file_path 拼出对象 Key 后预签名。"""
    prefix = _version_prefix_from_file_path(storage, file_path)
    if not prefix:
        return None
    icon_key = f"{prefix}icon.png"
    try:
        return storage.presigned_get_url(icon_key)
    except Exception as e:
        logger.warning("预签名图标链接失败 key=%s: %s", icon_key, e)
        return None


def list_plugins_service(
    query: PluginListQuery,
    db: Session,
    storage: S3StorageClient,
) -> PluginListResponse:
    logger.info(
        "List plugins request: page=%s page_size=%s asset_id=%s publisher_id=%s plugin_type=%s order_by=%s desc=%s",
        query.page,
        query.page_size,
        query.asset_id,
        query.publisher_id,
        query.plugin_type,
        query.order_by,
        query.desc,
    )
    repo = MarketAssetRepository(db)
    version_repo = MarketAssetVersionRepository(db)
    rows, total = repo.list_plugins(query)
    logger.info("List plugins query done: total=%s rows=%s", total, len(rows))
    asset_ids = [a.asset_id for a, _ in rows]
    versions_by_asset = version_repo.list_version_strings_by_asset_ids(asset_ids)
    items = []
    for asset, latest_file_path in rows:
        item = PluginListItem.model_validate(asset)
        item.icon_uri = _icon_presigned_url_from_file_path(storage, latest_file_path)
        item.all_versions = versions_by_asset.get(asset.asset_id, [])
        items.append(item)
    return PluginListResponse(
        page=query.page,
        page_size=query.page_size,
        total=total,
        items=items,
    )


def get_plugin_version_detail_service(
    asset_id: str,
    version: str,
    db: Session,
    storage: S3StorageClient,
) -> PluginVersionDetail:
    logger.info("Get plugin version detail request: asset_id=%s version=%s", asset_id, version)
    asset_repo = MarketAssetRepository(db)
    version_repo = MarketAssetVersionRepository(db)

    asset = asset_repo.get_by_asset_id(asset_id)
    if not asset:
        logger.warning("Get plugin version detail failed: asset not found, asset_id=%s", asset_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")

    version_row = version_repo.get_version(asset_id=asset_id, version=version)
    if not version_row:
        logger.warning(
            "Get plugin version detail failed: version not found, asset_id=%s version=%s",
            asset_id,
            version,
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")

    return PluginVersionDetail(
        asset_id=asset.asset_id,
        version=version_row.version,
        asset_type=asset.asset_type,
        plugin_type=asset.plugin_type,
        name=asset.name,
        display_name=asset.display_name,
        short_desc=asset.short_desc,
        detail_desc=asset.detail_desc,
        publisher_id=asset.publisher_id,
        publisher_name=asset.publisher_name,
        tags=asset.tags,
        certification=asset.certification,
        changelog=version_row.changelog,
        file_path=version_row.file_path,
        icon_uri=_icon_presigned_url_from_file_path(storage, version_row.file_path),
    )


def _key_from_object_uri(storage: Any, uri_or_key: str | None) -> str | None:
    if not uri_or_key:
        return None
    raw = uri_or_key.strip()
    if not raw:
        return None
    if "://" not in raw:
        return raw
    try:
        p = urlparse(raw)
        path = (p.path or "").lstrip("/")
        bucket = getattr(getattr(storage, "config", None), "bucket_name", None)
        if bucket and path.startswith(f"{bucket}/"):
            return path[len(bucket) + 1:]
        return path
    except Exception:
        return None


def _version_prefix_from_file_path(storage: Any, file_path: str | None) -> str | None:
    prefix = _key_from_object_uri(storage, file_path)
    if not prefix:
        return None
    prefix = prefix.strip()
    return prefix if prefix.endswith("/") else prefix + "/"


def delete_plugin_version_service(
    asset_id: str,
    version: str,
    auth: AuthContext,
    db: Session,
    storage: S3StorageClient,
) -> PluginVersionDeleteData:
    logger.info("Delete plugin version request: asset_id=%s version=%s", asset_id, version)
    asset_repo = MarketAssetRepository(db)
    version_repo = MarketAssetVersionRepository(db)

    asset = asset_repo.get_by_asset_id(asset_id)
    if not asset:
        logger.warning("Delete plugin version failed: asset not found, asset_id=%s", asset_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    if not auth.is_admin and auth.acting_user_id and asset.publisher_id != auth.acting_user_id:
        logger.warning(
            "Delete plugin version forbidden: asset_id=%s acting_user_id=%s publisher_id=%s",
            asset_id,
            auth.acting_user_id,
            asset.publisher_id,
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")

    prefixes: list[str] = []

    if version.strip().lower() == "all":
        logger.info("Delete all versions for asset_id=%s", asset_id)
        versions = version_repo.list_versions(asset_id)
        if not versions:
            logger.warning("Delete all versions failed: no versions found, asset_id=%s", asset_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No versions found for asset")
        for v in versions:
            p = _version_prefix_from_file_path(storage, v.file_path)
            if p:
                prefixes.append(p)
        version_repo.delete_all_versions(asset_id)
        asset_repo.delete_asset(asset_id)
        logger.info("Delete all versions done: asset deleted, asset_id=%s", asset_id)
    else:
        logger.info("Delete single version: asset_id=%s version=%s", asset_id, version)
        version_row = version_repo.get_version(asset_id=asset_id, version=version)
        if not version_row:
            logger.warning(
                "Delete single version failed: version not found, asset_id=%s version=%s",
                asset_id,
                version,
            )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
        p = _version_prefix_from_file_path(storage, version_row.file_path)
        if p:
            prefixes.append(p)
        version_repo.delete_version(asset_id, version)
        if version_repo.count_versions(asset_id) == 0:
            asset_repo.delete_asset(asset_id)
            logger.info("Delete single version done: no versions left, asset deleted, asset_id=%s", asset_id)
        else:
            remaining = version_repo.list_versions(asset_id)
            if remaining:
                new_latest = remaining[0].version
                fresh_asset = asset_repo.get_by_asset_id(asset_id)
                if fresh_asset:
                    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                    asset_repo.update(
                        fresh_asset,
                        {"latest_version": new_latest, "update_time": now_ms},
                    )
                    logger.info(
                        "Delete single version done: latest_version updated, asset_id=%s latest_version=%s",
                        asset_id,
                        new_latest,
                    )

    for p in prefixes:
        dr = storage.delete_prefix(p)
        if not dr.get("success"):
            logger.error(
                "Delete storage prefix failed: asset_id=%s version=%s prefix=%s errors=%s",
                asset_id,
                version,
                p,
                dr.get("errors", []),
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "message": "Object storage delete failed",
                    "prefix": p,
                    "errors": dr.get("errors", []),
                },
            )
        logger.info("Delete storage prefix success: asset_id=%s prefix=%s", asset_id, p)

    logger.info("Delete plugin version success: asset_id=%s version=%s", asset_id, version)
    return PluginVersionDeleteData(asset_id=asset_id, version=version)


def _build_artifact_key(
    publisher_id: str,
    asset_id: str,
    version: str,
    name: str,
    plugin_type: str | None = None,
) -> str:
    safe_name = name.strip().replace(" ", "-")
    root = _storage_root(plugin_type)
    return f"{root}/{publisher_id}/{asset_id}/{version}/{safe_name}_{version}.zip"


def _resolve_latest_version_for_download(
    *,
    asset_id: str,
    latest_version: str | None,
    version_repo: MarketAssetVersionRepository,
):
    if latest_version:
        row = version_repo.get_version(asset_id=asset_id, version=latest_version)
        if row:
            return row
    return version_repo.get_latest_version(asset_id=asset_id)


def get_download_info(
    *,
    asset_id: str,
    version: str | None = None,
    db: Session,
    storage: S3StorageClient,
    fetch_user_id: str | None = None,
) -> PluginDownloadData:
    """根据 asset_id（可选 version）返回预签名下载信息。"""
    asset_repo = MarketAssetRepository(db)
    version_repo = MarketAssetVersionRepository(db)
    fetch_repo = PluginFetchRecordRepository(db)

    asset = asset_repo.get_by_asset_id(asset_id)
    if not asset:
        raise PublishError(
            code=404,
            error="plugin_not_found",
            message=f"插件 '{asset_id}' 不存在",
        )

    version = (version or "").strip() or None
    if version is not None:
        if not VERSION_PATTERN.match(version):
            raise PublishError(
                code=422,
                error="invalid_version",
                data={"version": version},
                message="version 参数格式错误，应为 x.y.z（如 1.0.0）",
            )
        version_row = version_repo.get_version(asset_id=asset.asset_id, version=version)
        if not version_row:
            raise PublishError(
                code=404,
                error="version_not_found",
                data={"asset_id": asset.asset_id, "version": version},
                message=f"插件 '{asset.name}' 不存在版本 '{version}'",
            )
    else:
        version_row = _resolve_latest_version_for_download(
            asset_id=asset.asset_id,
            latest_version=asset.latest_version,
            version_repo=version_repo,
        )
    if not version_row:
        raise PublishError(
            code=404,
            error="plugin_not_found",
            message=f"插件 '{asset.asset_id}' 暂无可下载版本",
        )

    key = _build_artifact_key(
        publisher_id=asset.publisher_id,
        asset_id=asset.asset_id,
        version=version_row.version,
        name=asset.name,
        plugin_type=asset.plugin_type,
    )

    head = storage.head_object(key)
    if not head.get("success"):
        if head.get("not_found"):
            raise PublishError(
                code=404,
                error="version_deleted",
                message="插件版本已被删除",
            )
        raise PublishError(
            code=500,
            error="storage_error",
            message=f"读取插件包元数据失败: {head.get('error', 'unknown')}",
        )

    download_url = storage.presigned_get_url(key)

    # 下载元数据只从 HEAD + x-amz-meta-* 读取，避免读全量对象
    metadata = head.get("metadata") or {}
    checksum_sha256 = str(metadata.get("sha256") or "").strip()

    size_meta = str(metadata.get("size") or "").strip()
    size: int | None = None
    if size_meta:
        try:
            size = int(size_meta)
        except ValueError:
            size = None
    if size is None:
        # 兜底取 ContentLength（仍为 HEAD，无需读取对象 body）
        try:
            size = int(head.get("size")) if head.get("size") is not None else None
        except Exception:
            size = None

    if size is None or not checksum_sha256:
        raise PublishError(
            code=500,
            error="storage_error",
            message="插件包对象缺少必要的元数据（sha256/size），请重新发布该版本",
        )

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    try:
        updated_rows = asset_repo.increase_install_count_atomic(
            asset_id=asset.asset_id,
            now_ms=now_ms,
        )
        if updated_rows != 1:
            raise PublishError(
                code=500,
                error="db_error",
                message=f"更新下载统计失败：asset_id={asset.asset_id}",
            )

        fetch_repo.create_fetch_record(
            asset_id=asset.asset_id,
            version_id=version_row.version_id,
            fetch_user_id=fetch_user_id,
            create_time=now_ms,
        )
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise PublishError(
            code=500,
            error="db_error",
            message="更新下载统计失败",
        ) from e

    return PluginDownloadData(
        download_url=download_url,
        asset_id=asset.asset_id,
        name=asset.name,
        version=version_row.version,
        file_size=int(size),
        checksum_sha256=checksum_sha256,
    )
