"""Plugin publish, validation, and conflict handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import logging
import uuid
import zipfile
import hashlib
import re
from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException, status
import yaml
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

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


MAX_FILE_SIZE = 512 * 1024 * 1024
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")

# runtime.type values from design doc
RUNTIME_TOOLS = "tools"
RUNTIME_MCP_STDIO = "mcp-stdio"
RUNTIME_RESTFUL_API = "restful-api"
logger = logging.getLogger(__name__)


def _find_plugin_yaml_path(zf: zipfile.ZipFile) -> str | None:
    """
    Only accept the standard layout:
    <top>/plugin.yaml

    Where zip member path has exactly 2 segments and the last segment is plugin.yaml.
    """
    for name in zf.namelist():
        normalized = name.replace("\\", "/").strip("/")
        if not normalized:
            continue
        parts = normalized.split("/")
        if len(parts) == 2 and parts[-1] == "plugin.yaml":
            return name
    return None


def _plugin_prefix(plugin_yaml_path: str) -> str:
    """Return dir prefix of plugin.yaml (e.g. 'xxx/' or '' if at root)."""
    path = plugin_yaml_path.replace("\\", "/").strip("/")
    if "/" not in path:
        return ""
    return path.rsplit("/", 1)[0] + "/"


def _readme_path_in_zip(zf: zipfile.ZipFile, plugin_yaml_path: str) -> str | None:
    """Return zip member path for README.md in same dir as plugin.yaml."""
    prefix = _plugin_prefix(plugin_yaml_path)
    candidates = [prefix + "README.md"]
    names = set(zf.namelist())
    for c in candidates:
        if c in names:
            return c
    return None


def _icon_path_in_zip(zf: zipfile.ZipFile, plugin_yaml_path: str) -> str | None:
    """Return zip member path for icon.png in same dir as plugin.yaml."""
    prefix = _plugin_prefix(plugin_yaml_path)
    candidates = [prefix + "icon.png"]
    names = set(zf.namelist())
    for c in candidates:
        if c in names:
            return c
    return None


def _require_zip_member(names: set[str], path: str, *, message: str) -> None:
    if path not in names:
        raise PublishError(code=400, error="invalid_plugin_structure", message=message)


def _has_src_tree(names: set[str], prefix: str) -> bool:
    """True if there is at least one file under <prefix>src/."""
    p = prefix.replace("\\", "/")
    if p and not p.endswith("/"):
        p = p + "/"
    src_prefix = f"{p}src/"
    for n in names:
        n = n.replace("\\", "/")
        if n.startswith(src_prefix) and len(n) > len(src_prefix):
            return True
    return False


def _validate_tools_layout(zf: zipfile.ZipFile, plugin_yaml_path: str) -> dict[str, str]:
    """tools 类型：dist/*.whl、schemas/tools.json、icon.png、README.md。"""
    prefix = _plugin_prefix(plugin_yaml_path)
    names = set(zf.namelist())

    readme_path = _readme_path_in_zip(zf, plugin_yaml_path)
    if not readme_path:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message="插件包结构不符合要求：tools 类型缺少 README.md",
        )

    icon_path = _icon_path_in_zip(zf, plugin_yaml_path)
    if not icon_path:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message="插件包结构不符合要求：tools 类型缺少 icon.png",
        )

    tools_path = prefix + "schemas/tools.json"
    _require_zip_member(
        names,
        tools_path,
        message="插件包结构不符合要求：tools 类型缺少 dist/ 目录或 schemas/tools.json 文件",
    )

    dist_prefix = prefix + "dist/"
    whls = [n for n in names if n.startswith(dist_prefix) and n.lower().endswith(".whl")]
    if not whls:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message="插件包结构不符合要求：tools 类型缺少 dist/ 目录或 schemas/tools.json 文件",
        )

    return {
        "readme_path": readme_path,
        "icon_path": icon_path,
        "tools_path": tools_path,
    }


def _validate_src_based_layout(zf: zipfile.ZipFile, plugin_yaml_path: str) -> dict[str, str]:
    """mcp-stdio / restful-api：src/、icon.png、README.md。"""
    prefix = _plugin_prefix(plugin_yaml_path)
    names = set(zf.namelist())

    readme_path = _readme_path_in_zip(zf, plugin_yaml_path)
    if not readme_path:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message="插件包结构不符合要求：缺少 README.md",
        )

    icon_path = _icon_path_in_zip(zf, plugin_yaml_path)
    if not icon_path:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message="插件包结构不符合要求：缺少 icon.png",
        )

    if not _has_src_tree(names, prefix):
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message="插件包结构不符合要求：缺少 src/ 目录",
        )

    return {"readme_path": readme_path, "icon_path": icon_path}


def _normalize_runtime_type(raw: str) -> str:
    return (raw or "").strip()


@dataclass(frozen=True)
class PluginYamlPublicFields:
    """plugin.yaml 公共必填字段校验结果"""

    name: str
    display_name: str
    short_desc: str
    publisher_name: str
    tags: list[str]
    runtime_type: str


def _validate_plugin_yaml_public(data: dict[str, Any]) -> PluginYamlPublicFields:
    """校验 plugin.yaml 公共必填字段。"""
    name_raw = data.get("name")
    if not isinstance(name_raw, str) or not name_raw.strip():
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="plugin.yaml 配置文件格式错误或缺失：name 必填",
        )
    name = name_raw.strip()
    if not NAME_PATTERN.match(name):
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="plugin.yaml 中 name 必须符合 ^[a-z][a-z0-9-]*$",
        )

    display_name_raw = data.get("display_name")
    if not isinstance(display_name_raw, str) or not display_name_raw.strip():
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="plugin.yaml 配置文件格式错误或缺失：display_name 必填",
        )
    display_name = display_name_raw.strip()

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="plugin.yaml 配置文件格式错误或缺失：description 必填",
        )
    short_desc = description.strip()

    runtime = data.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("type"):
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="plugin.yaml 配置文件格式错误或缺失：runtime.type 必填",
        )
    runtime_type = _normalize_runtime_type(str(runtime.get("type")))

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="plugin.yaml 配置文件格式错误或缺失：metadata 必填",
        )
    author = metadata.get("author")
    if not isinstance(author, str) or not author.strip():
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="plugin.yaml 中 metadata.author 必填",
        )
    publisher_name = author.strip()

    tags: list[str] = []
    raw_tags = metadata.get("tags")
    if isinstance(raw_tags, list) and raw_tags:
        tags = [str(t).strip() for t in raw_tags if t is not None and str(t).strip()]
    elif raw_tags is None and metadata.get("tag") is not None:
        # 兼容旧字段 metadata.tag
        tag_raw = metadata.get("tag")
        if isinstance(tag_raw, list):
            tags = [str(t).strip() for t in tag_raw if t is not None and str(t).strip()]
        elif isinstance(tag_raw, str) and tag_raw.strip():
            tags = [tag_raw.strip()]

    if not tags:
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="plugin.yaml 中 metadata.tags 必须为非空字符串数组",
        )

    return PluginYamlPublicFields(
        name=name,
        display_name=display_name,
        short_desc=short_desc,
        publisher_name=publisher_name,
        tags=tags,
        runtime_type=runtime_type,
    )


def _validate_package_structure(
    zf: zipfile.ZipFile, plugin_yaml_path: str, runtime_type: str
) -> dict[str, str]:
    rt = runtime_type.lower()
    if rt == RUNTIME_TOOLS:
        return _validate_tools_layout(zf, plugin_yaml_path)
    if rt in (RUNTIME_MCP_STDIO.lower(), RUNTIME_RESTFUL_API.lower()):
        return _validate_src_based_layout(zf, plugin_yaml_path)
    raise PublishError(
        code=400,
        error="invalid_plugin_config",
        message=f"不支持的 runtime.type: {runtime_type!r}（支持 tools、mcp-stdio、restful-api）",
    )


def _extract_plugin_metadata(content: bytes) -> dict[str, Any]:
    """
    解析 zip：校验 plugin.yaml、包结构、README、icon。
    返回 name, version, display_name, short_desc, detail_desc, tags, publisher_name,
    plugin_type, icon_bytes。
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            plugin_yaml_path = _find_plugin_yaml_path(zf)
            if not plugin_yaml_path:
                raise PublishError(
                    code=400,
                    error="invalid_plugin_config",
                    message="plugin.yaml 配置文件格式错误或缺失",
                )

            raw = zf.read(plugin_yaml_path).decode("utf-8", errors="replace")
    except PublishError:
        raise
    except zipfile.BadZipFile as e:
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="上传文件不是有效的 ZIP 格式，请检查文件是否损坏或格式是否正确",
        ) from e
    except Exception as e:
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="无法解析插件包，请检查 zip 文件是否损坏",
        ) from e

    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message=f"plugin.yaml 配置文件格式错误或缺失: {e}",
        ) from e

    if not isinstance(data, dict):
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="plugin.yaml 配置文件格式错误或缺失",
        )

    yaml_public = _validate_plugin_yaml_public(data)

    version_raw = data.get("version")
    version = str(version_raw).strip() if version_raw is not None else ""

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            layout = _validate_package_structure(zf, plugin_yaml_path, yaml_public.runtime_type)
            detail_desc = zf.read(layout["readme_path"]).decode("utf-8", errors="replace")
            icon_bytes = zf.read(layout["icon_path"])
    except PublishError:
        raise
    except Exception as e:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message="无法读取插件包内文件，请确认目录结构正确",
        ) from e

    return {
        "name": yaml_public.name,
        "display_name": yaml_public.display_name,
        "version": version,
        "short_desc": yaml_public.short_desc,
        "detail_desc": detail_desc,
        "tags": yaml_public.tags,
        "publisher_name": yaml_public.publisher_name,
        "plugin_type": yaml_public.runtime_type,
        "icon_bytes": icon_bytes,
    }


def _normalize_version(version: str) -> str:
    """Normalize surrounding whitespace only; do not rewrite semantic content."""
    return version.strip()


def _validate_version(version: str) -> None:
    """Ensure version matches <major>.<minor>.<patch> (no v prefix)."""
    if not VERSION_PATTERN.match(version):
        raise PublishError(
            code=422,
            error="manifest_validation_failed",
            message=(
                "版本号格式错误，必须为 <主版本号>.<次版本号>.<修订号>，"
                "例如 1.0.0、1.0.1（不应有 v 前缀）"
            ),
        )


def _version_dir_prefix(publisher_id: str, asset_id: str, version: str) -> str:
    """Version directory key prefix: plugins/{publisher_id}/{asset_id}/{version}/"""
    return f"plugins/{publisher_id}/{asset_id}/{version}/"


def _build_storage_path(
    *,
    publisher_id: str,
    asset_id: str,
    version: str,
    asset_name: str,
) -> str:
    """Build object-key for zip: plugins/{publisher_id}/{asset_id}/{version}/{name}_{version}.zip"""
    prefix = _version_dir_prefix(publisher_id, asset_id, version)
    safe_name = asset_name.strip().replace(" ", "-")
    return f"{prefix}{safe_name}_{version}.zip"


def _compute_checksum(content: bytes) -> str:
    """SHA256 of content (for future client checksum comparison)."""
    return hashlib.sha256(content).hexdigest()


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
    return "uk_publisher_name" in low or (
        "unique" in low and "publisher_id" in low and "name" in low
    )


def _is_uk_asset_version_error(exc: IntegrityError) -> bool:
    msg = str(getattr(exc, "orig", None) or exc)
    low = msg.lower()
    return "uk_asset_version" in low or (
        "unique" in low and "asset_id" in low and "version" in low
    )


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

    meta = _extract_plugin_metadata(content)
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
            raise PublishError(
                code=409,
                error="plugin_name_exists",
                message=f"您已发布过同名插件 '{name}'，请使用其他名称或为现有插件添加新版本",
                data={"expected_plugin_id": matches[0].asset_id},
            )
        asset_id = uuid.uuid4().hex
        existing_asset = None
    existing_version = version_repo.get_version(asset_id=asset_id, version=version)

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

    version_dir = _version_dir_prefix(user_id, asset_id, version)
    zip_key = _build_storage_path(
        publisher_id=user_id,
        asset_id=asset_id,
        version=version,
        asset_name=name,
    )
    file_path = version_dir

    upload_result = storage.upload_bytes(content, zip_key)
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
    rows, total = repo.list_plugins(query)
    logger.info("List plugins query done: total=%s rows=%s", total, len(rows))
    items = []
    for asset, latest_file_path in rows:
        item = PluginListItem.model_validate(asset)
        item.icon_uri = _icon_presigned_url_from_file_path(storage, latest_file_path)
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
    auth: tuple,
    db: Session,
    storage: S3StorageClient,
) -> PluginVersionDeleteData:
    logger.info("Delete plugin version request: asset_id=%s version=%s", asset_id, version)
    is_admin, acting_user_id = auth
    asset_repo = MarketAssetRepository(db)
    version_repo = MarketAssetVersionRepository(db)

    asset = asset_repo.get_by_asset_id(asset_id)
    if not asset:
        logger.warning("Delete plugin version failed: asset not found, asset_id=%s", asset_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    if not is_admin and acting_user_id and asset.publisher_id != acting_user_id:
        logger.warning(
            "Delete plugin version forbidden: asset_id=%s acting_user_id=%s publisher_id=%s",
            asset_id,
            acting_user_id,
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


def _build_artifact_key(publisher_id: str, asset_id: str, version: str, name: str) -> str:
    safe_name = name.strip().replace(" ", "-")
    return f"plugins/{publisher_id}/{asset_id}/{version}/{safe_name}_{version}.zip"


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
    db: Session,
    storage: S3StorageClient,
    fetch_user_id: str | None = None,
) -> PluginDownloadData:
    """Resolve latest artifact and return public download info."""
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

    stat = storage.get_object_size_and_sha256(key)
    if not stat.get("success"):
        raise PublishError(
            code=500,
            error="storage_error",
            message=f"读取插件包元数据失败: {stat.get('error', 'unknown')}",
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
        file_size=int(stat.get("size", 0)),
        checksum_sha256=str(stat.get("checksum_sha256", "")),
    )
