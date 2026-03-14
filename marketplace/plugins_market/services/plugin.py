"""插件相关业务逻辑：发布、校验、冲突处理等。"""

from datetime import datetime, timezone
import io
import uuid
import zipfile

from sqlalchemy.orm import Session

from plugins_market.repositories import (
    MarketAssetRepository,
    MarketAssetVersionRepository,
)
from plugins_market.schemas.plugin import AssetCreate, AssetVersionCreate, PluginPublishResult


MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


class PublishError(Exception):
    """发布过程业务错误，由 Router 转为 HTTPException。"""

    def __init__(self, status_code: int, detail: dict):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"PublishError {status_code}: {detail.get('error', '')}")


def _parse_manifest_from_zip(content: bytes) -> str:
    """
    从 zip 字节流中解析 plugin.yaml，返回其文本内容。
    若未找到或解压失败，抛出 PublishError。
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            plugin_yaml_name = None
            for name in zf.namelist():
                parts = name.split("/")
                if len(parts) == 2 and parts[1] == "plugin.yaml":
                    plugin_yaml_name = name
                    break

            if not plugin_yaml_name:
                raise PublishError(
                    422,
                    {
                        "code": 422,
                        "data": None,
                        "error": "manifest_validation_failed",
                        "message": "plugin.yaml 文件缺失或路径不正确",
                    },
                )

            raw = zf.read(plugin_yaml_name).decode("utf-8")
            return raw
    except PublishError:
        raise
    except Exception as e:
        raise PublishError(
            422,
            {
                "code": 422,
                "data": None,
                "error": "manifest_validation_failed",
                "message": "无法解析插件包，请检查 zip 文件是否损坏",
            },
        ) from e


def _parse_name_version_from_yaml(raw: str) -> tuple[str | None, str | None]:
    """极简 YAML 解析：只抽取 name / version。"""
    manifest_name: str | None = None
    manifest_version: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            manifest_name = stripped.split(":", 1)[1].strip().strip("'\"")
        elif stripped.startswith("version:"):
            manifest_version = stripped.split(":", 1)[1].strip().strip("'\"")
    return manifest_name, manifest_version


def publish(
    *,
    space_id: str,
    content: bytes,
    filename: str | None,
    plugin_id: str | None,
    plugin_version: str | None,
    version_desc: str | None,
    force: bool,
    db: Session,
) -> PluginPublishResult:
    """
    执行插件发布：校验 → 冲突检测 → 写主表/版本表 → 返回结果。
    校验或冲突时抛出 PublishError，由调用方转为 HTTP 响应。
    """
    if not filename or not filename.lower().endswith(".zip"):
        raise PublishError(
            400,
            {
                "code": 400,
                "data": None,
                "error": "invalid_file_format",
                "message": "仅支持 .zip 格式的插件包文件",
            },
        )

    if len(content) > MAX_FILE_SIZE:
        raise PublishError(
            413,
            {
                "code": 413,
                "data": None,
                "error": "file_too_large",
                "message": "文件大小超过限制（最大100MB）",
            },
        )

    raw = _parse_manifest_from_zip(content)
    manifest_name, manifest_version = _parse_name_version_from_yaml(raw)

    if not manifest_name:
        raise PublishError(
            422,
            {
                "code": 422,
                "data": None,
                "error": "manifest_validation_failed",
                "message": "plugin.yaml 文件格式错误：缺少必需的 name 字段",
            },
        )

    if plugin_version is None:
        if not manifest_version:
            raise PublishError(
                422,
                {
                    "code": 422,
                    "data": None,
                    "error": "manifest_validation_failed",
                    "message": "plugin.yaml 文件格式错误：缺少必需的 version 字段",
                },
            )
        version = manifest_version
    else:
        version = plugin_version

    asset_id = plugin_id or f"asset_{uuid.uuid4().hex}"
    name = manifest_name

    asset_repo = MarketAssetRepository(db)
    version_repo = MarketAssetVersionRepository(db)

    existing_asset = asset_repo.get_by_asset_id(asset_id)
    existing_version = version_repo.get_version(asset_id=asset_id, version=version)

    if existing_version and not force:
        raise PublishError(
            409,
            {
                "code": 409,
                "data": {
                    "existing_plugin": {
                        "plugin_id": existing_asset.asset_id if existing_asset else asset_id,
                        "version": existing_version.version,
                    }
                },
                "error": "version_conflict",
                "message": f"插件 '{name}' 版本 '{version}' 已存在，如需覆盖请设置 force=true",
            },
        )

    if not existing_asset:
        asset_params = AssetCreate(
            asset_id=asset_id,
            name=name,
            asset_type="plugin",
            short_desc=version_desc,
            publisher_id=space_id,
            publisher_name="",
            category_id="",
            latest_version=version,
        )
        asset = asset_repo.create_asset(asset_params)
    else:
        asset = asset_repo.update_latest_version(existing_asset, version)

    version_id = (
        existing_version.version_id if (existing_version and force) else f"ver_{uuid.uuid4().hex}"
    )

    if existing_version and force:
        version_repo.update(
            existing_version,
            {
                "changelog": version_desc,
                "status": "pending_review",
            },
        )
        version_row = version_repo.get_version(asset_id=asset_id, version=version)
    else:
        version_params = AssetVersionCreate(
            version_id=version_id,
            asset_id=asset_id,
            version=version,
            source_publish_id=space_id,
            changelog=version_desc,
            status="pending_review",
        )
        version_row = version_repo.create_version(version_params)

    storage_url = f"openjiuwen-market/plugins/{space_id}/{asset_id}/plugin.zip"
    published_at = datetime.fromtimestamp(
        (version_row.create_time or asset.create_time) / 1000, tz=timezone.utc
    ).isoformat()

    return PluginPublishResult(
        plugin_id=asset.asset_id,
        name=asset.name,
        version=version_row.version,
        status=version_row.status or "pending_review",
        published_at=published_at,
        storage_url=storage_url,
    )
