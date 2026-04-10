import asyncio
import hashlib
import tempfile
import time
from collections import deque
from pathlib import Path as FsPath
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from common.security.security_utils import SecurityUtils
from plugins_market.core.auth import AuthContext, get_gitcode_user_id, require_auth
from plugins_market.core.config import settings
from plugins_market.core.database import get_db
from plugins_market.core.s3_storage_client import get_storage_client
from plugins_market.repositories import (
    MarketAssetRepository,
    MarketAssetVersionRepository,
)
from plugins_market.validation.constants import MAX_FILE_SIZE, ZIP_STREAM_READ_CHUNK_BYTES
from plugins_market.imports.skill_import_service import skill_import_from_bundle
from plugins_market.schemas.common import ResponseModel
from plugins_market.schemas.plugin import (
    PluginDownloadData,
    PluginListItem,
    PluginListQuery,
    PluginListResponse,
    PluginPublishForm,
    PluginPublishResult,
    PluginTemplatePresignData,
    PluginVersionDeleteData,
    PluginVersionDetail,
    SkillImportBundle,
    SkillImportResponse,
)
from plugins_market.services import (
    PublishError,
    delete_plugin_version_service,
    get_plugin_version_detail_service,
    list_plugins_service,
    get_download_info,
    publish as plugin_publish,
)

plugin_router = APIRouter(prefix="/plugins", tags=["plugins"])
artifact_router = APIRouter(prefix="/artifacts", tags=["plugins"])

_skill_import_req_times: deque[float] = deque()
_skill_import_rl_lock = asyncio.Lock()


async def _enforce_skill_import_rate_limit() -> None:
    limit = settings.skill_import_rate_limit_per_minute
    if limit <= 0:
        return
    async with _skill_import_rl_lock:
        now = time.monotonic()
        window = 60.0
        while _skill_import_req_times and _skill_import_req_times[0] < now - window:
            _skill_import_req_times.popleft()
        if len(_skill_import_req_times) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": status.HTTP_429_TOO_MANY_REQUESTS,
                    "data": None,
                    "error": "rate_limited",
                    "message": "skill-import 请求过于频繁，请稍后再试",
                },
            ) from None
        _skill_import_req_times.append(now)


def _auth_error(status_code: int, message: str, *, error: str = "permission_denied") -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": status_code,
            "data": None,
            "error": error,
            "message": message,
        },
    )


def _parse_form_bool(value: Optional[str]) -> bool:
    if not value:
        return False
    return str(value).strip().lower() in ("true", "1", "on")


def valid_checksum(
    checksum: str = Header(..., alias="X-Checksum-SHA256"),
) -> str:
    value = checksum.strip().lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise HTTPException(
            status_code=400,
            detail={
                "code": 400,
                "data": None,
                "error": "checksum_required",
                "message": "请求头 X-Checksum-SHA256 必填，且为 64 位小写十六进制字符串",
            },
        )
    return value


async def build_skill_import_bundle(
    file: UploadFile = File(..., description="技能集合包（ZIP，顶层为多个 skill 目录）"),
    checksum: str = Depends(valid_checksum),
    force: str = Form("false"),
    fail_fast: str = Form("false"),
) -> SkillImportBundle:
    return SkillImportBundle(
        file=file,
        checksum=checksum,
        force=_parse_form_bool(force),
        fail_fast=_parse_form_bool(fail_fast),
    )


class PublishFormRequired:
    """必填表单参数"""

    def __init__(
        self,
        file: UploadFile = File(..., description="插件包文件（.zip 格式）"),
        checksum: str = Depends(valid_checksum),
    ):
        self.file = file
        self.checksum = checksum


class PublishFormOptional:
    """可选表单参数"""

    def __init__(
        self,
        plugin_id: Optional[str] = Form(
            None,
            description="已存在插件发新版本时必填；首次发布请勿填写，由系统生成 plugin_id",
        ),
        plugin_version: Optional[str] = Form(None),
        version_desc: Optional[str] = Form(None),
        force: bool = Form(False),
    ):
        self.plugin_id = plugin_id.strip() if plugin_id else None
        self.plugin_version = plugin_version.strip() if plugin_version else None
        self.version_desc = version_desc.strip() if version_desc else None
        self.force = force


def build_publish_form(
    required: PublishFormRequired = Depends(),
    optional: PublishFormOptional = Depends(),
) -> PluginPublishForm:
    return PluginPublishForm(
        file=required.file,
        checksum=required.checksum,
        plugin_id=optional.plugin_id,
        plugin_version=optional.plugin_version,
        version_desc=optional.version_desc,
        force=optional.force,
    )


def get_publish_auth(
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token>"),
    x_system_token: Optional[str] = Header(None, alias="X-System-Token"),
) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    返回 (token, is_system_token, acting_user_id)
    - is_system_token=True：表示通过 X-System-Token
    - is_system_token=False：token 需要调用登录系统鉴权
    """
    has_auth = bool(authorization and authorization.strip().lower().startswith("bearer "))
    has_bearer_token = has_auth
    has_system = bool(x_system_token and x_system_token.strip())

    auth_count = int(has_system) + int(has_bearer_token)
    if auth_count != 1:
        raise _auth_error(
            status.HTTP_401_UNAUTHORIZED,
            "Missing/invalid authorization: provide exactly one of Authorization: Bearer <token>, or X-System-Token",
        )

    if has_system:
        system_admin_token = SecurityUtils.get_decrypt_secret("SYSTEM_ADMIN_TOKEN", default="") or ""
        if system_admin_token and x_system_token.strip() == system_admin_token:
            acting = settings.system_admin_user
            return (None, True, acting)
        raise _auth_error(status.HTTP_401_UNAUTHORIZED, "Invalid X-System-Token")

    token = authorization[7:].strip()
    if not token:
        raise _auth_error(status.HTTP_401_UNAUTHORIZED, "Invalid or empty token")
    return (token, False, None)


@plugin_router.post("", response_model=ResponseModel[PluginPublishResult])
async def publish_plugin(
    form: PluginPublishForm = Depends(build_publish_form),
    db: Session = Depends(get_db),
    storage=Depends(get_storage_client),
    auth: Tuple[Optional[str], bool, Optional[str]] = Depends(get_publish_auth),
):
    # Upload 到 S3 之前先校验 token
    token, is_system_token, acting_user_id = auth
    if not is_system_token:
        acting_user_id = await get_gitcode_user_id(token or "")

    content = await form.file.read()
    try:
        result = plugin_publish(
            user_id=acting_user_id or "",
            content=content,
            filename=form.file.filename,
            expected_checksum=form.checksum,
            plugin_id=form.plugin_id,
            plugin_version=form.plugin_version,
            version_desc=form.version_desc,
            force=form.force,
            db=db,
            storage=storage,
        )
    except PublishError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e

    return ResponseModel(
        code=status.HTTP_200_OK,
        message="Publish plugin successfully",
        data=result,
    )


def _template_filename_from_key(key: str) -> str:
    base = (key or "").strip().rstrip("/").split("/")[-1]
    return base or "plugin-template.zip"


@plugin_router.get(
    "/publish-template",
    response_model=ResponseModel[PluginTemplatePresignData],
)
async def get_publish_template_presigned(
    auth: AuthContext = Depends(require_auth),
    storage=Depends(get_storage_client),
):
    """为发布页「下载模板」生成私有桶对象的预签名 GET URL（需 Bearer 或 X-System-Token）。"""
    _ = auth
    key = (settings.plugin_template_object_key or "").strip()
    if not key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": 503,
                "data": None,
                "error": "template_not_configured",
                "message": "未配置发布模板对象路径（MARKET_PLUGIN_TEMPLATE_OBJECT_KEY）",
            },
        )
    exp_arg = settings.plugin_template_presigned_expires
    try:
        if exp_arg and exp_arg > 0:
            url = storage.presigned_get_url(key, expires_in=exp_arg)
            ttl = exp_arg
        else:
            url = storage.presigned_get_url(key)
            ttl = storage.config.presigned_expires_seconds
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": 500,
                "data": None,
                "error": "presign_failed",
                "message": f"生成模板下载链接失败：{e!s}",
            },
        ) from e

    return ResponseModel(
        code=status.HTTP_200_OK,
        message="ok",
        data=PluginTemplatePresignData(
            download_url=url,
            expires_in=int(ttl),
            filename=_template_filename_from_key(key),
        ),
    )


@plugin_router.post(
    "/skill-import",
    response_model=ResponseModel[SkillImportResponse],
)
async def skill_import(
    bundle: SkillImportBundle = Depends(build_skill_import_bundle),
    db: Session = Depends(get_db),
    storage=Depends(get_storage_client),
    auth: Tuple[Optional[str], bool, Optional[str]] = Depends(get_publish_auth),
):
    """批量导入 skill：仅 X-System-Token；须 X-Checksum-SHA256。"""
    await _enforce_skill_import_rate_limit()

    _token, is_system_token, acting_user_id = auth
    if not is_system_token:
        raise _auth_error(
            status.HTTP_403_FORBIDDEN,
            "批量导入仅支持 X-System-Token（系统管理员）",
            error="forbidden",
        )

    tmp_path: FsPath | None = None
    upload_tmp_name: str | None = None
    try:
        # NamedTemporaryFile + with：退出 with 时文件对象关闭（G.FIO.04，无裸 fd）
        with tempfile.NamedTemporaryFile(
            prefix="oj_skill_bundle_",
            suffix=".zip",
            delete=False,
            mode="wb",
        ) as out:
            upload_tmp_name = out.name
            tmp_path = FsPath(out.name)
            hasher = hashlib.sha256()
            written = 0
            while True:
                chunk = await bundle.file.read(ZIP_STREAM_READ_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "code": 400,
                            "data": None,
                            "error": "payload_too_large",
                            "message": "技能集合包原始大小超过 512MB 上限",
                        },
                    ) from None
                hasher.update(chunk)
                out.write(chunk)

        if hasher.hexdigest() != bundle.checksum:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": 400,
                    "data": None,
                    "error": "checksum_mismatch",
                    "message": "技能集合包 X-Checksum-SHA256 与实际上传内容不一致",
                },
            ) from None

        try:
            data = skill_import_from_bundle(
                bundle_path=tmp_path,
                user_id=acting_user_id or "",
                db=db,
                storage=storage,
                force=bundle.force,
                fail_fast=bundle.fail_fast,
            )
        except PublishError as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail) from e

        return ResponseModel(
            code=status.HTTP_200_OK,
            message="Import skills finished",
            data=data,
        )
    finally:
        if upload_tmp_name:
            try:
                FsPath(upload_tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


@plugin_router.get(
    "",
    response_model=ResponseModel[PluginListResponse],
)
def list_plugins(
    query: PluginListQuery = Depends(),
    db: Session = Depends(get_db),
    storage=Depends(get_storage_client),
):
    data = list_plugins_service(query=query, db=db, storage=storage)
    return ResponseModel(code=status.HTTP_200_OK, message="ok", data=data)


@artifact_router.get(
    "/{id}",
    response_model=ResponseModel[PluginDownloadData],
)
async def get_artifact_download(
    artifact_id: str = Path(..., alias="id"),
    version: Optional[str] = Query(None, description="版本号（如 1.0.0），不指定则返回最新版本"),
    db: Session = Depends(get_db),
    storage=Depends(get_storage_client),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token>"),
):
    fetch_user_id: Optional[str] = None
    token: str = ""
    if authorization and authorization.strip().lower().startswith("bearer "):
        token = authorization.strip()[7:].strip()

    if token:
        try:
            fetch_user_id = await get_gitcode_user_id(token)
        except HTTPException:
            # 下载接口允许匿名访问；若 token 无效则不写入 fetch_user_id，避免影响下载。
            fetch_user_id = None

    try:
        result = get_download_info(
            asset_id=artifact_id,
            version=version,
            db=db,
            storage=storage,
            fetch_user_id=fetch_user_id,
        )
    except PublishError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e

    return ResponseModel(
        code=status.HTTP_200_OK,
        message="ok",
        data=result,
    )


def _key_from_object_uri(storage: Any, uri_or_key: Optional[str]) -> Optional[str]:
    """
    将公开 URL（例如 http://localhost:9000/<bucket>/<key>）或直接 key 规范化为对象 key。
    """
    if not uri_or_key:
        return None
    raw = uri_or_key.strip()
    if not raw:
        return None
    if "://" not in raw:
        return raw

    try:
        p = urlparse(raw)
        path = (p.path or "").lstrip("/")  # <bucket>/<key>
        bucket = getattr(getattr(storage, "config", None), "bucket_name", None)
        if bucket and path.startswith(f"{bucket}/"):
            return path[len(bucket) + 1:]
        return path
    except Exception:
        return None


@plugin_router.get(
    "/{asset_id}/versions/{version}",
    response_model=ResponseModel[PluginVersionDetail],
)
def get_plugin_version_detail(
    asset_id: str,
    version: str,
    db: Session = Depends(get_db),
    storage=Depends(get_storage_client),
):
    data = get_plugin_version_detail_service(
        asset_id=asset_id,
        version=version,
        db=db,
        storage=storage,
    )
    return ResponseModel(code=status.HTTP_200_OK, message="ok", data=data)


@plugin_router.delete(
    "/{asset_id}/versions/{version}",
    response_model=ResponseModel[PluginVersionDeleteData],
)
async def delete_plugin_version(
    asset_id: str,
    version: str,
    auth: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
    storage: Any = Depends(get_storage_client),
):
    """
    删除指定资产的指定版本（version=all 时删除该资产下所有版本并删除资产主表记录）。
    鉴权：Authorization Bearer（GitCode token）或 X-System-Token 二选一。
    """
    data = delete_plugin_version_service(
        asset_id=asset_id,
        version=version,
        auth=auth,
        db=db,
        storage=storage,
    )
    return ResponseModel(code=status.HTTP_200_OK, message="ok", data=data)


router = APIRouter()
router.include_router(plugin_router)
router.include_router(artifact_router)
