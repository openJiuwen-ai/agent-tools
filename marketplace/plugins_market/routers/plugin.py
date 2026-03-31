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
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from common.security.security_utils import SecurityUtils
from plugins_market.core.auth import require_auth_with_user_id, verify_bearer_via_user
from plugins_market.core.config import settings
from plugins_market.core.database import get_db
from plugins_market.core.s3_storage_client import get_storage_client
from plugins_market.repositories import (
    MarketAssetRepository,
    MarketAssetVersionRepository,
)
from plugins_market.schemas.common import ResponseModel
from plugins_market.schemas.plugin import (
    PluginDownloadData,
    PluginListItem,
    PluginListQuery,
    PluginListResponse,
    PluginPublishForm,
    PluginPublishResult,
    PluginVersionDeleteData,
    PluginVersionDetail,
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


class PublishFormRequired:
    """必填表单参数"""

    def __init__(
        self,
        user_id: str = Form(..., description="用户ID"),
        file: UploadFile = File(..., description="插件包文件（.zip 格式）"),
        checksum: str = Depends(valid_checksum),
    ):
        self.user_id = user_id.strip()
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
        user_id=required.user_id,
        file=required.file,
        checksum=required.checksum,
        plugin_id=optional.plugin_id,
        plugin_version=optional.plugin_version,
        version_desc=optional.version_desc,
        force=optional.force,
    )


def get_publish_auth(
    authorization: Optional[str] = Header(
        None, description="Authorization: Bearer <token>"
    ),
    x_system_token: Optional[str] = Header(None, alias="X-System-Token"),
    token_header: Optional[str] = Header(
        None, alias="token", description="兼容：token: <token>（等同于 Bearer token）"
    ),
) -> Tuple[Optional[str], bool]:
    """
    返回 (token, is_system_token)
    - is_system_token=True：表示通过 X-System-Token
    - is_system_token=False：token 需要调用登录系统鉴权
    """
    has_auth = bool(authorization and authorization.strip().lower().startswith("bearer "))
    has_token_header = bool(token_header and token_header.strip())
    has_bearer_token = has_auth or has_token_header
    has_system = bool(x_system_token and x_system_token.strip())

    auth_count = int(has_system) + int(has_bearer_token)
    if auth_count != 1:
        raise _auth_error(
            status.HTTP_403_FORBIDDEN,
            "Missing/invalid authorization: provide exactly one of "
            "Authorization: Bearer <token>, token: <token>, or X-System-Token",
        )

    if has_system:
        system_admin_token = SecurityUtils.get_decrypt_secret("SYSTEM_ADMIN_TOKEN", default="") or ""
        if system_admin_token and x_system_token.strip() == system_admin_token:
            return (None, True)
        raise _auth_error(status.HTTP_403_FORBIDDEN, "Invalid X-System-Token")

    token = authorization[7:].strip() if has_auth else token_header.strip()
    if not token:
        raise _auth_error(status.HTTP_403_FORBIDDEN, "Invalid or empty token")
    return (token, False)


@plugin_router.post("", response_model=ResponseModel[PluginPublishResult])
async def publish_plugin(
    form: PluginPublishForm = Depends(build_publish_form),
    db: Session = Depends(get_db),
    storage=Depends(get_storage_client),
    auth: Tuple[Optional[str], bool] = Depends(get_publish_auth),
):
    # Upload 到 S3 之前先校验 token
    token, is_system_token = auth
    if not is_system_token:
        await verify_bearer_via_user(form.user_id, token or "")

    content = await form.file.read()
    try:
        result = plugin_publish(
            user_id=form.user_id,
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
def get_artifact_download(
    artifact_id: str = Path(..., alias="id"),
    db: Session = Depends(get_db),
    storage=Depends(get_storage_client),
):
    try:
        result = get_download_info(
            asset_id=artifact_id,
            db=db,
            storage=storage,
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
    auth: tuple = Depends(require_auth_with_user_id),
    db: Session = Depends(get_db),
    storage: Any = Depends(get_storage_client),
):
    """
    删除指定资产的指定版本（version=all 时删除该资产下所有版本并删除资产主表记录）。
    鉴权：Authorization Bearer + user_id 或 X-System-Token 二选一。
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
