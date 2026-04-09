from dataclasses import dataclass
from typing import Literal, List, Optional

from fastapi import UploadFile
from pydantic import BaseModel, Field, field_validator


@dataclass
class PluginPublishForm:
    file: UploadFile
    checksum: str
    plugin_id: Optional[str]
    plugin_version: Optional[str]
    version_desc: Optional[str]
    force: bool


class AssetCreate(BaseModel):
    """Parameters for creating a market asset."""

    asset_id: str
    name: str
    display_name: str
    asset_type: str = "plugin"
    short_desc: Optional[str] = None
    detail_desc: Optional[str] = None
    tags: Optional[List[str]] = None
    publisher_id: str = ""
    publisher_name: str = ""
    plugin_type: Optional[str] = None
    latest_version: Optional[str] = None


class AssetVersionCreate(BaseModel):
    """Parameters for creating an asset version."""

    version_id: str
    asset_id: str
    version: str
    changelog: Optional[str] = None
    status: str = "ACTIVE"
    file_path: Optional[str] = None


class PluginPublishResult(BaseModel):
    """Result of plugin publish operation."""

    plugin_id: str
    name: str
    version: str
    status: str
    published_at: str
    storage_url: str


# ----- DELETE /api/v1/plugins/{asset_id}/versions/{version} -----


class PluginVersionDeleteData(BaseModel):
    asset_id: str
    version: str  # 具体版本号或 "all"


class PluginTemplatePresignData(BaseModel):
    """GET /plugins/publish-template 返回的预签名下载信息。"""

    download_url: str
    expires_in: int
    filename: str


class PluginVersionDetail(BaseModel):
    asset_id: str
    version: str
    asset_type: str
    plugin_type: Optional[str] = None
    name: str
    display_name: str
    short_desc: Optional[str] = None
    detail_desc: Optional[str] = None
    publisher_id: str
    publisher_name: str
    tags: Optional[List[str]] = None
    certification: Optional[str] = None
    changelog: Optional[str] = None
    file_path: Optional[str] = None
    icon_uri: Optional[str] = None


# ----- GET /api/v1/plugins 列表 -----


class PluginDownloadData(BaseModel):
    """GET /api/v1/artifacts/{id} 响应体 data。"""

    download_url: str
    asset_id: str
    name: str
    version: str
    file_size: int
    checksum_sha256: str

PLUGIN_ORDER_BY_OPTIONS = ("install_count", "like_count", "create_time", "update_time", "review_count")


OrderByField = Literal["install_count", "like_count", "create_time", "update_time", "review_count"]


class PluginListQuery(BaseModel):
    """GET /api/v1/plugins 的 query 参数（非必填）。"""

    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页条数")
    asset_id: Optional[str] = Field(None, description="资产 ID")
    asset_type: Optional[str] = Field(None, description="资产类型")
    publisher_id: Optional[str] = Field(None, description="发布者 ID")
    publisher_name: Optional[str] = Field(None, description="发布者名称（模糊）")
    plugin_type: Optional[str] = Field(None, description="插件类型（精确匹配）")
    search_keyword: Optional[str] = Field(
        None, description="关键词（对 name/display_name/short_desc/detail_desc 模糊）"
    )
    order_by: str = Field(
        "install_count",
        description="排序字段: install_count, like_count, create_time, update_time, review_count",
    )
    desc: bool = Field(True, description="排序方向: true=降序, false=升序")  # True=降序，False=升序

    @field_validator("order_by", mode="before")
    @classmethod
    def normalize_order_by(cls, v: object) -> str:
        if v is None:
            return "install_count"
        s = str(v).strip()
        if not s:
            raise ValueError("order_by cannot be empty")
        if s in PLUGIN_ORDER_BY_OPTIONS:
            return s
        allowed = ", ".join(PLUGIN_ORDER_BY_OPTIONS)
        raise ValueError(f"order_by must be one of: {allowed}; got {v!r}")


class PluginListItem(BaseModel):
    """列表项，不包含 status。"""

    asset_id: str
    asset_type: str
    name: str
    display_name: Optional[str] = None
    short_desc: Optional[str] = None
    detail_desc: Optional[str] = None
    icon_uri: Optional[str] = None
    publisher_id: str
    publisher_name: str
    tags: Optional[List[str]] = None
    certification: Optional[str] = None
    plugin_type: Optional[str] = None
    latest_version: Optional[str] = None
    view_count: int = 0
    install_count: int = 0
    like_count: int = 0
    review_count: int = 0
    average_rating: float = 8.0
    create_time: Optional[int] = None
    update_time: Optional[int] = None

    model_config = {"from_attributes": True}


class PluginListResponse(BaseModel):
    """GET /api/v1/plugins 响应体 data。"""

    page: int
    page_size: int
    total: int
    items: list[PluginListItem]
