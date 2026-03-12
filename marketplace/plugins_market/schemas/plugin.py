from dataclasses import dataclass
from typing import Optional

from fastapi import UploadFile
from pydantic import BaseModel


@dataclass
class PluginPublishForm:
    space_id: str
    file: UploadFile
    plugin_id: Optional[str]
    plugin_version: Optional[str]
    version_desc: Optional[str]
    force: bool


class AssetCreate(BaseModel):
    """Parameters for creating a market asset."""

    asset_id: str
    name: str
    asset_type: str = "plugin"
    short_desc: Optional[str] = None
    publisher_id: str = ""
    publisher_name: str = ""
    category_id: str = ""
    latest_version: Optional[str] = None


class AssetVersionCreate(BaseModel):
    """Parameters for creating an asset version."""

    version_id: str
    asset_id: str
    version: str
    source_publish_id: str
    changelog: Optional[str] = None
    status: str = "pending_review"


class PluginPublishResult(BaseModel):
    """Result of plugin publish operation."""

    plugin_id: str
    name: str
    version: str
    status: str
    published_at: str
    storage_url: str
