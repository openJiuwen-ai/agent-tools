from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


# ---- Common wrapper --------------------------------------------------------
class ResponseModel(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="ignore")

    code: int = 200
    message: str = "Success"
    data: T | None = None


# ---- Request / Input models ------------------------------------------------
class PublishPluginInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    plugin_path: Path | None = None
    zip_path: Path | None = None
    plugin_id: str | None = None
    plugin_version: str | None = None
    version_desc: str | None = None
    force: bool = False

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "PublishPluginInput":
        user_id = str(self.user_id).strip()
        plugin_path = Path(self.plugin_path).resolve() if self.plugin_path else None
        zip_path = Path(self.zip_path).resolve() if self.zip_path else None
        plugin_id = self._norm_optional(self.plugin_id)
        plugin_version = self._norm_optional(self.plugin_version)
        version_desc = self._norm_optional(self.version_desc)

        if not user_id:
            raise ValueError("user_id cannot be empty")
        if plugin_path is None and zip_path is None:
            raise ValueError("either plugin_path or zip_path must be provided")

        object.__setattr__(self, "user_id", user_id)
        object.__setattr__(self, "plugin_path", plugin_path)
        object.__setattr__(self, "zip_path", zip_path)
        object.__setattr__(self, "plugin_id", plugin_id)
        object.__setattr__(self, "plugin_version", plugin_version)
        object.__setattr__(self, "version_desc", version_desc)
        return self

    @staticmethod
    def _norm_optional(value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class PublishRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    zip_path: Path
    checksum_sha256: str
    plugin_id: str | None = None
    plugin_version: str | None = None
    version_desc: str | None = None
    force: bool = False

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "PublishRequest":
        user_id = str(self.user_id).strip()
        zip_path = Path(self.zip_path).resolve()
        checksum = str(self.checksum_sha256).strip().lower()
        plugin_id = self._norm_optional(self.plugin_id)
        plugin_version = self._norm_optional(self.plugin_version)
        version_desc = self._norm_optional(self.version_desc)

        if not user_id:
            raise ValueError("user_id cannot be empty")
        if not zip_path.is_file():
            raise ValueError(f"zip file not found: {zip_path}")
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("checksum_sha256 must be 64-char hex")

        object.__setattr__(self, "user_id", user_id)
        object.__setattr__(self, "zip_path", zip_path)
        object.__setattr__(self, "checksum_sha256", checksum)
        object.__setattr__(self, "plugin_id", plugin_id)
        object.__setattr__(self, "plugin_version", plugin_version)
        object.__setattr__(self, "version_desc", version_desc)
        return self

    @staticmethod
    def _norm_optional(value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class PluginListQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    search_keyword: str = ""
    plugin_type: str | None = None
    publisher_name: str | None = None
    asset_id: str | None = None
    asset_type: str | None = None
    publisher_id: str | None = None
    page: int = 1
    page_size: int = 20
    order_by: str = "install_count"
    desc: bool = True


# ---- Response / Data models ------------------------------------------------
class PluginPublishResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    plugin_id: str = ""
    name: str = ""
    version: str = ""
    status: str = ""
    published_at: str = ""
    storage_url: str = ""


class PluginVersionDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    asset_id: str = ""
    version: str = ""
    asset_type: str = ""
    plugin_type: str | None = None
    name: str = ""
    display_name: str = ""
    short_desc: str | None = None
    detail_desc: str | None = None
    publisher_id: str = ""
    publisher_name: str = ""
    tags: list[str] | None = None
    certification: str | None = None
    changelog: str | None = None
    file_path: str | None = None
    icon_uri: str | None = None


class PluginListItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    asset_id: str = ""
    asset_type: str = ""
    name: str = ""
    display_name: str = ""
    latest_version: str = ""


class PluginListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page: int = 1
    page_size: int = 20
    total: int = 0
    items: list[PluginListItem] = Field(default_factory=list)


class PluginVersionDeleteData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    asset_id: str = ""
    version: str = ""


class PluginDownloadData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    download_url: str = ""
    asset_id: str = ""
    name: str = ""
    version: str = ""
    file_size: int = 0
    checksum_sha256: str = ""


class DownloadArtifactResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    download_url: str
    expected_checksum_sha256: str
    actual_checksum_sha256: str
    verified: bool
