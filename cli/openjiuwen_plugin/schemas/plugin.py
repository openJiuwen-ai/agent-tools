from __future__ import annotations

import re
from pathlib import Path
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")

MARKETPLACE_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def normalize_marketplace_version_optional(value: str | None) -> str | None:
    """规范化可选版本号：trim、可去单个 v/V，须为 x.y.z。"""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if len(s) > 1 and s[0] in ("v", "V"):
        s = s[1:].strip()
    if not MARKETPLACE_VERSION_PATTERN.match(s):
        raise ValueError(
            "version must match marketplace format: x.y.z (three numeric segments, e.g. 1.0.0); "
            "optional leading v/V is accepted; prerelease/build suffixes are not allowed"
        )
    return s


# ---- Common wrapper --------------------------------------------------------
class ResponseModel(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="ignore")

    code: int = 200
    message: str = "Success"
    data: T | None = None


# ---- Request / Input models ------------------------------------------------
class PublishPluginInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    plugin_path: Path | None = None
    zip_path: Path | None = None
    plugin_id: str | None = None
    plugin_version: str | None = None
    version_desc: str | None = None
    force: bool = False

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "PublishPluginInput":
        plugin_path = Path(self.plugin_path).resolve() if self.plugin_path else None
        zip_path = Path(self.zip_path).resolve() if self.zip_path else None
        plugin_id = self._norm_optional(self.plugin_id)
        plugin_version_raw = self._norm_optional(self.plugin_version)
        plugin_version = normalize_marketplace_version_optional(plugin_version_raw)
        version_desc = self._norm_optional(self.version_desc)

        if plugin_path is None and zip_path is None:
            raise ValueError("either plugin_path or zip_path must be provided")

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

    zip_path: Path
    checksum_sha256: str
    plugin_id: str | None = None
    plugin_version: str | None = None
    version_desc: str | None = None
    force: bool = False

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "PublishRequest":
        zip_path = Path(self.zip_path).resolve()
        checksum = str(self.checksum_sha256).strip().lower()
        plugin_id = self._norm_optional(self.plugin_id)
        plugin_version_raw = self._norm_optional(self.plugin_version)
        plugin_version = normalize_marketplace_version_optional(plugin_version_raw)
        version_desc = self._norm_optional(self.version_desc)

        if not zip_path.is_file():
            raise ValueError(f"zip file not found: {zip_path}")
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("checksum_sha256 must be 64-char hex")

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


class SkillImportItemResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entry: str = ""
    status: Literal["ok", "error"]
    plugin_id: str | None = None
    name: str | None = None
    version: str | None = None
    error: str | None = None
    message: str | None = None


class SkillImportSummary(BaseModel):
    """total 为集合包顶层 skill 目录数；fail_fast 提前结束时 ok+failed 可能小于 total。"""

    model_config = ConfigDict(extra="ignore")

    total: int = 0
    ok: int = 0
    failed: int = 0


class SkillImportResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: SkillImportSummary
    results: list[SkillImportItemResult] = Field(default_factory=list)


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
