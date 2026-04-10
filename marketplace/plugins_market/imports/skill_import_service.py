from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from plugins_market.core.errors import PublishError
from plugins_market.core.s3_storage_client import S3StorageClient
from plugins_market.imports.bundle_safe_extract import skill_import_extract_zip_to_dir
from plugins_market.validation.constants import (
    MAX_JSON_BYTES,
    MAX_ZIP_ENTRIES,
)
from plugins_market.imports.skill_entries import entry_to_publish_zip
from plugins_market.schemas.plugin import (
    SkillImportItemResult,
    SkillImportResponse,
    SkillImportSummary,
)
from plugins_market.services.plugin import publish

logger = logging.getLogger(__name__)


def _skill_import_merge_tags(raw: object, fallback: list[str]) -> list[str]:
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t is not None and str(t).strip()]
    return list(fallback)


def _skill_import_parse_entries_map(manifest: dict[str, Any]) -> dict[str, Any]:
    """解析 ``manifest.json``：根上每个 **值为 object** 的键视为「顶层目录名 → 条目配置」。"""
    return {k: v for k, v in manifest.items() if isinstance(v, dict)}


def _skill_import_entry_version_desc(entry_overrides: dict[str, Any]) -> str | None:
    """条目配置中的 ``version_desc``；缺省或非空串则 ``None``。"""
    raw = entry_overrides.get("version_desc")
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


# 简单包在 manifest 未写 ``version`` 时的 semver 兜底（无 HTTP/包级 defaults）
_SIMPLE_VERSION_FALLBACK = "0.0.1"
_SIMPLE_AUTHOR_FALLBACK = "system_admin"


def skill_import_from_staging_dir(
    tmp_root: Path,
    *,
    user_id: str,
    db: Session,
    storage: S3StorageClient,
    force: bool = False,
    fail_fast: bool = False,
) -> SkillImportResponse:
    """对已展开到 ``tmp_root`` 的集合包目录执行导入（与 ZIP 解压后布局一致）。

    调用方负责创建/清理 ``tmp_root``。``manifest.json`` 规则同 ``skill_import_from_bundle``。
    """
    results: list[SkillImportItemResult] = []

    manifest: dict[str, Any] = {}
    mf = tmp_root / "manifest.json"
    if mf.is_file():
        raw_mf = mf.read_bytes()
        if len(raw_mf) > MAX_JSON_BYTES:
            raise PublishError(
                code=400,
                error="manifest_too_large",
                message=f"manifest.json 超过 {MAX_JSON_BYTES} 字节上限",
            )
        try:
            text = raw_mf.decode("utf-8")
        except UnicodeDecodeError as e:
            raise PublishError(
                code=400,
                error="manifest_invalid",
                message=f"manifest.json 不是合法 UTF-8：{e}",
            ) from e
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise PublishError(
                code=400,
                error="manifest_invalid",
                message=f"manifest.json JSON 解析失败：{e}",
            ) from e
        if not isinstance(parsed, dict):
            raise PublishError(
                code=400,
                error="manifest_invalid",
                message="manifest.json 根节点必须为 JSON object",
            )
        manifest = parsed

    entry_dirs = sorted(
        [
            p
            for p in tmp_root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name != "__MACOSX"
        ],
        key=lambda p: p.name,
    )

    if not entry_dirs:
        raise PublishError(
            code=400,
            error="invalid_skill_bundle",
            message="无有效 skill 顶层目录（简单包：根目录 SKILL.md；标准包：plugin.yaml+icon+子目录 SKILL.md）",
        )

    if len(entry_dirs) > MAX_ZIP_ENTRIES:
        raise PublishError(
            code=400,
            error="too_many_skill_entries",
            message=(
                f"顶层 skill 目录数量 {len(entry_dirs)} 超过上限 {MAX_ZIP_ENTRIES} "
                f"（与 ZIP 条目数上限一致），请分批导入或拆分集合包"
            ),
        )

    entries_map = _skill_import_parse_entries_map(manifest)

    for entry in entry_dirs:
        entry_name = entry.name
        raw_eo = entries_map.get(entry_name)
        eo = raw_eo if isinstance(raw_eo, dict) else {}
        author_a = str(eo.get("author") or _SIMPLE_AUTHOR_FALLBACK).strip() or _SIMPLE_AUTHOR_FALLBACK
        tags_a = _skill_import_merge_tags(eo.get("tags"), [])

        entry_force = force or bool(eo.get("force"))
        plugin_id = eo.get("plugin_id")
        plugin_id_str = str(plugin_id).strip() if plugin_id else None

        publish_zip: Path | None = None
        name: str = ""
        version: str = ""
        try:
            publish_zip, name, version = entry_to_publish_zip(
                entry,
                entry_key=entry_name,
                entry_overrides=eo,
                version_fallback=_SIMPLE_VERSION_FALLBACK,
                default_author=author_a,
                default_tags=tags_a,
            )
        except ValueError as e:
            results.append(
                SkillImportItemResult(
                    entry=entry_name,
                    status="error",
                    error="import_normalize_failed",
                    message=str(e),
                )
            )
            logger.info(
                "skill import entry failed: entry=%s status=error error=import_normalize_failed",
                entry_name,
            )
            if fail_fast:
                break
            continue

        zip_bytes = publish_zip.read_bytes()
        inner_checksum = hashlib.sha256(zip_bytes).hexdigest()
        entry_version_desc = _skill_import_entry_version_desc(eo)
        try:
            pr = publish(
                user_id=user_id,
                content=zip_bytes,
                filename=f"{name}-{version}.zip",
                expected_checksum=inner_checksum,
                plugin_id=plugin_id_str,
                plugin_version=None,
                version_desc=entry_version_desc,
                force=entry_force,
                db=db,
                storage=storage,
            )
            results.append(
                SkillImportItemResult(
                    entry=entry_name,
                    status="ok",
                    plugin_id=pr.plugin_id,
                    name=pr.name,
                    version=pr.version,
                )
            )
            logger.info(
                "skill import entry ok: entry=%s plugin_id=%s name=%s version=%s",
                entry_name,
                pr.plugin_id,
                pr.name,
                pr.version,
            )
        except PublishError as e:
            msg = str(e.detail.get("message") or e)
            err = str(e.detail.get("error") or "publish_failed")
            results.append(
                SkillImportItemResult(
                    entry=entry_name,
                    status="error",
                    name=name,
                    version=version,
                    error=err,
                    message=msg,
                )
            )
            logger.info(
                "skill import entry failed: entry=%s status=error error=%s name=%s version=%s",
                entry_name,
                err,
                name,
                version,
            )
            if fail_fast:
                break
        finally:
            if publish_zip is not None:
                publish_zip.unlink(missing_ok=True)

    entry_total = len(entry_dirs)
    ok = sum(1 for r in results if r.status == "ok")
    failed = sum(1 for r in results if r.status == "error")
    return SkillImportResponse(
        summary=SkillImportSummary(total=entry_total, ok=ok, failed=failed),
        results=results,
    )


def skill_import_from_bundle(
    *,
    bundle_path: Path,
    user_id: str,
    db: Session,
    storage: S3StorageClient,
    force: bool = False,
    fail_fast: bool = False,
) -> SkillImportResponse:
    """解压集合 ZIP，逐条打成 skill 包并 ``publish``。

    HTTP 入口已在落盘时校验 ``X-Checksum-SHA256`` 与 ``MAX_FILE_SIZE``；若从其它路径调用，
    调用方须自行保证 ``bundle_path`` 内容与完整性。

    ``manifest.json``：若存在则须为合法 UTF-8 JSON object；根上**仅**「键 → JSON object」参与条目解析，值为非 object 的键忽略。
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="oj_skill_bundle_"))
    try:
        try:
            skill_import_extract_zip_to_dir(bundle_path, tmp_root)
        except ValueError as e:
            raise PublishError(
                code=400,
                error="invalid_skill_bundle",
                message=str(e) or "技能集合包解压失败",
            ) from e
        return skill_import_from_staging_dir(
            tmp_root,
            user_id=user_id,
            db=db,
            storage=storage,
            force=force,
            fail_fast=fail_fast,
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
