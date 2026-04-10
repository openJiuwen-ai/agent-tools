from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from plugins_market.core.errors import PublishError
from plugins_market.imports.yaml_util import dump_plugin_yaml, load_plugin_yaml, split_skill_frontmatter
from plugins_market.validation.constants import (
    MINIMAL_PNG_BYTES,
    NAME_PATTERN,
    SKILL_DESC_MAX_LEN,
    SKILL_NAME_MAX_LEN,
    SKILL_NAME_PATTERN,
)
from plugins_market.validation.zip_utils import validate_png_icon_bytes


def _validate_disk_icon_png(icon_file: Path) -> None:
    try:
        validate_png_icon_bytes(icon_file.read_bytes(), path="icon.png")
    except PublishError as e:
        raise ValueError(str(e.detail.get("message") or e)) from e


def _validate_plugin_skill_name(name: str) -> None:
    if not name or len(name) > SKILL_NAME_MAX_LEN:
        raise ValueError(f"skill name invalid or longer than {SKILL_NAME_MAX_LEN}")
    if not NAME_PATTERN.match(name):
        raise ValueError("skill name must match ^[a-z][a-z0-9-]*$")
    if not SKILL_NAME_PATTERN.match(name):
        raise ValueError(
            "skill name must use lowercase, digits, single hyphens between segments (Agent Skills rules)"
        )


def _normalize_description(desc: str) -> str:
    s = desc.strip()
    if not s:
        raise ValueError("description must be non-empty")
    if len(s) > SKILL_DESC_MAX_LEN:
        raise ValueError(f"description must be at most {SKILL_DESC_MAX_LEN} characters")
    return s


def _yaml_quote_string(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_skill_md(name: str, description: str, body: str) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {_yaml_quote_string(description)}\n"
        "---\n\n"
        f"{body.lstrip()}"
    )


def build_skill_plugin_zip_to_path(
    staging_root: Path, plugin_name: str, version: str, out_path: Path
) -> None:
    """将 staging 打成 `{name}-{version}/...` 布局的 skill 插件 ZIP，写入 ``out_path``（流式写入，不落整包于内存）。"""
    prefix = f"{plugin_name}-{version}"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in staging_root.rglob("*"):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(staging_root)
            zf.write(fpath, f"{prefix}/{rel.as_posix()}")


def is_standard_skill_entry(entry: Path) -> bool:
    """标准包：plugin.yaml、icon、{name}/SKILL.md。"""
    if not (entry / "plugin.yaml").is_file() or not (entry / "icon.png").is_file():
        return False
    try:
        data = load_plugin_yaml(str(entry / "plugin.yaml"))
    except ValueError:
        return False
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        return False
    name = name.strip()
    return (entry / name / "SKILL.md").is_file()


def is_simple_skill_entry(entry: Path) -> bool:
    """简单包：条目根目录有 SKILL.md。"""
    return (entry / "SKILL.md").is_file()


def validate_standard_skill_staging(staging: Path) -> tuple[str, str]:
    """标准包 staging 校验（只读，不写 yaml）。"""
    data = load_plugin_yaml(str(staging / "plugin.yaml"))
    name = str(data.get("name") or "").strip()
    _validate_plugin_skill_name(name)

    skill_md = staging / name / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    fm, _body = split_skill_frontmatter(text)
    fm_name = fm.get("name")
    if not isinstance(fm_name, str) or fm_name.strip() != name:
        raise ValueError("SKILL.md frontmatter name must equal plugin.yaml name")

    version = str(data.get("version") or "").strip()
    if not version or not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", version):
        raise ValueError("plugin.yaml version must be semver x.y.z")

    _validate_disk_icon_png(staging / "icon.png")

    return name, version


def build_simple_skill_staging(
    entry: Path,
    staging: Path,
    *,
    default_version: str,
    default_author: str,
    default_tags: list[str],
    display_name: str | None = None,
) -> tuple[str, str]:
    """简单包：根 SKILL.md -> 补全为标准包目录树（生成 plugin.yaml/icon）。"""
    text = (entry / "SKILL.md").read_text(encoding="utf-8")
    fm, body = split_skill_frontmatter(text)
    raw_name = fm.get("name")
    if not isinstance(raw_name, str):
        raise ValueError("SKILL.md frontmatter name is required and must be a string")
    name = raw_name.strip()
    _validate_plugin_skill_name(name)

    raw_desc = fm.get("description")
    if not isinstance(raw_desc, str):
        raise ValueError("SKILL.md frontmatter description is required and must be a string")
    description = _normalize_description(raw_desc)

    staging.mkdir(parents=True, exist_ok=True)
    skill_dir = staging / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    (skill_dir / "SKILL.md").write_text(
        _render_skill_md(name, description, body),
        encoding="utf-8",
    )

    for child in entry.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir() and child.name in {
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            ".eggs",
            "dist",
            "out",
        }:
            continue
        if child.name == "SKILL.md":
            continue
        dest = skill_dir / child.name
        if child.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(child, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dest)

    ver = default_version.strip()
    if not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", ver):
        raise ValueError("default_version must be semver x.y.z")

    disp = (display_name or "").strip() or name.replace("-", " ").title()
    plugin_data: dict[str, Any] = {
        "name": name,
        "version": ver,
        "display_name": disp,
        "description": description[:1024],
        "runtime": {"type": "skill"},
        "metadata": {"author": default_author.strip(), "tags": list(default_tags)},
    }
    (staging / "plugin.yaml").write_text(dump_plugin_yaml(plugin_data), encoding="utf-8")
    (staging / "icon.png").write_bytes(MINIMAL_PNG_BYTES)
    _validate_disk_icon_png(staging / "icon.png")
    return name, ver


def entry_to_publish_zip(
    entry: Path,
    *,
    entry_key: str,
    entry_overrides: dict[str, Any],
    version_fallback: str,
    default_author: str,
    default_tags: list[str],
) -> tuple[Path, str, str]:
    """归一化条目并打成临时 ZIP。返回 ``(zip_path, name, version)``；调用方须在完成后 ``unlink`` 该路径。

    ``entry_overrides`` 来自 ``manifest.json`` 根级对应**顶层目录名**的配置对象。
    """
    tmp = Path(tempfile.mkdtemp(prefix=f"oj_import_{entry_key}_"))
    zip_fd, zip_name = tempfile.mkstemp(prefix=f"oj_import_{entry_key}_pkg_", suffix=".zip")
    os.close(zip_fd)
    zip_path = Path(zip_name)
    try:
        staging = tmp / "staging"
        staging.mkdir()

        if is_standard_skill_entry(entry):
            shutil.copytree(entry, staging, dirs_exist_ok=True)
            name, version = validate_standard_skill_staging(staging)
        elif is_simple_skill_entry(entry):
            vo = entry_overrides.get("version")
            version_override = str(vo).strip() if vo else None
            v = version_override or version_fallback
            name, version = build_simple_skill_staging(
                entry,
                staging,
                default_version=v,
                default_author=default_author,
                default_tags=default_tags,
                display_name=None,
            )
        else:
            raise ValueError("skill_layout_unrecognized")

        build_skill_plugin_zip_to_path(staging, name, version, zip_path)
        return zip_path, name, version
    except Exception:
        zip_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
