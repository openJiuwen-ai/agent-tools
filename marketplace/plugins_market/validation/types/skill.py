"""Skill 插件：zip 目录布局与 SKILL.md frontmatter 校验。"""

from __future__ import annotations

import zipfile
from typing import Any

from plugins_market.core.errors import PublishError
from plugins_market.validation.base import raise_invalid_skill_md, raise_invalid_structure
from plugins_market.validation.constants import (
    MAX_YAML_BYTES,
    SKILL_DESC_MAX_LEN,
    SKILL_NAME_MAX_LEN,
    SKILL_NAME_PATTERN,
)
from plugins_market.validation.plugin_yaml import safe_load_yaml
from plugins_market.validation.zip_utils import (
    DecompressCounter,
    safe_read_zip_member,
    validate_png_icon_bytes,
)


def _non_hidden_subdirs(names: set[str], prefix: str) -> list[str]:
    """Return sorted list of non-hidden immediate subdirectory names under *prefix*."""
    seen: set[str] = set()
    for entry in names:
        normalized = entry.replace("\\", "/")
        if not normalized.startswith(prefix):
            continue
        rest = normalized[len(prefix):]
        if "/" not in rest:
            continue  # file directly under prefix, not a subdir
        subdir_name = rest.split("/")[0]
        if subdir_name.startswith("."):
            continue  # hidden
        seen.add(subdir_name)
    return sorted(seen)


def validate_skill_layout(
    zf: zipfile.ZipFile,
    prefix: str,
    plugin_name: str,
    counter: DecompressCounter,
) -> dict:
    """校验 skill 包结构：根目录 icon、唯一 skill 子目录、其下 SKILL.md、可选 README。

    Returns dict with keys: icon_path, icon_bytes, skill_md_path, readme_path (may be "").
    """
    names = set(zf.namelist())

    icon_zip_path = prefix + "icon.png"
    if icon_zip_path not in names:
        raise_invalid_structure("插件包结构不符合要求：skill 类型缺少 icon.png")

    subdirs = _non_hidden_subdirs(names, prefix)
    if len(subdirs) == 0:
        raise_invalid_structure(
            "插件包结构不符合要求：skill 类型必须有且仅有一个非隐藏 skill 子目录"
        )
    if len(subdirs) > 1:
        raise_invalid_structure(
            f"插件包结构不符合要求：skill 类型发现多个非隐藏子目录 {subdirs}，"
            "必须有且仅有一个"
        )
    actual_subdir = subdirs[0]

    if actual_subdir != plugin_name:
        raise_invalid_structure(
            f"skill 子目录名 {actual_subdir!r} 与 plugin.yaml name {plugin_name!r} 不一致"
        )

    skill_md_path = f"{prefix}{plugin_name}/SKILL.md"
    if skill_md_path not in names:
        raise_invalid_structure(
            f"插件包结构不符合要求：skill 类型缺少 {plugin_name}/SKILL.md"
        )

    readme_path = prefix + "README.md"
    if readme_path not in names:
        readme_path = ""

    icon_bytes = safe_read_zip_member(zf, icon_zip_path, counter)
    validate_png_icon_bytes(icon_bytes, path=icon_zip_path)

    return {
        "icon_path": icon_zip_path,
        "icon_bytes": icon_bytes,
        "skill_md_path": skill_md_path,
        "readme_path": readme_path,
    }


# ---------------------------------------------------------------------------
# SKILL.md frontmatter
# ---------------------------------------------------------------------------

def parse_skill_frontmatter(raw_bytes: bytes) -> tuple[dict[str, Any], str]:
    """Parse SKILL.md frontmatter with byte-limit and strict error reporting.

    Returns (frontmatter_dict, body_text).

    失败情形包括：缺少或格式错误的 --- 围栏、YAML 解析失败、根节点非 mapping。
    """
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise_invalid_skill_md(f"SKILL.md 编码必须为 UTF-8：{exc}")
    text = text.lstrip("\ufeff")

    if not text.startswith("---"):
        raise_invalid_skill_md(
            "SKILL.md frontmatter 格式错误：文件必须以 '---' 开头"
        )

    lines = text.splitlines()
    if len(lines) < 2 or lines[0].strip() != "---":
        raise_invalid_skill_md(
            "SKILL.md frontmatter 格式错误：开头 '---' 行格式不正确"
        )

    end_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        raise_invalid_skill_md(
            "SKILL.md frontmatter 格式错误：缺少闭合 '---'"
        )

    fm_text = "\n".join(lines[1:end_idx]).strip()
    body = "\n".join(lines[end_idx + 1:]).lstrip("\n")

    fm_bytes = fm_text.encode("utf-8")
    if len(fm_bytes) > MAX_YAML_BYTES:
        raise_invalid_skill_md(
            f"SKILL.md frontmatter 超过大小上限（最大 {MAX_YAML_BYTES // 1024} KB）"
        )

    if fm_text:
        fm = safe_load_yaml(fm_text, context="SKILL.md frontmatter")
    else:
        fm = {}

    if not isinstance(fm, dict):
        raise_invalid_skill_md(
            "SKILL.md frontmatter 格式错误：根类型必须为 mapping（字典），"
            f"实际类型为 {type(fm).__name__}"
        )

    return fm, body


def validate_skill_frontmatter(
    fm: dict[str, Any],
    *,
    dir_name: str,
    yaml_name: str,
) -> str:
    """校验 frontmatter 的 name / description，并与目录名、plugin.yaml 对齐。

    Returns frontmatter description.
    """
    fm_name = fm.get("name")
    if not isinstance(fm_name, str):
        raise_invalid_skill_md(
            f"SKILL.md frontmatter name 必须为字符串，"
            f"实际类型为 {type(fm_name).__name__}"
        )
    name_stripped = fm_name.strip()

    if len(name_stripped) > SKILL_NAME_MAX_LEN:
        raise_invalid_skill_md(
            f"SKILL.md frontmatter name 长度不得超过 {SKILL_NAME_MAX_LEN} 个字符"
        )
    if not SKILL_NAME_PATTERN.match(name_stripped):
        raise_invalid_skill_md(
            "SKILL.md frontmatter name 格式不符合 skill slug 规则"
        )

    if name_stripped != dir_name:
        raise_invalid_skill_md(
            f"SKILL.md frontmatter name {name_stripped!r} 与 skill 子目录名 {dir_name!r} 不一致"
        )
    if name_stripped != yaml_name:
        raise_invalid_skill_md(
            f"SKILL.md frontmatter name {name_stripped!r} 与 plugin.yaml name {yaml_name!r} 不一致"
        )

    fm_desc = fm.get("description")
    if not isinstance(fm_desc, str):
        raise_invalid_skill_md(
            f"SKILL.md frontmatter description 必须为字符串，"
            f"实际类型为 {type(fm_desc).__name__}"
        )
    desc_stripped = fm_desc.strip()
    if not desc_stripped:
        raise_invalid_skill_md(
            "SKILL.md frontmatter description 必填且不得为空"
        )
    if len(desc_stripped) > SKILL_DESC_MAX_LEN:
        raise_invalid_skill_md(
            f"SKILL.md frontmatter description 长度不得超过 {SKILL_DESC_MAX_LEN} 个字符"
        )

    return desc_stripped
