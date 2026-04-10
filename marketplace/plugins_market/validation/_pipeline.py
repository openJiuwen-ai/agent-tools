"""Validation pipeline: orchestrates all checks for a plugin zip upload.

Called by services/plugin.py::publish().
"""

from __future__ import annotations

import io
import zipfile
from typing import Any

from plugins_market.core.errors import PublishError
from plugins_market.validation.constants import (
    RUNTIME_MCP_STDIO,
    RUNTIME_RESTFUL_API,
    RUNTIME_SKILL,
    RUNTIME_TOOLS,
)
from plugins_market.validation.plugin_yaml import (
    PluginYamlPublicFields,
    safe_load_yaml,
    validate_plugin_yaml_bytes,
    validate_plugin_yaml_public,
)
from plugins_market.validation.zip_utils import (
    DecompressCounter,
    safe_read_zip_member,
    validate_zip_safety,
)
from plugins_market.validation.types.skill import (
    parse_skill_frontmatter,
    validate_skill_frontmatter,
    validate_skill_layout,
)
from plugins_market.validation.types.tools import (
    validate_tools_json,
    validate_tools_layout,
    validate_tools_schema_consistency,
)
from plugins_market.validation.types.mcp_stdio import validate_mcp_stdio_layout
from plugins_market.validation.types.restful_api import validate_restful_api_layout


def _find_plugin_yaml_path(zf: zipfile.ZipFile) -> str | None:
    """Accept only the standard layout: <top>/plugin.yaml (exactly 2 path segments)."""
    for name in zf.namelist():
        normalized = name.replace("\\", "/").strip("/")
        if not normalized:
            continue
        parts = normalized.split("/")
        if len(parts) == 2 and parts[-1] == "plugin.yaml":
            return name
    return None


def _plugin_prefix(plugin_yaml_path: str) -> str:
    """Return directory prefix of plugin.yaml, e.g. 'myplugin/' or ''."""
    path = plugin_yaml_path.replace("\\", "/").strip("/")
    if "/" not in path:
        return ""
    return path.rsplit("/", 1)[0] + "/"


def extract_plugin_metadata(content: bytes) -> dict[str, Any]:
    """Full validation pipeline for a plugin zip.

    Steps:
      1. Open zip (BadZipFile guard)
      2. zip safety pre-check (validate_zip_safety)
      3. Find and read plugin.yaml with streaming counter
      4. Parse plugin.yaml with bounded SafeLoader
      5. Validate public fields
      6. Type-specific layout + content validation
      7. Read icon and README with streaming counter

    Returns dict suitable for services/plugin.py::publish().
    """
    # ------------------------------------------------------------------
    # Open zip (magic bytes already verified before this call)
    # ------------------------------------------------------------------
    try:
        zf_obj = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise PublishError(
            code=400,
            error="invalid_plugin_config",
            message="上传文件不是有效的 ZIP 格式，请检查文件是否损坏或格式是否正确",
        ) from exc

    with zf_obj as zf:
        # ----------------------------------------------------------------
        # Layer 1: metadata pre-check
        # ----------------------------------------------------------------
        validate_zip_safety(zf)

        # Shared streaming counter for ALL reads inside this zip
        counter = DecompressCounter()

        # ----------------------------------------------------------------
        # Locate plugin.yaml
        # ----------------------------------------------------------------
        plugin_yaml_path = _find_plugin_yaml_path(zf)
        if not plugin_yaml_path:
            raise PublishError(
                code=400,
                error="invalid_plugin_config",
                message="plugin.yaml 配置文件格式错误或缺失",
            )

        # ----------------------------------------------------------------
        # Read plugin.yaml (byte limit enforced)
        # ----------------------------------------------------------------
        yaml_raw = safe_read_zip_member(zf, plugin_yaml_path, counter)
        yaml_text = validate_plugin_yaml_bytes(yaml_raw)

        # ----------------------------------------------------------------
        # Parse & validate plugin.yaml
        # ----------------------------------------------------------------
        yaml_data = safe_load_yaml(yaml_text, context="plugin.yaml")
        if not isinstance(yaml_data, dict):
            raise PublishError(
                code=400,
                error="invalid_plugin_config",
                message="plugin.yaml 根结构必须为 mapping（字典）",
            )

        public: PluginYamlPublicFields = validate_plugin_yaml_public(yaml_data)

        version_raw = yaml_data.get("version")
        version = str(version_raw).strip() if version_raw is not None else ""

        prefix = _plugin_prefix(plugin_yaml_path)
        rt = public.runtime_type

        # ----------------------------------------------------------------
        # Type-specific layout validation + icon/readme reads
        # ----------------------------------------------------------------
        detail_desc: str = ""
        icon_bytes: bytes = b""

        if rt == RUNTIME_SKILL:
            layout = validate_skill_layout(zf, prefix, public.name, counter)

            # Read SKILL.md and validate frontmatter
            skill_md_raw = safe_read_zip_member(zf, layout["skill_md_path"], counter)
            fm, _ = parse_skill_frontmatter(skill_md_raw)
            fm_desc = validate_skill_frontmatter(
                fm, dir_name=public.name, yaml_name=public.name
            )

            # detail_desc: prefer README.md; fall back to frontmatter description
            if layout["readme_path"]:
                readme_raw = safe_read_zip_member(zf, layout["readme_path"], counter)
                detail_desc = readme_raw.decode("utf-8", errors="replace")
            else:
                detail_desc = fm_desc

            icon_bytes = layout["icon_bytes"]

        elif rt == RUNTIME_TOOLS:
            layout = validate_tools_layout(zf, prefix, counter)

            # Read and validate tools.json
            tools_json_raw = safe_read_zip_member(zf, layout["tools_json_path"], counter)
            tools_list = validate_tools_json(tools_json_raw)

            # tool names consistency
            validate_tools_schema_consistency(
                zf, prefix, public.name, tools_list, counter
            )

            readme_raw = safe_read_zip_member(zf, layout["readme_path"], counter)
            detail_desc = readme_raw.decode("utf-8", errors="replace")
            icon_bytes = layout["icon_bytes"]

        elif rt == RUNTIME_MCP_STDIO:
            layout = validate_mcp_stdio_layout(zf, prefix, counter)
            readme_raw = safe_read_zip_member(zf, layout["readme_path"], counter)
            detail_desc = readme_raw.decode("utf-8", errors="replace")
            icon_bytes = layout["icon_bytes"]

        elif rt == RUNTIME_RESTFUL_API:
            layout = validate_restful_api_layout(zf, prefix, counter)
            readme_raw = safe_read_zip_member(zf, layout["readme_path"], counter)
            detail_desc = readme_raw.decode("utf-8", errors="replace")
            icon_bytes = layout["icon_bytes"]

        else:
            # Should not reach here; validate_plugin_yaml_public already guards this
            raise PublishError(
                code=400,
                error="invalid_plugin_config",
                message=f"不支持的 runtime.type: {rt!r}",
            )

    return {
        "name": public.name,
        "display_name": public.display_name,
        "version": version,
        "short_desc": public.short_desc,
        "detail_desc": detail_desc,
        "tags": public.tags,
        "publisher_name": public.publisher_name,
        "plugin_type": public.runtime_type,
        "icon_bytes": icon_bytes,
    }
