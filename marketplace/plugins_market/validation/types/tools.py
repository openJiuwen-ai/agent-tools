"""Tools 插件：zip 布局与 schemas/tools.json 形态校验（以 wheel 包为准）。"""

from __future__ import annotations

import json
import zipfile

from plugins_market.core.errors import PublishError
from plugins_market.validation.base import raise_invalid_structure
from plugins_market.validation.constants import (
    MAX_JSON_BYTES,
    TOOL_NAME_PATTERN,
)
from plugins_market.validation.zip_utils import (
    DecompressCounter,
    has_dist_wheels,
    safe_read_zip_member,
    validate_png_icon_bytes,
)


def validate_tools_layout(
    zf: zipfile.ZipFile,
    prefix: str,
    counter: DecompressCounter,
) -> dict:
    """校验 tools 包根目录：README、icon、``dist/*.whl``、schemas/tools.json，并校验 icon 为合法 PNG。

    Returns dict with keys: icon_path, icon_bytes, readme_path, tools_json_path.
    """
    names = set(zf.namelist())

    readme_path = prefix + "README.md"
    if readme_path not in names:
        raise_invalid_structure(
            "插件包结构不符合要求：tools 类型缺少 README.md"
        )

    icon_path = prefix + "icon.png"
    if icon_path not in names:
        raise_invalid_structure(
            "插件包结构不符合要求：tools 类型缺少 icon.png"
        )

    if not has_dist_wheels(names, prefix):
        raise_invalid_structure(
            "插件包结构不符合要求：tools 类型缺少 dist/ 下的 .whl 文件"
        )

    tools_json_path = prefix + "schemas/tools.json"
    if tools_json_path not in names:
        raise_invalid_structure(
            "插件包结构不符合要求：tools 类型缺少 schemas/tools.json"
        )

    icon_bytes = safe_read_zip_member(zf, icon_path, counter)
    validate_png_icon_bytes(icon_bytes, path=icon_path)

    return {
        "icon_path": icon_path,
        "icon_bytes": icon_bytes,
        "readme_path": readme_path,
        "tools_json_path": tools_json_path,
    }


def validate_tools_json(raw_bytes: bytes) -> list[dict]:
    """解析并校验 schemas/tools.json：tools 数组、各 tool 字段与 JSON Schema 形态约束。

    Returns the parsed tools list.
    """
    if len(raw_bytes) > MAX_JSON_BYTES:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message=f"schemas/tools.json 超过大小上限（最大 {MAX_JSON_BYTES // (1024 * 1024)} MB）",
        )

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message=f"schemas/tools.json 编码必须为 UTF-8：{exc}",
        ) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message=f"schemas/tools.json 解析失败：{exc}",
        ) from exc

    tools = data.get("tools") if isinstance(data, dict) else None
    if not isinstance(tools, list) or len(tools) == 0:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message="schemas/tools.json 中 tools 字段必须为非空数组",
        )

    seen_names: set[str] = set()
    for idx, tool in enumerate(tools):
        path = f"tools[{idx}]"

        if not isinstance(tool, dict):
            raise PublishError(
                code=400,
                error="invalid_plugin_structure",
                message=f"schemas/tools.json {path} 必须为 object",
            )

        t_name = tool.get("name")
        if not isinstance(t_name, str) or not TOOL_NAME_PATTERN.match(t_name):
            raise PublishError(
                code=400,
                error="invalid_plugin_structure",
                message=(
                    f"schemas/tools.json {path}.name 必须匹配 ^[a-z][a-z0-9-]*$，"
                    f"实际值: {t_name!r}"
                ),
            )
        if t_name in seen_names:
            raise PublishError(
                code=400,
                error="invalid_plugin_structure",
                message=f"schemas/tools.json 中 tool name {t_name!r} 重复",
            )
        seen_names.add(t_name)

        t_desc = tool.get("description")
        if not isinstance(t_desc, str) or not t_desc.strip():
            raise PublishError(
                code=400,
                error="invalid_plugin_structure",
                message=f"schemas/tools.json {path}.description 必须为非空字符串",
            )

        for schema_field in ("input_schema", "output_schema"):
            schema = tool.get(schema_field)
            if not isinstance(schema, dict):
                raise PublishError(
                    code=400,
                    error="invalid_plugin_structure",
                    message=f"schemas/tools.json {path}.{schema_field} 必须为 object",
                )
            if schema.get("type") != "object":
                raise PublishError(
                    code=400,
                    error="invalid_plugin_structure",
                    message=(
                        f"schemas/tools.json {path}.{schema_field}.type "
                        f'必须为 "object"，实际值: {schema.get("type")!r}'
                    ),
                )

    return tools
