"""Tools 插件：zip 布局、schemas/tools.json 与源码中 @tool 声明一致性校验。"""

from __future__ import annotations

import ast
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
    has_src_tree,
    safe_read_zip_member,
    validate_png_icon_bytes,
)


def validate_tools_layout(
    zf: zipfile.ZipFile,
    prefix: str,
    counter: DecompressCounter,
) -> dict:
    """校验 tools 包根目录：README、icon、非空 src/、schemas/tools.json，并校验 icon 为合法 PNG。

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

    if not has_src_tree(names, prefix):
        raise_invalid_structure(
            "插件包结构不符合要求：tools 类型缺少 src/ 目录"
        )

    tools_json_path = prefix + "schemas/tools.json"
    if tools_json_path not in names:
        raise_invalid_structure(
            "插件包结构不符合要求：tools 类型缺少 schemas/tools.json"
        )

    pyproject_path = prefix + "pyproject.toml"
    if pyproject_path not in names:
        raise_invalid_structure(
            "插件包结构不符合要求：tools 类型缺少 pyproject.toml"
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


def validate_tools_schema_consistency(
    zf: zipfile.ZipFile,
    prefix: str,
    plugin_name: str,
    tools: list[dict],
    counter: DecompressCounter,
) -> None:
    """保证 schemas/tools.json 中的 tool name 与 src/.../plugin.py 里 @tool(name=...) 完全一致（双向）。"""
    plugin_name_underscored = plugin_name.replace("-", "_")
    candidate_paths = [
        f"{prefix}src/{plugin_name}/plugin.py",
        f"{prefix}src/{plugin_name_underscored}/plugin.py",
    ]
    names = set(zf.namelist())
    plugin_py_path: str | None = None
    for p in candidate_paths:
        if p in names:
            plugin_py_path = p
            break

    if plugin_py_path is None:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message=(
                f"tools 类型缺少 src/{plugin_name}/plugin.py "
                f"（或 src/{plugin_name_underscored}/plugin.py）"
            ),
        )

    src_bytes = safe_read_zip_member(zf, plugin_py_path, counter)
    try:
        src_text = src_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message=f"{plugin_py_path} 编码必须为 UTF-8：{exc}",
        ) from exc

    schema_names = {t["name"] for t in tools if isinstance(t.get("name"), str)}
    code_names = _extract_tool_names_from_source(src_text, plugin_py_path)

    # If no literal @tool names found (all dynamic / non-literal), skip consistency check.
    if not code_names:
        return

    missing_in_schema = sorted(code_names - schema_names)
    if missing_in_schema:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message=(
                f"plugin.py 中存在但 schemas/tools.json 中缺失的 tool name：{missing_in_schema}"
            ),
        )

    missing_in_code = sorted(schema_names - code_names)
    if missing_in_code:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message=(
                f"schemas/tools.json 中声明但 plugin.py 中未找到的 tool name：{missing_in_code}"
            ),
        )


def _extract_tool_names_from_source(src_text: str, src_path: str) -> set[str]:
    """Extract literal tool names from @tool(...) decorators using AST."""
    try:
        tree = ast.parse(src_text, filename=src_path)
    except SyntaxError as exc:
        raise PublishError(
            code=400,
            error="invalid_plugin_structure",
            message=f"{src_path} 语法错误，无法解析 @tool 装饰器：{exc}",
        ) from exc

    names: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue

            func = dec.func
            is_tool_call = (
                isinstance(func, ast.Name) and func.id == "tool"
            ) or (
                isinstance(func, ast.Attribute) and func.attr == "tool"
            )
            if not is_tool_call:
                continue

            literal_name: str | None = None
            for kw in dec.keywords:
                if (
                    kw.arg == "name"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    literal_name = kw.value.value
                    break

            if literal_name is None and dec.args:
                arg0 = dec.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    literal_name = arg0.value

            # Non-literal name (e.g. constant or dynamic value): skip rather than error.
            if literal_name is None:
                continue

            names.add(literal_name)

    return names
