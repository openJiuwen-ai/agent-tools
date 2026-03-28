from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from openjiuwen_plugin.logging_config import get_logger
from openjiuwen_plugin.market import PublishError
from openjiuwen_plugin.market import upload_plugin as _market_upload_plugin
from openjiuwen_plugin.schemas import PluginPublishResult, PublishPluginInput, PublishRequest

logger = get_logger(__name__)

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
TOOL_NAME_PATTERN = re.compile(r'@tool\([^)]*name\s*=\s*["\']([a-z][a-z0-9-]*)["\']', re.DOTALL)
SUPPORTED_PLUGIN_TYPES = {"tools", "mcp-stdio", "restful-api"}
# tools 类型下 plugin.yaml 的 tools_schema 约定路径
TOOLS_SCHEMA_PATH = "schemas/tools.json"


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def init_plugin(plugin_name: str, base_path: Path, force: bool = False, plugin_type: str = "tools") -> Path:
    if not NAME_PATTERN.match(plugin_name):
        raise ValueError("plugin name must match ^[a-z][a-z0-9-]*$")
    if plugin_type not in SUPPORTED_PLUGIN_TYPES:
        supported = ", ".join(sorted(SUPPORTED_PLUGIN_TYPES))
        raise ValueError(f"plugin type must be one of: {supported}")

    plugin_root = (base_path / plugin_name).resolve()
    if plugin_root.exists() and any(plugin_root.iterdir()) and not force:
        raise FileExistsError(f"{plugin_root} already exists and is not empty. Use --force to continue.")

    package_name = plugin_name.replace("-", "_")
    package_dir = plugin_root / "src" / package_name
    schemas_dir = plugin_root / "schemas"

    dirs = [package_dir, schemas_dir]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)

    (plugin_root / "README.md").write_text(
        _render_readme(plugin_name, plugin_type),
        encoding="utf-8",
    )
    (plugin_root / "icon.png").write_bytes(b"")
    (package_dir / "__init__.py").write_text(
        '"""Plugin package."""\n',
        encoding="utf-8",
    )
    (schemas_dir / "tools.json").write_text(
        json.dumps(_default_tools_schema(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if plugin_type == "tools":
        (package_dir / "plugin.py").write_text(
            _render_plugin_impl(plugin_name),
            encoding="utf-8",
        )
    elif plugin_type == "mcp-stdio":
        (package_dir / "mcp_server.py").write_text(
            _render_mcp_stdio_impl(plugin_name),
            encoding="utf-8",
        )
    else:
        # restful-api
        (package_dir / "rest_api.py").write_text(
            _render_rest_api_impl(plugin_name),
            encoding="utf-8",
        )
    (plugin_root / "plugin.yaml").write_text(
        yaml.safe_dump(
            _default_plugin_yaml(plugin_name, plugin_type, package_name),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    # install_plugin_from_zip 对所有非 tools 类型也需要 pyproject.toml 来执行 `pip install .`
    (plugin_root / "pyproject.toml").write_text(
        _default_pyproject_toml(plugin_name, package_name, plugin_type),
        encoding="utf-8",
    )
    return plugin_root


def validate_plugin(plugin_path: Path) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    root = plugin_path.resolve()
    if not root.exists():
        return ValidationResult(False, [f"plugin path not found: {root}"], warnings)

    required_entries = [
        "plugin.yaml",
        "README.md",
        "icon.png",
        "src",
    ]
    plugin_yaml_path = root / "plugin.yaml"
    plugin_data: dict[str, Any] | None = None
    if plugin_yaml_path.exists():
        plugin_data = _load_yaml(plugin_yaml_path, errors)

    runtime_type = _runtime_type(plugin_data)
    if runtime_type == "tools":
        required_entries.append("schemas/tools.json")

    for rel in required_entries:
        if not (root / rel).exists():
            errors.append(f"missing required entry: {rel}")

    tools_rel = "schemas/tools.json"
    tools_schema_path = root / tools_rel
    tools_data: dict[str, Any] | None = None
    if runtime_type == "tools" and tools_schema_path.exists():
        tools_data = _load_json(tools_schema_path, errors)

    if plugin_data is not None:
        _validate_plugin_yaml(plugin_data, root, errors, warnings)

    if tools_data is not None and runtime_type == "tools":
        _validate_tools_json(tools_data, errors)
        _validate_tool_names_consistency(plugin_data, tools_data, root, errors, warnings)

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


# dist：tools 类型由 _pack_plugin_tools 单独写入 wheel；目录打包时避免重复/误打构建产物。
# out：pack 默认输出目录，不应递归打进 zip（否则会把历史 zip/sha256 一并打入）。
_PACK_IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", ".eggs", "dist", "out"}
_PACK_IGNORE_SUFFIXES = (".pyc", ".pyo", ".egg-info")


def pack_plugin(plugin_path: Path, output_dir: Path | None = None) -> Path:
    """将插件目录打成 zip；打包前会先执行 validate，校验不通过则抛错不生成 zip。"""
    root = plugin_path.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"plugin path not found or not a directory: {root}")

    result = validate_plugin(root)
    for w in result.warnings:
        logger.warning("%s", w)
    if result.errors:
        raise ValueError("plugin validation failed: " + "; ".join(result.errors))

    plugin_yaml_path = root / "plugin.yaml"
    if not plugin_yaml_path.exists():
        raise ValueError("plugin.yaml not found")
    plugin_data = yaml.safe_load(plugin_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(plugin_data, dict):
        raise ValueError("plugin.yaml must be an object")
    name = plugin_data.get("name")
    version = plugin_data.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError("plugin.yaml name and version required")

    out = (output_dir or (root / "out")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    zip_name = f"{name}-{version}.zip"
    zip_path = out / zip_name
    prefix = f"{name}-{version}"
    runtime_type = _runtime_type(plugin_data)

    if runtime_type == "tools":
        _pack_plugin_tools(root, name, version, prefix, zip_path)
    else:
        _pack_plugin_directory(root, prefix, zip_path)

    digest = _file_sha256_hex(zip_path)
    sha256_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha256_path.write_text(f"{digest}  {zip_name}\n", encoding="utf-8")
    return zip_path


def _pack_plugin_tools(root: Path, name: str, version: str, prefix: str, zip_path: Path) -> None:
    """tools 类型：先 build wheel，再打包基础元数据、src 与 dist/*.whl。"""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        raise ValueError("tools type requires pyproject.toml in plugin root")
    wheel_dir = root / "dist"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "-w", str(wheel_dir), "--no-deps"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(f"wheel build failed: {e.stderr or e.stdout or str(e)}") from e
    whls = list(wheel_dir.glob("*.whl"))
    if not whls:
        raise ValueError("wheel build produced no .whl files in dist/")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in ("plugin.yaml", "README.md", "icon.png", "schemas/tools.json"):
            p = root / rel
            if p.is_file():
                zf.write(p, f"{prefix}/{rel}".replace("\\", "/"))
        src_dir = root / "src"
        if src_dir.is_dir():
            for fpath in src_dir.rglob("*"):
                if fpath.is_file():
                    rel = fpath.relative_to(root)
                    zf.write(fpath, f"{prefix}/{rel}".replace("\\", "/"))
        for whl in whls:
            zf.write(whl, f"{prefix}/dist/{whl.name}".replace("\\", "/"))


def _pack_plugin_directory(root: Path, prefix: str, zip_path: Path) -> None:
    """mcp-stdio / restful-api：整目录打包，沿用忽略规则。"""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in root.rglob("*"):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(root)
            parts = rel.parts
            if any(p in _PACK_IGNORE_DIRS or p.endswith(".egg-info") for p in parts):
                continue
            if fpath.suffix in _PACK_IGNORE_SUFFIXES:
                continue
            arcname = str(Path(prefix) / rel).replace("\\", "/")
            zf.write(fpath, arcname)


def _file_sha256_hex(path: Path) -> str:
    """计算文件的 SHA256，返回十六进制字符串。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_version_prefix(version: str) -> str:
    """去掉版本号前的 v/V，与 plugin.yaml 及市场 SemVer 约定一致。"""
    s = version.strip()
    if len(s) > 1 and s[0] in ("v", "V"):
        return s[1:].strip()
    return s


def publish_plugin(
    market_url: str,
    user_token: str | None,
    system_token: str | None,
    publish_input: PublishPluginInput,
) -> PluginPublishResult:
    """
    上传插件到市场。指定 zip_path 时直接上传该 zip；否则从 plugin_path 先 pack 再上传。
    成功返回市场响应 data 字典；失败抛出 PublishError（由 market 抛出）。
    """
    if publish_input.zip_path is not None:
        z = publish_input.zip_path
        if not z.is_file():
            raise PublishError(400, f"zip file not found: {z}")
        checksum_sha256 = _file_sha256_hex(z)
        pv = (
            _strip_version_prefix(publish_input.plugin_version)
            if publish_input.plugin_version
            else None
        )
        req = PublishRequest(
            user_id=publish_input.user_id,
            zip_path=z,
            checksum_sha256=checksum_sha256,
            plugin_id=publish_input.plugin_id,
            plugin_version=pv,
            version_desc=publish_input.version_desc,
            force=publish_input.force,
        )
        return _market_upload_plugin(market_url, user_token, system_token, req)
    root = publish_input.plugin_path
    if root is None:
        raise PublishError(400, "either plugin_path or zip_path must be provided")
    if not root.exists():
        raise PublishError(400, f"plugin path not found: {root}")
    if not root.is_dir():
        raise PublishError(400, f"plugin path must be a directory: {root}")

    with tempfile.TemporaryDirectory(prefix="openjiuwen_publish_") as tmp:
        out_dir = Path(tmp)
        z = pack_plugin(root, out_dir)
        checksum_sha256 = _file_sha256_hex(z)
        pv = (
            _strip_version_prefix(publish_input.plugin_version)
            if publish_input.plugin_version
            else None
        )
        req = PublishRequest(
            user_id=publish_input.user_id,
            zip_path=z,
            checksum_sha256=checksum_sha256,
            plugin_id=publish_input.plugin_id,
            plugin_version=pv,
            version_desc=publish_input.version_desc,
            force=publish_input.force,
        )
        return _market_upload_plugin(market_url, user_token, system_token, req)


def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    """解压 zip，拒绝路径穿越。"""
    dest = dest.resolve()
    for member in zf.infolist():
        out = (dest / member.filename).resolve()
        try:
            out.relative_to(dest)
        except ValueError:
            raise ValueError(f"unsafe zip entry: {member.filename!r}") from None
    zf.extractall(dest)


def _find_plugin_root_in_extracted(extract_root: Path) -> Path:
    """归档内需恰好一个 plugin.yaml，返回其所在目录（插件根目录）。"""
    found = list(extract_root.rglob("plugin.yaml"))
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one plugin.yaml in archive, found {len(found)} under {extract_root}"
        )
    return found[0].parent


def install_plugin_from_zip(zip_path: Path, *, pip_prefix: Path | None = None) -> Path:
    """
    解压市场下载的插件 zip，校验结构后按 ``runtime.type`` 执行 ``pip install``。

    - **tools**：仅安装 ``dist/*.whl``（发布包语义）；若缺失 wheel 直接失败。
    - **mcp-stdio** / **restful-api**：在插件根目录执行 ``pip install .``。

    返回解压后的插件根目录路径（位于临时目录内；调用方若需保留目录应在同进程内复制）。
    """
    zpath = zip_path.resolve()
    if not zpath.is_file():
        raise ValueError(f"zip not found: {zpath}")

    with tempfile.TemporaryDirectory(prefix="openjiuwen_install_") as tmp:
        extract_root = Path(tmp)
        with zipfile.ZipFile(zpath, "r") as zf:
            _safe_extractall(zf, extract_root)
        plugin_root = _find_plugin_root_in_extracted(extract_root)

        result = validate_plugin(plugin_root)
        for w in result.warnings:
            logger.warning("%s", w)
        if result.errors:
            raise ValueError("plugin validation failed: " + "; ".join(result.errors))

        plugin_yaml = plugin_root / "plugin.yaml"
        data = yaml.safe_load(plugin_yaml.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("plugin.yaml must be an object")
        runtime_type = _runtime_type(data)

        pip_base = [sys.executable, "-m", "pip", "install"]
        if pip_prefix is not None:
            pip_base.extend(["--prefix", str(pip_prefix.resolve())])

        try:
            if runtime_type == "tools":
                whls = sorted((plugin_root / "dist").glob("*.whl"))
                if whls:
                    subprocess.run(
                        pip_base + [str(w) for w in whls],
                        check=True,
                    )
                else:
                    raise ValueError("tools plugin zip has no dist/*.whl")
            else:
                if not (plugin_root / "pyproject.toml").is_file():
                    raise ValueError(f"{runtime_type} plugin requires pyproject.toml in plugin root")
                subprocess.run(
                    pip_base + ["."],
                    cwd=plugin_root,
                    check=True,
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pip install failed (exit {e.returncode})") from e

        logger.info("installed plugin at %s (runtime.type=%s)", plugin_root, runtime_type)
        # 返回路径在 TemporaryDirectory 销毁后会失效；将整棵插件树复制到持久临时目录
        persist = Path(tempfile.mkdtemp(prefix="openjiuwen_plugin_installed_"))
        _copy_tree(plugin_root, persist / plugin_root.name)
        return persist / plugin_root.name


def _copy_tree(src: Path, dest: Path) -> None:
    """复制目录树（仅用于 install 后保留一份可读路径）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)


def _load_yaml(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"failed to parse YAML {path}: {exc}")
        return None
    if not isinstance(loaded, dict):
        errors.append(f"YAML root must be object: {path}")
        return None
    return loaded


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"failed to parse JSON {path}: {exc}")
        return None
    if not isinstance(loaded, dict):
        errors.append(f"JSON root must be object: {path}")
        return None
    return loaded


def _runtime_type(plugin_data: dict[str, Any] | None) -> str:
    if not isinstance(plugin_data, dict):
        return "tools"
    runtime = plugin_data.get("runtime")
    if not isinstance(runtime, dict):
        return "tools"
    runtime_type = runtime.get("type")
    if isinstance(runtime_type, str) and runtime_type in SUPPORTED_PLUGIN_TYPES:
        return runtime_type
    return "tools"


def _find_package_dir(root: Path, plugin_name: str) -> Path | None:
    pkg_candidates = [root / "src" / plugin_name, root / "src" / plugin_name.replace("-", "_")]
    return next((p for p in pkg_candidates if p.exists()), None)


def _is_valid_requires_python(value: str) -> bool:
    """与 pyproject requires-python 同类：PEP 440 版本说明符，可单条或多条逗号分隔。"""
    s = value.strip()
    if not s:
        return False
    try:
        SpecifierSet(s)
    except InvalidSpecifier:
        return False
    return True


def _validate_plugin_yaml(
    plugin_data: dict[str, Any],
    root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    runtime_type = _runtime_type(plugin_data)
    expected_tools_path = TOOLS_SCHEMA_PATH
    name = plugin_data.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        errors.append("plugin.yaml name must match ^[a-z][a-z0-9-]*$")

    version = plugin_data.get("version")
    if not isinstance(version, str) or not SEMVER_PATTERN.match(version):
        errors.append("plugin.yaml version must be semver, e.g. 1.2.3")

    display_name = plugin_data.get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        errors.append("plugin.yaml display_name must be non-empty string")

    description = plugin_data.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("plugin.yaml description must be non-empty string")

    runtime = plugin_data.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("plugin.yaml runtime must be object")
    else:
        declared_type = runtime.get("type")
        if declared_type not in SUPPORTED_PLUGIN_TYPES:
            supported = ", ".join(sorted(SUPPORTED_PLUGIN_TYPES))
            errors.append(f"runtime.type must be one of: {supported}")

    metadata = plugin_data.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("plugin.yaml metadata must be object")
    else:
        author = metadata.get("author")
        if not isinstance(author, str) or not author.strip():
            errors.append("metadata.author must be non-empty string")
        tags = metadata.get("tags")
        if not isinstance(tags, list) or not all(isinstance(t, str) and t.strip() for t in tags):
            errors.append("metadata.tags must be array of non-empty strings")

    compatibility = plugin_data.get("compatibility")
    if not isinstance(compatibility, dict):
        errors.append("plugin.yaml compatibility must be object")
    else:
        if "python" not in compatibility:
            errors.append("compatibility.python is required")
        else:
            py_val = compatibility["python"]
            if not isinstance(py_val, str):
                errors.append("compatibility.python must be a string")
            elif not _is_valid_requires_python(py_val):
                errors.append(
                    "compatibility.python must be PEP 440 version specifiers "
                    "(e.g. '>=3.11' or '>=3.11, <3.14'), same idea as pyproject requires-python"
                )

    if runtime_type == "tools":
        tools_schema = plugin_data.get("tools_schema")
        if tools_schema is None:
            warnings.append(f"plugin.yaml tools_schema missing, defaulting to {expected_tools_path}")
        elif not isinstance(tools_schema, str):
            errors.append("plugin.yaml tools_schema must be string path")
        elif tools_schema != expected_tools_path:
            errors.append(f"plugin.yaml tools_schema must be '{expected_tools_path}'")
        elif not (root / tools_schema).exists():
            errors.append(f"tools schema file not found: {tools_schema}")
    elif runtime_type == "mcp-stdio":
        mcp_data = plugin_data.get("mcp")
        if not isinstance(mcp_data, dict):
            errors.append("plugin.yaml mcp must be object for mcp-stdio type")
        else:
            if mcp_data.get("transport") != "stdio":
                errors.append("mcp.transport must be 'stdio'")
            command = mcp_data.get("command")
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(x, str) and x.strip() for x in command)
            ):
                errors.append("mcp.command must be non-empty string array")
    elif runtime_type == "restful-api":
        api_data = plugin_data.get("api")
        if not isinstance(api_data, dict):
            errors.append("plugin.yaml api must be object for restful-api type")
        elif not isinstance(api_data.get("base_url"), str) or not api_data.get("base_url", "").strip():
            errors.append("api.base_url must be non-empty string")

    # 不再根据 plugin.yaml 的 runtime/name 去强校验具体源码文件名；
    # 文件结构只要求存在 src 目录，避免实现入口文件名被写死。


def _validate_tools_json(tools_data: dict[str, Any], errors: list[str]) -> None:
    tools = tools_data.get("tools")
    if not isinstance(tools, list) or not tools:
        errors.append("tools.json tools must be non-empty array")
        return

    seen_names: set[str] = set()
    for i, tool in enumerate(tools):
        path = f"tools[{i}]"
        if not isinstance(tool, dict):
            errors.append(f"{path} must be object")
            continue

        name = tool.get("name")
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            errors.append(f"{path}.name must match ^[a-z][a-z0-9-]*$")
        elif name in seen_names:
            errors.append(f"duplicate tool name: {name}")
        else:
            seen_names.add(name)

        description = tool.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{path}.description must be non-empty string")

        for schema_key in ("input_schema", "output_schema"):
            schema_obj = tool.get(schema_key)
            if not isinstance(schema_obj, dict):
                errors.append(f"{path}.{schema_key} must be object")
                continue
            if schema_obj.get("type") != "object":
                errors.append(f"{path}.{schema_key}.type must be 'object'")


def _validate_tool_names_consistency(
    plugin_data: dict[str, Any] | None,
    tools_data: dict[str, Any],
    root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(plugin_data, dict):
        return
    plugin_name = plugin_data.get("name")
    if not isinstance(plugin_name, str) or not NAME_PATTERN.match(plugin_name):
        return

    package_path = root / "src" / plugin_name / "plugin.py"
    fallback_path = root / "src" / plugin_name.replace("-", "_") / "plugin.py"
    plugin_py = package_path if package_path.exists() else fallback_path
    if not plugin_py.exists():
        return

    tools = tools_data.get("tools")
    if not isinstance(tools, list):
        return

    schema_tool_names = {t["name"] for t in tools if isinstance(t.get("name"), str)}

    text = plugin_py.read_text(encoding="utf-8")
    code_tool_names = set(TOOL_NAME_PATTERN.findall(text))
    if not code_tool_names:
        warnings.append(f"no @tool(name=...) found in {plugin_py.relative_to(root)}")
        return

    missing_in_schema = sorted(code_tool_names - schema_tool_names)
    missing_in_code = sorted(schema_tool_names - code_tool_names)
    if missing_in_schema:
        errors.append(f"tools missing in schemas/tools.json: {', '.join(missing_in_schema)}")
    if missing_in_code:
        errors.append(f"tools in schemas/tools.json not found in plugin.py: {', '.join(missing_in_code)}")


def _default_plugin_yaml(plugin_name: str, plugin_type: str, package_name: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": plugin_name,
        "version": "0.1.0",
        "display_name": plugin_name.replace("-", " ").title(),
        "description": "TODO: describe your plugin",
        "runtime": {
            "type": plugin_type,
        },
        "metadata": {
            "author": "TODO: your name",
            "tags": ["demo"],
        },
        "compatibility": {
            "python": ">=3.11, <3.14",
        },
    }
    if plugin_type == "tools":
        base["tools_schema"] = "schemas/tools.json"
    elif plugin_type == "mcp-stdio":
        base["mcp"] = {
            "transport": "stdio",
            "command": ["python", "-m", f"{package_name}.mcp_server"],
        }
    else:
        # restful-api
        base["api"] = {
            "base_url": "TODO: your API base URL",
        }
    return base


def _default_tools_schema() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": "example",
                "description": "TODO: describe your tool",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "output_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            }
        ],
    }


def _render_plugin_impl(plugin_name: str) -> str:
    return f'''"""Plugin implementation for {plugin_name}."""

from openjiuwen.core.foundation.tool import tool


@tool(
    name="example",
    description="TODO: describe your tool",
    input_params={{}},
)
def example() -> dict:
    return {{}}
'''


def _render_mcp_stdio_impl(plugin_name: str) -> str:
    return f'''"""MCP stdio server for {plugin_name} (FastMCP)."""

from fastmcp import FastMCP

mcp = FastMCP("{plugin_name}")


@mcp.tool
def greet(name: str) -> str:
    """Say hello to the given name."""
    return f"Hello, {{name}}!"


if __name__ == "__main__":
    mcp.run()
'''


def _render_rest_api_impl(plugin_name: str) -> str:
    return f'''"""REST API entry for {plugin_name}."""

# TODO: implement your REST API client or server
# This module is the placeholder for restful-api type.
'''


def _default_pyproject_toml(plugin_name: str, package_name: str, plugin_type: str) -> str:
    # 注意：这里不填死版本号，只声明运行该脚手架所需的最小依赖。
    dependencies: list[str] = []
    if plugin_type == "mcp-stdio":
        dependencies = ["fastmcp"]

    deps_toml = ""
    if dependencies:
        # PEP 621: `dependencies = [...]`
        deps_toml = "\ndependencies = [" + ", ".join(f"\"{d}\"" for d in dependencies) + "]\n"

    return f"""[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{plugin_name}"
version = "0.1.0"
description = "openJiuwen plugin"
readme = "README.md"
requires-python = ">=3.11"
{deps_toml}

[tool.setuptools.packages.find]
where = ["src"]
include = ["{package_name}*"]
"""


def _render_readme(plugin_name: str, plugin_type: str) -> str:
    if plugin_type == "mcp-stdio":
        type_notes = "- `src/<package>/mcp_server.py`: MCP stdio entrypoint\n"
    elif plugin_type == "restful-api":
        type_notes = "- `src/<package>/rest_api.py`: REST API entry\n"
    else:
        type_notes = "- `schemas/tools.json`: tool definitions\n- `src/<package>/plugin.py`: tool implementation\n"
    return f"""# {plugin_name}

openjiuwen plugin scaffold.

## Structure

- `plugin.yaml`: plugin metadata and compatibility
{type_notes}"""


def info_plugin(plugin_path: Path) -> dict[str, Any]:
    """读取本地插件目录的 README、CHANGELOG 及 plugin.yaml 的 name/version，供展示。"""
    root = plugin_path.resolve()
    result: dict[str, Any] = {"name": None, "version": None, "readme": None, "changelog": None}
    if not root.exists() or not root.is_dir():
        return result
    plugin_yaml = root / "plugin.yaml"
    if plugin_yaml.exists():
        try:
            data = yaml.safe_load(plugin_yaml.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                result["name"] = data.get("name")
                result["version"] = data.get("version")
        except Exception as exc:
            logger.warning("failed to read/parse plugin.yaml: %s", exc)
    readme_path = root / "README.md"
    if readme_path.is_file():
        try:
            result["readme"] = readme_path.read_text(encoding="utf-8")
        except Exception:
            result["readme"] = "(read failed)"
    changelog_path = root / "CHANGELOG.md"
    if changelog_path.is_file():
        try:
            result["changelog"] = changelog_path.read_text(encoding="utf-8")
        except Exception:
            result["changelog"] = "(read failed)"
    return result
