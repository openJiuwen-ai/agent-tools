from __future__ import annotations

import json
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from openjiuwen_plugin.utils import sha256_file_hex
from openjiuwen_plugin.logging_config import get_logger
from openjiuwen_plugin.market import PublishError, plugin_upload
from openjiuwen_plugin.schemas import (
    MARKETPLACE_VERSION_PATTERN,
    PluginPublishResult,
    PublishPluginInput,
    PublishRequest,
)

logger = get_logger(__name__)

_PLACEHOLDER_ICON_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)

PACK_IGNORE_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".eggs",
        "dist",
        "out",
        "__MACOSX",
    }
)
PACK_IGNORE_SUFFIXES = (".pyc", ".pyo", ".egg-info")
SKILL_IMPORT_BUNDLE_MAX_BYTES = 512 * 1024 * 1024


def plugin_zip_write_directory_tree(
    zf: zipfile.ZipFile,
    root: Path,
    *,
    arcname_prefix: str = "",
) -> int:
    """Append files under ``root`` to ``zf``; return number of files written."""
    base = root.resolve()
    prefix = arcname_prefix.strip().replace("\\", "/").strip("/")
    n = 0
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base)
        parts = rel.parts
        if any(p in PACK_IGNORE_DIR_NAMES or p.endswith(".egg-info") for p in parts):
            continue
        if path.suffix in PACK_IGNORE_SUFFIXES:
            continue
        if path.name == ".DS_Store":
            continue
        rel_posix = rel.as_posix()
        arcname = f"{prefix}/{rel_posix}" if prefix else rel_posix
        zf.write(path, arcname)
        n += 1
    return n


def plugin_pack_skill_bundle(src_dir: Path, dest_zip: Path) -> None:
    """Build collection-bundle zip (zip root = ``src_dir``) for ``skill-import`` CLI upload."""
    src = src_dir.resolve()
    if not src.is_dir():
        raise ValueError(f"not a directory: {src}")
    dest_zip = dest_zip.resolve()
    dest_zip.parent.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            files_written = plugin_zip_write_directory_tree(zf, src, arcname_prefix="")
    except OSError as e:
        dest_zip.unlink(missing_ok=True)
        raise ValueError(f"failed to build bundle zip: {e}") from e

    if files_written == 0:
        dest_zip.unlink(missing_ok=True)
        raise ValueError(
            "no files packed (directory empty or only ignored paths such as .git / __pycache__)"
        )

    raw_size = dest_zip.stat().st_size
    if raw_size > SKILL_IMPORT_BUNDLE_MAX_BYTES:
        dest_zip.unlink(missing_ok=True)
        raise ValueError(
            f"packed bundle size {raw_size} bytes exceeds limit {SKILL_IMPORT_BUNDLE_MAX_BYTES} "
            "(same as server MAX_FILE_SIZE / 512MB)"
        )


NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_NAME_MAX_LEN = 64
SKILL_DESC_MAX_LEN = 1024
TOOL_NAME_PATTERN = re.compile(r'@tool\([^)]*name\s*=\s*["\']([a-z][a-z0-9-]*)["\']', re.DOTALL)
SUPPORTED_PLUGIN_TYPES = {"tools", "mcp-stdio", "restful-api", "skill"}
TOOLS_SCHEMA_PATH = "schemas/tools.json"


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def _validate_skill_slug(value: str, *, field: str = "name") -> str | None:
    if len(value) > SKILL_NAME_MAX_LEN:
        return f"{field} must be at most {SKILL_NAME_MAX_LEN} characters (Agent Skills rules)"
    if not SKILL_NAME_PATTERN.match(value):
        return (
            f"{field} must use lowercase letters, digits, and single hyphens between segments "
            "(no leading/trailing hyphen, no '--')"
        )
    return None


def _parse_skill_frontmatter(skill_md: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"cannot read SKILL.md: {e}"
    if not text.lstrip().startswith("---"):
        return None, "SKILL.md must start with YAML frontmatter (---)"
    rest = text.lstrip()[3:].lstrip("\n")
    end = rest.find("\n---")
    if end == -1:
        return None, "SKILL.md frontmatter not closed (missing closing ---)"
    raw_fm = rest[:end]
    try:
        fm = yaml.safe_load(raw_fm)
    except yaml.YAMLError as e:
        return None, f"invalid SKILL.md frontmatter YAML: {e}"
    if not isinstance(fm, dict):
        return None, "SKILL.md frontmatter must be a mapping"
    return fm, None


def _validate_skill_frontmatter_fields(
    fm: dict[str, Any], skill_dir_name: str, yaml_name: str
) -> list[str]:
    errors: list[str] = []
    name_val = fm.get("name")
    if not isinstance(name_val, str):
        errors.append("SKILL.md frontmatter name is required and must be a string")
    else:
        nm = name_val.strip()
        skill_nm_err = _validate_skill_slug(nm, field="SKILL.md frontmatter name")
        if skill_nm_err:
            errors.append(skill_nm_err)
        elif nm != skill_dir_name:
            errors.append("SKILL.md frontmatter name must equal the skill directory name")
        elif nm != yaml_name:
            errors.append("SKILL.md frontmatter name must equal plugin.yaml name")

    desc_val = fm.get("description")
    if not isinstance(desc_val, str):
        errors.append("SKILL.md frontmatter description is required and must be a string")
    else:
        stripped = desc_val.strip()
        if not stripped:
            errors.append("SKILL.md frontmatter description must be non-empty")
        elif len(stripped) > SKILL_DESC_MAX_LEN:
            errors.append(
                f"SKILL.md frontmatter description must be at most {SKILL_DESC_MAX_LEN} characters"
            )
    return errors


def _find_skill_subdirectory(root: Path) -> Path | None:
    """Exactly one non-hidden child directory that contains SKILL.md."""
    found: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (child / "SKILL.md").is_file():
            found.append(child)
    if len(found) != 1:
        return None
    return found[0]


def _init_plugin_skill(plugin_name: str, plugin_root: Path) -> Path:
    skill_dir = plugin_root / plugin_name
    plugin_root.mkdir(parents=True, exist_ok=True)
    skill_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("scripts", "references", "assets"):
        (skill_dir / sub).mkdir(parents=True, exist_ok=True)

    (plugin_root / "icon.png").write_bytes(_PLACEHOLDER_ICON_PNG_BYTES)

    skill_fm = (
        "---\n"
        f"name: {plugin_name}\n"
        'description: "TODO: describe this skill for models and users"\n'
        "---\n\n"
        "## Instructions\n\n"
        "TODO: add step-by-step guidance.\n"
    )
    (skill_dir / "SKILL.md").write_text(skill_fm, encoding="utf-8")

    pkg = plugin_name.replace("-", "_")
    (plugin_root / "plugin.yaml").write_text(
        yaml.safe_dump(
            _default_plugin_yaml(plugin_name, "skill", pkg),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return plugin_root


def plugin_init(plugin_name: str, base_path: Path, force: bool = False, plugin_type: str = "tools") -> Path:
    if not NAME_PATTERN.match(plugin_name):
        raise ValueError("plugin name must match ^[a-z][a-z0-9-]*$")
    if plugin_type not in SUPPORTED_PLUGIN_TYPES:
        supported = ", ".join(sorted(SUPPORTED_PLUGIN_TYPES))
        raise ValueError(f"plugin type must be one of: {supported}")
    if plugin_type == "skill":
        err = _validate_skill_slug(plugin_name, field="skill name")
        if err:
            raise ValueError(err)

    plugin_root = (base_path / plugin_name).resolve()
    if plugin_root.exists() and any(plugin_root.iterdir()) and not force:
        raise FileExistsError(f"{plugin_root} already exists and is not empty. Use --force to continue.")

    if plugin_type == "skill":
        return _init_plugin_skill(plugin_name, plugin_root)

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
    (plugin_root / "icon.png").write_bytes(_PLACEHOLDER_ICON_PNG_BYTES)
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
    (plugin_root / "pyproject.toml").write_text(
        _default_pyproject_toml(plugin_name, package_name, plugin_type),
        encoding="utf-8",
    )
    return plugin_root


def plugin_validate(
    plugin_path: Path,
    *,
    require_pyproject_for_tools: bool = True,
) -> ValidationResult:
    """Validate a plugin directory layout.

    ``require_pyproject_for_tools``:
      When True (default), ``runtime.type=tools`` requires ``pyproject.toml`` at the
      plugin root (local dev / ``validate`` / ``pack``).
      When False, tools may omit ``pyproject.toml`` and ``src/``; layout must include
      ``dist/*.whl`` (e.g. extracted pack / marketplace wheel-only bundle).
    """
    errors: list[str] = []
    warnings: list[str] = []
    root = plugin_path.resolve()
    if not root.exists():
        return ValidationResult(False, [f"plugin path not found: {root}"], warnings)

    plugin_yaml_path = root / "plugin.yaml"
    plugin_data: dict[str, Any] | None = None
    if plugin_yaml_path.exists():
        plugin_data = _load_yaml(plugin_yaml_path, errors)

    runtime_type = _runtime_type(plugin_data)
    required_entries = ["plugin.yaml", "icon.png"]
    if runtime_type is None or runtime_type != "skill":
        required_entries.append("README.md")

    if runtime_type == "tools":
        required_entries.append("schemas/tools.json")
        if require_pyproject_for_tools:
            required_entries.append("pyproject.toml")
            required_entries.append("src")
        else:
            dist_dir = root / "dist"
            if not dist_dir.is_dir() or not any(dist_dir.glob("*.whl")):
                errors.append("missing required: dist/*.whl (tools wheel bundle)")
    elif runtime_type is not None and runtime_type != "skill":
        required_entries.append("src")

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

    if runtime_type == "skill" and plugin_data is not None:
        yaml_name = plugin_data.get("name")
        if isinstance(yaml_name, str) and NAME_PATTERN.match(yaml_name):
            yaml_skill_err = _validate_skill_slug(yaml_name, field="plugin.yaml name")
            if yaml_skill_err:
                errors.append(yaml_skill_err)
        skill_sub = _find_skill_subdirectory(root)
        if skill_sub is None:
            errors.append(
                "skill plugin must have exactly one non-hidden child directory containing SKILL.md"
            )
        else:
            if isinstance(yaml_name, str) and skill_sub.name != yaml_name:
                errors.append(
                    f"skill directory name {skill_sub.name!r} must equal plugin.yaml name {yaml_name!r}"
                )
            fm, fm_err = _parse_skill_frontmatter(skill_sub / "SKILL.md")
            if fm_err:
                errors.append(fm_err)
            elif fm is not None and isinstance(yaml_name, str):
                errors.extend(_validate_skill_frontmatter_fields(fm, skill_sub.name, yaml_name))

    if tools_data is not None and runtime_type == "tools":
        _validate_tools_json(tools_data, errors)
        _validate_tool_names_consistency(plugin_data, tools_data, root, errors, warnings)

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def plugin_pack(plugin_path: Path, output_dir: Path | None = None) -> Path:
    """将插件目录打成 zip；打包前会先执行 validate，校验不通过则抛错不生成 zip。"""
    root = plugin_path.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"plugin path not found or not a directory: {root}")

    result = plugin_validate(root)
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
    if runtime_type is None:
        raise ValueError("plugin.yaml runtime.type is missing or not supported")

    if runtime_type == "tools":
        _pack_plugin_tools(root, name, version, prefix, zip_path)
    elif runtime_type == "skill":
        _pack_plugin_skill(root, name, version, prefix, zip_path)
    else:
        _pack_plugin_directory(root, prefix, zip_path)

    digest = sha256_file_hex(zip_path)
    sha256_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha256_path.write_text(f"{digest}  {zip_name}\n", encoding="utf-8")
    return zip_path


def _pack_plugin_tools(root: Path, name: str, version: str, prefix: str, zip_path: Path) -> None:
    """Pack tools type: build wheel first, then pack metadata and ``dist/*.whl`` (no ``src/`` tree)."""
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
        for whl in whls:
            zf.write(whl, f"{prefix}/dist/{whl.name}".replace("\\", "/"))


def _skill_optional_subdirs_non_empty(skill_dir: Path, sub: str) -> bool:
    d = skill_dir / sub
    if not d.is_dir():
        return False
    return any(d.rglob("*"))


def _pack_plugin_skill(root: Path, name: str, version: str, prefix: str, zip_path: Path) -> None:
    """Pack skill: plugin.yaml, icon.png, <name>/SKILL.md；README.md 可选（若存在则打入包）。"""
    skill_dir = root / name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise ValueError(f"skill plugin requires {name}/SKILL.md")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in ("plugin.yaml", "icon.png"):
            p = root / rel
            if p.is_file():
                zf.write(p, f"{prefix}/{rel}".replace("\\", "/"))
        readme = root / "README.md"
        if readme.is_file():
            zf.write(readme, f"{prefix}/README.md".replace("\\", "/"))
        zf.write(skill_md, f"{prefix}/{name}/SKILL.md".replace("\\", "/"))
        for sub in ("scripts", "references", "assets"):
            if not _skill_optional_subdirs_non_empty(skill_dir, sub):
                continue
            sub_root = skill_dir / sub
            for fpath in sub_root.rglob("*"):
                if fpath.is_file():
                    rel = fpath.relative_to(root)
                    zf.write(fpath, f"{prefix}/{rel.as_posix()}")


def _pack_plugin_directory(root: Path, prefix: str, zip_path: Path) -> None:
    """mcp-stdio / restful-api: full directory pack with ignore rules."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        plugin_zip_write_directory_tree(zf, root, arcname_prefix=prefix)


def plugin_publish(
    market_url: str,
    user_token: str | None,
    system_token: str | None,
    publish_input: PublishPluginInput,
) -> PluginPublishResult:
    """发布：可选先 ``plugin_pack``，再 ``market.plugin_upload``；也可直接传 zip。"""
    if publish_input.zip_path is not None:
        z = publish_input.zip_path
        if not z.is_file():
            raise PublishError(400, f"zip file not found: {z}")
        checksum_sha256 = sha256_file_hex(z)
        req = PublishRequest(
            zip_path=z,
            checksum_sha256=checksum_sha256,
            plugin_id=publish_input.plugin_id,
            plugin_version=publish_input.plugin_version,
            version_desc=publish_input.version_desc,
            force=publish_input.force,
        )
        return plugin_upload(market_url, user_token, system_token, req)
    root = publish_input.plugin_path
    if root is None:
        raise PublishError(400, "either plugin_path or zip_path must be provided")
    if not root.exists():
        raise PublishError(400, f"plugin path not found: {root}")
    if not root.is_dir():
        raise PublishError(400, f"plugin path must be a directory: {root}")

    with tempfile.TemporaryDirectory(prefix="openjiuwen_publish_") as tmp:
        out_dir = Path(tmp)
        z = plugin_pack(root, out_dir)
        checksum_sha256 = sha256_file_hex(z)
        req = PublishRequest(
            zip_path=z,
            checksum_sha256=checksum_sha256,
            plugin_id=publish_input.plugin_id,
            plugin_version=publish_input.plugin_version,
            version_desc=publish_input.version_desc,
            force=publish_input.force,
        )
        return plugin_upload(market_url, user_token, system_token, req)


def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    """解压 zip，拒绝路径穿越/绝对路径/符号链接等危险条目。"""
    dest = dest.resolve()
    for member in zf.infolist():
        name = member.filename
        if not isinstance(name, str) or not name:
            raise ValueError("unsafe zip entry: empty filename")
        if "\x00" in name:
            raise ValueError(f"unsafe zip entry: contains NUL byte: {name!r}")

        # Normalize separators to POSIX style for consistent checks.
        norm = name.replace("\\", "/")

        # Reject absolute paths and Windows drive paths.
        if norm.startswith("/") or norm.startswith("\\") or re.match(r"^[A-Za-z]:", norm):
            raise ValueError(f"unsafe zip entry: absolute/drive path: {name!r}")

        # Reject path traversal before filesystem resolution.
        norm2 = posixpath.normpath(norm)
        if norm2 in (".", "..") or norm2.startswith("../") or "/../" in f"/{norm2}/":
            raise ValueError(f"unsafe zip entry: path traversal: {name!r}")

        # Reject symlinks in archive (defense-in-depth).
        is_symlink = ((member.external_attr >> 16) & stat.S_IFMT(stat.S_IFLNK)) == stat.S_IFLNK
        if is_symlink:
            raise ValueError(f"unsafe zip entry: symlink not allowed: {name!r}")

        out = (dest / norm2).resolve()
        try:
            out.relative_to(dest)
        except ValueError:
            raise ValueError(f"unsafe zip entry: {name!r}") from None
    zf.extractall(dest)


def _find_plugin_root_in_extracted(extract_root: Path) -> Path:
    """归档内需恰好一个 plugin.yaml，返回其所在目录（插件根目录）。"""
    found = list(extract_root.rglob("plugin.yaml"))
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one plugin.yaml in archive, found {len(found)} under {extract_root}"
        )
    return found[0].parent


def _parse_plugin_yaml_for_install(plugin_root: Path) -> tuple[str, str]:
    """读取并校验 ``plugin.yaml``，返回 ``(runtime_type, yaml_name)``。"""
    plugin_yaml = plugin_root / "plugin.yaml"
    data = yaml.safe_load(plugin_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("plugin.yaml must be an object")
    runtime_type = _runtime_type(data)
    if runtime_type is None:
        raise ValueError("plugin.yaml runtime.type is missing or not supported")
    yaml_name = data.get("name")
    if not isinstance(yaml_name, str):
        raise ValueError("plugin.yaml name must be a string")
    return runtime_type, yaml_name


def _install_skill_from_staging(
    plugin_root: Path,
    yaml_name: str,
    dest_parent: Path,
    *,
    force: bool,
) -> Path:
    skill_src = plugin_root / yaml_name
    if not skill_src.is_dir() or not (skill_src / "SKILL.md").is_file():
        raise ValueError(f"skill plugin requires directory {yaml_name}/ containing SKILL.md")
    dest = dest_parent / yaml_name
    if dest.exists():
        if not force:
            raise FileExistsError(
                f"destination already exists: {dest} (use force=True or --force to overwrite)"
            )
        shutil.rmtree(dest)
    shutil.copytree(skill_src, dest)
    return dest


def _copy_bundle_to_output(plugin_root: Path, dest_parent: Path, *, force: bool) -> Path:
    """将插件根目录拷到 ``dest_parent / <staging_root_name>``。"""
    bundle_dest = dest_parent / plugin_root.name
    if bundle_dest.exists():
        if not force:
            raise FileExistsError(
                f"destination already exists: {bundle_dest} (use force=True or --force to overwrite)"
            )
        shutil.rmtree(bundle_dest)
    shutil.copytree(plugin_root, bundle_dest)
    return bundle_dest


def _pip_install_tools_wheels(bundle_dest: Path) -> None:
    """``pip install`` on ``bundle_dest/dist/*.whl`` into the current Python environment."""
    bd = bundle_dest.resolve()
    whls = sorted((bd / "dist").glob("*.whl"))
    if not whls:
        raise ValueError("tools plugin zip has no dist/*.whl")
    for w in whls:
        if w.name.startswith("-"):
            raise ValueError(f"unsafe wheel filename (starts with '-'): {w.name!r}")
    wheel_paths = [str(w) for w in whls]
    cmd: list[str] = [sys.executable, "-m", "pip", "install", "--"]
    cmd.extend(wheel_paths)
    subprocess.run(cmd, check=True)


def plugin_install(
    zip_path: Path,
    *,
    extract_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """解压 zip、校验后按 runtime 落盘；skill 只拷 slug 目录，tools 可 pip 装 wheels，mcp/api 仅拷贝。返回安装根路径。"""
    zpath = zip_path.resolve()
    if not zpath.is_file():
        raise ValueError(f"zip not found: {zpath}")

    dest_parent = (extract_dir if extract_dir is not None else Path.cwd()).resolve()
    dest_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="openjiuwen_install_") as tmp:
        extract_root = Path(tmp)
        with zipfile.ZipFile(zpath, "r") as zf:
            _safe_extractall(zf, extract_root)
        plugin_root = _find_plugin_root_in_extracted(extract_root)

        result = plugin_validate(plugin_root, require_pyproject_for_tools=False)
        for w in result.warnings:
            logger.warning("%s", w)
        if result.errors:
            raise ValueError("plugin validation failed: " + "; ".join(result.errors))

        runtime_type, yaml_name = _parse_plugin_yaml_for_install(plugin_root)

        if runtime_type == "skill":
            return _install_skill_from_staging(plugin_root, yaml_name, dest_parent, force=force)

        bundle_dest = _copy_bundle_to_output(plugin_root, dest_parent, force=force)

        try:
            if runtime_type == "tools":
                _pip_install_tools_wheels(bundle_dest)
                logger.info("tools: installed wheels into the current Python environment")
            elif runtime_type in ("mcp-stdio", "restful-api"):
                logger.info(
                    "%s: pip not run; install dependencies manually if required",
                    runtime_type,
                )
            else:
                raise ValueError(f"unexpected runtime type after bundle copy: {runtime_type}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pip install failed (exit {e.returncode})") from e

        return bundle_dest


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


def _runtime_type(plugin_data: dict[str, Any] | None) -> str | None:
    """解析并规范化 runtime.type。"""
    if not isinstance(plugin_data, dict):
        return None
    runtime = plugin_data.get("runtime")
    if not isinstance(runtime, dict):
        return None
    raw = runtime.get("type")
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    rt = raw.strip().lower()
    if not rt:
        return None
    canonical_map = {
        "tools": "tools",
        "mcp-stdio": "mcp-stdio",
        "restful-api": "restful-api",
        "skill": "skill",
    }
    return canonical_map.get(rt)


def _is_valid_requires_python(value: str) -> bool:
    """验证 PEP 440 版本说明符。"""
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
    if runtime_type == "skill":
        if not isinstance(name, str) or not name.strip():
            errors.append("plugin.yaml name is required")
        elif not NAME_PATTERN.match(name):
            errors.append("plugin.yaml name must match ^[a-z][a-z0-9-]*$")
        else:
            skill_slug_err = _validate_skill_slug(name, field="plugin.yaml name")
            if skill_slug_err:
                errors.append(skill_slug_err)
    elif not isinstance(name, str) or not NAME_PATTERN.match(name):
        errors.append("plugin.yaml name must match ^[a-z][a-z0-9-]*$")

    version = plugin_data.get("version")
    if not isinstance(version, str) or not MARKETPLACE_VERSION_PATTERN.match(version):
        errors.append(
            "plugin.yaml version must be x.y.z three numeric segments (marketplace rule), e.g. 1.2.3"
        )

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
        if not isinstance(declared_type, str) or not declared_type.strip() or declared_type.strip().lower() not in {
            t.lower() for t in SUPPORTED_PLUGIN_TYPES
        }:
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
    if runtime_type != "skill":
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
    else:
        pass

    if runtime_type == "tools":
        tools_schema = plugin_data.get("tools_schema")
        if tools_schema is None:
            warnings.append(f"plugin.yaml tools_schema missing, defaulting to {expected_tools_path}")
        elif not isinstance(tools_schema, str):
            errors.append("plugin.yaml tools_schema must be string path")
        elif tools_schema != expected_tools_path:
            errors.append(f"plugin.yaml tools_schema must be '{expected_tools_path}'")
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
    elif runtime_type == "skill":
        pass


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
        "version": "0.0.1",
        "display_name": plugin_name.replace("-", " ").title(),
        "description": "TODO: describe your plugin",
        "runtime": {
            "type": plugin_type,
        },
        "metadata": {
            "author": "TODO: your name",
            "tags": ["demo"],
        },
    }
    if plugin_type != "skill":
        base["compatibility"] = {"python": ">=3.11, <3.14"}
    if plugin_type == "tools":
        base["tools_schema"] = "schemas/tools.json"
    elif plugin_type == "mcp-stdio":
        base["mcp"] = {
            "transport": "stdio",
            "command": ["python", "-m", f"{package_name}.mcp_server"],
        }
    elif plugin_type == "skill":
        pass
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
    return f'''"""REST API entry for {plugin_name} (optional placeholder)."""


'''


def _default_pyproject_toml(plugin_name: str, package_name: str, plugin_type: str) -> str:
    dependencies: list[str] = []
    if plugin_type == "mcp-stdio":
        dependencies = ["fastmcp"]

    deps_toml = ""
    if dependencies:
        deps_toml = "\ndependencies = [" + ", ".join(f"\"{d}\"" for d in dependencies) + "]\n"

    return f"""[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{plugin_name}"
version = "0.0.1"
description = "openJiuwen plugin"
readme = "README.md"
requires-python = ">=3.11"
{deps_toml}

[tool.setuptools.packages.find]
where = ["src"]
include = ["{package_name}*"]
"""


def _render_readme(plugin_name: str, plugin_type: str) -> str:
    if plugin_type == "skill":
        type_notes = (
            f"- `{plugin_name}/SKILL.md`: skill instructions (Agent Skills layout)\n"
            f"- `{plugin_name}/scripts|references|assets/`: optional payloads\n"
        )
    elif plugin_type == "mcp-stdio":
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


def plugin_describe_local(plugin_path: Path) -> dict[str, Any]:
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