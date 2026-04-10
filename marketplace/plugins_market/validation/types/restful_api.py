"""restful-api 插件：zip 目录布局与 icon 校验。"""

from __future__ import annotations

import zipfile

from plugins_market.validation.base import raise_invalid_structure
from plugins_market.validation.zip_utils import (
    DecompressCounter,
    has_src_tree,
    safe_read_zip_member,
    validate_png_icon_bytes,
)


def validate_restful_api_layout(
    zf: zipfile.ZipFile,
    prefix: str,
    counter: DecompressCounter,
) -> dict:
    """校验 restful-api 包根目录：README、icon、非空 src/，并校验 icon 为合法 PNG。

    Returns dict with keys: icon_path, icon_bytes, readme_path.
    """
    names = set(zf.namelist())

    readme_path = prefix + "README.md"
    if readme_path not in names:
        raise_invalid_structure(
            "插件包结构不符合要求：restful-api 类型缺少 README.md"
        )

    icon_path = prefix + "icon.png"
    if icon_path not in names:
        raise_invalid_structure(
            "插件包结构不符合要求：restful-api 类型缺少 icon.png"
        )

    if not has_src_tree(names, prefix):
        raise_invalid_structure(
            "插件包结构不符合要求：restful-api 类型缺少 src/ 目录"
        )

    icon_bytes = safe_read_zip_member(zf, icon_path, counter)
    validate_png_icon_bytes(icon_bytes, path=icon_path)

    return {"icon_path": icon_path, "icon_bytes": icon_bytes, "readme_path": readme_path}
