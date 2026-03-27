from __future__ import annotations

import argparse


def _parse_bool_flag(value: str) -> bool:
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be true or false")


def _add_init_parser(plugin_subparsers) -> None:
    init_parser = plugin_subparsers.add_parser("init", help="Initialize a new plugin scaffold")
    init_parser.add_argument("name", help="Plugin name, e.g. weather-plugin")
    init_parser.add_argument("--path", default=".", help="Parent directory to create plugin in")
    init_parser.add_argument("--force", action="store_true", help="Allow non-empty target directory")
    init_parser.add_argument(
        "--type",
        dest="plugin_type",
        default="tools",
        choices=("tools", "mcp-stdio", "restful-api"),
        help="Plugin type, default is tools",
    )


def _add_validate_parser(plugin_subparsers) -> None:
    validate_parser = plugin_subparsers.add_parser("validate", help="Validate plugin structure and metadata")
    validate_parser.add_argument("path", help="Plugin root directory")


def _add_pack_parser(plugin_subparsers) -> None:
    pack_parser = plugin_subparsers.add_parser("pack", help="Pack validated plugin into a zip for upload")
    pack_parser.add_argument("path", help="Plugin root directory")
    pack_parser.add_argument(
        "--output",
        "-o",
        default="out",
        help="Output directory for the zip file (default: out)",
    )


def _add_publish_parser(plugin_subparsers) -> None:
    publish_parser = plugin_subparsers.add_parser("publish", help="Pack and upload plugin to market")
    publish_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Plugin root directory (required when not using --file)",
    )
    publish_parser.add_argument(
        "--file",
        "-f",
        metavar="PATH",
        help="Use existing zip to publish; if set, skip pack and upload this file",
    )
    publish_parser.add_argument(
        "--token",
        dest="user_token",
        help="用户 token（Bearer，普通用户发布/删除时使用）；与 --system-token 二选一。优先级：本参数 > OPENJIUWEN_USER_TOKEN > 交互输入",
    )
    publish_parser.add_argument(
        "--system-token",
        help="系统管理员 token，走 X-System-Token（与 --token 二选一；可用 OPENJIUWEN_SYSTEM_TOKEN）",
    )
    publish_parser.add_argument("--market-url", help="Market base URL (default: OPENJIUWEN_MARKET_URL)")
    publish_parser.add_argument(
        "--user-id",
        default=None,
        help="发布者 user_id（必填；命令行优先于 OPENJIUWEN_USER_ID）",
    )
    publish_parser.add_argument(
        "--plugin-id",
        help=(
            "Plugin id (optional for first publish; printed after first publish, "
            "or get via 'openjiuwen_plugin search')"
        ),
    )
    publish_parser.add_argument(
        "--plugin-version",
        help="Override version (SemVer e.g. 1.0.0; v1.0.0 accepted and stripped; optional)",
    )
    publish_parser.add_argument("--version-desc", help="Version description")
    publish_parser.add_argument("--force", action="store_true", help="Overwrite existing version")


def _add_info_parser(plugin_subparsers) -> None:
    info_parser = plugin_subparsers.add_parser(
        "info",
        help="Get plugin version details (GET /api/v1/plugins/{asset_id}/versions/{version})",
    )
    info_parser.add_argument(
        "asset_id",
        help="Asset id（与 publish 返回的 plugin_id 相同）",
    )
    info_parser.add_argument("--version", "-v", required=True, help="目标版本号")
    info_parser.add_argument("--market-url", help="Market base URL (default: OPENJIUWEN_MARKET_URL)")


def _add_search_parser(plugin_subparsers) -> None:
    search_parser = plugin_subparsers.add_parser(
        "search",
        help="Search plugins on market (no auth); query flags match marketplace PluginListQuery",
    )
    search_parser.add_argument("query", nargs="?", default="", help="search_keyword（关键词）")
    search_parser.add_argument("--market-url", help="Market base URL (default: OPENJIUWEN_MARKET_URL)")
    search_parser.add_argument(
        "--type",
        dest="plugin_type",
        default=None,
        metavar="STR",
        help="plugin_type（精确匹配 plugin.yaml runtime.type，如 tools / mcp-stdio / restful-api）",
    )
    search_parser.add_argument(
        "--author",
        metavar="NAME",
        default=None,
        help="publisher_name（发布者展示名模糊）；CLI 使用 --author 以兼容习惯",
    )
    search_parser.add_argument(
        "--asset-id",
        dest="search_asset_id",
        default=None,
        metavar="ID",
        help="asset_id",
    )
    search_parser.add_argument(
        "--asset-type",
        dest="search_asset_type",
        default=None,
        metavar="TYPE",
        help="asset_type",
    )
    search_parser.add_argument(
        "--publisher-id",
        dest="search_publisher_id",
        default=None,
        metavar="ID",
        help="publisher_id",
    )
    search_parser.add_argument(
        "--page",
        type=int,
        default=None,
        metavar="N",
        help="page（默认 1）",
    )
    search_parser.add_argument(
        "--page-size",
        dest="page_size",
        type=int,
        default=None,
        metavar="N",
        help="page_size（默认 20，最大 100）",
    )
    search_parser.add_argument(
        "--order-by",
        default=None,
        choices=("install_count", "like_count", "create_time", "update_time", "review_count"),
        help="order_by（默认 install_count）",
    )
    search_parser.add_argument(
        "--desc",
        type=_parse_bool_flag,
        default=True,
        metavar="BOOL",
        help="是否降序（对应 API desc）；可传 true/false，默认 true",
    )


def _add_delete_parser(plugin_subparsers) -> None:
    delete_parser = plugin_subparsers.add_parser("delete", help="Delete plugin from market (Store delete API)")
    delete_parser.add_argument(
        "plugin_id",
        help="Asset id（与 publish 返回的 plugin_id 相同）",
    )
    delete_parser.add_argument("--market-url", help="Market base URL (default: OPENJIUWEN_MARKET_URL)")
    delete_parser.add_argument(
        "--user-id",
        help="发布者 user_id（Bearer 时必填；与 OPENJIUWEN_USER_ID）",
    )
    delete_parser.add_argument(
        "--system-token",
        help="系统管理员 token，走 X-System-Token（与 Bearer 二选一；可用 OPENJIUWEN_SYSTEM_TOKEN）",
    )
    delete_parser.add_argument(
        "--token",
        dest="user_token",
        help="用户 token（Bearer；与 --system-token 二选一）。默认读取 OPENJIUWEN_USER_TOKEN 或交互输入",
    )
    delete_parser.add_argument(
        "--version",
        help="要删的版本号；省略则传 all（删光版本并删资产）",
    )


def _add_install_parser(plugin_subparsers) -> None:
    install_parser = plugin_subparsers.add_parser(
        "install",
        help="Download artifact zip from market and pip install (GET /api/v1/artifacts/{asset_id})",
    )
    install_parser.add_argument(
        "asset_id",
        help="市场 asset_id（与 publish 返回的 plugin_id 相同）",
    )
    install_parser.add_argument("--market-url", help="Market base URL (default: OPENJIUWEN_MARKET_URL)")
    install_parser.add_argument(
        "--version",
        dest="plugin_version",
        help="兼容保留参数（当前 artifacts 下载接口按 asset_id 获取）",
    )
    install_parser.add_argument(
        "--prefix",
        dest="pip_prefix",
        help="传给 pip install --prefix（可选）",
    )
    install_parser.add_argument(
        "--output",
        "-o",
        dest="output_zip",
        help="将下载的 zip 保存到此路径；省略则使用临时文件并在安装后删除",
    )


def build_plugin_parser(prog_name: str = "openjiuwen_plugin") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog_name)
    plugin_subparsers = parser.add_subparsers(dest="plugin_command")
    _add_init_parser(plugin_subparsers)
    _add_validate_parser(plugin_subparsers)
    _add_pack_parser(plugin_subparsers)
    _add_publish_parser(plugin_subparsers)
    _add_info_parser(plugin_subparsers)
    _add_search_parser(plugin_subparsers)
    _add_delete_parser(plugin_subparsers)
    _add_install_parser(plugin_subparsers)
    return parser
