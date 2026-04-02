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
        choices=("tools", "mcp-stdio", "restful-api", "skill"),
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
        help=(
            "End-user Bearer token (Authorization header). Mutually exclusive with --system-token. "
            "If omitted, reads OPENJIUWEN_USER_TOKEN"
        ),
    )
    publish_parser.add_argument(
        "--system-token",
        help=(
            "System-admin token (X-System-Token header). Mutually exclusive with --token. "
            "If omitted, can use OPENJIUWEN_SYSTEM_TOKEN"
        ),
    )
    publish_parser.add_argument("--market-url", help="Market base URL (default: OPENJIUWEN_MARKET_URL)")
    publish_parser.add_argument(
        "--plugin-id",
        help=(
            "Plugin id (optional for first publish; printed after first publish, "
            "or get via 'openjiuwen-plugin search')"
        ),
    )
    publish_parser.add_argument(
        "--plugin-version",
        help="Override version (marketplace: x.y.z e.g. 1.0.0; v1.0.0 accepted and stripped; optional)",
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
        help="Asset id (same as plugin_id returned by publish)",
    )
    info_parser.add_argument("--version", "-v", required=True, help="Target version")
    info_parser.add_argument("--market-url", help="Market base URL (default: OPENJIUWEN_MARKET_URL)")


def _add_search_parser(plugin_subparsers) -> None:
    search_parser = plugin_subparsers.add_parser(
        "search",
        help="Search plugins on market (no auth); query flags match marketplace PluginListQuery",
    )
    search_parser.add_argument("query", nargs="?", default="", help="search keyword")
    search_parser.add_argument("--market-url", help="Market base URL (default: OPENJIUWEN_MARKET_URL)")
    search_parser.add_argument(
        "--type",
        dest="plugin_type",
        default=None,
        metavar="STR",
        help="plugin type (exact match plugin.yaml runtime.type, such as tools / mcp-stdio / restful-api / skill)",
    )
    search_parser.add_argument(
        "--author",
        metavar="NAME",
        default=None,
        help="publisher name (fuzzy match)",
    )
    search_parser.add_argument(
        "--asset-id",
        dest="search_asset_id",
        default=None,
        metavar="ID",
        help="asset id",
    )
    search_parser.add_argument(
        "--asset-type",
        dest="search_asset_type",
        default=None,
        metavar="TYPE",
        help="asset type",
    )
    search_parser.add_argument(
        "--publisher-id",
        dest="search_publisher_id",
        default=None,
        metavar="ID",
        help="publisher id",
    )
    search_parser.add_argument(
        "--page",
        type=int,
        default=None,
        metavar="N",
        help="page (default 1)",
    )
    search_parser.add_argument(
        "--page-size",
        dest="page_size",
        type=int,
        default=None,
        metavar="N",
        help="page size (default 20, max 100)",
    )
    search_parser.add_argument(
        "--order-by",
        default=None,
        choices=("install_count", "like_count", "create_time", "update_time", "review_count"),
        help="order by (default install_count)",
    )
    search_parser.add_argument(
        "--desc",
        type=_parse_bool_flag,
        default=True,
        metavar="BOOL",
        help="descending order (default true)",
    )


def _add_delete_parser(plugin_subparsers) -> None:
    delete_parser = plugin_subparsers.add_parser("delete", help="Delete plugin from market (Store delete API)")
    delete_parser.add_argument(
        "plugin_id",
        help="Asset id (same as plugin_id returned by publish)",
    )
    delete_parser.add_argument("--market-url", help="Market base URL (default: OPENJIUWEN_MARKET_URL)")
    delete_parser.add_argument(
        "--system-token",
        help=(
            "System-admin token (X-System-Token header). Mutually exclusive with --token. "
            "If omitted, can use OPENJIUWEN_SYSTEM_TOKEN"
        ),
    )
    delete_parser.add_argument(
        "--token",
        dest="user_token",
        help=(
            "End-user Bearer token (Authorization header). Mutually exclusive with --system-token. "
            "If omitted, reads OPENJIUWEN_USER_TOKEN"
        ),
    )
    delete_parser.add_argument(
        "--version",
        help="Version to delete; if omitted then delete all versions",
    )


def _add_install_parser(plugin_subparsers) -> None:
    install_parser = plugin_subparsers.add_parser(
        "install",
        help="Download artifact zip from market and pip install (GET /api/v1/artifacts/{asset_id})",
    )
    install_parser.add_argument(
        "asset_id",
        help="Market asset_id (same as plugin_id returned by publish)",
    )
    install_parser.add_argument("--market-url", help="Market base URL (default: OPENJIUWEN_MARKET_URL)")
    install_parser.add_argument(
        "--version",
        dest="plugin_version",
        help="Preserved parameter (current artifacts download API gets by asset_id)",
    )
    install_parser.add_argument(
        "--prefix",
        dest="pip_prefix",
        help="Pass to pip install --prefix (optional; ignored for skill type)",
    )
    install_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Extract/install target root directory (default current working directory)",
    )
    install_parser.add_argument(
        "--save-zip",
        dest="save_zip",
        default=None,
        metavar="PATH",
        help="Save downloaded zip to this path additionally (optional)",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwrite when target directory already exists",
    )


def build_plugin_parser(prog_name: str = "openjiuwen-plugin") -> argparse.ArgumentParser:
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