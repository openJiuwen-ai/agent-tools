from __future__ import annotations

import os
import tempfile
from pathlib import Path

from openjiuwen_plugin.logging_config import get_logger
from openjiuwen_plugin.market import delete_plugin, download_artifact_zip, get_plugin_version_detail, search_plugins
from openjiuwen_plugin.plugin import (
    PublishError,
    init_plugin,
    install_plugin_from_zip,
    pack_plugin,
    publish_plugin,
    validate_plugin,
)
from openjiuwen_plugin.schemas import PluginListQuery
from openjiuwen_plugin.schemas import PublishPluginInput

logger = get_logger(__name__)


def _resolve_market_url(args_market_url: str | None, *, err_msg: str) -> str | None:
    market_url = args_market_url or os.getenv("OPENJIUWEN_MARKET_URL")
    if not market_url:
        logger.error(err_msg)
        return None
    return market_url


def _resolve_publish_auth(args) -> tuple[str | None, str | None, int]:
    user_token = args.user_token or os.getenv("OPENJIUWEN_USER_TOKEN")
    system_token = args.system_token or os.getenv("OPENJIUWEN_SYSTEM_TOKEN")
    has_user = bool(user_token and str(user_token).strip())
    has_sys = bool(system_token and str(system_token).strip())
    if has_user and has_sys:
        logger.error("publish supports exactly one auth method: use either --token or --system-token")
        return None, None, 1

    user_token = str(user_token).strip() if user_token else ""
    system_token = str(system_token).strip() if system_token else ""
    if system_token:
        return None, system_token, 0

    token_to_pass = user_token
    if not token_to_pass:
        try:
            token_to_pass = input("User token: ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.error("user token not provided")
            return None, None, 1
        if not token_to_pass:
            logger.error("user token cannot be empty")
            return None, None, 1
    return token_to_pass, None, 0


def _resolve_delete_auth(args) -> tuple[str | None, str | None, str | None, int]:
    user_token = args.user_token or os.getenv("OPENJIUWEN_USER_TOKEN")
    system_token = args.system_token or os.getenv("OPENJIUWEN_SYSTEM_TOKEN")

    has_user = bool(user_token and str(user_token).strip())
    has_sys = bool(system_token and str(system_token).strip())
    if has_user and has_sys:
        logger.error("delete supports exactly one auth method: use either --token or --system-token")
        return None, None, None, 1

    if has_sys:
        return None, str(system_token).strip(), None, 0

    if not user_token or not str(user_token).strip():
        try:
            user_token = input("User token: ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.error("user token not provided")
            return None, None, None, 1
        if not user_token:
            logger.error("user token cannot be empty")
            return None, None, None, 1

    user_id = args.user_id or os.getenv("OPENJIUWEN_USER_ID")
    if not user_id or not str(user_id).strip():
        logger.error("delete with Bearer requires --user-id or OPENJIUWEN_USER_ID")
        return None, None, None, 1
    return str(user_token).strip(), None, str(user_id).strip(), 0


def handle_init(args) -> int:
    try:
        plugin_root = init_plugin(args.name, Path(args.path), force=args.force, plugin_type=args.plugin_type)
    except Exception as exc:
        logger.error("init failed: %s", exc)
        return 1
    logger.info("plugin initialized at: %s", plugin_root)
    return 0


def handle_validate(args) -> int:
    result = validate_plugin(Path(args.path).resolve())
    if result.warnings:
        for warning in result.warnings:
            logger.warning("%s", warning)
    if result.errors:
        for err in result.errors:
            logger.error("%s", err)
        return 1
    logger.info("plugin validation passed")
    return 0


def handle_pack(args) -> int:
    try:
        plugin_root = Path(args.path).resolve()
        out_dir = (
            Path(args.output).resolve()
            if Path(args.output).is_absolute()
            else (plugin_root / args.output).resolve()
        )
        zip_path = pack_plugin(plugin_root, out_dir)
    except Exception as exc:
        logger.error("pack failed: %s", exc)
        return 1
    logger.info("packed: %s", zip_path)
    return 0


def handle_publish(args) -> int:
    token_to_pass, system_token, exit_code = _resolve_publish_auth(args)
    if exit_code != 0:
        return exit_code

    market_url = _resolve_market_url(
        args.market_url,
        err_msg="publish requires --market-url or OPENJIUWEN_MARKET_URL",
    )
    if not market_url:
        return 1
    if not args.file and not args.path:
        logger.error("path (plugin directory) is required when not using --file")
        return 1

    user_id = args.user_id or os.getenv("OPENJIUWEN_USER_ID")
    if not user_id or not str(user_id).strip():
        logger.error("publish requires --user-id or OPENJIUWEN_USER_ID")
        return 1
    user_id = str(user_id).strip()

    try:
        publish_input = PublishPluginInput(
            user_id=user_id,
            plugin_path=Path(args.path).resolve() if args.path else None,
            zip_path=Path(args.file).resolve() if args.file else None,
            plugin_id=args.plugin_id or None,
            plugin_version=args.plugin_version or None,
            version_desc=args.version_desc or None,
            force=args.force,
        )
        result = publish_plugin(
            market_url=market_url,
            user_token=token_to_pass if token_to_pass else None,
            system_token=system_token if system_token else None,
            publish_input=publish_input,
        )
    except PublishError as e:
        logger.error("publish failed: %s", e.detail)
        return 1
    except ValueError as e:
        logger.error("publish failed: %s", e)
        return 1

    logger.info(
        "published: plugin_id=%s name=%s version=%s status=%s",
        result.plugin_id,
        result.name,
        result.version,
        result.status,
    )
    logger.info(
        "提示：请保存上方 plugin_id，后续发新版本时需传 --plugin-id；若未保存可执行 openjiuwen-plugin search <关键词> 查询"
    )
    return 0


def handle_info(args) -> int:
    market_url = args.market_url or os.getenv("OPENJIUWEN_MARKET_URL")
    if not market_url:
        logger.error("info requires --market-url or OPENJIUWEN_MARKET_URL")
        return 1
    try:
        detail = get_plugin_version_detail(market_url, args.asset_id, args.version)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    except Exception as exc:
        logger.error("info failed: %s", exc)
        return 1

    # 直接按 PluginVersionDetail 字段输出（按 schema 定义顺序）
    logger.info("asset_id: %s", detail.asset_id or args.asset_id)
    for key in detail.__class__.model_fields:
        if key == "asset_id":
            continue
        val = getattr(detail, key)
        if val in (None, ""):
            continue
        if key == "tags":
            if isinstance(val, list) and val:
                logger.info("tags: %s", ", ".join(str(x) for x in val))
            continue
        logger.info("%s: %s", key, val)

    return 0


def handle_search(args) -> int:
    market_url = args.market_url or os.getenv("OPENJIUWEN_MARKET_URL")
    if not market_url:
        logger.error("search requires --market-url or OPENJIUWEN_MARKET_URL")
        return 1
    try:
        if args.page is not None and args.page < 1:
            logger.error("search --page must be >= 1")
            return 1
        if args.page_size is not None and (args.page_size < 1 or args.page_size > 100):
            logger.error("search --page-size must be between 1 and 100")
            return 1
        query = PluginListQuery(
            search_keyword=args.query or "",
            plugin_type=args.plugin_type,
            publisher_name=args.author,
            asset_id=args.search_asset_id,
            asset_type=args.search_asset_type,
            publisher_id=args.search_publisher_id,
            page=args.page or 1,
            page_size=args.page_size or 20,
            order_by=args.order_by or "install_count",
            desc=args.desc,
        )
        result = search_plugins(market_url, query)
        if not result.items:
            logger.info("no results.")
            return 0
        logger.info("page=%s page_size=%s total=%s", result.page, result.page_size, result.total)
        for item in result.items:
            aid = item.asset_id
            name = item.name
            ver = item.latest_version
            logger.info("  %s  %s  %s", aid, name, ver)
    except Exception as exc:
        logger.error("search failed: %s", exc)
        return 1
    return 0


def handle_delete(args) -> int:
    market_url = _resolve_market_url(
        args.market_url,
        err_msg="delete requires --market-url or OPENJIUWEN_MARKET_URL",
    )
    if not market_url:
        return 1
    try:
        user_token, system_token, user_id, exit_code = _resolve_delete_auth(args)
        if exit_code != 0:
            return exit_code
        if system_token:
            delete_plugin(
                market_url,
                args.plugin_id,
                logger,
                version=args.version,
                system_token=system_token,
            )
        else:
            delete_plugin(
                market_url,
                args.plugin_id,
                logger,
                version=args.version,
                user_token=user_token,
                user_id=user_id,
            )
    except Exception as exc:
        logger.error("delete failed: %s", exc)
        return 1
    return 0


def handle_install(args) -> int:
    market_url = _resolve_market_url(
        args.market_url,
        err_msg="install requires --market-url or OPENJIUWEN_MARKET_URL",
    )
    if not market_url:
        return 1

    asset_id = str(args.asset_id).strip()
    if args.plugin_version:
        logger.warning("install --version is ignored for artifacts endpoint")

    zip_path: Path
    own_tmp: Path | None = None
    if args.output_zip:
        zip_path = Path(args.output_zip).resolve()
    else:
        fd, tmp_name = tempfile.mkstemp(suffix=".zip", prefix="openjiuwen_dl_")
        os.close(fd)
        own_tmp = Path(tmp_name)
        zip_path = own_tmp

    pip_prefix = Path(args.pip_prefix).resolve() if args.pip_prefix else None
    try:
        dl_info = download_artifact_zip(market_url, asset_id, zip_path)
        if dl_info.verified:
            logger.info(
                "download checksum verified: %s",
                dl_info.actual_checksum_sha256,
            )
        else:
            logger.warning("download checksum not provided by server; skip verification")
        installed_root = install_plugin_from_zip(zip_path, pip_prefix=pip_prefix)
        logger.info("plugin installed under: %s", installed_root)
    except Exception as exc:
        logger.error("install failed: %s", exc)
        return 1
    finally:
        if own_tmp is not None:
            try:
                own_tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return 0


COMMAND_HANDLERS = {
    "init": handle_init,
    "validate": handle_validate,
    "pack": handle_pack,
    "publish": handle_publish,
    "info": handle_info,
    "search": handle_search,
    "delete": handle_delete,
    "install": handle_install,
}
