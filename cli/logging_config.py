"""统一日志配置：格式、级别、控制台输出；禁止使用 print，请使用 logger。"""

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(
    level: str = "INFO",
    log_format: str | None = None,
    date_format: str | None = None,
) -> None:
    """配置根日志：级别与控制台 Handler。在入口处调用一次。"""
    global _initialized
    if _initialized:
        return

    fmt = log_format or _DEFAULT_FORMAT
    date_fmt = date_format or _DATE_FORMAT
    level_value = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level_value)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level_value)
        handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        root.addHandler(handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """按模块名获取 logger，便于在各模块内使用。"""
    return logging.getLogger(name)
