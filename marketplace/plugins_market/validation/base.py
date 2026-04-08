"""Shared helper utilities for validation modules."""

from plugins_market.core.errors import PublishError


def raise_invalid_config(message: str) -> None:
    raise PublishError(code=400, error="invalid_plugin_config", message=message)


def raise_invalid_structure(message: str) -> None:
    raise PublishError(code=400, error="invalid_plugin_structure", message=message)


def raise_invalid_skill_md(message: str) -> None:
    raise PublishError(code=400, error="invalid_skill_md", message=message)


def require_string_field(value: object, field_name: str) -> str:
    """Return stripped string value; raise PublishError if missing or not a string."""
    if not isinstance(value, str) or not value.strip():
        raise_invalid_config(f"plugin.yaml 配置文件格式错误或缺失：{field_name} 必填")
    return value.strip()  # type: ignore[return-value]
