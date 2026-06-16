"""Timezone-aware datetime helpers for reports."""

from __future__ import annotations

from datetime import datetime, timezone


def local_now() -> datetime:
    """Return current local time with explicit timezone."""
    return datetime.now(timezone.utc).astimezone()


def format_now(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return local_now().strftime(fmt)
