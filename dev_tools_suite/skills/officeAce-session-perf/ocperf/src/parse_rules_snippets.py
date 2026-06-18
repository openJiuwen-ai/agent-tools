"""Guide link helper for HTML reports."""

from __future__ import annotations

import html
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GUIDE_LABEL = "解析规则说明"


def resolve_guide_href(out_html: Path | None, explicit: str | None = None) -> str:
    """Relative path from output HTML to docs/GUIDE.md, or explicit href."""
    if explicit:
        return explicit
    guide = _REPO_ROOT / "docs" / "GUIDE.md"
    if out_html is not None:
        try:
            return Path(os.path.relpath(guide, out_html.parent.resolve())).as_posix()
        except ValueError:
            pass
    return "GUIDE.md"


def guide_link_html(href: str, label: str = DEFAULT_GUIDE_LABEL) -> str:
    return (
        f'<a class="guide-link" href="{html.escape(href, quote=True)}" '
        f'target="_blank" rel="noopener noreferrer">{html.escape(label)}</a>'
    )
