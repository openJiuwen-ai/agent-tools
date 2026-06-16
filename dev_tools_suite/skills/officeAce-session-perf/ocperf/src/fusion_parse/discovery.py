"""Locate history.json and full*.json under a log directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiscoveredLogs:
    log_dir: Path
    history_files: list[Path]
    full_files: list[Path]

    @property
    def history_label(self) -> str:
        if len(self.history_files) == 1:
            return self.history_files[0].name
        return f"{len(self.history_files)}× history*.json"


def discover_session_logs(log_dir: Path) -> DiscoveredLogs:
    """Resolve history + full inputs from a session log folder."""
    root = log_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")

    history: list[Path] = []
    exact = root / "history.json"
    if exact.is_file():
        history = [exact]
    else:
        history = sorted(root.glob("history*.json"))
        if not history:
            raise FileNotFoundError(f"未找到 history.json 或 history*.json: {root}")

    full_files = sorted(root.glob("full*.json"))
    single_full = root / "full.json"
    if single_full.is_file() and single_full not in full_files:
        full_files.append(single_full)
    for pattern in ("full*.txt", "full*.log"):
        full_files.extend(sorted(root.glob(pattern)))
    full_files = sorted({p.resolve() for p in full_files}, key=lambda p: (p.stat().st_mtime, p.name))
    if not full_files:
        raise FileNotFoundError(f"未找到 full.json / full*.json / full*.txt: {root}")

    return DiscoveredLogs(log_dir=root, history_files=history, full_files=full_files)
