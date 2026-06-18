"""Resolve OfficeClaw session logs (user-agnostic; paths via env or %USERPROFILE%)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _user_home() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home()).expanduser()


def _default_logs_roots() -> list[Path]:
    env = os.environ.get("OFFICECLAW_LOGS")
    if env:
        return [Path(env)]
    home = _user_home()
    return [
        home / ".office-claw" / ".jiuwenclaw" / "service_default" / ".logs",
        home / "office-claw.jiuwenclaw" / "service_default.logs",
    ]


def _default_sessions_roots() -> list[Path]:
    env = os.environ.get("OFFICECLAW_SESSIONS")
    if env:
        return [Path(env)]
    home = _user_home()
    return [
        home
        / ".office-claw"
        / ".jiuwenclaw"
        / "service_default"
        / "agent_default"
        / "agent"
        / "sessions",
        home
        / "office-claw.jiuwenclaw"
        / "service_default"
        / "agent_default"
        / "agent"
        / "sessions",
    ]


@dataclass
class OfficeClawSessionLayout:
    session_id: str
    history_path: Path
    full_files: list[Path]
    sessions_root: Path
    logs_root: Path

    @property
    def session_dir(self) -> Path:
        return self.sessions_root / self.session_id


def _normalize_session_id(session_id: str) -> str:
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id 不能为空")
    if not sid.startswith("officeclaw_"):
        sid = f"officeclaw_{sid}"
    return sid


def _find_history(session_dir: Path) -> Path | None:
    exact = session_dir / "history.json"
    if exact.is_file():
        return exact
    cands = sorted(session_dir.glob("history*.json"))
    return cands[0] if cands else None


def _find_full_logs(logs_root: Path) -> list[Path]:
    if not logs_root.is_dir():
        return []
    found: list[Path] = []
    for pattern in ("full*.json", "full*.txt", "full*.log"):
        found.extend(sorted(logs_root.glob(pattern)))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in sorted(found, key=lambda x: (x.stat().st_mtime, x.name)):
        rp = p.resolve()
        if rp not in seen and rp.is_file():
            seen.add(rp)
            out.append(rp)
    return out


def _first_existing_dir(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    return None


def _paths_hint() -> str:
    home = _user_home()
    return (
        "请向用户确认 OfficeClaw 日志路径并设置环境变量，或传入 --logs-root / --sessions-root：\n"
        f"  OFFICECLAW_SESSIONS（history.json 所在 sessions 根目录）\n"
        f"  OFFICECLAW_LOGS（含 full*.json/txt 的目录）\n"
        f"常见位置（当前用户 {home.name} 下，因安装方式而异）：\n"
        f"  {home / '.office-claw' / '.jiuwenclaw' / '...'}\n"
        f"  {home / 'office-claw.jiuwenclaw' / '...'}\n"
        "也可使用 -DataDir 指向同时含 history.json 与 full 日志的测试目录。"
    )


def resolve_officeclaw_session(
    session_id: str,
    *,
    logs_root: Path | None = None,
    sessions_root: Path | None = None,
) -> OfficeClawSessionLayout:
    sid = _normalize_session_id(session_id)

    if sessions_root:
        sessions = sessions_root.resolve()
    else:
        found = _first_existing_dir(_default_sessions_roots())
        if not found:
            raise FileNotFoundError(_paths_hint())
        sessions = found

    if logs_root:
        logs = logs_root.resolve()
    else:
        found = _first_existing_dir(_default_logs_roots())
        if not found:
            raise FileNotFoundError(_paths_hint())
        logs = found

    session_dir = sessions / sid
    if not session_dir.is_dir():
        raise FileNotFoundError(
            f"会话目录不存在: {session_dir}\n"
            "请向用户确认 session_id 是否正确，或提供正确的 OFFICECLAW_SESSIONS。"
        )

    history = _find_history(session_dir)
    if history is None:
        raise FileNotFoundError(f"未找到 history.json: {session_dir}")

    full_files = _find_full_logs(logs)
    if not full_files:
        raise FileNotFoundError(
            f"未在 {logs} 找到 full*.json / full*.txt。\n"
            "请向用户确认 OFFICECLAW_LOGS 或传入 --logs-root。"
        )

    return OfficeClawSessionLayout(
        session_id=sid,
        history_path=history.resolve(),
        full_files=full_files,
        sessions_root=sessions,
        logs_root=logs,
    )
