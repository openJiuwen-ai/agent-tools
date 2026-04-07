"""OAuth 一次性会话（state / pending）：优先 Redis，否则进程内内存（单 worker 可用）。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

from plugins_market.core.config import settings

logger = logging.getLogger(__name__)


class OAuthStrStore(Protocol):
    def get(self, key: str) -> str | None:
        ...

    def set_ex(self, key: str, value: str, ttl_seconds: int) -> None:
        ...

    def delete(self, key: str) -> None:
        ...


class _MemoryOAuthStore:
    _data: dict[str, tuple[float, str]] = {}
    _lock = threading.Lock()

    def _purge_expired_locked(self) -> None:
        now = time.monotonic()
        dead = [k for k, (exp_at, _) in self._data.items() if exp_at <= now]
        for k in dead:
            self._data.pop(k, None)

    def get(self, key: str) -> str | None:
        with self._lock:
            self._purge_expired_locked()
            item = self._data.get(key)
            if not item:
                return None
            exp_at, val = item
            if exp_at <= time.monotonic():
                self._data.pop(key, None)
                return None
            return val

    def set_ex(self, key: str, value: str, ttl_seconds: int) -> None:
        with self._lock:
            self._purge_expired_locked()
            self._data[key] = (time.monotonic() + max(1, ttl_seconds), value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class _RedisOAuthStore:
    def __init__(self, *, host: str, port: int, db: int, password: str) -> None:
        import redis  # type: ignore[import-not-found]

        kwargs: dict = {
            "host": host,
            "port": port,
            "db": db,
            "decode_responses": True,
        }
        if password:
            kwargs["password"] = password
        self._r = redis.Redis(**kwargs)
        # 首次建连：未 PING 则仅创建客户端对象，网络/密码错误要在第一次命令时才会暴露
        self._r.ping()

    def get(self, key: str) -> str | None:
        v = self._r.get(key)
        return v if isinstance(v, str) else None

    def set_ex(self, key: str, value: str, ttl_seconds: int) -> None:
        self._r.setex(key, max(1, ttl_seconds), value)

    def delete(self, key: str) -> None:
        self._r.delete(key)


_memory = _MemoryOAuthStore()
_redis_store: _RedisOAuthStore | None = None
_redis_init_attempted = False
_warned_memory_no_redis_host = False


def get_oauth_str_store() -> OAuthStrStore:
    global _redis_store, _redis_init_attempted, _warned_memory_no_redis_host
    host = (settings.redis_host or "").strip()
    if not host and not _warned_memory_no_redis_host:
        _warned_memory_no_redis_host = True
        logger.warning(
            "OAuth session store: no REDIS_HOST/MARKET_REDIS_HOST; using in-process memory only. "
            "GitCode OAuth breaks when /start and /callback hit different workers or pods — configure shared Redis."
        )
    if host and not _redis_init_attempted:
        _redis_init_attempted = True
        try:
            _redis_store = _RedisOAuthStore(
                host=host,
                port=int(settings.redis_port),
                db=int(settings.redis_db),
                password=(settings.redis_password or "").strip(),
            )
            logger.info("OAuth session store: Redis (PING OK, host=%s port=%s db=%s)", host, settings.redis_port,
                        settings.redis_db)
            return _redis_store
        except Exception as e:
            logger.warning("OAuth session store: Redis init failed (%s), using memory", e)
    if _redis_store is not None:
        return _redis_store
    return _memory
