"""
openjiuwentools-infer-router

KV Cache aware inference routing system for vLLM and SGLang.
"""

__version__ = "0.1.0"

# 命名空间包设置
try:
    import pkgutil

    _extended_path = pkgutil.extend_path(__path__, __name__)
    __path__[:] = _extended_path
except ImportError:
    pass


def __getattr__(name):
    if name == "settings":
        from openjiuwentools.infer_router.config.config import settings
        return settings
    if name == "KVCacheManager":
        from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
        return KVCacheManager
    if name == "Router":
        from openjiuwentools.infer_router.routing.router import Router
        return Router
    if name == "WorkerManager":
        from openjiuwentools.infer_router.worker.worker_manager import WorkerManager
        return WorkerManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "settings",
    "KVCacheManager",
    "Router",
    "WorkerManager",
]
