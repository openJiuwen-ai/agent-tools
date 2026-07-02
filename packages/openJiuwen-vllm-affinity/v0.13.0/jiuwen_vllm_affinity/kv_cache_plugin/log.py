import os

from vllm.logger import init_logger

# Must live under the "vllm" logger namespace so vLLM's logging config
# (handler + VLLM_LOGGING_LEVEL) applies. Loggers named jiuwen_vllm_affinity.*
# propagate to the root logger (default WARNING) and INFO is silently dropped.
logger = init_logger("vllm.jiuwen_vllm_affinity.kv_cache")

_KV_DEBUG_ENV = "JIUWEN_KV_DEBUG"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def kv_debug_enabled() -> bool:
    return os.environ.get(_KV_DEBUG_ENV, "").strip().lower() in _TRUTHY


def kv_log_info(msg: str, *args) -> None:
    """Emit [jiuwen-kv] trace logs when ``JIUWEN_KV_DEBUG=1`` (or true/yes/on)."""
    if kv_debug_enabled():
        logger.info(msg, *args)
