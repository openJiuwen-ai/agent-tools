from jiuwen_vllm_affinity.kv_cache_plugin.log import logger
from jiuwen_vllm_affinity.kv_cache_plugin.patcher import apply_patches


def register():
    logger.info("register jiuwen_vllm_affinity kv_cache_plugin")
    apply_patches()
