import logging
from jiuwen_vllm_affinity.kv_cache_plugin.engine.protocol import (
    register_engine_protocol,
)
import jiuwen_vllm_affinity.kv_cache_plugin.entrypoints.openai.api_server
from jiuwen_vllm_affinity.kv_cache_plugin.entrypoints.openai.protocol import (
    register_chat_request,
)
from jiuwen_vllm_affinity.kv_cache_plugin.entrypoints.openai.serving_chat import (
    register_openai_serving,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.sched.interface import (
    register_scheduler_interface,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.sched.scheduler import (
    register_scheduler,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.block_pool import register_block_pool
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_coordinator import (
    register_kv_cache_coordinator,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_manager import (
    register_kv_cache_manager,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.single_type_kv_cache_manager import (
    register_single_type_kv_cache_manager,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.engine.async_llm import (
    register_engine_client,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.engine.core import register_engine_core
from jiuwen_vllm_affinity.kv_cache_plugin.v1.engine.core_client import (
    register_engine_core_client,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.request import register_request


def apply_patches():
    register_engine_protocol()
    register_chat_request()
    register_openai_serving()
    register_scheduler_interface()
    register_scheduler()
    register_block_pool()
    register_kv_cache_coordinator()
    register_kv_cache_manager()
    register_single_type_kv_cache_manager()
    register_engine_client()
    register_engine_core()
    register_engine_core_client()
    register_request()
