"""KVBM 与外部推理引擎的集成入口。"""

from jiuwen_kvbm.integrations.scheduler_kv_bridge import (
    SchedulerKVBridge,
    SchedulerKVOwnershipMode,
)
from jiuwen_kvbm.integrations.vllm_ascend_018 import (
    get_bound_vllm_ascend_018_dataplane,
    get_bound_vllm_ascend_018_physical_block_resolver,
    install_vllm_ascend_018_block_pool_hook,
    install_vllm_ascend_018_engine_core_hook,
    install_vllm_ascend_018_kv_cache_manager_hook,
    install_vllm_ascend_018_kv_event_publisher_hook,
    install_vllm_ascend_018_multiprocess_hooks,
)
from jiuwen_kvbm.integrations.vllm_ascend_023 import (
    get_bound_vllm_ascend_023_dataplane,
    get_bound_vllm_ascend_023_physical_block_resolver,
    install_vllm_ascend_023_block_pool_hook,
    install_vllm_ascend_023_engine_core_hook,
    install_vllm_ascend_023_kv_cache_manager_hook,
    install_vllm_ascend_023_kv_event_publisher_hook,
    install_vllm_ascend_023_multiprocess_hooks,
    install_vllm_ascend_023_scheduler_diagnostics_hook,
    validate_vllm_ascend_023_runtime,
)

__all__ = [
    "SchedulerKVBridge",
    "SchedulerKVOwnershipMode",
    "get_bound_vllm_ascend_018_dataplane",
    "get_bound_vllm_ascend_018_physical_block_resolver",
    "install_vllm_ascend_018_block_pool_hook",
    "install_vllm_ascend_018_engine_core_hook",
    "install_vllm_ascend_018_kv_cache_manager_hook",
    "install_vllm_ascend_018_kv_event_publisher_hook",
    "install_vllm_ascend_018_multiprocess_hooks",
    "get_bound_vllm_ascend_023_dataplane",
    "get_bound_vllm_ascend_023_physical_block_resolver",
    "install_vllm_ascend_023_block_pool_hook",
    "install_vllm_ascend_023_engine_core_hook",
    "install_vllm_ascend_023_kv_cache_manager_hook",
    "install_vllm_ascend_023_kv_event_publisher_hook",
    "install_vllm_ascend_023_multiprocess_hooks",
    "install_vllm_ascend_023_scheduler_diagnostics_hook",
    "validate_vllm_ascend_023_runtime",
]
