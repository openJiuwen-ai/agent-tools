from collections.abc import Sequence

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.serial_utils import bytestr
from jiuwen_vllm_affinity.kv_cache_plugin.engine.protocol import EngineClientEx


class AsyncLLMEx(EngineClientEx):
    async def release_kv_cache(
        self, session_id: str, token_requests: list[tuple[Sequence[bytestr], int]]
    ) -> int:
        return await self.engine_core.release_kv_cache(session_id, token_requests)


def register_engine_client():
    AsyncLLM.release_kv_cache = AsyncLLMEx.release_kv_cache
