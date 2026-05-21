from typing import Sequence

from vllm.v1.engine.core_client import AsyncMPClient, EngineCoreClient, InprocClient
from vllm.v1.serial_utils import bytestr


class EngineCoreClientEx:
    async def release_kv_cache(
        self, session_id: str, token_requests: list[tuple[Sequence[bytestr], int]]
    ) -> int:
        pass


class AsyncMPClientEx:
    async def release_kv_cache(
        self, session_id: str, token_requests: list[tuple[Sequence[bytestr], int]]
    ) -> int:
        return await self.call_utility_async(
            "release_kv_cache", session_id, token_requests
        )


class InprocClientEx:
    async def release_kv_cache(
        self, session_id: str, token_requests: list[tuple[Sequence[bytestr], int]]
    ) -> int:
        return self.engine_core.release_kv_cache(session_id, token_requests)


def register_engine_core_client():
    EngineCoreClient.release_kv_cache = EngineCoreClientEx.release_kv_cache
    AsyncMPClient.release_kv_cache = AsyncMPClientEx.release_kv_cache
    InprocClient.release_kv_cache = InprocClientEx.release_kv_cache
