from typing import Callable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request
from jiuwen_vllm_affinity.kv_cache_plugin.v1.engine.core import unpack_sharing_cache_salt


def _packed_request_id_source(request: EngineCoreRequest) -> str:
    if request.external_req_id is not None:
        return request.external_req_id
    return request.request_id


@classmethod
def from_engine_core_request_v2(
    cls,
    request: EngineCoreRequest,
    block_hasher: Callable[["Request"], list["BlockHash"]] | None,
) -> "Request":
    sharing_cache_salt = unpack_sharing_cache_salt(_packed_request_id_source(request))
    req = cls(
        request_id=request.request_id,
        client_index=request.client_index,
        prompt_token_ids=request.prompt_token_ids,
        prompt_embeds=request.prompt_embeds,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        priority=request.priority,
        trace_headers=request.trace_headers,
        block_hasher=block_hasher,
        resumable=request.resumable,
        reasoning_ended=request.reasoning_ended,
    )
    if sharing_cache_salt is not None:
        req.sharing_cache_salt = sharing_cache_salt
    return req


def request_get_sharing_cache_salt(request) -> str | None:
    return (
        None
        if not hasattr(request, "sharing_cache_salt")
        else request.sharing_cache_salt
    )


def register_request():
    Request.from_engine_core_request = from_engine_core_request_v2
