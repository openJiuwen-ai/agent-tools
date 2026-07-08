# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2026 Huawei Technologies Co., Ltd.
# 本文件基于vLLM v0.21.0的Request实现修改而来，用于实现双区缓存特性

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
        prompt_is_token_ids=getattr(request, "prompt_is_token_ids", None),
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
        reasoning_parser_kwargs=getattr(request, "reasoning_parser_kwargs", None),
        abort_immediately=getattr(request, "abort_immediately", False),
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


def request_get_session_binding_id(request) -> str | None:
    """Session id used for block_to_sessions binding and release_kv_cache.

    - cache_sharing=true: sharing_cache_salt packed in request_id
    - cache_sharing=false: cache_salt on the request (prefix isolation per client)
    """
    sharing_salt = request_get_sharing_cache_salt(request)
    if sharing_salt:
        return sharing_salt
    cache_salt = getattr(request, "cache_salt", None)
    if cache_salt:
        return cache_salt
    return None


def register_request():
    Request.from_engine_core_request = from_engine_core_request_v2
