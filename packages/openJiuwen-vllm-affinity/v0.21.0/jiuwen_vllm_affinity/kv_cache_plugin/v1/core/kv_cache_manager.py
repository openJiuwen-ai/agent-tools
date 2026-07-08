# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2026 Huawei Technologies Co., Ltd.
# 本文件基于vLLM v0.21.0的KVCacheManager实现修改而来，用于实现双区缓存特性

from jiuwen_vllm_affinity.kv_cache_plugin.v1.request import (
    request_get_session_binding_id,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request


class KVCacheManagerEx(KVCacheManager):
    _orig_allocate_slots = KVCacheManager.allocate_slots

    def release_kv_cache(
        self,
        session_id: str,
        block_hashes: list[BlockHash],
        *,
        release_token_index: int | None = None,
        num_tokens: int | None = None,
    ) -> int:
        return self.coordinator.aging_block(
            session_id,
            block_hashes,
            release_token_index=release_token_index,
            num_tokens=num_tokens,
        )

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
    ) -> KVCacheBlocks | None:
        session_binding_id = request_get_session_binding_id(request)
        self.coordinator.set_jiuwen_sharing_session_id(session_binding_id)
        try:
            kv_blocks = KVCacheManagerEx._orig_allocate_slots(
                self,
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=num_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=delay_cache_blocks,
                num_encoder_tokens=num_encoder_tokens,
                full_sequence_must_fit=full_sequence_must_fit,
            )
        finally:
            self.coordinator.clear_jiuwen_sharing_session_id()

        if kv_blocks is None:
            return None

        # Partial tail blocks (e.g. block 213 for 1000 tokens) are never prefix-cached
        # but must be session-bound for release_kv_cache orphan collection.
        if session_binding_id is not None:
            for manager in self.coordinator.single_type_managers:
                req_blocks = manager.req_to_blocks.get(request.request_id, [])
                live_blocks = [b for b in req_blocks if not b.is_null]
                if live_blocks:
                    manager.kv_cache_session_manager.add_blocks(
                        live_blocks, session_binding_id
                    )
        return kv_blocks


def register_kv_cache_manager():
    KVCacheManager.allocate_slots = KVCacheManagerEx.allocate_slots
    KVCacheManager.release_kv_cache = KVCacheManagerEx.release_kv_cache
