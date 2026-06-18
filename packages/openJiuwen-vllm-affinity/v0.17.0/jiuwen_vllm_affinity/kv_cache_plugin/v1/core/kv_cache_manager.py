# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2026 Huawei Technologies Co., Ltd.
# 本文件基于vLLM v0.17.0的KVCacheManager实现修改而来，用于实现双区缓存特性

from jiuwen_vllm_affinity.kv_cache_plugin.v1.request import request_get_sharing_cache_salt
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request


class KVCacheManagerEx(KVCacheManager):
    def release_kv_cache(self, session_id: str, block_hashes: list[BlockHash]) -> int:
        return self.coordinator.aging_block(session_id, block_hashes)

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
    ) -> KVCacheBlocks | None:
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens,
            self.max_model_len,
        )

        self.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None

        sharing_salt = request_get_sharing_cache_salt(request)
        self.coordinator.set_jiuwen_sharing_session_id(sharing_salt)
        try:
            if (
                new_computed_block_list is not self.empty_kv_cache_blocks.blocks
                or num_external_computed_tokens > 0
            ):
                self.coordinator.allocate_new_computed_blocks(
                    request_id=request.request_id,
                    new_computed_blocks=new_computed_block_list,
                    num_local_computed_tokens=num_local_computed_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                )

            new_blocks = self.coordinator.allocate_new_blocks(
                request.request_id,
                num_tokens_need_slot,
                num_tokens_main_model,
                num_encoder_tokens,
            )
        finally:
            self.coordinator.clear_jiuwen_sharing_session_id()

        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)


def register_kv_cache_manager():
    KVCacheManager.allocate_slots = KVCacheManagerEx.allocate_slots
    KVCacheManager.release_kv_cache = KVCacheManagerEx.release_kv_cache
