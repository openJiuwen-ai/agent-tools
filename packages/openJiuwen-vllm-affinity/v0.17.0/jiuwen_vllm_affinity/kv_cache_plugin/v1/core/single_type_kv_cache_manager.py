# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2026 Huawei Technologies Co., Ltd.
# 本文件基于vLLM v0.17.0的SingleTypeKVCacheManager实现修改而来，用于实现双区缓存特性

from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_session_manager import (
    KvCacheSessionManager,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_allocate_new_orig = SingleTypeKVCacheManager.allocate_new_blocks


def allocate_new_computed_blocks_jiuwen(
    self,
    request_id: str,
    new_computed_blocks: Sequence[KVCacheBlock],
    num_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> None:
    session_id = self.get_jiuwen_sharing_session_id()

    if request_id in self.num_cached_block:
        if len(new_computed_blocks) != 0:
            raise ValueError(
                "Running request should not have new computed blocks"
            )
        return

    req_blocks = self.req_to_blocks[request_id]
    if len(req_blocks) != 0:
        raise ValueError("New request should have no allocated blocks yet")
    num_total_computed_tokens = (
        num_local_computed_tokens + num_external_computed_tokens
    )
    num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
    num_skipped_blocks = num_skipped_tokens // self.block_size
    if num_skipped_blocks > 0:
        new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
        num_external_computed_tokens = min(
            num_total_computed_tokens - num_skipped_tokens,
            num_external_computed_tokens,
        )

    if self.enable_caching:
        self.block_pool.touch(new_computed_blocks)
    elif any(new_computed_blocks):
        raise ValueError(
            "Computed blocks should be empty when prefix caching is disabled"
        )

    req_blocks.extend([self.block_pool.null_block] * num_skipped_blocks)
    req_blocks.extend(new_computed_blocks)
    self.num_cached_block[request_id] = len(req_blocks)

    if session_id is not None and len(new_computed_blocks) > 0:
        self.kv_cache_session_manager.add_blocks(new_computed_blocks, session_id)

    if num_external_computed_tokens > 0:
        allocated_blocks = self.block_pool.get_new_blocks(
            cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        req_blocks.extend(allocated_blocks)
        if session_id is not None and allocated_blocks:
            self.kv_cache_session_manager.reset_blocks(allocated_blocks, session_id)


def allocate_new_blocks_jiuwen(
    self, request_id: str, num_tokens: int, num_tokens_main_model: int
):
    blocks = _allocate_new_orig(self, request_id, num_tokens, num_tokens_main_model)
    session_id = self.get_jiuwen_sharing_session_id()
    if len(blocks) > 0 and session_id is not None:
        self.kv_cache_session_manager.reset_blocks(blocks, session_id)
        logger.debug("new block cnt %s", len(blocks))
    return blocks


class SingleTypeKVCacheManagerEx(SingleTypeKVCacheManager):
    def set_jiuwen_sharing_session_id(self, session_id: str | None) -> None:
        self.jiuwen_sharing_session_id = session_id

    def get_jiuwen_sharing_session_id(self) -> str | None:
        return getattr(self, "jiuwen_sharing_session_id", None)

    def clear_jiuwen_sharing_session_id(self) -> None:
        self.jiuwen_sharing_session_id = None

    def aging_block(self, session_id, block_hashes) -> int:
        aging_blocks = []
        for block_hash in block_hashes:
            cached_block = self.block_pool.get_cached_block(
                block_hash, [self.kv_cache_group_id]
            )
            if cached_block:
                aging_blocks.append(cached_block[0])
            else:
                break
        aging_blocks = self.kv_cache_session_manager.release_blocks(
            aging_blocks, session_id
        )
        return self.block_pool.aging_block(aging_blocks)


def replace_single_type_kv_cache_manager_init():
    origin_init = SingleTypeKVCacheManager.__init__

    def new_init(
        self,
        kv_cache_spec,
        block_pool,
        enable_caching,
        kv_cache_group_id,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        origin_init(
            self,
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        self.kv_cache_session_manager = KvCacheSessionManager()

    SingleTypeKVCacheManager.__init__ = new_init


def register_single_type_kv_cache_manager():
    replace_single_type_kv_cache_manager_init()
    SingleTypeKVCacheManager.set_jiuwen_sharing_session_id = (
        SingleTypeKVCacheManagerEx.set_jiuwen_sharing_session_id
    )
    SingleTypeKVCacheManager.get_jiuwen_sharing_session_id = (
        SingleTypeKVCacheManagerEx.get_jiuwen_sharing_session_id
    )
    SingleTypeKVCacheManager.clear_jiuwen_sharing_session_id = (
        SingleTypeKVCacheManagerEx.clear_jiuwen_sharing_session_id
    )
    SingleTypeKVCacheManager.aging_block = SingleTypeKVCacheManagerEx.aging_block
    SingleTypeKVCacheManager.allocate_new_computed_blocks = (
        allocate_new_computed_blocks_jiuwen
    )
    SingleTypeKVCacheManager.allocate_new_blocks = allocate_new_blocks_jiuwen
