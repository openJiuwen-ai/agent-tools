# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2026 Huawei Technologies Co., Ltd.
# 本文件基于vLLM v0.13.0的SingleTypeKVCacheManager实现修改而来，用于实现双区缓存特性

from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_session_manager import (
    KvCacheSessionManager,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.request import request_get_sharing_cache_salt
from jiuwen_vllm_affinity.kv_cache_plugin.log import kv_log_info

_cache_blocks_orig = SingleTypeKVCacheManager.cache_blocks


def expected_release_block_count(
    release_token_index: int | None,
    num_tokens: int | None,
    block_size: int,
) -> int | None:
    if release_token_index is None or num_tokens is None:
        return None
    if block_size <= 0 or num_tokens <= 0:
        return 0
    remaining = max(0, num_tokens - release_token_index)
    if remaining == 0:
        return 0
    return cdiv(remaining, block_size)


def should_collect_orphan_tail(
    release_token_index: int | None,
    num_tokens: int | None,
    block_size: int,
    hash_resolved_count: int,
    hash_suffix_len: int,
) -> bool:
    """Collect session orphans when prefix hash did not resolve every physical block."""
    if hash_suffix_len > 0 and hash_resolved_count == hash_suffix_len:
        return True
    expected = expected_release_block_count(
        release_token_index, num_tokens, block_size
    )
    if expected is None:
        return True
    return hash_resolved_count < expected


def _session_owns_block(
    session_manager: KvCacheSessionManager,
    session_id: str,
    block_id: int,
) -> bool:
    if block_id < 0:
        return False
    sessions = session_manager.block_to_sessions
    return block_id < len(sessions) and session_id in sessions[block_id]


def collect_consecutive_session_tail(
    session_manager: KvCacheSessionManager,
    session_id: str | None,
    pool_blocks: list[KVCacheBlock],
    hash_resolved_blocks: list[KVCacheBlock],
    *,
    only_idle: bool = True,
) -> list[KVCacheBlock]:
    """Session-bound blocks immediately after the hash-resolved prefix (e.g. 213 after 206-212)."""
    if session_id is None or not hash_resolved_blocks:
        return []

    max_resolved_id = max(b.block_id for b in hash_resolved_blocks)
    tail: list[KVCacheBlock] = []
    next_id = max_resolved_id + 1
    while next_id < len(pool_blocks):
        blk = pool_blocks[next_id]
        if blk.is_null:
            break
        if not _session_owns_block(session_manager, session_id, next_id):
            break
        if only_idle and blk.ref_cnt != 0:
            break
        tail.append(blk)
        next_id += 1
    return tail


def collect_release_tail_orphans(
    session_manager: KvCacheSessionManager,
    session_id: str | None,
    pool_blocks: list[KVCacheBlock],
    hash_resolved_blocks: list[KVCacheBlock],
    *,
    release_token_index: int | None,
    num_tokens: int | None,
    block_size: int,
    hash_suffix_len: int,
    only_idle: bool = True,
) -> list[KVCacheBlock]:
    expected = expected_release_block_count(
        release_token_index, num_tokens, block_size
    )
    exclude_ids = {b.block_id for b in hash_resolved_blocks}

    tail: list[KVCacheBlock] = []
    if hash_resolved_blocks:
        tail = collect_consecutive_session_tail(
            session_manager,
            session_id,
            pool_blocks,
            hash_resolved_blocks,
            only_idle=only_idle,
        )
        if not tail and only_idle:
            tail = collect_consecutive_session_tail(
                session_manager,
                session_id,
                pool_blocks,
                hash_resolved_blocks,
                only_idle=False,
            )

    if expected is None:
        return tail

    need = expected - len(hash_resolved_blocks)
    if need <= len(tail):
        return tail

    still_need = need - len(tail)
    tail_ids = {b.block_id for b in tail}
    orphans = session_manager.collect_orphan_blocks(
        session_id,
        pool_blocks,
        exclude_block_ids=exclude_ids | tail_ids,
        only_idle=only_idle,
    )
    if still_need > 0 and only_idle and len(orphans) < still_need:
        orphans = session_manager.collect_orphan_blocks(
            session_id,
            pool_blocks,
            exclude_block_ids=exclude_ids | tail_ids,
            only_idle=False,
        )
    orphans.sort(key=lambda b: b.block_id, reverse=True)
    orphans = orphans[:still_need]
    orphans.sort(key=lambda b: b.block_id)
    return tail + orphans


def cache_blocks_jiuwen(self, request, num_tokens: int) -> None:
    _cache_blocks_orig(self, request, num_tokens)
    session_id = request_get_sharing_cache_salt(request)
    if session_id is None:
        return
    req_blocks = self.req_to_blocks.get(request.request_id, [])
    num_blocks_needed = cdiv(num_tokens, self.block_size)
    blocks_to_bind = [
        b for b in req_blocks[:num_blocks_needed] if not b.is_null
    ]
    if blocks_to_bind:
        self.kv_cache_session_manager.add_blocks(blocks_to_bind, session_id)


class SingleTypeKVCacheManagerEx(SingleTypeKVCacheManager):
    def aging_block(
        self,
        session_id,
        block_hashes,
        *,
        release_token_index: int | None = None,
        num_tokens: int | None = None,
    ) -> int:
        hash_resolved_blocks: list[KVCacheBlock] = []
        for block_hash in block_hashes:
            cached_block = self.block_pool.get_cached_block(
                block_hash, [self.kv_cache_group_id]
            )
            if cached_block:
                hash_resolved_blocks.append(cached_block[0])
            else:
                break

        tail_blocks: list[KVCacheBlock] = []
        hash_suffix_len = len(block_hashes)
        if should_collect_orphan_tail(
            release_token_index,
            num_tokens,
            self.block_size,
            len(hash_resolved_blocks),
            hash_suffix_len,
        ):
            tail_blocks = collect_release_tail_orphans(
                self.kv_cache_session_manager,
                session_id,
                self.block_pool.blocks,
                hash_resolved_blocks,
                release_token_index=release_token_index,
                num_tokens=num_tokens,
                block_size=self.block_size,
                hash_suffix_len=hash_suffix_len,
            )

        resolved_blocks = list(hash_resolved_blocks)
        seen_ids = {b.block_id for b in resolved_blocks}
        for blk in tail_blocks:
            if blk.block_id not in seen_ids:
                resolved_blocks.append(blk)
                seen_ids.add(blk.block_id)

        session_released = self.kv_cache_session_manager.release_blocks(
            resolved_blocks, session_id
        )
        aged_count = self.block_pool.aging_block(session_released)
        if session_released:
            block_ids = [b.block_id for b in session_released]
            pending = len(session_released) - aged_count
            if aged_count > 0:
                if pending > 0:
                    kv_log_info(
                        "[jiuwen-kv] 3-release: session_unbound block_ids=%s, "
                        "aged_immediately=%d, pending_aging=%d",
                        block_ids,
                        aged_count,
                        pending,
                    )
                else:
                    kv_log_info(
                        "[jiuwen-kv] 3-release: session_unbound block_ids=%s, "
                        "aged_immediately=%d",
                        block_ids,
                        aged_count,
                    )
            else:
                kv_log_info(
                    "[jiuwen-kv] 3-release: session_unbound block_ids=%s, "
                    "pending_aging=%d",
                    block_ids,
                    pending,
                )
        # Return session-unbound count: blocks still referenced (ref_cnt>0) or
        # not yet in the free queue are deferred via pending_aging on free.
        return len(session_released)

    def save_new_computed_blocks_with_session(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        session_id: str | None,
    ) -> None:
        if request_id not in self.num_cached_block and session_id is not None:
            self.kv_cache_session_manager.add_blocks(new_computed_blocks, session_id)

        reused_block_ids = [
            b.block_id for b in new_computed_blocks if not b.is_null
        ]
        if reused_block_ids:
            kv_log_info(
                "[jiuwen-kv] 2-prefix_hit: block_ids=%s",
                reused_block_ids,
            )

        self.save_new_computed_blocks(request_id, new_computed_blocks)

    def allocate_new_blocks_with_session(
        self, request_id: str, num_tokens: int, session_id: str | None
    ) -> list[KVCacheBlock]:
        blocks = self.allocate_new_blocks(request_id, num_tokens)
        if len(blocks) > 0 and session_id is not None:
            self.kv_cache_session_manager.add_blocks(blocks, session_id)
        return blocks


def replace_single_type_kv_cache_manager_init():
    origin_init = SingleTypeKVCacheManager.__init__

    def new_init(
        self,
        kv_cache_spec,
        block_pool,
        kv_cache_group_id,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        origin_init(
            self,
            kv_cache_spec,
            block_pool,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        self.kv_cache_session_manager = KvCacheSessionManager()

    SingleTypeKVCacheManager.__init__ = new_init


def register_single_type_kv_cache_manager():
    replace_single_type_kv_cache_manager_init()
    SingleTypeKVCacheManager.aging_block = SingleTypeKVCacheManagerEx.aging_block
    SingleTypeKVCacheManager.allocate_new_blocks_with_session = (
        SingleTypeKVCacheManagerEx.allocate_new_blocks_with_session
    )
    SingleTypeKVCacheManager.save_new_computed_blocks_with_session = (
        SingleTypeKVCacheManagerEx.save_new_computed_blocks_with_session
    )
    SingleTypeKVCacheManager.cache_blocks = cache_blocks_jiuwen
