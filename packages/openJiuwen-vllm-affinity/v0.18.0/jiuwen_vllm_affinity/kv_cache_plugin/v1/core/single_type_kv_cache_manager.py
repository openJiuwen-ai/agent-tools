# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2026 Huawei Technologies Co., Ltd.
# 本文件基于vLLM v0.18.0的SingleTypeKVCacheManager实现修改而来，用于实现双区缓存特性

from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_session_manager import (
    KvCacheSessionManager,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.request import (
    request_get_session_binding_id,
)
from jiuwen_vllm_affinity.kv_cache_plugin.kv_debug import (
    kv_debug_enabled,
    kv_label_session,
    logger,
)
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.two_phase_block_queue import (
    jiuwen_clear_reclaimed,
    jiuwen_is_reclaimed,
)

_allocate_new_orig = SingleTypeKVCacheManager.allocate_new_blocks
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


def _set_alloc_log_context(block_pool, session_id: str | None) -> None:
    block_pool.jiuwen_alloc_session_id = session_id


def _clear_alloc_log_context(block_pool) -> None:
    block_pool.jiuwen_alloc_session_id = None


def _sessions_on_block(
    session_manager: KvCacheSessionManager, block_id: int
) -> list[str]:
    sessions = session_manager.block_to_sessions
    if block_id < 0 or block_id >= len(sessions):
        return []
    return list(sessions[block_id])


def bind_allocated_blocks(
    session_manager: KvCacheSessionManager,
    blocks: Sequence[KVCacheBlock],
    session_id: str | None,
) -> None:
    """Bind newly allocated blocks to a session.

    Fresh blocks use append semantics (``add_blocks``). Blocks reclaimed from the
    free queue (``reclaimed`` sidecar flag) replace any stale ``block_to_sessions``
    entries via ``reset_blocks`` before the new owner is recorded (or cleared
    when ``session_id`` is None).
    """
    if not blocks:
        return

    fresh: list[KVCacheBlock] = []
    reclaimed: list[KVCacheBlock] = []
    for blk in blocks:
        if blk.is_null or blk.block_id < 0:
            continue
        if jiuwen_is_reclaimed(blk):
            reclaimed.append(blk)
            jiuwen_clear_reclaimed(blk)
        else:
            fresh.append(blk)

    if reclaimed:
        session_manager.reset_blocks(reclaimed, session_id)
    if fresh and session_id is not None:
        session_manager.add_blocks(fresh, session_id)


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

    if kv_debug_enabled():
        reused_block_ids = [
            b.block_id for b in new_computed_blocks if not b.is_null
        ]
        if reused_block_ids:
            owners_sample: list[str] = []
            if session_id is not None:
                for bid in reused_block_ids[:3]:
                    owners = _sessions_on_block(
                        self.kv_cache_session_manager, bid
                    )
                    owners_sample.append(f"{bid}:{owners}")
            logger.info(
                "[jiuwen-kv] 2-prefix_hit: session=%s block_ids=%s "
                "owners_before_bind=%s",
                kv_label_session(session_id),
                reused_block_ids,
                owners_sample,
            )

    if session_id is not None and len(new_computed_blocks) > 0:
        self.kv_cache_session_manager.add_blocks(new_computed_blocks, session_id)

    if num_external_computed_tokens > 0:
        if kv_debug_enabled():
            try:
                _set_alloc_log_context(self.block_pool, session_id)
                allocated_blocks = self.block_pool.get_new_blocks(
                    cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
                )
            finally:
                _clear_alloc_log_context(self.block_pool)
        else:
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
        req_blocks.extend(allocated_blocks)
        if isinstance(self.kv_cache_spec, FullAttentionSpec):
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)
        bind_allocated_blocks(
            self.kv_cache_session_manager, allocated_blocks, session_id
        )


def allocate_new_blocks_jiuwen(
    self, request_id: str, num_tokens: int, num_tokens_main_model: int
):
    session_id = self.get_jiuwen_sharing_session_id()
    if kv_debug_enabled():
        try:
            _set_alloc_log_context(self.block_pool, session_id)
            blocks = _allocate_new_orig(
                self, request_id, num_tokens, num_tokens_main_model
            )
        finally:
            _clear_alloc_log_context(self.block_pool)
    else:
        blocks = _allocate_new_orig(
            self, request_id, num_tokens, num_tokens_main_model
        )
    bind_allocated_blocks(self.kv_cache_session_manager, blocks, session_id)
    return blocks


def cache_blocks_jiuwen(self, request, num_tokens: int) -> None:
    _cache_blocks_orig(self, request, num_tokens)
    session_id = request_get_session_binding_id(request)
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
    def set_jiuwen_sharing_session_id(self, session_id: str | None) -> None:
        self.jiuwen_sharing_session_id = session_id

    def get_jiuwen_sharing_session_id(self) -> str | None:
        return getattr(self, "jiuwen_sharing_session_id", None)

    def clear_jiuwen_sharing_session_id(self) -> None:
        self.jiuwen_sharing_session_id = None

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

        debug_kv = kv_debug_enabled()
        owned_count = 0
        owned_block_ids: set[int] = set()
        if debug_kv and session_id is not None:
            sessions = self.kv_cache_session_manager.block_to_sessions
            for blk in resolved_blocks:
                bid = blk.block_id
                if bid >= 0 and bid < len(sessions) and session_id in sessions[bid]:
                    owned_count += 1
                    owned_block_ids.add(bid)

        session_released = self.kv_cache_session_manager.release_blocks(
            resolved_blocks, session_id
        )
        aged_count = self.block_pool.aging_block(session_released)

        if not debug_kv or session_id is None or owned_count == 0:
            return len(session_released)

        shared_remain = 0
        sessions = self.kv_cache_session_manager.block_to_sessions
        for bid in owned_block_ids:
            if bid < len(sessions) and len(sessions[bid]) > 0:
                shared_remain += 1

        freed_ids = [b.block_id for b in session_released]
        if freed_ids:
            pending = len(session_released) - aged_count
            logger.info(
                "[jiuwen-kv] 3-release: session=%s unbound=%d freed=%d "
                "shared_remain=%d block_ids_freed=%s aged=%d pending=%d",
                kv_label_session(session_id),
                owned_count,
                len(session_released),
                shared_remain,
                freed_ids,
                aged_count,
                pending,
            )
        else:
            logger.info(
                "[jiuwen-kv] 3-release: session=%s unbound=%d freed=0 "
                "shared_remain=%d",
                kv_label_session(session_id),
                owned_count,
                shared_remain,
            )
        return len(session_released)


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
    SingleTypeKVCacheManager.cache_blocks = cache_blocks_jiuwen
