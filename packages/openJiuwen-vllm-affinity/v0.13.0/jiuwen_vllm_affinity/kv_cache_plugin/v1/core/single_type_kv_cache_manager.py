from typing import Sequence
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_session_manager import (
    KvCacheSessionManager,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def should_supplement_partial_tail(
    release_token_index: int | None,
    num_tokens: int | None,
    block_size: int,
) -> bool:
    if release_token_index is None or num_tokens is None:
        return True
    if block_size <= 0 or num_tokens <= 0:
        return False
    remainder = num_tokens % block_size
    if remainder == 0:
        return False
    partial_start = (num_tokens // block_size) * block_size
    return release_token_index <= partial_start


class SingleTypeKVCacheManagerEx(SingleTypeKVCacheManager):
    def __init__(
        self,
        kv_cache_spec,
        block_pool,
        kv_cache_group_id,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        super().__init__(
            kv_cache_spec, block_pool, kv_cache_group_id, dcp_world_size, pcp_world_size
        )
        self.kv_cache_session_manager = KvCacheSessionManager()

    def aging_block(
        self,
        session_id,
        block_hashes,
        *,
        release_token_index: int | None = None,
        num_tokens: int | None = None,
    ) -> int:
        resolved_blocks = []
        for block_hash in block_hashes:
            cached_block = self.block_pool.get_cached_block(
                block_hash, [self.kv_cache_group_id]
            )
            if cached_block:
                resolved_blocks.append(cached_block[0])
            else:
                break

        tail_blocks: list[KVCacheBlock] = []
        if should_supplement_partial_tail(
            release_token_index, num_tokens, self.block_size
        ):
            resolved_ids = {b.block_id for b in resolved_blocks}
            tail_blocks = self.kv_cache_session_manager.collect_orphan_blocks(
                session_id,
                self.block_pool.blocks,
                exclude_block_ids=resolved_ids,
                only_idle=True,
            )

        seen_ids = {b.block_id for b in resolved_blocks}
        for blk in tail_blocks:
            if blk.block_id not in seen_ids:
                resolved_blocks.append(blk)
                seen_ids.add(blk.block_id)

        session_released = self.kv_cache_session_manager.release_blocks(
            resolved_blocks, session_id
        )
        return self.block_pool.aging_block(session_released)

    def save_new_computed_blocks_with_session(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        session_id: str | None,
    ) -> None:
        """ """
        if request_id not in self.num_cached_block and session_id is not None:
            self.kv_cache_session_manager.add_blocks(new_computed_blocks, session_id)
        self.save_new_computed_blocks(request_id, new_computed_blocks)

    def allocate_new_blocks_with_session(
        self, request_id: str, num_tokens: int, session_id: str | None
    ) -> list[KVCacheBlock]:
        """ """
        blocks = self.allocate_new_blocks(request_id, num_tokens)
        if len(blocks) > 0 and session_id is not None:
            self.kv_cache_session_manager.reset_blocks(blocks, session_id)
        logger.debug("new block cnt %s", len(blocks))
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
