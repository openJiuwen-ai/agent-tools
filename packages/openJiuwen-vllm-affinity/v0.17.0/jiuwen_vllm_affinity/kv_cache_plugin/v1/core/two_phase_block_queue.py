# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2026 Huawei Technologies Co., Ltd.
# 本文件基于vLLM v0.17.0的FreeKVCacheBlockQueue实现修改而来，用于实现双区缓存特性

from vllm.v1.core.kv_cache_utils import KVCacheBlock


class TwoPhaseBlockQueue:
    """Doubly-linked free block queue with free / release / normal tiers.

    Physical layout (LRU -> MRU, head is always the next block to allocate)::

        fake_head -> [free ...][release ...] -> released_tail -> [normal ...] -> fake_tail

    Three pointers fully describe the queue:

    - ``fake_free_list_head`` / head: allocation always pops from here. Because the
      free segment physically precedes release, which precedes normal, popping from
      the head naturally yields the priority ``free > release > normal`` without any
      per-segment scanning.
    - ``released_tail``: MRU of the *reusable* region (free + release); i.e. the last
      block before the normal segment. It is ``fake_head`` when that region is empty.
      ``aging_block`` inserts right after it, so it only ever advances toward the tail.
    - ``fake_free_list_tail`` / tail: ``append`` puts freed (but not released) blocks
      here, forming the normal segment.

    Logical states while sitting in the queue:

    - free:    never popped from this queue (block has no ``jiuwen_allocated`` mark)
    - release: popped before, then moved here by ``aging_block`` (release_kv_cache)
    - normal:  popped before, then returned via ``append`` (plain free)

    We do not track the free/release boundary at all: both are drained from the head
    in order, so ``released_tail`` (the reusable|normal divider) is the only structural
    pointer. Whether a popped block is fresh or reused is recorded on the block itself
    (``jiuwen_allocated``) purely to label the ``4-evict`` debug log.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
            # All blocks start in the (free) reusable region; its MRU is the last block.
            self.released_tail = blocks[-1]
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
            self.released_tail = self.fake_free_list_head

    @staticmethod
    def _unlink(block: KVCacheBlock) -> None:
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"unlink() called on an invalid block: {block}")
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block

    @staticmethod
    def _link_after(anchor: KVCacheBlock, block: KVCacheBlock) -> None:
        after = anchor.next_free_block
        block.prev_free_block = anchor
        block.next_free_block = after
        anchor.next_free_block = block
        after.prev_free_block = block

    def _take_head(self) -> KVCacheBlock:
        """Detach and return the head block (no num_free_blocks bookkeeping)."""
        head = self.fake_free_list_head.next_free_block
        if head is None or head is self.fake_free_list_tail:
            raise ValueError("No free blocks available")

        # free if it was never popped before; release/normal otherwise.
        reclaimed = getattr(head, "jiuwen_allocated", False)
        head.jiuwen_allocated = True

        # Popping the last reusable block empties the reusable region.
        if head is self.released_tail:
            self.released_tail = self.fake_free_list_head

        self._unlink(head)
        head.prev_free_block = head.next_free_block = None
        head.jiuwen_reclaimed = reclaimed
        return head

    def popleft(self) -> KVCacheBlock:
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            if self.num_free_blocks != 0:
                raise RuntimeError(
                    f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                    "with the free list."
                )
            raise ValueError("No free blocks available")

        block = self._take_head()
        self.num_free_blocks -= 1
        return block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n == 0:
            return []
        if self.num_free_blocks < n:
            raise ValueError(
                f"Not enough free blocks: requested {n}, "
                f"available {self.num_free_blocks}"
            )

        ret = [self._take_head() for _ in range(n)]
        self.num_free_blocks -= n
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """Pull an in-queue block out (e.g. prefix-cache hit via ``touch``)."""
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        block.jiuwen_allocated = True
        if block is self.released_tail:
            self.released_tail = block.prev_free_block

        self._unlink(block)
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Return a freed (not released) block to the normal segment MRU (tail)."""
        last_block = self.fake_free_list_tail.prev_free_block
        if last_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        self._link_after(last_block, block)
        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        if last_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block
        self.num_free_blocks += len(blocks)

    def aging_block(self, block: KVCacheBlock) -> int:
        """Move an in-queue block to the release MRU (just before the normal segment).

        Returns 1 if the block is (now) in the reusable region, 0 if it could not be
        aged (still referenced or not currently in the queue); the caller then defers
        aging until the block is freed.
        """
        if block.ref_cnt != 0:
            return 0
        if block.prev_free_block is None or block.next_free_block is None:
            return 0
        if block is self.released_tail:
            # Already the release MRU; nothing to move.
            return 1

        self._unlink(block)
        # Insert right after the current reusable MRU so release sits before normal.
        self._link_after(self.released_tail, block)
        self.released_tail = block
        return 1

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret

    def lru_head_block_id(self) -> int | None:
        head = self.fake_free_list_head.next_free_block
        if head is None or head is self.fake_free_list_tail:
            return None
        return head.block_id

    def queue_block_ids(self, limit: int = 12) -> list[int]:
        ids: list[int] = []
        curr = self.fake_free_list_head.next_free_block
        while curr is not None and curr is not self.fake_free_list_tail:
            ids.append(curr.block_id)
            if len(ids) >= limit:
                break
            curr = curr.next_free_block
        return ids
