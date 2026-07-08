# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2026 Huawei Technologies Co., Ltd.
# 本文件基于vLLM v0.21.0的FreeKVCacheBlockQueue实现修改而来，用于实现双区缓存特性

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import KVCacheBlock


@dataclass
class _JiuwenBlockState:
    allocated: bool = False
    reclaimed: bool = False


_block_states: dict[int, _JiuwenBlockState] = {}


def _state(block: KVCacheBlock) -> _JiuwenBlockState:
    st = _block_states.get(block.block_id)
    if st is None:
        st = _JiuwenBlockState()
        _block_states[block.block_id] = st
    return st


def init_block_states(blocks: list[KVCacheBlock]) -> None:
    for block in blocks:
        if block.block_id >= 0 and block.block_id not in _block_states:
            _block_states[block.block_id] = _JiuwenBlockState()


def jiuwen_is_reclaimed(block: KVCacheBlock) -> bool:
    return _state(block).reclaimed


def jiuwen_clear_reclaimed(block: KVCacheBlock) -> None:
    _state(block).reclaimed = False


def jiuwen_mark_reclaimed_for_test(block: KVCacheBlock, reclaimed: bool = True) -> None:
    _state(block).reclaimed = reclaimed


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

    - free:    never popped from this queue (block has no ``allocated`` mark)
    - release: popped before, then moved here by ``aging_block`` (release_kv_cache)
    - normal:  popped before, then returned via ``append`` (plain free)

    We do not track the free/release boundary at all: both are drained from the head
    in order, so ``released_tail`` (the reusable|normal divider) is the only structural
    pointer. Whether the head block is from the never-used *free* segment or from a
    previously used *release/normal* segment is tracked in the sidecar state map.

    Per-block allocation state (set in ``_take_head`` / ``remove``) is stored in a
    sidecar map (``_block_states``), not on ``KVCacheBlock`` instances, because v0.18
    uses ``@dataclass(slots=True)``:

    - ``allocated`` (persistent): ``True`` once the block has ever been popped from
      this queue or removed via ``remove`` (prefix-cache touch). Never cleared. At the
      next ``popleft``, the pre-pop value distinguishes a virgin free block (``False``)
      from a reused release/normal block (``True``).
    - ``reclaimed`` (transient): set on each ``popleft`` to the pre-pop value of
      ``allocated``. Consumed by ``bind_allocated_blocks`` to choose ``reset_blocks``
      (reused; clear stale ``block_to_sessions``) vs ``add_blocks`` (fresh). Also read
      by the ``4-evict`` debug log before bind clears it.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        init_block_states(blocks)
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

        # Snapshot whether this pop reuses a release/normal block (allocated was
        # True) vs a never-used free block (False). Persist allocated=True and pass
        # the snapshot to bind_allocated_blocks via reclaimed.
        meta = _state(head)
        reclaimed = meta.allocated
        meta.allocated = True

        # Popping the last reusable block empties the reusable region.
        if head is self.released_tail:
            self.released_tail = self.fake_free_list_head

        self._unlink(head)
        head.prev_free_block = head.next_free_block = None
        meta.reclaimed = reclaimed
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

        # Prefix-cache touch: block leaves the queue without reclaimed, but must be
        # treated as previously allocated on any later popleft.
        _state(block).allocated = True
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
