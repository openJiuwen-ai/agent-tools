from vllm.v1.core.kv_cache_utils import KVCacheBlock


class TwoPhaseBlockQueue:
    """Doubly-linked free block queue with a released (aged) zone and a normal zone.

    Layout (LRU -> MRU within each zone):

        fake_head -> [released LRU ... released MRU]
                  -> [normal LRU ... normal MRU] -> fake_tail

    Eviction (popleft) always takes from the list head:
    1. Released blocks are evicted before any normal block.
    2. Within each zone, standard LRU applies.
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
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

        # MRU end of the released zone; fake_head when the zone is empty.
        self.released_tail = self.fake_free_list_head

    def _first_normal_block(self) -> KVCacheBlock | None:
        if self.released_tail is self.fake_free_list_head:
            return self.fake_free_list_head.next_free_block
        nxt = self.released_tail.next_free_block
        return None if nxt is self.fake_free_list_tail else nxt

    def _is_in_released_zone(self, block: KVCacheBlock) -> bool:
        if self.released_tail is self.fake_free_list_head:
            return False
        curr = self.fake_free_list_head.next_free_block
        while curr is not None and curr is not self.fake_free_list_tail:
            if curr is block:
                return True
            if curr is self.released_tail:
                break
            curr = curr.next_free_block
        return False

    @staticmethod
    def _unlink(block: KVCacheBlock) -> None:
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"unlink() called on an invalid block: {block}")
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block

    def _insert_released_mru(self, block: KVCacheBlock) -> None:
        after = self.released_tail
        before = after.next_free_block
        block.prev_free_block = after
        block.next_free_block = before
        after.next_free_block = block
        before.prev_free_block = block
        self.released_tail = block

    def _on_released_block_removed(self, block: KVCacheBlock) -> None:
        if block == self.released_tail:
            self.released_tail = block.prev_free_block

    def _repair_released_tail(self) -> None:
        rt = self.released_tail
        if rt is self.fake_free_list_head:
            return
        if rt.prev_free_block is None or rt.next_free_block is None:
            self.released_tail = self.fake_free_list_head

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

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        if first_block.next_free_block is None:
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        self._on_released_block_removed(first_block)
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head
        first_block.prev_free_block = first_block.next_free_block = None
        self._repair_released_tail()

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n == 0:
            return []
        if self.num_free_blocks < n:
            raise ValueError(
                f"Not enough free blocks: requested {n}, "
                f"available {self.num_free_blocks}"
            )
        self.num_free_blocks -= n

        ret: list[KVCacheBlock] = []
        curr_block = self.fake_free_list_head.next_free_block
        for _ in range(n):
            if curr_block is None:
                raise RuntimeError(
                    "Unexpected None block while popping from free list"
                )
            ret.append(curr_block)
            self._on_released_block_removed(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        self._repair_released_tail()
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        self._on_released_block_removed(block)
        self._unlink(block)
        block.prev_free_block = block.next_free_block = None
        self._repair_released_tail()
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block
        last_block.next_free_block = block
        block.prev_free_block = last_block
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block
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

    def aging_block(self, block: KVCacheBlock) -> int:
        """Move a free block into the released zone (MRU side); LRU evicted first."""
        if block.ref_cnt != 0:
            return 0
        if self.num_free_blocks <= 1:
            return 1
        if block.prev_free_block is None or block.next_free_block is None:
            return 0
        if self._is_in_released_zone(block) and block is self.released_tail:
            return 1

        self._on_released_block_removed(block)
        self._unlink(block)
        self._insert_released_mru(block)
        return 1
