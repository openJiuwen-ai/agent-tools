from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

_orig_allocate_new_computed = KVCacheCoordinator.allocate_new_computed_blocks
_orig_allocate_new = KVCacheCoordinator.allocate_new_blocks


def _set_manager_session_ids(coordinator: KVCacheCoordinator, session_id) -> None:
    for m in coordinator.single_type_managers:
        m.set_jiuwen_sharing_session_id(session_id)


def _clear_manager_session_ids(coordinator: KVCacheCoordinator) -> None:
    for m in coordinator.single_type_managers:
        m.clear_jiuwen_sharing_session_id()


def allocate_new_computed_blocks_jiuwen(
    self,
    request_id: str,
    new_computed_blocks,
    num_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> None:
    sid = self.get_jiuwen_sharing_session_id()
    _set_manager_session_ids(self, sid)
    try:
        return _orig_allocate_new_computed(
            self,
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
            num_external_computed_tokens,
        )
    finally:
        _clear_manager_session_ids(self)


def allocate_new_blocks_jiuwen(
    self,
    request_id: str,
    num_tokens: int,
    num_tokens_main_model: int,
    num_encoder_tokens: int = 0,
):
    sid = self.get_jiuwen_sharing_session_id()
    _set_manager_session_ids(self, sid)
    try:
        return _orig_allocate_new(
            self, request_id, num_tokens, num_tokens_main_model, num_encoder_tokens
        )
    finally:
        _clear_manager_session_ids(self)


class KVCacheCoordinatorEx(KVCacheCoordinator):
    def set_jiuwen_sharing_session_id(self, session_id: str | None) -> None:
        self.jiuwen_sharing_session_id = session_id

    def get_jiuwen_sharing_session_id(self) -> str | None:
        return getattr(self, "jiuwen_sharing_session_id", None)

    def clear_jiuwen_sharing_session_id(self) -> None:
        self.jiuwen_sharing_session_id = None

    def aging_block(self, session_id, block_hashes) -> int:
        num = 0
        for manager in self.single_type_managers:
            num += manager.aging_block(session_id, block_hashes)
        return num


def register_kv_cache_coordinator():
    KVCacheCoordinator.set_jiuwen_sharing_session_id = (
        KVCacheCoordinatorEx.set_jiuwen_sharing_session_id
    )
    KVCacheCoordinator.get_jiuwen_sharing_session_id = (
        KVCacheCoordinatorEx.get_jiuwen_sharing_session_id
    )
    KVCacheCoordinator.clear_jiuwen_sharing_session_id = (
        KVCacheCoordinatorEx.clear_jiuwen_sharing_session_id
    )
    KVCacheCoordinator.aging_block = KVCacheCoordinatorEx.aging_block
    KVCacheCoordinator.allocate_new_computed_blocks = allocate_new_computed_blocks_jiuwen
    KVCacheCoordinator.allocate_new_blocks = allocate_new_blocks_jiuwen
