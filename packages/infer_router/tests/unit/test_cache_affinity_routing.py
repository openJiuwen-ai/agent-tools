"""Cache-overlap must decide worker selection.

Before this fix, KV-cache overlap only credited a worker's score when the request
was a same-group re-use (`is_same_group and reuse_after > 0`), and select_worker's
single-top-overlap branch picked ``max(all_workers, key=score)`` -- so a lone
cache-overlap winner could be discarded in favour of a higher-scoring but
cache-cold worker (the never-learning LinTS term dominates). These tests pin the
corrected behaviour: a worker holding the request's prefix is selected even when
it carries higher load / a lower LinTS score, and cache-cold ties still fall back
to load balancing.
"""

from unittest.mock import Mock

from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.routing.load_balance import WorkloadWeightedAlgorithm
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint, WorkerInfo, WorkerType

TOKENS = list(range(200))


def _algo(overlap: dict):
    km = Mock(spec=KVCacheManager)
    km.block_size = 16
    km.find_matches = Mock(return_value=overlap)
    return WorkloadWeightedAlgorithm(km)


def _worker(wid, load):
    return WorkerInfo(
        worker_id=wid, model="m", url=f"http://{wid}",
        current_load=load, worker_type=WorkerType.COMBINED, group=wid,
    )


def _hint():
    return RouteHint(
        request_id="r", model="m", priority=0, estimated_output_tokens=64,
        next_turn_prefill=False, token_ids=TOKENS, prefix_id=None,
    )


def test_overlap_worker_selected_despite_higher_load():
    """The worker holding the prefix wins even under much higher load."""
    algo = _algo({"hot": len(TOKENS)})  # 'hot' holds the whole prefix, 'cold' none
    workers = [_worker("cold", load=0.0), _worker("hot", load=90.0)]
    # Repeat to defeat any residual LinTS/exploration randomness.
    picks = {algo.select_worker(workers, _hint()).worker_id for _ in range(25)}
    assert picks == {"hot"}, f"cache-overlap worker not consistently selected: {picks}"


def test_partial_but_clear_overlap_still_wins():
    """A >10%-of-prompt overlap lead is decisive, not just a full hit."""
    algo = _algo({"hot": int(len(TOKENS) * 0.8)})  # 80% overlap vs 0
    workers = [_worker("cold", load=0.0), _worker("hot", load=50.0)]
    picks = {algo.select_worker(workers, _hint()).worker_id for _ in range(25)}
    assert picks == {"hot"}


def test_no_overlap_falls_back_to_load_balancing():
    """With no cache signal, selection stays valid (load-balanced), not crashing."""
    algo = _algo({})  # nobody holds the prefix
    workers = [_worker("a", load=10.0), _worker("b", load=10.0)]
    chosen = {algo.select_worker(workers, _hint()).worker_id for _ in range(25)}
    assert chosen <= {"a", "b"} and chosen, "should still return a valid worker"
