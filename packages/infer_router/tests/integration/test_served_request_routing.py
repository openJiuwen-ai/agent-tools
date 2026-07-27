"""Regression test for the served-request / routed-request mismatch.

The router must select workers for the SAME request the handler forwards. The
scheduler's ``get_next_request()`` can return a *different* queued request than
the one a handler is serving (WSPT/LCFS, or FCFS under concurrent coroutine
interleaving), so routing must be driven by the handler's own ``route_hint`` --
not by whatever request the scheduler happens to pop -- or KV-cache prefix
affinity is silently lost.

This test forces the mismatch deterministically: it selects the WSPT strategy
and pre-seeds the queue with a competitor request engineered to have a strictly
higher WSPT score than the request under test, so ``get_next_request()`` provably
returns the competitor. It then spies on the argument passed to
``router.route_to_workers`` and asserts it is the *served* request's hint.

    Before the fix: routing is called with the popped competitor's hint  -> FAIL
    After the fix:  routing is called with the served request's own hint  -> PASS

The spy asserts on the routing *argument*, not on a concrete worker id, so it is
independent of the Thompson-sampling randomness in the load-balancing algorithm.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from openjiuwentools.infer_router.api import server
from openjiuwentools.infer_router.api.server import get_app
from openjiuwentools.infer_router.schemas.agent_hints import (
    RouteHint,
    WorkerInfo,
    WorkerType,
)

SERVED_ID = "served-A"
COMPETITOR_ID = "competitor-B"


@pytest.fixture
def client():
    return TestClient(get_app())


async def _mock_forward_request(*args, **kwargs):
    return {"id": SERVED_ID, "choices": [{"message": {"content": "ok"}}]}


def _make_worker(worker_id):
    return WorkerInfo(
        worker_id=worker_id,
        model="test-model",
        url="http://localhost:8000/v1",
        current_load=10,
        cached_prefixes=[],
        worker_type=WorkerType.COMBINED,
        group="test-group",
    )


def _seed_higher_priority_competitor():
    """Put a competitor in the real scheduler queue with a strictly higher WSPT
    score than the request under test.

    WSPT score = (1 + priority) / tokens_to_process. The competitor has a large
    priority and few tokens to process; the served request (priority 0, ~13
    uncached tokens) scores far lower, so WSPT pops the competitor.
    """
    server.scheduler.global_queue.clear()
    server.scheduler.set_strategy("WSPT")
    competitor_hint = RouteHint(
        priority=1000,
        estimated_output_tokens=1,
        next_turn_prefill=False,
        request_id=COMPETITOR_ID,
        model="test-model",
        token_ids=[1, 2],
    )
    server.scheduler.submit(competitor_hint, overlap_scores=None)


def test_chat_routes_for_served_request(client):
    """The chat handler must route for its own request, not the popped one."""
    routed_ids: list[str] = []

    served_hint = RouteHint(
        priority=0,
        estimated_output_tokens=128,
        next_turn_prefill=False,
        request_id=SERVED_ID,
        model="test-model",
        token_ids=[60, 124, 117, 115, 101, 114, 124, 62, 72, 101, 108, 108, 111],
    )

    def _spy_route_to_workers(route_hint, *args, **kwargs):
        routed_ids.append(route_hint.request_id)
        return ("worker-1", None)

    with patch("openjiuwentools.infer_router.api.server.settings") as mock_settings, \
        patch("openjiuwentools.infer_router.api.server.preprocessor") as mock_preprocessor, \
        patch("openjiuwentools.infer_router.api.server.worker_manager") as mock_manager, \
        patch("openjiuwentools.infer_router.api.server.router") as mock_router:
        mock_settings.enable_auth = False
        mock_preprocessor.process.return_value = served_hint
        mock_manager.get_worker.side_effect = _make_worker
        mock_manager.forward_request = _mock_forward_request
        mock_router.route_to_workers.side_effect = _spy_route_to_workers

        _seed_higher_priority_competitor()

        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert routed_ids, "route_to_workers was never called"
    # The invariant: routing is driven by the served request's hint, even though
    # the scheduler popped the higher-WSPT competitor.
    assert routed_ids[-1] == SERVED_ID, (
        f"routed for {routed_ids[-1]!r} but served {SERVED_ID!r} -- "
        "worker selection is decoupled from the request actually forwarded"
    )


def test_completions_routes_for_served_request(client):
    """Same invariant on the /v1/completions path."""
    routed_ids: list[str] = []

    served_hint = RouteHint(
        priority=0,
        estimated_output_tokens=128,
        next_turn_prefill=False,
        request_id=SERVED_ID,
        model="test-model",
        token_ids=[60, 124, 117, 115, 101, 114, 124, 62, 72, 101, 108, 108, 111],
    )

    def _spy_route_to_workers(route_hint, *args, **kwargs):
        routed_ids.append(route_hint.request_id)
        return ("worker-1", None)

    with patch("openjiuwentools.infer_router.api.server.settings") as mock_settings, \
        patch("openjiuwentools.infer_router.api.server.preprocessor") as mock_preprocessor, \
        patch("openjiuwentools.infer_router.api.server.worker_manager") as mock_manager, \
        patch("openjiuwentools.infer_router.api.server.router") as mock_router:
        mock_settings.enable_auth = False
        mock_preprocessor.process_completion.return_value = served_hint
        mock_manager.get_worker.side_effect = _make_worker
        mock_manager.forward_request = _mock_forward_request
        mock_router.route_to_workers.side_effect = _spy_route_to_workers

        _seed_higher_priority_competitor()

        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hi"},
        )

    assert response.status_code == 200
    assert routed_ids, "route_to_workers was never called"
    assert routed_ids[-1] == SERVED_ID, (
        f"routed for {routed_ids[-1]!r} but served {SERVED_ID!r} -- "
        "worker selection is decoupled from the request actually forwarded"
    )
