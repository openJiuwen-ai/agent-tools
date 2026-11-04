"""Regression checks for diagnostic and compatibility interfaces."""

from copy import deepcopy
from types import SimpleNamespace

from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.preprocess.tokenizer import TokenizerManager
from openjiuwentools.infer_router.routing.scorer import (
    LatencyPredictScorer,
    PrefixCacheScorer,
    create_scorer,
    get_scorer_registry,
    register_scorer,
)
from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo
from openjiuwentools.infer_router.worker.worker_manager import WorkerManager
from jiuwen_kvbm.runtime.worker_runtime import WorkerKVBMRuntime


def test_cache_lock_decorator_preserves_method_identity():
    method = KVCacheManager.update_cache_state
    assert method.__name__ == "update_cache_state"
    assert method.__wrapped__.__name__ == "update_cache_state"


def test_explicit_scorer_registration_preserves_identity_and_factory(monkeypatch):
    registry = get_scorer_registry()
    assert registry["prefix-cache-scorer"] is PrefixCacheScorer
    assert isinstance(create_scorer("prefix-cache-scorer"), PrefixCacheScorer)

    class CustomScorer(PrefixCacheScorer):
        scorer_type = "test-explicit-registration"

    monkeypatch.setitem(registry, CustomScorer.scorer_type, PrefixCacheScorer)
    assert register_scorer(CustomScorer) is CustomScorer
    assert registry[CustomScorer.scorer_type] is CustomScorer
    assert isinstance(create_scorer(CustomScorer.scorer_type), CustomScorer)


def test_diagnostic_tokenizer_lookup_does_not_load_models():
    manager = TokenizerManager(load_from_file=False)
    assert manager.get_cached_tokenizer("missing-model") is None


def test_public_chat_formatting_keeps_tool_history_isolated():
    messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]}]
    expected = deepcopy(messages)
    formatted = TokenizerManager.format_chat_messages(messages)
    assert formatted == expected
    formatted[0]["tool_calls"][0]["id"] = "changed"
    assert messages == expected


def test_public_worker_control_url_retains_normalization():
    worker = WorkerInfo(worker_id="d0", model="test", kvbm_control_url="http://127.0.0.1:18010/v1/")
    assert WorkerManager.worker_control_url(worker) == "http://127.0.0.1:18010"


def test_prediction_batch_can_be_replaced_and_cleared():
    scorer = LatencyPredictScorer()
    scorer.set_predictions({"d0": {"ttft_ms": 12.0}})
    scorer.set_predictions(None)
    worker = WorkerInfo(worker_id="d0", model="test")
    assert scorer.score(worker, [worker], SimpleNamespace(estimated_output_tokens=1), None) == 0.5


def test_public_offload_entrypoints_delegate_without_changing_results():
    events = []

    def offload_ram(block):
        events.append(("ram", block))
        return "ram-event"

    def offload_disk(block):
        events.append(("disk", block))
        return "disk-event"

    # The compatibility bridge must use the existing lifecycle implementation.
    runtime = SimpleNamespace(_offload_to_ram=offload_ram, _offload_to_disk=offload_disk)
    block = object()
    assert WorkerKVBMRuntime.offload_to_ram(runtime, block) == "ram-event"
    assert WorkerKVBMRuntime.offload_to_disk(runtime, block) == "disk-event"
    assert events == [("ram", block), ("disk", block)]
