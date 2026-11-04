# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openjiuwentools.infer_router.data_plane.server import (
    _build_complete_fn,
    _build_disagg_result,
    _build_models_fn,
    _build_pd_decode_budget_fn,
    _build_pd_dispatch_snapshot_fn,
    _build_pd_prefill_complete_fn,
    _build_route_fn,
    _get_prefix_match_worker_ids,
    _normalize_max_tokens,
    _pd_engine_request_id,
    _preload_router_tokenizers,
    _resolve_pd_prefill_max_inflight,
    _should_use_pd_decode_direct,
)
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint, WorkerInfo


class _ImmediateScheduler:
    def __init__(self):
        self._scheduled = None

    def submit(self, route_hint, overlap_scores):
        route_hint.precomputed_overlap_scores = overlap_scores
        self._scheduled = SimpleNamespace(route_hint=route_hint)

    def get_next_request(self):
        scheduled = self._scheduled
        self._scheduled = None
        return scheduled

    def mark_task_completed(self):
        pass


def test_models_callback_returns_unique_registered_models():
    """Router 模型接口应汇总并稳定排序当前注册 worker 的模型。"""
    worker_manager = Mock()
    worker_manager.get_all_workers.return_value = {
        "worker-b": SimpleNamespace(model="model-b"),
        "worker-a-1": SimpleNamespace(model="model-a"),
        "worker-a-2": SimpleNamespace(model="model-a"),
        "worker-empty": SimpleNamespace(model=""),
    }

    result = json.loads(_build_models_fn(worker_manager)())

    assert result == {
        "object": "list",
        "data": [
            {"id": "model-a", "object": "model", "owned_by": "jiuwen"},
            {"id": "model-b", "object": "model", "owned_by": "jiuwen"},
        ],
    }


def test_models_callback_returns_empty_list_without_workers():
    """没有注册 worker 时仍应返回合法的 OpenAI 模型列表。"""
    worker_manager = Mock()
    worker_manager.get_all_workers.return_value = {}

    assert json.loads(_build_models_fn(worker_manager)()) == {
        "object": "list",
        "data": [],
    }


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({}, 512),
        ({"max_tokens": None}, 512),
        ({"max_tokens": 64}, 64),
    ],
)
def test_normalize_max_tokens_handles_missing_null_and_explicit_values(body, expected):
    """缺失和 null 应使用默认值，客户端显式设置的上限应保留。"""
    _normalize_max_tokens(body, 512)

    assert body["max_tokens"] == expected


@pytest.mark.parametrize("body", [{}, {"max_tokens": None}])
def test_normalize_max_tokens_is_transparent_without_configured_default(body):
    """默认配置不应限制未设置输出上限的 Agent 请求。"""
    original = dict(body)

    _normalize_max_tokens(body, None)

    assert body == original


def test_normalize_max_tokens_respects_max_completion_tokens():
    """Chat Completions 新字段存在时不应再注入旧 max_tokens。"""
    body = {"max_completion_tokens": 2048}

    _normalize_max_tokens(body, 512, prefer_max_completion_tokens=True)

    assert body == {"max_completion_tokens": 2048}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "request_max_tokens",
    [pytest.param(None, id="null"), pytest.param("missing", id="missing")],
)
def test_completion_route_forwards_default_max_tokens(
    monkeypatch,
    stream,
    request_max_tokens,
):
    """流式和非流式 completion 都应把缺失或 null 归一化为默认上限。"""
    cfg = SimpleNamespace(
        router_queue_timeout=1.0,
        router_queue_poll_interval=0.01,
        default_max_tokens=512,
        default_ignore_eos=False,
        kvbm_enabled=False,
        kvbm_router_pd_enabled=False,
    )
    monkeypatch.setattr(
        "openjiuwentools.infer_router.config.config.get_global_settings",
        lambda: cfg,
    )
    route_hint = RouteHint(
        priority=0,
        estimated_output_tokens=128,
        next_turn_prefill=False,
        request_id="default-max-tokens-request",
        model="test-model",
        token_ids=[],
    )
    preprocessor = Mock()
    preprocessor.process_completion.return_value = route_hint
    router = Mock()
    router.route_to_workers.return_value = ("combined-0", None)
    worker_manager = Mock()
    worker_manager.get_worker.return_value = WorkerInfo(
        worker_id="combined-0",
        model="test-model",
        url="http://worker:8001",
    )
    route_fn = _build_route_fn(
        preprocessor=preprocessor,
        kvcache_manager=Mock(),
        scheduler=_ImmediateScheduler(),
        router=router,
        worker_manager=worker_manager,
    )
    body = {"model": "test-model", "prompt": "hello", "stream": stream}
    if request_max_tokens != "missing":
        body["max_tokens"] = request_max_tokens

    result = route_fn(json.dumps(body), "/v1/completions")

    assert json.loads(result["body_json"])["max_tokens"] == 512
    forwarded_request = preprocessor.process_completion.call_args.args[0]
    assert forwarded_request.max_tokens == 512


def test_chat_route_preserves_jiuwenswarm_tool_request_without_output_limit(
    monkeypatch,
):
    """JiuwenSwarm 的工具和流选项应完整透传，且不凭空增加生成上限。"""
    cfg = SimpleNamespace(
        router_queue_timeout=1.0,
        router_queue_poll_interval=0.01,
        default_max_tokens=None,
        default_ignore_eos=False,
        kvbm_enabled=False,
        kvbm_router_pd_enabled=False,
    )
    monkeypatch.setattr(
        "openjiuwentools.infer_router.config.config.get_global_settings",
        lambda: cfg,
    )
    route_hint = RouteHint(
        priority=0,
        estimated_output_tokens=128,
        next_turn_prefill=False,
        request_id="agent-tool-request",
        model="test-model",
        token_ids=[],
    )
    preprocessor = Mock()
    preprocessor.process.return_value = route_hint
    router = Mock()
    router.route_to_workers.return_value = ("combined-0", None)
    worker_manager = Mock()
    worker_manager.get_worker.return_value = WorkerInfo(
        worker_id="combined-0",
        model="test-model",
        url="http://worker:8001",
    )
    route_fn = _build_route_fn(
        preprocessor=preprocessor,
        kvcache_manager=Mock(),
        scheduler=_ImmediateScheduler(),
        router=router,
        worker_manager=worker_manager,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Look up weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "北京天气如何？"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "frequency_penalty": 0.2,
    }

    result = route_fn(json.dumps(body), "/v1/chat/completions")
    forwarded = json.loads(result["body_json"])

    assert forwarded == body
    parsed_request = preprocessor.process.call_args.args[0]
    assert parsed_request.tools == tools
    assert parsed_request.tool_choice == "auto"
    assert parsed_request.model_dump(exclude_none=True)["frequency_penalty"] == 0.2


@pytest.mark.parametrize(
    ("generic_limit", "legacy_limit", "expected"),
    [
        (12, 8, 12),
        (None, 6, 6),
        (0, 8, 0),
    ],
)
def test_pd_prefill_admission_prefers_generic_config_and_supports_legacy(
    generic_limit,
    legacy_limit,
    expected,
):
    """通用PD准入配置应优先，并兼容旧KVBM配置。"""
    cfg = SimpleNamespace(
        router_pd_prefill_max_inflight=generic_limit,
        kvbm_router_pd_prefill_max_inflight=legacy_limit,
    )

    assert _resolve_pd_prefill_max_inflight(cfg) == expected


def test_pd_prefill_complete_callback_releases_only_prefill_worker():
    """P阶段回调应提前释放P预算，并保留D供最终完成回调释放。"""
    router = Mock()
    request_workers = {"request-1": ["prefill-0", "decode-0"]}
    callback = _build_pd_prefill_complete_fn(router, request_workers)

    callback("request-1", "prefill-0")

    assert request_workers["request-1"] == ["decode-0"]
    router.release_pd_prefill.assert_called_once_with("prefill-0", "request-1")


def test_pd_dispatch_snapshot_callback_compares_selection_and_dispatch(monkeypatch):
    router = Mock()
    router.inflight.get.return_value = 7
    worker_manager = Mock()
    worker_manager.get_worker.return_value = SimpleNamespace(
        running_requests=5,
        queue_depth=3,
    )
    request_contexts = {
        "request-1": {
            "decode_selection_snapshot": {
                "selected_at_monotonic": 10.0,
                "inflight": 2,
                "running": 1,
                "waiting": 0,
            }
        }
    }
    monkeypatch.setattr(
        "openjiuwentools.infer_router.data_plane.server.time.monotonic",
        lambda: 10.125,
    )
    callback = _build_pd_dispatch_snapshot_fn(
        router,
        worker_manager,
        request_contexts,
    )

    snapshot = json.loads(callback("request-1", "decode-0"))

    assert snapshot == {
        "selection_to_dispatch_ms": 125.0,
        "selection_inflight": 2,
        "selection_running": 1,
        "selection_waiting": 0,
        "dispatch_inflight": 7,
        "dispatch_running": 5,
        "dispatch_waiting": 3,
    }


def test_pd_decode_budget_callback_reconciles_attached_request():
    """D attach回调应收缩预算并立即发布完整NPU前缀。"""
    router = Mock()
    request_workers = {"request-1": ["decode-0"]}
    route_hint = Mock(token_ids=list(range(284)))
    request_contexts = {"request-1": {"route_hint": route_hint}}
    cache_manager = Mock()
    cache_manager.record_pd_import.return_value = 2
    callback = _build_pd_decode_budget_fn(
        router,
        request_workers,
        request_contexts,
        cache_manager,
    )

    callback(
        "request-1",
        "decode-0",
        "attached",
        json.dumps(
            {
                "source_worker_id": "prefill-0",
                "block_hashes": ["hash-0", "hash-1"],
                "destination_block_ids": [70, 71],
                "partial_valid_tokens": 28,
            }
        ),
    )

    assert request_workers["request-1"] == ["decode-0"]
    router.reconcile_pd_decode_budget.assert_called_once_with("decode-0", "request-1")
    cache_manager.record_pd_import.assert_called_once_with(
        "decode-0",
        ["hash-0", "hash-1"],
        [70, 71],
        list(range(284)),
        block_size=128,
    )


def test_pd_decode_budget_callback_marks_scheduler_allocation():
    """D目标block分配事件应提前启动Router容量物化确认。"""
    router = Mock()
    request_workers = {"request-1": ["decode-0"]}
    callback = _build_pd_decode_budget_fn(
        router,
        request_workers,
        {},
        Mock(),
    )

    callback(
        "request-1",
        "decode-0",
        "allocated",
        json.dumps({"allocated_blocks": 50}),
    )

    router.mark_pd_decode_blocks_allocated.assert_called_once_with("decode-0", "request-1")
    router.reconcile_pd_decode_budget.assert_not_called()


def test_pd_decode_budget_callback_forwards_capacity_generation():
    """D allocation和attach回调应把EngineCore容量代际传给Router。"""
    router = Mock()
    request_workers = {"request-1": ["decode-0"]}
    callback = _build_pd_decode_budget_fn(router, request_workers, {}, Mock())

    callback(
        "request-1",
        "decode-0",
        "allocated",
        json.dumps({"capacity_generation": 101}),
    )
    callback(
        "request-1",
        "decode-0",
        "attached",
        json.dumps({"capacity_generation": 102}),
    )

    router.mark_pd_decode_blocks_allocated.assert_called_once_with("decode-0", "request-1", 101)
    router.reconcile_pd_decode_budget.assert_called_once_with("decode-0", "request-1", 102)


def test_pd_decode_budget_callback_restores_fallback_reservation():
    """PD降级回调应恢复完整预算且不触发attach容量确认。"""
    router = Mock()
    request_workers = {"request-1": ["decode-0"]}
    callback = _build_pd_decode_budget_fn(
        router,
        request_workers,
        {},
        Mock(),
    )

    callback(
        "request-1",
        "decode-0",
        "fallback",
        json.dumps({"reason": "pd_import_failed"}),
    )

    router.restore_pd_decode_budget_after_fallback.assert_called_once_with("decode-0", "request-1")
    router.reconcile_pd_decode_budget.assert_not_called()


def test_pd_decode_budget_callback_does_not_treat_first_byte_as_attach():
    """fallback后的first byte不能再次把完整D预算收缩为attach预算。"""
    router = Mock()
    callback = _build_pd_decode_budget_fn(
        router,
        {"request-1": ["decode-0"]},
        {},
        Mock(),
    )

    callback("request-1", "decode-0", "fallback", "{}")
    callback("request-1", "decode-0", "first_byte", "{}")

    router.restore_pd_decode_budget_after_fallback.assert_called_once_with("decode-0", "request-1")
    router.reconcile_pd_decode_budget.assert_not_called()


@pytest.mark.parametrize(
    ("include_decode_workers", "expected"),
    [
        (True, {"prefill-0", "decode-0"}),
        (False, {"prefill-0"}),
    ],
)
def test_prefix_match_worker_scope_can_exclude_decode_workers(
    include_decode_workers,
    expected,
):
    """配置开关应允许 Mooncake 路径恢复 P-only 前缀匹配。"""
    prefill = WorkerInfo(
        worker_id="prefill-0",
        model="test-model",
        url="http://prefill:8011",
    )
    decode = WorkerInfo(
        worker_id="decode-0",
        model="test-model",
        url="http://decode:8020",
    )
    worker_manager = Mock()
    worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    worker_manager.get_all_healthy_decode_workers.return_value = [decode]

    worker_ids = _get_prefix_match_worker_ids(
        worker_manager=worker_manager,
        model="test-model",
        include_decode_workers=include_decode_workers,
    )

    assert worker_ids == expected
    if include_decode_workers:
        worker_manager.get_all_healthy_decode_workers.assert_called_once_with("test-model")
    else:
        worker_manager.get_all_healthy_decode_workers.assert_not_called()


@pytest.mark.asyncio
async def test_router_preloads_discovered_worker_models_once():
    """Router 监听前应从 Worker 发现结果提取并预加载去重后的模型。"""

    preprocessor = Mock()
    preprocessor.preload_tokenizers.return_value = {
        "model-a": True,
        "model-b": True,
    }
    worker_manager = Mock()
    worker_manager.get_all_workers.return_value = {
        "worker-a": SimpleNamespace(model="model-a"),
        "worker-b": SimpleNamespace(model="model-b"),
        "worker-c": SimpleNamespace(model="model-a"),
    }

    result = await _preload_router_tokenizers(
        preprocessor=preprocessor,
        worker_manager=worker_manager,
        enabled=True,
    )

    assert result == {"model-a": True, "model-b": True}
    preprocessor.preload_tokenizers.assert_called_once_with(["model-a", "model-b"])


@pytest.mark.parametrize("arity", [4, 6, 11])
def test_complete_callback_short_and_full_signatures_release_all_workers(arity):
    router = Mock()
    workers = {"req": ["p", "d"]}
    contexts = {"req": {}}
    complete = _build_complete_fn(router, Mock(), workers, contexts)
    args = ["req", "d", 0.1, 1.0, 4, 10, 0.1, 0.0, 0.0, 0.0, "{}"]
    complete(*args[:arity])
    assert workers == {}
    assert contexts == {}
    assert router.release_worker.call_count == 2
    router.release_worker.assert_any_call("p", request_id="req")
    router.release_worker.assert_any_call("d", request_id="req")


def test_complete_callback_rejects_invalid_arguments_before_releasing_workers():
    router = Mock()
    workers = {"req": ["p", "d"]}
    contexts = {"req": {}}
    complete = _build_complete_fn(router, Mock(), workers, contexts)
    with pytest.raises(TypeError):
        complete("req", "d", 0.1, 1.0, unexpected=1)
    assert workers == {"req": ["p", "d"]}
    assert contexts == {"req": {}}
    router.release_worker.assert_not_called()


def test_complete_callback_supports_keyword_arguments():
    router = Mock()
    workers = {"req": ["p", "d"]}
    contexts = {"req": {}}
    complete = _build_complete_fn(router, Mock(), workers, contexts)
    complete("req", "d", ttft_s=0.1, total_s=1.0, completion_tokens=4)
    assert not workers
    assert not contexts
    router.record_decode_tpot.assert_called_once_with("d", 300.0)


def test_complete_fn_logs_one_line_pd_request_summary(monkeypatch):
    """请求完成时应把 Router、Worker 和端到端耗时合并为一条固定字段日志。"""

    from openjiuwentools.infer_router.data_plane import server

    router = Mock()
    worker_manager = Mock()
    info = Mock()
    monkeypatch.setattr(server.logger, "info", info)
    request_workers = {"request-1": ["prefill-0", "decode-0"]}
    request_contexts = {
        "request-1": {
            "request_index": 7,
            "route_hint": SimpleNamespace(),
            "overlap_scores": {},
            "decode_id": "decode-0",
            "prefill_id": "prefill-0",
            "prompt_tokens": 6300,
            "router_phase_ms": {},
        }
    }
    complete = _build_complete_fn(
        router,
        worker_manager,
        request_workers,
        request_contexts,
    )

    complete(
        "request-1",
        "decode-0",
        0.4,
        20.0,
        512,
        6300,
        0.32,
        0.05,
        0.18,
        0.07,
        json.dumps(
            {
                "free_blocks": 700,
                "p_request_ms": 83.0,
                "p_http_dispatch_ms": 0.2,
                "p_prepare_to_http_dispatch_ms": 1.2,
                "p_dispatch_to_scheduler_seen_ms": 29.0,
                "p_worker_request_received_to_scheduler_seen_ms": 4.0,
                "p_worker_request_received_timing_source": "arrival_time:monotonic",
                "p_scheduler_queue_cycles": 2,
                "p_scheduler_seen_cycle_id": 101,
                "p_first_scheduled_cycle_id": 104,
                "p_kv_ready_cycle_id": 105,
                "p_scheduler_compute_cycles": 2,
                "p_first_scheduled_token_count": 28,
                "p_scheduler_running_request_count": 3,
                "p_scheduler_waiting_request_count": 1,
                "p_first_scheduled_running_request_count": 4,
                "p_first_scheduled_waiting_request_count": 0,
                "p_first_scheduled_cycle_ms": 31.0,
                "p_scheduler_seen_to_kv_ready_ms": 45.0,
                "p_kv_ready_to_export_ready_ms": 2.5,
                "p_export_ready_to_http_response_ms": 3.0,
                "p_response_network_ms": 1.0,
                "p_prefill_admission_wait_ms": 12.5,
                "delta_plan_ms": 20.0,
                "delta_preplanned": True,
                "delta_plan_task_queue_ms": 0.0,
                "delta_hash_validate_ms": 0.4,
                "delta_reserve_ms": 1.5,
                "delta_plan_execution_ms": 0.6,
                "delta_preplan_age_ms": 80.0,
                "delta_inflight_wait_ms": 42.0,
                "delta_inflight_retry_count": 1,
                "delta_capacity_wait_ms": 17.0,
                "delta_capacity_retry_count": 2,
                "import_executor_queue_ms": 3.0,
                "reserve_ms": 1.5,
                "hixl_pull_ms": 39.0,
                "attach_task_queue_ms": 2.0,
                "attach_execution_ms": 8.0,
                "transferred_blocks": 1,
                "reused_blocks": 49,
            }
        ),
    )

    summary_call = next(call for call in info.call_args_list if "[KVBMPDRequestSummary]" in call.args[0])
    assert summary_call.args[1] == 7
    assert summary_call.args[4] == 700
    assert summary_call.args[6] == 0.2
    assert summary_call.args[7] == 1.2
    assert summary_call.args[9] == 4.0
    assert summary_call.args[10] == "arrival_time:monotonic"
    assert summary_call.args[11] == 2
    assert summary_call.args[12] == 101
    assert summary_call.args[13] == 104
    assert summary_call.args[14] == 105
    assert summary_call.args[26] == 12.5
    assert summary_call.args[28] is True
    assert summary_call.args[29] == 0.0
    assert summary_call.args[30] == 0.4
    assert summary_call.args[34] == 42.0
    assert summary_call.args[35] == 1
    assert summary_call.args[36] == 17.0
    assert summary_call.args[37] == 2
    assert summary_call.args[38] == 3.0
    assert summary_call.args[47] == 512
    assert summary_call.args[48] == pytest.approx((20.0 - 0.4) * 1000 / 511)


def test_pd_engine_request_id_matches_worker_cache_sharing_encoding():
    """Router handoff ID 应与 Worker 传入 engine-core 的 ID 完全一致。"""

    forwarded = "router-pd-request"

    assert _pd_engine_request_id(forwarded, {}) == forwarded
    assert json.loads(
        _pd_engine_request_id(
            forwarded,
            {"cache_sharing": True, "cache_salt": "session-a"},
        )
    ) == {
        "request_id": forwarded,
        "sharing_cache_salt": "session-a",
    }


def test_disagg_result_contains_kvbm_pd_orchestration_metadata():
    """PD 路由决策应携带 Rust 数据面完成 handoff 所需的控制元数据。"""

    prefill = WorkerInfo(
        worker_id="prefill-0",
        model="test-model",
        url="http://prefill:8011",
        kvbm_control_url="http://prefill:18011",
        kv_addr="10.0.0.1:19011",
    )
    decode = WorkerInfo(
        worker_id="decode-0",
        model="test-model",
        url="http://decode:8020",
        kvbm_control_url="http://decode:18020",
        kv_addr="10.0.0.2:19020",
    )
    worker_manager = Mock()
    worker_manager.get_worker.side_effect = {
        "prefill-0": prefill,
        "decode-0": decode,
    }.get

    def worker_control_url(worker):
        return worker.kvbm_control_url

    worker_manager.worker_control_url.side_effect = worker_control_url
    route_hint = RouteHint(
        priority=0,
        estimated_output_tokens=64,
        next_turn_prefill=False,
        request_id="router-request-1",
        model="test-model",
        token_ids=list(range(256)),
    )
    cfg = SimpleNamespace(
        kvbm_enabled=True,
        kvbm_router_pd_enabled=True,
        kvbm_router_pd_timeout_ms=7000,
        kvbm_router_pd_export_timeout_ms=90000,
    )

    result = _build_disagg_result(
        "prefill-0",
        "decode-0",
        {"model": "test-model", "prompt": "hello", "max_tokens": 8},
        route_hint,
        False,
        worker_manager,
        cfg,
    )

    assert result["pd_kvbm_enabled"] is True
    assert result["pd_decode_direct"] is False
    assert result["pd_handoff_id"] == "pd-router-request-1"
    assert result["pd_decode_admission_ticket_id"] == ""
    assert result["pd_request_id"] == result["headers"]["X-Request-Id"]
    assert result["headers"]["X-Jiuwen-Request-Id"] == "router-request-1"
    assert result["pd_prefill_control_url"] == "http://prefill:18011"
    assert result["pd_decode_control_url"] == "http://decode:18020"
    assert result["pd_prompt_token_count"] == 256
    assert result["pd_timeout_ms"] == 7000
    assert result["pd_export_timeout_ms"] == 90000


def test_native_disagg_result_enables_connector_transfer_on_prefill():
    """Native Connector PD must ask P to return metadata for the later D request."""

    prefill = WorkerInfo(
        worker_id="prefill-0",
        model="test-model",
        url="http://prefill:8011",
        kv_addr="10.0.0.1:19011",
    )
    decode = WorkerInfo(
        worker_id="decode-0",
        model="test-model",
        url="http://decode:8020",
        kv_addr="10.0.0.2:19020",
    )
    worker_manager = Mock()
    worker_manager.get_worker.side_effect = {
        "prefill-0": prefill,
        "decode-0": decode,
    }.get
    worker_manager._worker_control_url.return_value = ""
    route_hint = RouteHint(
        priority=0,
        estimated_output_tokens=64,
        next_turn_prefill=False,
        request_id="router-request-native",
        model="test-model",
        token_ids=list(range(256)),
    )
    cfg = SimpleNamespace(
        kvbm_enabled=False,
        kvbm_router_pd_enabled=False,
        kvbm_router_pd_timeout_ms=7000,
        kvbm_router_pd_export_timeout_ms=90000,
    )

    result = _build_disagg_result(
        "prefill-0",
        "decode-0",
        {"model": "test-model", "prompt": "hello", "max_tokens": 8},
        route_hint,
        False,
        worker_manager,
        cfg,
    )

    prefill_body = json.loads(result["body_json"])
    decode_body = json.loads(result["decode_body_json"])
    assert prefill_body["max_tokens"] == 1
    assert prefill_body["min_tokens"] == 1
    assert prefill_body["kv_transfer_params"] == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    assert "kv_transfer_params" not in decode_body


def test_kvbm_disagg_result_does_not_add_native_connector_metadata():
    """HIXL must continue to use its own control path instead of native metadata."""

    prefill = WorkerInfo(
        worker_id="prefill-0",
        model="test-model",
        url="http://prefill:8011",
        kvbm_control_url="http://prefill:18011",
    )
    decode = WorkerInfo(
        worker_id="decode-0",
        model="test-model",
        url="http://decode:8020",
        kvbm_control_url="http://decode:18020",
    )
    worker_manager = Mock()
    worker_manager.get_worker.side_effect = {
        "prefill-0": prefill,
        "decode-0": decode,
    }.get

    def worker_control_url(worker):
        return worker.kvbm_control_url

    worker_manager.worker_control_url.side_effect = worker_control_url
    result = _build_disagg_result(
        "prefill-0",
        "decode-0",
        {"model": "test-model", "prompt": "hello", "max_tokens": 8},
        RouteHint(
            priority=0,
            estimated_output_tokens=64,
            next_turn_prefill=False,
            request_id="router-request-hixl",
            model="test-model",
            token_ids=list(range(256)),
        ),
        False,
        worker_manager,
        SimpleNamespace(
            kvbm_enabled=True,
            kvbm_router_pd_enabled=True,
            kvbm_router_pd_timeout_ms=7000,
            kvbm_router_pd_export_timeout_ms=90000,
        ),
    )

    assert "kv_transfer_params" not in json.loads(result["body_json"])


def test_pd_decode_direct_requires_continuous_npu_prefix_over_thresholds():
    """D 直达应同时满足连续 NPU token 数与 prompt 命中比例阈值。"""

    class PrefixBlock:
        def __init__(self, token_count, on_npu=True):
            self.alias = SimpleNamespace(token_count=token_count) if on_npu else None

        def select(self, _locations):
            return self.alias

    manager = Mock()
    manager.get_worker_prefix_snapshot.return_value = SimpleNamespace(
        blocks=tuple(PrefixBlock(128) for _ in range(8)),
    )
    cfg = SimpleNamespace(
        kvbm_enabled=True,
        kvbm_router_pd_enabled=True,
        kvbm_router_pd_decode_direct_enabled=True,
        kvbm_router_pd_decode_direct_min_npu_tokens=1024,
        kvbm_router_pd_decode_direct_min_ratio=0.8,
    )
    route_hint = SimpleNamespace(token_ids=list(range(1250)))

    result = _should_use_pd_decode_direct(
        decode_worker_id="decode-0",
        route_hint=route_hint,
        cfg=cfg,
        kvcache_manager=manager,
    )

    assert result["enabled"] is True
    assert result["npu_prefix_tokens"] == 1024
    assert result["ratio"] == 0.8192


def test_pd_decode_direct_defaults_to_disabled_when_option_is_missing():
    """未提供直达配置时必须保持严格PD，不能让D隐式执行Prefill。"""
    manager = Mock()
    cfg = SimpleNamespace(
        kvbm_enabled=True,
        kvbm_router_pd_enabled=True,
    )

    result = _should_use_pd_decode_direct(
        decode_worker_id="decode-0",
        route_hint=SimpleNamespace(token_ids=list(range(1250))),
        cfg=cfg,
        kvcache_manager=manager,
    )

    assert result["enabled"] is False
    assert result["reason"] == "disabled"
    manager.get_worker_prefix_snapshot.assert_not_called()


def test_pd_decode_direct_stops_at_first_non_npu_block():
    """中间出现 RAM/DISK block 时，后续 NPU block 不能计入连续直达前缀。"""

    class PrefixBlock:
        def __init__(self, token_count, on_npu=True):
            self.alias = SimpleNamespace(token_count=token_count) if on_npu else None

        def select(self, _locations):
            return self.alias

    manager = Mock()
    manager.get_worker_prefix_snapshot.return_value = SimpleNamespace(
        blocks=(
            PrefixBlock(128),
            PrefixBlock(128, on_npu=False),
            PrefixBlock(128),
        ),
    )
    cfg = SimpleNamespace(
        kvbm_enabled=True,
        kvbm_router_pd_enabled=True,
        kvbm_router_pd_decode_direct_enabled=True,
        kvbm_router_pd_decode_direct_min_npu_tokens=128,
        kvbm_router_pd_decode_direct_min_ratio=0.8,
    )

    result = _should_use_pd_decode_direct(
        decode_worker_id="decode-0",
        route_hint=SimpleNamespace(token_ids=list(range(300))),
        cfg=cfg,
        kvcache_manager=manager,
    )

    assert result["enabled"] is False
    assert result["npu_prefix_tokens"] == 128
    assert result["reason"] == "below_min_ratio"


def test_route_capacity_timeout_error_contains_request_id(monkeypatch):
    """准入排队超时错误应包含可追踪的Router请求ID。"""
    cfg = SimpleNamespace(
        router_queue_timeout=0.0,
        router_queue_poll_interval=0.01,
        default_max_tokens=64,
        default_ignore_eos=False,
    )
    monkeypatch.setattr(
        "openjiuwentools.infer_router.config.config.get_global_settings",
        lambda: cfg,
    )
    preprocessor = Mock()
    preprocessor.process_completion.return_value = RouteHint(
        priority=0,
        estimated_output_tokens=64,
        next_turn_prefill=False,
        request_id="capacity-timeout-request-1",
        model="test-model",
        token_ids=[1, 2, 3],
    )
    kvcache_manager = Mock()
    kvcache_manager.find_matches.return_value = {}
    router = Mock()
    router.route_to_workers.return_value = (None, None)
    router.get_worker_admission_snapshot.return_value = {
        "blocker": "routing",
        "prefill_admitted": 3,
        "prefill_healthy": 3,
        "prefill_configured": 3,
        "decode_admitted": 1,
        "decode_healthy": 3,
        "decode_configured": 3,
    }
    worker_manager = Mock()
    worker_manager.get_all_healthy_prefill_workers.return_value = []
    worker_manager.get_all_healthy_decode_workers.return_value = []
    worker_manager.get_all_healthy_combined_workers.return_value = []
    route_fn = _build_route_fn(
        preprocessor=preprocessor,
        kvcache_manager=kvcache_manager,
        scheduler=_ImmediateScheduler(),
        router=router,
        worker_manager=worker_manager,
    )

    with pytest.raises(RuntimeError, match="request_id=capacity-timeout-request-1"):
        route_fn(
            json.dumps(
                {
                    "model": "test-model",
                    "prompt": "hello",
                    "max_tokens": 64,
                    "stream": True,
                }
            ),
            "/v1/completions",
        )


def test_route_decode_direct_releases_unused_prefill_capacity(monkeypatch):
    """D 直达时应立即释放未使用的 P，并只登记 D 等待完成回调释放。"""

    class PrefixBlock:
        @staticmethod
        def select(_locations):
            return SimpleNamespace(token_count=128)

    cfg = SimpleNamespace(
        router_queue_timeout=1.0,
        router_queue_poll_interval=0.01,
        default_max_tokens=64,
        default_ignore_eos=False,
        kvbm_enabled=True,
        kvbm_router_pd_enabled=True,
        kvbm_router_pd_timeout_ms=7000,
        kvbm_router_pd_decode_direct_enabled=True,
        kvbm_router_pd_decode_direct_min_npu_tokens=1024,
        kvbm_router_pd_decode_direct_min_ratio=0.8,
    )
    monkeypatch.setattr(
        "openjiuwentools.infer_router.config.config.get_global_settings",
        lambda: cfg,
    )
    preprocessor = Mock()
    preprocessor.process_completion.return_value = RouteHint(
        priority=0,
        estimated_output_tokens=64,
        next_turn_prefill=False,
        request_id="direct-request-1",
        model="test-model",
        token_ids=list(range(1250)),
    )
    kvcache_manager = Mock()
    decode_snapshot = SimpleNamespace(
        blocks=tuple(PrefixBlock() for _ in range(8)),
    )

    def find_matches(*_args, **kwargs):
        kwargs["snapshot_output"]["decode-0"] = decode_snapshot
        return {"prefill-0": 0, "decode-0": 1024}

    kvcache_manager.find_matches.side_effect = find_matches
    scheduler = _ImmediateScheduler()
    router = Mock()
    router.route_to_workers.return_value = ("prefill-0", "decode-0")
    prefill = WorkerInfo(
        worker_id="prefill-0",
        model="test-model",
        url="http://prefill:8011",
        kvbm_control_url="http://prefill:18011",
    )
    decode = WorkerInfo(
        worker_id="decode-0",
        model="test-model",
        url="http://decode:8020",
        kvbm_control_url="http://decode:18020",
    )
    worker_manager = Mock()
    worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    worker_manager.get_worker.side_effect = {
        "prefill-0": prefill,
        "decode-0": decode,
    }.get

    def worker_control_url(worker):
        return worker.kvbm_control_url

    worker_manager.worker_control_url.side_effect = worker_control_url
    request_workers = {}
    route_fn = _build_route_fn(
        preprocessor=preprocessor,
        kvcache_manager=kvcache_manager,
        scheduler=scheduler,
        router=router,
        worker_manager=worker_manager,
        request_workers=request_workers,
    )

    result = route_fn(
        json.dumps(
            {
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 64,
                "stream": True,
            }
        ),
        "/v1/completions",
    )

    assert result["pd_decode_direct"] is True
    assert result["headers"]["X-Request-Id"] == "direct-request-1"
    assert "X-KV-Target" not in result["headers"]
    assert request_workers.get("direct-request-1") == ["decode-0"]
    router.release_worker.assert_called_once_with("prefill-0", request_id="direct-request-1")
    kvcache_manager.get_worker_prefix_snapshot.assert_not_called()


def test_route_skips_generic_prefetch_when_dedicated_pd_handoff_is_ready(
    monkeypatch,
):
    """专用 PD HIXL 事务启用时不应在 Rust 早派发前重复执行通用预取。"""

    cfg = SimpleNamespace(
        router_queue_timeout=1.0,
        router_queue_poll_interval=0.01,
        default_max_tokens=64,
        default_ignore_eos=False,
        kvbm_enabled=True,
        kvbm_router_pd_enabled=True,
        kvbm_router_pd_timeout_ms=7000,
        kvbm_router_pd_decode_direct_enabled=False,
    )
    monkeypatch.setattr(
        "openjiuwentools.infer_router.config.config.get_global_settings",
        lambda: cfg,
    )
    generic_prefetch = Mock()
    monkeypatch.setattr(
        "openjiuwentools.infer_router.data_plane.server._prefetch_kvbm_blocks_for_worker",
        generic_prefetch,
    )
    preprocessor = Mock()
    preprocessor.process_completion.return_value = RouteHint(
        priority=0,
        estimated_output_tokens=64,
        next_turn_prefill=False,
        request_id="pd-handoff-request-1",
        model="test-model",
        token_ids=list(range(1250)),
    )
    kvcache_manager = Mock()
    kvcache_manager.find_matches.return_value = {}
    scheduler = _ImmediateScheduler()
    router = Mock()
    router.route_to_workers.return_value = ("prefill-0", "decode-0")
    prefill = WorkerInfo(
        worker_id="prefill-0",
        model="test-model",
        url="http://prefill:8011",
        kvbm_control_url="http://prefill:18011",
    )
    decode = WorkerInfo(
        worker_id="decode-0",
        model="test-model",
        url="http://decode:8020",
        kvbm_control_url="http://decode:18020",
    )
    worker_manager = Mock()
    worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    worker_manager.get_worker.side_effect = {
        "prefill-0": prefill,
        "decode-0": decode,
    }.get

    def worker_control_url(worker):
        return worker.kvbm_control_url

    worker_manager.worker_control_url.side_effect = worker_control_url
    route_fn = _build_route_fn(
        preprocessor=preprocessor,
        kvcache_manager=kvcache_manager,
        scheduler=scheduler,
        router=router,
        worker_manager=worker_manager,
    )

    result = route_fn(
        json.dumps(
            {
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 64,
                "stream": True,
            }
        ),
        "/v1/completions",
    )

    assert result["pd_kvbm_enabled"] is True
    assert result["pd_decode_direct"] is False
    generic_prefetch.assert_not_called()


def test_hixl_route_failure_temporarily_yields_capacity_turn(monkeypatch):
    """A failed HIXL attempt must park its FIFO waiter before capacity wait."""
    cfg = SimpleNamespace(
        router_queue_timeout=1.0,
        router_queue_poll_interval=0.01,
        default_max_tokens=64,
        default_ignore_eos=False,
        kvbm_enabled=True,
        kvbm_router_pd_enabled=True,
        kvbm_router_pd_timeout_ms=7000,
        kvbm_router_pd_decode_direct_enabled=False,
    )
    monkeypatch.setattr(
        "openjiuwentools.infer_router.config.config.get_global_settings",
        lambda: cfg,
    )
    preprocessor = Mock()
    preprocessor.process_completion.return_value = RouteHint(
        priority=0,
        estimated_output_tokens=64,
        next_turn_prefill=False,
        request_id="yield-request-1",
        model="test-model",
        token_ids=list(range(1250)),
    )
    kvcache_manager = Mock()
    kvcache_manager.find_matches.return_value = {}
    router = Mock()
    router.wait_for_pd_prefix_affinity = None
    router.refresh_pd_prefix_affinity = None
    router.get_capacity_generation.return_value = 17
    router.enqueue_capacity_waiter.side_effect = [False, True]
    router.wait_for_capacity_change.return_value = False
    router.wait_for_capacity_turn.return_value = True
    router.route_to_workers.side_effect = [
        (None, None),
        ("prefill-0", "decode-0"),
    ]
    router.get_worker_admission_snapshot.return_value = {
        "blocker": "decode",
        "prefill_admitted": 1,
        "prefill_healthy": 1,
        "prefill_configured": 1,
        "decode_admitted": 0,
        "decode_healthy": 1,
        "decode_configured": 1,
        "prefill_details": {},
        "decode_details": {
            "decode-0": {
                "reason": "decode_free_block_budget",
            }
        },
    }
    router.take_pd_decode_admission_ticket.return_value = ""
    prefill = WorkerInfo(
        worker_id="prefill-0",
        model="test-model",
        url="http://prefill:8011",
        kvbm_control_url="http://prefill:18011",
    )
    decode = WorkerInfo(
        worker_id="decode-0",
        model="test-model",
        url="http://decode:8020",
        kvbm_control_url="http://decode:18020",
    )
    worker_manager = Mock()
    worker_manager.get_all_healthy_prefill_workers.return_value = [prefill]
    worker_manager.get_all_healthy_decode_workers.return_value = [decode]
    worker_manager.get_worker.side_effect = {
        "prefill-0": prefill,
        "decode-0": decode,
    }.get

    def worker_control_url(worker):
        return worker.kvbm_control_url

    worker_manager.worker_control_url.side_effect = worker_control_url
    route_fn = _build_route_fn(
        preprocessor=preprocessor,
        kvcache_manager=kvcache_manager,
        scheduler=_ImmediateScheduler(),
        router=router,
        worker_manager=worker_manager,
    )

    result = route_fn(
        json.dumps(
            {
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 64,
                "stream": True,
            }
        ),
        "/v1/completions",
    )

    assert result["pd_kvbm_enabled"] is True
    router.defer_capacity_waiter.assert_called_once()
    deferred_request_id, deferred_model, deferred_generation = router.defer_capacity_waiter.call_args.args
    assert deferred_model == "test-model"
    assert deferred_generation == 17
    assert router.defer_capacity_waiter.call_args.kwargs == {
        "blocked_worker_ids": ("decode-0",),
    }
    router.wait_for_capacity_change.assert_called_once()
    wait_args = router.wait_for_capacity_change.call_args
    assert wait_args.args == (17,)
    assert wait_args.kwargs["timeout"] == pytest.approx(1.0, abs=0.1)
    assert wait_args.kwargs["request_id"] == deferred_request_id


def test_overlap_snapshot_is_serialized_with_worker_selection():
    """并发请求不能在路由锁外同时计算随后复用的overlap快照。"""
    preprocess_barrier = threading.Barrier(2)
    first_match_entered = threading.Event()
    second_match_entered = threading.Event()
    release_first_match = threading.Event()
    call_count_lock = threading.Lock()
    match_calls = 0

    def process_completion(_request, _hints, request_id):
        preprocess_barrier.wait(timeout=2)
        return RouteHint(
            priority=0,
            estimated_output_tokens=64,
            next_turn_prefill=False,
            request_id=request_id,
            model="test-model",
            token_ids=list(range(16)),
        )

    def find_matches(**_kwargs):
        nonlocal match_calls
        with call_count_lock:
            match_calls += 1
            call_number = match_calls
        if call_number == 1:
            first_match_entered.set()
            assert release_first_match.wait(timeout=2)
        else:
            second_match_entered.set()
        return {"combined-0": 16}

    preprocessor = Mock()
    preprocessor.process_completion.side_effect = process_completion
    kvcache_manager = Mock()
    kvcache_manager.find_matches.side_effect = find_matches
    scheduler = _ImmediateScheduler()
    router = Mock()
    router.route_to_workers.return_value = ("combined-0", None)
    worker = WorkerInfo(worker_id="combined-0", model="test-model", url="http://worker")
    worker_manager = Mock()
    worker_manager.get_all_healthy_prefill_workers.return_value = []
    worker_manager.get_all_healthy_combined_workers.return_value = [worker]
    worker_manager.get_worker.return_value = worker
    request_contexts = {}

    route_fn = _build_route_fn(
        preprocessor=preprocessor,
        kvcache_manager=kvcache_manager,
        scheduler=scheduler,
        router=router,
        worker_manager=worker_manager,
        request_contexts=request_contexts,
    )
    body = json.dumps(
        {
            "model": "test-model",
            "prompt": "hello",
            "max_tokens": 64,
            "stream": True,
        }
    )
    errors = []

    def invoke_route():
        try:
            route_fn(body, "/v1/completions")
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=invoke_route) for _ in range(2)]
    for thread in threads:
        thread.start()

    assert first_match_entered.wait(timeout=2)
    assert not second_match_entered.wait(timeout=0.2)
    release_first_match.set()

    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert not errors
    assert match_calls == 2
    assert len(request_contexts) == 2
    assert all(context["router_phase_ms"]["router_route_ms"] >= 0.0 for context in request_contexts.values())
    assert all(
        context["router_phase_ms"]["router_request_parse_ms"] >= 0.0
        and context["router_phase_ms"]["router_preprocessor_ms"] >= 0.0
        and context["router_phase_ms"]["router_worker_selection_ms"] >= 0.0
        and context["router_phase_ms"]["router_route_attempt_ms"] >= 0.0
        and context["router_phase_ms"]["router_capacity_wait_ms"] >= 0.0
        and context["router_phase_ms"]["router_capacity_turn_wait_ms"] >= 0.0
        and context["router_phase_ms"]["router_route_build_ms"] >= 0.0
        and context["router_phase_ms"]["router_route_unaccounted_ms"] >= 0.0
        and context["router_phase_ms"]["router_route_attempt_count"] == 1
        for context in request_contexts.values()
    )
