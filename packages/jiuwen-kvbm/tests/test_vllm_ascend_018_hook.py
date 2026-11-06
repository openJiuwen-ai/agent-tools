# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
import logging
import os
import sys
import threading
import time
import types


from jiuwen_kvbm.integrations import vllm_ascend_018 as integration
from jiuwen_kvbm.integrations.vllm_ascend_018 import (
    _drain_runtime_event_outbox,
    _enqueue_runtime_events,
    _runtime_event_to_relay_event,
    get_bound_vllm_ascend_018_dataplane,
    get_bound_vllm_ascend_018_physical_block_resolver,
    get_bound_vllm_ascend_018_provider,
    install_vllm_ascend_018_block_pool_hook,
    install_vllm_ascend_018_engine_core_hook,
    install_vllm_ascend_018_kv_cache_manager_hook,
    install_vllm_ascend_018_kv_event_publisher_hook,
    install_vllm_ascend_018_multiprocess_hooks,
    install_vllm_ascend_018_scheduler_diagnostics_hook,
    jiuwen_kvbm_vllm_ascend_018_run_engine_core,
    uninstall_vllm_ascend_018_engine_core_hook,
)
from jiuwen_kvbm.runtime import (
    BlockHashesInFlightError,
    BlockReservationCapacityError,
    KVBMRuntimeEvent,
)


class FakeBlock:
    def __init__(self, values):
        self.values = list(values)
        self.device = "npu"
        self.dtype = "float32"
        self.shape = (len(self.values),)

    def __getitem__(self, index):
        return self.values[index]

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return FakeBlock(self.values)

    def to(self, device=None, dtype=None):
        return self

    def copy_(self, other):
        self.values = list(other.values)
        return self

    def numel(self):
        return len(self.values)

    @staticmethod
    def element_size():
        return 4


def test_scheduler_pd_import_gate_blocks_prefill_until_released(monkeypatch):
    """prepare 后 D 请求不应被调度，解除 gate 后才允许本地 prefill。"""

    monkeypatch.setenv("JIUWEN_KVBM_PD_BLOCK_OWNERSHIP", "legacy")

    class RequestQueue:
        def __init__(self):
            self.requests = []
            self.remove_calls = 0
            self.prepend_calls = 0

        def remove_requests(self, requests):
            self.remove_calls += 1
            for request in requests:
                if request in self.requests:
                    self.requests.remove(request)

        def prepend_request(self, request):
            self.prepend_calls += 1
            if request not in self.requests:
                self.requests.insert(0, request)

    class SchedulerOutput:
        scheduled_new_reqs = []
        num_scheduled_tokens = {}

    class Scheduler:
        def __init__(self):
            self.requests = {}
            self.waiting = RequestQueue()
            self.skipped_waiting = RequestQueue()
            self.running = []
            self.scheduled = []

        def add_request(self, request):
            self.requests[request.request_id] = request
            self.waiting.requests.append(request)

        def schedule(self):
            if self.waiting.requests:
                self.scheduled.append(self.waiting.requests.pop(0))
            return SchedulerOutput()

    class AsyncScheduler(Scheduler):
        """模拟 vLLM 0.18 重写 schedule() 的异步调度器。"""

        def schedule(self):
            if self.waiting.requests:
                self.scheduled.append(self.waiting.requests.pop(0))
            return SchedulerOutput()

    class AscendScheduler(AsyncScheduler):
        """模拟继承 AsyncScheduler 但未再次重写 schedule() 的 Ascend 类型。"""

    def reject_preplan(*_args, **_kwargs):
        raise BlockReservationCapacityError(
            50,
            19,
            0,
            owner_id="pd-gate-1",
        )

    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {})
    monkeypatch.setattr(integration, "_PD_TARGET_ATTACH_TASKS", [])
    monkeypatch.setattr(integration, "_PD_TARGET_PLAN_TASKS", [])
    monkeypatch.setattr(integration, "_PD_FAST_PATH_PENDING", {})
    monkeypatch.setattr(integration, "_OBSERVED_SCHEDULERS", [])
    monkeypatch.setattr(integration, "_PD_REQUEST_BLOCK_HASHES", {})
    event_broker = integration._PDControlEventBroker()
    monkeypatch.setattr(integration, "_PD_CONTROL_EVENTS", event_broker)
    monkeypatch.setattr(
        integration,
        "_scheduler_preplan_pd_import",
        reject_preplan,
    )
    try:
        assert install_vllm_ascend_018_scheduler_diagnostics_hook(
            scheduler_cls=Scheduler,
            new_request_data_cls=None,
        )
        prepared = integration._local_pd_import_prepare({"handoff_id": "pd-gate-1", "request_id": "decode-gated-1"})
        scheduler = AscendScheduler()
        request = types.SimpleNamespace(
            request_id="decode-gated-1-8a773759",
            block_hashes=[b"decode-block-hash"],
        )
        scheduler.add_request(request)
        scheduler.schedule()
        gate_status = integration._local_pd_import_status({"request_id": "decode-gated-1"})
        _subscriber, events = event_broker.subscribe(
            handoff_id="pd-gate-1",
            after_sequence=0,
        )

        assert prepared["ok"] is True
        assert AsyncScheduler.__dict__.get("_jiuwen_kvbm_018_runtime_schedule_hook_installed") is True
        assert integration._PD_REQUEST_BLOCK_HASHES["decode-gated-1"] == (b"decode-block-hash",)
        assert prepared["gate"]["engine_request_id"] is None
        assert gate_status["gate"]["engine_request_id"] == ("decode-gated-1-8a773759")
        assert scheduler.scheduled == []
        assert scheduler.waiting.requests == [request]
        assert scheduler.waiting.remove_calls == 1
        assert scheduler.waiting.prepend_calls == 1
        scheduler.schedule()
        assert scheduler.scheduled == []
        assert scheduler.waiting.requests == [request]
        assert scheduler.waiting.remove_calls == 2
        assert scheduler.waiting.prepend_calls == 2
        request_seen = next(event for event in events if event["event_type"] == "REQUEST_SEEN")
        assert request_seen["result"] == {
            "preplan_ready": False,
            "preplan_error": ("insufficient free vLLM blocks for delta import: requested=50, available=19, reused=0"),
            "capacity_unavailable": False,
            "capacity_waiting": True,
            "scheduler_allocation_ready": False,
            "allocated_blocks": 0,
        }

        released = integration._local_pd_import_release({"handoff_id": "pd-gate-1", "request_id": "decode-gated-1"})
        scheduler.schedule()

        assert released["released"] is True
        assert scheduler.scheduled == [request]
        assert scheduler.waiting.prepend_calls == 2
    finally:
        runtime_original = getattr(
            AsyncScheduler,
            "_jiuwen_kvbm_018_runtime_original_schedule",
            None,
        )
        if runtime_original is not None:
            AsyncScheduler.schedule = runtime_original
        original_schedule = getattr(
            Scheduler,
            "_jiuwen_kvbm_018_original_schedule",
            None,
        )
        original_add = getattr(
            Scheduler,
            "_jiuwen_kvbm_018_original_add_request",
            None,
        )
        if original_schedule is not None:
            Scheduler.schedule = original_schedule
        if original_add is not None:
            Scheduler.add_request = original_add


def test_scheduler_pd_attach_fast_path_schedules_request_in_same_cycle(
    monkeypatch,
    caplog,
):
    """attach 完成后应把 D 请求提升到队首并在当前周期获得 token budget。"""

    class RequestQueue:
        def __init__(self, requests):
            self.requests = list(requests)

        def remove_requests(self, requests):
            for request in requests:
                if request in self.requests:
                    self.requests.remove(request)

        def prepend_request(self, request):
            if request not in self.requests:
                self.requests.insert(0, request)

    class Scheduler:
        def __init__(self):
            self.target = types.SimpleNamespace(request_id="decode-fast-path")
            self.other = types.SimpleNamespace(request_id="decode-existing")
            self.requests = {
                self.target.request_id: self.target,
                self.other.request_id: self.other,
            }
            self.waiting = RequestQueue([self.other, self.target])
            self.skipped_waiting = RequestQueue([])
            self.running = []
            self.model_outputs = []

        def schedule(self):
            request = self.waiting.requests.pop(0)
            return types.SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=[],
                num_scheduled_tokens={request.request_id: 1},
            )

        def update_from_output(self, scheduler_output, model_output):
            self.model_outputs.append((scheduler_output, model_output))
            return (scheduler_output, model_output)

    def fake_process_attach_tasks(scheduler, _log, *, cycle_id=0):
        promoted = integration._promote_pd_request_for_fast_path(
            scheduler,
            scheduler.target.request_id,
        )
        return [
            {
                "handoff_id": "pd-fast-path-1",
                "request_id": scheduler.target.request_id,
                "attach_completed_at": time.monotonic() - 0.002,
                "attach_cycle_id": cycle_id,
                "d_request_seen_to_attach_start_ms": 10.0,
                "attach_task_queue_ms": 1.0,
                "attach_execution_ms": 0.5,
                "promoted_to_waiting_head": promoted,
                "running_request_count": 0,
            }
        ]

    monkeypatch.setattr(
        integration,
        "_process_pd_target_attach_tasks",
        fake_process_attach_tasks,
    )
    monkeypatch.setattr(
        integration,
        "_pd_scheduler_has_pending_work",
        lambda: True,
    )
    monkeypatch.setattr(integration, "_PD_FAST_PATH_PENDING", {})
    monkeypatch.setattr(integration, "_PD_D_EXECUTION_TIMINGS", {})
    caplog.set_level(logging.INFO)
    try:
        assert install_vllm_ascend_018_scheduler_diagnostics_hook(
            scheduler_cls=Scheduler,
            new_request_data_cls=None,
        )
        scheduler = Scheduler()

        output = scheduler.schedule()
        model_output = types.SimpleNamespace()
        scheduler.update_from_output(output, model_output)

        assert output.num_scheduled_tokens == {"decode-fast-path": 1}
        assert scheduler.model_outputs == [(output, model_output)]
        assert scheduler.waiting.requests == [scheduler.other]
        assert "scheduled_same_cycle=True" in caplog.text
        assert "promoted_to_waiting_head=True" in caplog.text
        assert "D first model-output timing" in caplog.text
        assert "attach_to_schedule_output_ms=" in caplog.text
        assert "schedule_output_to_model_output_ms=" in caplog.text
        assert "scheduler_update_ms=" in caplog.text
        assert integration._PD_D_EXECUTION_TIMINGS == {}
    finally:
        original_schedule = getattr(
            Scheduler,
            "_jiuwen_kvbm_018_original_schedule",
            None,
        )
        if original_schedule is not None:
            Scheduler.schedule = original_schedule
        original_update = getattr(
            Scheduler,
            "_jiuwen_kvbm_018_original_update_from_output",
            None,
        )
        if original_update is not None:
            Scheduler.update_from_output = original_update


def test_scheduler_idle_fast_path_skips_pd_scans_and_samples_timing(
    monkeypatch,
    caplog,
):
    """无 PD 工作时 Decode 周期应绕过任务和 gate 扫描，并输出采样耗时。"""

    class SchedulerOutput:
        scheduled_new_reqs = []
        num_scheduled_tokens = {}

    class Scheduler:
        def __init__(self):
            self.calls = 0

        def schedule(self):
            self.calls += 1
            return SchedulerOutput()

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("idle fast path must not scan PD state")

    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {})
    monkeypatch.setattr(integration, "_PD_TARGET_ATTACH_TASKS", [])
    monkeypatch.setattr(integration, "_PD_TARGET_PLAN_TASKS", [])
    monkeypatch.setattr(integration, "_PD_FAST_PATH_PENDING", {})
    monkeypatch.setattr(integration, "_OBSERVED_SCHEDULERS", [])
    monkeypatch.setattr(integration, "_PD_SCHEDULER_IDLE_SAMPLE_EVERY", 1)
    monkeypatch.setattr(integration, "_process_pd_target_plan_tasks", unexpected_call)
    monkeypatch.setattr(integration, "_process_pd_target_attach_tasks", unexpected_call)
    monkeypatch.setattr(
        integration,
        "_remove_gated_pd_requests_from_queues",
        unexpected_call,
    )
    try:
        assert install_vllm_ascend_018_scheduler_diagnostics_hook(
            scheduler_cls=Scheduler,
            new_request_data_cls=None,
        )
        scheduler = Scheduler()
        with caplog.at_level("INFO", logger="jiuwen.kvbm.vllm_ascend_018"):
            output = scheduler.schedule()

        assert isinstance(output, SchedulerOutput)
        assert scheduler.calls == 1
        timing_message = next(
            record.getMessage() for record in caplog.records if "scheduler hook timing" in record.getMessage()
        )
        assert "active_pd=False" in timing_message
        assert "idle_fast_path=True" in timing_message
        assert "pd_hook_total_us=" in timing_message
        assert "original_schedule_us=" in timing_message
        assert "gate_scan_us=0.000" in timing_message
        assert "task_scan_us=0.000" in timing_message
    finally:
        original_schedule = getattr(
            Scheduler,
            "_jiuwen_kvbm_018_original_schedule",
            None,
        )
        if original_schedule is not None:
            Scheduler.schedule = original_schedule


def test_p_source_intent_enables_scheduler_cycle_diagnostics(monkeypatch, caplog):
    """仅存在P source intent时也应记录首次调度周期和等待周期。"""

    class Scheduler:
        def __init__(self):
            self.requests = {}
            self.waiting = []
            self.running = []

        def add_request(self, request):
            self.requests[request.request_id] = request
            self.waiting.append(request)

        def schedule(self):
            request = self.waiting.pop(0)
            self.running.append(request)
            return types.SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=[],
                num_scheduled_tokens={request.request_id: 28},
            )

    monkeypatch.setattr(integration, "_PD_SOURCE_EXPORT_INTENTS", {})
    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {})
    monkeypatch.setattr(integration, "_PD_TARGET_ATTACH_TASKS", [])
    monkeypatch.setattr(integration, "_PD_TARGET_PLAN_TASKS", [])
    monkeypatch.setattr(integration, "_PD_FAST_PATH_PENDING", {})
    monkeypatch.setattr(integration, "_OBSERVED_SCHEDULERS", [])
    integration._local_pd_export_prepare(
        {
            "handoff_id": "pd-source-cycle-1",
            "request_id": "prefill-cycle-1",
            "prompt_token_count": 128,
            "ttl_ms": 30_000,
        }
    )
    try:
        assert install_vllm_ascend_018_scheduler_diagnostics_hook(
            scheduler_cls=Scheduler,
            new_request_data_cls=None,
        )
        scheduler = Scheduler()
        request = types.SimpleNamespace(
            request_id="prefill-cycle-1-8a773759",
            block_hashes=[b"prefill-cycle-hash"],
            prompt_token_ids=list(range(128)),
            arrival_time=time.monotonic() - 0.004,
        )
        scheduler.add_request(request)
        with caplog.at_level("INFO", logger="jiuwen.kvbm.vllm_ascend_018"):
            scheduler.schedule()

        intent = integration._PD_SOURCE_EXPORT_INTENTS["prefill-cycle-1"]
        assert intent.scheduler_seen_cycle_id == 0
        assert intent.worker_request_received_to_scheduler_seen_ms >= 4.0
        assert intent.worker_request_received_timing_source == "arrival_time:monotonic"
        assert intent.first_scheduled_cycle_id == 1
        assert intent.scheduler_wait_cycles == 0
        assert intent.first_scheduled_token_count == 28
        assert intent.first_scheduled_running_request_count == 1
        assert intent.first_scheduled_waiting_request_count == 0
        assert integration._pd_scheduler_has_pending_work() is True
        assert "P scheduler cycle diagnostics: cycle_id=1" in caplog.text
        assert "P request first scheduled" in caplog.text
    finally:
        original_schedule = getattr(
            Scheduler,
            "_jiuwen_kvbm_018_original_schedule",
            None,
        )
        original_add = getattr(
            Scheduler,
            "_jiuwen_kvbm_018_original_add_request",
            None,
        )
        if original_schedule is not None:
            Scheduler.schedule = original_schedule
        if original_add is not None:
            Scheduler.add_request = original_add


def test_attached_gate_aliases_and_stale_fast_path_state_are_cleaned(
    monkeypatch,
    caplog,
):
    """attach 后应清理 gate 别名，未被调度的诊断状态也必须有限期回收。"""

    gate = integration._PDTargetRequestGate(
        handoff_id="handoff-cleanup",
        request_id="decode-cleanup",
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 30,
        request_seen=True,
        engine_request_id="decode-cleanup-8a773759",
    )
    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {"decode-cleanup": gate})
    monkeypatch.setattr(integration, "_PD_TARGET_ATTACH_TASKS", [])
    monkeypatch.setattr(integration, "_PD_TARGET_PLAN_TASKS", [])
    monkeypatch.setattr(integration, "_PD_FAST_PATH_PENDING", {})

    removed = integration._remove_attached_pd_target_gates(
        "decode-cleanup-8a773759",
        "handoff-cleanup",
    )
    assert removed == 1
    assert integration._PD_TARGET_GATES == {}

    output = types.SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={},
    )
    record = {
        "handoff_id": "handoff-cleanup",
        "request_id": "decode-cleanup-8a773759",
        "attach_cycle_id": 1,
        "d_request_seen_to_attach_start_ms": 1.0,
        "attach_task_queue_ms": 1.0,
        "attach_execution_ms": 1.0,
        "promoted_to_waiting_head": True,
        "running_request_count": 0,
    }
    with caplog.at_level("WARNING", logger="jiuwen.kvbm.vllm_ascend_018"):
        integration._log_pd_fast_path_schedule_result(
            output,
            [record],
            logging.getLogger("jiuwen.kvbm.vllm_ascend_018"),
            cycle_id=1,
        )
        for cycle_id in range(2, integration._PD_FAST_PATH_PENDING_MAX_CYCLES + 1):
            integration._log_pd_fast_path_schedule_result(
                output,
                [],
                logging.getLogger("jiuwen.kvbm.vllm_ascend_018"),
                cycle_id=cycle_id,
            )

    assert integration._PD_FAST_PATH_PENDING == {}
    assert integration._pd_scheduler_has_pending_work() is False
    assert "discarded stale D attach fast-path diagnostics" in caplog.text


def test_pd_delta_preplan_is_adopted_without_scheduler_mailbox_wait(monkeypatch):
    """D请求提前预留与P布局一致时应直接采用，不再提交同步plan task。"""

    class Reservation:
        reservation_id = "reservation-preplanned"
        raw_blocks = (types.SimpleNamespace(block_id=101),)
        planned_blocks = (
            types.SimpleNamespace(block_id=7),
            types.SimpleNamespace(block_id=8),
            types.SimpleNamespace(block_id=101),
        )
        block_ids = (101,)
        transfer_positions = (2,)

        def to_dict(self):
            return {
                "reservation_id": self.reservation_id,
                "block_ids": list(self.block_ids),
                "planned_block_ids": [7, 8, 101],
            }

    class ReservationManager:
        def __init__(self):
            self.reservation = Reservation()
            self.reserve_calls = 0
            self.renew_calls = 0
            self.rollback_calls = 0

        def reserve_delta(self, block_hashes, *, ttl_ms, owner_id=""):
            assert len(block_hashes) == 3
            assert ttl_ms == 120_000
            assert owner_id == ""
            self.reserve_calls += 1
            return self.reservation

        def renew(self, reservation_id, *, ttl_ms):
            assert reservation_id == self.reservation.reservation_id
            assert ttl_ms == 120_000
            self.renew_calls += 1
            return self.reservation

        def rollback(self, reservation_id):
            assert reservation_id == self.reservation.reservation_id
            self.rollback_calls += 1
            return self.reservation

        @staticmethod
        def stats():
            return {"free_blocks": 900}

    request = types.SimpleNamespace(
        request_id="decode-preplanned-engine",
        prompt_token_ids=list(range(10)),
        block_hashes=[b"hash-0", b"hash-1"],
        cache_salt=None,
        lora_request=None,
        mm_features=None,
        prompt_embeds=None,
        num_computed_tokens=0,
    )
    scheduler = types.SimpleNamespace(
        block_size=4,
        kv_cache_manager=types.SimpleNamespace(num_kv_cache_groups=1),
        requests={request.request_id: request},
        running=[],
    )
    manager = ReservationManager()
    monkeypatch.setattr(integration, "_process_block_reservations", lambda: manager)
    preplan = integration._scheduler_preplan_pd_import(scheduler, request)
    gate = integration._PDTargetRequestGate(
        handoff_id="handoff-preplanned",
        request_id="decode-preplanned",
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 30,
        request_seen=True,
        engine_request_id=request.request_id,
        request_seen_at=time.monotonic(),
        preplan=preplan,
    )
    monkeypatch.setattr(
        integration,
        "_PD_TARGET_GATES",
        {gate.request_id: gate},
    )
    monkeypatch.setattr(integration, "_PD_TARGET_PLAN_TASKS", [])
    monkeypatch.setattr(integration, "_OBSERVED_SCHEDULERS", [scheduler])

    result = integration._local_pd_import_plan(
        {
            "handoff_id": gate.handoff_id,
            "request_id": gate.request_id,
            "block_groups": list(preplan.block_groups),
            "token_block_fingerprints": integration._token_block_fingerprints(
                request.prompt_token_ids,
                scheduler.block_size,
            ),
            "partial_token_fingerprint": integration._partial_token_fingerprint(
                request.prompt_token_ids,
                scheduler.block_size,
            ),
            "partial_valid_tokens": 2,
            "semantic_plain_text": True,
            "reservation_ttl_ms": 120_000,
        }
    )

    assert result["ok"] is True
    assert result["delta_preplanned"] is True
    assert result["delta_plan_task_queue_ms"] == 0.0
    assert result["delta_hash_validate_ms"] >= 0.0
    assert manager.reserve_calls == 1
    assert manager.renew_calls == 1
    assert manager.rollback_calls == 0
    assert gate.preplan is None
    assert integration._PD_TARGET_PLAN_TASKS == []


def test_pd_import_prepare_does_not_recreate_required_gate(monkeypatch):
    """延迟import只能续租原gate，不能在请求已离开Scheduler后重建空gate。"""

    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {})

    result = integration._local_pd_import_prepare(
        {
            "handoff_id": "handoff-delayed",
            "request_id": "decode-delayed",
            "ttl_ms": 120_000,
            "require_existing": True,
        }
    )

    assert result["ok"] is False
    assert "gate disappeared" in result["error"]
    assert integration._PD_TARGET_GATES == {}


def test_pd_delta_plan_waits_for_inflight_leader_then_retries(monkeypatch):
    """Follower应在控制线程等待leader发布，再由Scheduler重新plan。"""

    gate = integration._PDTargetRequestGate(
        handoff_id="handoff-inflight",
        request_id="decode-inflight",
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 30,
        request_seen=True,
        engine_request_id="decode-inflight-engine",
        request_seen_at=time.monotonic(),
    )
    manager = types.SimpleNamespace(
        wait_calls=0,
        wait_for_inflight_reservations=lambda reservation_ids, timeout_seconds: (
            setattr(manager, "wait_calls", manager.wait_calls + 1) or True
        ),
    )
    plan_calls = 0

    def plan(_scheduler, _task):
        nonlocal plan_calls
        plan_calls += 1
        if plan_calls == 1:
            raise BlockHashesInFlightError(46, ["leader-reservation"])
        return {
            "ok": True,
            "request_id": gate.engine_request_id,
            "reservation": {"reservation_id": "follower-reservation"},
        }

    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {gate.request_id: gate})
    monkeypatch.setattr(integration, "_PD_TARGET_PLAN_TASKS", [])
    monkeypatch.setattr(integration, "_process_block_reservations", lambda: manager)
    monkeypatch.setattr(integration, "_scheduler_plan_pd_import", plan)
    result_holder = {}

    def run_plan():
        result_holder["result"] = integration._local_pd_import_plan(
            {
                "handoff_id": gate.handoff_id,
                "request_id": gate.request_id,
                "block_groups": [{"logical_index": 0, "group_id": 0}],
                "timeout_ms": 1_000,
            }
        )

    control_thread = threading.Thread(target=run_plan)
    control_thread.start()
    deadline = time.monotonic() + 1.0
    for _ in range(2):
        while not integration._PD_TARGET_PLAN_TASKS:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        integration._process_pd_target_plan_tasks(
            object(),
            logging.getLogger("jiuwen.kvbm.vllm_ascend_018"),
        )
    control_thread.join(timeout=1.0)

    assert not control_thread.is_alive()
    assert result_holder["result"]["ok"] is True
    assert result_holder["result"]["delta_inflight_retry_count"] == 1
    assert result_holder["result"]["delta_inflight_wait_ms"] >= 0.0
    assert manager.wait_calls == 1


def test_pd_delta_plan_waits_for_block_pool_capacity_then_retries(monkeypatch):
    """容量不足时应在控制线程等待BlockPool释放，而不是立即回退。"""

    gate = integration._PDTargetRequestGate(
        handoff_id="handoff-capacity",
        request_id="decode-capacity",
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 30,
        request_seen=True,
        engine_request_id="decode-capacity-engine",
        request_seen_at=time.monotonic(),
        preplan_capacity_waiting=True,
    )

    class Manager:
        def __init__(self):
            self.wait_calls = 0
            self.release_calls = 0

        def wait_for_free_blocks(self, minimum, *, timeout_seconds):
            assert minimum == 4
            assert timeout_seconds > 0
            self.wait_calls += 1
            return True

        def release_pending_delta(self, owner_id):
            assert owner_id == gate.handoff_id
            self.release_calls += 1
            return True

    manager = Manager()
    plan_calls = 0

    def plan(_scheduler, _task):
        nonlocal plan_calls
        plan_calls += 1
        if plan_calls == 1:
            raise BlockReservationCapacityError(
                4,
                3,
                46,
                owner_id=gate.handoff_id,
            )
        return {
            "ok": True,
            "request_id": gate.engine_request_id,
            "reservation": {"reservation_id": "capacity-reservation"},
        }

    monkeypatch.setattr(integration, "_PD_TARGET_GATES", {gate.request_id: gate})
    monkeypatch.setattr(integration, "_PD_TARGET_PLAN_TASKS", [])
    monkeypatch.setattr(integration, "_process_block_reservations", lambda: manager)
    monkeypatch.setattr(integration, "_scheduler_plan_pd_import", plan)
    result_holder = {}

    def run_plan():
        result_holder["result"] = integration._local_pd_import_plan(
            {
                "handoff_id": gate.handoff_id,
                "request_id": gate.request_id,
                "block_groups": [{"logical_index": 0, "group_id": 0}],
                "timeout_ms": 1_000,
            }
        )

    control_thread = threading.Thread(target=run_plan)
    control_thread.start()
    deadline = time.monotonic() + 1.0
    for _ in range(2):
        while not integration._PD_TARGET_PLAN_TASKS:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        integration._process_pd_target_plan_tasks(
            object(),
            logging.getLogger("jiuwen.kvbm.vllm_ascend_018"),
        )
    control_thread.join(timeout=1.0)

    assert not control_thread.is_alive()
    assert result_holder["result"]["ok"] is True
    assert result_holder["result"]["delta_capacity_retry_count"] == 1
    assert result_holder["result"]["delta_capacity_wait_ms"] >= 0.0
    assert manager.wait_calls == 1
    assert manager.release_calls == 0


def test_pd_delta_preplan_is_rolled_back_when_gate_released(monkeypatch):
    """D gate失败释放时必须回滚尚未采用的提前reservation。"""

    rolled_back = []
    preplan = integration._PDTargetPreplan(
        target_packed_hashes=(b"hash",),
        block_groups=(
            {
                "logical_index": 0,
                "group_id": 0,
                "request_scoped_partial": False,
            },
        ),
        result={"reservation": {"reservation_id": "reservation-to-rollback"}},
        created_at=time.monotonic(),
        hash_prepare_ms=1.0,
        reserve_ms=2.0,
    )
    gate = integration._PDTargetRequestGate(
        handoff_id="handoff-rollback",
        request_id="decode-rollback",
        created_at=time.monotonic(),
        expires_at=time.monotonic() + 30,
        preplan=preplan,
    )
    manager = types.SimpleNamespace(
        rollback=lambda reservation_id: rolled_back.append(reservation_id),
        release_pending_delta=lambda _owner_id: False,
    )
    monkeypatch.setattr(integration, "_process_block_reservations", lambda: manager)
    monkeypatch.setattr(
        integration,
        "_PD_TARGET_GATES",
        {gate.request_id: gate},
    )

    result = integration._local_pd_import_release({"handoff_id": gate.handoff_id, "request_id": gate.request_id})

    assert result["released"] is True
    assert rolled_back == ["reservation-to-rollback"]
    assert integration._PD_TARGET_GATES == {}


def test_scheduled_request_ids_accepts_aggregated_cache_request_data():
    """vLLM 0.18 的不可迭代 CacheRequestData 应通过 req_ids 解析。"""

    cache_request_data = types.SimpleNamespace(req_ids=["decode-cached-1", "decode-cached-2"])
    output = types.SimpleNamespace(
        scheduled_new_reqs=[types.SimpleNamespace(req_id="decode-new-1")],
        scheduled_cached_reqs=cache_request_data,
        num_scheduled_tokens={},
    )

    assert integration._scheduled_request_ids(output) == {
        "decode-new-1",
        "decode-cached-1",
        "decode-cached-2",
    }


def test_new_request_data_hook_logs_when_scheduler_is_replaced(caplog, monkeypatch):
    """Scheduler 被 Ascend 替换时仍应在 NewRequestData 构造点输出命中诊断。"""

    class FakeRequest:
        request_id = "request-data-cache-hit"
        prompt_token_ids = list(range(2048))
        num_computed_tokens = 1024
        block_hashes = [b"request-data-block-hash"]

    class FakeNewRequestData:
        @classmethod
        def from_request(cls, request, *_args, **_kwargs):
            return types.SimpleNamespace(
                req_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                num_computed_tokens=request.num_computed_tokens,
            )

    try:
        monkeypatch.setattr(integration, "_PD_REQUEST_BLOCK_HASHES", {})
        assert install_vllm_ascend_018_scheduler_diagnostics_hook(
            scheduler_cls=None,
            new_request_data_cls=FakeNewRequestData,
        )
        with caplog.at_level("INFO", logger="jiuwen.kvbm.vllm_ascend_018"):
            result = FakeNewRequestData.from_request(FakeRequest())

        assert result.req_id == "request-data-cache-hit"
        assert integration._PD_REQUEST_BLOCK_HASHES["request-data-cache-hit"] == (b"request-data-block-hash",)
        message = None
        for record in caplog.records:
            text = record.getMessage()
            if "Jiuwen KVBM request cache-hit diagnostics" in text and "request-data-cache-hit" in text:
                message = text
                break
        assert message is not None
        assert "'cached_prompt_tokens': 1024" in message
        assert "'computed_prompt_tokens': 1024" in message
        assert "'source': 'new_request_data'" in message
    finally:
        original = getattr(
            FakeNewRequestData,
            "_jiuwen_kvbm_018_original_from_request",
            None,
        )
        if original is not None:
            FakeNewRequestData.from_request = original


def test_kv_cache_manager_hook_skips_expensive_prefix_diagnostics_by_default(
    caplog,
    monkeypatch,
):
    """正常 PD 路径不应遍历 BlockPool 生成详细指纹诊断。"""

    class ExplodingCachedMap:
        @property
        def _cache(self):
            raise AssertionError("未开启 diagnostics 时不应读取全量缓存 key")

    class FakeBlockPool:
        cached_block_hash_to_block = ExplodingCachedMap()

    class FakeRequest:
        request_id = "request-no-prefix-diagnostics"
        block_hashes = [b"hash-a"]
        prompt_token_ids = list(range(128))

    class FakeKVCacheManager:
        block_pool = FakeBlockPool()
        block_size = 128

        def __init__(self):
            self.events = []
            self.computed_result = (object(), 0)

        def take_events(self):
            return list(self.events)

        def get_computed_blocks(self, _request):
            return self.computed_result

    monkeypatch.setenv("JIUWEN_KVBM_VLLM_ASCEND_018_DIAGNOSTICS", "0")
    assert install_vllm_ascend_018_kv_cache_manager_hook(
        kv_cache_manager_cls=FakeKVCacheManager,
    )

    manager = FakeKVCacheManager()
    result = manager.get_computed_blocks(FakeRequest())

    assert result is manager.computed_result
    assert result[1] == 0
    assert not any("prefix-cache lookup diagnostics" in record.getMessage() for record in caplog.records)


def test_pd_source_capture_ignores_scheduler_preemption_free(
    monkeypatch,
    caplog,
):
    """P请求被抢占时不应把不完整block table误判为最终export结果。"""

    captures = []

    class FakeKVCacheManager:
        def __init__(self):
            self.free_calls = []
            self.events = []

        def take_events(self):
            return list(self.events)

        def free(self, request):
            self.free_calls.append(request.request_id)

    class FakeScheduler:
        def __init__(self, manager):
            self.kv_cache_manager = manager

        def _preempt_request(self, request):
            self.kv_cache_manager.free(request)

    request = types.SimpleNamespace(
        request_id="prefill-preempted-1-8a773759",
        num_computed_tokens=1408,
    )
    intent = integration._PDSourceExportIntent(
        handoff_id="pd-prefill-preempted-1",
        request_id="prefill-preempted-1",
        prompt_token_count=6300,
        ttl_ms=30_000,
        created_at=time.monotonic(),
    )
    monkeypatch.setattr(
        integration,
        "_PD_SOURCE_EXPORT_INTENTS",
        {"prefill-preempted-1": intent},
    )
    monkeypatch.setattr(integration, "_PD_PREEMPTING_REQUEST_IDS", {})
    monkeypatch.setattr(
        integration,
        "_capture_pd_source_request_blocks",
        lambda _manager, captured_request, _log: captures.append(captured_request.request_id),
    )
    monkeypatch.setattr(integration, "_record_kv_cache_manager", lambda _manager: None)
    monkeypatch.setattr(
        integration,
        "_install_coordinator_cache_blocks_hook",
        lambda _manager, _log: True,
    )
    monkeypatch.setattr(
        integration,
        "_start_engine_core_reservation_server",
        lambda _log: None,
    )

    assert install_vllm_ascend_018_kv_cache_manager_hook(
        kv_cache_manager_cls=FakeKVCacheManager,
    )
    assert integration._install_scheduler_preemption_hook(
        FakeScheduler,
        integration.logger,
    )
    manager = FakeKVCacheManager()
    scheduler = FakeScheduler(manager)

    with caplog.at_level("INFO", logger="jiuwen.kvbm.vllm_ascend_018"):
        scheduler._preempt_request(request)

    assert manager.free_calls == [request.request_id]
    assert captures == []
    assert intent.capture_complete is False
    assert intent.preemption_count == 1
    assert intent.last_preempted_computed_tokens == 1408
    assert integration._PD_PREEMPTING_REQUEST_IDS == {}
    assert any("deferred P export capture across vLLM preemption" in record.getMessage() for record in caplog.records)

    manager.free(request)

    assert manager.free_calls == [request.request_id, request.request_id]
    assert captures == [request.request_id]


def test_engine_core_runtime_event_outbox_converts_move_events():
    """engine-core outbox 应把 KVBM move 事件转换为 router 可消费的 KV 事件。"""
    _drain_runtime_event_outbox()

    _enqueue_runtime_events(
        [
            KVBMRuntimeEvent(
                event_type="move",
                block_hashes=["hash-1"],
                location="disk",
                token_count=16,
                engine_specific={"from": "ram", "to": "disk"},
            )
        ]
    )

    events = _drain_runtime_event_outbox()

    assert events == [
        {
            "type": "BlockMoved",
            "data": {
                "block_hashes": ["hash-1"],
                "block_size": 16,
                "medium": "disk",
                "to": "disk",
                "from": "ram",
                "token_count": 16,
            },
        }
    ]
    assert _drain_runtime_event_outbox() == []


class FakeCache:
    def __init__(self, blocks):
        self.blocks = [FakeBlock(block) for block in blocks]
        self.shape = (len(self.blocks),) + self.blocks[0].shape
        self.dtype = self.blocks[0].dtype
        self.device = self.blocks[0].device

    def __getitem__(self, index):
        return self.blocks[index]


def _kv_caches():
    return {
        "layer.0": (
            FakeCache([[1, 2, 3, 4], [5, 6, 7, 8]]),
            FakeCache([[9, 10, 11, 12], [13, 14, 15, 16]]),
        )
    }


class FakeKVCacheRunner:
    """Keep initialized tensors and configuration on the simulated runner."""

    def __init__(self):
        self.kv_cache_config = None
        self.kv_caches = {}

    def initialize_kv_cache_tensors(self, kv_cache_config):
        self.kv_cache_config = kv_cache_config
        self.kv_caches = _kv_caches()
        return self.kv_caches


def test_fake_kv_cache_runner_keeps_configuration_and_tensors_per_instance():
    first, second = FakeKVCacheRunner(), FakeKVCacheRunner()
    first_config, second_config = object(), object()
    first_cache = first.initialize_kv_cache_tensors(first_config)
    second_cache = second.initialize_kv_cache_tensors(second_config)
    assert first.kv_cache_config is first_config
    assert second.kv_cache_config is second_config
    assert first.kv_caches is first_cache
    assert second.kv_caches is second_cache
    assert first_cache is not second_cache


def test_vllm_ascend_018_hook_binds_provider_and_diagnostics():
    """engine-core hook 应在 KV cache 初始化后绑定 provider 并记录诊断。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    try:
        installed = install_vllm_ascend_018_engine_core_hook(
            diagnostics=True,
            model_runner_cls=FakeModelRunner,
        )

        assert installed is True
        runner = FakeModelRunner()
        kv_caches = runner.initialize_kv_cache_tensors(object())

        provider = get_bound_vllm_ascend_018_provider(runner)
        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        resolver = get_bound_vllm_ascend_018_physical_block_resolver(runner)
        assert provider is not None
        assert dataplane is not None
        assert resolver is not None
        assert provider.kv_caches is kv_caches
        assert runner._jiuwen_kvbm_diagnostics["key_device"] == "npu"
        assert runner._jiuwen_kvbm_diagnostics["num_blocks"] == 2
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_hook_is_idempotent():
    """重复安装 hook 不应多次包裹 initialize_kv_cache_tensors。"""

    class FakeModelRunner:
        calls = 0

        def initialize_kv_cache_tensors(self, kv_cache_config):
            self.calls += 1
            return _kv_caches()

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)

        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        assert runner.calls == 1
        assert get_bound_vllm_ascend_018_provider(runner) is not None
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_dataplane_can_copy_block_to_ram_and_disk(tmp_path, monkeypatch):
    """绑定的数据面应能把真实 provider block 复制到 RAM/磁盘后端。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        assert dataplane is not None

        store_event = dataplane.register_physical_block(1, "block-1")
        ram_event = dataplane.offload_block_to_ram(1)
        disk_event = dataplane.offload_block_to_disk(1)
        stats = dataplane.stats()

        assert store_event.location == "npu"
        assert ram_event.location == "ram"
        assert disk_event.location == "disk"
        assert stats["disk_blocks"] == 1
        assert stats["disk_bytes"] > 0
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_dataplane_spills_ram_by_payload_nbytes(tmp_path, monkeypatch):
    """RAM 水位应按完整 VllmAscend018BlockPayload.nbytes 触发磁盘 spill。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_NPU_HIGH_WATERMARK_BLOCKS", "0")
    monkeypatch.setenv("JIUWEN_KVBM_RAM_HIGH_WATERMARK_BYTES", "32")
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        assert dataplane is not None
        dataplane.register_physical_blocks()

        events = dataplane.enforce_watermarks()
        stats = dataplane.stats()

        assert [event.location for event in events] == ["ram", "ram", "disk"]
        assert stats["ram_blocks"] == 1
        assert stats["disk_blocks"] == 1
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_dataplane_applies_block_stored_event(tmp_path, monkeypatch):
    """数据面应以 tensor shape 修正配置，并在 store relay 保留 token 链。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        events = dataplane.apply_kv_event(
            {
                "type": "BlockStored",
                "data": {
                    "block_ids": [1],
                    "block_hashes": ["hash-1"],
                    "block_size": 128,
                    "token_ids": [10, 11, 12, 13],
                    "parent_block_hash": "hash-0",
                },
            }
        )

        assert len(events) == 1
        assert events[0].event_type == "store"
        assert dataplane.config.block_size == 4
        assert os.environ["JIUWEN_KVBM_BLOCK_SIZE"] == "4"
        assert events[0].token_count == 4
        assert events[0].engine_specific["block_size"] == 4
        assert events[0].engine_specific["token_ids"] == [10, 11, 12, 13]
        assert events[0].engine_specific["parent_block_hash"] == "hash-0"
        assert dataplane.stats()["cached_npu_blocks"] == 1
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_dataplane_batches_store_and_preserves_block_tokens(
    tmp_path,
    monkeypatch,
):
    """批量 store 注册应保留每个 block 的 token 与 parent 链。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())
        dataplane = get_bound_vllm_ascend_018_dataplane(runner)

        events = dataplane.apply_kv_event(
            {
                "type": "BlockStored",
                "data": {
                    "block_ids": [0, 1],
                    "block_hashes": ["hash-0", "hash-1"],
                    "token_ids": list(range(8)),
                    "parent_block_hash": "root-hash",
                },
            }
        )
        diagnostics = dataplane.consume_kv_event_runtime_diagnostics()

        assert [event.engine_specific["token_ids"] for event in events] == [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
        assert events[0].engine_specific["parent_block_hash"] == "root-hash"
        assert events[1].engine_specific["parent_block_hash"] == "hash-0"
        assert diagnostics["registered_blocks"] == 2
        assert diagnostics["unchanged_blocks"] == 0
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_dataplane_ignores_hash_only_event(tmp_path, monkeypatch):
    """只有 block hash 而没有物理 block id 的事件不应触发真实 tensor 读写。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        result = dataplane.apply_kv_event(
            {
                "type": "BlockStored",
                "data": {
                    "block_hashes": ["hash-only"],
                    "block_size": 128,
                },
            }
        )

        assert result is None
        assert dataplane.stats()["total_blocks"] == 0
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_dataplane_resolves_hash_only_event_from_block_table(tmp_path, monkeypatch):
    """数据面应能从 engine-core block table 解析 hash 对应的物理 block id。"""

    class FakeModelRunner(FakeKVCacheRunner):
        def __init__(self):
            super().__init__()
            self.kv_cache_manager = {
                "block_table": {
                    "hash-from-table": {"physical_block_id": 1},
                }
            }

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        resolver = get_bound_vllm_ascend_018_physical_block_resolver(runner)
        assert resolver.resolve_block_hashes(["hash-from-table"]) == [1]
        assert resolver.diagnostics()["mapped_hashes"] >= 1

        events = dataplane.apply_kv_event(
            {
                "type": "BlockStored",
                "data": {
                    "block_hashes": ["hash-from-table"],
                    "block_size": 128,
                },
            }
        )

        assert len(events) == 1
        assert events[0].event_type == "store"
        assert dataplane.stats()["cached_npu_blocks"] == 1
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_publisher_hook_consumes_hash_only_events(tmp_path, monkeypatch):
    """engine-core publisher hook 应在事件发布前把 hash-only 事件交给 KVBM 数据面。"""

    class FakeModelRunner(FakeKVCacheRunner):
        def __init__(self):
            super().__init__()
            self.kv_cache_manager = {
                "block_table": {
                    "hash-from-publisher": {"physical_block_id": 1},
                }
            }

    class FakePublisher:
        def __init__(self):
            self.published = []

        def publish(self, events):
            self.published.append(events)

    class BlockStored:
        def __init__(self):
            self.block_hashes = ["hash-from-publisher"]
            self.parent_block_hash = None
            self.token_ids = [1, 2, 3, 4]
            self.block_size = 128
            self.medium = "GPU"

    class EventBatch:
        def __init__(self):
            self.events = [BlockStored()]

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        assert install_vllm_ascend_018_kv_event_publisher_hook(publisher_cls=FakePublisher)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        publisher = FakePublisher()
        batch = EventBatch()
        publisher.publish(batch)
        dataplane = get_bound_vllm_ascend_018_dataplane(runner)

        assert publisher.published == [batch]
        assert dataplane.stats()["cached_npu_blocks"] == 1
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_resolver_matches_vllm_packed_block_hash(tmp_path, monkeypatch):
    """resolver 应能把 vLLM 内部 packed hash 匹配到外部事件 hash。"""

    external_hash = b"0123456789abcdef0123456789abcdef"
    packed_hash = external_hash + (0).to_bytes(4, "big")

    class FakeKVCacheBlock:
        def __init__(self):
            self.block_id = 1
            self._block_hash = packed_hash

        @property
        def block_hash(self):
            return self._block_hash

    class FakeModelRunner(FakeKVCacheRunner):
        def __init__(self):
            super().__init__()
            self.kv_cache_manager = {
                "block_pool": {
                    "cached_block_hash_to_block": {
                        "_cache": {
                            packed_hash: FakeKVCacheBlock(),
                        }
                    }
                }
            }

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        resolver = get_bound_vllm_ascend_018_physical_block_resolver(runner)

        assert resolver.resolve_block_hashes([external_hash]) == [1]
        events = dataplane.apply_kv_event(
            {
                "type": "BlockStored",
                "data": {
                    "block_hashes": [external_hash],
                    "block_size": 128,
                },
            }
        )

        assert len(events) == 1
        assert events[0].event_type == "store"
        assert dataplane.stats()["cached_npu_blocks"] == 1
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_block_pool_hook_records_physical_mapping(tmp_path, monkeypatch):
    """BlockPool hook 应精确记录 vLLM 插入的 hash 到物理 block id 映射。"""

    external_hash = b"abcdef0123456789abcdef0123456789"
    packed_hash = external_hash + (0).to_bytes(4, "big")

    class FakeModelRunner(FakeKVCacheRunner):
        def __init__(self):
            super().__init__()
            self.cache_config = {
                "pin_memory": True,
                "max_model_len": 4096,
            }

    class FakeKVCacheBlock:
        block_id = 1

    class FakeBlockPool:
        def __init__(self):
            self.cached_blocks = {}

        def _insert_block_hash(self, block_hash_with_group_id, block, num_tokens=None):
            self.cached_blocks[block_hash_with_group_id] = block
            return "inserted"

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        assert install_vllm_ascend_018_block_pool_hook(block_pool_cls=FakeBlockPool)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        block_pool = FakeBlockPool()
        assert block_pool._insert_block_hash(packed_hash, FakeKVCacheBlock(), num_tokens=128) == "inserted"
        assert block_pool.cached_blocks[packed_hash].block_id == 1

        resolver = get_bound_vllm_ascend_018_physical_block_resolver(runner)
        diagnostics = resolver.diagnostics()

        assert resolver.resolve_block_hashes([external_hash]) == [1]
        assert resolver.resolve_block_hashes(["pin_memory"]) == []
        assert resolver.resolve_block_hashes(["max_model_len"]) == []
        assert diagnostics["observed_mapped_hashes"] > 0
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_block_pool_hook_records_cache_full_blocks_mapping(
    tmp_path,
    monkeypatch,
):
    """cache_full_blocks 应在 store 事件前增量记录最终 block hash。"""

    external_hash = b"cache-full-hash-0000000000000001"
    packed_hash = external_hash + (0).to_bytes(4, "big")

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    class FakeKVCacheBlock:
        def __init__(self):
            self.block_id = 2
            self._block_hash = None

        @property
        def block_hash(self):
            return self._block_hash

    class FakeBlockPool:
        def __init__(self):
            self.cached_blocks = {}

        def _insert_block_hash(self, block_hash_with_group_id, block, num_tokens=None):
            block._block_hash = block_hash_with_group_id
            self.cached_blocks[block_hash_with_group_id] = block

        def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks):
            for block in blocks[num_cached_blocks:num_full_blocks]:
                self._insert_block_hash(packed_hash, block)
            return num_full_blocks

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        assert install_vllm_ascend_018_block_pool_hook(block_pool_cls=FakeBlockPool)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())
        block = FakeKVCacheBlock()
        pool = FakeBlockPool()

        assert pool.cache_full_blocks(object(), [block], 0, 1) == 1
        assert pool.cached_blocks[packed_hash] is block
        resolver = get_bound_vllm_ascend_018_physical_block_resolver(runner)
        assert resolver.resolve_block_hashes([external_hash]) == [2]
        assert resolver.diagnostics()["cache_full_mappings"] == 1
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_cache_full_mapping_walker_supports_keyword_blocks():
    """增量捕获应兼容关键字参数与嵌套 block 容器。"""

    class FakeKVCacheBlock:
        block_id = 7
        block_hash = b"keyword-cache-full-hash"

    captured = []
    original_record = integration._record_engine_core_physical_mapping
    try:

        def record_physical_mapping(block_hash, block_id):
            return captured.append((block_hash, block_id))

        integration._record_engine_core_physical_mapping = record_physical_mapping
        recorded = integration._record_cache_full_block_mappings(
            (),
            {"blocks": [[FakeKVCacheBlock()]]},
        )
    finally:
        integration._record_engine_core_physical_mapping = original_record

    assert recorded == 1
    assert captured == [(b"keyword-cache-full-hash", 7)]


def test_vllm_ascend_018_coordinator_hook_records_request_block_mappings(
    tmp_path,
    monkeypatch,
):
    """coordinator.cache_blocks 应捕获绕过 BlockPool hook 的映射。"""

    external_hashes = [
        b"coordinator-hash-000000000000001",
        b"coordinator-hash-000000000000002",
        b"coordinator-hash-000000000000003",
    ]

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    class RequestBlock:
        def __init__(self, block_id):
            self.block_id = block_id
            self.block_hash = None

    class FakeRequest:
        request_id = "coordinator-request"
        block_hashes = external_hashes
        prompt_token_ids = list(range(12))

    class FakeCoordinator:
        def __init__(self):
            self._groups = [[RequestBlock(4), RequestBlock(5), RequestBlock(6)]]
            self.get_blocks_count = 0

        def cache_blocks(self, request, num_tokens_to_cache):
            for index, block in enumerate(self._groups[0]):
                block.block_hash = external_hashes[index] + (0).to_bytes(4, "big")
            return num_tokens_to_cache

        def get_blocks(self, request_id):
            assert request_id == "coordinator-request"
            self.get_blocks_count += 1
            return self._groups

    class FakeKVCacheManager:
        def __init__(self):
            self.coordinator = FakeCoordinator()
            self.block_size = 4
            self.events = []

        def take_events(self):
            return list(self.events)

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        assert install_vllm_ascend_018_kv_cache_manager_hook(kv_cache_manager_cls=FakeKVCacheManager)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())
        manager = FakeKVCacheManager()

        assert manager.coordinator.cache_blocks(FakeRequest(), 8) == 8
        resolver = get_bound_vllm_ascend_018_physical_block_resolver(runner)
        assert resolver.resolve_block_hashes(external_hashes[:2]) == [4, 5]
        assert resolver.diagnostics()["coordinator_cache_mappings"] == 2
        first_timing = resolver.consume_coordinator_cache_mapping_timing()
        assert first_timing["visited_blocks"] == 2
        assert manager.coordinator.get_blocks_count == 1

        assert manager.coordinator.cache_blocks(FakeRequest(), 8) == 8
        second_timing = resolver.consume_coordinator_cache_mapping_timing()
        assert second_timing["visited_blocks"] == 0
        assert second_timing["skipped_blocks"] == 2
        assert manager.coordinator.get_blocks_count == 1

        assert manager.coordinator.cache_blocks(FakeRequest(), 12) == 12
        third_timing = resolver.consume_coordinator_cache_mapping_timing()
        assert third_timing["visited_blocks"] == 1
        assert third_timing["skipped_blocks"] == 2
        assert resolver.resolve_block_hashes(external_hashes) == [4, 5, 6]
        assert manager.coordinator.get_blocks_count == 2
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_block_pool_hook_exposes_safe_reservation_api(tmp_path, monkeypatch):
    """BlockPool hook 应记录实例，并让 dataplane 通过原生引用计数安全预留 block。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    class RequestBlock:
        def __init__(self, block_id):
            self.block_id = block_id
            self.ref_cnt = 0

    class FakeBlockPool:
        def __init__(self):
            self.free = [RequestBlock(7), RequestBlock(8)]
            self.cached_blocks = {}

        def _insert_block_hash(self, block_hash_with_group_id, block, num_tokens=None):
            self.cached_blocks[block_hash_with_group_id] = block

        def get_num_free_blocks(self):
            return len(self.free)

        def get_new_blocks(self, count):
            blocks = self.free[:count]
            del self.free[:count]
            for block in blocks:
                block.ref_cnt += 1
            return blocks

        def free_blocks(self, blocks):
            for block in blocks:
                block.ref_cnt -= 1
                self.free.append(block)

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        assert install_vllm_ascend_018_block_pool_hook(block_pool_cls=FakeBlockPool)
        pool = FakeBlockPool()
        initial_generation = getattr(
            pool,
            integration._BLOCK_POOL_CAPACITY_GENERATION_ATTR,
        )
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())
        dataplane = get_bound_vllm_ascend_018_dataplane(runner)

        reservation = dataplane.block_reservations.reserve(2)
        assert reservation.block_ids == (7, 8)
        assert pool.get_num_free_blocks() == 0
        allocated_generation = getattr(
            pool,
            integration._BLOCK_POOL_CAPACITY_GENERATION_ATTR,
        )
        assert allocated_generation > initial_generation

        dataplane.block_reservations.rollback(reservation.reservation_id)
        assert pool.get_num_free_blocks() == 2
        assert (
            getattr(
                pool,
                integration._BLOCK_POOL_CAPACITY_GENERATION_ATTR,
            )
            > allocated_generation
        )
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_kv_cache_manager_hook_syncs_block_pool_snapshot(tmp_path, monkeypatch):
    """KVCacheManager hook 应从 block_pool 快照同步 hash 到物理 block id 映射。"""

    external_hash = b"fedcba9876543210fedcba9876543210"
    packed_hash = external_hash + (0).to_bytes(4, "big")

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    class FakeKVCacheBlock:
        def __init__(self):
            self.block_id = 1
            self.block_hash = packed_hash

    class FakeHashMap:
        def __init__(self):
            self._cache = {packed_hash: FakeKVCacheBlock()}

    class FakeBlockPool:
        def __init__(self):
            self.cached_block_hash_to_block = FakeHashMap()

    class FakeKVCacheManager:
        def __init__(self):
            self.block_pool = FakeBlockPool()
            self.events = [BlockStored()]

        def take_events(self):
            return list(self.events)

    class BlockStored:
        def __init__(self):
            self.block_hashes = [external_hash]
            self.parent_block_hash = None
            self.token_ids = [1, 2, 3, 4]
            self.block_size = 128
            self.medium = "GPU"

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        assert install_vllm_ascend_018_kv_cache_manager_hook(kv_cache_manager_cls=FakeKVCacheManager)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        manager = FakeKVCacheManager()
        events = manager.take_events()
        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        resolver = get_bound_vllm_ascend_018_physical_block_resolver(runner)

        assert resolver.resolve_block_hashes([external_hash]) == [1]
        assert resolver.diagnostics()["observed_mapped_hashes"] > 0
        produced = dataplane.apply_kv_events(events)
        assert len(produced) == 1
        assert produced[0].event_type == "store"

        snapshot_calls = 0
        original_snapshot = integration._record_from_kv_cache_manager

        def counted_snapshot(observed_manager):
            nonlocal snapshot_calls
            snapshot_calls += 1
            return original_snapshot(observed_manager)

        monkeypatch.setattr(
            integration,
            "_record_from_kv_cache_manager",
            counted_snapshot,
        )
        manager.take_events()
        assert snapshot_calls == 0
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_resolver_uses_incremental_mapping_without_refresh():
    """增量映射命中时不应扫描 engine-core 对象树。"""

    resolver = integration.VllmAscend018PhysicalBlockResolver(root=object())
    block_hash = b"incremental-hash-0000000000000001"
    resolver.record_physical_block(block_hash, 7)

    def unexpected_refresh():
        raise AssertionError("增量映射命中时不应执行 refresh")

    resolver.refresh = unexpected_refresh
    assert resolver.resolve_block_hashes([block_hash]) == [7]


def test_vllm_ascend_018_kv_cache_manager_uses_targeted_packed_hash_lookup(
    tmp_path,
    monkeypatch,
):
    """P 节点新 hash 应定向查询 packed key，不遍历 BlockPool。"""

    external_hash = b"targeted-hash-000000000000000001"
    packed_hash = external_hash + (3).to_bytes(4, "big")

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    class FakeKVCacheBlock:
        def __init__(self):
            self.block_id = 9
            self.block_hash = packed_hash

    class FakeHashMap:
        def __init__(self):
            self._cache = {packed_hash: FakeKVCacheBlock()}

    class FakeBlockPool:
        def __init__(self):
            self.cached_block_hash_to_block = FakeHashMap()

    class FakeKVCacheManager:
        def __init__(self):
            self.block_pool = FakeBlockPool()
            self.events = [BlockStored()]

        def take_events(self):
            return list(self.events)

    class BlockStored:
        def __init__(self):
            self.block_hashes = [external_hash]
            self.group_idx = 3
            self.parent_block_hash = None
            self.token_ids = [1, 2, 3, 4]
            self.block_size = 128
            self.medium = "GPU"

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        assert install_vllm_ascend_018_kv_cache_manager_hook(kv_cache_manager_cls=FakeKVCacheManager)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())
        manager = FakeKVCacheManager()

        def unexpected_snapshot(observed_manager):
            raise AssertionError("定向 packed hash 命中时不应扫描 BlockPool")

        monkeypatch.setattr(
            integration,
            "_record_from_kv_cache_manager",
            unexpected_snapshot,
        )
        manager.take_events()
        resolver = get_bound_vllm_ascend_018_physical_block_resolver(runner)
        assert resolver.resolve_block_hashes([external_hash]) == [9]
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_removed_event_miss_skips_block_pool_snapshot(
    tmp_path,
    monkeypatch,
):
    """删除事件的 hash 未命中时不应扫描已删除的 BlockPool。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    class FakeHashMap:
        def __init__(self):
            self._cache = {}

    class FakeBlockPool:
        def __init__(self):
            self.cached_block_hash_to_block = FakeHashMap()

    class FakeKVCacheManager:
        def __init__(self):
            self.block_pool = FakeBlockPool()
            self.events = [BlockRemoved()]

        def take_events(self):
            return list(self.events)

    class BlockRemoved:
        def __init__(self):
            self.block_hashes = [b"already-removed-block-hash"]
            self.group_idx = 0

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        assert install_vllm_ascend_018_kv_cache_manager_hook(kv_cache_manager_cls=FakeKVCacheManager)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())
        manager = FakeKVCacheManager()

        def unexpected_snapshot(observed_manager):
            raise AssertionError("BlockRemoved 未命中时不应扫描 BlockPool")

        monkeypatch.setattr(
            integration,
            "_record_from_kv_cache_manager",
            unexpected_snapshot,
        )
        manager.take_events()
        resolver = get_bound_vllm_ascend_018_physical_block_resolver(runner)
        diagnostics = resolver.consume_fallback_diagnostics()
        assert diagnostics[-1]["fallback_reason"] == "non_store_snapshot_skipped"
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_packed_hash_candidates_include_both_byte_orders():
    """packed hash 候选应覆盖 group id 的大端和小端表示。"""

    external_hash = b"byte-order-hash"
    candidates = integration._packed_block_hash_candidates(external_hash, 3)

    assert external_hash + (3).to_bytes(4, "big") in candidates
    assert external_hash + (3).to_bytes(4, "little") in candidates


def test_vllm_ascend_018_engine_core_events_enforce_watermarks(tmp_path, monkeypatch):
    """真实 engine-core KV event 消费后应立即执行一次水位策略。"""

    external_hashes = [
        b"watermark-block-hash-000000000001",
        b"watermark-block-hash-000000000002",
    ]
    packed_hashes = [
        external_hashes[0] + (0).to_bytes(4, "big"),
        external_hashes[1] + (0).to_bytes(4, "big"),
    ]

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    class FakeKVCacheBlock:
        def __init__(self, block_id, block_hash):
            self.block_id = block_id
            self.block_hash = block_hash

    class FakeHashMap:
        def __init__(self):
            self._cache = {
                packed_hashes[0]: FakeKVCacheBlock(0, packed_hashes[0]),
                packed_hashes[1]: FakeKVCacheBlock(1, packed_hashes[1]),
            }

    class FakeBlockPool:
        def __init__(self):
            self.cached_block_hash_to_block = FakeHashMap()

    class FakeKVCacheManager:
        def __init__(self):
            self.block_pool = FakeBlockPool()
            self.events = [BlockStored()]

        def take_events(self):
            return list(self.events)

    class BlockStored:
        def __init__(self):
            self.block_hashes = external_hashes
            self.parent_block_hash = None
            self.token_ids = list(range(256))
            self.block_size = 128
            self.medium = "GPU"

    class EventBatch:
        def __init__(self, events):
            self.events = events

    class FakePublisher:
        def __init__(self):
            self.published = []

        def publish(self, events):
            self.published.append(events)

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")
    monkeypatch.setenv("JIUWEN_KVBM_NPU_HIGH_WATERMARK_BLOCKS", "1")
    monkeypatch.setenv("JIUWEN_KVBM_RAM_HIGH_WATERMARK_BYTES", "1")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        assert install_vllm_ascend_018_kv_cache_manager_hook(kv_cache_manager_cls=FakeKVCacheManager)
        assert install_vllm_ascend_018_kv_event_publisher_hook(publisher_cls=FakePublisher)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        manager = FakeKVCacheManager()
        batch = EventBatch(manager.take_events())
        publisher = FakePublisher()
        publisher.publish(batch)
        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        stats = dataplane.stats()

        assert publisher.published == [batch]
        assert stats["total_blocks"] == 2
        assert stats["cached_npu_blocks"] == 1
        assert stats["ram_blocks"] == 0
        assert stats["disk_blocks"] == 1
        assert stats["disk_bytes"] > 0
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_dataplane_prefetches_disk_blocks_to_npu(tmp_path, monkeypatch):
    """dataplane 应能按 block hash 批量把磁盘 block 预取回 NPU。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_NPU_HIGH_WATERMARK_BLOCKS", "1")
    monkeypatch.setenv("JIUWEN_KVBM_RAM_HIGH_WATERMARK_BYTES", "1")
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        dataplane.apply_kv_event(
            {
                "type": "BlockStored",
                "data": {
                    "physical_block_ids": [0, 1],
                    "block_hashes": ["prefetch-0", "prefetch-1"],
                    "block_size": 128,
                },
            }
        )
        dataplane.enforce_watermarks()
        offloaded_stats = dataplane.stats()
        assert offloaded_stats["disk_blocks"] == 1
        assert offloaded_stats["cached_npu_blocks"] == 1

        events = dataplane.prefetch_block_hashes(["prefetch-0", "missing"])
        stats = dataplane.stats()

        assert len(events) == 1
        assert events[0].event_type == "move"
        assert events[0].location == "npu"
        assert events[0].engine_specific["from"] == "disk"
        assert stats["disk_blocks"] == 0
        assert stats["cached_npu_blocks"] == 2
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_prefetch_deadline_expires_without_loading_ram(
    tmp_path,
    monkeypatch,
):
    """请求 deadline 已过时不应写回 NPU，也不应删除 RAM 源副本。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())
        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        dataplane.register_physical_block(0, "deadline-hash", active=False)
        dataplane.offload_block_to_ram(0)

        result = dataplane.prefetch_block_hashes_for_request(
            ["deadline-hash"],
            request_id="req-expired",
            deadline_monotonic=time.monotonic() - 0.001,
        )

        assert result["prefetched_count"] == 0
        assert result["expired_count"] == 1
        assert result["outcomes"] == {"deadline-hash": "expired_before_copy"}
        assert dataplane.stats()["ram_blocks"] == 1
        assert dataplane.stats()["cached_npu_blocks"] == 0
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_prefetch_deduplicates_inflight_hash(
    tmp_path,
    monkeypatch,
):
    """同一 hash 已有 in-flight owner 时，新请求不应重复执行 payload 搬运。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())
        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        dataplane.register_physical_block(0, "shared-hash", active=False)
        dataplane.offload_block_to_ram(0)
        dataplane._prefetch_inflight["shared-hash"] = "req-owner"

        result = dataplane.prefetch_block_hashes_for_request(
            ["shared-hash", "shared-hash"],
            request_id="req-duplicate",
            deadline_monotonic=float("inf"),
        )

        assert result["prefetched_count"] == 0
        assert result["deduplicated_count"] == 2
        assert dataplane.stats()["ram_blocks"] == 1
        assert dataplane._prefetch_inflight["shared-hash"] == "req-owner"
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_prefetch_rolls_back_when_deadline_expires_during_copy(
    tmp_path,
    monkeypatch,
):
    """payload 写入期间超过 deadline 时不应发布 NPU location。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())
        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        dataplane.register_physical_block(0, "late-hash", active=False)
        dataplane.offload_block_to_ram(0)
        accessor = dataplane.runtime.manager.accessor
        original_write = accessor.write_block

        def _delayed_write(block_id, payload):
            original_write(block_id, payload)
            time.sleep(0.01)

        monkeypatch.setattr(accessor, "write_block", _delayed_write)
        result = dataplane.prefetch_block_hashes_for_request(
            ["late-hash"],
            request_id="req-late",
            deadline_monotonic=time.monotonic() + 0.002,
        )

        assert result["prefetched_count"] == 0
        assert result["expired_count"] == 1
        assert result["rolled_back_count"] == 1
        assert result["outcomes"] == {"late-hash": "rolled_back:deadline"}
        assert dataplane.stats()["ram_blocks"] == 1
        assert dataplane.stats()["cached_npu_blocks"] == 0
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_prefetch_rolls_back_when_request_publishes_same_hash(
    tmp_path,
    monkeypatch,
):
    """推理请求在 copy 期间发布同一 hash 时，预取应保留源副本并去重。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())
        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        dataplane.register_physical_block(0, "publish-race-hash", active=False)
        dataplane.offload_block_to_ram(0)
        accessor = dataplane.runtime.manager.accessor
        original_write = accessor.write_block

        def _write_then_publish_duplicate(block_id, payload):
            original_write(block_id, payload)
            dataplane.register_physical_block(
                1,
                "publish-race-hash",
                active=False,
            )

        monkeypatch.setattr(accessor, "write_block", _write_then_publish_duplicate)
        result = dataplane.prefetch_block_hashes_for_request(
            ["publish-race-hash"],
            request_id="req-publish-race",
            deadline_monotonic=float("inf"),
        )

        assert result["prefetched_count"] == 0
        assert result["rolled_back_count"] == 1
        assert result["outcomes"] == {"publish-race-hash": "rolled_back:already_published"}
        assert dataplane.stats()["ram_blocks"] == 1
        assert dataplane.stats()["cached_npu_blocks"] == 1
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_dataplane_applies_move_and_remove_events(tmp_path, monkeypatch):
    """数据面应能根据 KV move/remove 事件更新 RAM/Disk/evicted 状态。"""

    class FakeModelRunner(FakeKVCacheRunner):
        pass

    monkeypatch.setenv("JIUWEN_KVBM_WORKER_DISK_ROOT", str(tmp_path))
    monkeypatch.setenv("JIUWEN_KVBM_BLOCK_SIZE", "128")

    try:
        assert install_vllm_ascend_018_engine_core_hook(model_runner_cls=FakeModelRunner)
        runner = FakeModelRunner()
        runner.initialize_kv_cache_tensors(object())

        dataplane = get_bound_vllm_ascend_018_dataplane(runner)
        dataplane.apply_kv_event(
            {
                "type": "BlockStored",
                "data": {"physical_block_id": 1, "block_hash": "hash-1", "block_size": 128},
            }
        )
        move_events = dataplane.apply_kv_event(
            {
                "type": "BlockMoved",
                "data": {"physical_block_id": 1, "block_hash": "hash-1", "to": "ram"},
            }
        )
        stats = dataplane.stats()

        assert len(move_events) == 1
        assert move_events[0].location == "ram"
        assert stats["ram_blocks"] == 1

        removed = dataplane.apply_kv_event(
            {
                "type": "BlockRemoved",
                "data": {"physical_block_id": 1, "block_hash": "hash-1"},
            }
        )

        assert len(removed) == 1
        assert dataplane.stats()["total_blocks"] == 1
        assert dataplane.runtime.manager.stats().evicted_blocks == 1
    finally:
        uninstall_vllm_ascend_018_engine_core_hook(FakeModelRunner)


def test_vllm_ascend_018_multiprocess_hook_replaces_engine_core_entrypoint(monkeypatch):
    """多进程 hook 应替换 EngineCoreProc.run_engine_core 为可被 spawn 导入的顶层函数。"""

    class EngineCoreProc:
        @staticmethod
        def run_engine_core(*args, **kwargs):
            return None

    vllm_module = types.ModuleType("vllm")
    vllm_v1_module = types.ModuleType("vllm.v1")
    engine_module = types.ModuleType("vllm.v1.engine")
    core_module = types.ModuleType("vllm.v1.engine.core")
    core_module.EngineCoreProc = EngineCoreProc

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core_module)
    monkeypatch.delenv("JIUWEN_KVBM_VLLM_ASCEND_018_DIAGNOSTICS", raising=False)

    installed = install_vllm_ascend_018_multiprocess_hooks(
        diagnostics=True,
        worker_id="worker-1",
        disk_root="/tmp/kvbm-test",
        clean_disk_on_start=False,
        npu_high_watermark_blocks=8,
        ram_high_watermark_bytes=1024,
        block_size=128,
        auto_register_physical_blocks=True,
        auto_enforce_watermarks_on_init=True,
        transfer_max_inflight=3,
        transfer_retention_seconds=600.0,
        benchmark_enabled=True,
        benchmark_tcp_host="10.0.0.7",
        benchmark_tcp_port=29007,
        benchmark_max_blocks=8,
    )

    assert installed is True
    assert EngineCoreProc.run_engine_core is jiuwen_kvbm_vllm_ascend_018_run_engine_core
    assert os.environ["JIUWEN_KVBM_VLLM_ASCEND_018_DIAGNOSTICS"] == "1"
    assert os.environ["JIUWEN_KVBM_WORKER_ID"] == "worker-1"
    assert os.environ["JIUWEN_KVBM_WORKER_DISK_ROOT"] == "/tmp/kvbm-test"
    assert os.environ["JIUWEN_KVBM_CLEAN_DISK_ON_START"] == "0"
    assert os.environ["JIUWEN_KVBM_NPU_HIGH_WATERMARK_BLOCKS"] == "8"
    assert os.environ["JIUWEN_KVBM_RAM_HIGH_WATERMARK_BYTES"] == "1024"
    assert os.environ["JIUWEN_KVBM_BLOCK_SIZE"] == "128"
    assert os.environ["JIUWEN_KVBM_AUTO_REGISTER_PHYSICAL_BLOCKS"] == "1"
    assert os.environ["JIUWEN_KVBM_AUTO_ENFORCE_WATERMARKS_ON_INIT"] == "1"
    assert os.environ["JIUWEN_KVBM_TRANSFER_MAX_INFLIGHT"] == "3"
    assert os.environ["JIUWEN_KVBM_TRANSFER_RETENTION_SECONDS"] == "600.0"
    assert os.environ["JIUWEN_KVBM_BENCHMARK_ENABLED"] == "1"
    assert os.environ["JIUWEN_KVBM_BENCHMARK_TCP_HOST"] == "10.0.0.7"
    assert os.environ["JIUWEN_KVBM_BENCHMARK_TCP_PORT"] == "29007"
    assert os.environ["JIUWEN_KVBM_BENCHMARK_MAX_BLOCKS"] == "8"
    assert os.environ["JIUWEN_KVBM_PD_BLOCK_OWNERSHIP"] == "scheduler"


def test_vllm_ascend_018_scheduler_ownership_can_be_enabled(monkeypatch):
    """Scheduler原生bridge可启用，并应把模式传入engine-core子进程。"""

    monkeypatch.delenv("JIUWEN_KVBM_PD_BLOCK_OWNERSHIP", raising=False)

    install_vllm_ascend_018_multiprocess_hooks(
        pd_block_ownership="scheduler",
    )

    assert os.environ["JIUWEN_KVBM_PD_BLOCK_OWNERSHIP"] == "scheduler"


def test_runtime_relay_event_preserves_physical_block_id():
    """KVBM location relay 不应丢失 Router 跨 Worker 传输所需的物理 id。"""
    relay = _runtime_event_to_relay_event(
        {
            "event_type": "store",
            "location": "npu",
            "block_hashes": ["hash-7"],
            "token_count": 16,
            "engine_specific": {
                "to": "npu",
                "physical_block_id": 7,
            },
        }
    )

    assert relay is not None
    assert relay["data"]["physical_block_id"] == 7


def test_runtime_relay_event_preserves_batched_physical_block_ids():
    """聚合 store relay 应保持 block hash 与物理 ID 列表顺序对齐。"""

    relay = _runtime_event_to_relay_event(
        {
            "event_type": "store",
            "location": "npu",
            "block_hashes": ["hash-7", "hash-8"],
            "token_count": 16,
            "engine_specific": {
                "to": "npu",
                "physical_block_ids": [7, 8],
            },
        }
    )

    assert relay is not None
    assert relay["data"]["block_hashes"] == ["hash-7", "hash-8"]
    assert relay["data"]["physical_block_ids"] == [7, 8]


def test_local_block_pool_capacity_confirms_only_contiguous_prefix(monkeypatch):
    """D容量查询只能扣减第一个缺失位置之前的连续前缀。"""

    external_hashes = [value.to_bytes(16, "big") for value in (100, 101, 102)]
    packed_hashes = [value + (0).to_bytes(4, "big") for value in external_hashes]
    request_hashes = [str(value) for value in (100, 101, 102)]

    class FakeKVCacheBlock:
        def __init__(self, block_id, block_hash):
            self.block_id = block_id
            self.block_hash = block_hash

    class FakeBlockPool:
        cached_block_hash_to_block = {
            packed_hashes[0]: FakeKVCacheBlock(10, packed_hashes[0]),
            packed_hashes[2]: FakeKVCacheBlock(12, packed_hashes[2]),
        }

        @staticmethod
        def get_num_free_blocks():
            return 7

    monkeypatch.setattr(
        integration,
        "_latest_reservable_block_pool",
        lambda: FakeBlockPool(),
    )

    result = integration._local_block_pool_capacity(
        {
            "block_hashes": request_hashes,
            "prompt_blocks": 4,
            "decode_blocks": 1,
        }
    )

    assert result == {
        "ok": True,
        "free_blocks": 7,
        "free_blocks_before_ticket": 7,
        "capacity_generation": -1,
        "confirmed_reusable_blocks": 1,
        "required_new_blocks": 4,
        "margin_blocks": 0,
        "outstanding_ticket_reserved_blocks": 0,
        "router_pending_reserved_blocks": 0,
        "committed_reserved_blocks": 0,
        "ticket_capacity_admitted": False,
        "queried_block_hashes": 3,
        "admission_ticket_id": "",
        "admission_ticket_pinned_blocks": 0,
    }


def test_block_pool_capacity_ticket_pins_prefix_until_release(monkeypatch):
    """容量确认ticket应固定连续命中，候选释放后归还引用。"""

    request_hash = (100).to_bytes(16, "big")
    packed_hash = request_hash + (0).to_bytes(4, "big")

    class FakeKVCacheBlock:
        def __init__(self):
            self.block_id = 10
            self.block_hash = packed_hash
            self.ref_cnt = 0

    block = FakeKVCacheBlock()

    class FakeBlockPool:
        cached_block_hash_to_block = {packed_hash: block}

        @staticmethod
        def get_num_free_blocks():
            return 7 - block.ref_cnt

        @staticmethod
        def touch(blocks):
            for item in blocks:
                item.ref_cnt += 1

        @staticmethod
        def free_blocks(blocks):
            for item in blocks:
                item.ref_cnt -= 1

    integration._PD_DECODE_ADMISSION_TICKETS.clear()
    monkeypatch.setattr(
        integration,
        "_latest_reservable_block_pool",
        lambda: FakeBlockPool(),
    )

    result = integration._local_block_pool_capacity(
        {
            "block_hashes": ["100"],
            "prompt_blocks": 1,
            "decode_blocks": 1,
            "request_id": "request-1",
            "handoff_id": "pd-request-1",
        }
    )

    assert result["admission_ticket_id"]
    assert result["admission_ticket_pinned_blocks"] == 1
    assert block.ref_cnt == 1
    released = integration._local_pd_decode_admission_ticket_release(
        {
            "handoff_id": "pd-request-1",
            "ticket_id": result["admission_ticket_id"],
        }
    )
    assert released["released"] is True
    assert block.ref_cnt == 0


def test_block_pool_capacity_rejects_ticket_that_overcommits_pending_delta(
    monkeypatch,
):
    """后续ticket不能挤占已承诺给前序D请求的缺失block容量。"""

    packed_hashes = [value.to_bytes(16, "big") + (0).to_bytes(4, "big") for value in (100, 200)]

    class FakeKVCacheBlock:
        def __init__(self, block_id, block_hash):
            self.block_id = block_id
            self.block_hash = block_hash
            self.ref_cnt = 0

    blocks = [FakeKVCacheBlock(index + 10, block_hash) for index, block_hash in enumerate(packed_hashes)]

    class FakeBlockPool:
        cached_block_hash_to_block = dict(zip(packed_hashes, blocks, strict=True))

        @staticmethod
        def get_num_free_blocks():
            return 8 - sum(block.ref_cnt > 0 for block in blocks)

        @staticmethod
        def touch(selected):
            for block in selected:
                block.ref_cnt += 1

        @staticmethod
        def free_blocks(selected):
            for block in selected:
                block.ref_cnt -= 1

    integration._PD_DECODE_ADMISSION_TICKETS.clear()
    pool = FakeBlockPool()
    monkeypatch.setattr(integration, "_latest_reservable_block_pool", lambda: pool)

    first = integration._local_block_pool_capacity(
        {
            "block_hashes": ["100"],
            "prompt_blocks": 2,
            "decode_blocks": 1,
            "request_id": "request-1",
            "handoff_id": "pd-request-1",
            "margin_blocks": 1,
        }
    )
    second = integration._local_block_pool_capacity(
        {
            "block_hashes": ["200"],
            "prompt_blocks": 5,
            "decode_blocks": 1,
            "request_id": "request-2",
            "handoff_id": "pd-request-2",
            "margin_blocks": 1,
            "pending_reserved_blocks": 4,
        }
    )

    assert first["ticket_capacity_admitted"] is True
    assert first["required_new_blocks"] == 2
    assert second["outstanding_ticket_reserved_blocks"] == 2
    assert second["router_pending_reserved_blocks"] == 4
    assert second["committed_reserved_blocks"] == 4
    assert second["ticket_capacity_admitted"] is False
    assert second["admission_ticket_id"] == ""
    assert blocks[0].ref_cnt == 1
    assert blocks[1].ref_cnt == 0
    integration._local_pd_decode_admission_ticket_release(
        {
            "handoff_id": "pd-request-1",
            "ticket_id": first["admission_ticket_id"],
        }
    )
