"""Low-overhead per-request timing diagnostics for native Mooncake decode."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable


logger = logging.getLogger("jiuwen.mooncake")

_ENTRYPOINT_PATCH_ATTR = "_jiuwen_mooncake_diagnostics_entrypoint_installed"
_ORIGINAL_ENTRYPOINT_ATTR = "_jiuwen_mooncake_original_engine_core_entrypoint"
_SCHEDULER_PATCH_ATTR = "_jiuwen_mooncake_diagnostics_scheduler_installed"
_RUNTIME_SCHEDULER_PATCH_ATTR = "_jiuwen_mooncake_diagnostics_runtime_scheduler_installed"
_RUNTIME_UPDATE_PATCH_ATTR = "_jiuwen_mooncake_diagnostics_runtime_update_installed"
_RUNTIME_XFER_PATCH_ATTR = "_jiuwen_mooncake_diagnostics_runtime_xfer_installed"
_CONNECTOR_PATCH_ATTR = "_jiuwen_mooncake_diagnostics_connector_installed"
_ORIGINAL_SCHEDULE_ATTR = "_jiuwen_mooncake_original_schedule"
_ORIGINAL_UPDATE_REMOTE_ATTR = "_jiuwen_mooncake_original_update_remote"
_ORIGINAL_ADD_REQUEST_ATTR = "_jiuwen_mooncake_original_add_request"
_ORIGINAL_XFER_FINISHED_ATTR = "_jiuwen_mooncake_original_xfer_finished"
_RUNTIME_ORIGINAL_SCHEDULE_ATTR = "_jiuwen_mooncake_runtime_original_schedule"
_RUNTIME_ORIGINAL_UPDATE_ATTR = "_jiuwen_mooncake_runtime_original_update"
_RUNTIME_ORIGINAL_XFER_ATTR = "_jiuwen_mooncake_runtime_original_xfer"
_EXPECTED_COMPUTED_TOKENS_ATTR = "_jiuwen_mooncake_expected_computed_tokens"
_MAX_RECORDS = 8192


@dataclass
class _RemoteKVTiming:
    request_id: str
    logical_request_id: str
    arrival_time: float
    wait_observed_at: float
    connector_ready_at: float = 0.0


_REMOTE_KV_TIMINGS: dict[str, _RemoteKVTiming] = {}
_CONNECTOR_MATCH_STATES: dict[str, tuple[Any, ...]] = {}


def _logical_request_id(request: Any) -> str:
    trace_headers = getattr(request, "trace_headers", None)
    if not isinstance(trace_headers, Mapping):
        return ""
    return str(trace_headers.get("x-jiuwen-request-id") or trace_headers.get("X-Jiuwen-Request-Id") or "")


def _queue_length(queue: Any) -> int:
    try:
        return len(queue)
    except (TypeError, AttributeError):
        return 0


def _scheduled_token_counts(output: Any) -> dict[str, int]:
    raw = getattr(output, "num_scheduled_tokens", None)
    if not isinstance(raw, Mapping):
        return {}
    return {str(request_id): int(token_count or 0) for request_id, token_count in raw.items()}


def _request_ids_from_scheduler_data(value: Any) -> set[str]:
    if value is None:
        return set()
    for attribute in ("req_ids", "request_ids"):
        request_ids = getattr(value, attribute, None)
        if isinstance(request_ids, (list, tuple, set)):
            return {str(request_id) for request_id in request_ids if request_id}
    request_id = getattr(value, "req_id", None) or getattr(
        value,
        "request_id",
        None,
    )
    if request_id:
        return {str(request_id)}
    if isinstance(value, (list, tuple, set)):
        request_ids: set[str] = set()
        for item in value:
            request_ids.update(_request_ids_from_scheduler_data(item))
        return request_ids
    return set()


def _scheduled_request_ids(output: Any) -> set[str]:
    request_ids = set(_scheduled_token_counts(output))
    for attribute in ("scheduled_new_reqs", "scheduled_cached_reqs"):
        request_ids.update(_request_ids_from_scheduler_data(getattr(output, attribute, None)))
    return request_ids


def _trim_records() -> None:
    while len(_REMOTE_KV_TIMINGS) > _MAX_RECORDS:
        _REMOTE_KV_TIMINGS.pop(next(iter(_REMOTE_KV_TIMINGS)), None)
    while len(_CONNECTOR_MATCH_STATES) > _MAX_RECORDS:
        _CONNECTOR_MATCH_STATES.pop(next(iter(_CONNECTOR_MATCH_STATES)), None)


def _new_remote_kv_timing(request: Any, observed_at: float) -> _RemoteKVTiming:
    return _RemoteKVTiming(
        request_id=str(getattr(request, "request_id", "") or ""),
        logical_request_id=_logical_request_id(request),
        arrival_time=float(getattr(request, "arrival_time", observed_at) or observed_at),
        wait_observed_at=observed_at,
    )


def _record_remote_waiting_requests(scheduler: Any, observed_at: float) -> None:
    requests = getattr(scheduler, "requests", None)
    if not isinstance(requests, Mapping):
        return
    for request in requests.values():
        status = getattr(request, "status", None)
        status_name = str(getattr(status, "name", status) or "")
        if status_name != "WAITING_FOR_REMOTE_KVS":
            continue
        request_id = str(getattr(request, "request_id", "") or "")
        expected_computed_tokens = int(getattr(request, _EXPECTED_COMPUTED_TOKENS_ATTR, 0) or 0)
        if expected_computed_tokens > 0 and int(getattr(request, "num_computed_tokens", 0) or 0) == 0:
            # vLLM-Ascend 0.18 allocates async remote-KV slots but omits the
            # upstream state assignment before returning to the next cycle.
            request.num_computed_tokens = expected_computed_tokens
            logger.info(
                "[MooncakeCompatibility] restored async remote computed tokens "
                "request=%s engine_request_id=%s num_computed_tokens=%s",
                _logical_request_id(request) or "unavailable",
                request_id,
                expected_computed_tokens,
            )
        if expected_computed_tokens > 0:
            delattr(request, _EXPECTED_COMPUTED_TOKENS_ATTR)
        if request_id and request_id not in _REMOTE_KV_TIMINGS:
            _REMOTE_KV_TIMINGS[request_id] = _new_remote_kv_timing(
                request,
                observed_at,
            )
            _trim_records()


def _scheduler_connector(scheduler: Any) -> tuple[str, Any] | tuple[None, None]:
    for attribute in ("connector", "kv_connector"):
        connector = getattr(scheduler, attribute, None)
        if connector is not None:
            return attribute, connector
    return None, None


def _install_connector_diagnostics_hook(
    scheduler: Any,
    hook_logger: logging.Logger,
) -> bool:
    connector_attribute, connector = _scheduler_connector(scheduler)
    if connector is None:
        return False
    if getattr(connector, _CONNECTOR_PATCH_ATTR, False):
        return True
    original = getattr(connector, "get_num_new_matched_tokens", None)
    if original is None:
        return False

    def _wrapped_get_num_new_matched_tokens(
        request: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original(request, *args, **kwargs)
        matched_tokens = result[0] if isinstance(result, tuple) and result else result
        try:
            has_remote_tokens = int(matched_tokens or 0) > 0
        except (TypeError, ValueError):
            has_remote_tokens = False
        waits_async = isinstance(result, tuple) and len(result) >= 2 and bool(result[1])
        if waits_async and has_remote_tokens:
            local_computed_tokens = (
                args[0]
                if args
                else kwargs.get(
                    "num_new_local_computed_tokens",
                    0,
                )
            )
            try:
                expected_computed_tokens = int(local_computed_tokens or 0) + int(matched_tokens or 0)
            except (TypeError, ValueError):
                expected_computed_tokens = 0
            if expected_computed_tokens > 0:
                setattr(
                    request,
                    _EXPECTED_COMPUTED_TOKENS_ATTR,
                    expected_computed_tokens,
                )
        request_id = str(getattr(request, "request_id", "") or "")
        params = getattr(request, "kv_transfer_params", None)
        params_mapping = params if isinstance(params, Mapping) else {}
        match_state = (
            bool(params_mapping),
            bool(params_mapping.get("do_remote_prefill")),
            bool(params_mapping.get("do_remote_decode")),
            matched_tokens,
            waits_async,
        )
        if request_id and _CONNECTOR_MATCH_STATES.get(request_id) != match_state:
            _CONNECTOR_MATCH_STATES[request_id] = match_state
            _trim_records()
            hook_logger.info(
                "[MooncakeDRequestTiming] connector match result "
                "request=%s engine_request_id=%s has_transfer_params=%s "
                "do_remote_prefill=%s do_remote_decode=%s "
                "matched_tokens=%s waits_async=%s connector_attribute=%s",
                _logical_request_id(request) or "unavailable",
                request_id,
                bool(params_mapping),
                bool(params_mapping.get("do_remote_prefill")),
                bool(params_mapping.get("do_remote_decode")),
                matched_tokens,
                waits_async,
                connector_attribute,
            )
        if has_remote_tokens or waits_async:
            if request_id and request_id not in _REMOTE_KV_TIMINGS:
                _REMOTE_KV_TIMINGS[request_id] = _new_remote_kv_timing(
                    request,
                    time.time(),
                )
                _trim_records()
                hook_logger.info(
                    "[MooncakeDRequestTiming] connector wait observed "
                    "request=%s engine_request_id=%s matched_tokens=%s "
                    "waits_async=%s connector_attribute=%s",
                    _REMOTE_KV_TIMINGS[request_id].logical_request_id or "unavailable",
                    request_id,
                    matched_tokens,
                    waits_async,
                    connector_attribute,
                )
        return result

    setattr(
        connector,
        "get_num_new_matched_tokens",
        _wrapped_get_num_new_matched_tokens,
    )
    setattr(connector, _CONNECTOR_PATCH_ATTR, True)
    hook_logger.info(
        "Mooncake connector diagnostics hook installed on %s attribute=%s",
        type(connector),
        connector_attribute,
    )
    return True


def _record_finished_recving(
    kv_connector_output: Any,
    ready_at: float,
    hook_logger: logging.Logger,
) -> None:
    request_ids = getattr(kv_connector_output, "finished_recving", None) or ()
    for request_id in request_ids:
        record = _REMOTE_KV_TIMINGS.get(str(request_id))
        if record is not None and record.connector_ready_at <= 0:
            record.connector_ready_at = ready_at
            hook_logger.info(
                "[MooncakeDRequestTiming] connector ready observed request=%s engine_request_id=%s",
                record.logical_request_id or "unavailable",
                record.request_id,
            )


def _record_kv_xfer_finished(
    original: Callable[..., Any],
    scheduler: Any,
    kv_connector_output: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    hook_logger: logging.Logger,
) -> Any:
    _record_finished_recving(kv_connector_output, time.time(), hook_logger)
    result = original(scheduler, kv_connector_output, *args, **kwargs)
    return result


def _record_remote_kv_update(
    original: Callable[..., Any],
    scheduler: Any,
    request: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    request_id = str(getattr(request, "request_id", "") or "")
    now = time.time()
    record = _REMOTE_KV_TIMINGS.get(request_id)
    if record is None:
        record = _new_remote_kv_timing(request, now)
        _REMOTE_KV_TIMINGS[request_id] = record
        _trim_records()
    result = original(scheduler, request, *args, **kwargs)
    # vLLM 0.18 invokes this method only after finished_recving is observed
    # and returns None. Older implementations may return an explicit bool.
    if result is not False and record.connector_ready_at <= 0:
        record.connector_ready_at = time.time()
    return result


def _log_first_scheduled_requests(
    scheduler: Any,
    output: Any,
    scheduled_at: float,
    hook_logger: logging.Logger,
) -> None:
    running_count = _queue_length(getattr(scheduler, "running", None))
    waiting_count = _queue_length(getattr(scheduler, "waiting", None))
    token_counts = _scheduled_token_counts(output)
    for request_id in _scheduled_request_ids(output):
        record = _REMOTE_KV_TIMINGS.pop(request_id, None)
        if record is None:
            continue
        ready_at = record.connector_ready_at or scheduled_at
        hook_logger.info(
            "[MooncakeDRequestTiming] scheduler phases "
            "request=%s engine_request_id=%s "
            "d_engine_arrival_to_connector_wait_ms=%.3f "
            "connector_kv_wait_ms=%.3f "
            "d_engine_arrival_to_connector_ready_ms=%.3f "
            "connector_ready_to_first_scheduled_ms=%.3f "
            "d_engine_arrival_to_first_scheduled_ms=%.3f "
            "first_scheduled_tokens=%s running_requests=%s waiting_requests=%s",
            record.logical_request_id or "unavailable",
            record.request_id,
            max(0.0, (record.wait_observed_at - record.arrival_time) * 1000.0),
            max(0.0, (ready_at - record.wait_observed_at) * 1000.0),
            max(0.0, (ready_at - record.arrival_time) * 1000.0),
            max(0.0, (scheduled_at - ready_at) * 1000.0),
            max(0.0, (scheduled_at - record.arrival_time) * 1000.0),
            token_counts.get(request_id, 0),
            running_count,
            waiting_count,
        )


def _install_runtime_scheduler_diagnostics_hook(
    scheduler_cls: type[Any],
    hook_logger: logging.Logger,
    *,
    base_scheduler_cls: type[Any],
) -> bool:
    installed = False
    schedule_owner = next(
        (candidate for candidate in scheduler_cls.__mro__ if "schedule" in candidate.__dict__),
        None,
    )
    if schedule_owner is not None and schedule_owner is not base_scheduler_cls:
        if schedule_owner.__dict__.get(_RUNTIME_SCHEDULER_PATCH_ATTR, False):
            installed = True
        else:
            original_schedule = schedule_owner.__dict__["schedule"]

            def _wrapped_runtime_schedule(
                self: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                _install_connector_diagnostics_hook(self, hook_logger)
                output = original_schedule(self, *args, **kwargs)
                scheduled_at = time.time()
                _record_remote_waiting_requests(self, scheduled_at)
                _log_first_scheduled_requests(
                    self,
                    output,
                    scheduled_at,
                    hook_logger,
                )
                return output

            setattr(schedule_owner, _RUNTIME_ORIGINAL_SCHEDULE_ATTR, original_schedule)
            setattr(schedule_owner, "schedule", _wrapped_runtime_schedule)
            setattr(schedule_owner, _RUNTIME_SCHEDULER_PATCH_ATTR, True)
            installed = True

    update_owner = next(
        (candidate for candidate in scheduler_cls.__mro__ if "_update_waiting_for_remote_kv" in candidate.__dict__),
        None,
    )
    if update_owner is not None and update_owner is not base_scheduler_cls:
        if update_owner.__dict__.get(_RUNTIME_UPDATE_PATCH_ATTR, False):
            installed = True
        else:
            original_update = update_owner.__dict__["_update_waiting_for_remote_kv"]

            def _wrapped_runtime_update(
                self: Any,
                request: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                return _record_remote_kv_update(
                    original_update,
                    self,
                    request,
                    args,
                    kwargs,
                )

            setattr(update_owner, _RUNTIME_ORIGINAL_UPDATE_ATTR, original_update)
            setattr(
                update_owner,
                "_update_waiting_for_remote_kv",
                _wrapped_runtime_update,
            )
            setattr(update_owner, _RUNTIME_UPDATE_PATCH_ATTR, True)
            installed = True

    xfer_owner = next(
        (candidate for candidate in scheduler_cls.__mro__ if "_update_from_kv_xfer_finished" in candidate.__dict__),
        None,
    )
    if xfer_owner is not None and xfer_owner is not base_scheduler_cls:
        if xfer_owner.__dict__.get(_RUNTIME_XFER_PATCH_ATTR, False):
            installed = True
        else:
            original_xfer = xfer_owner.__dict__["_update_from_kv_xfer_finished"]

            def _wrapped_runtime_xfer(
                self: Any,
                kv_connector_output: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                return _record_kv_xfer_finished(
                    original_xfer,
                    self,
                    kv_connector_output,
                    args,
                    kwargs,
                    hook_logger,
                )

            setattr(xfer_owner, _RUNTIME_ORIGINAL_XFER_ATTR, original_xfer)
            setattr(
                xfer_owner,
                "_update_from_kv_xfer_finished",
                _wrapped_runtime_xfer,
            )
            setattr(xfer_owner, _RUNTIME_XFER_PATCH_ATTR, True)
            installed = True

    if installed:
        hook_logger.info(
            "Mooncake runtime scheduler diagnostics hook installed on %s "
            "schedule_owner=%s update_owner=%s xfer_owner=%s",
            scheduler_cls,
            schedule_owner,
            update_owner,
            xfer_owner,
        )
    return installed


def install_mooncake_scheduler_diagnostics_hook(
    scheduler_cls: type[Any] | None = None,
    *,
    log: logging.Logger | None = None,
) -> bool:
    """Patch a vLLM Scheduler to trace async Mooncake receive phases."""

    hook_logger = log or logger
    if scheduler_cls is None:
        try:
            from vllm.v1.core.sched.scheduler import Scheduler

            scheduler_cls = Scheduler
        except Exception as exc:
            hook_logger.warning("Mooncake scheduler diagnostics skipped: %s", exc)
            return False

    if scheduler_cls.__dict__.get(_SCHEDULER_PATCH_ATTR, False):
        return True
    original_schedule = getattr(scheduler_cls, "schedule", None)
    original_update_remote = getattr(
        scheduler_cls,
        "_update_waiting_for_remote_kv",
        None,
    )
    original_add_request = getattr(scheduler_cls, "add_request", None)
    original_xfer_finished = getattr(
        scheduler_cls,
        "_update_from_kv_xfer_finished",
        None,
    )
    if original_schedule is None or original_update_remote is None:
        hook_logger.warning("Mooncake scheduler diagnostics skipped: required Scheduler methods missing")
        return False

    def _wrapped_update_remote(
        self: Any,
        request: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return _record_remote_kv_update(
            original_update_remote,
            self,
            request,
            args,
            kwargs,
        )

    def _wrapped_schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
        _install_connector_diagnostics_hook(self, hook_logger)
        output = original_schedule(self, *args, **kwargs)
        scheduled_at = time.time()
        _record_remote_waiting_requests(self, scheduled_at)
        _log_first_scheduled_requests(self, output, scheduled_at, hook_logger)
        return output

    def _wrapped_xfer_finished(
        self: Any,
        kv_connector_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not (original_xfer_finished is not None):
            raise RuntimeError("Required invariant failed: original_xfer_finished is not None")
        return _record_kv_xfer_finished(
            original_xfer_finished,
            self,
            kv_connector_output,
            args,
            kwargs,
            hook_logger,
        )

    def _wrapped_add_request(
        self: Any,
        request: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not (original_add_request is not None):
            raise RuntimeError("Required invariant failed: original_add_request is not None")
        # Hook the concrete subclass; isinstance would also match the base class.
        runtime_scheduler_cls = type(self)
        if runtime_scheduler_cls is not scheduler_cls:
            _install_runtime_scheduler_diagnostics_hook(
                runtime_scheduler_cls,
                hook_logger,
                base_scheduler_cls=scheduler_cls,
            )
        _install_connector_diagnostics_hook(self, hook_logger)
        return original_add_request(self, request, *args, **kwargs)

    setattr(scheduler_cls, _ORIGINAL_SCHEDULE_ATTR, original_schedule)
    setattr(scheduler_cls, _ORIGINAL_UPDATE_REMOTE_ATTR, original_update_remote)
    if original_xfer_finished is not None:
        setattr(
            scheduler_cls,
            _ORIGINAL_XFER_FINISHED_ATTR,
            original_xfer_finished,
        )
        setattr(
            scheduler_cls,
            "_update_from_kv_xfer_finished",
            _wrapped_xfer_finished,
        )
    if original_add_request is not None:
        setattr(scheduler_cls, _ORIGINAL_ADD_REQUEST_ATTR, original_add_request)
        setattr(scheduler_cls, "add_request", _wrapped_add_request)
    setattr(scheduler_cls, "schedule", _wrapped_schedule)
    setattr(
        scheduler_cls,
        "_update_waiting_for_remote_kv",
        _wrapped_update_remote,
    )
    setattr(scheduler_cls, _SCHEDULER_PATCH_ATTR, True)
    hook_logger.info("Mooncake scheduler diagnostics hook installed on %s", scheduler_cls)
    return True


def _is_mooncake_engine_core_entrypoint(entrypoint: Any) -> bool:
    """Return whether an entrypoint resolves back to this diagnostics wrapper."""

    if entrypoint is mooncake_diagnostics_run_engine_core:
        return True
    return getattr(entrypoint, "func", None) is mooncake_diagnostics_run_engine_core


def mooncake_diagnostics_run_engine_core(
    *args: Any,
    _original_run_engine_core: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Spawn-safe vLLM-Ascend EngineCore entrypoint wrapper."""

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    install_mooncake_scheduler_diagnostics_hook()
    from vllm_ascend.patch.platform import patch_balance_schedule

    ascend_original = getattr(
        patch_balance_schedule,
        "_ORIGINAL_RUN_ENGINE_CORE",
        None,
    )
    if _is_mooncake_engine_core_entrypoint(ascend_original):
        if _original_run_engine_core is None:
            raise RuntimeError("Mooncake diagnostics lost the original EngineCore entrypoint")
        # Restore the vLLM-Ascend spawn entrypoint required by its patch module.
        patch_balance_schedule._ORIGINAL_RUN_ENGINE_CORE = _original_run_engine_core  # pylint: disable=protected-access

    return patch_balance_schedule.run_engine_core(*args, **kwargs)


def install_mooncake_engine_core_diagnostics(
    *,
    log: logging.Logger | None = None,
) -> bool:
    """Install the spawn-safe EngineCore wrapper for native Mooncake D nodes."""

    hook_logger = log or logger
    try:
        import vllm.v1.engine.core as core_module
        from vllm.v1.engine.core import EngineCoreProc
    except Exception as exc:
        hook_logger.warning("Mooncake EngineCore diagnostics skipped: %s", exc)
        return False

    if getattr(EngineCoreProc, _ENTRYPOINT_PATCH_ATTR, False):
        return True
    original_entrypoint = EngineCoreProc.run_engine_core
    ascend_module = sys.modules.get("vllm_ascend.patch.platform.patch_balance_schedule")
    if ascend_module is not None and original_entrypoint is getattr(ascend_module, "run_engine_core", None):
        ascend_original = getattr(
            ascend_module,
            "_ORIGINAL_RUN_ENGINE_CORE",
            None,
        )
        if callable(ascend_original) and not _is_mooncake_engine_core_entrypoint(ascend_original):
            original_entrypoint = ascend_original
    wrapped_entrypoint = partial(
        mooncake_diagnostics_run_engine_core,
        _original_run_engine_core=original_entrypoint,
    )
    setattr(EngineCoreProc, _ORIGINAL_ENTRYPOINT_ATTR, original_entrypoint)
    EngineCoreProc.run_engine_core = staticmethod(wrapped_entrypoint)
    setattr(EngineCoreProc, _ENTRYPOINT_PATCH_ATTR, True)
    core_module.EngineCoreProc.run_engine_core = EngineCoreProc.run_engine_core
    hook_logger.info("Mooncake EngineCore diagnostics entrypoint installed")
    return True


def restore_mooncake_scheduler_diagnostics_hook(scheduler_cls: type[Any]) -> None:
    """Restore a patched fake/real Scheduler; intended for unit tests."""

    original_schedule: Callable[..., Any] | None = getattr(scheduler_cls, _ORIGINAL_SCHEDULE_ATTR, None)
    original_update: Callable[..., Any] | None = getattr(scheduler_cls, _ORIGINAL_UPDATE_REMOTE_ATTR, None)
    original_add_request: Callable[..., Any] | None = getattr(scheduler_cls, _ORIGINAL_ADD_REQUEST_ATTR, None)
    original_xfer: Callable[..., Any] | None = getattr(scheduler_cls, _ORIGINAL_XFER_FINISHED_ATTR, None)
    if original_schedule is not None:
        setattr(scheduler_cls, "schedule", original_schedule)
    if original_update is not None:
        setattr(scheduler_cls, "_update_waiting_for_remote_kv", original_update)
    if original_add_request is not None:
        setattr(scheduler_cls, "add_request", original_add_request)
    if original_xfer is not None:
        setattr(scheduler_cls, "_update_from_kv_xfer_finished", original_xfer)
    for attribute in (
        _ORIGINAL_SCHEDULE_ATTR,
        _ORIGINAL_UPDATE_REMOTE_ATTR,
        _ORIGINAL_ADD_REQUEST_ATTR,
        _ORIGINAL_XFER_FINISHED_ATTR,
        _SCHEDULER_PATCH_ATTR,
    ):
        if hasattr(scheduler_cls, attribute):
            delattr(scheduler_cls, attribute)
    _REMOTE_KV_TIMINGS.clear()
    _CONNECTOR_MATCH_STATES.clear()
