# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
import hashlib
import logging
from types import SimpleNamespace

from jiuwen_kvbm.integrations import vllm_ascend_018 as integration


def test_token_digest_matches_router_canonical_encoding():
    assert integration._pd_token_digest([1, 2, 300]) == hashlib.sha256(b"[1,2,300]").hexdigest()


def test_fallback_observation_logs_schedule_once_then_removal(monkeypatch, caplog):
    observations = {"req": {"handoff_id": "pd-req", "started": 10, "scheduled": False}}
    monkeypatch.setattr(integration, "_PD_FALLBACK_OBSERVATIONS", observations)
    monkeypatch.setattr(integration.time, "monotonic", lambda: 11)
    request = object()
    scheduler = SimpleNamespace(requests={"req": request})
    output = SimpleNamespace(num_scheduled_tokens={"req": 1})
    log = logging.getLogger("test.fallback")
    with caplog.at_level(logging.INFO):
        integration._observe_pd_fallback_schedule(scheduler, output, log)
        integration._observe_pd_fallback_schedule(scheduler, output, log)
        assert caplog.text.count("phase=first_scheduled_after_gate_release") == 1
        assert scheduler.requests["req"] is request
        scheduler.requests.clear()
        integration._observe_pd_fallback_schedule(scheduler, SimpleNamespace(), log)
    assert "left_scheduler_not_proof_of_success" in caplog.text
    assert not observations


def test_fallback_observation_expires(monkeypatch, caplog):
    observations = {"req": {"handoff_id": "pd-req", "started": 0, "scheduled": False}}
    monkeypatch.setattr(integration, "_PD_FALLBACK_OBSERVATIONS", observations)
    monkeypatch.setattr(integration.time, "monotonic", lambda: 601)
    with caplog.at_level(logging.INFO):
        integration._observe_pd_fallback_schedule(
            SimpleNamespace(requests={"req": object()}), SimpleNamespace(), logging.getLogger("test.fallback")
        )
    assert not observations
    assert "observation_expired" in caplog.text
