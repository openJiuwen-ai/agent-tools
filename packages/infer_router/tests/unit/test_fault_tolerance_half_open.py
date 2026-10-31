"""Circuit breaker half-open accounting through check_state (the retry path)."""

import time

import pytest

from openjiuwentools.infer_router.fault_tolerance.fault_tolerance import CircuitBreaker


def test_check_state_enforces_half_open_max_calls() -> None:
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0, half_open_max_calls=2)
    breaker.state = "OPEN"
    breaker.last_failure_time = time.time() - 10.0

    assert breaker.check_state() is True
    assert breaker.check_state() is True
    assert breaker.half_open_calls == 2
    with pytest.raises(Exception, match="HALF_OPEN with max calls"):
        breaker.check_state()
    assert breaker.total_calls == 3
