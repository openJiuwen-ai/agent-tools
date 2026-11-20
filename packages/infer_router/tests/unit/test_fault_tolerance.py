"""测试容错模块"""

import time

import pytest

from openjiuwentools.infer_router.fault_tolerance.fault_tolerance import (
    CircuitBreaker,
)


@pytest.fixture
def circuit_breaker():
    """创建熔断器实例"""
    return CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)


@pytest.mark.asyncio
async def test_circuit_breaker_closed(circuit_breaker):
    """测试熔断器关闭状态"""

    async def mock_func():
        return "success"

    result = await circuit_breaker.call(mock_func)

    assert result == "success"
    assert circuit_breaker.state == "CLOSED"


@pytest.mark.asyncio
async def test_circuit_breaker_open(circuit_breaker):
    """测试熔断器打开状态"""

    async def mock_func():
        raise Exception("test error")

    # 触发多次失败，使熔断器打开
    for _ in range(3):
        with pytest.raises(Exception, match="test error"):
            await circuit_breaker.call(mock_func)

    # 熔断器应该打开
    assert circuit_breaker.state == "OPEN"

    # 再次调用应该直接失败
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        await circuit_breaker.call(mock_func)


def test_half_open_probe_limit_via_check_state():
    """HALF_OPEN 下 check_state 自身受探测上限约束"""
    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=30.0, half_open_max_calls=2,
    )
    breaker.handle_failure()
    breaker.last_failure_time = time.time() - 60

    assert breaker.check_state() is True
    assert breaker.state == "HALF_OPEN"
    assert breaker.half_open_calls == 1

    assert breaker.check_state() is True
    assert breaker.half_open_calls == 2

    with pytest.raises(Exception, match="HALF_OPEN with max calls"):
        breaker.check_state()


@pytest.mark.asyncio
async def test_call_counts_half_open_probe_once():
    """call() 路径的探测计数不应重复累加"""
    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=30.0, half_open_max_calls=3,
    )
    breaker.handle_failure()
    breaker.last_failure_time = time.time() - 60

    async def mock_func():
        return "success"

    assert await breaker.call(mock_func) == "success"
    assert breaker.state == "HALF_OPEN"
    assert breaker.half_open_calls == 1
