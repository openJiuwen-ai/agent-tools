"""测试容错模块"""

import asyncio
import time

import pytest

from openjiuwentools.infer_router.fault_tolerance.fault_tolerance import (
    CircuitBreaker,
    retry_async,
    retry_sync,
)


def _open_breaker() -> CircuitBreaker:
    """返回一个已进入 OPEN 状态的断路器"""
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    breaker.handle_failure()
    assert breaker.state == "OPEN"
    return breaker


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


@pytest.mark.asyncio
async def test_retry_async_backoff_when_circuit_open(monkeypatch):
    """断路器拒绝时应退避等待，而不是立即忙等重试"""
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def mock_func():
        return "unreachable"

    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        await retry_async(
            mock_func,
            max_attempts=3,
            delay=0.01,
            jitter=False,
            circuit_breaker=_open_breaker(),
        )

    assert sleeps == [0.01, 0.02]


def test_retry_sync_backoff_when_circuit_open(monkeypatch):
    """同步重试在断路器拒绝时同样退避等待"""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        retry_sync(
            lambda: "unreachable",
            max_attempts=3,
            delay=0.01,
            jitter=False,
            circuit_breaker=_open_breaker(),
        )

    assert sleeps == [0.01, 0.02]
