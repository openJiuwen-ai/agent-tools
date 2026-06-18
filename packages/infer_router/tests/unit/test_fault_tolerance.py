"""测试容错模块"""

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
