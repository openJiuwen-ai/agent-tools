"""测试调度模块"""

from unittest.mock import Mock

import pytest

from openjiuwentools.infer_router.schedule.scheduler import Scheduler
from openjiuwentools.infer_router.schemas.agent_hints import RouteHint


@pytest.fixture
def scheduler():
    """创建调度器实例"""
    mock_kv_cache_manager = Mock()
    return Scheduler(mock_kv_cache_manager)


@pytest.fixture
def route_hint():
    """创建路由提示"""
    return RouteHint(
        request_id="test-1",
        model="test-model",
        prompt="Hello world",
        estimated_tokens=100,
        estimated_output_tokens=100,
        priority=5,
        prefixes=["Hello"],
        next_turn_prefill=False,
    )


def test_scheduler_submit(scheduler, route_hint):
    """测试任务提交"""
    scheduler.submit(route_hint)

    assert scheduler.get_queue_size() == 1


def test_scheduler_get_next_request(scheduler, route_hint):
    """测试获取下一个请求"""
    scheduler.submit(route_hint)

    next_request = scheduler.get_next_request()

    assert next_request is not None
    assert next_request.route_hint.request_id == "test-1"
    assert scheduler.get_queue_size() == 0


def test_scheduler_fcfs_strategy(scheduler):
    """测试 FCFS 策略"""
    scheduler.set_strategy("FCFS")

    hint1 = RouteHint(
        request_id="test-1",
        model="test-model",
        prompt="First",
        estimated_tokens=100,
        estimated_output_tokens=100,
        priority=5,
        prefixes=[],
        next_turn_prefill=False,
    )
    hint2 = RouteHint(
        request_id="test-2",
        model="test-model",
        prompt="Second",
        estimated_tokens=100,
        estimated_output_tokens=100,
        priority=5,
        prefixes=[],
        next_turn_prefill=False,
    )

    scheduler.submit(hint1)
    scheduler.submit(hint2)

    next_request = scheduler.get_next_request()
    assert next_request.route_hint.request_id == "test-1"


def test_scheduler_lcfs_strategy(scheduler):
    """测试 LCFS 策略"""
    scheduler.set_strategy("LCFS")

    hint1 = RouteHint(
        request_id="test-1",
        model="test-model",
        prompt="First",
        estimated_tokens=100,
        estimated_output_tokens=100,
        priority=5,
        prefixes=[],
        next_turn_prefill=False,
    )
    hint2 = RouteHint(
        request_id="test-2",
        model="test-model",
        prompt="Second",
        estimated_tokens=100,
        estimated_output_tokens=100,
        priority=5,
        prefixes=[],
        next_turn_prefill=False,
    )

    scheduler.submit(hint1)
    scheduler.submit(hint2)

    next_request = scheduler.get_next_request()
    assert next_request.route_hint.request_id == "test-2"


def test_scheduler_wspt_strategy(scheduler):
    """测试 WS-PT 策略

    WS-PT 策略选择收益最高的请求：
    eval = (1 + priority) / 待计算的token数
    待计算的token数 = 输入token数 - KV缓存命中token数

    test-2 的待计算 token 数更少，所以优先级更高
    """
    scheduler.set_strategy("WSPT")

    hint1 = RouteHint(
        request_id="test-1",
        model="test-model",
        prompt="First",
        estimated_tokens=100,
        estimated_output_tokens=100,
        priority=5,
        prefixes=[],
        next_turn_prefill=False,
        token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    )
    hint2 = RouteHint(
        request_id="test-2",
        model="test-model",
        prompt="Second",
        estimated_tokens=100,
        estimated_output_tokens=50,
        priority=5,
        prefixes=[],
        next_turn_prefill=False,
        token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )

    scheduler.submit(hint1)
    scheduler.submit(hint2, overlap_scores={"worker-1": 10})

    next_request = scheduler.get_next_request()
    assert next_request.route_hint.request_id == "test-2"


def test_scheduler_cancel_request(scheduler, route_hint):
    """测试取消请求"""
    scheduler.submit(route_hint)

    result = scheduler.cancel_request("test-1")
    assert result is True
    assert scheduler.get_queue_size() == 0

    result = scheduler.cancel_request("non-existent")
    assert result is False


def test_scheduler_get_stats(scheduler, route_hint):
    """测试获取统计信息"""
    scheduler.submit(route_hint)

    stats = scheduler.get_stats()

    assert stats["total_tasks"] == 1
    assert stats["total_queue_size"] == 1
    assert "current_strategy" in stats


@pytest.mark.parametrize("priority", [1, 5, 10])
def test_scheduler_submit_with_different_priorities(scheduler, route_hint, priority):
    """测试不同优先级的任务提交"""
    route_hint.priority = priority

    scheduler.submit(route_hint)

    assert scheduler.get_queue_size() == 1
