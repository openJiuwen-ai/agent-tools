"""测试数据模型"""

from openjiuwentools.infer_router.schemas.agent_hints import (
    AgentHints,
    CacheControl,
    ChatCompletionRequest,
    JWExt,
    RouteHint,
    WorkerInfo,
)


def test_worker_info_defaults():
    """测试WorkerInfo默认值"""
    worker = WorkerInfo(
        worker_id="worker-1",
        model="test-model",
        total_tokens=500000,
        current_load=0.0,
        cached_prefixes=[],
        url="http://localhost:8001/v1",
    )

    assert worker.worker_id == "worker-1"
    assert worker.model == "test-model"
    assert worker.url == "http://localhost:8001/v1"
    assert worker.total_tokens == 500000
    assert worker.available_memory == 500000
    assert worker.current_load == 0
    assert worker.cached_prefixes == []
    assert worker.engine_type == "vllm"
    assert worker.api_key is None


def test_worker_info_with_values():
    """测试WorkerInfo带值"""
    worker = WorkerInfo(
        worker_id="worker-1",
        model="test-model",
        url="http://localhost:8001/v1",
        available_memory=1000000,
        current_load=10,
        cached_prefixes=["Hello", "Hi"],
    )

    assert worker.worker_id == "worker-1"
    assert worker.model == "test-model"
    assert worker.url == "http://localhost:8001/v1"
    assert worker.available_memory == 1000000
    assert worker.current_load == 10
    assert worker.cached_prefixes == ["Hello", "Hi"]


def test_agent_hints_defaults():
    """测试AgentHints默认值"""
    hints = AgentHints()

    assert hints.priority == 0
    assert hints.estimated_output_tokens == 128
    assert hints.next_turn_prefill is False


def test_agent_hints_with_values():
    """测试AgentHints带值"""
    hints = AgentHints(priority=5, estimated_output_tokens=256, next_turn_prefill=True)

    assert hints.priority == 5
    assert hints.estimated_output_tokens == 256
    assert hints.next_turn_prefill is True


def test_jwext_defaults():
    """测试JWExt默认值"""
    jwext = JWExt()

    assert jwext.agent_hints is None


def test_jwext_with_hints():
    """测试JWExt带AgentHints"""
    hints = AgentHints(priority=5)
    jwext = JWExt(agent_hints=hints)

    assert jwext.agent_hints is not None
    assert jwext.agent_hints.priority == 5


def test_chat_completion_request():
    """测试ChatCompletionRequest"""
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        jiuwenext=JWExt(agent_hints=AgentHints(priority=5)),
        max_tokens=100,
        temperature=0.7,
    )

    assert request.model == "test-model"
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"
    assert request.jiuwenext.agent_hints.priority == 5
    assert request.max_tokens == 100
    assert request.temperature == 0.7


def test_route_hint():
    """测试RouteHint"""
    route_hint = RouteHint(
        priority=5,
        estimated_output_tokens=100,
        next_turn_prefill=False,
        request_id="test-1",
        model="test-model",
        token_ids=[72, 101, 108, 108, 111],
    )

    assert route_hint.request_id == "test-1"
    assert route_hint.model == "test-model"
    assert route_hint.estimated_output_tokens == 100
    assert route_hint.priority == 5
    assert route_hint.token_ids == [72, 101, 108, 108, 111]


def test_worker_info_model_dump():
    """测试WorkerInfo模型转储"""
    worker = WorkerInfo(
        worker_id="worker-1",
        model="test-model",
        available_memory=0,
        current_load=0,
        cached_prefixes=[],
        url="http://localhost:8001/v1",
    )

    data = worker.model_dump()
    assert data["worker_id"] == "worker-1"
    assert data["model"] == "test-model"
    assert data["url"] == "http://localhost:8001/v1"
    assert data["engine_type"] == "vllm"
    assert data["api_key"] is None


def test_agent_hints_model_dump():
    """测试AgentHints模型转储"""
    hints = AgentHints(priority=5, estimated_output_tokens=256)

    data = hints.model_dump()
    assert data["priority"] == 5
    assert data["estimated_output_tokens"] == 256
    assert data["next_turn_prefill"] is False


def test_cache_control_defaults():
    """测试CacheControl默认值"""
    cache_control = CacheControl()

    assert cache_control.type == "ephemeral"
    assert cache_control.ttl is None


def test_cache_control_with_values():
    """测试CacheControl带值"""
    cache_control = CacheControl(type="persistent", ttl="5m")

    assert cache_control.type == "persistent"
    assert cache_control.ttl == "5m"


def test_agent_hints_with_new_fields():
    """测试AgentHints带新字段"""
    hints = AgentHints(
        priority=5,
        estimated_output_tokens=256,
        next_turn_prefill=True,
        prefix_id="prefix-123",
        total_requests=10,
        iat=250,
    )

    assert hints.priority == 5
    assert hints.estimated_output_tokens == 256
    assert hints.next_turn_prefill is True
    assert hints.prefix_id == "prefix-123"
    assert hints.total_requests == 10
    assert hints.iat == 250


def test_jwext_with_cache_control():
    """测试JWExt带CacheControl"""
    hints = AgentHints(priority=5)
    cache_control = CacheControl(type="persistent")
    jwext = JWExt(agent_hints=hints, cache_control=cache_control)

    assert jwext.agent_hints is not None
    assert jwext.agent_hints.priority == 5
    assert jwext.cache_control is not None
    assert jwext.cache_control.type == "persistent"


def test_route_hint_with_new_fields():
    """测试RouteHint带新字段"""
    route_hint = RouteHint(
        priority=5,
        estimated_output_tokens=256,
        next_turn_prefill=True,
        request_id="test-1",
        model="test-model",
        prefix_id="prefix-123",
        total_requests=10,
        iat=250,
    )

    assert route_hint.priority == 5
    assert route_hint.estimated_output_tokens == 256
    assert route_hint.next_turn_prefill is True
    assert route_hint.request_id == "test-1"
    assert route_hint.model == "test-model"
    assert route_hint.prefix_id == "prefix-123"
    assert route_hint.total_requests == 10
    assert route_hint.iat == 250
