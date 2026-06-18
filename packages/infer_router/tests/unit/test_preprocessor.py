"""测试预处理模块"""

import pytest

from openjiuwentools.infer_router.preprocess.preprocessor import Preprocessor
from openjiuwentools.infer_router.schemas.agent_hints import (
    AgentHints,
    ChatCompletionRequest,
    JWExt,
)


@pytest.fixture
def preprocessor():
    """创建预处理器实例"""
    return Preprocessor()


@pytest.fixture
def chat_request():
    """创建聊天请求"""
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello world"}],
        jiuwenext=JWExt(
            agent_hints=AgentHints(
                priority=5,
                estimated_output_tokens=100,
                next_turn_prefill=True,
            )
        ),
        max_tokens=100,
        temperature=0.7,
    )


@pytest.fixture
def chat_request_no_hints():
    """创建无Agent Hints的聊天请求"""
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello world"}],
        max_tokens=100,
        temperature=0.7,
    )


def test_preprocessor_process_with_hints(preprocessor, chat_request):
    """测试带Agent Hints的预处理"""
    request_id = "test-1"
    agent_hints = chat_request.jiuwenext.agent_hints if chat_request.jiuwenext else None

    route_hint = preprocessor.process(chat_request, agent_hints, request_id)

    assert route_hint is not None
    assert route_hint.request_id == request_id
    assert route_hint.model == chat_request.model
    assert route_hint.priority == agent_hints.priority
    assert route_hint.estimated_output_tokens == agent_hints.estimated_output_tokens


def test_preprocessor_process_without_hints(preprocessor, chat_request_no_hints):
    """测试无Agent Hints的预处理"""
    request_id = "test-1"

    route_hint = preprocessor.process(chat_request_no_hints, None, request_id)

    assert route_hint is not None
    assert route_hint.request_id == request_id
    assert route_hint.model == chat_request_no_hints.model
    assert route_hint.priority == 0  # 默认优先级
    assert route_hint.estimated_output_tokens == 128  # 默认token数


def test_preprocessor_extract_prefixes(preprocessor, chat_request):
    """测试前缀提取"""
    request_id = "test-1"
    agent_hints = chat_request.jiuwenext.agent_hints if chat_request.jiuwenext else None

    route_hint = preprocessor.process(chat_request, agent_hints, request_id)

    # 验证token_ids已生成
    assert route_hint.token_ids is not None
    assert len(route_hint.token_ids) > 0


def test_preprocessor_empty_messages(preprocessor):
    """测试空消息的情况"""
    chat_request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        max_tokens=100,
        temperature=0.7,
    )
    request_id = "test-1"

    route_hint = preprocessor.process(chat_request, None, request_id)

    assert route_hint is not None
    assert route_hint.token_ids == []
