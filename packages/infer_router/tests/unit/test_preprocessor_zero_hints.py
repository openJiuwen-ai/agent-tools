"""Explicit 0 in AgentHints must not be replaced by the default value."""

from openjiuwentools.infer_router.preprocess.preprocessor import Preprocessor
from openjiuwentools.infer_router.schemas.agent_hints import (
    AgentHints,
    ChatCompletionRequest,
)


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])


def test_explicit_zero_hints_are_clamped_not_defaulted() -> None:
    hint = Preprocessor().process(
        _request(),
        AgentHints(estimated_output_tokens=0, total_requests=0, iat=0),
        "req-1",
    )

    assert hint.estimated_output_tokens == 1
    assert hint.total_requests == 1
    assert hint.iat == 1
