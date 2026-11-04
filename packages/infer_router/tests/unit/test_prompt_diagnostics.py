import asyncio
import logging
from types import SimpleNamespace

import pytest

from openjiuwentools.infer_router.prompt_diagnostics import (
    digest,
    enabled,
    prompt_observation,
    token_observation,
    trace_generation,
)


def test_token_containers_have_same_identity():
    values = [1, 2, 300]
    assert token_observation({"input_ids": [values], "attention_mask": [[1, 1, 1]]})[
        "token_sha256"
    ] == digest(values)
    assert token_observation(values)["token_count"] == 3
    assert token_observation({"input_ids": values})["token_count"] == 3


def test_prompt_observation_is_private_and_nonmutating():
    body = {
        "messages": [{"role": "user", "content": "private user text"}],
        "tools": [{"name": "private tool"}],
        "api_key": "secret-key",
    }
    before = repr(body)
    result = prompt_observation(body, [1, 2], stage="test", request_id="r1")
    assert "private" not in repr(result)
    assert "secret-key" not in repr(result)
    assert repr(body) == before
    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})


def test_disabled_returns_original_generator(monkeypatch):
    monkeypatch.delenv("JIUWEN_PD_PROMPT_DIAGNOSTICS", raising=False)
    sentinel = object()
    assert not enabled()
    assert trace_generation(sentinel, None, "r", "e") is sentinel


@pytest.mark.parametrize("failure", [False, True])
def test_generation_terminal_diagnostics(monkeypatch, caplog, failure):
    monkeypatch.setenv("JIUWEN_PD_PROMPT_DIAGNOSTICS", "1")
    log = logging.getLogger("test.pd.diagnostics")

    async def generator():
        yield SimpleNamespace(prompt_token_ids=[1, 2])
        if failure:
            raise ValueError("test error")

    async def run():
        result = []
        async for item in trace_generation(generator(), log, "r", "e"):
            result.append(item)
        return result

    with caplog.at_level(logging.INFO):
        if failure:
            with pytest.raises(ValueError):
                asyncio.run(run())
        else:
            assert len(asyncio.run(run())) == 1
    assert "phase=first_engine_output" in caplog.text
    assert f"phase={'error' if failure else 'completed'}" in caplog.text


def test_generation_close_logged(monkeypatch, caplog):
    monkeypatch.setenv("JIUWEN_PD_PROMPT_DIAGNOSTICS", "true")

    async def generator():
        yield SimpleNamespace(prompt_token_ids=[1])

    async def run():
        stream = trace_generation(generator(), logging.getLogger("test.pd"), "r", "e")
        await anext(stream)
        await stream.aclose()

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    assert "phase=cancelled_or_closed" in caplog.text
