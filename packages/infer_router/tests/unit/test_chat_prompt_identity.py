# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
"""Router must tokenize the same complete messages as Worker templates."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from openjiuwentools.infer_router.preprocess.preprocessor import Preprocessor
from openjiuwentools.infer_router.preprocess.tokenizer import TokenizerManager
from openjiuwentools.infer_router.schemas.agent_hints import ChatCompletionRequest


@pytest.fixture
def history():
    return [
        {"role": "system", "content": "Retail policy"},
        {"role": "user", "content": "Find my order"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "get_order_details",
                        "arguments": '{"order_id":"123"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "get_order_details",
            "content": '{"status":"delivered"}',
        },
        {"role": "user", "content": "Please continue"},
    ]


def test_full_message_copy_preserves_tool_history(history):
    formatted = TokenizerManager._format_chat_messages(history)
    assert formatted == history
    assert formatted[2]["content"] is None
    formatted[2]["tool_calls"][0]["function"]["arguments"] = "{}"
    assert history[2]["tool_calls"][0]["function"]["arguments"] != "{}"


@pytest.mark.parametrize("content", [None, [{"type": "text", "text": "Hello"}], "Hello"])
def test_content_is_never_coerced(content):
    messages = [{"role": "assistant", "content": content, "name": "agent"}]
    assert TokenizerManager._format_chat_messages(messages) == messages


@pytest.mark.parametrize("with_tools", [False, True])
def test_real_local_template_matches_worker_input(history, monkeypatch, with_tools):
    """Use a local HF tokenizer/template, with no model files or network access."""
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, **{chr(i): i for i in range(32, 127)}}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{{ messages | tojson }}{% if tools %}{{ tools | tojson }}{% endif %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )
    tools = (
        [
            {
                "type": "function",
                "function": {"name": "get_order_details", "parameters": {"type": "object"}},
            }
        ]
        if with_tools
        else None
    )
    manager = TokenizerManager(load_from_file=False)
    monkeypatch.setattr(manager, "_get_tokenizer", lambda model: tokenizer)
    incremental = Mock(side_effect=AssertionError("Structured history must not use text cache"))
    monkeypatch.setattr(manager, "_tokenize_chat_with_prefix_cache", incremental)
    original = deepcopy(history)
    for arguments in ('{"order_id":"123"}', '{"order_id":"45678"}'):
        history[2]["tool_calls"][0]["function"]["arguments"] = arguments
        kwargs = {"tools": tools} if tools else {}
        expected = tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True, **kwargs)
        if hasattr(expected, "input_ids"):
            expected = expected.input_ids
        actual = manager.tokenize_messages(history, "local-test", tools=tools)
        assert actual == expected
        assert len(actual) > 2
        preprocessor = Preprocessor()
        preprocessor.tokenizer_manager = manager
        request = ChatCompletionRequest(model="local-test", messages=history, tools=tools)
        assert preprocessor.process(request, None, "pd-chat-test").token_ids == expected
    assert not incremental.called
    original[2]["tool_calls"][0]["function"]["arguments"] = arguments
    assert history == original


def test_cache_boundary_is_restricted_to_plain_text(history):
    assert not TokenizerManager._supports_text_prefix_cache(history)
    assert TokenizerManager._supports_text_prefix_cache(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "Hi"},
        ]
    )
    assert not TokenizerManager._supports_text_prefix_cache([{"role": "tool", "content": "result"}])
    assert not TokenizerManager._supports_text_prefix_cache([{"role": "assistant", "content": None}])


def test_history_fingerprints_include_tool_ids_and_arguments(history):
    before = TokenizerManager._chat_message_fingerprints(history)
    changed = deepcopy(history)
    changed[2]["tool_calls"][0]["function"]["arguments"] = "{}"
    changed[3]["tool_call_id"] = "call-2"
    after = TokenizerManager._chat_message_fingerprints(changed)
    assert before[:2] == after[:2]
    assert before[2:4] != after[2:4]


def test_tool_schema_changes_are_not_cached(history, monkeypatch):
    seen = []

    class Tokenizer:
        chat_template = "test-template"

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            seen.append((deepcopy(messages), deepcopy(kwargs["tools"])))
            return [len(kwargs["tools"][0]["function"]["name"])]

    manager = TokenizerManager(load_from_file=False)
    monkeypatch.setattr(manager, "_get_tokenizer", lambda model: Tokenizer())
    for name in ("first", "second-tool"):
        tools = [{"type": "function", "function": {"name": name}}]
        assert manager.tokenize_messages(history, "m", tools=tools) == [len(name)]
    assert len(seen) == 2
    assert all(messages == history for messages, _ in seen)
