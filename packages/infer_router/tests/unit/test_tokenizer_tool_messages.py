"""Tokenizer message projection must preserve tool-call fields."""

from openjiuwentools.infer_router.preprocess.tokenizer import TokenizerManager


class _CapturingTokenizer:
    chat_template = "template"

    def __init__(self) -> None:
        self.messages = []

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return [1, 2, 3]


def test_chat_template_keeps_tool_calls_and_none_content() -> None:
    tokenizer = _CapturingTokenizer()
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "result", "tool_call_id": "call-1", "name": "f"},
    ]

    TokenizerManager._tokenize_with_chat_template(tokenizer, messages)

    assistant, tool = tokenizer.messages
    assert assistant["content"] == ""
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert tool["tool_call_id"] == "call-1"
    assert tool["name"] == "f"
