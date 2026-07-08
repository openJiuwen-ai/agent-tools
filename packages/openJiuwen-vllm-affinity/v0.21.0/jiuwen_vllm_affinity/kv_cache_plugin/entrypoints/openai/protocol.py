from typing import ClassVar

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


def register_chat_request():
    """ChatCompletionRequest already allows extra fields (extra=\"allow\")."""
    if getattr(ChatCompletionRequest, "grammar_from_tool_parser", None) is not None:
        return

    @property
    def grammar_from_tool_parser(self: ChatCompletionRequest) -> bool:
        # vLLM PrivateAttr; only set by ToolParser.adjust_request (e.g. Mistral).
        return bool(getattr(self, "_grammar_from_tool_parser", False))

    ChatCompletionRequest.grammar_from_tool_parser = grammar_from_tool_parser


class ReleaseKvCacheResponse(OpenAIBaseModel):
    cache_salt: str | None = None
    block_released: int


class ReleaseKvCacheRequest(ChatCompletionRequest):
    """Chat completion-shaped body plus fields for partial KV release."""

    # Shadow OpenAIBaseModel.field_names so __log_extra_fields__ recomputes from
    # this class's model_fields; otherwise we inherit ChatCompletionRequest's
    # cached set and wrongly warn that subclass-only keys are "ignored".
    field_names: ClassVar[set[str] | None] = None

    messages_released_index: int = 0
    tools_released_index: int | None = None
    cache_sharing: bool | None = None
