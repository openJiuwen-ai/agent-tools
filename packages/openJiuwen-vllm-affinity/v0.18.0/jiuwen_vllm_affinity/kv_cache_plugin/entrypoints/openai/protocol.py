from typing import ClassVar

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class ReleaseKvCacheResponse(OpenAIBaseModel):
    cache_salt: str | None = None
    block_released: int


def register_chat_request():
    """ChatCompletionRequest already allows extra fields (extra=\"allow\")."""


class ReleaseKvCacheRequest(ChatCompletionRequest):
    """Chat completion-shaped body plus fields for partial KV release."""

    # Shadow OpenAIBaseModel.field_names so __log_extra_fields__ recomputes from
    # this class's model_fields; otherwise we inherit ChatCompletionRequest's
    # cached set and wrongly warn that subclass-only keys are "ignored".
    field_names: ClassVar[set[str] | None] = None

    messages_released_index: int = 0
    tools_released_index: int | None = None
    cache_sharing: bool | None = None
