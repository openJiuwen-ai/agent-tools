# White-box tests intentionally inspect the internal state and hooks under test.
# pylint: disable=protected-access
from unittest.mock import Mock

from openjiuwentools.infer_router.preprocess import tokenizer as tokenizer_module
from openjiuwentools.infer_router.preprocess.tokenizer import TokenizerManager


def test_fallback_tokenizer_logs_phase_timing(monkeypatch):
    """未加载真实 tokenizer 时也应输出可比较的分阶段耗时。"""

    info = Mock()
    monkeypatch.setattr(tokenizer_module.logger, "info", info)
    manager = TokenizerManager(load_from_file=False)

    token_ids = manager.tokenize_prompt("abc", "test-model")

    assert token_ids == [97, 98, 99]
    timing_call = next(call for call in info.call_args_list if "[TokenizerTiming]" in call.args[0])
    assert timing_call.args[1] == "test-model"
    assert timing_call.args[2] == "completion"
    assert timing_call.args[3] >= 0.0
    assert timing_call.args[4] >= 0.0
    assert timing_call.args[5] >= timing_call.args[3]
    assert timing_call.args[6] is False


def test_preload_models_deduplicates_and_warms_chat_tokenizer(monkeypatch):
    """启动预加载应按模型去重，并同时预热普通编码和 chat template。"""

    tokenizer = Mock()
    tokenizer.chat_template = "{{ messages }}"
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    manager = TokenizerManager(load_from_file=True)
    get_tokenizer = Mock(return_value=tokenizer)
    monkeypatch.setattr(manager, "_get_tokenizer", get_tokenizer)

    results = manager.preload_models(["model-b", "model-a", "model-a"])

    assert results == {"model-a": True, "model-b": True}
    assert get_tokenizer.call_count == 2
    assert tokenizer.encode.call_count == 2
    assert tokenizer.apply_chat_template.call_count == 2


class _ComposableChatTokenizer:
    """以特殊END token作为安全消息边界的测试tokenizer。"""

    chat_template = "test-template"
    name_or_path = "test-tokenizer"
    special_tokens_map = {"eos_token": "<END>"}
    all_special_ids = [999]

    def __init__(self):
        self.encoded_texts = []

    @staticmethod
    def _render(messages, add_generation_prompt):
        text = "".join(f"<{item['role']}>{item['content']}<END>" for item in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        rendered = self._render(messages, add_generation_prompt)
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        self.encoded_texts.append(text)
        result = []
        cursor = 0
        while cursor < len(text):
            if text.startswith("<END>", cursor):
                result.append(999)
                cursor += len("<END>")
            else:
                result.append(ord(text[cursor]))
                cursor += 1
        return result


class _QwenStyleChatTokenizer(_ComposableChatTokenizer):
    """模拟前缀以换行结束、下一消息以特殊token开始的Qwen模板。"""

    all_special_ids = [998, 999]

    @staticmethod
    def _render(messages, add_generation_prompt):
        text = "".join(f"<START>{item['role']}\n{item['content']}<END>\n" for item in messages)
        if add_generation_prompt:
            text += "<START>assistant\n"
        return text

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        self.encoded_texts.append(text)
        result = []
        cursor = 0
        while cursor < len(text):
            if text.startswith("<START>", cursor):
                result.append(998)
                cursor += len("<START>")
            elif text.startswith("<END>", cursor):
                result.append(999)
                cursor += len("<END>")
            else:
                result.append(ord(text[cursor]))
                cursor += 1
        return result


class _ToolAwareChatTokenizer(_ComposableChatTokenizer):
    """将工具名编码进模板的测试 tokenizer。"""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        tools=None,
    ):
        tool_text = "".join(item["function"]["name"] for item in (tools or []))
        rendered = f"<tools>{tool_text}</tools>" + self._render(
            messages,
            add_generation_prompt,
        )
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class _OffsetCompletionTokenizer:
    """提供字符offset mapping的Completion测试tokenizer。"""

    name_or_path = "completion-tokenizer"
    chat_template = None
    special_tokens_map = {}

    def __init__(self):
        self.encoded_texts = []

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del add_special_tokens
        self.encoded_texts.append(text)
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        self.encoded_texts.append(text)
        return [ord(char) for char in text]


class _ContextSensitiveCompletionTokenizer(_OffsetCompletionTokenizer):
    """模拟子串独立编码与完整文本编码边界不一致的tokenizer。"""

    @staticmethod
    def _ids(text):
        return [ord(char) + (1000 if index == 0 else 0) for index, char in enumerate(text)]

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del add_special_tokens
        self.encoded_texts.append(text)
        result = {"input_ids": self._ids(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        self.encoded_texts.append(text)
        return self._ids(text)


def test_chat_prefix_cache_uses_validated_message_boundary(monkeypatch):
    """边界验证完成后，共享消息前缀请求应只编码变化后缀。"""

    info = Mock()
    monkeypatch.setattr(tokenizer_module.logger, "info", info)
    tokenizer = _ComposableChatTokenizer()
    manager = TokenizerManager(load_from_file=True)
    manager._tokenizers["model-a"] = tokenizer
    shared = {"role": "system", "content": "S" * 100}

    manager.tokenize_messages([shared, {"role": "user", "content": "q1"}], "model-a")
    manager.tokenize_messages([shared, {"role": "user", "content": "q2"}], "model-a")
    calls_before_fast_hit = len(tokenizer.encoded_texts)
    token_ids = manager.tokenize_messages(
        [shared, {"role": "user", "content": "q3"}],
        "model-a",
    )

    expected = _ComposableChatTokenizer().apply_chat_template(
        [shared, {"role": "user", "content": "q3"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert token_ids == expected
    assert len(tokenizer.encoded_texts) == calls_before_fast_hit + 1
    assert "S" * 100 not in tokenizer.encoded_texts[-1]
    timing_call = next(call for call in reversed(info.call_args_list) if "[TokenizerTiming]" in call.args[0])
    assert timing_call.args[9] is True
    assert timing_call.args[11] > 100


def test_chat_full_cache_skips_repeated_prompt_encoding(monkeypatch):
    """完全相同的chat请求应直接复用完整token结果。"""

    info = Mock()
    monkeypatch.setattr(tokenizer_module.logger, "info", info)
    tokenizer = _ComposableChatTokenizer()
    manager = TokenizerManager(load_from_file=True)
    manager._tokenizers["model-a"] = tokenizer
    messages = [{"role": "user", "content": "same prompt"}]

    first = manager.tokenize_messages(messages, "model-a")
    calls_after_first = len(tokenizer.encoded_texts)
    second = manager.tokenize_messages(messages, "model-a")

    assert second == first
    assert len(tokenizer.encoded_texts) == calls_after_first
    timing_call = next(call for call in reversed(info.call_args_list) if "[TokenizerTiming]" in call.args[0])
    assert timing_call.args[10] is True


def test_tool_chat_tokenization_includes_tools_and_bypasses_prefix_cache(monkeypatch):
    """不同工具集必须产生不同 token，且不能共享仅按消息构造的缓存。"""
    monkeypatch.setattr(tokenizer_module.logger, "info", Mock())
    tokenizer = _ToolAwareChatTokenizer()
    manager = TokenizerManager(load_from_file=True)
    manager._tokenizers["model-tools"] = tokenizer
    messages = [{"role": "user", "content": "use a tool"}]

    first = manager.tokenize_messages(
        messages,
        "model-tools",
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    second = manager.tokenize_messages(
        messages,
        "model-tools",
        tools=[{"type": "function", "function": {"name": "search"}}],
    )

    assert first != second
    assert any("lookup" in text for text in tokenizer.encoded_texts)
    assert any("search" in text for text in tokenizer.encoded_texts)


def test_chat_prefix_cache_accepts_special_token_at_suffix_start(monkeypatch):
    """Qwen风格换行前缀应利用下一消息开头的特殊token确认安全边界。"""

    monkeypatch.setattr(tokenizer_module.logger, "info", Mock())
    tokenizer = _QwenStyleChatTokenizer()
    manager = TokenizerManager(load_from_file=True)
    manager._tokenizers["model-qwen"] = tokenizer
    shared = {"role": "system", "content": "S" * 100}

    for question in ("q1", "q2"):
        manager.tokenize_messages(
            [shared, {"role": "user", "content": question}],
            "model-qwen",
        )
    calls_before_fast_hit = len(tokenizer.encoded_texts)
    manager.tokenize_messages(
        [shared, {"role": "user", "content": "q3"}],
        "model-qwen",
    )

    assert len(tokenizer.encoded_texts) == calls_before_fast_hit + 1
    assert tokenizer.encoded_texts[-1].startswith("<START>user")


def test_chat_prefix_cache_falls_back_without_safe_special_boundary(monkeypatch):
    """无法证明消息边界安全时必须持续返回完整分词结果。"""

    monkeypatch.setattr(tokenizer_module.logger, "info", Mock())
    tokenizer = _ComposableChatTokenizer()
    tokenizer.all_special_ids = []
    manager = TokenizerManager(load_from_file=True)
    manager._tokenizers["model-unsafe"] = tokenizer
    shared = {"role": "system", "content": "S" * 100}

    for question in ("q1", "q2"):
        manager.tokenize_messages(
            [shared, {"role": "user", "content": question}],
            "model-unsafe",
        )
    calls_before_third = len(tokenizer.encoded_texts)
    actual = manager.tokenize_messages(
        [shared, {"role": "user", "content": "q3"}],
        "model-unsafe",
    )
    expected = _ComposableChatTokenizer().apply_chat_template(
        [shared, {"role": "user", "content": "q3"}],
        tokenize=True,
        add_generation_prompt=True,
    )

    assert actual == expected
    assert len(tokenizer.encoded_texts) == calls_before_third + 1
    assert "S" * 100 in tokenizer.encoded_texts[-1]


def test_completion_prefix_cache_uses_validated_text_prefix(monkeypatch):
    """边界校验后，Completion共享文本前缀应只重新编码变化尾部。"""

    info = Mock()
    monkeypatch.setattr(tokenizer_module.logger, "info", info)
    tokenizer = _OffsetCompletionTokenizer()
    manager = TokenizerManager(load_from_file=True)
    manager._tokenizers["model-completion"] = tokenizer
    shared = "S" * 300

    manager.tokenize_prompt(shared + "question-a", "model-completion")
    manager.tokenize_prompt(shared + "question-b", "model-completion")
    calls_before_fast_hit = len(tokenizer.encoded_texts)
    actual = manager.tokenize_prompt(
        shared + "question-c",
        "model-completion",
    )

    assert actual == [ord(char) for char in shared + "question-c"]
    assert len(tokenizer.encoded_texts) == calls_before_fast_hit + 1
    assert len(tokenizer.encoded_texts[-1]) < len(shared)
    timing_call = next(call for call in reversed(info.call_args_list) if "[TokenizerTiming]" in call.args[0])
    assert timing_call.args[9] is True
    assert timing_call.args[10] is False
    assert timing_call.args[11] >= 128
    assert timing_call.args[12] >= 0.0
    assert timing_call.args[13] >= 0.0
    assert timing_call.args[18] == 0.0
    assert len(manager._completion_prefix_cache) == 1


def test_completion_prefix_cache_validates_before_fast_path(monkeypatch):
    """首次公共文本候选必须完整编码校验，不能立即报告前缀命中。"""

    info = Mock()
    monkeypatch.setattr(tokenizer_module.logger, "info", info)
    tokenizer = _OffsetCompletionTokenizer()
    manager = TokenizerManager(load_from_file=True)
    manager._tokenizers["model-completion"] = tokenizer
    shared = "P" * 300

    manager.tokenize_prompt(shared + "one", "model-completion")
    actual = manager.tokenize_prompt(shared + "two", "model-completion")

    assert actual == [ord(char) for char in shared + "two"]
    timing_call = next(call for call in reversed(info.call_args_list) if "[TokenizerTiming]" in call.args[0])
    assert timing_call.args[9] is False
    assert timing_call.args[14] is True


def test_completion_prefix_cache_respects_existing_switch(monkeypatch):
    """关闭统一前缀缓存开关后，Completion请求应始终执行完整编码。"""

    monkeypatch.setattr(tokenizer_module.logger, "info", Mock())
    tokenizer = _OffsetCompletionTokenizer()
    manager = TokenizerManager(load_from_file=True)
    manager._prefix_cache_enabled = False
    manager._tokenizers["model-disabled"] = tokenizer
    shared = "D" * 300

    manager.tokenize_prompt(shared + "first", "model-disabled")
    manager.tokenize_prompt(shared + "second", "model-disabled")

    assert tokenizer.encoded_texts == [
        shared + "first",
        shared + "second",
    ]


def test_completion_prefix_cache_disables_unsafe_boundary(monkeypatch):
    """增量结果不一致时应返回完整结果，并停用对应文本边界。"""

    info = Mock()
    monkeypatch.setattr(tokenizer_module.logger, "info", info)
    tokenizer = _ContextSensitiveCompletionTokenizer()
    manager = TokenizerManager(load_from_file=True)
    manager._tokenizers["model-sensitive"] = tokenizer
    shared = "X" * 300

    manager.tokenize_prompt(shared + "first", "model-sensitive")
    second = manager.tokenize_prompt(shared + "second", "model-sensitive")
    calls_before_third = len(tokenizer.encoded_texts)
    third = manager.tokenize_prompt(shared + "third", "model-sensitive")

    assert second == tokenizer._ids(shared + "second")
    assert third == tokenizer._ids(shared + "third")
    assert len(tokenizer.encoded_texts) == calls_before_third + 1
    timing_call = next(call for call in reversed(info.call_args_list) if "[TokenizerTiming]" in call.args[0])
    assert timing_call.args[9] is False


def test_tokenizer_namespace_is_cached_per_instance(monkeypatch):
    """同一个tokenizer实例的配置摘要只应序列化计算一次。"""

    monkeypatch.setattr(tokenizer_module.logger, "info", Mock())
    tokenizer = _OffsetCompletionTokenizer()
    manager = TokenizerManager(load_from_file=True)
    manager._tokenizers["model-namespace"] = tokenizer
    namespace_builder = Mock(wraps=manager._chat_cache_namespace)
    monkeypatch.setattr(manager, "_chat_cache_namespace", namespace_builder)

    manager.tokenize_prompt("N" * 300 + "first", "model-namespace")
    manager.tokenize_prompt("N" * 300 + "second", "model-namespace")
    manager.tokenize_prompt("N" * 300 + "third", "model-namespace")

    assert namespace_builder.call_count == 1
