import argparse
import asyncio
import json

from types import SimpleNamespace

from args import _add_jiuwen_args
from transport import (
    _get_tool_call_settings,
    _make_engine_request_id,
    _requested_max_tokens,
    _resolve_max_tokens,
    _resolve_worker_parser_settings,
    _stream_response,
)


TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]


def test_tools_do_not_enable_auto_calls_by_default():
    tools, tool_choice, inject_tools = _get_tool_call_settings({"tools": TOOLS}, False)

    assert tools == TOOLS
    assert tool_choice == "none"
    assert inject_tools is False


def test_explicit_auto_requires_enable_auto_tools():
    body = {"tools": TOOLS, "tool_choice": "auto"}

    assert _get_tool_call_settings(body, False) == (TOOLS, "auto", False)
    assert _get_tool_call_settings(body, True) == (TOOLS, "auto", True)


def test_required_or_named_tool_still_injects_tools():
    named_tool = {"type": "function", "function": {"name": "lookup"}}

    assert _get_tool_call_settings({"tools": TOOLS, "tool_choice": "required"}, False) == (
        TOOLS,
        "required",
        True,
    )
    assert _get_tool_call_settings({"tools": TOOLS, "tool_choice": named_tool}, False) == (
        TOOLS,
        named_tool,
        True,
    )


def test_cache_sharing_packs_engine_request_id():
    packed = _make_engine_request_id(
        "chatcmpl-1",
        {"cache_sharing": True, "cache_salt": "session-1"},
    )

    assert json.loads(packed) == {
        "request_id": "chatcmpl-1",
        "sharing_cache_salt": "session-1",
    }
    assert _make_engine_request_id("chatcmpl-1", {"cache_salt": "session-1"}) == "chatcmpl-1"


def test_max_completion_tokens_takes_precedence():
    assert _requested_max_tokens({"max_tokens": 512, "max_completion_tokens": 2048}) == 2048


def test_missing_output_limit_uses_remaining_model_context():
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=8192),
        )
    )

    assert _resolve_max_tokens(engine, {}, prompt_tokens=1200) == 6992
    assert _resolve_max_tokens(engine, {"max_tokens": 1000}, 1200) == 1000


def test_official_auto_tool_choice_cli_alias_is_supported():
    parser = argparse.ArgumentParser()
    _add_jiuwen_args(parser)

    args = parser.parse_args(["--enable-auto-tool-choice"])

    assert args.enable_auto_tools is True


def test_qwen3_worker_parsers_are_auto_detected():
    args = SimpleNamespace(
        model="/models/Qwen3-4B",
        served_model_name="qwen3-4b",
        tool_call_parser=None,
        reasoning_parser=None,
        enable_auto_tools=None,
        auto_detect_parsers=True,
    )

    assert _resolve_worker_parser_settings(args) == ("hermes", "qwen3", True)


def test_qwen3_coder_uses_xml_parser_without_reasoning_parser():
    args = SimpleNamespace(
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        served_model_name=None,
        tool_call_parser=None,
        reasoning_parser=None,
        enable_auto_tools=None,
        auto_detect_parsers=True,
    )

    assert _resolve_worker_parser_settings(args) == ("qwen3_xml", None, True)


def test_parser_auto_detection_avoids_unverified_qwen_variants():
    args = SimpleNamespace(
        model="Qwen/Qwen3-VL-4B-Instruct",
        served_model_name=None,
        tool_call_parser=None,
        reasoning_parser=None,
        enable_auto_tools=None,
        auto_detect_parsers=True,
    )

    assert _resolve_worker_parser_settings(args) == (None, None, False)


def test_explicit_parser_settings_override_auto_detection():
    args = SimpleNamespace(
        model="Qwen/Qwen3-4B",
        served_model_name=None,
        tool_call_parser="custom_parser",
        reasoning_parser="custom_reasoning",
        enable_auto_tools=False,
        auto_detect_parsers=True,
    )

    assert _resolve_worker_parser_settings(args) == (
        "custom_parser",
        "custom_reasoning",
        False,
    )


# ---------------------------------------------------------------------------
# 模型工件检测（config.json model_type + chat template 签名）
# ---------------------------------------------------------------------------

QWEN3_HERMES_TEMPLATE = (
    "{%- if tools %}<tool_call>{{ tool | tojson }}</tool_call>{%- endif %}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "<think>\n\n</think>\n\n{%- endif %}"
)

QWEN3_CODER_TEMPLATE = "<tool_call_start>```json\n{{ tool | tojson }}\n```<tool_call_end>"


def _write_model_artifacts(
    tmp_path,
    *,
    model_type=None,
    chat_template=None,
    template_file=None,
    dir_name="myqwen3-finetune",
):
    """构造一个本地模型目录；dir_name 故意含 qwen3 子串以验证不看名字。"""
    model_dir = tmp_path / dir_name
    model_dir.mkdir()
    if model_type is not None:
        (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    if chat_template is not None:
        (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": chat_template}), encoding="utf-8")
    if template_file is not None:
        (model_dir / "chat_template.jinja").write_text(template_file, encoding="utf-8")
    return str(model_dir)


def _parser_args(model, served_model_name=None):
    return SimpleNamespace(
        model=model,
        served_model_name=served_model_name,
        tool_call_parser=None,
        reasoning_parser=None,
        enable_auto_tools=None,
        auto_detect_parsers=True,
    )


def test_artifact_detection_uses_model_type_not_directory_name(tmp_path):
    """finetune 目录名任意：以 config.json 的 model_type 为准。"""
    model_dir = _write_model_artifacts(
        tmp_path,
        model_type="qwen3",
        chat_template=QWEN3_HERMES_TEMPLATE,
    )

    assert _resolve_worker_parser_settings(_parser_args(model_dir)) == (
        "hermes",
        "qwen3",
        True,
    )


def test_artifact_detection_family_map_without_template(tmp_path):
    """tokenizer 工件缺失时家族映射仍然生效。"""
    model_dir = _write_model_artifacts(tmp_path, model_type="qwen3")

    assert _resolve_worker_parser_settings(_parser_args(model_dir)) == (
        "hermes",
        "qwen3",
        True,
    )


def test_artifact_detection_template_signature_wins_over_family(tmp_path):
    """模板是格式级证据：Coder 风格模板优先于 model_type 家族推断。"""
    model_dir = _write_model_artifacts(
        tmp_path,
        model_type="qwen3",
        chat_template=QWEN3_CODER_TEMPLATE,
    )

    assert _resolve_worker_parser_settings(_parser_args(model_dir)) == (
        "qwen3_xml",
        None,
        True,
    )


def test_artifact_detection_unknown_family_uses_template_signature(tmp_path):
    """未知家族的自定义模型只要模板是 hermes 格式即可识别。"""
    model_dir = _write_model_artifacts(
        tmp_path,
        model_type="my_custom_arch",
        chat_template=QWEN3_HERMES_TEMPLATE,
    )

    assert _resolve_worker_parser_settings(_parser_args(model_dir)) == (
        "hermes",
        "qwen3",
        True,
    )


def test_artifact_detection_reads_standalone_chat_template_jinja(tmp_path):
    """chat template 独立成 chat_template.jinja 文件时也能识别。"""
    model_dir = _write_model_artifacts(
        tmp_path,
        model_type="qwen3",
        template_file=QWEN3_HERMES_TEMPLATE,
    )

    assert _resolve_worker_parser_settings(_parser_args(model_dir)) == (
        "hermes",
        "qwen3",
        True,
    )


def test_artifact_detection_refused_variant_blocks_name_fallback(tmp_path):
    """工件明确是 VL 变体时，即使 served 名像 Qwen3 也不得回退名字匹配。"""
    model_dir = _write_model_artifacts(
        tmp_path,
        model_type="qwen3_vl",
        chat_template=QWEN3_HERMES_TEMPLATE,
    )

    assert _resolve_worker_parser_settings(_parser_args(model_dir, served_model_name="qwen3-4b")) == (None, None, False)


def test_artifact_detection_conflict_refuses_guessing(tmp_path):
    """家族说支持但模板被定制得没有已知标记：宁可不猜。"""
    model_dir = _write_model_artifacts(
        tmp_path,
        model_type="qwen3",
        chat_template="<|im_end|>",
    )

    assert _resolve_worker_parser_settings(_parser_args(model_dir)) == (
        None,
        None,
        False,
    )


def test_artifact_detection_partial_result_keeps_reasoning_only(tmp_path):
    """模板只有思考标记时：工具不猜，reasoning 保留。"""
    model_dir = _write_model_artifacts(
        tmp_path,
        model_type="qwen3",
        chat_template="{%- if enable_thinking %}<think>{%- endif %}",
    )

    assert _resolve_worker_parser_settings(_parser_args(model_dir)) == (
        None,
        "qwen3",
        False,
    )


def test_artifact_detection_qwen25_maps_to_hermes_without_reasoning(tmp_path):
    model_dir = _write_model_artifacts(
        tmp_path,
        model_type="qwen2",
        chat_template=QWEN3_HERMES_TEMPLATE.replace(
            "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}",
            "",
        ),
    )

    assert _resolve_worker_parser_settings(_parser_args(model_dir)) == (
        "hermes",
        None,
        True,
    )


def test_stream_tool_parser_receives_request_and_emits_openai_delta():
    tool_request = object()

    class ToolCall:
        @staticmethod
        def model_dump(exclude_none=True):
            del exclude_none
            return {
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }

    class Parser:
        def __init__(self):
            self.requests = []

        def extract_tool_calls_streaming(self, *args, request):
            del args
            self.requests.append(request)
            return SimpleNamespace(content=None, tool_calls=[ToolCall()])

    parser = Parser()
    output = SimpleNamespace(
        prompt_token_ids=[1, 2],
        outputs=[
            SimpleNamespace(
                index=0,
                text="<tool_call>{}</tool_call>",
                token_ids=[3],
                finish_reason="stop",
            )
        ],
    )

    async def generator():
        yield output

    async def collect():
        selected_block_ids = []
        async for selected_chunk in _stream_response(
            generator(), "test-model", "chatcmpl-1", tool_parser=parser, tool_request=tool_request
        ):
            selected_block_ids.append(selected_chunk)
        return selected_block_ids

    chunks = asyncio.run(collect())
    payload = json.loads(chunks[0].removeprefix("data: "))

    assert parser.requests == [tool_request]
    assert payload["choices"][0]["delta"]["tool_calls"][0]["id"] == "call-1"
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
