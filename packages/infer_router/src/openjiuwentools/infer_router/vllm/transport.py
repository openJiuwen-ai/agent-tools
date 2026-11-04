import asyncio
import inspect
import json
import logging
import os
import struct
import time
import uuid
import hmac
from typing import AsyncGenerator

import orjson
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from openjiuwentools.infer_router.vllm.health_check import HealthChecker
from openjiuwentools.infer_router.vllm.prometheus_logger import collect_all_metrics
from openjiuwentools.infer_router.prompt_diagnostics import (
    enabled as prompt_diagnostics_enabled,
    prompt_observation,
    trace_generation,
)

logger = logging.getLogger("jiuwen.transport")

_HEADER_SIZE = 4


class WorkerRequestBoundary:
    """Authenticate before parsing and bound actual bytes, including chunked bodies."""

    def __init__(self, app, *, api_key: str | None, max_body_size: int):
        self.app = app
        self.api_key = api_key
        self.max_body_size = max_body_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if self.api_key and scope.get("path") != "/health":
            expected = f"Bearer {self.api_key}".encode()
            if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
                return await JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"error": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
        if declared < 0 or declared > self.max_body_size:
            return await JSONResponse({"error": "Request body too large"}, status_code=413)(scope, receive, send)
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_body_size:
                return await JSONResponse({"error": "Request body too large"}, status_code=413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


def _request_summary(body: dict) -> dict:
    return {
        "messages_count": len(body.get("messages") or []),
        "tools_count": len(body.get("tools") or []),
        "stream": bool(body.get("stream", False)),
    }


# ---------------------------------------------------------------------------
# Tool call parser (vLLM ToolParser integration)
# ---------------------------------------------------------------------------


def _resolve_tool_parser_cls(parser_name: str | None):
    """根据名称从 vLLM ToolParserManager 加载 tool parser 类。"""
    if not parser_name:
        return None
    from vllm.tool_parsers import ToolParserManager

    return ToolParserManager.get_tool_parser(parser_name)


def _resolve_reasoning_parser_cls(parser_name: str | None):
    """根据名称从 vLLM ReasoningParserManager 加载 reasoning parser 类。"""
    if not parser_name:
        return None
    from vllm.reasoning import ReasoningParserManager

    return ReasoningParserManager.get_reasoning_parser(parser_name)


def _infer_qwen_parser_names(model_names: list[str]) -> tuple[str | None, str | None]:
    """为已确认格式的 Qwen 文本生成模型推断 parser。"""
    normalized = " ".join(str(name).lower() for name in model_names if name)
    unsupported_variants = (
        "qwen3-vl",
        "qwen3_vl",
        "qwen3-next",
        "qwen3_next",
        "qwen3.5",
        "embedding",
        "reranker",
        "asr",
        "omni",
    )
    if any(variant in normalized for variant in unsupported_variants):
        return None, None
    if any(name in normalized for name in ("qwen3-coder", "qwen3_coder", "qwen3coder")):
        return "qwen3_xml", None
    if "qwen3" in normalized:
        return "hermes", "qwen3"
    if any(name in normalized for name in ("qwen2.5", "qwen2_5", "qwen2-5", "qwq")):
        return "hermes", None
    return None, None


# ---------------------------------------------------------------------------
# Parser 自动识别：模型工件（config.json / chat template）优先于名字匹配
# ---------------------------------------------------------------------------

# config.json 的 model_type -> (tool_call_parser, reasoning_parser)。
# model_type 是模型作者声明的结构化字段，微调模型继承基座配置，不存在子串误伤。
_MODEL_TYPE_PARSERS: dict[str, tuple[str | None, str | None]] = {
    "qwen3": ("hermes", "qwen3"),
    "qwen3_moe": ("hermes", "qwen3"),
    "qwen3_coder": ("qwen3_xml", None),
    "qwen2": ("hermes", None),
    "qwen2_moe": ("hermes", None),
}

# 工件明确指向的变体：不猜，且不再回退名字匹配。
_REFUSED_MODEL_TYPES = {
    "qwen3_vl",
    "qwen3_next",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen2.5_vl",
}

# chat template 内容签名（格式级证据）。顺序即优先级：更具体的在前。
_TOOL_TEMPLATE_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("qwen3_xml", ("<tool_call_start>",)),
    ("hermes", ("<tool_call>",)),
)
_REASONING_TEMPLATE_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (("qwen3", ("enable_thinking", "<think>")),)


def _load_artifact_json(model_path: str, filename: str) -> dict | None:
    """直接读取模型目录下的 JSON 工件，避免引入 transformers 依赖和网络访问。"""
    path = os.path.join(model_path, filename)
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _extract_chat_template(model_path: str, tokenizer_config: dict | None) -> str:
    """chat template 可能内嵌在 tokenizer_config.json，也可能独立为 chat_template.jinja。"""
    if tokenizer_config:
        template = tokenizer_config.get("chat_template")
        if isinstance(template, str):
            return template
        if isinstance(template, list):
            # transformers 多模板格式：合并全部渲染串，签名匹配足够。
            parts = [item.get("template", "") for item in template if isinstance(item, dict)]
            joined = "\n".join(part for part in parts if part)
            if joined:
                return joined
    try:
        with open(
            os.path.join(model_path, "chat_template.jinja"),
            encoding="utf-8",
        ) as handle:
            return handle.read()
    except OSError:
        return ""


def _detect_from_template(template: str) -> tuple[str | None, str | None]:
    """按 chat template 中的格式标记识别 parser。"""
    if not template:
        return None, None
    tool_parser = None
    for parser, markers in _TOOL_TEMPLATE_SIGNATURES:
        if any(marker in template for marker in markers):
            tool_parser = parser
            break
    reasoning_parser = None
    for parser, markers in _REASONING_TEMPLATE_SIGNATURES:
        if any(marker in template for marker in markers):
            reasoning_parser = parser
            break
    return tool_parser, reasoning_parser


def _detect_parsers_from_artifacts(
    model_path: str | None,
) -> tuple[tuple[str | None, str | None], str]:
    """从本地模型工件推断 parser：config.json 定家族，chat template 定格式。

    返回 ((tool_parser, reasoning_parser), source)。source == "unavailable"
    表示本地工件缺失/不可读（hub id、路径不存在），此时调用方可回退名字
    匹配；"refused:*" 表示工件明确指向不支持自动识别的变体，"no_match:*"
    表示工件可读但无法得出结论——两者都不应再回退名字匹配，否则会重新
    引入按目录名误猜的问题。
    """
    if not model_path or not os.path.isdir(model_path):
        return (None, None), "unavailable"
    config = _load_artifact_json(model_path, "config.json")
    tokenizer_config = _load_artifact_json(model_path, "tokenizer_config.json")
    model_type = str((config or {}).get("model_type") or "").lower()
    template = _extract_chat_template(model_path, tokenizer_config)
    if not model_type and not template:
        return (None, None), "unavailable"
    if model_type in _REFUSED_MODEL_TYPES:
        return (None, None), f"refused:model_type={model_type}"

    family_tool, family_reasoning = _MODEL_TYPE_PARSERS.get(model_type, (None, None))
    template_tool, template_reasoning = _detect_from_template(template)

    # 模板是格式级证据，优先于家族推断；家族结论只补齐模板未覆盖的部分。
    tool = template_tool or family_tool
    reasoning = template_reasoning or family_reasoning
    # 家族认为支持但模板存在且没有对应标记：模板被定制过，宁可不猜。
    if template and family_tool and not template_tool:
        tool = None
    if template and family_reasoning and not template_reasoning:
        reasoning = None

    if tool is None and reasoning is None:
        return (None, None), f"no_match:model_type={model_type or 'unknown'}"
    source_parts = [f"model_type={model_type or 'unknown'}"]
    if template_tool or template_reasoning:
        source_parts.append("template_signatures")
    else:
        source_parts.append("family_map")
    return (tool, reasoning), "+".join(source_parts)


def _resolve_worker_parser_settings(args) -> tuple[str | None, str | None, bool]:
    """解析 worker parser 配置。

    优先级：显式配置 > 模型工件检测（config.json + chat template）>
    模型名匹配（仅当本地工件不可读时兜底）。
    """
    tool_parser_name = getattr(args, "tool_call_parser", None)
    reasoning_parser_name = getattr(args, "reasoning_parser", None)
    auto_detect = bool(getattr(args, "auto_detect_parsers", True))
    model_path = getattr(args, "model", None)
    if auto_detect and (tool_parser_name is None or reasoning_parser_name is None):
        (inferred_tool, inferred_reasoning), source = _detect_parsers_from_artifacts(model_path)
        if source == "unavailable":
            # 无本地工件（hub id / 路径不存在）：退回模型名匹配。
            inferred_tool, inferred_reasoning = _infer_qwen_parser_names(
                [
                    getattr(args, "served_model_name", None),
                    getattr(args, "model", None),
                ]
            )
            source = "model_name_fallback" if inferred_tool or inferred_reasoning else "no_match"
        logger.info(
            "Parser auto-detection: model=%s source=%s tool=%r reasoning=%r",
            model_path,
            source,
            inferred_tool,
            inferred_reasoning,
        )
        if tool_parser_name is None:
            tool_parser_name = inferred_tool
        if reasoning_parser_name is None:
            reasoning_parser_name = inferred_reasoning

    configured_auto_tools = getattr(args, "enable_auto_tools", None)
    enable_auto_tools = tool_parser_name is not None if configured_auto_tools is None else bool(configured_auto_tools)
    return tool_parser_name, reasoning_parser_name, enable_auto_tools


def _get_tool_call_settings(body: dict, enable_auto_tools: bool) -> tuple[list | None, object, bool]:
    """Return tools, tool_choice, and whether tools should be injected/parsed."""
    tools = body.get("tools", None)
    tool_choice = body.get("tool_choice", "none")
    if not tools:
        return tools, tool_choice, False

    if tool_choice == "none":
        return tools, tool_choice, False
    if tool_choice == "auto":
        return tools, tool_choice, enable_auto_tools
    if tool_choice == "required" or isinstance(tool_choice, dict):
        return tools, tool_choice, True

    return tools, tool_choice, False


def _build_tool_parser_request(body: dict):
    """构造 vLLM parser 需要的 OpenAI ChatCompletionRequest。"""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest as VLLMChatCompletionRequest,
    )

    fields = VLLMChatCompletionRequest.model_fields
    compatible_body = {key: value for key, value in body.items() if key in fields}
    return VLLMChatCompletionRequest.model_validate(compatible_body)


def _requested_max_tokens(body: dict) -> int | None:
    """按 OpenAI Chat Completions 语义解析显式输出上限。"""
    max_completion_tokens = body.get("max_completion_tokens")
    if max_completion_tokens is not None:
        return int(max_completion_tokens)
    max_tokens = body.get("max_tokens")
    return int(max_tokens) if max_tokens is not None else None


def _engine_max_model_len(engine) -> int | None:
    """从不同 vLLM AsyncLLM 包装层读取模型上下文长度。"""
    candidates = (
        getattr(engine, "vllm_config", None),
        getattr(getattr(engine, "engine_core", None), "vllm_config", None),
        getattr(getattr(engine, "llm_engine", None), "vllm_config", None),
    )
    for config in candidates:
        model_config = getattr(config, "model_config", None)
        value = getattr(model_config, "max_model_len", None)
        if value is not None:
            return int(value)
    value = getattr(getattr(engine, "model_config", None), "max_model_len", None)
    return int(value) if value is not None else None


def _resolve_max_tokens(engine, body: dict, prompt_tokens: int) -> int | None:
    """模拟 vLLM OpenAI server 的动态输出上限。"""
    requested = _requested_max_tokens(body)
    max_model_len = _engine_max_model_len(engine)
    if max_model_len is None:
        return requested
    if prompt_tokens > max_model_len:
        raise ValueError(f"Input length ({prompt_tokens}) exceeds model context length ({max_model_len})")
    model_config = getattr(engine, "model_config", None)
    get_defaults = getattr(model_config, "get_diff_sampling_param", None)
    default_sampling_params = get_defaults() if callable(get_defaults) else {}
    try:
        from vllm.entrypoints.utils import get_max_tokens

        return get_max_tokens(
            max_model_len,
            requested,
            prompt_tokens,
            default_sampling_params,
        )
    except ImportError:
        fallback = requested
        if fallback is None:
            fallback = default_sampling_params.get("max_tokens")
        limits = [max_model_len - prompt_tokens]
        if fallback is not None:
            limits.append(int(fallback))
        return min(limits)


def _make_engine_request_id(request_id: str, body: dict) -> str:
    cache_salt = body.get("cache_salt")
    if body.get("cache_sharing") and cache_salt:
        return json.dumps(
            {
                "request_id": request_id,
                "sharing_cache_salt": cache_salt,
            }
        )
    return request_id


def _mooncake_trace_headers(logical_request_id: str) -> dict[str, str] | None:
    if not logical_request_id:
        return None
    return {"x-jiuwen-request-id": logical_request_id}


def _sampling_extra_args(body: dict) -> dict | None:
    """Forward native vLLM connector metadata into EngineCore requests."""
    kv_transfer_params = body.get("kv_transfer_params")
    if not isinstance(kv_transfer_params, dict):
        return None
    return {"kv_transfer_params": kv_transfer_params}


def _attach_kv_transfer_params(response: dict, final_output: object) -> dict:
    """Expose connector-generated P metadata to the Router data plane."""
    kv_transfer_params = getattr(final_output, "kv_transfer_params", None)
    if isinstance(kv_transfer_params, dict):
        response["kv_transfer_params"] = kv_transfer_params
    return response


def _metric_timestamp(metrics: object, name: str) -> float:
    value = getattr(metrics, name, 0.0) if metrics is not None else 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _log_mooncake_first_output_timing(
    output: object,
    *,
    logical_request_id: str,
    engine_request_id: str,
    http_received_at: float,
) -> None:
    if not logical_request_id or http_received_at <= 0:
        return
    first_byte_at = time.time()
    metrics = getattr(output, "metrics", None)
    arrival_at = _metric_timestamp(metrics, "arrival_time")
    first_scheduled_at = _metric_timestamp(metrics, "first_scheduled_time")
    first_token_at = _metric_timestamp(metrics, "first_token_time") or first_byte_at
    logger.info(
        "[MooncakeDRequestTiming] first output request=%s engine_request_id=%s "
        "d_http_received_to_engine_arrival_ms=%.3f "
        "d_http_received_to_first_scheduled_ms=%.3f "
        "first_scheduled_to_first_token_ms=%.3f "
        "first_token_to_first_byte_ms=%.3f "
        "first_scheduled_to_first_byte_ms=%.3f "
        "d_http_received_to_first_byte_ms=%.3f timing_source=%s",
        logical_request_id,
        engine_request_id,
        max(0.0, (arrival_at - http_received_at) * 1000.0) if arrival_at else 0.0,
        (max(0.0, (first_scheduled_at - http_received_at) * 1000.0) if first_scheduled_at else 0.0),
        (max(0.0, (first_token_at - first_scheduled_at) * 1000.0) if first_scheduled_at else 0.0),
        max(0.0, (first_byte_at - first_token_at) * 1000.0),
        (max(0.0, (first_byte_at - first_scheduled_at) * 1000.0) if first_scheduled_at else 0.0),
        max(0.0, (first_byte_at - http_received_at) * 1000.0),
        "vllm_metrics" if first_scheduled_at else "worker_first_output",
    )


# ---------------------------------------------------------------------------
# Shared inference logic
# ---------------------------------------------------------------------------


async def do_chat_completion(
    engine,
    body: dict,
    model_name: str,
    tool_parser_cls=None,
    reasoning_parser_cls=None,
    enable_auto_tools: bool = False,
    request_id: str | None = None,
) -> dict:
    """执行 chat completion 推理。

    参数:
        engine: vLLM 异步推理引擎。
        body: OpenAI chat completion 请求体。
        model_name: 响应中使用的模型名称。
        tool_parser_cls: 可选的工具调用解析器类型。
        reasoning_parser_cls: 可选的推理内容解析器类型。
        enable_auto_tools: 是否允许自动工具调用。
        request_id: Router 传入的 engine request ID；为空时本地生成。

    返回:
        OpenAI 兼容的 chat completion 响应字典。
    """
    from vllm.sampling_params import SamplingParams

    messages = body.get("messages", [])
    tools, tool_choice, inject_tools = _get_tool_call_settings(body, enable_auto_tools)
    parse_tool_calls = inject_tools and tool_parser_cls is not None

    tokenizer = engine.get_tokenizer()
    if inspect.isawaitable(tokenizer):
        tokenizer = await tokenizer

    template_kwargs = {}
    if inject_tools:
        template_kwargs["tools"] = tools
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )

    prompt_token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    )
    prompt_tokens = len(prompt_token_ids)
    last_msg = messages[-1] if messages else {}
    req_max_tokens = _resolve_max_tokens(engine, body, prompt_tokens)
    logger.info(
        "[ChatCompletion] messages=%d last_role=%s tools=%d tool_choice=%s parse_tool_calls=%s "
        "enable_auto_tools=%s prompt_tokens=%d max_tokens=%r",
        len(messages),
        last_msg.get("role", "?"),
        len(tools) if tools else 0,
        tool_choice,
        parse_tool_calls,
        enable_auto_tools,
        prompt_tokens,
        req_max_tokens,
    )
    logger.debug("[ChatCompletion] summary=%s", _request_summary(body))

    sampling_params = SamplingParams(
        max_tokens=req_max_tokens,
        min_tokens=body.get("min_tokens", 0),
        temperature=body.get("temperature", 1.0),
        top_p=body.get("top_p", 1.0),
        n=body.get("n", 1),
        ignore_eos=body.get("ignore_eos", False),
        extra_args=_sampling_extra_args(body),
    )
    if body.get("stop"):
        sampling_params.stop = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]

    request_id = request_id or f"chatcmpl-{uuid.uuid4()}"
    engine_request_id = _make_engine_request_id(request_id, body)
    if prompt_diagnostics_enabled():
        logger.info(
            "[PDPromptDiagnostics] %s",
            json.dumps(
                prompt_observation(
                    body,
                    prompt_token_ids,
                    stage="worker_nonstream",
                    request_id=request_id,
                    tokenizer=tokenizer,
                    template_kwargs={**template_kwargs, "add_generation_prompt": True},
                    engine_request_id=engine_request_id,
                    inject_tools=inject_tools,
                    enable_auto_tools=enable_auto_tools,
                ),
                sort_keys=True,
            ),
        )
    generator = engine.generate(
        prompt=prompt_text,
        sampling_params=sampling_params,
        request_id=engine_request_id,
    )
    generator = trace_generation(generator, logger, request_id, engine_request_id)

    final_output = None
    async for output in generator:
        final_output = output

    if final_output is None:
        return {"error": "No output generated"}

    reasoning_parser = reasoning_parser_cls(tokenizer) if reasoning_parser_cls else None
    tool_request = _build_tool_parser_request(body) if parse_tool_calls else None

    choices = []
    called_tool_names: list[str] = []
    for i, out in enumerate(final_output.outputs):
        if reasoning_parser:
            _, content_text = reasoning_parser.extract_reasoning(out.text, request=None)
            if content_text is None and reasoning_parser.start_token not in out.text:
                content_text = out.text
        else:
            content_text = out.text

        if parse_tool_calls and content_text:
            parser = tool_parser_cls(tokenizer)
            extracted = parser.extract_tool_calls(content_text, request=tool_request)
            if extracted.tools_called:
                tool_calls = [tc.model_dump() for tc in extracted.tool_calls]
                called_tool_names.extend(
                    tc.get("function", {}).get("name") for tc in tool_calls if tc.get("function", {}).get("name")
                )
                choices.append(
                    {
                        "index": i,
                        "message": {"role": "assistant", "content": extracted.content, "tool_calls": tool_calls},
                        "finish_reason": "tool_calls",
                    }
                )
            else:
                choices.append(
                    {
                        "index": i,
                        "message": {"role": "assistant", "content": extracted.content or content_text},
                        "finish_reason": out.finish_reason,
                    }
                )
        else:
            choices.append(
                {
                    "index": i,
                    "message": {"role": "assistant", "content": content_text or ""},
                    "finish_reason": out.finish_reason,
                }
            )

    prompt_token_count = len(final_output.prompt_token_ids)
    completion_token_count = sum(len(out.token_ids) for out in final_output.outputs)

    if called_tool_names:
        logger.info(
            "[ToolCall] request=%s stream=false calls=%d names=%s",
            request_id,
            len(called_tool_names),
            called_tool_names,
        )

    return _attach_kv_transfer_params(
        {
            "id": request_id,
            "object": "chat.completion",
            "model": model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        },
        final_output,
    )


async def do_completion(
    engine,
    body: dict,
    model_name: str,
    request_id: str | None = None,
) -> dict:
    """执行 text completion 推理。

    参数:
        engine: vLLM 异步推理引擎。
        body: OpenAI text completion 请求体。
        model_name: 响应中使用的模型名称。
        request_id: Router 传入的 engine request ID；为空时本地生成。

    返回:
        OpenAI 兼容的 text completion 响应字典。
    """
    from vllm.sampling_params import SamplingParams

    prompt = body.get("prompt", "")

    tokenizer = engine.get_tokenizer()
    if inspect.isawaitable(tokenizer):
        tokenizer = await tokenizer
    if isinstance(prompt, list):
        prompt_text = tokenizer.decode(prompt)
        prompt_tokens = len(prompt)
    else:
        prompt_text = prompt
        prompt_tokens = len(tokenizer.encode(prompt_text))

    sampling_params = SamplingParams(
        max_tokens=_resolve_max_tokens(engine, body, prompt_tokens),
        min_tokens=body.get("min_tokens", 0),
        temperature=body.get("temperature", 1.0),
        top_p=body.get("top_p", 1.0),
        n=body.get("n", 1),
        ignore_eos=body.get("ignore_eos", False),
        extra_args=_sampling_extra_args(body),
    )
    if body.get("stop"):
        sampling_params.stop = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]

    request_id = request_id or f"cmpl-{uuid.uuid4().hex}"
    generator = engine.generate(
        prompt=prompt_text,
        sampling_params=sampling_params,
        request_id=request_id,
    )

    final_output = None
    async for output in generator:
        final_output = output

    if final_output is None:
        return {"error": "No output generated"}

    echo = body.get("echo", False)
    choices = []
    for i, out in enumerate(final_output.outputs):
        text = (prompt_text + out.text) if echo else out.text
        choices.append(
            {
                "index": i,
                "text": text,
                "finish_reason": out.finish_reason,
                "logprobs": None,
            }
        )

    prompt_token_count = len(final_output.prompt_token_ids)
    completion_token_count = sum(len(out.token_ids) for out in final_output.outputs)

    return _attach_kv_transfer_params(
        {
            "id": request_id,
            "object": "text_completion",
            "model": model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        },
        final_output,
    )


# ---------------------------------------------------------------------------
# HTTP server (FastAPI + uvicorn)
# ---------------------------------------------------------------------------


def create_http_app(args, engine) -> FastAPI:
    app = FastAPI(title="Jiuwen vLLM Worker")
    app.add_middleware(
        WorkerRequestBoundary,
        api_key=getattr(args, "api_key", None) or os.environ.get("JIUWEN_WORKER_API_KEY"),
        max_body_size=max(1, int(getattr(args, "max_request_body_bytes", 64 * 1024 * 1024))),
    )
    model_name = args.served_model_name or args.model
    checker = HealthChecker(engine, model_name)
    tool_parser_name, reasoning_parser_name, enable_auto_tools = _resolve_worker_parser_settings(args)
    logger.info(
        "Parser config: tool_call_parser=%r, reasoning_parser=%r, enable_auto_tools=%s",
        tool_parser_name,
        reasoning_parser_name,
        enable_auto_tools,
    )
    tool_parser_cls = _resolve_tool_parser_cls(tool_parser_name)
    reasoning_parser_cls = _resolve_reasoning_parser_cls(reasoning_parser_name)
    if tool_parser_cls:
        logger.info("Tool call parser loaded: %s", tool_parser_cls.__name__)
    if reasoning_parser_cls:
        logger.info("Reasoning parser loaded: %s", reasoning_parser_cls.__name__)

    @app.get("/health")
    async def health():
        healthy, detail = await checker.check()
        status = 200 if healthy else 503
        return JSONResponse({"status": "healthy" if healthy else "unhealthy", "detail": detail}, status_code=status)

    @app.get("/metrics")
    async def metrics():
        return Response(collect_all_metrics(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": model_name, "object": "model", "owned_by": "jiuwen"}],
            }
        )

    @app.post("/kvbm/prefetch")
    async def kvbm_prefetch(raw_request: Request):
        """将 router 的 KVBM 预取请求转发到 engine-core 子进程。

        参数:
            raw_request: FastAPI 原始请求，JSON 中支持 block_hashes、request_id、
                deadline_ms 和 deadline_epoch_ms。

        返回:
            JSONResponse，包含 engine-core 预取结果或不可用原因。
        """

        if not getattr(args, "kvbm_worker_enabled", False):
            return JSONResponse(
                {"ok": False, "error": "worker KVBM is disabled"},
                status_code=503,
            )
        body = await raw_request.json()
        prefetch_port = getattr(args, "kvbm_prefetch_port", None)
        if prefetch_port is None:
            prefetch_port = int(getattr(args, "port", 8001)) + 10000
        prefetch_port = int(prefetch_port)
        if prefetch_port <= 0:
            return JSONResponse(
                {"ok": False, "error": "worker KVBM prefetch endpoint is disabled"},
                status_code=503,
            )

        host = str(getattr(args, "kvbm_prefetch_host", "127.0.0.1"))
        deadline_ms = int(body.get("deadline_ms") or 100)
        timeout = httpx.Timeout((max(deadline_ms, 1) + 100) / 1000.0, connect=0.1)
        url = f"http://{host}:{prefetch_port}/prefetch"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=body)
            return JSONResponse(response.json(), status_code=response.status_code)
        except Exception as exc:
            logger.warning("KVBM prefetch forward failed: url=%s error=%s", url, exc)
            return JSONResponse(
                {"ok": False, "error": "KVBM prefetch service unavailable"},
                status_code=503,
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(raw_request: Request):
        http_received_at = time.time()
        body = await raw_request.json()
        forwarded_request_id = raw_request.headers.get("X-Request-Id")
        logical_request_id = raw_request.headers.get("X-Jiuwen-Request-Id", "")

        if body.get("stream", False):
            request_id = forwarded_request_id or f"chatcmpl-{uuid.uuid4().hex}"
            engine_request_id = _make_engine_request_id(request_id, body)
            from vllm.sampling_params import SamplingParams

            messages = body.get("messages", [])
            tools, tool_choice, inject_tools = _get_tool_call_settings(body, enable_auto_tools)
            parse_tool_calls = inject_tools and tool_parser_cls is not None
            tokenizer = engine.get_tokenizer()
            if inspect.isawaitable(tokenizer):
                tokenizer = await tokenizer
            template_kwargs = {}
            if inject_tools:
                template_kwargs["tools"] = tools
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )

            prompt_token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **template_kwargs,
            )
            prompt_tokens = len(prompt_token_ids)
            last_msg = messages[-1] if messages else {}
            if prompt_diagnostics_enabled():
                logger.info(
                    "[PDPromptDiagnostics] %s",
                    json.dumps(
                        prompt_observation(
                            body,
                            prompt_token_ids,
                            stage="worker_stream",
                            request_id=request_id,
                            tokenizer=tokenizer,
                            template_kwargs={**template_kwargs, "add_generation_prompt": True},
                            engine_request_id=engine_request_id,
                            logical_request_id=logical_request_id,
                            inject_tools=inject_tools,
                            enable_auto_tools=enable_auto_tools,
                        ),
                        sort_keys=True,
                    ),
                )
            req_max_tokens = _resolve_max_tokens(engine, body, prompt_tokens)
            logger.info(
                "[ChatCompletion][Stream] id=%s messages=%d last_role=%s tools=%d tool_choice=%s "
                "enable_auto_tools=%s parse_tool_calls=%s prompt_tokens=%d max_tokens=%r",
                request_id,
                len(messages),
                last_msg.get("role", "?"),
                len(tools) if tools else 0,
                tool_choice,
                enable_auto_tools,
                parse_tool_calls,
                prompt_tokens,
                req_max_tokens,
            )
            logger.debug("[ChatCompletion][Stream] summary=%s", _request_summary(body))

            sampling_params = SamplingParams(
                max_tokens=req_max_tokens,
                min_tokens=body.get("min_tokens", 0),
                temperature=body.get("temperature", 1.0),
                top_p=body.get("top_p", 1.0),
                n=body.get("n", 1),
                ignore_eos=body.get("ignore_eos", False),
                extra_args=_sampling_extra_args(body),
            )
            if body.get("stop"):
                sampling_params.stop = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
            generator = engine.generate(
                prompt=prompt_text,
                sampling_params=sampling_params,
                request_id=engine_request_id,
                trace_headers=_mooncake_trace_headers(logical_request_id),
            )
            stream_options = body.get("stream_options") or {}
            include_usage = stream_options.get("include_usage", True)
            tool_parser = tool_parser_cls(tokenizer) if parse_tool_calls else None
            generator = trace_generation(generator, logger, logical_request_id or request_id, engine_request_id)
            tool_request = _build_tool_parser_request(body) if parse_tool_calls else None
            reasoning_parser = reasoning_parser_cls(tokenizer) if reasoning_parser_cls else None
            return StreamingResponse(
                _stream_response(
                    generator,
                    model_name,
                    request_id,
                    include_usage,
                    tool_parser,
                    reasoning_parser,
                    tool_request,
                    logical_request_id=logical_request_id,
                    engine_request_id=engine_request_id,
                    http_received_at=http_received_at,
                ),
                media_type="text/event-stream",
            )

        result = await do_chat_completion(
            engine,
            body,
            model_name,
            tool_parser_cls,
            reasoning_parser_cls,
            enable_auto_tools,
            request_id=forwarded_request_id,
        )
        status = 500 if "error" in result else 200
        return JSONResponse(result, status_code=status)

    @app.post("/v1/completions")
    async def completions(raw_request: Request):
        http_received_at = time.time()
        body = await raw_request.json()
        forwarded_request_id = raw_request.headers.get("X-Request-Id")
        logical_request_id = raw_request.headers.get("X-Jiuwen-Request-Id", "")

        if body.get("stream", False):
            request_id = forwarded_request_id or f"cmpl-{uuid.uuid4().hex}"
            from vllm.sampling_params import SamplingParams

            prompt = body.get("prompt", "")
            tokenizer = engine.get_tokenizer()
            if inspect.isawaitable(tokenizer):
                tokenizer = await tokenizer
            if isinstance(prompt, list):
                prompt_text = tokenizer.decode(prompt)
                prompt_tokens = len(prompt)
            else:
                prompt_text = prompt
                prompt_tokens = len(tokenizer.encode(prompt_text))

            sampling_params = SamplingParams(
                max_tokens=_resolve_max_tokens(engine, body, prompt_tokens),
                min_tokens=body.get("min_tokens", 0),
                temperature=body.get("temperature", 1.0),
                top_p=body.get("top_p", 1.0),
                n=body.get("n", 1),
                ignore_eos=body.get("ignore_eos", False),
                extra_args=_sampling_extra_args(body),
            )
            if body.get("stop"):
                sampling_params.stop = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
            engine_request_id = request_id
            generator = engine.generate(
                prompt=prompt_text,
                sampling_params=sampling_params,
                request_id=engine_request_id,
                trace_headers=_mooncake_trace_headers(logical_request_id),
            )
            stream_options = body.get("stream_options") or {}
            include_usage = stream_options.get("include_usage", True)
            return StreamingResponse(
                _stream_completion_response(
                    generator,
                    model_name,
                    request_id,
                    include_usage,
                    logical_request_id=logical_request_id,
                    engine_request_id=engine_request_id,
                    http_received_at=http_received_at,
                ),
                media_type="text/event-stream",
            )

        result = await do_completion(
            engine,
            body,
            model_name,
            request_id=forwarded_request_id,
        )
        status = 500 if "error" in result else 200
        return JSONResponse(result, status_code=status)

    @app.post("/release_kv_cache")
    async def release_kv_cache(raw_request: Request):
        """释放指定消息范围的 KV 缓存块，转发给 vLLM 插件的 release_kv_cache。"""
        if not hasattr(engine, "engine_core") or not hasattr(engine.engine_core, "release_kv_cache"):
            return JSONResponse(
                {"error": "release_kv_cache not supported, plugin not loaded"},
                status_code=501,
            )

        body = await raw_request.json()
        messages = body.get("messages", [])
        messages_released_index = body.get("messages_released_index", 0)
        cache_salt = body.get("cache_salt", "")
        tools = body.get("tools", None)

        message_length = len(messages)
        message_begin = max(0, min(messages_released_index, message_length))

        if message_begin >= message_length:
            return JSONResponse({"cache_salt": cache_salt, "block_released": 0})

        tokenizer = engine.get_tokenizer()
        if inspect.isawaitable(tokenizer):
            tokenizer = await tokenizer

        template_kwargs = {}
        if tools:
            template_kwargs["tools"] = tools
        before_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )

        if message_begin > 0:
            after_token_ids = tokenizer.apply_chat_template(
                messages[:message_begin],
                tokenize=True,
                add_generation_prompt=True,
                **template_kwargs,
            )
        else:
            after_token_ids = []

        if len(before_token_ids) <= len(after_token_ids):
            return JSONResponse({"cache_salt": cache_salt, "block_released": 0})

        released_token_index = len(after_token_ids)
        if after_token_ids:
            last_match_index_after = 0
            last_match_index_before = 0
            match_count = len(after_token_ids)
            eos_token_count = 5
            if match_count > eos_token_count:
                match_count -= eos_token_count
            for idx, val in enumerate(before_token_ids):
                if val == after_token_ids[last_match_index_after]:
                    last_match_index_before = idx
                    last_match_index_after += 1
                    if last_match_index_after >= match_count:
                        break
            released_token_index = last_match_index_before + 1

        try:
            from vllm.sampling_params import SamplingParams
            from vllm.v1.engine import EngineCoreRequest
            from vllm.v1.serial_utils import MsgpackEncoder

            engine_request = EngineCoreRequest(
                request_id=f"release-{uuid.uuid4()}",
                prompt_token_ids=before_token_ids,
                mm_features=None,
                sampling_params=SamplingParams(),
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=cache_salt or None,
                data_parallel_rank=None,
            )

            encoded = MsgpackEncoder().encode(engine_request)
            released = await engine.engine_core.release_kv_cache(
                cache_salt,
                [(encoded, released_token_index)],
            )

            logger.info(
                "release_kv_cache: salt=%s released_index=%d blocks=%d",
                cache_salt,
                released_token_index,
                released,
            )
            return JSONResponse({"cache_salt": cache_salt, "block_released": released})
        except Exception:
            logger.exception("release_kv_cache failed")
            return JSONResponse({"error": "release_kv_cache failed"}, status_code=500)

    return app


async def _stream_response(
    generator: AsyncGenerator,
    model_name: str,
    request_id: str,
    include_usage: bool = False,
    tool_parser=None,
    reasoning_parser=None,
    tool_request=None,
    *,
    logical_request_id: str = "",
    engine_request_id: str = "",
    http_received_at: float = 0.0,
):
    """流式输出 chat completion 的 SSE 响应。

    vLLM 的 CompletionOutput.text 是累积值（每次包含从开头到当前的全部文本），
    需要通过 previous_texts 追踪已发送内容，计算增量 delta 发送给客户端。
    首个 chunk 的 delta 包含 role="assistant"，符合 OpenAI 流式协议规范。
    当 tool_parser 不为 None 时，使用 vLLM ToolParser 进行流式 tool call 解析。
    当 reasoning_parser 不为 None 时，先分离推理内容再做 tool call 解析。
    """
    previous_texts: dict[int, str] = {}
    previous_token_ids: dict[int, tuple] = {}
    thinking_started: dict[int, bool] = {}
    reasoning_ended: dict[int, bool] = {}
    post_think_texts: dict[int, str] = {}
    first_chunk = True
    final_output = None
    has_tool_calls = False
    tool_call_names: dict[int, str] = {}
    first_output_logged = False

    async for output in generator:
        if not first_output_logged:
            _log_mooncake_first_output_timing(
                output,
                logical_request_id=logical_request_id,
                engine_request_id=engine_request_id or request_id,
                http_received_at=http_received_at,
            )
            first_output_logged = True
        final_output = output
        for out in output.outputs:
            prev_text = previous_texts.get(out.index, "")
            prev_tokens = previous_token_ids.get(out.index, ())
            current_text = out.text
            current_tokens = out.token_ids
            delta_text = current_text[len(prev_text):]
            delta_tokens = current_tokens[len(prev_tokens):]
            previous_texts[out.index] = current_text
            previous_token_ids[out.index] = current_tokens

            content = None
            tc_deltas = []

            if tool_parser is not None and reasoning_parser is not None:
                ended = reasoning_ended.get(out.index, False)
                started = thinking_started.get(out.index, False)

                if not started and not ended:
                    start_detected = reasoning_parser.start_token_id in tuple(delta_tokens)
                    if not start_detected:
                        start_token = reasoning_parser.start_token
                        if start_token in current_text and start_token not in prev_text:
                            start_detected = True
                    if start_detected:
                        thinking_started[out.index] = True
                        is_end = reasoning_parser.is_reasoning_end_streaming(
                            current_tokens,
                            delta_tokens,
                        )
                        end_token = reasoning_parser.end_token
                        if not is_end and end_token in current_text and end_token not in prev_text:
                            is_end = True
                        if is_end:
                            reasoning_ended[out.index] = True
                            post_think_texts[out.index] = ""
                        if delta_text:
                            content = delta_text
                    else:
                        delta_msg = tool_parser.extract_tool_calls_streaming(
                            prev_text,
                            current_text,
                            delta_text,
                            prev_tokens,
                            current_tokens,
                            delta_tokens,
                            request=tool_request,
                        )
                        if delta_msg:
                            content = delta_msg.content
                            tc_deltas = delta_msg.tool_calls or []

                elif started and not ended:
                    is_end = reasoning_parser.is_reasoning_end_streaming(
                        current_tokens,
                        delta_tokens,
                    )
                    end_token = reasoning_parser.end_token
                    if not is_end and end_token in current_text and end_token not in prev_text:
                        is_end = True
                    if is_end:
                        reasoning_ended[out.index] = True
                        post_think_texts[out.index] = ""
                    if delta_text:
                        content = delta_text

                elif ended:
                    prev_post = post_think_texts.get(out.index, "")
                    cur_post = prev_post + delta_text
                    post_think_texts[out.index] = cur_post
                    delta_msg = tool_parser.extract_tool_calls_streaming(
                        prev_post,
                        cur_post,
                        delta_text,
                        prev_tokens,
                        current_tokens,
                        delta_tokens,
                        request=tool_request,
                    )
                    if delta_msg:
                        content = delta_msg.content
                        tc_deltas = delta_msg.tool_calls or []

            elif tool_parser is not None:
                delta_msg = tool_parser.extract_tool_calls_streaming(
                    prev_text,
                    current_text,
                    delta_text,
                    prev_tokens,
                    current_tokens,
                    delta_tokens,
                    request=tool_request,
                )
                if delta_msg:
                    content = delta_msg.content
                    tc_deltas = delta_msg.tool_calls or []

            elif reasoning_parser is not None:
                content = delta_text if delta_text else None

            else:
                content = delta_text if delta_text else None

            if not content and not tc_deltas and out.finish_reason is None:
                continue

            delta: dict = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if content:
                delta["content"] = content
            if tc_deltas:
                has_tool_calls = True
                for tc in tc_deltas:
                    fn = getattr(getattr(tc, "function", None), "name", None)
                    if fn:
                        tool_call_names[tc.index] = fn
                delta["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in tc_deltas]

            finish_reason = None
            if out.finish_reason is not None:
                finish_reason = "tool_calls" if has_tool_calls else out.finish_reason

            if delta or finish_reason is not None:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{"index": out.index, "delta": delta, "finish_reason": finish_reason}],
                }
                yield f"data: {orjson.dumps(chunk).decode()}\n\n"

    if has_tool_calls:
        logger.info(
            "[ToolCall] request=%s stream=true calls=%d names=%s",
            request_id,
            len(tool_call_names),
            sorted(tool_call_names.values()),
        )

    if include_usage and final_output is not None:
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = sum(len(output.token_ids) for output in final_output.outputs)
        usage_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": model_name,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {orjson.dumps(usage_chunk).decode()}\n\n"

    yield "data: [DONE]\n\n"


async def _stream_completion_response(
    generator: AsyncGenerator,
    model_name: str,
    request_id: str,
    include_usage: bool = False,
    *,
    logical_request_id: str = "",
    engine_request_id: str = "",
    http_received_at: float = 0.0,
):
    """流式输出 text completion 的 SSE 响应。

    与 _stream_response 类似，通过 previous_texts 将 vLLM 累积文本转换为增量输出。
    """
    previous_texts: dict[int, str] = {}
    final_output = None
    first_output_logged = False
    async for output in generator:
        if not first_output_logged:
            _log_mooncake_first_output_timing(
                output,
                logical_request_id=logical_request_id,
                engine_request_id=engine_request_id or request_id,
                http_received_at=http_received_at,
            )
            first_output_logged = True
        final_output = output
        for out in output.outputs:
            prev = previous_texts.get(out.index, "")
            delta_text = out.text[len(prev):]
            previous_texts[out.index] = out.text

            if not delta_text and out.finish_reason is None:
                continue

            chunk = {
                "id": request_id,
                "object": "text_completion",
                "model": model_name,
                "choices": [
                    {
                        "index": out.index,
                        "text": delta_text,
                        "finish_reason": out.finish_reason,
                        "logprobs": None,
                    }
                ],
            }
            yield f"data: {orjson.dumps(chunk).decode()}\n\n"

    if include_usage and final_output is not None:
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = sum(len(output.token_ids) for output in final_output.outputs)
        usage_chunk = {
            "id": request_id,
            "object": "text_completion",
            "model": model_name,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {orjson.dumps(usage_chunk).decode()}\n\n"

    yield "data: [DONE]\n\n"


async def serve_http(args, engine, shutdown_event) -> callable:
    """启动 HTTP 服务，返回 close 回调供优雅退出使用。"""
    app = create_http_app(args, engine)
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
    )
    server = uvicorn.Server(config)

    def _noop_signal_handlers() -> None:
        pass

    server.install_signal_handlers = _noop_signal_handlers

    log_stats_interval = getattr(args, "log_stats_interval", 10)

    async def _periodic_log_stats():
        while not shutdown_event.is_set():
            await asyncio.sleep(log_stats_interval)
            if hasattr(engine, "do_log_stats"):
                await engine.do_log_stats()

    def close():
        server.should_exit = True

    async def _serve():
        asyncio.create_task(_periodic_log_stats())
        await server.serve()

    logger.info("HTTP server starting on %s:%d", args.host, args.port)
    return close, _serve()


# ---------------------------------------------------------------------------
# TCP server (length-prefixed JSON)
# ---------------------------------------------------------------------------

_MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64 MB


async def _tcp_read_message(reader: asyncio.StreamReader) -> dict | None:
    """读取 4 字节大端长度头 + JSON body。"""
    header = await reader.readexactly(_HEADER_SIZE)
    length = struct.unpack("!I", header)[0]
    if length > _MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {length} bytes (max {_MAX_MESSAGE_SIZE})")
    payload = await reader.readexactly(length)
    return orjson.loads(payload)


def _tcp_pack_message(obj: dict) -> bytes:
    """将 dict 编码为 4 字节大端长度头 + JSON body。"""
    payload = orjson.dumps(obj)
    return struct.pack("!I", len(payload)) + payload


async def _tcp_client_handler(
    reader,
    writer,
    engine,
    model_name,
    tool_parser_cls=None,
    reasoning_parser_cls=None,
    enable_auto_tools: bool = False,
    api_key: str | None = None,
):
    addr = writer.get_extra_info("peername")
    logger.info("TCP client connected: %s", addr)
    try:
        while True:
            body = await _tcp_read_message(reader)
            if body is None:
                break
            supplied_key = body.pop("api_key", "")
            if api_key and not hmac.compare_digest(str(supplied_key), api_key):
                writer.write(_tcp_pack_message({"error": "Unauthorized"}))
                await writer.drain()
                break
            result = await do_chat_completion(
                engine, body, model_name, tool_parser_cls, reasoning_parser_cls, enable_auto_tools
            )
            writer.write(_tcp_pack_message(result))
            await writer.drain()
    except asyncio.IncompleteReadError:
        pass
    except Exception:
        logger.exception("TCP handler error for %s", addr)
    finally:
        writer.close()
        await writer.wait_closed()
        logger.info("TCP client disconnected: %s", addr)


async def serve_tcp(args, engine, shutdown_event) -> callable:
    """启动 TCP 服务，返回 close 回调供优雅退出使用。"""
    model_name = args.served_model_name or args.model
    tool_parser_name, reasoning_parser_name, enable_auto_tools = _resolve_worker_parser_settings(args)
    logger.info(
        "Parser config: tool_call_parser=%r, reasoning_parser=%r, enable_auto_tools=%s",
        tool_parser_name,
        reasoning_parser_name,
        enable_auto_tools,
    )
    tool_parser_cls = _resolve_tool_parser_cls(tool_parser_name)
    reasoning_parser_cls = _resolve_reasoning_parser_cls(reasoning_parser_name)
    tcp_server = await asyncio.start_server(
        lambda r, w: _tcp_client_handler(
            r,
            w,
            engine,
            model_name,
            tool_parser_cls,
            reasoning_parser_cls,
            enable_auto_tools,
            getattr(args, "api_key", None) or os.environ.get("JIUWEN_WORKER_API_KEY"),
        ),
        host=args.host,
        port=args.port,
    )
    logger.info("TCP server listening on %s:%d", args.host, args.port)

    async def run():
        async with tcp_server:
            await shutdown_event.wait()

    return tcp_server.close, run()
