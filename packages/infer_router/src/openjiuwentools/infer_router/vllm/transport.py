import asyncio
import inspect
import logging
import struct
import uuid
from typing import AsyncGenerator

import orjson
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from openjiuwentools.infer_router.vllm.health_check import HealthChecker
from openjiuwentools.infer_router.vllm.prometheus_logger import collect_all_metrics

logger = logging.getLogger("jiuwen.transport")

_HEADER_SIZE = 4


# ---------------------------------------------------------------------------
# Shared inference logic
# ---------------------------------------------------------------------------

async def do_chat_completion(engine, body: dict, model_name: str) -> dict:
    """执行 chat completion 推理，返回 OpenAI 兼容的响应 dict。"""
    from vllm.sampling_params import SamplingParams

    messages = body.get("messages", [])

    tokenizer = engine.get_tokenizer()
    if inspect.isawaitable(tokenizer):
        tokenizer = await tokenizer
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        max_tokens=body.get("max_tokens", 128),
        temperature=body.get("temperature", 1.0),
        top_p=body.get("top_p", 1.0),
        n=body.get("n", 1),
    )

    request_id = f"chatcmpl-{uuid.uuid4()}"
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

    choices = []
    for i, out in enumerate(final_output.outputs):
        choices.append({
            "index": i,
            "message": {"role": "assistant", "content": out.text},
            "finish_reason": out.finish_reason,
        })

    prompt_token_count = len(final_output.prompt_token_ids)
    completion_token_count = sum(len(out.token_ids) for out in final_output.outputs)

    return {
        "id": request_id,
        "object": "chat.completion",
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": prompt_token_count + completion_token_count,
        },
    }


async def do_completion(engine, body: dict, model_name: str) -> dict:
    """执行 text completion 推理，返回 OpenAI 兼容的响应 dict。"""
    from vllm.sampling_params import SamplingParams

    prompt = body.get("prompt", "")

    if isinstance(prompt, list):
        tokenizer = engine.get_tokenizer()
        if inspect.isawaitable(tokenizer):
            tokenizer = await tokenizer
        prompt_text = tokenizer.decode(prompt)
    else:
        prompt_text = prompt

    sampling_params = SamplingParams(
        max_tokens=body.get("max_tokens", 128),
        temperature=body.get("temperature", 1.0),
        top_p=body.get("top_p", 1.0),
        n=body.get("n", 1),
    )
    if body.get("stop"):
        sampling_params.stop = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]

    request_id = f"cmpl-{id(body)}"
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
        choices.append({
            "index": i,
            "text": text,
            "finish_reason": out.finish_reason,
            "logprobs": None,
        })

    prompt_token_count = len(final_output.prompt_token_ids)
    completion_token_count = sum(len(out.token_ids) for out in final_output.outputs)

    return {
        "id": request_id,
        "object": "text_completion",
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": prompt_token_count + completion_token_count,
        },
    }


# ---------------------------------------------------------------------------
# HTTP server (FastAPI + uvicorn)
# ---------------------------------------------------------------------------

def create_http_app(args, engine) -> FastAPI:
    app = FastAPI(title="Jiuwen vLLM Worker")
    model_name = args.served_model_name or args.model
    checker = HealthChecker(engine, model_name)

    @app.get("/health")
    async def health():
        healthy, detail = await checker.check()
        status = 200 if healthy else 503
        return JSONResponse({"status": "healthy" if healthy else "unhealthy", "detail": detail},
                            status_code=status)

    @app.get("/metrics")
    async def metrics():
        return Response(collect_all_metrics(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "jiuwen"}],
        })

    @app.post("/v1/chat/completions")
    async def chat_completions(raw_request: Request):
        body = await raw_request.json()

        if body.get("stream", False):
            request_id = f"chatcmpl-{id(raw_request)}"
            from vllm.sampling_params import SamplingParams

            messages = body.get("messages", [])
            tokenizer = engine.get_tokenizer()
            if inspect.isawaitable(tokenizer):
                tokenizer = await tokenizer
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            sampling_params = SamplingParams(
                max_tokens=body.get("max_tokens", 128),
                temperature=body.get("temperature", 1.0),
                top_p=body.get("top_p", 1.0),
                n=body.get("n", 1),
            )
            generator = engine.generate(
                prompt=prompt_text,
                sampling_params=sampling_params,
                request_id=request_id,
            )
            return StreamingResponse(
                _stream_response(generator, model_name, request_id),
                media_type="text/event-stream",
            )

        result = await do_chat_completion(engine, body, model_name)
        status = 500 if "error" in result else 200
        return JSONResponse(result, status_code=status)

    @app.post("/v1/completions")
    async def completions(raw_request: Request):
        body = await raw_request.json()

        if body.get("stream", False):
            request_id = f"cmpl-{id(raw_request)}"
            from vllm.sampling_params import SamplingParams

            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                tokenizer = engine.get_tokenizer()
                if inspect.isawaitable(tokenizer):
                    tokenizer = await tokenizer
                prompt_text = tokenizer.decode(prompt)
            else:
                prompt_text = prompt

            sampling_params = SamplingParams(
                max_tokens=body.get("max_tokens", 128),
                temperature=body.get("temperature", 1.0),
                top_p=body.get("top_p", 1.0),
                n=body.get("n", 1),
            )
            if body.get("stop"):
                sampling_params.stop = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
            generator = engine.generate(
                prompt=prompt_text,
                sampling_params=sampling_params,
                request_id=request_id,
            )
            return StreamingResponse(
                _stream_completion_response(generator, model_name, request_id),
                media_type="text/event-stream",
            )

        result = await do_completion(engine, body, model_name)
        status = 500 if "error" in result else 200
        return JSONResponse(result, status_code=status)

    return app


async def _stream_response(
    generator: AsyncGenerator, model_name: str, request_id: str,
):
    """流式输出 chat completion 的 SSE 响应。

    vLLM 的 CompletionOutput.text 是累积值（每次包含从开头到当前的全部文本），
    需要通过 previous_texts 追踪已发送内容，计算增量 delta 发送给客户端。
    首个 chunk 的 delta 包含 role="assistant"，符合 OpenAI 流式协议规范。
    """
    previous_texts: dict[int, str] = {}
    first_chunk = True
    async for output in generator:
        for out in output.outputs:
            # 计算增量：out.text 是累积文本，减去已发送部分得到新增内容
            prev = previous_texts.get(out.index, "")
            delta_text = out.text[len(prev):]
            previous_texts[out.index] = out.text

            if not delta_text and out.finish_reason is None:
                continue

            delta: dict = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if delta_text:
                delta["content"] = delta_text

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [{
                    "index": out.index,
                    "delta": delta,
                    "finish_reason": out.finish_reason,
                }],
            }
            yield f"data: {orjson.dumps(chunk).decode()}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_completion_response(
    generator: AsyncGenerator, model_name: str, request_id: str,
):
    """流式输出 text completion 的 SSE 响应。

    与 _stream_response 类似，通过 previous_texts 将 vLLM 累积文本转换为增量输出。
    """
    previous_texts: dict[int, str] = {}
    async for output in generator:
        for out in output.outputs:
            # 计算增量：out.text 是累积文本，减去已发送部分得到新增内容
            prev = previous_texts.get(out.index, "")
            delta_text = out.text[len(prev):]
            previous_texts[out.index] = out.text

            if not delta_text and out.finish_reason is None:
                continue

            chunk = {
                "id": request_id,
                "object": "text_completion",
                "model": model_name,
                "choices": [{
                    "index": out.index,
                    "text": delta_text,
                    "finish_reason": out.finish_reason,
                    "logprobs": None,
                }],
            }
            yield f"data: {orjson.dumps(chunk).decode()}\n\n"
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

    def close():
        server.should_exit = True

    logger.info("HTTP server starting on %s:%d", args.host, args.port)
    return close, server.serve()


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


async def _tcp_client_handler(reader, writer, engine, model_name):
    addr = writer.get_extra_info("peername")
    logger.info("TCP client connected: %s", addr)
    try:
        while True:
            body = await _tcp_read_message(reader)
            if body is None:
                break
            result = await do_chat_completion(engine, body, model_name)
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
    tcp_server = await asyncio.start_server(
        lambda r, w: _tcp_client_handler(r, w, engine, model_name),
        host=args.host,
        port=args.port,
    )
    logger.info("TCP server listening on %s:%d", args.host, args.port)

    async def run():
        async with tcp_server:
            await shutdown_event.wait()

    return tcp_server.close, run()
