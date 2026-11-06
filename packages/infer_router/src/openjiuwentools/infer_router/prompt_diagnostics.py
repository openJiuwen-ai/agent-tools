"""Opt-in prompt identity observations; never alter inference inputs."""

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping


def enabled():
    return os.getenv("JIUWEN_PD_PROMPT_DIAGNOSTICS", "").lower() in ("1", "true", "yes")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def token_observation(value):
    """Inspect BatchEncoding/tensor/list results without changing the caller's value."""
    original_type = type(value).__name__
    if isinstance(value, Mapping):
        value = value.get("input_ids", [])
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    tokens = [int(item) for item in value]
    return {
        "token_container": original_type,
        "token_count": len(tokens),
        "token_sha256": digest(tokens),
    }


def prompt_observation(
    body, tokens, *, stage, request_id, tokenizer=None, template_kwargs=None, **extra
):
    """Hash only content; log template controls, never messages or tools verbatim."""
    result = {
        "stage": stage,
        "request_id": request_id,
        "messages_sha256": digest(body.get("messages", [])),
        "effective_messages_sha256": digest(body.get("messages", [])),
        "tools_sha256": digest(body.get("tools")),
        "tool_choice_sha256": digest(body.get("tool_choice", "auto")),
        "requested_template_kwargs_sha256": digest(body.get("chat_template_kwargs")),
        **token_observation(tokens),
        **extra,
    }
    if tokenizer is not None:
        result.update(
            tokenizer_class=type(tokenizer).__name__,
            template_sha256=digest(getattr(tokenizer, "chat_template", None)),
        )
    if template_kwargs is not None:
        result["effective_template_kwargs_sha256"] = digest(template_kwargs)
    return result


def trace_generation(generator, log, request_id, engine_request_id):
    """Log first engine output and terminal status, including client cancellation."""
    if not enabled():
        return generator

    async def observe():
        phase = "completed"
        first = True
        try:
            async for output in generator:
                if first:
                    tokens = getattr(output, "prompt_token_ids", None)
                    log.info(
                        "[PDGenerationDiagnostics] request=%s engine_request_id=%s "
                        "phase=first_engine_output prompt=%s",
                        request_id,
                        engine_request_id,
                        token_observation(tokens) if tokens is not None else None,
                    )
                    first = False
                yield output
        except (asyncio.CancelledError, GeneratorExit):
            phase = "cancelled_or_closed"
            raise
        except Exception:
            phase = "error"
            raise
        finally:
            log.info(
                "[PDGenerationDiagnostics] request=%s engine_request_id=%s phase=%s had_output=%s",
                request_id,
                engine_request_id,
                phase,
                not first,
            )

    return observe()
