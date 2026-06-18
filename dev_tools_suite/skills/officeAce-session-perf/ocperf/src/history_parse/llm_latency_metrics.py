"""LLM latency helpers: TTFT, TPOT, tokens/s, cache tokens."""

from __future__ import annotations

from typing import Any

from history_parse.models import LLMRound

_CACHE_TOKEN_KEYS = (
    "cache_tokens",
    "cache_token",
    "cached_tokens",
    "prompt_cache_hit_tokens",
    "cache_read_input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
)


def extract_cache_tokens(meta: dict[str, Any] | None) -> int | None:
    """Read cache/prompt-cache token count from usage_metadata dict or string body."""
    if not meta:
        return None
    if isinstance(meta, str):
        import re

        for key in _CACHE_TOKEN_KEYS:
            m = re.search(rf"{key}[=:]?\s*(\d+)", meta)
            if m:
                return int(m.group(1))
        return None
    total = 0
    found = False
    for key in _CACHE_TOKEN_KEYS:
        val = meta.get(key)
        if isinstance(val, (int, float)) and val > 0:
            total += int(val)
            found = True
    return total if found else None


def compute_tpot(inference_sec: float, output_tokens: int | None) -> float | None:
    """Time per output token (seconds)."""
    if not output_tokens or output_tokens <= 0 or inference_sec < 0:
        return None
    return round(inference_sec / output_tokens, 6)


def compute_tokens_per_sec(output_tokens: int | None, inference_sec: float) -> float | None:
    """Output throughput during inference phase."""
    if not output_tokens or output_tokens <= 0 or inference_sec <= 0:
        return None
    return round(output_tokens / inference_sec, 2)


def llm_round_metrics(r: LLMRound) -> dict[str, Any]:
    """Per-round latency + token metrics for HTML/JSON export."""
    tpot = compute_tpot(r.inference_sec, r.output_tokens)
    tps = compute_tokens_per_sec(r.output_tokens, r.inference_sec)
    cache = r.cache_tokens
    parts = [
        f"TTFT {r.ttft_sec:.2f}s",
        f"推理 {r.inference_sec:.2f}s",
    ]
    if tpot is not None:
        parts.append(f"TPOT {tpot * 1000:.1f}ms")
    if tps is not None:
        parts.append(f"{tps:.1f} tok/s")
    tok = f"in {r.input_tokens or 0} · out {r.output_tokens or 0}"
    if cache:
        tok += f" · cache {cache}"
    parts.append(tok)
    return {
        "ttft_sec": round(r.ttft_sec, 6),
        "inference_sec": round(r.inference_sec, 6),
        "tpot_sec": tpot,
        "tokens_per_sec": tps,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "total_tokens": r.total_tokens,
        "cache_tokens": cache,
        "detail": " · ".join(parts),
    }


def aggregate_llm_latency(rounds: list[LLMRound]) -> dict[str, Any]:
    """Session-level LLM latency + token totals."""
    if not rounds:
        return {
            "llm_calls": 0,
            "input_tokens_sum": 0,
            "output_tokens_sum": 0,
            "cache_tokens_sum": 0,
            "total_tokens_sum": 0,
            "llm_ttft_sum_sec": 0.0,
            "llm_inference_sum_sec": 0.0,
            "avg_tpot_sec": None,
            "avg_tokens_per_sec": None,
        }
    ttft_sum = sum(r.ttft_sec for r in rounds)
    infer_sum = sum(r.inference_sec for r in rounds)
    tpots = [
        compute_tpot(r.inference_sec, r.output_tokens)
        for r in rounds
        if compute_tpot(r.inference_sec, r.output_tokens) is not None
    ]
    tpss = [
        compute_tokens_per_sec(r.output_tokens, r.inference_sec)
        for r in rounds
        if compute_tokens_per_sec(r.output_tokens, r.inference_sec) is not None
    ]
    return {
        "llm_calls": len(rounds),
        "input_tokens_sum": sum(r.input_tokens or 0 for r in rounds),
        "output_tokens_sum": sum(r.output_tokens or 0 for r in rounds),
        "cache_tokens_sum": sum(r.cache_tokens or 0 for r in rounds),
        "total_tokens_sum": sum(r.total_tokens or 0 for r in rounds),
        "llm_ttft_sum_sec": round(ttft_sum, 3),
        "llm_inference_sum_sec": round(infer_sum, 3),
        "avg_tpot_sec": round(sum(tpots) / len(tpots), 6) if tpots else None,
        "avg_tokens_per_sec": round(sum(tpss) / len(tpss), 2) if tpss else None,
    }
