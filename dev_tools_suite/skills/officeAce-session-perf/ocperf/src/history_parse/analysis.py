"""Build timeline from history.json events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from history_parse.llm_latency_metrics import extract_cache_tokens
from history_parse.models import (
    ContextEvent,
    HistoryExtras,
    LLMRound,
    ToolExecution,
    UserTurn,
)
from history_parse.parser import filter_sessions
from history_parse.session import child_info
from history_parse.spawn_orphan import resolve_orphan_spawns
from history_parse.todo_tracker import (
    build_todo_timeline,
    build_todo_timeline_from_task_events,
    merge_todo_timelines,
)


def _ts(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone()


def _parse_tool_args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _context_event_from_tool(tool_name: str, args: Any, result: str) -> ContextEvent | None:
    if tool_name == "reload_original_context_messages":
        handle = storage = ""
        if isinstance(args, dict):
            handle = str(args.get("offload_handle") or "")
            storage = str(args.get("offload_type") or "")
        return ContextEvent(
            kind="reload",
            timestamp=datetime.min,
            tool_name=tool_name,
            summary=f"重载上下文 · {handle or '—'}",
            detail=result[:3000],
        )
    if tool_name == "skill_complete":
        skill = args.get("skill_name", "") if isinstance(args, dict) else ""
        return ContextEvent(
            kind="compress",
            timestamp=datetime.min,
            tool_name=tool_name,
            summary=f"释放技能 · {skill}",
            detail=result[:2000],
        )
    if "OFFLOAD" in result or "offload" in result.lower():
        return ContextEvent(
            kind="compress",
            timestamp=datetime.min,
            tool_name=tool_name,
            summary="上下文卸载",
            detail=result[:2000],
        )
    return None


def build_timeline_from_history(
    events: list[dict[str, Any]],
    root_session: str,
    *,
    full_log_paths: list[Path] | None = None,
) -> tuple[list[LLMRound], list[ToolExecution], HistoryExtras]:
    filtered = filter_sessions(events, root_session)
    extras = HistoryExtras()

    rounds: list[LLMRound] = []
    tools: list[ToolExecution] = []

    reasoning_buf: dict[tuple[str, str], list[str]] = {}
    delta_buf: dict[tuple[str, str], list[str]] = {}
    first_token_ts: dict[tuple[str, str], datetime] = {}
    # 同 session 内：工具结束 / 上轮模型结束 / 用户消息 → 作为下一轮 TTFT 起点
    session_boundary: dict[str, datetime] = {}
    pending_calls: dict[str, dict[str, Any]] = {}

    tool_success = tool_failure = 0
    tool_by_name: dict[str, dict[str, int]] = {}

    def _key(e: dict[str, Any]) -> tuple[str, str]:
        return (str(e.get("session_id") or ""), str(e.get("request_id") or ""))

    for event in filtered:
        et = event.get("event_type") or ""
        role = event.get("role") or ""
        ts = _ts(float(event.get("timestamp") or 0))
        key = _key(event)

        if role == "user" and not et:
            extras.user_turns.append(
                UserTurn(request_id=key[1], timestamp=ts, content=str(event.get("content") or ""), session_id=key[0])
            )
            session_boundary[key[0]] = ts
            continue

        if et == "chat.reasoning":
            reasoning_buf.setdefault(key, []).append(str(event.get("content") or ""))
            first_token_ts.setdefault(key, ts)
            continue

        if et == "chat.delta":
            first_token_ts.setdefault(key, ts)
            delta_buf.setdefault(key, []).append(str(event.get("content") or ""))
            continue

        if et == "chat.tool_call":
            tc = event.get("tool_call") or {}
            tool_call_id = str(tc.get("tool_call_id") or "")
            if not tool_call_id:
                continue
            pending_calls[tool_call_id] = {
                "session_id": key[0],
                "request_id": key[1],
                "tool_call_id": tool_call_id,
                "name": str(tc.get("name") or event.get("tool_name") or "?"),
                "arguments": _parse_tool_args(tc.get("arguments") or event.get("arguments") or ""),
                "start_ts": ts,
            }
            continue

        if et == "chat.tool_result":
            tool_name = str(event.get("tool_name") or "")
            tool_call_id = str(event.get("tool_call_id") or "")
            result = str(event.get("result") or "")
            pending = pending_calls.pop(tool_call_id, None)
            if pending:
                start_ts = pending["start_ts"]
                name = str(pending.get("name") or tool_name or "?")
                arguments = pending.get("arguments")
                session_id = str(pending.get("session_id") or key[0])
                request_id = str(pending.get("request_id") or key[1])
            else:
                start_ts = ts
                name = tool_name or "?"
                arguments = _parse_tool_args(event.get("arguments") or "")
                session_id = key[0]
                request_id = key[1]

            is_child, label, path = child_info(session_id, root_session)
            duration = max(0.0, (ts - start_ts).total_seconds())
            tools.append(
                ToolExecution(
                    session_id=session_id,
                    request_id=request_id,
                    tool_call_id=tool_call_id or f"unknown-{len(tools)}",
                    name=name,
                    arguments=arguments,
                    start_ts=start_ts,
                    end_ts=ts,
                    duration_sec=duration,
                    result=result[:8000],
                    is_child_session=is_child,
                    child_label=label,
                    child_path=path,
                )
            )

            if "success=True" in result:
                tool_success += 1
                tool_by_name.setdefault(name, {"success": 0, "failure": 0})["success"] += 1
            elif "success=False" in result:
                tool_failure += 1
                tool_by_name.setdefault(name, {"success": 0, "failure": 0})["failure"] += 1

            ctx = _context_event_from_tool(name, arguments, result)
            if ctx:
                ctx.timestamp = ts
                extras.context_events.append(ctx)
            session_boundary[session_id] = ts
            continue

        if et == "chat.usage_metadata":
            meta = (event.get("metadata") or {}).get("usage_metadata") or {}
            reasoning = "".join(reasoning_buf.pop(key, []))
            delta = "".join(delta_buf.pop(key, []))
            first_out = first_token_ts.pop(key, None)
            output_ts = ts
            think_start = session_boundary.get(key[0])
            if first_out is None:
                first_out = output_ts
            if think_start is None:
                think_start = first_out
            ttft_sec = max(0.0, (first_out - think_start).total_seconds())
            inference_sec = max(0.0, (output_ts - first_out).total_seconds())
            is_child, label, path = child_info(key[0], root_session)
            rounds.append(
                LLMRound(
                    session_id=key[0],
                    request_id=key[1],
                    model_name=str(meta.get("model_name") or ""),
                    request_ts=think_start,
                    first_token_ts=first_out,
                    output_ts=output_ts,
                    ttft_sec=ttft_sec,
                    inference_sec=inference_sec,
                    duration_sec=round(ttft_sec + inference_sec, 6),
                    input_tokens=int(meta.get("input_tokens") or 0) or None,
                    output_tokens=int(meta.get("output_tokens") or 0) or None,
                    total_tokens=int(meta.get("total_tokens") or 0) or None,
                    cache_tokens=extract_cache_tokens(meta),
                    reasoning_full=reasoning,
                    assistant_text=delta,
                    is_child_session=is_child,
                    child_label=label,
                    child_path=path,
                )
            )
            session_boundary[key[0]] = output_ts

    extras.tool_stats = {"success": tool_success, "failure": tool_failure, "by_name": tool_by_name}

    orphan_info, backfilled = resolve_orphan_spawns(
        pending_calls,
        root_session,
        full_log_paths=full_log_paths,
        existing_tools=tools,
    )
    extras.orphan_spawn_calls = orphan_info
    extras.spawn_orphan_backfilled = len(backfilled)
    if backfilled:
        tools.extend(backfilled)
        tools.sort(key=lambda t: (t.start_ts, t.tool_call_id))
        for tool in backfilled:
            tool_success += 1
            tool_by_name.setdefault(tool.name, {"success": 0, "failure": 0})["success"] += 1
        extras.tool_stats = {"success": tool_success, "failure": tool_failure, "by_name": tool_by_name}

    tool_todo = build_todo_timeline(tools)
    event_todo = build_todo_timeline_from_task_events(filtered, root_session)
    extras.todo_timeline = merge_todo_timelines(event_todo, tool_todo)
    return rounds, tools, extras
