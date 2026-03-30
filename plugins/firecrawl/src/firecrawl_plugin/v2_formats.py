"""将工具层参数转为 firecrawl-py V2 的 formats / ScrapeOptions 片段。"""

from __future__ import annotations

from typing import Any

from firecrawl.v2.types import ScreenshotFormat


def parse_format_strings(raw: list[str] | None) -> list[Any]:
    """逗号拆分后的格式列表 -> V2 FormatOption 列表。"""
    if not raw:
        return []
    out: list[Any] = []
    for item in raw:
        s = (item or "").strip()
        if not s:
            continue
        if s.lower() == "screenshot@fullpage":
            out.append(ScreenshotFormat(full_page=True))
        elif s == "extract":
            # 结构化提取在 V2 中走 json format，具体 schema 由 build_formats_with_extract 合并
            out.append("json")
        else:
            out.append(s)
    return out


def build_formats_with_extract(
    format_strings: list[str] | None,
    *,
    schema: Any = None,
    prompt: str | None = None,
    system_prompt: str | None = None,
) -> list[Any] | None:
    """
    合并普通 formats 与 JSON 提取配置。
    若需要 json 提取，使用 JsonFormat 或带 type 的 dict（便于传 system_prompt 等扩展字段）。
    """
    parts = parse_format_strings(format_strings)
    want_json = schema is not None or (prompt and str(prompt).strip()) or (system_prompt and str(system_prompt).strip())
    if want_json:
        # 去掉重复的裸 "json"/extract 占位，合并为一条 json 配置（dict 便于带 systemPrompt）
        parts = [p for p in parts if p not in ("json", "extract")]
        jf: dict[str, Any] = {"type": "json"}
        if schema is not None:
            jf["schema"] = schema
        if prompt and str(prompt).strip():
            jf["prompt"] = str(prompt).strip()
        if system_prompt and str(system_prompt).strip():
            jf["systemPrompt"] = str(system_prompt).strip()
        parts.append(jf)
    if not parts:
        return None
    return parts
