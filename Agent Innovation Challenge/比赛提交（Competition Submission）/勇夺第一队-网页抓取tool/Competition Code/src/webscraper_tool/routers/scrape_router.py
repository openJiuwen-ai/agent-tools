import json
import re
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import APIRouter, Body, Request
from pydantic import BaseModel, Field

scrape_router = APIRouter()


def _slice_html(html: str, start_markers: Optional[List[str]], end_markers: Optional[List[str]], max_len: int) -> str:
    if not start_markers:
        return html

    start = -1
    for m in start_markers:
        idx = html.find(m)
        if idx != -1 and (start == -1 or idx < start):
            start = idx

    if start == -1:
        return html

    end = -1
    if end_markers:
        for m in end_markers:
            idx = html.find(m, start)
            if idx != -1 and (end == -1 or idx < end):
                end = idx

    if end == -1:
        end = min(len(html), start + max_len)

    return html[start:end]


def _parse_json_maybe(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


class ExtractCssRule(BaseModel):
    name: str = Field(..., description="Field name in output")
    selector: str = Field(..., description="CSS selector")
    attr: Optional[str] = Field(default=None, description="If set, extract attribute value")
    multiple: bool = Field(False, description="Return list instead of single")
    text_join: str = Field(" ", description="Joiner for text nodes")


class ExtractRegexRule(BaseModel):
    name: str = Field(..., description="Field name in output")
    pattern: str = Field(..., description="Regex pattern")
    group: int = Field(0, ge=0, le=50, description="Capture group index")
    multiple: bool = Field(True, description="Return list of matches")


class ExtractRequest(BaseModel):
    url: str = Field(..., description="Target URL. Server will fetch HTML internally.")
    css: Optional[List[ExtractCssRule]] = Field(default=None, description="CSS extraction rules")
    regex: Optional[List[ExtractRegexRule]] = Field(default=None, description="Regex extraction rules")
    slice_markers_start: Optional[List[str]] = Field(default=None, description="Optional slice start markers")
    slice_markers_end: Optional[List[str]] = Field(default=None, description="Optional slice end markers")
    slice_max_len: int = Field(60000, ge=1000, le=500000, description="Max slice length when slicing")


class ExtractResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


async def _coerce_extract_request(request: Request, body_req: Optional[ExtractRequest]) -> ExtractRequest:
    if body_req is not None:
        return body_req

    qp = dict(request.query_params)

    try:
        form = await request.form()
        for k, v in form.items():
            qp.setdefault(k, v)
    except Exception:
        pass

    url = str(qp.get("url", "")).strip()

    css_val = qp.get("css")
    regex_val = qp.get("regex")
    css_rules = _parse_json_maybe(css_val) if isinstance(css_val, str) else None
    regex_rules = _parse_json_maybe(regex_val) if isinstance(regex_val, str) else None

    slice_start = _parse_json_maybe(qp.get("slice_markers_start", "")) if isinstance(qp.get("slice_markers_start"), str) else None
    slice_end = _parse_json_maybe(qp.get("slice_markers_end", "")) if isinstance(qp.get("slice_markers_end"), str) else None

    def _to_int(x: Any, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    # Pydantic will validate rule item shapes.
    return ExtractRequest(
        url=url,
        css=css_rules,
        regex=regex_rules,
        slice_markers_start=slice_start if isinstance(slice_start, list) else None,
        slice_markers_end=slice_end if isinstance(slice_end, list) else None,
        slice_max_len=_to_int(qp.get("slice_max_len"), 60000),
    )


async def _extract_impl(request: Request, req: Optional[ExtractRequest]) -> ExtractResponse:
    try:
        req = await _coerce_extract_request(request, req)
    except Exception as e:
        return ExtractResponse(success=False, data={}, error=f"bad_request: {str(e)}")

    url_in = str(req.url or "").strip()
    if not url_in:
        return ExtractResponse(success=False, data={}, error="bad_request: url is required")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            resp = await client.get(url_in, headers=headers)
        content_type = resp.headers.get("content-type", "")
        if "text" not in content_type and "json" not in content_type and "html" not in content_type:
            return ExtractResponse(success=False, data={}, error=f"unsupported content-type: {content_type}")
        html = resp.text

        html = _slice_html(html, req.slice_markers_start, req.slice_markers_end, req.slice_max_len)
        soup = BeautifulSoup(html, "html.parser")

        out: Dict[str, Any] = {}

        if req.css:
            for rule in req.css:
                nodes = soup.select(rule.selector)
                if rule.multiple:
                    vals: List[str] = []
                    for n in nodes:
                        if rule.attr:
                            v = n.get(rule.attr)
                            if v is not None:
                                vals.append(str(v).strip())
                        else:
                            vals.append(n.get_text(rule.text_join, strip=True))
                    out[rule.name] = [v for v in vals if v]
                else:
                    n = nodes[0] if nodes else None
                    if not n:
                        out[rule.name] = ""
                    elif rule.attr:
                        out[rule.name] = str(n.get(rule.attr) or "").strip()
                    else:
                        out[rule.name] = n.get_text(rule.text_join, strip=True)

        if req.regex:
            for rule in req.regex:
                try:
                    if rule.multiple:
                        matches = re.findall(rule.pattern, html, flags=re.S)
                        vals: List[str] = []
                        for m in matches:
                            if isinstance(m, tuple):
                                vals.append(str(m[rule.group] if rule.group < len(m) else m[0]))
                            else:
                                vals.append(str(m))
                        out[rule.name] = [v.strip() for v in vals if str(v).strip()]
                    else:
                        m = re.search(rule.pattern, html, flags=re.S)
                        if not m:
                            out[rule.name] = ""
                        else:
                            out[rule.name] = str(m.group(rule.group)).strip()
                except Exception as e:
                    out[rule.name] = f"__error__: {str(e)}"

        return ExtractResponse(success=True, data=out)
    except Exception as e:
        return ExtractResponse(success=False, data={}, error=f"extract failed: {str(e)}")


@scrape_router.get("/extract", response_model=ExtractResponse)
async def extract_get(request: Request):
    return await _extract_impl(request, None)


@scrape_router.post("/extract", response_model=ExtractResponse)
async def extract_post(request: Request, req: Optional[ExtractRequest] = Body(default=None)):
    return await _extract_impl(request, req)
