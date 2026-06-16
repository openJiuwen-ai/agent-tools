"""Fuse history.json (tools) + full.json (model) into reconciled latency reports."""

from fusion_parse.discovery import discover_session_logs
from fusion_parse.fusion_report import render_fusion_html, write_fusion_report
from fusion_parse.reconcile import build_fusion_session

__all__ = [
    "build_fusion_session",
    "discover_session_logs",
    "render_fusion_html",
    "write_fusion_report",
]
