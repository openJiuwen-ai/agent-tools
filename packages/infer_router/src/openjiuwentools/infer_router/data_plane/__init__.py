"""In-process Rust data plane proxy (PyO3 extension).

The Rust HTTP server runs on tokio threads within this Python process.
Routing decisions are made via direct Python function calls (no gRPC).
"""

from openjiuwentools.infer_router._data_plane import start_proxy

__all__ = ["start_proxy"]
