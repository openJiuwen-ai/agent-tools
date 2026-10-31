"""启动期依赖：WorkerManager 的 HTTP 客户端固定开启 HTTP/2。"""

import httpx


def test_worker_http_client_can_enable_http2():
    """worker_manager.py 以 ``httpx.AsyncClient(http2=True)`` 建客户端。

    httpx 的 HTTP/2 支持由 ``h2`` 提供，只有安装 ``httpx[http2]`` 额外依赖才
    可用；若 pyproject 只声明 ``httpx``，干净安装后启动 jiuwen-infer-router
    会在 worker manager 初始化时抛 ImportError 并退出（Application startup
    failed）。
    """
    # trust_env=False 只隔离本机代理环境变量，不影响 HTTP/2 额外依赖检查
    client = httpx.AsyncClient(http2=True, trust_env=False)

    assert client is not None
