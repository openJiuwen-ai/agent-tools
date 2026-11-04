from __future__ import annotations

import json
import socket
import socketserver
import struct
import threading
import time
from typing import Any

from jiuwen_kvbm.accessor.vllm_ascend_018 import (
    VllmAscend018BlockPayload,
    VllmAscend018LayerPayload,
    VllmAscend018TensorProvider,
)

_PROTOCOL_VERSION = 1
_MAX_HEADER_BYTES = 8 * 1024 * 1024


class TcpStagingProtocolError(RuntimeError):
    """TCP staging 对端返回了非法协议数据。"""


class TcpStagingServer(socketserver.ThreadingTCPServer):
    """从源 NPU 读取 KV block 并通过 TCP 输出 CPU staging 数据的服务。

    参数:
        server_address: TCP 监听地址二元组。
        provider: 源 worker 的真实 KV tensor provider。
        access_lock: 保护 provider 读取过程的进程内锁。
        max_blocks: 单次请求允许读取的最大 block 数。
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        provider: VllmAscend018TensorProvider,
        access_lock: threading.Lock,
        *,
        max_blocks: int = 64,
    ) -> None:
        self.provider = provider
        self.access_lock = access_lock
        self.max_blocks = max(1, int(max_blocks))
        super().__init__(server_address, _TcpStagingHandler)


class _TcpStagingHandler(socketserver.BaseRequestHandler):
    """处理一个可复用 TCP staging 连接。"""

    def handle(self) -> None:
        """循环处理同一连接上的 block 拉取请求。

        参数:
            无显式参数；请求从 self.request 读取。

        返回:
            无返回值；客户端关闭连接或协议失败时结束。
        """

        while True:
            try:
                request = _recv_json(self.request)
            except EOFError:
                return
            try:
                response, chunks = self._read_blocks(request)
            except Exception as exc:
                _send_json(self.request, {"ok": False, "error": str(exc), "payload_nbytes": 0})
                continue
            _send_json(self.request, response)
            for chunk in chunks:
                self.request.sendall(chunk)

    def _read_blocks(self, request: dict[str, Any]) -> tuple[dict[str, Any], list[bytes]]:
        """读取并序列化一批源 block。

        参数:
            request: TCP 协议请求字典，必须包含 source_blocks。

        返回:
            (响应元数据, tensor 字节 chunks) 二元组。
        """

        if int(request.get("version", 0)) != _PROTOCOL_VERSION:
            raise TcpStagingProtocolError("unsupported TCP staging protocol version")
        source_blocks = _required_block_ids(request, "source_blocks")
        server = self.server
        if not isinstance(server, TcpStagingServer):
            raise RuntimeError("invalid TCP staging server")
        if len(source_blocks) > server.max_blocks:
            raise ValueError(f"source_blocks exceeds max_blocks={server.max_blocks}")

        started = time.perf_counter()
        with server.access_lock:
            blocks = [server.provider.read_block(block_id) for block_id in source_blocks]
        read_ms = (time.perf_counter() - started) * 1000.0
        serialize_started = time.perf_counter()
        metadata, chunks = _serialize_blocks(blocks)
        serialize_ms = (time.perf_counter() - serialize_started) * 1000.0
        return {
            "ok": True,
            "version": _PROTOCOL_VERSION,
            "blocks": metadata,
            "payload_nbytes": sum(len(chunk) for chunk in chunks),
            "source_read_ms": read_ms,
            "source_serialize_ms": serialize_ms,
        }, chunks


class TcpStagingClient:
    """执行 NPU -> CPU -> TCP -> CPU -> NPU 基线搬运的持久连接客户端。

    参数:
        host: 源 worker TCP staging 服务地址。
        port: 源 worker TCP staging 服务端口。
        provider: 目标 worker 的真实 KV tensor provider。
        access_lock: 保护目标 provider 写入过程的进程内锁。
        timeout_seconds: 建连和读写超时时间，单位秒。
    """

    def __init__(
        self,
        host: str,
        port: int,
        provider: VllmAscend018TensorProvider,
        access_lock: threading.Lock,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.provider = provider
        self.access_lock = access_lock
        self.timeout_seconds = float(timeout_seconds)
        self._socket: socket.socket | None = None

    def __enter__(self) -> TcpStagingClient:
        """建立持久 TCP 连接并返回客户端自身。

        返回:
            已连接的 TcpStagingClient。
        """

        self._socket = socket.create_connection(
            (self.host, self.port),
            timeout=self.timeout_seconds,
        )
        self._socket.settimeout(self.timeout_seconds)
        return self

    def __exit__(self, *_args: Any) -> None:
        """关闭持久 TCP 连接。

        参数:
            _args: context manager 传入的异常信息，本实现不使用。

        返回:
            无返回值。
        """

        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def pull_blocks(
        self,
        source_blocks: list[int],
        destination_blocks: list[int],
        *,
        expected_nbytes: int | None = None,
    ) -> dict[str, float]:
        """通过 CPU staging 和 TCP 将源 block 写入目标物理 block。

        参数:
            source_blocks: 源 worker 物理 block ID 列表。
            destination_blocks: 目标 worker 物理 block ID 列表，顺序与源列表对应。
            expected_nbytes: 目标 worker 根据本地 cache 布局计算的预期字节数；
                提供时，对端载荷大小必须完全一致。

        返回:
            包含端到端 elapsed_ms、源端 D2H/序列化和目标端 H2D 耗时的字典。

        异常:
            ValueError: 两侧 block 数不一致或列表为空。
            TcpStagingProtocolError: 对端响应失败或载荷不完整。
        """

        if len(source_blocks) != len(destination_blocks) or not source_blocks:
            raise ValueError("source_blocks and destination_blocks must be non-empty and equal")
        sock = self._socket
        if sock is None:
            raise RuntimeError("TCP staging client is not connected")

        started = time.perf_counter()
        _send_json(
            sock,
            {"version": _PROTOCOL_VERSION, "source_blocks": source_blocks},
        )
        response = _recv_json(sock)
        if not response.get("ok"):
            raise TcpStagingProtocolError(str(response.get("error", "source read failed")))
        payload_nbytes = int(response.get("payload_nbytes", 0))
        if expected_nbytes is not None and payload_nbytes != int(expected_nbytes):
            raise TcpStagingProtocolError(
                f"source payload size {payload_nbytes} does not match "
                f"destination layout {expected_nbytes}"
            )
        raw_payload = _recv_exact(sock, payload_nbytes)
        receive_ms = (time.perf_counter() - started) * 1000.0

        decode_started = time.perf_counter()
        blocks = _deserialize_blocks(
            response.get("blocks"),
            raw_payload,
            destination_blocks,
        )
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        write_started = time.perf_counter()
        with self.access_lock:
            for block in blocks:
                self.provider.write_block(block.block_id, block)
        target_write_ms = (time.perf_counter() - write_started) * 1000.0
        return {
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "source_read_ms": float(response.get("source_read_ms", 0.0)),
            "source_serialize_ms": float(response.get("source_serialize_ms", 0.0)),
            "network_receive_ms": receive_ms,
            "target_decode_ms": decode_ms,
            "target_write_ms": target_write_ms,
        }


def start_tcp_staging_server(
    provider: VllmAscend018TensorProvider,
    host: str,
    port: int,
    access_lock: threading.Lock,
    *,
    max_blocks: int = 64,
) -> tuple[TcpStagingServer, threading.Thread]:
    """启动后台 TCP staging 服务。

    参数:
        provider: 源 worker 的真实 KV tensor provider。
        host: 监听地址。
        port: 监听端口。
        access_lock: 保护 provider 读取的进程内锁。
        max_blocks: 单次请求允许传输的最大 block 数。

    返回:
        (server, daemon thread) 二元组，调用方需保存引用。
    """

    server = TcpStagingServer(
        (host, int(port)),
        provider,
        access_lock,
        max_blocks=max_blocks,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        name="jiuwen-kvbm-tcp-staging-server",
        daemon=True,
    )
    thread.start()
    return server, thread


def _serialize_blocks(
    blocks: list[VllmAscend018BlockPayload],
) -> tuple[list[dict[str, Any]], list[bytes]]:
    """将 CPU tensor block 编码为 JSON 元数据和连续字节块。

    参数:
        blocks: provider.read_block 返回的 block 列表。

    返回:
        (block 元数据列表, 原始字节 chunks) 二元组。
    """

    metadata: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    for block in blocks:
        layers: list[dict[str, Any]] = []
        for layer in block.layers:
            key = _tensor_to_bytes(layer.key)
            value = _tensor_to_bytes(layer.value)
            layers.append(
                {
                    "layer_name": layer.layer_name,
                    "key": _tensor_metadata(layer.key, len(key)),
                    "value": _tensor_metadata(layer.value, len(value)),
                }
            )
            chunks.extend((key, value))
        metadata.append({"source_block_id": block.block_id, "layers": layers})
    return metadata, chunks


def _deserialize_blocks(
    raw_metadata: Any,
    payload: bytes,
    destination_blocks: list[int],
) -> list[VllmAscend018BlockPayload]:
    """把 TCP 原始载荷还原为目标 CPU tensor block。

    参数:
        raw_metadata: 源端返回的 block/layer/tensor 元数据。
        payload: 连续原始 tensor 字节。
        destination_blocks: 目标物理 block ID 列表。

    返回:
        可传给 provider.write_block 的 block payload 列表。
    """

    if not isinstance(raw_metadata, list) or len(raw_metadata) != len(destination_blocks):
        raise TcpStagingProtocolError("response block metadata count mismatch")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required by TCP staging target") from exc

    offset = 0
    result: list[VllmAscend018BlockPayload] = []
    backing = bytearray(payload)
    for block_meta, destination_block in zip(raw_metadata, destination_blocks, strict=True):
        layers: list[VllmAscend018LayerPayload] = []
        for layer_meta in block_meta.get("layers", []):
            key, offset = _tensor_from_bytes(torch, backing, offset, layer_meta["key"])
            value, offset = _tensor_from_bytes(torch, backing, offset, layer_meta["value"])
            layers.append(
                VllmAscend018LayerPayload(
                    layer_name=str(layer_meta["layer_name"]),
                    key=key,
                    value=value,
                )
            )
        result.append(VllmAscend018BlockPayload(block_id=destination_block, layers=layers))
    if offset != len(payload):
        raise TcpStagingProtocolError("response payload size does not match metadata")
    return result


def _tensor_from_bytes(
    torch: Any,
    backing: bytearray,
    offset: int,
    metadata: dict[str, Any],
) -> tuple[Any, int]:
    """从连续字节缓冲区还原一个 CPU tensor。

    参数:
        torch: 延迟导入的 torch 模块。
        backing: 保存完整响应载荷的可写缓冲区。
        offset: 当前 tensor 的起始偏移。
        metadata: 包含 dtype、shape 和 nbytes 的 tensor 元数据。

    返回:
        (独立 CPU tensor, 下一个字节偏移) 二元组。
    """

    nbytes = int(metadata["nbytes"])
    end = offset + nbytes
    if nbytes < 0 or end > len(backing):
        raise TcpStagingProtocolError("tensor payload is truncated")
    dtype_name = str(metadata["dtype"]).removeprefix("torch.")
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise TcpStagingProtocolError(f"unsupported torch dtype: {dtype_name}")
    tensor = torch.frombuffer(
        memoryview(backing)[offset:end],
        dtype=dtype,
    ).reshape(tuple(int(dim) for dim in metadata["shape"])).clone()
    return tensor, end


def _tensor_metadata(tensor: Any, nbytes: int) -> dict[str, Any]:
    """生成还原 CPU tensor 所需的元数据。

    参数:
        tensor: CPU tensor。
        nbytes: tensor 连续字节长度。

    返回:
        包含 dtype、shape 和 nbytes 的字典。
    """

    return {
        "dtype": str(getattr(tensor, "dtype", "")),
        "shape": [int(dim) for dim in tensor.shape],
        "nbytes": nbytes,
    }


def _tensor_to_bytes(tensor: Any) -> bytes:
    """导出 CPU tensor 的连续原始字节。

    参数:
        tensor: provider.read_block 产生的 CPU tensor。

    返回:
        tensor 连续存储对应的 bytes。
    """

    import torch

    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _required_block_ids(payload: dict[str, Any], name: str) -> list[int]:
    """读取非空物理 block ID 列表。

    参数:
        payload: 协议请求字典。
        name: block 列表字段名。

    返回:
        转换为 int 的 block ID 列表。
    """

    value = payload.get(name)
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return [int(item) for item in value]


def _send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    """发送四字节长度前缀的 JSON 帧。

    参数:
        sock: 已连接 TCP socket。
        payload: 可 JSON 序列化字典。

    返回:
        无返回值。
    """

    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(body) > _MAX_HEADER_BYTES:
        raise TcpStagingProtocolError("JSON frame exceeds maximum header size")
    sock.sendall(struct.pack("!I", len(body)) + body)


def _recv_json(sock: socket.socket) -> dict[str, Any]:
    """接收四字节长度前缀的 JSON 帧。

    参数:
        sock: 已连接 TCP socket。

    返回:
        解码后的 JSON 字典。
    """

    size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if size <= 0 or size > _MAX_HEADER_BYTES:
        raise TcpStagingProtocolError(f"invalid JSON frame size: {size}")
    value = json.loads(_recv_exact(sock, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise TcpStagingProtocolError("JSON frame must be an object")
    return value


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    """从 socket 精确读取指定字节数。

    参数:
        sock: 已连接 TCP socket。
        size: 期望读取的字节数。

    返回:
        长度等于 size 的 bytes。

    异常:
        EOFError: 对端在载荷完整前关闭连接。
    """

    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("TCP staging peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
