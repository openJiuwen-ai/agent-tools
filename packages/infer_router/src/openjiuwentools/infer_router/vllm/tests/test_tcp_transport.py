import asyncio
import json
import struct

import pytest

from transport import _tcp_pack_message

_HEADER_SIZE = 4


def pack_message(obj: dict) -> bytes:
    payload = json.dumps(obj).encode("utf-8")
    return struct.pack("!I", len(payload)) + payload


def unpack_message(data: bytes) -> dict:
    length = struct.unpack("!I", data[:_HEADER_SIZE])[0]
    return json.loads(data[_HEADER_SIZE:_HEADER_SIZE + length])


# ---------------------------------------------------------------------------
# Unit tests: message framing
# ---------------------------------------------------------------------------

class TestMessageFraming:
    @staticmethod
    def test_pack_unpack_roundtrip():
        original = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        packed = _tcp_pack_message(original)
        assert len(packed) > _HEADER_SIZE
        assert unpack_message(packed) == original

    @staticmethod
    def test_pack_empty_dict():
        packed = _tcp_pack_message({})
        assert unpack_message(packed) == {}

    @staticmethod
    def test_pack_unicode():
        original = {"content": "你好世界"}
        packed = _tcp_pack_message(original)
        assert unpack_message(packed) == original

    @staticmethod
    def test_header_is_4_bytes_big_endian():
        obj = {"a": 1}
        packed = _tcp_pack_message(obj)
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        expected_header = struct.pack("!I", len(body))
        assert packed[:4] == expected_header
        assert packed[4:] == body


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tcp_single_request(tcp_server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", tcp_server_port)

    writer.write(pack_message({"messages": [{"role": "user", "content": "hello"}]}))
    await writer.drain()

    header = await reader.readexactly(_HEADER_SIZE)
    length = struct.unpack("!I", header)[0]
    response = json.loads(await reader.readexactly(length))

    assert response["object"] == "chat.completion"
    assert response["model"] == "test-model"
    assert response["choices"][0]["message"]["content"] == "mock reply"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"]["total_tokens"] == 7

    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_tcp_multiple_requests_same_connection(tcp_server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", tcp_server_port)

    for i in range(3):
        writer.write(pack_message({"messages": [{"role": "user", "content": f"msg {i}"}]}))
        await writer.drain()

        header = await reader.readexactly(_HEADER_SIZE)
        length = struct.unpack("!I", header)[0]
        response = json.loads(await reader.readexactly(length))
        assert response["object"] == "chat.completion"

    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_tcp_client_disconnect_graceful(tcp_server_port):
    reader, writer = await asyncio.open_connection("127.0.0.1", tcp_server_port)

    writer.write(pack_message({"messages": [{"role": "user", "content": "bye"}]}))
    await writer.drain()

    header = await reader.readexactly(_HEADER_SIZE)
    length = struct.unpack("!I", header)[0]
    await reader.readexactly(length)

    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_tcp_concurrent_clients(tcp_server_port):
    async def single_client(port, cid):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(pack_message({"messages": [{"role": "user", "content": f"c{cid}"}]}))
        await writer.drain()

        header = await reader.readexactly(_HEADER_SIZE)
        length = struct.unpack("!I", header)[0]
        response = json.loads(await reader.readexactly(length))
        assert response["object"] == "chat.completion"

        writer.close()
        await writer.wait_closed()
        return response

    results = await asyncio.gather(*[single_client(tcp_server_port, i) for i in range(5)])
    assert len(results) == 5
    assert all(r["model"] == "test-model" for r in results)
