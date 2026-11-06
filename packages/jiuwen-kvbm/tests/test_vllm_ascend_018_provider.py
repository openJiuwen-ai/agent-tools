from jiuwen_kvbm.accessor import (
    VllmAscend018TensorProvider,
    VllmAscendKVAccessor,
    detect_vllm_async_mp_client,
    locate_vllm_ascend_018_kv_caches,
)


class FakeBlock:
    def __init__(self, values):
        self.values = list(values)
        self.device = "npu"
        self.dtype = "float32"
        self.shape = (len(self.values),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return FakeBlock(self.values)

    def to(self, device=None, dtype=None):
        return self

    def copy_(self, other):
        self.values = list(other.values)
        return self

    def numel(self):
        return len(self.values)

    @staticmethod
    def element_size():
        return 4

    def zero_(self):
        self.values = [0 for _ in self.values]
        return self

    def sum(self):
        return sum(self.values)


class FakeCache:
    def __init__(self, blocks):
        self.blocks = [FakeBlock(block) for block in blocks]
        self.shape = (len(self.blocks),) + self.blocks[0].shape
        self.dtype = self.blocks[0].dtype
        self.device = self.blocks[0].device

    def __getitem__(self, index):
        return self.blocks[index]


def _kv_caches():
    return {
        "layer.0": (
            FakeCache([[0, 1, 2, 3], [12, 13, 14, 15], [24, 25, 26, 27]]),
            FakeCache([[4, 5, 6, 7], [16, 17, 18, 19], [28, 29, 30, 31]]),
        ),
        "layer.1": (
            FakeCache([[100, 101, 102, 103], [112, 113, 114, 115], [124, 125, 126, 127]]),
            FakeCache([[104, 105, 106, 107], [116, 117, 118, 119], [128, 129, 130, 131]]),
        ),
    }


def test_vllm_ascend_018_provider_reads_and_writes_block():
    """v0.18 provider 应按 [0]/[1] 拆分 key/value，并按 block 维读写。"""
    provider = VllmAscend018TensorProvider(_kv_caches())
    accessor = VllmAscendKVAccessor(provider)

    payload = accessor.read_block(1)

    assert payload.block_id == 1
    assert [layer.layer_name for layer in payload.layers] == ["layer.0", "layer.1"]
    assert payload.layers[0].key.values == [12, 13, 14, 15]
    assert payload.layers[0].value.values == [16, 17, 18, 19]

    provider.kv_caches["layer.0"][0][1].values = [-1, -1, -1, -1]
    provider.kv_caches["layer.0"][1][1].values = [-2, -2, -2, -2]
    accessor.write_block(1, payload)

    assert provider.kv_caches["layer.0"][0][1].values == payload.layers[0].key.values
    assert provider.kv_caches["layer.0"][1][1].values == payload.layers[0].value.values


def test_vllm_ascend_018_provider_describes_block_size():
    """describe_block 应返回跨所有层 key/value 的总字节数。"""
    provider = VllmAscend018TensorProvider(_kv_caches(), device_id=3)

    region = provider.describe_block(1)

    assert region.device_type == "npu"
    assert region.device_id == 3
    assert region.nbytes == 4 * 4 * 4
    assert region.backend_handle == 1
    assert region.layout.startswith("vllm-ascend-0.18:")


def test_vllm_ascend_018_provider_reuses_block_region_template():
    """同一 KV cache 的后续 block 应复用首次计算的布局模板。"""

    caches = _kv_caches()
    provider = VllmAscend018TensorProvider(caches, device_id=2)
    first = provider.describe_block(0)

    class UnexpectedCaches:
        @staticmethod
        def items():
            raise AssertionError("后续 describe_block 不应重新遍历层布局")

    provider.kv_caches = UnexpectedCaches()
    second = provider.describe_block(2)

    assert second.nbytes == first.nbytes
    assert second.layout == first.layout
    assert second.backend_handle == 2


def test_vllm_ascend_018_provider_infers_block_tokens_from_tensor_shape():
    """provider 应以真实 KV tensor 的第 1 维作为 block token 数。

    返回:
        无返回值；shape 推导错误时由断言报告失败。
    """

    provider = VllmAscend018TensorProvider(_kv_caches())

    assert provider.blocks_cache_shape() == (3, 4)
    assert provider.block_size_tokens() == 4


def test_vllm_ascend_018_delete_defaults_to_noop_and_can_zero():
    """delete_block 默认不破坏 vLLM 生命周期，显式 zero 时才清零。"""
    caches = _kv_caches()
    noop_provider = VllmAscend018TensorProvider(caches)
    noop_provider.delete_block(1)
    assert caches["layer.0"][0][1].sum() != 0

    zero_provider = VllmAscend018TensorProvider(caches, delete_mode="zero")
    zero_provider.delete_block(1)
    assert caches["layer.0"][0][1].values == [0, 0, 0, 0]
    assert caches["layer.0"][1][1].values == [0, 0, 0, 0]


def test_vllm_ascend_018_provider_reports_diagnostics():
    """diagnostics 应输出真实环境排查需要的 shape、dtype、device 和大小信息。"""
    provider = VllmAscend018TensorProvider(_kv_caches())

    diagnostics = provider.diagnostics(1)

    assert diagnostics["layer_count"] == 2
    assert diagnostics["first_layer"] == "layer.0"
    assert diagnostics["key_shape"] == (3, 4)
    assert diagnostics["value_shape"] == (3, 4)
    assert diagnostics["block_size"] == 4
    assert diagnostics["key_device"] == "npu"
    assert diagnostics["key_dtype"] == "float32"
    assert diagnostics["num_blocks"] == 3
    assert diagnostics["first_layer_block_nbytes"] == 32
    assert diagnostics["total_block_nbytes"] == 64
    assert diagnostics["block_nbytes"] == 64


def test_vllm_ascend_018_payload_reports_total_nbytes():
    """read_block 返回的 payload 应暴露跨所有层的真实字节数。"""
    provider = VllmAscend018TensorProvider(_kv_caches())

    payload = provider.read_block(1)

    assert payload.layers[0].nbytes == 32
    assert payload.nbytes == 64


def test_vllm_ascend_018_provider_digest_detects_equal_and_different_blocks():
    """block 摘要应稳定识别内容相同或不同的跨层 K/V payload。"""

    caches = _kv_caches()
    provider = VllmAscend018TensorProvider(caches)
    first = provider.block_digest(1, include_layers=True)
    repeated = provider.block_digest(1)

    assert first["sha256"] == repeated["sha256"]
    assert first["nbytes"] == 64
    assert first["layer_count"] == 2
    assert len(first["layers"]) == 2

    caches["layer.0"][0][1].values[0] = -1
    changed = provider.block_digest(1)

    assert changed["sha256"] != first["sha256"]


def test_locate_vllm_ascend_018_kv_caches_from_engine_object():
    """定位器应能从真实 engine 类似的对象树中找到 kv_caches，并返回路径。"""

    class Runner:
        def __init__(self):
            self.kv_caches = _kv_caches()

    class Worker:
        def __init__(self):
            self.model_runner = Runner()

    class Engine:
        def __init__(self):
            self.worker = Worker()

    kv_caches, path = locate_vllm_ascend_018_kv_caches(Engine())

    assert path == "engine.worker.model_runner.kv_caches"
    assert "layer.0" in kv_caches


def test_locate_vllm_ascend_018_failure_reports_candidates():
    """定位失败时应带出疑似路径，方便真实环境继续确认 engine 结构。"""

    class Runner:
        def __init__(self):
            self.cache_config = {"num_gpu_blocks": 8}

    class Engine:
        def __init__(self):
            self.model_runner = Runner()

    try:
        locate_vllm_ascend_018_kv_caches(Engine(), max_depth=4)
    except LookupError as exc:
        message = str(exc)
    else:
        raise AssertionError("locator should fail without kv_caches")

    assert "diagnostics=" in message
    assert "engine.model_runner" in message
    assert "cache_config" in message


def test_locate_vllm_ascend_018_reports_async_mp_boundary():
    """真实 AsyncLLM 前端进程只持有 AsyncMPClient 时，应明确提示多进程边界。"""

    AsyncMPClient = type("AsyncMPClient", (), {})
    AsyncMPClient.__module__ = "vllm.v1.engine.core_client"

    class Engine:
        def __init__(self):
            self.engine_core = AsyncMPClient()

    engine = Engine()

    assert detect_vllm_async_mp_client(engine) == "engine.engine_core"

    try:
        locate_vllm_ascend_018_kv_caches(engine)
    except LookupError as exc:
        message = str(exc)
    else:
        raise AssertionError("locator should fail at the AsyncMPClient boundary")

    assert "AsyncMPClient" in message
    assert "engine-core worker subprocess" in message
