from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger("jiuwen.kvbm.llm_datadist")


class LLMDataDistBackendError(RuntimeError):
    """LLM-DataDist backend 错误。"""


class LLMDataDistRole(str, Enum):
    """LLM-DataDist 节点角色。

    枚举值:
        PROMPT: Prefill/Prompt 角色。
        DECODER: Decode 角色。
        MIX: 同时承担 Prompt 和 Decode 的混合角色。
    """

    PROMPT = "prompt"
    DECODER = "decoder"
    MIX = "mix"


@dataclass(frozen=True)
class LLMDataDistEndpoint:
    """LLM-DataDist 节点监听端点。

    属性:
        cluster_id: LLM-DataDist cluster ID。
        role: 当前节点角色。
        listen_ip: 当前节点监听 IP。
        listen_port: 当前节点监听端口。
        device_id: 当前节点绑定的 NPU 设备编号。
    """

    cluster_id: int
    role: LLMDataDistRole
    listen_ip: str
    listen_port: int
    device_id: int


@dataclass(frozen=True)
class LLMDataDistConfig:
    """LLM-DataDist backend 初始化配置。

    属性:
        enabled: 是否启用 LLM-DataDist backend。
        endpoint: 本地 LLM-DataDist 端点信息。
        transfer_backend: 底层传输后端，例如 hixl。
        enable_cache_manager: 是否启用 DataDist cache manager 模式。
        enable_remote_cache_accessible: 是否启用远端 cache 可访问模式。
        extra_options: 透传给 LLM-DataDist 初始化的额外选项。
    """

    enabled: bool
    endpoint: LLMDataDistEndpoint
    transfer_backend: str = "hixl"
    enable_cache_manager: bool = True
    enable_remote_cache_accessible: bool = False
    extra_options: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMDataDistCacheDesc:
    """LLM-DataDist cache 描述。

    属性:
        placement: cache 位置，例如 device 或 host。
        data_type: cache dtype 字符串，例如 bf16、float16。
        shape: 单个 cache tensor 的形状。
        num_tensors: cache 中 tensor 数量。
        tensor_addresses: cache 中每个 tensor 的首地址，顺序需要与 LLM-DataDist
            按层传输约定一致，通常为 layer0_key、layer0_value、layer1_key、
            layer1_value。
        tensor_num_per_layer: 每层 KV tensor 数量，通常为 2。
        layer_count: KV cache 层数。
    """

    placement: str
    data_type: str
    shape: tuple[int, ...]
    num_tensors: int
    tensor_addresses: tuple[int, ...] = ()
    tensor_num_per_layer: int = 2
    layer_count: int = 0


@dataclass(frozen=True)
class LLMDataDistCacheKey:
    """LLM-DataDist cache key。

    属性:
        key: 上层稳定 cache key，第一版通常使用 block_hash 或 prefix hash。
        prompt_cluster_id: Prompt 侧 cluster ID。
        prompt_batch_index: Prompt batch index；block 级 pull 第一版固定为 0。
    """

    key: str
    prompt_cluster_id: int = 0
    prompt_batch_index: int = 0


@dataclass(frozen=True)
class LLMDataDistCluster:
    """LLM-DataDist 远端 cluster 建链信息。

    属性:
        remote_cluster_id: 对端 LLMDataDist 实例的 cluster ID。
        remote_ip: 对端 Host 通信 IP。
        remote_port: 对端 Host 通信端口。
        local_ip: 本端用于建立通信链路的 Host IP。
        local_port: 本端用于建立通信链路的 Host 端口。
    """

    remote_cluster_id: int
    remote_ip: str
    remote_port: int
    local_ip: str
    local_port: int


@dataclass(frozen=True)
class LLMDataDistCacheHandle:
    """LLM-DataDist cache 注册结果。

    属性:
        cache_id: LLM-DataDist 返回的 cache ID。
        desc: 注册时使用的 cache 描述。
        keys: 已注册的 cache key 列表。
        raw: 底层 wrapper 返回的原始对象，供调试和后续适配使用。
    """

    cache_id: int
    desc: LLMDataDistCacheDesc
    keys: tuple[LLMDataDistCacheKey, ...] = ()
    raw: Any = None


@dataclass(frozen=True)
class LLMDataDistBlockMapping:
    """LLM-DataDist block pull 映射。

    属性:
        cache_key: 远端 cache key。
        prompt_blocks: 源端 prompt block id 列表。
        decoder_blocks: 目标端 decoder block id 列表。
        layer_range: 可选 layer 范围；为空时表示全层。
    """

    cache_key: LLMDataDistCacheKey
    prompt_blocks: tuple[int, ...]
    decoder_blocks: tuple[int, ...]
    layer_range: tuple[int, int] | None = None


class LLMDataDistBackend(Protocol):
    """LLM-DataDist KV 语义数据面 backend 协议。"""

    def initialize(self) -> None:
        """初始化 backend。

        返回:
            无返回值。
        """

    def finalize(self) -> None:
        """释放 backend。

        返回:
            无返回值。
        """

    def health(self) -> dict[str, Any]:
        """返回 backend 健康状态。

        返回:
            backend 状态字典。
        """

    def register_cache(
        self,
        desc: LLMDataDistCacheDesc,
        keys: list[LLMDataDistCacheKey] | tuple[LLMDataDistCacheKey, ...] = (),
    ) -> LLMDataDistCacheHandle:
        """注册 KV cache。

        参数:
            desc: cache 描述。
            keys: 需要绑定到该 cache 的 cache key 列表。

        返回:
            cache 注册句柄。
        """

    def unregister_cache(self, cache_id: int) -> None:
        """解注册 KV cache。

        参数:
            cache_id: 待解注册 cache ID。

        返回:
            无返回值。
        """

    def link_clusters(
        self,
        clusters: list[LLMDataDistCluster] | tuple[LLMDataDistCluster, ...],
        *,
        timeout_ms: int = 3000,
    ) -> None:
        """与远端 LLM-DataDist cluster 建链。

        参数:
            clusters: 待建链的远端 cluster 列表。
            timeout_ms: 建链超时时间，单位毫秒。

        返回:
            无返回值。
        """

    def unlink_clusters(
        self,
        clusters: list[LLMDataDistCluster] | tuple[LLMDataDistCluster, ...],
        *,
        timeout_ms: int = 3000,
        force: bool = False,
    ) -> None:
        """断开远端 LLM-DataDist cluster 链路。

        参数:
            clusters: 待断链的远端 cluster 列表。
            timeout_ms: 断链超时时间，单位毫秒。
            force: 是否只强制清理本端链路。

        返回:
            无返回值。
        """

    def pull_blocks(self, cache_id: int, mapping: LLMDataDistBlockMapping) -> None:
        """按 block 语义从远端拉取 KV cache。

        参数:
            cache_id: 本地目标 cache ID。
            mapping: 源端和目标端 block 映射。

        返回:
            无返回值。
        """


class FakeLLMDataDistBackend:
    """测试和无 CANN 环境下使用的 LLM-DataDist backend。

    该实现只记录调用，不做真实传输，用于验证 KVBM 生命周期和配置链路。
    """

    def __init__(self, config: LLMDataDistConfig) -> None:
        """初始化 fake backend。

        参数:
            config: LLM-DataDist backend 配置。

        返回:
            无返回值。
        """

        self.config = config
        self.initialized = False
        self.caches: dict[int, LLMDataDistCacheHandle] = {}
        self.pull_requests: list[tuple[int, LLMDataDistBlockMapping]] = []
        self.linked_clusters: dict[int, LLMDataDistCluster] = {}
        self._next_cache_id = 1

    def initialize(self) -> None:
        """标记 fake backend 已初始化。

        返回:
            无返回值。
        """

        self.initialized = True

    def finalize(self) -> None:
        """标记 fake backend 已释放。

        返回:
            无返回值。
        """

        self.initialized = False

    def health(self) -> dict[str, Any]:
        """返回 fake backend 状态。

        返回:
            包含 initialized、mode、endpoint 的状态字典。
        """

        return {
            "ok": self.initialized,
            "mode": "fake",
            "initialized": self.initialized,
            "endpoint": _endpoint_dict(self.config.endpoint),
            "cache_count": len(self.caches),
            "pull_count": len(self.pull_requests),
            "linked_cluster_ids": sorted(self.linked_clusters),
        }

    def register_cache(
        self,
        desc: LLMDataDistCacheDesc,
        keys: list[LLMDataDistCacheKey] | tuple[LLMDataDistCacheKey, ...] = (),
    ) -> LLMDataDistCacheHandle:
        """记录 cache 注册请求。

        参数:
            desc: cache 描述。
            keys: cache key 列表。

        返回:
            fake cache 句柄。
        """

        self._require_initialized()
        cache_id = self._next_cache_id
        self._next_cache_id += 1
        handle = LLMDataDistCacheHandle(cache_id=cache_id, desc=desc, keys=tuple(keys))
        self.caches[cache_id] = handle
        return handle

    def unregister_cache(self, cache_id: int) -> None:
        """删除 fake cache 记录。

        参数:
            cache_id: 待删除 cache ID。

        返回:
            无返回值。
        """

        self.caches.pop(cache_id, None)

    def pull_blocks(self, cache_id: int, mapping: LLMDataDistBlockMapping) -> None:
        """记录 block pull 请求。

        参数:
            cache_id: 本地目标 cache ID。
            mapping: block 映射。

        返回:
            无返回值。
        """

        self._require_initialized()
        if cache_id not in self.caches:
            raise LLMDataDistBackendError(f"cache_id {cache_id} is not registered")
        self.pull_requests.append((cache_id, mapping))

    def link_clusters(
        self,
        clusters: list[LLMDataDistCluster] | tuple[LLMDataDistCluster, ...],
        *,
        timeout_ms: int = 3000,
    ) -> None:
        """记录 fake cluster 建链。

        参数:
            clusters: 待建链 cluster 列表。
            timeout_ms: 建链超时时间，fake backend 仅校验其为正数。

        返回:
            无返回值。
        """

        self._require_initialized()
        if timeout_ms <= 0:
            raise LLMDataDistBackendError("link timeout_ms must be positive")
        for cluster in clusters:
            self.linked_clusters[cluster.remote_cluster_id] = cluster

    def unlink_clusters(
        self,
        clusters: list[LLMDataDistCluster] | tuple[LLMDataDistCluster, ...],
        *,
        timeout_ms: int = 3000,
        force: bool = False,
    ) -> None:
        """删除 fake cluster 链路记录。

        参数:
            clusters: 待断链 cluster 列表。
            timeout_ms: 断链超时时间，fake backend 仅校验其为正数。
            force: 是否强制断链，fake backend 不区分具体行为。

        返回:
            无返回值。
        """

        self._require_initialized()
        if timeout_ms <= 0:
            raise LLMDataDistBackendError("unlink timeout_ms must be positive")
        for cluster in clusters:
            self.linked_clusters.pop(cluster.remote_cluster_id, None)

    def _require_initialized(self) -> None:
        """校验 fake backend 是否已初始化。

        返回:
            无返回值。
        """

        if not self.initialized:
            raise LLMDataDistBackendError("fake LLM-DataDist backend is not initialized")


class PythonLLMDataDistBackend:
    """基于 CANN llm_datadist Python wrapper 的 backend 骨架。

    该实现先负责 import、initialize_v2/finalize_v2 生命周期。cache 注册和
    pull_blocks 的精确参数需要在目标 CANN 容器确认 Python wrapper 签名后补齐。
    """

    def __init__(self, config: LLMDataDistConfig, *, module_name: str = "llm_datadist") -> None:
        """初始化 Python wrapper backend。

        参数:
            config: LLM-DataDist backend 配置。
            module_name: Python wrapper 模块名，默认 llm_datadist。

        返回:
            无返回值。
        """

        self.config = config
        self.module_name = module_name
        self.module: Any | None = None
        self.datadist: Any | None = None
        self.initialized = False
        self.caches: dict[int, LLMDataDistCacheHandle] = {}
        self.linked_clusters: dict[int, LLMDataDistCluster] = {}

    def initialize(self) -> None:
        """导入 llm_datadist 并调用 initialize_v2。

        返回:
            无返回值。
        """

        try:
            self.module = importlib.import_module(self.module_name)
        except ImportError as exc:
            raise LLMDataDistBackendError(
                f"failed to import {self.module_name}; install CANN LLM-DataDist Python package"
            ) from exc
        if hasattr(self.module, "LLMDataDist"):
            self._initialize_object_api()
            self.initialized = True
            logger.info("LLM-DataDist backend initialized: %s", self.health())
            return
        initialize = getattr(self.module, "initialize_v2", None)
        if initialize is None:
            raise LLMDataDistBackendError(
                f"{self.module_name} exposes neither LLMDataDist nor initialize_v2; unsupported LLM-DataDist wrapper"
            )
        options = self._initialize_options()
        try:
            result = initialize(options)
        except TypeError as exc:
            raise LLMDataDistBackendError(
                f"{self.module_name}.initialize_v2 rejected options={options!r}; "
                "confirm the CANN 8.5.1 Python wrapper signature"
            ) from exc
        _raise_if_bad_status(result, "initialize_v2")
        self.initialized = True
        logger.info("LLM-DataDist backend initialized: %s", self.health())

    def finalize(self) -> None:
        """调用 finalize_v2 释放 wrapper。

        返回:
            无返回值。
        """

        try:
            if self.module is None:
                self.initialized = False
                return
            if self.datadist is not None and hasattr(self.datadist, "finalize"):
                result = self.datadist.finalize()
                _raise_if_bad_status(result, "LLMDataDist.finalize")
            else:
                finalize = getattr(self.module, "finalize_v2", None)
                if finalize is None:
                    self.initialized = False
                    return
                result = finalize()
                _raise_if_bad_status(result, "finalize_v2")
            self.initialized = False
        finally:
            self.initialized = False

    def health(self) -> dict[str, Any]:
        """返回 Python wrapper backend 状态。

        返回:
            状态字典。
        """

        return {
            "ok": self.initialized,
            "mode": "python",
            "module": self.module_name,
            "initialized": self.initialized,
            "endpoint": _endpoint_dict(self.config.endpoint),
            "transfer_backend": self.config.transfer_backend,
            "api": "object" if self.datadist is not None else "function",
            "linked_cluster_ids": sorted(self.linked_clusters),
        }

    def register_cache(
        self,
        desc: LLMDataDistCacheDesc,
        keys: list[LLMDataDistCacheKey] | tuple[LLMDataDistCacheKey, ...] = (),
    ) -> LLMDataDistCacheHandle:
        """注册 KV cache。

        参数:
            desc: cache 描述。
            keys: cache key 列表。

        返回:
            cache 注册句柄。
        """

        self._require_initialized()
        cache_manager = getattr(self.datadist, "cache_manager", None)
        if cache_manager is not None and hasattr(cache_manager, "register_blocks_cache"):
            return self._register_blocks_cache_object_api(cache_manager, desc, tuple(keys))
        raise LLMDataDistBackendError(
            "llm_datadist object CacheManager.register_blocks_cache is missing; "
            "function-style register_cache is intentionally unsupported for blocks cache"
        )

    def unregister_cache(self, cache_id: int) -> None:
        """解注册 KV cache。

        参数:
            cache_id: 待解注册 cache ID。

        返回:
            无返回值。
        """

        self._require_initialized()
        handle = self.caches.get(cache_id)
        cache_manager = getattr(self.datadist, "cache_manager", None)
        unregister = getattr(cache_manager, "unregister_cache", None) if cache_manager is not None else None
        if unregister is None:
            raise LLMDataDistBackendError("llm_datadist.unregister_cache is missing")
        result = unregister(cache_id)
        _raise_if_bad_status(result, "unregister_cache")
        if handle is not None:
            self.caches.pop(cache_id, None)

    def pull_blocks(self, cache_id: int, mapping: LLMDataDistBlockMapping) -> None:
        """按 block 语义拉取 KV cache。

        参数:
            cache_id: 本地目标 cache ID。
            mapping: block 映射。

        返回:
            无返回值。
        """

        self._require_initialized()
        cache_manager = getattr(self.datadist, "cache_manager", None)
        pull = None
        if cache_manager is not None:
            pull = getattr(cache_manager, "pull_blocks", None)
        if pull is None and self.module is not None:
            pull = getattr(self.module, "pull_cache_v2", None)
        if pull is None:
            raise LLMDataDistBackendError("llm_datadist pull_blocks/pull_cache_v2 is missing")
        if cache_manager is None:
            raise LLMDataDistBackendError("function-style pull_cache_v2 is intentionally unsupported for blocks cache")
        handle = self.caches.get(cache_id)
        if handle is None:
            raise LLMDataDistBackendError(f"cache_id {cache_id} is not registered")
        src_cache_key = self._blocks_cache_key(mapping.cache_key)
        kwargs: dict[str, Any] = {"tensor_num_per_layer": int(handle.desc.tensor_num_per_layer)}
        if mapping.layer_range is not None:
            start, stop = mapping.layer_range
            kwargs["src_layer_range"] = range(start, stop)
            kwargs["dst_layer_range"] = range(start, stop)
        try:
            result = pull(
                src_cache_key,
                handle.raw,
                list(mapping.prompt_blocks),
                list(mapping.decoder_blocks),
                **kwargs,
            )
        except TypeError as exc:
            raise LLMDataDistBackendError(
                "llm_datadist CacheManager.pull_blocks rejected mapped arguments: "
                f"cache_id={cache_id}, mapping={mapping!r}, kwargs={kwargs!r}"
            ) from exc
        _raise_if_bad_status(result, "CacheManager.pull_blocks")

    def link_clusters(
        self,
        clusters: list[LLMDataDistCluster] | tuple[LLMDataDistCluster, ...],
        *,
        timeout_ms: int = 3000,
    ) -> None:
        """调用 CANN LLMDataDist.link_clusters 建立远端链路。

        参数:
            clusters: 待建链的远端 cluster 列表。
            timeout_ms: 建链超时时间，单位毫秒。

        返回:
            所有 cluster 建链成功时无返回值。
        """

        self._require_initialized()
        if self.datadist is None or not hasattr(self.datadist, "link_clusters"):
            raise LLMDataDistBackendError("llm_datadist LLMDataDist.link_clusters is missing")
        raw_clusters = [self._cluster_info(cluster) for cluster in clusters]
        result = self.datadist.link_clusters(raw_clusters, timeout=int(timeout_ms))
        logger.info(
            "LLM-DataDist link_clusters returned: remote_cluster_ids=%s result=%r",
            [cluster.remote_cluster_id for cluster in clusters],
            result,
        )
        _raise_cluster_statuses(
            result,
            "LLMDataDist.link_clusters",
            clusters,
            accepted_status_names={"LLM_ALREADY_LINK", "ALREADY_LINK"},
        )
        for cluster in clusters:
            self.linked_clusters[cluster.remote_cluster_id] = cluster
        logger.info(
            "LLM-DataDist linked clusters: remote_cluster_ids=%s",
            [cluster.remote_cluster_id for cluster in clusters],
        )

    def unlink_clusters(
        self,
        clusters: list[LLMDataDistCluster] | tuple[LLMDataDistCluster, ...],
        *,
        timeout_ms: int = 3000,
        force: bool = False,
    ) -> None:
        """调用 CANN LLMDataDist.unlink_clusters 断开远端链路。

        参数:
            clusters: 待断链的远端 cluster 列表。
            timeout_ms: 断链超时时间，单位毫秒。
            force: 是否只强制清理本端链路。

        返回:
            所有 cluster 断链成功时无返回值。
        """

        self._require_initialized()
        if self.datadist is None or not hasattr(self.datadist, "unlink_clusters"):
            raise LLMDataDistBackendError("llm_datadist LLMDataDist.unlink_clusters is missing")
        raw_clusters = [self._cluster_info(cluster) for cluster in clusters]
        result = self.datadist.unlink_clusters(
            raw_clusters,
            timeout=int(timeout_ms),
            force=bool(force),
        )
        logger.info(
            "LLM-DataDist unlink_clusters returned: remote_cluster_ids=%s result=%r",
            [cluster.remote_cluster_id for cluster in clusters],
            result,
        )
        _raise_cluster_statuses(result, "LLMDataDist.unlink_clusters", clusters)
        for cluster in clusters:
            self.linked_clusters.pop(cluster.remote_cluster_id, None)
        logger.info(
            "LLM-DataDist unlinked clusters: remote_cluster_ids=%s force=%s",
            [cluster.remote_cluster_id for cluster in clusters],
            force,
        )

    def _cluster_info(self, cluster: LLMDataDistCluster) -> Any:
        """转换中性 cluster 描述为 CANN LLMClusterInfo。

        参数:
            cluster: Jiuwen 中性的远端 cluster 描述。

        返回:
            已填写本地和远端 IP 信息的 CANN LLMClusterInfo。
        """

        if not (self.module is not None):
            raise RuntimeError("Required invariant failed: self.module is not None")
        cluster_cls = getattr(self.module, "LLMClusterInfo", None)
        if cluster_cls is None:
            raise LLMDataDistBackendError("llm_datadist.LLMClusterInfo is missing")
        raw = cluster_cls()
        raw.remote_cluster_id = int(cluster.remote_cluster_id)
        raw.append_local_ip_info(cluster.local_ip, int(cluster.local_port))
        raw.append_remote_ip_info(cluster.remote_ip, int(cluster.remote_port))
        return raw

    def _register_blocks_cache_object_api(
        self,
        cache_manager: Any,
        desc: LLMDataDistCacheDesc,
        keys: tuple[LLMDataDistCacheKey, ...],
    ) -> LLMDataDistCacheHandle:
        """使用 CacheManager.register_blocks_cache 注册 PA blocks cache。

        参数:
            cache_manager: CANN LLM-DataDist 暴露的 cache manager 对象。
            desc: Jiuwen 中性的 cache 描述，必须携带 tensor_addresses。
            keys: 可选 cache key；第一版使用第一个 BlocksCacheKey 作为注册索引。

        返回:
            注册完成后的 cache 句柄。
        """

        if self.module is None:
            raise LLMDataDistBackendError("llm_datadist module is not loaded")
        if not desc.tensor_addresses:
            raise LLMDataDistBackendError("LLM-DataDist register_blocks_cache requires tensor_addresses")
        if len(desc.tensor_addresses) != int(desc.num_tensors):
            raise LLMDataDistBackendError(
                "LLM-DataDist tensor address count does not match num_tensors: "
                f"addresses={len(desc.tensor_addresses)}, num_tensors={desc.num_tensors}"
            )
        cache_desc = self._cache_desc(desc)
        blocks_cache_key = self._blocks_cache_key(keys[0]) if keys else None
        register = getattr(cache_manager, "register_blocks_cache")
        try:
            if self.config.enable_remote_cache_accessible:
                raw_cache = register(
                    cache_desc,
                    list(desc.tensor_addresses),
                    blocks_cache_key,
                    True,
                )
            else:
                raw_cache = register(cache_desc, list(desc.tensor_addresses), blocks_cache_key)
        except TypeError:
            raw_cache = register(cache_desc, list(desc.tensor_addresses), blocks_cache_key)
        cache_id = _cache_id(raw_cache)
        handle = LLMDataDistCacheHandle(
            cache_id=cache_id,
            desc=desc,
            keys=keys,
            raw=raw_cache,
        )
        self.caches[cache_id] = handle
        logger.info(
            "LLM-DataDist registered blocks cache: cache_id=%s, num_tensors=%s, shape=%s, keys=%s",
            cache_id,
            desc.num_tensors,
            desc.shape,
            keys,
        )
        return handle

    def _cache_desc(self, desc: LLMDataDistCacheDesc) -> Any:
        """转换 Jiuwen cache 描述为 CANN CacheDesc。

        参数:
            desc: Jiuwen 中性的 cache 描述。

        返回:
            CANN llm_datadist.CacheDesc 实例。
        """

        if not (self.module is not None):
            raise RuntimeError("Required invariant failed: self.module is not None")
        cache_desc_cls = getattr(self.module, "CacheDesc", None)
        if cache_desc_cls is None:
            raise LLMDataDistBackendError("llm_datadist.CacheDesc is missing")
        return cache_desc_cls(
            int(desc.num_tensors),
            list(desc.shape),
            self._data_type(desc.data_type),
            self._placement(desc.placement),
        )

    def _blocks_cache_key(self, key: LLMDataDistCacheKey) -> Any:
        """转换 Jiuwen cache key 为 CANN BlocksCacheKey。

        参数:
            key: Jiuwen 中性的 cache key。

        返回:
            CANN llm_datadist.BlocksCacheKey 实例。
        """

        if not (self.module is not None):
            raise RuntimeError("Required invariant failed: self.module is not None")
        key_cls = getattr(self.module, "BlocksCacheKey", None)
        if key_cls is None:
            raise LLMDataDistBackendError("llm_datadist.BlocksCacheKey is missing")
        return key_cls(int(key.prompt_cluster_id), int(key.prompt_batch_index))

    def _data_type(self, data_type: str) -> Any:
        """转换 dtype 字符串为 CANN DataType 枚举。

        参数:
            data_type: dtype 字符串，例如 torch.bfloat16、bf16、float16。

        返回:
            CANN llm_datadist.DataType 枚举值。
        """

        if not (self.module is not None):
            raise RuntimeError("Required invariant failed: self.module is not None")
        enum = getattr(self.module, "DataType", None)
        if enum is None:
            raise LLMDataDistBackendError("llm_datadist.DataType is missing")
        normalized = data_type.lower().replace("torch.", "").replace("_", "")
        candidates = {
            "bfloat16": "DT_BF16",
            "bf16": "DT_BF16",
            "float16": "DT_FLOAT16",
            "fp16": "DT_FLOAT16",
            "half": "DT_FLOAT16",
            "float32": "DT_FLOAT",
            "float": "DT_FLOAT",
            "fp32": "DT_FLOAT",
            "int8": "DT_INT8",
        }
        enum_name = candidates.get(normalized)
        if enum_name is None or not hasattr(enum, enum_name):
            raise LLMDataDistBackendError(f"unsupported LLM-DataDist dtype: {data_type}")
        return getattr(enum, enum_name)

    def _placement(self, placement: str) -> Any:
        """转换 placement 字符串为 CANN Placement 枚举。

        参数:
            placement: cache 位置字符串，支持 device、npu、host、cpu。

        返回:
            CANN llm_datadist.Placement 枚举值。
        """

        if not (self.module is not None):
            raise RuntimeError("Required invariant failed: self.module is not None")
        enum = getattr(self.module, "Placement", None)
        if enum is None:
            raise LLMDataDistBackendError("llm_datadist.Placement is missing")
        normalized = placement.lower()
        enum_name = "HOST" if normalized in {"host", "cpu", "ram"} else "DEVICE"
        if not hasattr(enum, enum_name):
            raise LLMDataDistBackendError(f"llm_datadist.Placement.{enum_name} is missing")
        return getattr(enum, enum_name)

    def _initialize_object_api(self) -> None:
        """使用 CANN LLMDataDist 面向对象 API 初始化。

        返回:
            无返回值。
        """

        if not (self.module is not None):
            raise RuntimeError("Required invariant failed: self.module is not None")
        endpoint = self.config.endpoint
        role = _module_role(self.module, endpoint.role)
        datadist_cls = getattr(self.module, "LLMDataDist")
        self.datadist = datadist_cls(role, endpoint.cluster_id)
        config_cls = getattr(self.module, "LLMConfig", None)
        if config_cls is not None:
            llm_config = config_cls()
            llm_config.device_id = endpoint.device_id
            llm_config.listen_ip_info = f"{endpoint.listen_ip}:{endpoint.listen_port}"
            llm_config.enable_cache_manager = self.config.enable_cache_manager
            if self.config.enable_remote_cache_accessible:
                llm_config.enable_remote_cache_accessible = True
            if self.config.transfer_backend:
                try:
                    llm_config.transfer_backend = self.config.transfer_backend
                except Exception:
                    logger.debug("Installed LLMConfig does not expose transfer_backend", exc_info=True)
            options = llm_config.generate_options()
            options.update(self.config.extra_options)
        else:
            options = self._initialize_options()
        result = self.datadist.init(options)
        _raise_if_bad_status(result, "LLMDataDist.init")

    def _initialize_options(self) -> dict[str, str]:
        """构造 LLM-DataDist initialize_v2 options。

        返回:
            传入 Python wrapper 的 options 字典。
        """

        endpoint = self.config.endpoint
        options = {
            "llm.DeviceId": str(endpoint.device_id),
            "llm.Role": _role_value(endpoint.role),
            "llm.ListenIpInfo": f"{endpoint.listen_ip}:{endpoint.listen_port}",
            "llm.TransferBackend": self.config.transfer_backend,
            "llm.EnableCacheManager": "1" if self.config.enable_cache_manager else "0",
            "llm.EnableRemoteCacheAccessible": ("1" if self.config.enable_remote_cache_accessible else "0"),
        }
        options.update(self.config.extra_options)
        return options

    def _require_initialized(self) -> None:
        """校验 Python wrapper backend 已初始化。

        返回:
            无返回值。
        """

        if not self.initialized:
            raise LLMDataDistBackendError("LLM-DataDist backend is not initialized")


def create_llm_datadist_backend(
    config: LLMDataDistConfig,
    *,
    mode: str = "python",
) -> LLMDataDistBackend:
    """创建 LLM-DataDist backend。

    参数:
        config: backend 配置。
        mode: backend 模式，支持 fake、python、disabled。

    返回:
        backend 实例。
    """

    normalized = mode.lower()
    if normalized in {"fake", "mock"}:
        return FakeLLMDataDistBackend(config)
    if normalized in {"python", "real", "llm_datadist"}:
        return PythonLLMDataDistBackend(config)
    raise LLMDataDistBackendError(f"unsupported LLM-DataDist backend mode: {mode}")


def _endpoint_dict(endpoint: LLMDataDistEndpoint) -> dict[str, Any]:
    """转换 endpoint 为日志友好的字典。

    参数:
        endpoint: LLM-DataDist endpoint。

    返回:
        字典形式 endpoint。
    """

    return {
        "cluster_id": endpoint.cluster_id,
        "role": endpoint.role.value,
        "listen_ip": endpoint.listen_ip,
        "listen_port": endpoint.listen_port,
        "device_id": endpoint.device_id,
    }


def _role_value(role: LLMDataDistRole) -> str:
    """转换角色为 CANN wrapper 常见字符串。

    参数:
        role: LLMDataDistRole。

    返回:
        Prompt、Decoder 或 Mix。
    """

    if role == LLMDataDistRole.PROMPT:
        return "Prompt"
    if role == LLMDataDistRole.DECODER:
        return "Decoder"
    return "Mix"


def _module_role(module: Any, role: LLMDataDistRole) -> Any:
    """转换角色为 llm_datadist 模块内 LLMRole 枚举。

    参数:
        module: llm_datadist Python 模块。
        role: Jiuwen 中性的 LLMDataDistRole。

    返回:
        模块内 LLMRole 枚举值；不存在时退化为字符串。
    """

    llm_role = getattr(module, "LLMRole", None)
    if llm_role is None:
        return _role_value(role)
    if role == LLMDataDistRole.PROMPT:
        return getattr(llm_role, "PROMPT", getattr(llm_role, "Prompt", _role_value(role)))
    if role == LLMDataDistRole.DECODER:
        return getattr(llm_role, "DECODER", getattr(llm_role, "Decoder", _role_value(role)))
    return getattr(llm_role, "MIX", getattr(llm_role, "Mix", _role_value(role)))


def _raise_if_bad_status(
    result: Any,
    operation: str,
    *,
    accepted_status_names: set[str] | frozenset[str] = frozenset(),
) -> None:
    """检查 CANN wrapper 返回状态。

    参数:
        result: wrapper 调用返回值。
        operation: 操作名，用于错误信息。
        accepted_status_names: 除 SUCCESS 外允许按成功处理的状态名称。

    返回:
        状态成功时无返回值。
    """

    if result is None:
        return
    status = result[0] if isinstance(result, tuple) and result else result
    name = str(getattr(status, "name", "")).upper()
    if name:
        if name in {"LLM_SUCCESS", "SUCCESS"} or name in accepted_status_names:
            return
        raise LLMDataDistBackendError(f"{operation} failed with status={status!r}")
    value = getattr(status, "value", status)
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        text = str(status).upper()
        if text.endswith(".LLM_SUCCESS") or text.endswith(".SUCCESS"):
            return
        if any(
            text == accepted or text.endswith(f".{accepted}") or f".{accepted}:" in text
            for accepted in accepted_status_names
        ):
            return
        return
    if numeric != 0:
        raise LLMDataDistBackendError(f"{operation} failed with status={status!r} numeric_status={numeric}")


def _raise_cluster_statuses(
    result: Any,
    operation: str,
    clusters: list[LLMDataDistCluster] | tuple[LLMDataDistCluster, ...],
    *,
    accepted_status_names: set[str] | frozenset[str] = frozenset(),
) -> None:
    """检查 link/unlink 的总状态和逐 cluster 状态。

    参数:
        result: CANN link_clusters 或 unlink_clusters 返回值。
        operation: 操作名称，用于错误信息。
        clusters: 与逐项状态位置对应的 cluster 列表。
        accepted_status_names: 除 SUCCESS 外允许按成功处理的状态名称。

    返回:
        所有状态成功时无返回值。
    """

    if not isinstance(result, tuple) or len(result) < 2:
        _raise_if_bad_status(
            result,
            operation,
            accepted_status_names=accepted_status_names,
        )
        return
    _raise_if_bad_status(
        result[0],
        operation,
        accepted_status_names=accepted_status_names,
    )
    statuses = result[1] or []
    for index, status in enumerate(statuses):
        cluster_id = clusters[index].remote_cluster_id if index < len(clusters) else "unknown"
        _raise_if_bad_status(
            status,
            f"{operation}[remote_cluster_id={cluster_id}]",
            accepted_status_names=accepted_status_names,
        )


def _cache_id(raw_cache: Any) -> int:
    """从 CANN Cache 对象提取 cache_id。

    参数:
        raw_cache: register_blocks_cache 返回的 CANN Cache 对象。

    返回:
        cache_id 整数；无法提取时抛出异常。
    """

    value = getattr(raw_cache, "cache_id", None)
    if callable(value):
        value = value()
    if value is None:
        raise LLMDataDistBackendError("registered LLM-DataDist cache has no cache_id")
    return int(value)
