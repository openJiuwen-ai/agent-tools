from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("jiuwen.worker.kvbm")


@dataclass(frozen=True)
class VllmWorkerKVBMConfig:
    """vLLM worker 侧 KVBM 配置。

    属性:
        enabled: 是否启用 worker 侧 KVBM 数据面。
        worker_id: worker 唯一标识。
        disk_root: 本地磁盘后端目录。
        clean_disk_on_start: 是否在启动时清理旧磁盘 payload。
        npu_high_watermark_blocks: NPU cached block 数超过该值时触发 NPU->RAM。
        ram_high_watermark_bytes: RAM 占用超过该值时触发 RAM->DISK。
        interval: 周期性执行水位策略的间隔秒数。
        block_size: 默认 block token 数。
        provider: worker 侧 KV provider 类型，空字符串表示不自动定位。
        diagnostics: 是否在启动时打印只读 KV cache 诊断信息。
        locator_depth: 自动定位 vLLM-Ascend kv_caches 时遍历对象树的最大深度。
    """

    enabled: bool
    worker_id: str
    disk_root: Path
    clean_disk_on_start: bool
    npu_high_watermark_blocks: int
    ram_high_watermark_bytes: int
    interval: float
    block_size: int
    provider: str
    diagnostics: bool
    locator_depth: int

    @classmethod
    def from_args(cls, args: Any, worker_id: str) -> "VllmWorkerKVBMConfig":
        """从 worker CLI/YAML 参数创建配置。

        参数:
            args: worker argparse Namespace。
            worker_id: 当前 worker 唯一标识。

        返回:
            VllmWorkerKVBMConfig 实例。
        """

        return cls(
            enabled=bool(getattr(args, "kvbm_worker_enabled", False)),
            worker_id=worker_id,
            disk_root=Path(getattr(args, "kvbm_worker_disk_root", "/tmp/jiuwen-kvbm-worker")),
            clean_disk_on_start=bool(getattr(args, "kvbm_clean_disk_on_start", True)),
            npu_high_watermark_blocks=int(getattr(args, "kvbm_npu_high_watermark_blocks", 0)),
            ram_high_watermark_bytes=int(getattr(args, "kvbm_ram_high_watermark_bytes", 0)),
            interval=float(getattr(args, "kvbm_worker_interval", 5.0)),
            block_size=int(getattr(args, "kvbm_block_size", 16)),
            provider=str(getattr(args, "kvbm_worker_provider", "")),
            diagnostics=bool(getattr(args, "kvbm_worker_diagnostics", False)),
            locator_depth=int(getattr(args, "kvbm_worker_locator_depth", 16)),
        )


class VllmWorkerKVBMAdapter:
    """vLLM/vLLM-Ascend worker 侧 KVBM 适配器。

    参数:
        config: worker KVBM 配置。
        engine: vLLM AsyncLLM 引擎对象。
        worker_publisher: worker 事件发布器，用于向 router 上报 move 事件。
        provider_factory: 可选 provider 工厂。该工厂接收 engine 并返回
            BlockPayloadProvider。未提供时 adapter 保持 inactive，不访问 engine 私有字段。
    """

    def __init__(
        self,
        config: VllmWorkerKVBMConfig,
        engine: Any,
        worker_publisher: Any | None = None,
        provider_factory: Callable[[Any], Any] | None = None,
    ):
        self.config = config
        self.engine = engine
        self.worker_publisher = worker_publisher
        self.provider_factory = provider_factory
        self.runtime = None
        self.active = False
        self.inactive_reason = ""
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """启动 worker 侧 KVBM adapter。

        返回:
            无返回值。
        """

        if not self.config.enabled:
            self.inactive_reason = "disabled"
            logger.info("Worker KVBM adapter disabled")
            return
        if self._defer_to_engine_core_hook():
            self.inactive_reason = "engine_core_hook_mode"
            logger.info(
                "Worker KVBM adapter deferred to vLLM-Ascend engine-core hook; "
                "waiting for Jiuwen KVBM engine-core diagnostics from the worker subprocess"
            )
            return
        provider_factory = self.provider_factory or self._build_provider_factory()
        if provider_factory is None:
            self.inactive_reason = "provider_factory_missing"
            logger.warning("Worker KVBM adapter inactive: provider factory is missing")
            return

        try:
            from jiuwen_kvbm import WorkerKVBMConfig, WorkerKVBMRuntime
            from jiuwen_kvbm.accessor import VllmAscendKVAccessor

            provider = provider_factory(self.engine)
            self._log_provider_diagnostics(provider)
            accessor = VllmAscendKVAccessor(provider)
            self.runtime = WorkerKVBMRuntime(
                WorkerKVBMConfig(
                    worker_id=self.config.worker_id,
                    disk_root=self.config.disk_root,
                    clean_disk_on_start=self.config.clean_disk_on_start,
                    npu_high_watermark_blocks=self.config.npu_high_watermark_blocks,
                    ram_high_watermark_bytes=self.config.ram_high_watermark_bytes,
                    block_size=self.config.block_size,
                ),
                accessor=accessor,
            )
            self.active = True
            self._task = asyncio.create_task(self._run_watermark_loop())
            logger.info("Worker KVBM adapter started")
        except Exception as exc:
            self.active = False
            self.inactive_reason = str(exc)
            logger.warning("Worker KVBM adapter failed to start: %s", exc)

    def _defer_to_engine_core_hook(self) -> bool:
        """判断当前外层 adapter 是否应让位给 engine-core hook。

        返回:
            当 provider 为 vLLM-Ascend 0.18，且当前 engine 是 vLLM 前端
            AsyncMPClient 多进程模式时返回 True；否则返回 False。
        """

        if self.provider_factory is not None:
            return False
        from openjiuwentools.infer_router.vllm.kvbm_provider import (
            get_kvbm_provider_spec,
        )

        spec = get_kvbm_provider_spec(self.config.provider)
        if spec is None or not spec.engine_core_hook:
            return False
        try:
            from jiuwen_kvbm.accessor import detect_vllm_async_mp_client
        except Exception:
            return False
        return detect_vllm_async_mp_client(self.engine) is not None

    def _build_provider_factory(self) -> Callable[[Any], Any] | None:
        """根据配置创建内置 provider 工厂。

        返回:
            provider 工厂函数；当 provider 配置为空或未知时返回 None。
        """

        from openjiuwentools.infer_router.vllm.kvbm_provider import (
            get_kvbm_provider_spec,
        )

        spec = get_kvbm_provider_spec(self.config.provider)
        if spec is None:
            return None
        if spec.name == "vllm-ascend-0.18":
            from jiuwen_kvbm.accessor import (
                VllmAscend018TensorProvider,
                locate_vllm_ascend_018_kv_caches,
            )

            def _factory(engine: Any) -> Any:
                kv_caches, path = locate_vllm_ascend_018_kv_caches(
                    engine,
                    max_depth=self.config.locator_depth,
                )
                provider = VllmAscend018TensorProvider(kv_caches)
                setattr(provider, "_jiuwen_kvbm_kv_caches_path", path)
                return provider

            return _factory
        if spec.name == "vllm-ascend-0.23":
            from jiuwen_kvbm.accessor import (
                VllmAscend023TensorProvider,
                locate_vllm_ascend_023_kv_caches,
            )

            def _factory(engine: Any) -> Any:
                kv_caches, path = locate_vllm_ascend_023_kv_caches(
                    engine,
                    max_depth=self.config.locator_depth,
                )
                provider = VllmAscend023TensorProvider(kv_caches)
                setattr(provider, "_jiuwen_kvbm_kv_caches_path", path)
                return provider

            return _factory
        logger.warning("Worker KVBM provider %s is not supported", self.config.provider)
        return None

    def _log_provider_diagnostics(self, provider: Any) -> None:
        """记录 provider 的只读诊断信息。

        参数:
            provider: 已创建的 BlockPayloadProvider。

        返回:
            无返回值。
        """

        if not self.config.diagnostics:
            return
        path = getattr(provider, "_jiuwen_kvbm_kv_caches_path", "<provided-by-factory>")
        logger.info("Worker KVBM diagnostics: kv_caches_path=%s", path)
        diagnostics_fn = getattr(provider, "diagnostics", None)
        if diagnostics_fn is None:
            logger.info("Worker KVBM diagnostics: provider has no diagnostics() method")
            return
        try:
            diagnostics = diagnostics_fn(0)
        except Exception as exc:
            logger.warning("Worker KVBM diagnostics failed: %s", exc)
            return
        logger.info("Worker KVBM diagnostics: %s", diagnostics)

    async def stop(self) -> None:
        """停止 worker 侧 KVBM adapter。

        返回:
            无返回值。
        """

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        self.active = False

    def on_block_stored(
        self,
        block_id: int,
        block_hash: str,
        *,
        token_count: int | None = None,
        session_id: str | None = None,
    ) -> None:
        """通知 adapter 有新的 KV Block 生成。

        参数:
            block_id: vLLM-Ascend 物理 Block 编号。
            block_hash: Block hash。
            token_count: 该 Block 覆盖 token 数；为空时使用配置默认值。
            session_id: 可选 session 标识。

        返回:
            无返回值。
        """

        if not self.active or self.runtime is None:
            return
        self.runtime.register_block(
            block_id,
            block_hash,
            token_count=token_count,
            session_id=session_id,
            active=False,
        )

    def apply_kv_events(self, events: list[Any] | tuple[Any, ...]) -> list[Any]:
        """批量消费 vLLM/vLLM-Ascend KV 事件。

        参数:
            events: vLLM KV event 对象或 relay dict 列表。

        返回:
            KVBM runtime 产生的事件列表；adapter 未激活时返回空列表。
        """

        results = []
        for event in events:
            result = self.apply_kv_event(event)
            if result is None:
                continue
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        return results

    def apply_kv_event(self, event: Any) -> Any | list[Any] | None:
        """消费单个 vLLM/vLLM-Ascend KV 事件。

        参数:
            event: vLLM KV event 对象，或形如 {"type": ..., "data": ...} 的
                relay dict。

        返回:
            事件成功映射时返回 KVBM runtime 事件；无法识别、adapter 未激活、
            或事件缺少物理 block id 时返回 None。
        """

        if not self.active or self.runtime is None:
            return None
        normalized = _normalize_kv_event(event)
        block_ids = normalized["block_ids"]
        block_hashes = normalized["block_hashes"]
        token_count = normalized["token_count"] or self.config.block_size
        if not block_ids:
            logger.debug("Worker KVBM ignored KV event without physical block ids: %s", normalized)
            return None

        event_type = normalized["event_type"]
        if event_type == "store":
            return [
                self.runtime.register_block(
                    block_id,
                    _block_hash_for_index(block_hashes, index, block_id),
                    token_count=token_count,
                    active=False,
                )
                for index, block_id in enumerate(block_ids)
            ]
        if event_type in {"removed", "remove", "evict", "cleared", "clear"}:
            results = []
            for block_id in block_ids:
                try:
                    results.append(self.runtime.manager.evict_block(block_id))
                except KeyError:
                    continue
            return results
        if event_type in {"move", "transfer"}:
            target = str(normalized["target"]).lower()
            results = []
            for index, block_id in enumerate(block_ids):
                block_hash = _block_hash_for_index(block_hashes, index, block_id)
                if not _is_registered(self.runtime.manager, block_id):
                    self.runtime.register_block(
                        block_id,
                        block_hash,
                        token_count=token_count,
                        active=False,
                    )
                block = self.runtime.manager.get_metadata(block_id)
                if target in {"cpu", "host", "ram", "host_ram"}:
                    results.append(self.runtime.offload_to_ram(block))
                elif target in {"disk", "ssd", "nvme", "file", "filesystem", "remote"}:
                    results.append(self.runtime.offload_to_disk(block))
                elif target in {"gpu", "npu", "device"}:
                    event_result = self.runtime.prefetch_block(block_hash)
                    if event_result is not None:
                        results.append(event_result)
            return results
        if event_type == "hit":
            for block_hash in block_hashes:
                self.runtime.manager.match([block_hash])
        return None

    def get_metrics(self) -> dict[str, Any]:
        """返回 worker KVBM 指标。

        返回:
            可合入 worker metrics 的字典。
        """

        data = {
            "kvbm_worker_enabled": self.config.enabled,
            "kvbm_worker_active": self.active,
        }
        if self.inactive_reason:
            data["kvbm_worker_inactive_reason"] = self.inactive_reason
        if self.runtime is not None:
            data["kvbm_worker_stats"] = self.runtime.stats()
        return data

    async def _run_watermark_loop(self) -> None:
        """周期性执行 KVBM 水位策略并上报事件。

        返回:
            无返回值。
        """

        while True:
            try:
                if self.runtime is not None:
                    for event in self.runtime.enforce_watermarks():
                        await self._publish_runtime_event(event)
            except Exception as exc:
                logger.error("Worker KVBM watermark loop failed: %s", exc)
            await asyncio.sleep(self.config.interval)

    async def _publish_runtime_event(self, event: Any) -> None:
        """发布 runtime 产生的 KVBM 事件。

        参数:
            event: KVBMRuntimeEvent 实例。

        返回:
            无返回值。
        """

        if self.worker_publisher is None:
            return
        from openjiuwentools.infer_router.vllm.worker_publisher import TOPIC_KV_EVENTS

        payload = event.to_dict()
        relay_msg = {
            "worker_id": self.config.worker_id,
            "seq": 0,
            "ts": "",
            "events": [
                {
                    "type": "BlockMoved" if payload["event_type"] == "move" else payload["event_type"],
                    "data": {
                        "block_hashes": payload["block_hashes"],
                        "block_size": self.config.block_size,
                        "medium": payload["location"],
                        "to": payload["location"],
                        "from": payload["engine_specific"].get("from"),
                    },
                }
            ],
        }
        await self.worker_publisher.publish(TOPIC_KV_EVENTS, relay_msg)


def _normalize_kv_event(event: Any) -> dict[str, Any]:
    """将 vLLM KV event 对象或 dict 标准化为 worker 数据面事件。

    参数:
        event: vLLM KV event 对象，或 relay dict。

    返回:
        包含 event_type、block_ids、block_hashes、target、token_count 的字典。
    """

    raw = _event_to_mapping(event)
    event_type = str(raw.get("type") or raw.get("event_type") or raw.get("name") or type(event).__name__)
    data = raw.get("data")
    if isinstance(data, dict):
        raw = {**raw, **data}
    return {
        "event_type": _normalize_event_type(event_type),
        "block_ids": _extract_int_list(raw, ("block_ids", "block_id", "physical_block_ids", "physical_block_id")),
        "block_hashes": [str(value) for value in _extract_list(raw, ("block_hashes", "block_hash", "hashes", "hash"))],
        "target": raw.get("to") or raw.get("destination") or raw.get("medium") or raw.get("location") or "",
        "token_count": _int_or_zero(raw.get("token_count") or raw.get("block_size")),
    }


def _event_to_mapping(event: Any) -> dict[str, Any]:
    """将事件对象转换为浅层 dict。

    参数:
        event: vLLM KV event 对象或 dict。

    返回:
        事件字段字典。
    """

    if isinstance(event, dict):
        return dict(event)
    if hasattr(event, "__struct_fields__"):
        return {key: getattr(event, key) for key in event.__struct_fields__}
    if hasattr(event, "__dict__"):
        return dict(event.__dict__)
    return {}


def _normalize_event_type(event_type: str) -> str:
    """标准化 KV event 类型名。

    参数:
        event_type: vLLM 原始事件名。

    返回:
        KVBM 数据面使用的事件类型名。
    """

    mapping = {
        "BlockStored": "store",
        "BlockRemoved": "removed",
        "BlockEvicted": "evict",
        "BlockCleared": "cleared",
        "BlockHit": "hit",
        "BlockMoved": "move",
        "BlockTransferred": "move",
    }
    return mapping.get(event_type, event_type).lower()


def _extract_list(raw: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    """从多个候选字段中提取列表值。

    参数:
        raw: 事件字段字典。
        keys: 候选字段名。

    返回:
        提取到的列表；字段不存在时返回空列表。
    """

    for key in keys:
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        if isinstance(value, list | tuple):
            return list(value)
        return [value]
    return []


def _extract_int_list(raw: dict[str, Any], keys: tuple[str, ...]) -> list[int]:
    """从事件字段中提取整数列表。

    参数:
        raw: 事件字段字典。
        keys: 候选字段名。

    返回:
        可解析为 int 的字段值列表。
    """

    values = []
    for value in _extract_list(raw, keys):
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return values


def _int_or_zero(value: Any) -> int:
    """将值转换为 int，失败时返回 0。

    参数:
        value: 任意值。

    返回:
        整数值或 0。
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _block_hash_for_index(block_hashes: list[str], index: int, block_id: int) -> str:
    """按下标获取 block hash，不存在时生成 synthetic hash。

    参数:
        block_hashes: 事件携带的 block hash 列表。
        index: 当前 block 下标。
        block_id: 物理 block id。

    返回:
        block hash 字符串。
    """

    if index < len(block_hashes):
        return block_hashes[index]
    return f"physical:{block_id}"


def _is_registered(manager: Any, block_id: int) -> bool:
    """判断 block 是否已注册。

    参数:
        manager: KVBMManager 实例。
        block_id: 物理 block id。

    返回:
        已注册返回 True，否则返回 False。
    """

    try:
        manager.get_metadata(block_id)
        return True
    except KeyError:
        return False
