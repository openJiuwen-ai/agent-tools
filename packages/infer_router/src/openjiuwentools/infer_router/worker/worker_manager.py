import asyncio
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException, status
from loguru import logger

from openjiuwentools.infer_router.config.config import settings
from openjiuwentools.infer_router.discovery import (
    ConfigDiscovery,
    EtcdDiscovery,
    WorkerDiscovery,
)
from openjiuwentools.infer_router.kv_cache.event import CacheEvent
from openjiuwentools.infer_router.kv_cache.kv_cache import KVCacheManager
from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType
from openjiuwentools.infer_router.worker.atomic_load_store import AtomicLoadStore


@dataclass
class WorkerStatus:
    """工作器状态"""

    worker_id: str
    last_health_check: float
    is_healthy: bool
    response_time: float
    consecutive_failures: int = 0  # 连续失败次数


def worker_endpoint_url(worker_url: str, endpoint: str) -> str:
    """Join a worker base URL and an API endpoint without duplicating ``/v1``.

    Worker configs document ``url`` as ``http://host:port/v1`` while forwarding
    passes endpoints such as ``/v1/chat/completions``. Plain concatenation
    produced ``/v1/v1/chat/completions``, which every worker answers with 404.
    Etcd registrations use the bare ``http://host:port`` form, so both shapes
    must keep working.
    """
    base = worker_url.rstrip("/")
    path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if base.endswith("/v1") and (path == "/v1" or path.startswith("/v1/")):
        path = path[len("/v1"):]
    return base + path


class WorkerManager:
    """工作器管理器，负责动态发现和健康检查"""

    def __init__(self, kv_cache_manager: KVCacheManager | None = None):
        self.workers: dict[str, WorkerInfo] = {}
        self.worker_statuses: dict[str, WorkerStatus] = {}
        self.groups: dict[str, list[str]] = {}
        self.discovery_task: asyncio.Task | None = None
        self.health_check_task: asyncio.Task | None = None
        self.discovery: WorkerDiscovery | None = None
        self.http_client: httpx.AsyncClient | None = None
        self._kvbm_sync_client: httpx.Client | None = None
        self._kvbm_sync_client_lock = threading.Lock()
        self._kvbm_capacity_flights_lock = threading.Lock()
        self._kvbm_capacity_flights: dict[tuple, dict] = {}
        self.kv_cache_manager = kv_cache_manager
        self._workload_manager = None
        self._load_store = AtomicLoadStore()  # 线程安全的负载存储
        self._on_worker_discovered_callbacks: list = []
        self._kvbm_datadist_status_cache: dict[str, tuple[float, dict]] = {}
        self._kvbm_link_locks: dict[tuple[str, str], asyncio.Lock] = {}

    def on_worker_discovered(self, callback) -> None:
        """注册 worker 发现回调，callback(worker: WorkerInfo)"""
        self._on_worker_discovered_callbacks.append(callback)

    def set_workload_manager(self, workload_manager):
        """设置 workload_manager 引用"""
        self._workload_manager = workload_manager

    @property
    def workload_manager(self):
        """获取 workload_manager 引用"""
        return self._workload_manager

    def register_event_handlers(self) -> None:
        """向 event_manager 注册事件处理器"""
        if not self.kv_cache_manager or not self.kv_cache_manager.event_manager:
            return

        event_manager = self.kv_cache_manager.event_manager

        event_manager.register_handler("prefill_start", self._on_prefill_start_event)
        event_manager.register_handler("prefill_end", self._on_prefill_end_event)
        event_manager.register_handler("decode_start", self._on_decode_start_event)
        event_manager.register_handler("decode_end", self._on_decode_end_event)

        logger.info("Registered workload event handlers to event_manager")

    def _on_prefill_start_event(self, event: CacheEvent) -> None:
        """处理 prefill_start 事件"""
        if not self._workload_manager:
            return
        self._workload_manager.process_cache_event(event)
        self._update_worker_load(event.worker_id)

    def _on_prefill_end_event(self, event: CacheEvent) -> None:
        """处理 prefill_end 事件"""
        if not self._workload_manager:
            return
        self._workload_manager.process_cache_event(event)
        self._update_worker_load(event.worker_id)

    def _on_decode_start_event(self, event: CacheEvent) -> None:
        """处理 decode_start 事件"""
        if not self._workload_manager:
            return
        self._workload_manager.process_cache_event(event)
        self._update_worker_load(event.worker_id)

    def _on_decode_end_event(self, event: CacheEvent) -> None:
        """处理 decode_end 事件"""
        if not self._workload_manager:
            return
        self._workload_manager.process_cache_event(event)
        self._update_worker_load(event.worker_id)

    def _update_worker_load(self, worker_id: str) -> None:
        """更新指定工作器的负载（线程安全）"""
        if not self._workload_manager:
            return
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            new_load = self._workload_manager.calculate_load(worker)
            self._load_store.set_load(worker_id, new_load)
            # 同时更新 WorkerInfo 中的 current_load（保持向后兼容）
            worker.current_load = new_load

    def get_worker_load(self, worker_id: str) -> float:
        """获取工作器负载（线程安全）"""
        # 优先从线程安全存储获取
        load = self._load_store.get_load(worker_id)
        if load > 0:
            return load
        # 降级到 WorkerInfo（兼容旧代码和初始化状态）
        if worker_id in self.workers:
            return self.workers[worker_id].current_load
        return 0.0

    def add_to_group(self, worker: WorkerInfo):
        """将工作器添加到对应的group"""
        group = worker.group
        if group not in self.groups:
            self.groups[group] = []
        if worker.worker_id not in self.groups[group]:
            self.groups[group].append(worker.worker_id)

    def remove_from_group(self, worker: WorkerInfo):
        """将工作器从group中移除"""
        group = worker.group
        if group in self.groups and worker.worker_id in self.groups[group]:
            self.groups[group].remove(worker.worker_id)
            if not self.groups[group]:
                del self.groups[group]

    def get_groups(self) -> list[str]:
        """获取所有group列表"""
        return list(self.groups.keys())

    def get_workers_in_group(self, group: str) -> list[WorkerInfo]:
        """获取指定group内的所有工作器"""
        if group not in self.groups:
            return []
        return [self.workers[worker_id] for worker_id in self.groups[group] if worker_id in self.workers]

    @staticmethod
    def _is_prefill_capable(worker: WorkerInfo, model: str) -> bool:
        """判断工作器是否为指定模型的 prefill 节点"""
        return worker.model == model and worker.worker_type == WorkerType.PREFILL

    @staticmethod
    def _is_decode_capable(worker: WorkerInfo, model: str) -> bool:
        """判断工作器是否为指定模型的 decode 节点"""
        return worker.model == model and worker.worker_type == WorkerType.DECODE

    def get_prefill_workers_in_group(self, group: str, model: str) -> list[WorkerInfo]:
        """获取指定group内的prefill工作器（包括combined类型）"""
        workers = self.get_workers_in_group(group)
        return [w for w in workers if self._is_prefill_capable(w, model)]

    def get_decode_workers_in_group(self, group: str, model: str) -> list[WorkerInfo]:
        """获取指定group内的decode工作器（包括combined类型）"""
        workers = self.get_workers_in_group(group)
        return [w for w in workers if self._is_decode_capable(w, model)]

    def get_healthy_groups(self, model: str) -> list[str]:
        """获取包含指定模型健康工作器的group列表"""
        healthy_groups = []
        all_groups = []

        for group in self.groups:
            workers = self.get_workers_in_group(group)
            has_matching_model = any(worker.model == model for worker in workers)
            if has_matching_model:
                all_groups.append(group)
                for worker in workers:
                    worker_status = self.worker_statuses.get(worker.worker_id)
                    if worker_status and worker_status.is_healthy and worker.model == model:
                        healthy_groups.append(group)
                        break

        # 如果没有找到健康的 group，返回所有包含该模型的 group（降级模式）
        if not healthy_groups and all_groups:
            logger.warning(f"No healthy groups found for model {model}, using all groups in degraded mode")
            return all_groups

        return healthy_groups

    def get_group_health_status(self, group: str, model: str) -> bool:
        """检查group是否有健康的工作器"""
        workers = self.get_workers_in_group(group)
        has_matching_model = any(worker.model == model for worker in workers)
        if not has_matching_model:
            return False

        for worker in workers:
            if worker.model == model:
                worker_status = self.worker_statuses.get(worker.worker_id)
                if worker_status and worker_status.is_healthy:
                    return True

        # 如果没有健康的 worker，但 group 中有匹配模型的 worker，返回 True（降级模式）
        return True

    def is_combined_group(self, group: str) -> bool:
        """判断group是否是组合型group（全部由组合型节点组成）"""
        workers = self.get_workers_in_group(group)
        if not workers:
            return False
        return all(worker.worker_type == WorkerType.COMBINED for worker in workers)

    def get_group_model(self, group: str) -> str | None:
        """获取group内工作器的模型类型（同一个group内模型类型相同）"""
        workers = self.get_workers_in_group(group)
        if not workers:
            return None
        return workers[0].model

    def validate_group_consistency(self, worker: WorkerInfo) -> bool:
        """验证工作器加入group是否符合规则：
        1. 组合型节点所在group只能包含组合型节点
        2. 同一个group内的节点模型类型必须相同

        Returns:
            bool: 是否符合规则
        """
        workers_in_group = self.get_workers_in_group(worker.group)

        if not workers_in_group:
            return True

        group_model = workers_in_group[0].model
        if worker.model != group_model:
            logger.warning(
                "Worker {} model {} does not match group {} model {}",
                worker.worker_id,
                worker.model,
                worker.group,
                group_model,
            )
            return False

        has_combined = any(w.worker_type == WorkerType.COMBINED for w in workers_in_group)

        if has_combined and worker.worker_type != WorkerType.COMBINED:
            logger.warning(
                "Worker {} type {} cannot join group {} which contains combined workers",
                worker.worker_id,
                worker.worker_type,
                worker.group,
            )
            return False

        if not has_combined and worker.worker_type == WorkerType.COMBINED:
            logger.warning(
                "Worker {} type COMBINED cannot join group {} which contains non-combined workers",
                worker.worker_id,
                worker.group,
            )
            return False

        return True

    def create_discovery(self) -> WorkerDiscovery:
        """根据配置创建工作器发现器

        Returns:
            WorkerDiscovery: 工作器发现器实例

        """
        discovery_type = settings.worker_discovery_type.lower()

        if discovery_type == "config":
            logger.info(f"Using config-based worker discovery with file: {settings.worker_config_path}")
            return ConfigDiscovery(config_path=settings.worker_config_path)
        elif discovery_type == "etcd":
            logger.info(
                f"Using etcd-based worker discovery with hosts: {settings.etcd_hosts}, prefix: {settings.etcd_prefix}"
            )
            discovery = EtcdDiscovery(
                etcd_hosts=settings.etcd_hosts,
                etcd_port=settings.etcd_port,
                etcd_prefix=settings.etcd_prefix,
                etcd_user=settings.etcd_user,
                etcd_password=settings.etcd_password,
                enable_watch=settings.etcd_enable_watch,
            )

            # 如果启用了watch机制，设置回调函数
            if settings.etcd_enable_watch:
                discovery.set_on_change_callback(self._on_worker_change)

            return discovery
        else:
            logger.warning(f"Unknown discovery type: {discovery_type}, falling back to config-based discovery")
            return ConfigDiscovery(config_path=settings.worker_config_path)

    async def _on_worker_change(self, event_type: str, data):
        """处理工作器变化事件

        Args:
            event_type: 事件类型（put或delete）
            data: 工作器信息或worker_id

        """
        try:
            if event_type == "put":
                # 新增或更新工作器
                worker = data
                if worker.worker_id not in self.workers:
                    logger.info(f"Worker {worker.worker_id} added via watch")
                    self.workers[worker.worker_id] = worker
                    self.worker_statuses[worker.worker_id] = WorkerStatus(
                        worker_id=worker.worker_id,
                        last_health_check=0,
                        is_healthy=True,
                        response_time=0,
                        consecutive_failures=0,
                    )

                    # 注册到 KV Cache 管理器
                    if self.kv_cache_manager:
                        self.kv_cache_manager.register_worker(
                            worker_id=worker.worker_id,
                            engine_type=worker.engine_type,
                            model=worker.model,
                        )

                    # 添加到group
                    self.add_to_group(worker)
                else:
                    logger.info(f"Worker {worker.worker_id} updated via watch")
                    # 更新前先从旧的group移除
                    old_worker = self.workers[worker.worker_id]
                    if old_worker.group != worker.group:
                        self.remove_from_group(old_worker)
                    self.workers[worker.worker_id] = worker
                    # 添加到新的group
                    self.add_to_group(worker)

            elif event_type == "delete":
                # 删除工作器
                worker_id = data
                if worker_id in self.workers:
                    logger.info(f"Worker {worker_id} removed via watch")
                    worker = self.workers[worker_id]
                    # 从group移除
                    self.remove_from_group(worker)
                    del self.workers[worker_id]
                    if worker_id in self.worker_statuses:
                        del self.worker_statuses[worker_id]

                    # 从 KV Cache 管理器注销
                    if self.kv_cache_manager:
                        self.kv_cache_manager.unregister_worker(worker_id)

        except Exception as e:
            logger.error(f"Error handling worker change event: {e}")

    async def start(self):
        """启动工作器管理服务"""
        logger.info("Starting worker manager...")

        # 创建HTTP客户端，使用优化的连接池配置
        limits = httpx.Limits(
            max_connections=settings.http_pool_connections,
            max_keepalive_connections=settings.http_pool_max_keepalive,
            keepalive_expiry=settings.http_keepalive_expiry,
        )

        # 创建单独的超时配置：健康检查有较长的超时，而请求转发有更长的超时
        health_timeout = httpx.Timeout(
            timeout=settings.worker_health_check_timeout,
            connect=5.0,
        )

        self.http_client = httpx.AsyncClient(
            timeout=health_timeout,
            limits=limits,
            http2=True,  # 启用HTTP/2支持
            verify=False,  # 跳过SSL验证，适用于内部网络
        )

        # 创建工作器发现器
        self.discovery = self.create_discovery()
        await self.discovery.start()

        # 如果没有启用watch机制，启动定期发现任务
        if not (settings.worker_discovery_type == "etcd" and settings.etcd_enable_watch):
            self.discovery_task = asyncio.create_task(self._discovery_loop())

        # 启动健康检查任务
        self.health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Worker manager started")

    async def stop(self):
        """停止工作器管理服务"""
        logger.info("Stopping worker manager...")

        if self.discovery_task:
            self.discovery_task.cancel()

        if self.health_check_task:
            self.health_check_task.cancel()

        if self.discovery:
            await self.discovery.stop()

        if self.http_client:
            await self.http_client.aclose()
        with self._kvbm_sync_client_lock:
            if self._kvbm_sync_client is not None:
                self._kvbm_sync_client.close()
                self._kvbm_sync_client = None

        logger.info("Worker manager stopped")

    async def _discovery_loop(self):
        """工作器发现循环"""
        while True:
            try:
                await self.discover_workers()
            except Exception as e:
                logger.error(f"Worker discovery failed: {e}")

            await asyncio.sleep(settings.worker_discovery_interval)

    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await self.check_all_workers_health()
                self._check_worker_timeouts()
            except Exception as e:
                logger.error(f"Worker health check failed: {e}")

            await asyncio.sleep(settings.worker_health_check_interval)

    def _check_worker_timeouts(self):
        """检查工作器是否超时

        如果工作器超时，清空其缓存并生成 Cleared 事件

        """
        current_time = time.time()
        for worker_id, worker_status in self.worker_statuses.items():
            if (
                worker_status.last_health_check
                and current_time - worker_status.last_health_check > settings.worker_timeout
            ):
                logger.warning(f"Worker {worker_id} is timeout, clearing cache")
                # 清空工作器的事件
                if self.kv_cache_manager:
                    self.kv_cache_manager.clear_worker_events(worker_id)

    async def discover_workers(self):
        """发现工作器"""
        if not self.discovery:
            logger.warning("Discovery mechanism not initialized")
            return

        logger.info("Discovering workers...")

        # 使用发现器获取工作器列表
        discovered_workers = await self.discovery.discover()

        # 更新工作器列表
        for worker in discovered_workers:
            if worker.worker_id not in self.workers:
                logger.info(f"Discovered new worker: {worker.worker_id}")
                self.workers[worker.worker_id] = worker
                self.worker_statuses[worker.worker_id] = WorkerStatus(
                    worker_id=worker.worker_id,
                    last_health_check=0,
                    is_healthy=True,
                    response_time=0,
                    consecutive_failures=0,
                )

                # 注册到 KV Cache 管理器
                if self.kv_cache_manager:
                    self.kv_cache_manager.register_worker(
                        worker_id=worker.worker_id,
                        engine_type=worker.engine_type,
                        model=worker.model,
                    )

                # 添加到group
                self.add_to_group(worker)

                # 通知回调
                for cb in self._on_worker_discovered_callbacks:
                    try:
                        cb(worker)
                    except Exception as e:
                        logger.error(f"Worker discovered callback error: {e}")
            else:
                # 更新现有工作器信息
                old_worker = self.workers[worker.worker_id]
                if old_worker.group != worker.group:
                    self.remove_from_group(old_worker)
                self.workers[worker.worker_id] = worker
                # 添加到新的group
                self.add_to_group(worker)

        # 移除不再存在的工作器
        discovered_ids = {worker.worker_id for worker in discovered_workers}
        removed_ids = set(self.workers.keys()) - discovered_ids
        for worker_id in removed_ids:
            logger.info(f"Worker {worker_id} no longer exists, removing")
            worker = self.workers[worker_id]
            # 从group移除
            self.remove_from_group(worker)
            del self.workers[worker_id]
            if worker_id in self.worker_statuses:
                del self.worker_statuses[worker_id]

            # 从 KV Cache 管理器注销
            if self.kv_cache_manager:
                self.kv_cache_manager.unregister_worker(worker_id)

        logger.info(f"Worker discovery completed. Total workers: {len(self.workers)}")

    async def check_all_workers_health(self):
        """检查所有工作器的健康状态"""
        logger.debug("Checking health of all workers...")

        tasks = []
        for worker_id in list(self.workers.keys()):
            tasks.append(self.check_worker_health(worker_id))

        await asyncio.gather(*tasks)

    async def check_worker_health(self, worker_id: str):
        """检查单个工作器的健康状态"""
        start_time = time.time()

        try:
            worker = self.workers.get(worker_id)
            if not worker:
                logger.warning(f"Worker {worker_id} not found")
                return

            # 获取旧的状态
            old_status = self.worker_statuses.get(worker_id)
            old_consecutive_failures = old_status.consecutive_failures if old_status else 0

            base_url = worker.url.rstrip("/").removesuffix("/v1")
            health_url = f"{base_url}/health"

            headers = {}
            if worker.api_key:
                headers["Authorization"] = f"Bearer {worker.api_key}"

            try:
                response = await self.http_client.get(health_url, headers=headers)
                check_passed = response.status_code == 200

                if check_passed:
                    logger.debug(f"Worker {worker_id} health check passed")
                    new_consecutive_failures = 0
                    is_healthy = True
                else:
                    logger.warning(
                        "Worker {} health check failed: url={}, status={}, body={}",
                        worker_id,
                        health_url,
                        response.status_code,
                        response.text[:500],
                    )
                    new_consecutive_failures = old_consecutive_failures + 1
                    is_healthy = new_consecutive_failures < settings.worker_health_check_max_failures

            except httpx.HTTPStatusError as e:
                logger.error(
                    "Worker {} health check HTTP error: url={}, status={}, body={}",
                    worker_id,
                    health_url,
                    e.response.status_code,
                    e.response.text[:500],
                )
                new_consecutive_failures = old_consecutive_failures + 1
                is_healthy = new_consecutive_failures < settings.worker_health_check_max_failures
            except httpx.RequestError as e:
                logger.warning(
                    "Worker {} health check connection error: url={}, type={}, detail={}",
                    worker_id,
                    health_url,
                    type(e).__name__,
                    e,
                )
                new_consecutive_failures = old_consecutive_failures + 1
                is_healthy = new_consecutive_failures < settings.worker_health_check_max_failures

            response_time = time.time() - start_time

            # 如果健康状态改变了，记录日志
            if old_status and old_status.is_healthy != is_healthy:
                if is_healthy:
                    logger.info(f"Worker {worker_id} is now healthy")
                else:
                    logger.warning(
                        f"Worker {worker_id} marked as unhealthy after {new_consecutive_failures} consecutive failures"
                    )

            self.worker_statuses[worker_id] = WorkerStatus(
                worker_id=worker_id,
                last_health_check=time.time(),
                is_healthy=is_healthy,
                response_time=response_time,
                consecutive_failures=new_consecutive_failures,
            )

        except Exception as e:
            logger.error(f"Health check failed for worker {worker_id}: {e}")

            old_status = self.worker_statuses.get(worker_id)
            old_consecutive_failures = old_status.consecutive_failures if old_status else 0
            new_consecutive_failures = old_consecutive_failures + 1
            is_healthy = new_consecutive_failures < settings.worker_health_check_max_failures

            # 更新工作器状态
            self.worker_statuses[worker_id] = WorkerStatus(
                worker_id=worker_id,
                last_health_check=time.time(),
                is_healthy=is_healthy,
                response_time=time.time() - start_time,
                consecutive_failures=new_consecutive_failures,
            )

    def get_all_healthy_prefill_workers(self, model: str) -> list[WorkerInfo]:
        """获取所有健康的 prefill 可用工作器（PREFILL 或 COMBINED），跨 group"""
        return [w for w in self.get_healthy_workers(model) if self._is_prefill_capable(w, model)]

    def get_all_healthy_decode_workers(self, model: str) -> list[WorkerInfo]:
        """获取所有健康的 decode 可用工作器（DECODE 或 COMBINED），跨 group"""
        return [w for w in self.get_healthy_workers(model) if self._is_decode_capable(w, model)]

    def get_all_healthy_combined_workers(self, model: str) -> list[WorkerInfo]:
        """获取所有健康的 COMBINED 工作器，跨 group"""
        return [w for w in self.get_healthy_workers(model) if w.worker_type == WorkerType.COMBINED]

    def has_disaggregated_workers(self, model: str) -> bool:
        """判断指定模型是否配置了 PD 分离工作器。

        Args:
            model: 待检查的模型名称。

        Returns:
            bool: 存在 PREFILL 或 DECODE 类型工作器时返回 True，否则返回 False。
        """
        return any(
            worker.model == model and worker.worker_type in (WorkerType.PREFILL, WorkerType.DECODE)
            for worker in self.workers.values()
        )

    def get_healthy_workers(self, model: str) -> list[WorkerInfo]:
        """获取指定模型的健康工作器列表"""
        healthy_workers = []
        all_workers = []

        for worker_id, worker in self.workers.items():
            if worker.model == model:
                all_workers.append(worker)
                worker_status = self.worker_statuses.get(worker_id)
                if worker_status and worker_status.is_healthy:
                    healthy_workers.append(worker)

        # 如果没有找到健康的工作器，返回所有工作器（降级模式）
        if not healthy_workers and all_workers:
            logger.warning(f"No healthy workers found for model {model}, using all workers in degraded mode")
            return all_workers

        return healthy_workers

    def get_worker(self, worker_id: str) -> WorkerInfo | None:
        """获取指定ID的工作器"""
        return self.workers.get(worker_id)

    def get_worker_status(self, worker_id: str) -> WorkerStatus | None:
        """获取指定ID的工作器状态"""
        return self.worker_statuses.get(worker_id)

    def get_all_workers(self) -> dict[str, WorkerInfo]:
        """获取所有工作器"""
        return self.workers.copy()

    def get_all_worker_statuses(self) -> dict[str, WorkerStatus]:
        """获取所有工作器状态"""
        return self.worker_statuses.copy()

    async def prefetch_kv_blocks(
        self,
        worker_id: str,
        block_hashes: list[str],
        *,
        request_id: str = "",
        deadline_ms: int = 100,
    ) -> dict:
        """向目标 worker 发起一次 best-effort KVBM 预取请求。

        参数:
            worker_id: 需要预取的目标 worker ID。
            block_hashes: router 判断本次请求会命中的 block hash 列表。
            request_id: 当前请求 ID，用于 worker 侧日志串联。
            deadline_ms: 从 Router 发起时起算的预取有效预算，单位毫秒；Worker
                超时后必须回滚，HTTP 客户端也据此设置控制面超时。

        返回:
            worker 返回的预取结果字典；失败或未启用时返回 ok=False 的字典。
        """

        if not block_hashes:
            return {"ok": True, "prefetched_count": 0, "requested_count": 0}

        worker = self.workers.get(worker_id)
        if not worker:
            return {"ok": False, "error": f"worker {worker_id} not found"}
        if self.http_client is None:
            return {"ok": False, "error": "worker http client is not initialized"}

        base_url = worker.url.rstrip("/").removesuffix("/v1")
        url = f"{base_url}/kvbm/prefetch"
        headers = {"Content-Type": "application/json"}
        if worker.api_key:
            headers["Authorization"] = f"Bearer {worker.api_key}"
        payload = {
            "request_id": request_id,
            "block_hashes": block_hashes,
            "deadline_ms": deadline_ms,
            "deadline_epoch_ms": int(time.time() * 1000) + max(deadline_ms, 0),
        }
        timeout = httpx.Timeout((max(deadline_ms, 1) + 100) / 1000.0, connect=0.1)
        try:
            response = await self.http_client.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.debug(
                "KVBM prefetch skipped for worker {}: url={}, blocks={}, error={}",
                worker_id,
                url,
                len(block_hashes),
                exc,
            )
            return {"ok": False, "error": str(exc), "requested_count": len(block_hashes)}

    @staticmethod
    def worker_control_url(worker: WorkerInfo) -> str:
        """Return the normalized control endpoint for request orchestration."""
        return WorkerManager._worker_control_url(worker)

    @staticmethod
    def _worker_control_url(worker: WorkerInfo) -> str:
        """返回 Worker 的 KVBM 控制服务根地址。

        参数:
            worker: Router 已发现的 Worker 元数据。

        返回:
            显式配置存在时返回规范化后的地址；配置为空时按 Worker 推理端口
            加 10000 推导默认 engine-core KVBM 控制地址。URL 缺少合法端口时
            保留旧的推理地址回退行为。
        """

        configured = worker.kvbm_control_url.strip()
        if configured:
            return configured.rstrip("/").removesuffix("/v1")
        inference_url = worker.url.rstrip("/").removesuffix("/v1")
        parsed = urlsplit(inference_url)
        try:
            inference_port = parsed.port
        except ValueError:
            inference_port = None
        if inference_port is None or inference_port > 55_535 or not parsed.hostname:
            return inference_url
        hostname = parsed.hostname
        control_host = f"[{hostname}]" if ":" in hostname else hostname
        control_netloc = f"{control_host}:{inference_port + 10_000}"
        return urlunsplit((parsed.scheme, control_netloc, "", "", ""))

    def confirm_kvbm_block_pool_capacity(
        self,
        worker_id: str,
        *,
        timeout_ms: int,
        block_hashes: list[str] | None = None,
        prompt_blocks: int = 0,
        decode_blocks: int = 0,
        margin_blocks: int = 0,
        pending_reserved_blocks: int = 0,
        request_id: str = "",
        handoff_id: str = "",
        ticket_ttl_ms: int = 120_000,
    ) -> dict:
        """在路由派发前同步读取D Worker的最新BlockPool容量。

        参数:
            worker_id: 待确认容量的D Worker ID。
            timeout_ms: 控制面HTTP请求超时，单位毫秒。
            block_hashes: Router快照中D可能复用的连续NPU前缀hash。
            prompt_blocks: 当前请求完整Prompt block数。
            decode_blocks: 当前请求预计输出block数。

        返回:
            成功时返回包含``ok=True``、``free_blocks``和``block_size``
            的字典，并同步更新Worker容量快照；网络、协议或
            Worker状态异常时返回``ok=False``和``error``。
        """

        started = time.perf_counter()
        normalized_hashes = tuple(str(value) for value in (block_hashes or []))
        flight_key = (
            worker_id,
            normalized_hashes,
            max(int(prompt_blocks), 0),
            max(int(decode_blocks), 0),
            max(int(margin_blocks), 0),
            max(int(pending_reserved_blocks), 0),
            str(handoff_id),
        )
        with self._kvbm_capacity_flights_lock:
            flight = self._kvbm_capacity_flights.get(flight_key)
            if flight is None:
                flight = {"event": threading.Event(), "result": None}
                self._kvbm_capacity_flights[flight_key] = flight
                leader = True
            else:
                leader = False

        if not leader:
            timeout_seconds = max(int(timeout_ms), 1) / 1000.0
            if not flight["event"].wait(timeout=timeout_seconds):
                return {
                    "ok": False,
                    "error": "timed out waiting for shared D Worker capacity query",
                    "error_type": "SharedQueryWaitTimeout",
                    "failure_stage": "singleflight_wait",
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "singleflight_shared": True,
                }
            result = dict(flight["result"] or {})
            result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
            result["singleflight_shared"] = True
            return result

        try:
            fetch_kwargs = {"timeout_ms": timeout_ms}
            if request_id:
                fetch_kwargs["request_id"] = request_id
            if normalized_hashes:
                fetch_kwargs.update(
                    block_hashes=list(normalized_hashes),
                    prompt_blocks=prompt_blocks,
                    decode_blocks=decode_blocks,
                    margin_blocks=margin_blocks,
                    pending_reserved_blocks=pending_reserved_blocks,
                    request_id=request_id,
                    handoff_id=handoff_id,
                    ticket_ttl_ms=ticket_ttl_ms,
                )
            result = self._fetch_kvbm_block_pool_capacity(
                worker_id,
                **fetch_kwargs,
            )
            result["singleflight_shared"] = False
            return result
        finally:
            with self._kvbm_capacity_flights_lock:
                flight["result"] = locals().get(
                    "result",
                    {
                        "ok": False,
                        "error": "D Worker capacity query terminated unexpectedly",
                    },
                )
                flight["event"].set()
                self._kvbm_capacity_flights.pop(flight_key, None)

    def _fetch_kvbm_block_pool_capacity(
        self,
        worker_id: str,
        *,
        timeout_ms: int,
        block_hashes: list[str] | None = None,
        prompt_blocks: int = 0,
        decode_blocks: int = 0,
        margin_blocks: int = 0,
        pending_reserved_blocks: int = 0,
        request_id: str = "",
        handoff_id: str = "",
        ticket_ttl_ms: int = 120_000,
    ) -> dict:
        """执行一次真实的D Worker BlockPool容量查询。

        参数:
            worker_id: 待查询的D Worker ID。
            timeout_ms: 控制面HTTP请求超时，单位毫秒。

        返回:
            容量查询结果。该函数只由singleflight主请求调用。
        """

        worker = self.workers.get(worker_id)
        if worker is None:
            return {"ok": False, "error": f"worker {worker_id} not found"}
        timeout_seconds = max(int(timeout_ms), 1) / 1000.0
        with self._kvbm_sync_client_lock:
            if self._kvbm_sync_client is None:
                self._kvbm_sync_client = httpx.Client(
                    limits=httpx.Limits(
                        max_connections=32,
                        max_keepalive_connections=16,
                        keepalive_expiry=settings.http_keepalive_expiry,
                    ),
                    verify=False,
                )
            client = self._kvbm_sync_client
        url = f"{self._worker_control_url(worker)}/datadist/block-pool/capacity"
        headers = {"Content-Type": "application/json"}
        if worker.api_key:
            headers["Authorization"] = f"Bearer {worker.api_key}"
        started = time.perf_counter()
        failure_stage = "http_request"
        try:
            request_payload = {
                "block_hashes": list(block_hashes or []),
                "prompt_blocks": max(int(prompt_blocks), 0),
                "decode_blocks": max(int(decode_blocks), 0),
                "margin_blocks": max(int(margin_blocks), 0),
                "pending_reserved_blocks": max(int(pending_reserved_blocks), 0),
            }
            if request_id:
                request_payload["request_id"] = str(request_id)
            if handoff_id and request_id:
                request_payload.update(
                    {
                        "request_id": str(request_id),
                        "handoff_id": str(handoff_id),
                        "ticket_ttl_ms": max(int(ticket_ttl_ms), 1_000),
                    }
                )
            response = client.post(
                url,
                json=request_payload,
                headers=headers,
                timeout=httpx.Timeout(
                    timeout_seconds,
                    connect=min(timeout_seconds, 0.1),
                ),
            )
            response.raise_for_status()
            http_ms = (time.perf_counter() - started) * 1000.0
            failure_stage = "response_decode"
            payload = response.json()
            block_pool = payload.get("block_pool") or {}
            free_blocks = block_pool.get("free_blocks")
            if not payload.get("ok") or not block_pool.get("ready"):
                return {
                    "ok": False,
                    "error": "D Worker BlockPool is not ready",
                    "url": url,
                }
            if free_blocks is None:
                return {
                    "ok": False,
                    "error": "D Worker did not return free_blocks",
                    "url": url,
                }
            free_blocks = max(int(free_blocks), 0)
            raw_generation = block_pool.get("capacity_generation")
            capacity_generation = int(raw_generation) if raw_generation is not None else -1
            current_generation = int(getattr(worker, "kvbm_capacity_generation", -1))
            if capacity_generation >= 0 and current_generation >= 0 and capacity_generation < current_generation:
                return {
                    "ok": False,
                    "error": (
                        "stale D Worker capacity generation: "
                        f"sample={capacity_generation}, current={current_generation}"
                    ),
                    "free_blocks": free_blocks,
                    "capacity_generation": capacity_generation,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "url": url,
                }
            worker.kvbm_free_blocks = free_blocks
            worker.kvbm_block_pool_updated_at = time.monotonic()
            if capacity_generation >= 0:
                worker.kvbm_capacity_generation = capacity_generation
            block_size = block_pool.get("block_size")
            if block_size is not None and int(block_size) > 0:
                worker.kvbm_block_size = int(block_size)
            return {
                "ok": True,
                "free_blocks": free_blocks,
                "capacity_generation": capacity_generation,
                "block_size": int(block_size) if block_size is not None else None,
                "confirmed_reusable_blocks": int(block_pool.get("confirmed_reusable_blocks", 0) or 0),
                "required_new_blocks": int(block_pool.get("required_new_blocks", 0) or 0),
                "outstanding_ticket_reserved_blocks": int(block_pool.get("outstanding_ticket_reserved_blocks", 0) or 0),
                "router_pending_reserved_blocks": int(block_pool.get("router_pending_reserved_blocks", 0) or 0),
                "committed_reserved_blocks": int(block_pool.get("committed_reserved_blocks", 0) or 0),
                "ticket_capacity_admitted": bool(block_pool.get("ticket_capacity_admitted", False)),
                "admission_ticket_id": str(block_pool.get("admission_ticket_id", "") or ""),
                "admission_ticket_pinned_blocks": int(block_pool.get("admission_ticket_pinned_blocks", 0) or 0),
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "http_ms": http_ms,
                "control_timing": payload.get("control_timing", {}),
                "url": url,
            }
        except Exception as exc:
            error_body = exc.response.text[:1024] if isinstance(exc, httpx.HTTPStatusError) else ""
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}" + (f" body={error_body}" if error_body else ""),
                "error_type": type(exc).__name__,
                "failure_stage": failure_stage,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "url": url,
            }

    def release_kvbm_decode_admission_ticket(
        self,
        worker_id: str,
        *,
        handoff_id: str,
        ticket_id: str = "",
        timeout_ms: int = 200,
    ) -> dict:
        """释放未被最终路由采用的D前缀准入ticket。"""

        worker = self.workers.get(worker_id)
        if worker is None:
            return {"ok": False, "error": f"worker {worker_id} not found"}
        timeout_seconds = max(int(timeout_ms), 1) / 1000.0
        with self._kvbm_sync_client_lock:
            if self._kvbm_sync_client is None:
                self._kvbm_sync_client = httpx.Client(verify=False)
            client = self._kvbm_sync_client
        url = f"{self._worker_control_url(worker)}/datadist/block-pool/admission-ticket/release"
        try:
            response = client.post(
                url,
                json={"handoff_id": handoff_id, "ticket_id": ticket_id},
                timeout=httpx.Timeout(
                    timeout_seconds,
                    connect=min(timeout_seconds, 0.1),
                ),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": url}

    async def _post_kvbm_control(
        self,
        worker: WorkerInfo,
        path: str,
        payload: dict,
        *,
        timeout_ms: int,
    ) -> dict:
        """调用 Worker KVBM 控制接口并返回 JSON。

        参数:
            worker: 接收控制请求的 Worker。
            path: 以 `/` 开头的控制接口路径。
            payload: JSON 请求体。
            timeout_ms: HTTP 总超时，单位毫秒。

        返回:
            Worker 返回的 JSON 字典；网络或协议错误时返回 ok=False。
        """

        if self.http_client is None:
            return {"ok": False, "error": "worker http client is not initialized"}
        headers = {"Content-Type": "application/json"}
        if worker.api_key:
            headers["Authorization"] = f"Bearer {worker.api_key}"
        url = f"{self._worker_control_url(worker)}{path}"
        try:
            response = await self.http_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(
                    max(timeout_ms, 1) / 1000.0,
                    connect=min(max(timeout_ms, 1) / 1000.0, 0.5),
                ),
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                return {"ok": False, "error": "worker returned non-object JSON"}
            return result
        except httpx.HTTPStatusError as exc:
            response_payload: object | None = None
            try:
                response_payload = exc.response.json()
            except Exception:
                response_payload = exc.response.text
            return {
                "ok": False,
                "error": str(exc),
                "url": url,
                "status_code": exc.response.status_code,
                "response": response_payload,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": url}

    async def get_datadist_status(
        self,
        worker_id: str,
        *,
        cache_ttl_seconds: float = 10.0,
        force_refresh: bool = False,
        timeout_ms: int = 1000,
    ) -> dict:
        """读取并短期缓存 Worker 的 LLM-DataDist endpoint 状态。

        参数:
            worker_id: 待读取的 Worker ID。
            cache_ttl_seconds: 成功状态的缓存时间，单位秒。
            force_refresh: 是否忽略缓存并立即访问 Worker。
            timeout_ms: 访问 Worker 状态接口的超时时间，单位毫秒。

        返回:
            `/datadist/status` 的响应；Worker 不存在或请求失败时 ok=False。
        """

        now = time.monotonic()
        cached = self._kvbm_datadist_status_cache.get(worker_id)
        if not force_refresh and cached is not None and now - cached[0] <= max(cache_ttl_seconds, 0.0):
            return cached[1]
        worker = self.workers.get(worker_id)
        if worker is None:
            return {"ok": False, "error": f"worker {worker_id} not found"}
        result = await self._post_kvbm_control(
            worker,
            "/datadist/status",
            {},
            timeout_ms=timeout_ms,
        )
        if result.get("ok"):
            self._kvbm_datadist_status_cache[worker_id] = (now, result)
        return result

    async def ensure_datadist_link(
        self,
        target_worker_id: str,
        source_worker_id: str,
        *,
        timeout_ms: int = 3000,
        status_ttl_seconds: float = 10.0,
    ) -> dict:
        """确保目标 Worker 已建立到源 Worker 的单向拉取链路。

        参数:
            target_worker_id: 执行 pull 的目标 Worker ID。
            source_worker_id: 提供 KV block 的源 Worker ID。
            timeout_ms: LLM-DataDist 建链超时，单位毫秒。
            status_ttl_seconds: endpoint 状态缓存时间，单位秒。

        返回:
            已建链或新建链成功时 ok=True，否则返回错误诊断。
        """

        total_started = time.perf_counter()
        phase_ms: dict[str, float] = {}

        def finish(result: dict) -> dict:
            """补齐本次状态检查和建链的阶段耗时。"""

            phase_ms["link_total"] = (time.perf_counter() - total_started) * 1000.0
            return {**result, "phase_ms": dict(phase_ms)}

        target = self.workers.get(target_worker_id)
        source = self.workers.get(source_worker_id)
        if target is None or source is None:
            return finish({"ok": False, "error": "source or target worker not found"})
        key = (target_worker_id, source_worker_id)
        lock = self._kvbm_link_locks.setdefault(key, asyncio.Lock())
        async with lock:
            status_started = time.perf_counter()
            target_status, source_status = await asyncio.gather(
                self.get_datadist_status(
                    target_worker_id,
                    cache_ttl_seconds=status_ttl_seconds,
                    timeout_ms=timeout_ms,
                ),
                self.get_datadist_status(
                    source_worker_id,
                    cache_ttl_seconds=status_ttl_seconds,
                    timeout_ms=timeout_ms,
                ),
            )
            phase_ms["status"] = (time.perf_counter() - status_started) * 1000.0
            if not target_status.get("ok") or not source_status.get("ok"):
                return finish(
                    {
                        "ok": False,
                        "error": "source or target LLM-DataDist backend is not ready",
                        "target_worker_id": target_worker_id,
                        "target_control_url": self._worker_control_url(target),
                        "target_status": target_status,
                        "source_worker_id": source_worker_id,
                        "source_control_url": self._worker_control_url(source),
                        "source_status": source_status,
                    }
                )
            target_health = target_status.get("health") or {}
            source_health = source_status.get("health") or {}
            target_endpoint = target_health.get("endpoint") or {}
            source_endpoint = source_health.get("endpoint") or {}
            try:
                source_cluster_id = int(source_endpoint["cluster_id"])
                payload = {
                    "remote_cluster_id": source_cluster_id,
                    "remote_ip": str(source_endpoint["listen_ip"]),
                    "remote_port": int(source_endpoint["listen_port"]),
                    "local_ip": str(target_endpoint["listen_ip"]),
                    "local_port": int(target_endpoint["listen_port"]),
                    "timeout_ms": timeout_ms,
                }
            except (KeyError, TypeError, ValueError) as exc:
                return finish(
                    {
                        "ok": False,
                        "error": f"invalid datadist endpoint metadata: {exc}",
                    }
                )
            if source_cluster_id in {int(value) for value in target_health.get("linked_cluster_ids", [])}:
                phase_ms["link"] = 0.0
                return finish({"ok": True, "already_linked": True, **payload})
            link_started = time.perf_counter()
            result = await self._post_kvbm_control(
                target,
                "/datadist/link",
                payload,
                timeout_ms=timeout_ms + 200,
            )
            phase_ms["link"] = (time.perf_counter() - link_started) * 1000.0
            if result.get("ok"):
                self._kvbm_datadist_status_cache.pop(target_worker_id, None)
                return finish({**result, **payload})

            # status 缓存可能仍是其他请求建链前的旧快照。link 报错后强制读取
            # 目标端真实状态；若底层链路已经存在，则把重复建链视为幂等成功。
            refreshed_target_status = await self.get_datadist_status(
                target_worker_id,
                cache_ttl_seconds=status_ttl_seconds,
                force_refresh=True,
                timeout_ms=timeout_ms,
            )
            refreshed_health = refreshed_target_status.get("health") or {}
            try:
                refreshed_linked_cluster_ids = {int(value) for value in refreshed_health.get("linked_cluster_ids", [])}
            except (TypeError, ValueError):
                refreshed_linked_cluster_ids = set()
            if refreshed_target_status.get("ok") and source_cluster_id in refreshed_linked_cluster_ids:
                return finish(
                    {
                        "ok": True,
                        "already_linked": True,
                        "reconciled_after_link_error": True,
                        "link_error": result,
                        **payload,
                    }
                )
            return finish(
                {
                    **result,
                    "refreshed_target_status": refreshed_target_status,
                    **payload,
                }
            )

    async def transfer_kv_blocks(
        self,
        target_worker_id: str,
        source_worker_id: str,
        source_blocks: list[int],
        block_hashes: list[str],
        *,
        request_id: str,
        timeout_ms: int = 3000,
        link_timeout_ms: int = 3000,
        status_ttl_seconds: float = 10.0,
        source_lease_ttl_ms: int = 30000,
        reservation_ttl_ms: int = 120000,
    ) -> dict:
        """通过目标 Worker 的事务接口拉取并发布远端 KV block。

        参数:
            target_worker_id: 目标 Worker ID。
            source_worker_id: 源 Worker ID。
            source_blocks: 源 Worker 内的物理 block id 列表。
            block_hashes: Router 观察到的对应外部 hash，用于源端复用竞态校验。
            request_id: 当前推理请求 ID，用于生成和串联 transfer_id。
            timeout_ms: pull-and-publish HTTP 请求预算，单位毫秒。
            link_timeout_ms: 首次建链预算，单位毫秒。
            status_ttl_seconds: endpoint 状态缓存时间，单位秒。
            source_lease_ttl_ms: 源端 pin lease 生存时间，单位毫秒。
            reservation_ttl_ms: 目标 BlockPool 预留生存时间，单位毫秒。

        返回:
            事务成功时 ok=True；失败或超时时返回 ok=False，且超时会请求取消。
        """

        total_started = time.perf_counter()
        if not source_blocks or len(source_blocks) != len(block_hashes):
            return {"ok": False, "error": "source blocks and hashes must be non-empty and aligned"}
        link = await self.ensure_datadist_link(
            target_worker_id,
            source_worker_id,
            timeout_ms=link_timeout_ms,
            status_ttl_seconds=status_ttl_seconds,
        )
        if not link.get("ok"):
            return {"ok": False, "stage": "link", "link": link}
        phase_ms = dict(link.get("phase_ms") or {})
        target = self.workers.get(target_worker_id)
        source = self.workers.get(source_worker_id)
        if target is None or source is None:
            return {"ok": False, "error": "source or target worker not found"}
        transfer_id = f"router-{request_id}-{uuid.uuid4().hex[:12]}"
        payload = {
            "transfer_id": transfer_id,
            "remote_cluster_id": int(link["remote_cluster_id"]),
            "source_control_url": self._worker_control_url(source),
            "source_blocks": source_blocks,
            "external_block_hashes": block_hashes,
            "source_lease_ttl_ms": source_lease_ttl_ms,
            "reservation_ttl_ms": reservation_ttl_ms,
            "timeout_ms": timeout_ms,
            "deadline_epoch_ms": int(time.time() * 1000) + max(timeout_ms, 0),
        }
        worker_transaction_started = time.perf_counter()
        result = await self._post_kvbm_control(
            target,
            "/datadist/transfers/pull",
            payload,
            timeout_ms=timeout_ms + 200,
        )
        phase_ms.update(result.get("phase_ms") or {})
        phase_ms["worker_transaction_http"] = (time.perf_counter() - worker_transaction_started) * 1000.0
        phase_ms["router_transfer_total"] = (time.perf_counter() - total_started) * 1000.0
        result["phase_ms"] = phase_ms
        if result.get("ok"):
            return result
        # Worker 重启会清空进程内链路，但 Router 可能仍持有短期状态缓存。
        # 失败后立即失效目标状态，使下一次请求重新探测并按需建链。
        self._kvbm_datadist_status_cache.pop(target_worker_id, None)
        # HTTP 超时或连接中断时，Worker 事务可能仍在运行；协作式取消可触发
        # publish 前检查及 reservation rollback，避免迟到结果占用 NPU。
        cancel = await self._post_kvbm_control(
            target,
            "/datadist/transfers/cancel",
            {"transfer_id": transfer_id},
            timeout_ms=500,
        )
        return {
            **result,
            "ok": False,
            "transfer_id": transfer_id,
            "cancel": cancel,
        }

    async def forward_request(
        self, worker_id: str, endpoint: str, method: str, data: dict, headers: dict | None = None
    ) -> dict:
        """转发请求到工作器

        Args:
            worker_id: 工作器ID
            endpoint: API端点（如 /v1/chat/completions）
            method: HTTP方法
            data: 请求数据
            headers: 额外的请求头

        Returns:
            工作器响应

        """
        logger.debug(f"[WORKER: FORWARD] Forwarding request to worker {worker_id}, endpoint={endpoint}")

        worker = self.workers.get(worker_id)
        if not worker:
            logger.error(f"[WORKER: ERROR] Worker {worker_id} not found")
            raise ValueError(f"Worker {worker_id} not found")

        url = worker_endpoint_url(worker.url, endpoint)

        request_headers = {"Content-Type": "application/json"}
        if worker.api_key:
            request_headers["Authorization"] = f"Bearer {worker.api_key}"
        if headers:
            request_headers.update(headers)

        logger.debug(
            f"[WORKER: REQUEST] {worker_id} url={url}, model={data.get('model')}, max_tokens={data.get('max_tokens')}"
        )

        timeout = httpx.Timeout(settings.request_forward_timeout, connect=30.0)

        try:
            if method.upper() == "POST":
                response = await self.http_client.post(url, json=data, headers=request_headers, timeout=timeout)
            elif method.upper() == "GET":
                response = await self.http_client.get(url, headers=request_headers, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.text
            except Exception:
                logger.debug("Failed to read error response body for worker {}", worker_id)
            logger.error(
                f"Forward request failed for worker {worker_id}: HTTP {e.response.status_code}, "
                f"response: {error_detail}"
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Forward request timeout for worker {worker_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request timeout when forwarding to worker",
            ) from None
        except httpx.RequestError as e:
            logger.error(f"Forward request failed for worker {worker_id}: {type(e).__name__}: {e}")
            raise

    async def forward_request_stream(
        self, worker_id: str, endpoint: str, method: str, data: dict, headers: dict | None = None
    ) -> AsyncGenerator[bytes, None]:
        """转发流式请求到工作器

        Args:
            worker_id: 工作器ID
            endpoint: API端点（如 /v1/chat/completions）
            method: HTTP方法
            data: 请求数据
            headers: 额外的请求头

        Returns:
            流式响应生成器

        """
        logger.debug(f"[WORKER: FORWARD_STREAM] Forwarding stream request to worker {worker_id}, endpoint={endpoint}")

        worker = self.workers.get(worker_id)
        if not worker:
            logger.error(f"[WORKER: ERROR] Worker {worker_id} not found")
            raise ValueError(f"Worker {worker_id} not found")

        url = worker_endpoint_url(worker.url, endpoint)

        request_headers = {"Content-Type": "application/json"}
        if worker.api_key:
            request_headers["Authorization"] = f"Bearer {worker.api_key}"
        if headers:
            request_headers.update(headers)

        logger.debug(
            "[WORKER: REQUEST_STREAM] {} url={}, model={}, max_tokens={}",
            worker_id,
            url,
            data.get("model"),
            data.get("max_tokens"),
        )

        timeout = httpx.Timeout(settings.request_forward_timeout, connect=30.0)

        try:
            if method.upper() == "POST":
                async with self.http_client.stream(
                    "POST", url, json=data, headers=request_headers, timeout=timeout
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
            else:
                raise ValueError(f"Unsupported HTTP method for streaming: {method}")

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.text
            except Exception:
                logger.debug("Failed to read error response body for stream worker {}", worker_id)
            logger.error(
                f"Forward stream request failed for worker {worker_id}: HTTP {e.response.status_code}, "
                f"response: {error_detail}"
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Forward stream request timeout for worker {worker_id}: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Forward stream request failed for worker {worker_id}: {type(e).__name__}: {e}")
            raise
