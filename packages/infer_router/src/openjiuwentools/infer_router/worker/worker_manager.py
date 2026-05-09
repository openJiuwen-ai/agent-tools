import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

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
        self.kv_cache_manager = kv_cache_manager
        self._workload_manager = None
        self._load_store = AtomicLoadStore()  # 线程安全的负载存储

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
    def _is_prefill_worker(worker: WorkerInfo) -> bool:
        """检查是否为prefill工作器（包括combined类型）"""
        return worker.worker_type in (WorkerType.PREFILL, WorkerType.COMBINED)

    @staticmethod
    def _is_decode_worker(worker: WorkerInfo) -> bool:
        """检查是否为decode工作器（包括combined类型）"""
        return worker.worker_type in (WorkerType.DECODE, WorkerType.COMBINED)

    def get_prefill_workers_in_group(self, group: str, model: str) -> list[WorkerInfo]:
        """获取指定group内的prefill工作器（包括combined类型）"""
        return [w for w in self.get_workers_in_group(group) if w.model == model and self._is_prefill_worker(w)]

    def get_decode_workers_in_group(self, group: str, model: str) -> list[WorkerInfo]:
        """获取指定group内的decode工作器（包括combined类型）"""
        return [w for w in self.get_workers_in_group(group) if w.model == model and self._is_decode_worker(w)]

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
                f"Worker {worker.worker_id} model {worker.model} "
                f"does not match group {worker.group} model {group_model}"
            )
            return False

        has_combined = any(w.worker_type == WorkerType.COMBINED for w in workers_in_group)

        if has_combined and worker.worker_type != WorkerType.COMBINED:
            logger.warning(
                f"Worker {worker.worker_id} type {worker.worker_type} "
                f"cannot join group {worker.group} which contains combined workers"
            )
            return False

        if not has_combined and worker.worker_type == WorkerType.COMBINED:
            logger.warning(
                f"Worker {worker.worker_id} type COMBINED "
                f"cannot join group {worker.group} which contains non-combined workers"
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

            base_url = worker.url.rstrip("/v1").rstrip("/v")
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
                    logger.warning(f"Worker {worker_id} health check failed with status {response.status_code}")
                    new_consecutive_failures = old_consecutive_failures + 1
                    # 只有连续失败次数超过阈值时才标记为不健康
                    is_healthy = new_consecutive_failures < settings.worker_health_check_max_failures

            except httpx.HTTPStatusError as e:
                logger.error(f"Worker {worker_id} health check failed: HTTP {e.response.status_code}")
                new_consecutive_failures = old_consecutive_failures + 1
                is_healthy = new_consecutive_failures < settings.worker_health_check_max_failures
            except httpx.RequestError as e:
                logger.error(f"Worker {worker_id} health check failed: {e}")
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

    async def forward_request(
        self,
        worker_id: str,
        endpoint: str,
        method: str,
        data: dict,
        headers: dict | None = None,
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
        logger.info(f"[WORKER: FORWARD] Forwarding request to worker {worker_id}")
        logger.debug(f"[WORKER: FORWARD] Endpoint: {endpoint}, Method: {method}")
        logger.debug(f"[WORKER: FORWARD] Data keys: {list(data.keys()) if data else None}")

        worker = self.workers.get(worker_id)
        if not worker:
            logger.error(f"[WORKER: ERROR] Worker {worker_id} not found")
            raise ValueError(f"Worker {worker_id} not found")

        url = f"{worker.url}{endpoint}"
        logger.info(f"[WORKER: URL] Forwarding to {url}")

        request_headers = {"Content-Type": "application/json"}
        if worker.api_key:
            request_headers["Authorization"] = f"Bearer {worker.api_key}"
            logger.debug(f"[WORKER: AUTH] Using API key for worker {worker_id}")
        if headers:
            request_headers.update(headers)

        logger.info(f"[WORKER: FINAL_HEADERS] {request_headers}")
        logger.info(
            f"[WORKER: REQUEST_DATA] model={data.get('model')}, "
            f"max_tokens={data.get('max_tokens')}, "
            f"messages_count={len(data.get('messages', []))}"
        )

        timeout = httpx.Timeout(settings.request_forward_timeout, connect=30.0)
        logger.debug(f"[WORKER: TIMEOUT] Request timeout: {settings.request_forward_timeout}s")
        logger.info(f"[WORKER: REQUEST] URL: {url}")
        logger.info(f"[WORKER: REQUEST] Headers: {request_headers}")

        try:
            if method.upper() == "POST":
                logger.info(f"[WORKER: POST] Sending POST request to worker {worker_id}")
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
                error_detail = "Unable to read response text"
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
        self,
        worker_id: str,
        endpoint: str,
        method: str,
        data: dict,
        headers: dict | None = None,
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
        logger.info(f"[WORKER: FORWARD_STREAM] Forwarding stream request to worker {worker_id}")
        logger.debug(f"[WORKER: FORWARD_STREAM] Endpoint: {endpoint}, Method: {method}")
        logger.debug(f"[WORKER: FORWARD_STREAM] Data keys: {list(data.keys()) if data else None}")

        worker = self.workers.get(worker_id)
        if not worker:
            logger.error(f"[WORKER: ERROR] Worker {worker_id} not found")
            raise ValueError(f"Worker {worker_id} not found")

        url = f"{worker.url}{endpoint}"
        logger.info(f"[WORKER: URL] Forwarding to {url}")

        request_headers = {"Content-Type": "application/json"}
        if worker.api_key:
            request_headers["Authorization"] = f"Bearer {worker.api_key}"
            logger.debug(f"[WORKER: AUTH] Using API key for worker {worker_id}")
        if headers:
            request_headers.update(headers)

        logger.info(f"[WORKER: FINAL_HEADERS] {request_headers}")
        logger.info(
            f"[WORKER: REQUEST_DATA] model={data.get('model')}, "
            f"max_tokens={data.get('max_tokens')}, "
            f"messages_count={len(data.get('messages', []))}"
        )

        timeout = httpx.Timeout(settings.request_forward_timeout, connect=30.0)
        logger.debug(f"[WORKER: TIMEOUT] Request timeout: {settings.request_forward_timeout}s")

        try:
            if method.upper() == "POST":
                logger.info(f"[WORKER: POST_STREAM] Sending POST stream request to worker {worker_id}")
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
                error_detail = "Unable to read response text"
            logger.error(
                f"Forward stream request failed for worker {worker_id}: "
                f"HTTP {e.response.status_code}, response: {error_detail}"
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Forward stream request timeout for worker {worker_id}: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Forward stream request failed for worker {worker_id}: {type(e).__name__}: {e}")
            raise
