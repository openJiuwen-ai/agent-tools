"""从etcd发现工作器"""

import asyncio
import json
import threading
from collections.abc import Callable

import httpx
from loguru import logger

from openjiuwentools.infer_router.discovery.base import WorkerDiscovery
from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType


class EtcdDiscovery(WorkerDiscovery):
    """从etcd发现工作器"""

    def __init__(
        self,
        etcd_hosts: list[str] = None,
        etcd_port: int = 2379,
        etcd_prefix: str = "/jiuwen/workers",
        etcd_user: str | None = None,
        etcd_password: str | None = None,
        enable_watch: bool = False,
    ):
        """初始化etcd发现器

        Args:
            etcd_hosts: etcd主机列表
            etcd_port: etcd端口
            etcd_prefix: 工作器信息存储的前缀
            etcd_user: etcd用户名（可选）
            etcd_password: etcd密码（可选）
            enable_watch: 是否启用watch机制实时监听变化

        """
        self.etcd_hosts = etcd_hosts or ["localhost"]
        self.etcd_port = etcd_port
        self.etcd_prefix = etcd_prefix
        self.etcd_user = etcd_user
        self.etcd_password = etcd_password
        self.enable_watch = enable_watch
        self.client: httpx.AsyncClient | None = None
        self._watch_task: asyncio.Task | None = None
        self._watch_thread: threading.Thread | None = None
        self._watch_stop_event: threading.Event | None = None
        self._on_change_callback: Callable | None = None

    async def _get_client(self):
        """获取或创建HTTP客户端"""
        if self.client is None:
            try:
                # 构建etcd HTTP客户端
                endpoints = [f"http://{host}:{self.etcd_port}" for host in self.etcd_hosts]

                # 创建带有认证信息的客户端
                auth = None
                if self.etcd_user or self.etcd_password:
                    auth = (self.etcd_user or "", self.etcd_password or "")

                self.client = httpx.AsyncClient(base_url=endpoints[0], auth=auth)

                logger.info(f"Connected to etcd at {endpoints}")
            except Exception as e:
                logger.error(f"Failed to create etcd HTTP client: {e}")
                raise

        return self.client

    async def discover(self) -> list[WorkerInfo]:
        """从etcd读取工作器信息

        Returns:
            List[WorkerInfo]: 工作器列表

        """
        try:
            client = await self._get_client()

            # 构建etcd key前缀的URL
            key_prefix = f"/v2/keys{self.etcd_prefix}"
            url = f"{key_prefix}?recursive=true"

            # 发送HTTP GET请求
            response = await client.get(url)
            response.raise_for_status()

            workers = []
            result = response.json()

            # 处理etcd响应
            if "node" in result and "nodes" in result["node"]:
                for node in result["node"]["nodes"]:
                    if "value" in node:
                        try:
                            worker_data = json.loads(node["value"])
                            total_tokens = worker_data.get("total_tokens", 0)
                            if total_tokens <= 0:
                                total_tokens = worker_data.get("available_memory", 0)

                            worker_type_str = worker_data.get("worker_type", "combined").lower()
                            worker_type = (
                                WorkerType(worker_type_str)
                                if worker_type_str in [t.value for t in WorkerType]
                                else WorkerType.COMBINED
                            )

                            worker_kwargs = {
                                "worker_id": worker_data["worker_id"],
                                "model": worker_data["model"],
                                "url": worker_data["url"],
                                "current_load": worker_data.get("current_load", 0),
                                "cached_prefixes": worker_data.get("cached_prefixes", []),
                                "engine_type": worker_data.get("engine_type", "vllm"),
                                "api_key": worker_data.get("api_key"),
                                "worker_type": worker_type,
                                "group": worker_data.get("group", "default"),
                                "kv_addr": worker_data.get("kv_addr", ""),
                                "publisher_endpoint": worker_data.get("publisher_endpoint", ""),
                            }
                            if total_tokens > 0:
                                worker_kwargs["total_tokens"] = total_tokens

                            worker = WorkerInfo(**worker_kwargs)
                            workers.append(worker)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse worker data from etcd: {e}")
                        except KeyError as e:
                            logger.error(f"Invalid worker data in etcd, missing field {e}")
                        except Exception as e:
                            logger.error(f"Failed to process worker data from etcd: {e}")

            logger.info(f"Discovered {len(workers)} workers from etcd")
            return workers

        except httpx.HTTPError as e:
            logger.error(f"Failed to discover workers from etcd: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to discover workers from etcd: {e}")
            return []

    def set_on_change_callback(self, callback: Callable):
        """设置变化回调函数

        Args:
            callback: 回调函数，当工作器信息发生变化时调用

        """
        self._on_change_callback = callback

    async def start(self):
        """启动发现服务"""
        try:
            await self._get_client()
            logger.info(f"Etcd discovery started with prefix: {self.etcd_prefix}")

        except Exception as e:
            logger.error(f"Failed to start etcd discovery: {e}")
            raise

    async def stop(self):
        """停止发现服务"""
        # 关闭HTTP客户端
        if self.client:
            await self.client.aclose()
            self.client = None

        logger.info("Etcd discovery stopped")
