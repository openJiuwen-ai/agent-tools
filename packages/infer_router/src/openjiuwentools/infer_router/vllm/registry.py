import asyncio
import json
import logging
import uuid

logger = logging.getLogger("jiuwen.registry")


class EtcdRegistry:
    """将 worker 实例注册到 etcd，通过 lease 保活，供 router 发现。

    key 结构: /jiuwen/services/{service_name}/{instance_id}
    value: JSON 元数据（host, port, model, request_plane 等）
    """

    def __init__(
        self,
        etcd_endpoints: str,
        service_name: str,
        host: str,
        port: int,
        ttl: int = 10,
        model: str = "",
        worker_type: str = "combined",
        group: str = "default",
        engine_type: str = "vllm",
        kv_addr: str = "",
        publisher_endpoint: str = "",
        api_key: str | None = None,
        total_tokens: int = 1000000,
        metadata: dict | None = None,
    ) -> None:
        self.service_name = service_name
        self.host = host
        self.port = port
        self.ttl = ttl
        self.model = model
        self.worker_type = worker_type
        self.group = group
        self.engine_type = engine_type
        self.kv_addr = kv_addr
        self.publisher_endpoint = publisher_endpoint
        self.api_key = api_key
        self.total_tokens = total_tokens
        self.metadata = metadata or {}
        self.instance_id = uuid.uuid4().hex[:8]
        self._etcd_host, self._etcd_port = self._parse_endpoint(etcd_endpoints)
        self._client = None
        self._lease_id: int = 0
        self._keepalive_task: asyncio.Task | None = None

    @property
    def worker_id(self) -> str:
        """工作器 ID，格式：{service_name}-{instance_id}"""
        return f"{self.service_name}-{self.instance_id}"

    @property
    def key(self) -> str:
        return f"/jiuwen/workers/{self.worker_id}"

    @property
    def value(self) -> str:
        """构建 router 发现所需的完整工作器信息"""
        data = {
            "worker_id": self.worker_id,
            "model": self.model,
            "url": f"http://{self.host}:{self.port}",
            "engine_type": self.engine_type,
            "worker_type": self.worker_type,
            "group": self.group,
            "total_tokens": self.total_tokens,
            "current_load": 0,
            "cached_prefixes": [],
            "kv_addr": self.kv_addr,
            "publisher_endpoint": self.publisher_endpoint,
            "host": self.host,
            "port": self.port,
            "instance_id": self.instance_id,
        }
        data.update(self.metadata)
        return json.dumps(data, ensure_ascii=False)

    async def register(self) -> None:
        from etcd3 import Client

        self._client = Client(host=self._etcd_host, port=self._etcd_port)
        lease_resp = await asyncio.to_thread(self._client.lease_grant, self.ttl)
        self._lease_id = lease_resp.ID
        await asyncio.to_thread(self._client.put, self.key, self.value, lease=self._lease_id)
        self._keepalive_task = asyncio.create_task(self._keepalive())
        logger.info(
            "Service registered: name=%s key=%s addr=%s:%d ttl=%ds",
            self.service_name, self.key, self.host, self.port, self.ttl,
        )

    async def deregister(self) -> None:
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

        if self._client:
            try:
                await asyncio.to_thread(self._client.delete_range, self.key)
            except Exception:
                logger.warning("Failed to delete key %s", self.key, exc_info=True)
            try:
                if self._lease_id:
                    await asyncio.to_thread(self._client.lease_revoke, self._lease_id)
            except Exception:
                logger.warning("Failed to revoke lease", exc_info=True)
            try:
                await asyncio.to_thread(self._client.close)
            except Exception:
                logger.debug("Failed to close etcd client", exc_info=True)

        logger.info("Service deregistered: name=%s instance=%s", self.service_name, self.instance_id)

    async def _keepalive(self) -> None:
        interval = max(self.ttl // 3, 1)
        while True:
            await asyncio.sleep(interval)
            try:
                await asyncio.to_thread(self._client.lease_keep_alive_once, self._lease_id)
            except Exception:
                logger.exception("Lease keepalive failed, re-registering")
                try:
                    await self._re_register()
                except Exception:
                    logger.exception("Re-registration failed")

    async def _re_register(self) -> None:
        lease_resp = await asyncio.to_thread(self._client.lease_grant, self.ttl)
        self._lease_id = lease_resp.ID
        await asyncio.to_thread(self._client.put, self.key, self.value, lease=self._lease_id)
        logger.info("Re-registered service %s", self.key)

    @staticmethod
    def _parse_endpoint(endpoint: str) -> tuple[str, int]:
        endpoint = endpoint.replace("http://", "").replace("https://", "")
        parts = endpoint.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 2379
        return host, port
