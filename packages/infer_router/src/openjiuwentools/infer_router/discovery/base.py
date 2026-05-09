"""工作器发现抽象基类"""

from abc import ABC, abstractmethod

from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo


class WorkerDiscovery(ABC):
    """工作器发现抽象基类"""

    @abstractmethod
    async def discover(self) -> list[WorkerInfo]:
        """发现工作器

        Returns:
            List[WorkerInfo]: 发现的工作器列表

        """
        pass

    @abstractmethod
    async def start(self):
        """启动发现服务（如果需要）"""
        pass

    @abstractmethod
    async def stop(self):
        """停止发现服务（如果需要）"""
        pass
