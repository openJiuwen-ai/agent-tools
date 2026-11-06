"""推理引擎 Scheduler 与 KVBM 之间的 KV block 所有权边界。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class SchedulerKVOwnershipMode(str, Enum):
    """KVBM 在 PD 路径中使用的 NPU block 所有权模式。"""

    LEGACY = "legacy"
    SCHEDULER = "scheduler"


class SchedulerKVBridge(ABC):
    """隔离推理引擎 Scheduler block 生命周期的适配接口。

    KVBM 通过该接口获取源 block 或准备目标 block，但不直接假设具体引擎的
    BlockPool、请求表和 prefix cache 实现。后续 scheduler 模式会由推理引擎
    分配目标 block，并只把可传输的物理布局交给 KVBM。
    """

    @property
    @abstractmethod
    def ownership_mode(self) -> SchedulerKVOwnershipMode:
        """返回当前 bridge 使用的 block 所有权模式。"""

    @abstractmethod
    def prepare_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """登记P请求的源block捕获意图。

        参数:
            payload: 包含handoff、request和捕获TTL的请求字典。

        返回:
            包含创建状态或错误信息的可序列化字典。
        """

    @abstractmethod
    def export_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """读取并pin P请求可传输的源block布局。

        参数:
            payload: 包含request、block限制和等待预算的请求字典。

        返回:
            成功时包含源block、hash和lease；尚未就绪时返回pending状态。
        """

    @abstractmethod
    def release_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        """释放P请求的捕获意图。

        参数:
            payload: 包含request_id和handoff_id的请求字典。

        返回:
            包含幂等释放结果的可序列化字典。
        """

    @abstractmethod
    def prepare_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """为D请求建立Scheduler等待状态。

        参数:
            payload: 包含handoff、request和门控TTL的请求字典。

        返回:
            包含目标请求门控状态的可序列化字典。
        """

    @abstractmethod
    def plan_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """为D请求生成接收KV的目标物理布局。

        参数:
            payload: 包含P端block语义、请求ID和等待预算的请求字典。

        返回:
            包含复用位置、传输位置和目标block ID的计划字典。
        """

    @abstractmethod
    def complete_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """在HIXL写入完成后提交D请求的目标block。

        参数:
            payload: 包含reservation、block语义和请求ID的请求字典。

        返回:
            包含attach、publish和Scheduler提交结果的字典。
        """

    @abstractmethod
    def abort_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """终止D接收事务并解除Scheduler等待。

        参数:
            payload: 包含request_id和可选handoff_id的请求字典。

        返回:
            包含幂等释放及回滚结果的可序列化字典。
        """

    @abstractmethod
    def target_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        """读取D请求的Scheduler接收状态。

        参数:
            payload: 包含可选request_id的查询字典。

        返回:
            指定请求或全部等待事务的状态字典。
        """

    def diagnostics(self) -> dict[str, Any]:
        """返回bridge模式和实现类型诊断。

        返回:
            可直接加入Worker status响应的诊断字典。
        """

        return {
            "ownership_mode": self.ownership_mode.value,
            "implementation": type(self).__name__,
        }
