"""线程安全的负载存储"""

import threading
from collections.abc import Callable


class AtomicLoadStore:
    """原子负载存储，提供线程安全的负载读写操作"""

    def __init__(self):
        self._loads: dict[str, float] = {}
        self._lock = threading.RLock()  # 可重入锁，支持嵌套调用

    def get_load(self, worker_id: str) -> float:
        """原子读取工作器负载"""
        with self._lock:
            return self._loads.get(worker_id, 0.0)

    def set_load(self, worker_id: str, load: float) -> None:
        """原子设置工作器负载"""
        with self._lock:
            self._loads[worker_id] = load

    def update_load(self, worker_id: str, update_func: Callable[[float], float]) -> float:
        """原子更新工作器负载"""
        with self._lock:
            current = self._loads.get(worker_id, 0.0)
            new_value = update_func(current)
            self._loads[worker_id] = new_value
            return new_value

    def remove_load(self, worker_id: str) -> None:
        """移除工作器负载记录"""
        with self._lock:
            if worker_id in self._loads:
                del self._loads[worker_id]

    def get_all_loads(self) -> dict[str, float]:
        """获取所有工作器负载的快照"""
        with self._lock:
            return dict(self._loads)  # 返回副本，避免外部修改
