import hashlib
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger

from openjiuwentools.infer_router.kv_cache.event import CacheEvent, CacheLocation
from openjiuwentools.infer_router.kv_cache.kv_event_manager import KVEventManager
from openjiuwentools.infer_router.monitoring import metrics


@dataclass
class CacheUpdateParams:
    """缓存更新参数"""

    block_hashes: list[str] = field(default_factory=list)
    token_count: int = 0
    location: str = "gpu"
    token_ids: list[int] = field(default_factory=list)


class EngineType(Enum):
    """推理引擎类型"""

    VLLM = "vllm"
    SGLANG = "sglang"


@dataclass
class BlockInfo:
    """缓存块信息"""

    block_hash: str
    token_count: int
    location: CacheLocation = CacheLocation.NPU
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ref_count: int = 1
    is_pinned: bool = False
    decay_factor: float = 1.0  # 衰减因子，1.0表示刚开始，0.0表示接近完成
    ttl: float | None = None  # 生存时间（秒），None表示永不过期


@dataclass
class RadixNode:
    """Radix Tree 节点（用于 SGLang）"""

    node_id: str
    token_id: int = 0  # 只存储当前 token，不存储完整路径
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    parent: Optional["RadixNode"] = None
    ref_count: int = 0
    is_pinned: bool = False
    location: CacheLocation = CacheLocation.NPU
    block_ids: list[str] = field(default_factory=list)
    last_accessed: float = 0.0

    def get_token_ids(self) -> list[int]:
        """通过 parent 回溯获取完整 token 路径"""
        path = []
        node: RadixNode | None = self
        while node and node.parent is not None:  # 跳过根节点
            path.append(node.token_id)
            node = node.parent
        return path[::-1]  # 反转得到正确顺序


@dataclass
class PodCacheState:
    """Pod 级缓存状态"""

    pod_id: str
    engine_type: EngineType
    model_id: str
    cached_blocks: dict[str, BlockInfo] = field(default_factory=dict)
    radix_tree_root: RadixNode | None = None
    gpu_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    queue_depth: int = 0
    running_requests: int = 0
    last_updated: float = field(default_factory=time.time)


class _ReadWriteLock:
    """简单的读写锁实现"""

    def __init__(self):
        self._read_lock = threading.Lock()  # 保护读计数器
        self._write_lock = threading.Lock()  # 写操作互斥锁
        self._read_count = 0  # 当前读操作数量

    def acquire_read(self):
        """获取读锁"""
        with self._read_lock:
            self._read_count += 1
            if self._read_count == 1:
                # 第一个读操作需要获取写锁（阻止写操作）
                self._write_lock.acquire()

    def release_read(self):
        """释放读锁"""
        with self._read_lock:
            self._read_count -= 1
            if self._read_count == 0:
                # 最后一个读操作释放写锁
                self._write_lock.release()

    def acquire_write(self):
        """获取写锁"""
        self._write_lock.acquire()

    def release_write(self):
        """释放写锁"""
        self._write_lock.release()


class RadixTree:
    """Radix Tree 实现（使用读写锁优化并发性能）"""

    def __init__(self):
        self.root = RadixNode(node_id="root")
        self.node_count = 0
        self.total_tokens = 0
        self._rw_lock = _ReadWriteLock()  # 使用读写锁

    def insert(self, token_ids: list[int]) -> tuple[RadixNode, int]:
        """插入 token 序列，返回最终节点和匹配长度（写操作）

        Args:
            token_ids: token ID 列表

        Returns:
            (最终节点, 新插入的 token 数量)

        """
        self._rw_lock.acquire_write()
        try:
            current = self.root
            matched_len = 0

            for token_id in token_ids:
                if token_id in current.children:
                    current = current.children[token_id]
                    matched_len += 1
                else:
                    break

            new_tokens = len(token_ids) - matched_len

            if new_tokens > 0:
                for i in range(matched_len, len(token_ids)):
                    token_id = token_ids[i]
                    new_node = RadixNode(
                        node_id=f"node_{self.node_count}",
                        token_id=token_id,  # 只存储单个 token
                        parent=current,
                    )
                    current.children[token_id] = new_node
                    current = new_node
                    self.node_count += 1
                    self.total_tokens += 1

            current.ref_count += 1
            current.last_accessed = time.time()

            return current, new_tokens
        finally:
            self._rw_lock.release_write()

    def find_longest_prefix(self, token_ids: list[int]) -> tuple[RadixNode | None, int]:
        """查找最长前缀匹配（读操作）

        Args:
            token_ids: token ID 列表

        Returns:
            (匹配的节点, 匹配的 token 数量)

        """
        self._rw_lock.acquire_read()
        try:
            current = self.root
            matched_len = 0

            for token_id in token_ids:
                if token_id in current.children:
                    current = current.children[token_id]
                    matched_len += 1
                else:
                    break

            if matched_len == 0:
                return None, 0

            return current, matched_len
        finally:
            self._rw_lock.release_read()

    def get_stats(self) -> dict[str, Any]:
        """获取 Radix Tree 统计信息（读操作）"""
        self._rw_lock.acquire_read()
        try:
            return {
                "node_count": self.node_count,
                "total_tokens": self.total_tokens,
            }
        finally:
            self._rw_lock.release_read()


class KVCacheIndex:
    """KV Cache 状态索引，维护全局缓存视图"""

    def __init__(self, block_size: int = 16):
        """初始化 KV Cache 索引

        Args:
            block_size: 块大小（token 数量）

        """
        self.block_size = block_size
        self.pod_states: dict[str, PodCacheState] = {}
        self.session_affinity: dict[str, str] = {}
        self.model_versions: dict[str, set[str]] = {}
        self.prefix_index: dict[str, set[str]] = {}

    def register_pod(
        self,
        pod_id: str,
        engine_type: EngineType,
        model_id: str,
        initial_state: PodCacheState | None = None,
    ):
        """注册新的 Pod

        Args:
            pod_id: Pod ID
            engine_type: 引擎类型
            model_id: 模型 ID
            initial_state: 初始状态

        """
        if pod_id in self.pod_states:
            logger.warning(f"Pod {pod_id} already registered, updating state")

        if initial_state:
            self.pod_states[pod_id] = initial_state
        else:
            self.pod_states[pod_id] = PodCacheState(
                pod_id=pod_id,
                engine_type=engine_type,
                model_id=model_id,
                radix_tree_root=RadixTree().root,
            )

        if model_id not in self.model_versions:
            self.model_versions[model_id] = set()
        self.model_versions[model_id].add(pod_id)

        logger.info(f"Registered pod {pod_id} with engine {engine_type.value}, model {model_id}")

    def unregister_pod(self, pod_id: str):
        """注销 Pod

        Args:
            pod_id: Pod ID

        """
        if pod_id not in self.pod_states:
            return

        pod_state = self.pod_states[pod_id]

        if pod_state.model_id in self.model_versions:
            self.model_versions[pod_state.model_id].discard(pod_id)

        del self.pod_states[pod_id]

        for _prefix_hash, pod_set in self.prefix_index.items():
            pod_set.discard(pod_id)

        sessions_to_remove = [sid for sid, pid in self.session_affinity.items() if pid == pod_id]
        for sid in sessions_to_remove:
            del self.session_affinity[sid]

        logger.info(f"Unregistered pod {pod_id}")

    @staticmethod
    def _handle_store_event(pod_state: PodCacheState, event: CacheEvent):
        """处理 store 事件"""
        if event.token_ids and pod_state.radix_tree_root:
            radix_tree = RadixTree()
            radix_tree.root = pod_state.radix_tree_root
            radix_tree.insert(event.token_ids)
            pod_state.radix_tree_root = radix_tree.root

    @staticmethod
    def _handle_evict_event(pod_state: PodCacheState, event: CacheEvent):
        """处理 evict 事件"""
        if event.token_ids and pod_state.radix_tree_root:
            radix_tree = RadixTree()
            radix_tree.root = pod_state.radix_tree_root
            for token_id in event.token_ids:
                radix_tree.delete(token_id)
            pod_state.radix_tree_root = radix_tree.root

    def _handle_hit_event(self, pod_state: PodCacheState, event: CacheEvent):
        """处理 hit 事件"""
        pass

    def update_pod_cache(
        self,
        pod_id: str,
        event: CacheEvent,
    ):
        """更新 Pod 缓存状态

        Args:
            pod_id: Pod ID
            event: 缓存事件

        """
        if pod_id not in self.pod_states:
            logger.warning(f"Pod {pod_id} not registered")
            return

        pod_state = self.pod_states[pod_id]
        pod_state.last_updated = time.time()

        event_handlers = {
            "store": self._handle_store_event,
            "evict": self._handle_evict_event,
            "hit": self._handle_hit_event,
        }

        handler = event_handlers.get(event.event_type)
        if handler:
            handler(pod_state, event)

        metrics.record_cache_hit() if event.event_type == "hit" else metrics.record_cache_miss()

    def get_pods_with_prefix(self, prefix_hash: str) -> list[str]:
        """获取拥有指定前缀的 Pod 列表

        Args:
            prefix_hash: 前缀哈希

        Returns:
            Pod ID 列表

        """
        return list(self.prefix_index.get(prefix_hash, set()))

    def get_session_affinity(self, session_id: str) -> str | None:
        """获取会话亲和性

        Args:
            session_id: 会话 ID

        Returns:
            Pod ID 或 None

        """
        return self.session_affinity.get(session_id)

    def set_session_affinity(self, session_id: str, pod_id: str):
        """设置会话亲和性

        Args:
            session_id: 会话 ID
            pod_id: Pod ID

        """
        self.session_affinity[session_id] = pod_id

    def get_pods_for_model(self, model_id: str) -> list[str]:
        """获取指定模型的 Pod 列表

        Args:
            model_id: 模型 ID

        Returns:
            Pod ID 列表

        """
        return list(self.model_versions.get(model_id, set()))


class PrefixCacheScorer:
    """前缀缓存评分器"""

    def __init__(self, block_size: int = 16):
        """初始化评分器

        Args:
            block_size: 块大小

        """
        self.block_size = block_size

    @staticmethod
    def compute_block_hash(token_ids: list[int]) -> str:
        """计算块哈希

        Args:
            token_ids: token ID 列表

        Returns:
            块哈希

        """
        token_str = ",".join(map(str, token_ids))
        return hashlib.sha256(token_str.encode()).hexdigest()

    @staticmethod
    def score_vllm(pod_state: PodCacheState, token_ids: list[int]) -> float:
        """VLLM 评分：基于块级前缀匹配（已弃用，保留兼容）

        Args:
            pod_state: Pod 缓存状态
            token_ids: token ID 列表

        Returns:
            评分（0-1）

        """
        return PrefixCacheScorer.score(pod_state, token_ids)

    @staticmethod
    def score_sglang(pod_state: PodCacheState, token_ids: list[int]) -> float:
        """SGLang 评分：基于 Radix Tree 最长前缀匹配（已弃用，保留兼容）

        Args:
            pod_state: Pod 缓存状态
            token_ids: token ID 列表

        Returns:
            评分（0-1）

        """
        return PrefixCacheScorer.score(pod_state, token_ids)

    @staticmethod
    def score(pod_state: PodCacheState, token_ids: list[int]) -> float:
        """计算评分（统一使用 Radix Tree）

        Args:
            pod_state: Pod 缓存状态
            token_ids: token ID 列表

        Returns:
            评分（0-1）

        """
        if not pod_state.radix_tree_root or not token_ids:
            return 0.0

        radix_tree = RadixTree()
        radix_tree.root = pod_state.radix_tree_root

        node, matched_len = radix_tree.find_longest_prefix(token_ids)

        if matched_len == 0 or not node:
            return 0.0

        match_ratio = matched_len / len(token_ids)

        pin_bonus = 1.2 if node.is_pinned else 1.0
        ref_count_bonus = min(node.ref_count / 10, 0.5)

        return min(match_ratio * pin_bonus + ref_count_bonus, 1.0)


class KVCacheManager:
    """KV Cache 管理器"""

    def __init__(
        self,
        block_size: int = None,
        event_manager=None,
        event_generator=None,
        enable_radix_tree: bool = False,
    ):
        """初始化 KV Cache 管理器

        Args:
            block_size: 块大小
            event_manager: 事件管理器，如果为 None 则创建默认管理器
            event_generator: 事件生成器，如果为 None 则根据配置创建
            enable_radix_tree: 是否启用 Radix Tree 前缀匹配功能（默认启用）

        """
        from openjiuwentools.infer_router.config.config import settings

        self.block_size = block_size or settings.kv_cache_block_size
        self.index = KVCacheIndex(block_size=self.block_size)
        self.scorer = PrefixCacheScorer(block_size=self.block_size)

        self.max_blocks = settings.kv_cache_max_blocks
        self.aging_block_factor = settings.kv_cache_aging_block_factor
        self.decay_factor = settings.kv_cache_decay_factor
        self.enable_session_affinity = settings.kv_cache_enable_session_affinity
        self.enable_radix_tree = enable_radix_tree  # 新增：Radix Tree 开关

        # 初始化事件管理器

        self.event_manager = event_manager or KVEventManager()

        # 注册事件处理器
        self._register_event_handlers()

        # 初始化事件生成器（仅实现二使用）
        self.event_generator = event_generator
        if not self.event_generator and settings.kv_event_mode == "inner_event":
            from openjiuwentools.infer_router.kv_cache.kv_event_generator import (
                KVEventGenerator,
            )

            self.event_generator = KVEventGenerator(enable_radix_tree=self.enable_radix_tree)
            logger.info(f"KVEventGenerator initialized (inner_event mode, enable_radix_tree={self.enable_radix_tree})")
        elif self.event_generator:
            # 如果提供了外部事件生成器，同步 enable_radix_tree 设置
            if hasattr(self.event_generator, "enable_radix_tree"):
                self.event_generator.enable_radix_tree = self.enable_radix_tree
            logger.info(f"Using provided KVEventGenerator (enable_radix_tree={self.enable_radix_tree})")
        else:
            logger.info("Using worker event mode")

    def register_worker(
        self,
        worker_id: str,
        engine_type: str,
        model: str,
    ):
        """注册工作器

        Args:
            worker_id: 工作器 ID
            engine_type: 引擎类型（vllm 或 sglang）
            model: 模型名称

        """
        engine = EngineType.VLLM if engine_type.lower() == "vllm" else EngineType.SGLANG
        self.index.register_pod(worker_id, engine, model)

    def unregister_worker(self, worker_id: str):
        """注销工作器

        Args:
            worker_id: 工作器 ID

        """
        self.index.unregister_pod(worker_id)

        # 清空工作器的事件（实现二）
        if self.event_generator:
            self.event_generator.remove_worker(worker_id)
        logger.info(f"Unregistered worker: {worker_id}")

    def _register_event_handlers(self):
        """注册事件处理器"""
        self.event_manager.register_handler("store", self._handle_store_event)
        self.event_manager.register_handler("removed", self._handle_removed_event)
        self.event_manager.register_handler("Cleared", self._handle_cleared_event)
        self.event_manager.register_handler("evict", self._handle_evict_event)
        self.event_manager.register_handler("hit", self._handle_hit_event)
        logger.debug("Registered event handlers")

    def _handle_store_event(self, event):
        """处理 store 事件

        Args:
            event: 缓存事件

        """
        # 如果 Radix Tree 功能未启用，跳过更新
        if not self.enable_radix_tree:
            return

        worker_id = event.engine_specific.get("worker_id")
        if worker_id:
            pod_state = self.index.pod_states.get(worker_id)
            if pod_state and pod_state.radix_tree_root and event.token_ids:
                radix_tree = RadixTree()
                radix_tree.root = pod_state.radix_tree_root
                radix_tree.insert(event.token_ids)
                pod_state.radix_tree_root = radix_tree.root

    def _handle_removed_event(self, event):
        """处理 removed 事件

        Args:
            event: 缓存事件

        """
        pass

    def _handle_cleared_event(self, event):
        """处理 Cleared 事件

        Args:
            event: 缓存事件

        """
        worker_id = event.engine_specific.get("worker_id")
        if worker_id:
            # 清空工作器的缓存
            self._clear_worker_cache(worker_id)
            logger.debug(f"Cleared cache for worker {worker_id}")

    def _handle_evict_event(self, event):
        """处理 evict 事件

        Args:
            event: 缓存事件

        """
        # 如果 Radix Tree 功能未启用，跳过更新
        if not self.enable_radix_tree:
            return

        worker_id = event.engine_specific.get("worker_id")
        if worker_id:
            pod_state = self.index.pod_states.get(worker_id)
            if pod_state and pod_state.radix_tree_root and event.token_ids:
                radix_tree = RadixTree()
                radix_tree.root = pod_state.radix_tree_root
                for token_id in event.token_ids:
                    radix_tree.delete(token_id)
                pod_state.radix_tree_root = radix_tree.root

    def _handle_hit_event(self, event):
        """处理 hit 事件

        Args:
            event: 缓存事件

        """
        pass

    def update_cache_state(
        self,
        worker_id: str,
        event_type: str,
        params: CacheUpdateParams = None,
    ):
        """更新缓存状态

        Args:
            worker_id: 工作器 ID
            event_type: 事件类型（store, evict, hit）
            params: 缓存更新参数

        """
        if params is None:
            params = CacheUpdateParams()

        block_hashes = params.block_hashes
        location = params.location
        token_ids = params.token_ids

        if block_hashes is None:
            block_hashes = []
        if token_ids is None:
            token_ids = []

        location_map = {
            "npu": CacheLocation.NPU,
            "cpu": CacheLocation.CPU,
            "remote": CacheLocation.REMOTE,
        }

        event = CacheEvent(
            event_type=event_type,
            block_hashes=block_hashes,
            location=location_map.get(location, CacheLocation.NPU),
            token_ids=token_ids,
            engine_specific={"worker_id": worker_id},
        )

        self.index.update_pod_cache(worker_id, event)

    def get_best_worker_for_prefix(
        self,
        token_ids: list[int],
        model: str,
        exclude_workers: set[str] | None = None,
    ) -> str | None:
        """获取指定前缀的最佳工作器

        Args:
            token_ids: token ID 列表
            model: 模型名称
            exclude_workers: 排除的工作器 ID 集合

        Returns:
            最佳工作器 ID 或 None

        """
        exclude_workers = exclude_workers or set()

        candidate_workers = self.index.get_pods_for_model(model)
        candidate_workers = [w for w in candidate_workers if w not in exclude_workers]

        if not candidate_workers:
            return None

        best_worker = None
        best_score = -1.0

        for worker_id in candidate_workers:
            pod_state = self.index.pod_states.get(worker_id)
            if not pod_state:
                continue

            score = self.scorer.score(pod_state, token_ids)

            if score > best_score:
                best_score = score
                best_worker = worker_id

        return best_worker

    def find_matches(
        self,
        token_ids: list[int],
        model: str,
        exclude_workers: set[str] | None = None,
    ) -> dict[str, int]:
        """查找所有工作器的KV缓存匹配情况

        根据设计文档，通过KV索引器为输入令牌序列查找所有工作器的KV缓存匹配情况。
        结果存储为字典，键为工作器ID，值为匹配的块数量。

        Args:
            token_ids: token ID 列表
            model: 模型名称
            exclude_workers: 排除的工作器 ID 集合

        Returns:
            工作器ID -> 匹配块数量的字典

        """
        exclude_workers = exclude_workers or set()

        candidate_workers = self.index.get_pods_for_model(model)
        candidate_workers = [w for w in candidate_workers if w not in exclude_workers]

        # 如果 Radix Tree 功能未启用，返回所有 worker 的匹配长度为 0
        if not self.enable_radix_tree:
            return dict.fromkeys(candidate_workers, 0)

        overlap_scores: dict[str, int] = {}

        for worker_id in candidate_workers:
            pod_state = self.index.pod_states.get(worker_id)
            if not pod_state or not pod_state.radix_tree_root:
                overlap_scores[worker_id] = 0
                continue

            radix_tree = RadixTree()
            radix_tree.root = pod_state.radix_tree_root
            _, matched_len = radix_tree.find_longest_prefix(token_ids)
            overlap_scores[worker_id] = matched_len

        return overlap_scores

    def get_workers_with_prefix(self, prefix_hash: str) -> list[str]:
        """获取拥有指定前缀的工作器列表

        Args:
            prefix_hash: 前缀哈希

        Returns:
            工作器 ID 列表

        """
        return self.index.get_pods_with_prefix(prefix_hash)

    def set_session_affinity(self, session_id: str, worker_id: str):
        """设置会话亲和性

        Args:
            session_id: 会话 ID
            worker_id: 工作器 ID

        """
        self.index.set_session_affinity(session_id, worker_id)

    def get_session_affinity(self, session_id: str) -> str | None:
        """获取会话亲和性

        Args:
            session_id: 会话 ID

        Returns:
            工作器 ID 或 None

        """
        return self.index.get_session_affinity(session_id)

    def compute_prefix_hash(self, token_ids: list[int]) -> str:
        """计算前缀哈希

        Args:
            token_ids: token ID 列表

        Returns:
            前缀哈希

        """
        num_blocks = len(token_ids) // self.block_size
        if num_blocks == 0:
            block_tokens = token_ids
        else:
            block_tokens = token_ids[: self.block_size]

        return self.scorer.compute_block_hash(block_tokens)

    def get_cache_stats(self) -> dict[str, Any]:
        """获取缓存统计信息

        Returns:
            统计信息字典

        """
        total_blocks = 0
        total_pods = len(self.index.pod_states)
        engine_stats = {"vllm": 0, "sglang": 0}

        for pod_state in self.index.pod_states.values():
            total_blocks += len(pod_state.cached_blocks)
            engine_stats[pod_state.engine_type.value] += 1

        stats = {
            "total_blocks": total_blocks,
            "total_pods": total_pods,
            "engine_stats": engine_stats,
            "session_count": len(self.index.session_affinity),
            "model_count": len(self.index.model_versions),
            "prefix_count": len(self.index.prefix_index),
            "max_blocks": self.max_blocks,
            "event_stats": self.event_manager.get_event_stats(),
        }

        if self.event_generator:
            stats["event_generator_stats"] = self.event_generator.get_stats()

        return stats

    def get_worker_cache_size(self, worker_id: str) -> int:
        """获取工作器的缓存块数量（树大小）

        Args:
            worker_id: 工作器 ID

        Returns:
            缓存块数量

        """
        pod_state = self.index.pod_states.get(worker_id)
        if pod_state is None:
            return 0
        return len(pod_state.cached_blocks)

    @staticmethod
    def compute_decay_factor(current_output_tokens: int, expected_output_tokens: int) -> float:
        """计算输出块衰减因子

        Args:
            current_output_tokens: 当前已生成的输出token数量
            expected_output_tokens: 预期输出token数量

        Returns:
            衰减因子（0.0-1.0）

        """
        if expected_output_tokens <= 0:
            return 1.0

        decay_factor = 1.0 - (current_output_tokens / expected_output_tokens)
        return max(0.0, min(1.0, decay_factor))

    def update_block_decay_factor(self, worker_id: str, block_hash: str, decay_factor: float):
        """更新缓存块的衰减因子

        Args:
            worker_id: 工作器ID
            block_hash: 块哈希
            decay_factor: 衰减因子

        """
        if worker_id not in self.index.pod_states:
            return

        pod_state = self.index.pod_states[worker_id]
        if block_hash in pod_state.cached_blocks:
            pod_state.cached_blocks[block_hash].decay_factor = decay_factor

    def get_block_effective_weight(self, worker_id: str, block_hash: str) -> float:
        """获取缓存块的有效权重（考虑衰减因子）

        Args:
            worker_id: 工作器ID
            block_hash: 块哈希

        Returns:
            有效权重

        """
        if worker_id not in self.index.pod_states:
            return 0.0

        pod_state = self.index.pod_states[worker_id]
        if block_hash not in pod_state.cached_blocks:
            return 0.0

        block_info = pod_state.cached_blocks[block_hash]
        location_weights = {
            CacheLocation.NPU: 1.0,
            CacheLocation.CPU: 0.7,
            CacheLocation.REMOTE: 0.4,
        }
        location_weight = location_weights.get(block_info.location, 0.0)

        # 考虑衰减因子的有效权重
        effective_weight = location_weight * block_info.decay_factor
        return effective_weight

    def clean_expired_blocks(self, worker_id: str) -> int:
        """清理过期的缓存块

        Args:
            worker_id: 工作器ID

        Returns:
            清理的块数量

        """
        if worker_id not in self.index.pod_states:
            return 0

        pod_state = self.index.pod_states[worker_id]
        current_time = time.time()
        expired_blocks = []

        for block_hash, block_info in pod_state.cached_blocks.items():
            # 检查是否过期
            if block_info.ttl is not None and current_time - block_info.created_at > block_info.ttl:
                expired_blocks.append(block_hash)

        # 清理过期的块
        for block_hash in expired_blocks:
            del pod_state.cached_blocks[block_hash]
            if block_hash in self.index.prefix_index:
                self.index.prefix_index[block_hash].discard(worker_id)

        if expired_blocks:
            logger.debug(f"Cleaned {len(expired_blocks)} expired blocks from worker {worker_id}")

        return len(expired_blocks)

    def trim_cache(self, worker_id: str, target_size: int | None = None) -> int:
        """基于大小修剪缓存

        Args:
            worker_id: 工作器ID
            target_size: 目标缓存大小（块数），如果为None则使用配置值

        Returns:
            修剪的块数量

        """
        if worker_id not in self.index.pod_states:
            return 0

        pod_state = self.index.pod_states[worker_id]
        current_size = len(pod_state.cached_blocks)
        target_size = target_size or self.max_blocks

        if current_size <= target_size:
            return 0

        # 需要修剪的块数量
        to_trim = current_size - target_size

        # 按优先级排序缓存块（优先级低的先被修剪）
        sorted_blocks = self._sort_blocks_by_priority(pod_state.cached_blocks)

        # 修剪块
        trimmed = 0
        for block_hash, _ in sorted_blocks[:to_trim]:
            if block_hash in pod_state.cached_blocks:
                # 跳过固定的块
                if pod_state.cached_blocks[block_hash].is_pinned:
                    continue

                del pod_state.cached_blocks[block_hash]
                if block_hash in self.index.prefix_index:
                    self.index.prefix_index[block_hash].discard(worker_id)
                trimmed += 1

        if trimmed > 0:
            logger.debug(f"Trimmed {trimmed} blocks from worker {worker_id}, new size: {len(pod_state.cached_blocks)}")

        return trimmed

    @staticmethod
    def _sort_blocks_by_priority(blocks: dict[str, BlockInfo]) -> list[tuple[str, float]]:
        """按优先级排序缓存块

        Args:
            blocks: 缓存块字典

        Returns:
            排序后的块哈希和优先级列表

        """

        def get_priority(block_info: BlockInfo) -> float:
            """计算块的优先级"""
            # 固定的块优先级最高
            if block_info.is_pinned:
                return float("inf")

            # 基于以下因素计算优先级：
            # 1. 引用计数（越高优先级越高）
            # 2. 最后访问时间（越近优先级越高）
            # 3. 衰减因子（越高优先级越高）
            current_time = time.time()
            recency = 1.0 / (1.0 + current_time - block_info.last_accessed)
            return block_info.ref_count * 0.5 + recency * 0.3 + block_info.decay_factor * 0.2

        # 按优先级升序排序（优先级低的先被修剪）
        return sorted(
            [(block_hash, get_priority(block_info)) for block_hash, block_info in blocks.items()],
            key=lambda x: x[1],
        )

    @staticmethod
    def warm_cache(prefix_hash: str, model: str, worker_id: str):
        """预热KV缓存

        Args:
            prefix_hash: 前缀哈希
            model: 模型名称
            worker_id: 工作器ID

        """
        # 实际实现中，这里应该：
        # 1. 检查工作器是否存在
        # 2. 构建预热请求
        # 3. 发送预热请求到工作器
        logger.debug(f"Warming cache for prefix {prefix_hash} on worker {worker_id} for model {model}")

    def process_event(self, event: CacheEvent) -> bool:
        """处理事件

        Args:
            event: 缓存事件

        Returns:
            是否处理成功

        """
        return self.event_manager.process_event(event)

    def generate_events(self, worker_id: str, token_ids: list[int]) -> list[CacheEvent]:
        """生成事件（仅实现二使用）

        Args:
            worker_id: 工作器 ID
            token_ids: token ID 列表

        Returns:
            CacheEvent 对象列表

        """
        if self.event_generator:
            events = self.event_generator.generate_events(worker_id, token_ids)
            for event in events:
                self.event_manager.process_event(event)
            return events
        return []

    def clear_worker_events(self, worker_id: str) -> list[CacheEvent]:
        """清空工作器的事件（仅实现二使用）

        Args:
            worker_id: 工作器 ID

        Returns:
            CacheEvent 对象列表

        """
        if self.event_generator:
            events = self.event_generator.clear_worker_events(worker_id)
            for event in events:
                self.event_manager.process_event(event)
            return events
        return []

    def get_event_stats(self) -> dict:
        """获取事件统计信息

        Returns:
            事件统计字典

        """
        return self.event_manager.get_event_stats()

    def get_event_generator_stats(self) -> dict:
        """获取事件生成器统计信息

        Returns:
            事件生成器统计字典

        """
        if self.event_generator:
            return self.event_generator.get_stats()
        return {}

    def clear_events(self) -> None:
        """清空事件缓冲区"""
        self.event_manager.clear_events()

    def _add_block(self, worker_id: str, block_hash: str, token_count: int) -> None:
        """添加缓存块

        Args:
            worker_id: 工作器 ID
            block_hash: 块哈希
            token_count: token 数量

        """
        pod_state = self.index.pod_states.get(worker_id)
        if pod_state:
            block_info = BlockInfo(
                block_hash=block_hash,
                token_count=token_count,
                location=CacheLocation.NPU,
            )
            pod_state.cached_blocks[block_hash] = block_info
            # 更新前缀索引
            if block_hash not in self.index.prefix_index:
                self.index.prefix_index[block_hash] = set()
            self.index.prefix_index[block_hash].add(worker_id)

    def _remove_block(self, worker_id: str, block_hash: str) -> None:
        """移除缓存块

        Args:
            worker_id: 工作器 ID
            block_hash: 块哈希

        """
        pod_state = self.index.pod_states.get(worker_id)
        if pod_state and block_hash in pod_state.cached_blocks:
            del pod_state.cached_blocks[block_hash]
            # 更新前缀索引
            if block_hash in self.index.prefix_index:
                self.index.prefix_index[block_hash].discard(worker_id)
                if not self.index.prefix_index[block_hash]:
                    del self.index.prefix_index[block_hash]

    def _clear_worker_cache(self, worker_id: str) -> None:
        """清空工作器的缓存

        Args:
            worker_id: 工作器 ID

        """
        pod_state = self.index.pod_states.get(worker_id)
        if pod_state:
            # 清理前缀索引
            for block_hash in list(pod_state.cached_blocks.keys()):
                if block_hash in self.index.prefix_index:
                    self.index.prefix_index[block_hash].discard(worker_id)
                    if not self.index.prefix_index[block_hash]:
                        del self.index.prefix_index[block_hash]
            # 清空缓存块
            pod_state.cached_blocks.clear()

    def _update_block_access_time(self, worker_id: str, block_hash: str) -> None:
        """更新块的访问时间

        Args:
            worker_id: 工作器 ID
            block_hash: 块哈希

        """
        pod_state = self.index.pod_states.get(worker_id)
        if pod_state and block_hash in pod_state.cached_blocks:
            pod_state.cached_blocks[block_hash].last_accessed = time.time()
