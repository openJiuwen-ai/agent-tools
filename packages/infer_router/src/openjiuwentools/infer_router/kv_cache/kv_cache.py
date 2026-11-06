import hashlib
from functools import wraps
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger

from openjiuwentools.infer_router.kv_cache.event import (
    CacheEvent,
    CacheLocation,
    cache_location_from_medium,
)
from openjiuwentools.infer_router.kv_cache.kv_event_manager import KVEventManager
from openjiuwentools.infer_router.monitoring import metrics


_PREFIX_CURSOR_CACHE_MAX_ENTRIES = 2048


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
    physical_block_id: int | None = None  # 所属 worker 内的 vLLM 物理 block 编号


@dataclass(frozen=True)
class CrossWorkerTransferCandidate:
    """Router 选出的单源跨 Worker KV block 传输候选。

    属性:
        source_worker_id: 提供 KV cache 的源 Worker ID。
        block_hashes: 按请求前缀顺序排列的待传输 block hash。
        source_block_ids: 与 block_hashes 对齐的源端物理 block ID。
        source_matched_blocks: 源端匹配当前请求连续前缀的 block 数。
        source_npu_blocks: 源端匹配块中位于 NPU 的 block 数。
        physical_mapped_npu_blocks: 源端匹配块中同时位于 NPU 且具有物理 ID 的数量。
        target_matched_blocks: 目标端已经匹配的任意层级 block 数。
        target_npu_blocks: 目标端已经位于 NPU 的匹配 block 数。
        first_unselected_reason: 第一个未继续选择 block 的原因。
    """

    source_worker_id: str
    block_hashes: tuple[str, ...]
    source_block_ids: tuple[int, ...]
    source_matched_blocks: int
    source_npu_blocks: int
    physical_mapped_npu_blocks: int
    target_matched_blocks: int
    target_npu_blocks: int
    first_unselected_reason: str


@dataclass(frozen=True)
class PrefixBlockAliasSnapshot:
    """一个逻辑 block 边界上的缓存 hash 别名快照。

    属性:
        block_hash: Worker 上报的 block hash。
        location: 该别名当前所在的缓存层级。
        physical_block_id: Worker 内物理 block ID；未解析时为 None。
        token_count: 该 block 覆盖的有效 token 数。
        priority: 与实时索引相同的别名选择优先级，值越大越优先。
    """

    block_hash: str
    location: CacheLocation
    physical_block_id: int | None
    token_count: int
    priority: tuple[int, int, float, int]


@dataclass(frozen=True)
class PrefixBlockSnapshot:
    """请求前缀中一个逻辑 block 边界的只读快照。

    属性:
        token_end: 该 block 在请求 token 序列中的结束位置。
        aliases: 同一逻辑 block 的有效 hash 别名，已按优先级降序排列。
    """

    token_end: int
    aliases: tuple[PrefixBlockAliasSnapshot, ...]

    def select(
        self,
        locations: set[CacheLocation] | None = None,
    ) -> PrefixBlockAliasSnapshot | None:
        """选择满足位置约束的最高优先级 hash 别名。

        参数:
            locations: 允许的缓存位置集合；为 None 时允许所有位置。

        返回:
            满足条件的最高优先级别名；不存在时返回 None。
        """

        for alias in self.aliases:
            if locations is None or alias.location in locations:
                return alias
        return None


@dataclass(frozen=True)
class WorkerPrefixSnapshot:
    """单个 Worker 对一次请求的连续前缀匹配快照。

    属性:
        worker_id: Worker ID。
        model_id: Worker 服务的模型 ID；Worker 不存在时为空字符串。
        matched_tokens: 最深有效 block 边界对应的连续匹配 token 数。
        blocks: 按请求前缀顺序排列的逻辑 block 快照。
    """

    worker_id: str
    model_id: str
    matched_tokens: int
    blocks: tuple[PrefixBlockSnapshot, ...]

    def block_hashes(
        self,
        *,
        limit: int | None = None,
        locations: set[CacheLocation] | None = None,
    ) -> list[str]:
        """从快照提取满足位置约束的 block hash。

        参数:
            limit: 最多返回的 block 数；为 None 时不限制。
            locations: 允许的缓存位置集合；为 None 时允许所有位置。

        返回:
            按请求前缀顺序排列的 block hash 列表。
        """

        result: list[str] = []
        max_items = limit if limit is not None and limit >= 0 else None
        for block in self.blocks:
            alias = block.select(locations)
            if alias is None:
                continue
            result.append(alias.block_hash)
            if max_items is not None and len(result) >= max_items:
                break
        return result

    def location_counts(self) -> dict[CacheLocation, int]:
        """统计每个逻辑 block 的最高优先级缓存位置。

        返回:
            CacheLocation 到逻辑 block 数量的映射。
        """

        counts: dict[CacheLocation, int] = {}
        for block in self.blocks:
            alias = block.select()
            if alias is not None:
                counts[alias.location] = counts.get(alias.location, 0) + 1
        return counts


@dataclass(frozen=True)
class CrossWorkerTransferPlan:
    """一次跨 Worker 传输候选计算及诊断的共享结果。

    属性:
        candidate: 可执行的最佳传输候选；不存在时为 None。
        diagnostics: 与本次候选计算使用同一快照生成的跳过诊断。
        target_snapshot: 目标 Worker 的请求级前缀快照，可供 local prefetch 复用。
    """

    candidate: CrossWorkerTransferCandidate | None
    diagnostics: dict[str, Any]
    target_snapshot: WorkerPrefixSnapshot


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
    last_accessed: float = field(default_factory=time.time)

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
    block_node_map: dict[str, RadixNode] = field(default_factory=dict)
    active_node_ref_counts: dict[int, int] = field(default_factory=dict)
    radix_block_size: int | None = None
    radix_query_generation: int = 0
    prefix_cursor_cache: OrderedDict[tuple[int, int, tuple[int, ...]], RadixNode] = field(
        default_factory=OrderedDict, repr=False
    )
    gpu_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    queue_depth: int = 0
    running_requests: int = 0
    last_updated: float = field(default_factory=time.time)
    state_lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)


def _bind_block_to_radix_node(
    pod_state: PodCacheState,
    block_hash: str,
    node: RadixNode,
    location: CacheLocation,
) -> None:
    """将 block hash 绑定到事件声明的真实 block 边界节点。

    参数:
        pod_state: 当前 Worker 的 Router KV cache 状态。
        block_hash: Worker 上报的稳定 block hash。
        node: 按该事件 block_size 定位的 RadixTree 节点。
        location: 该 block 当前所在的缓存层级。

    返回:
        无返回值。

    说明:
        Worker 自动识别出的 block_size 事件可能晚于缺少该字段的原生事件。
        重新绑定时必须从旧节点移除 hash，才能纠正 Router 兜底尺寸造成的错误边界。
    """

    previous = pod_state.block_node_map.get(block_hash)
    if previous is node:
        if block_hash not in node.block_ids:
            node.block_ids.append(block_hash)
        node.location = location
        return
    if previous is not None and previous is not node:
        previous.block_ids = [existing for existing in previous.block_ids if existing != block_hash]
        previous_id = id(previous)
        remaining = pod_state.active_node_ref_counts.get(previous_id, 0) - 1
        if remaining > 0:
            pod_state.active_node_ref_counts[previous_id] = remaining
        else:
            pod_state.active_node_ref_counts.pop(previous_id, None)
    pod_state.block_node_map[block_hash] = node
    node_id = id(node)
    pod_state.active_node_ref_counts[node_id] = pod_state.active_node_ref_counts.get(node_id, 0) + 1
    if block_hash not in node.block_ids:
        node.block_ids.append(block_hash)
    node.location = location


def _unbind_block_from_radix_node(
    pod_state: PodCacheState,
    block_hash: str,
) -> None:
    """移除 block 到 Radix 节点的绑定并更新活跃节点引用计数。

    参数:
        pod_state: 当前 Worker 的 Router KV cache 状态。
        block_hash: 要解除绑定的 block hash。

    返回:
        无返回值；不存在的绑定按幂等操作处理。
    """

    node = pod_state.block_node_map.pop(block_hash, None)
    if node is None:
        return
    node.block_ids = [value for value in node.block_ids if value != block_hash]
    node_id = id(node)
    remaining = pod_state.active_node_ref_counts.get(node_id, 0) - 1
    if remaining > 0:
        pod_state.active_node_ref_counts[node_id] = remaining
    else:
        pod_state.active_node_ref_counts.pop(node_id, None)


def _rebuild_active_radix_node_refs(pod_state: PodCacheState) -> None:
    """从现有 block 绑定重建一次活跃 Radix 节点引用计数。

    参数:
        pod_state: 可能由快照恢复的 Worker 缓存状态。

    返回:
        无返回值。
    """

    counts: dict[int, int] = {}
    for block_hash, node in pod_state.block_node_map.items():
        if block_hash not in pod_state.cached_blocks:
            continue
        node_id = id(node)
        counts[node_id] = counts.get(node_id, 0) + 1
    pod_state.active_node_ref_counts = counts
    _invalidate_prefix_cursor_cache(pod_state)


def _invalidate_prefix_cursor_cache(pod_state: PodCacheState) -> None:
    """使 Worker 的公共前缀 Radix 游标缓存失效。

    参数:
        pod_state: 需要提升查询代次并清空游标缓存的 Worker 状态。

    返回:
        无返回值。
    """

    pod_state.radix_query_generation += 1
    pod_state.prefix_cursor_cache.clear()


def _set_radix_block_size(pod_state: PodCacheState, block_size: int) -> None:
    """记录 Worker 实际 KV block size，并在尺寸变化时失效查询缓存。

    参数:
        pod_state: 接收 Worker KV 事件的缓存状态。
        block_size: Worker 事件上报的每个物理 block token 数。

    返回:
        无返回值；非法尺寸会被忽略。
    """

    if block_size <= 0:
        return
    if pod_state.radix_block_size not in (None, block_size):
        _invalidate_prefix_cursor_cache(pod_state)
    pod_state.radix_block_size = block_size


def _advance_radix_cursor(
    pod_state: PodCacheState,
    start_node: RadixNode,
    token_chunk: tuple[int, ...],
) -> tuple[RadixNode | None, bool]:
    """沿一段 token 推进 Radix 游标，并缓存成功的公共前缀路径。

    参数:
        pod_state: 当前 Worker 的 KV cache 状态。
        start_node: 本段 token 查询开始前所在的 RadixTree 节点。
        token_chunk: 按 Worker block size 切分出的连续 token 元组。

    返回:
        二元组 ``(end_node, cache_hit)``。完整路径存在时 end_node 为末端
        节点；路径中断时为 None。cache_hit 表示本段是否直接复用了游标缓存。

    说明:
        缓存只保存树路径，不保存 location、物理 block ID 等可变元数据。
        RadixTree 追加新分支不会改变已有路径，因此普通 store 无需全量失效；
        clear、状态重建或 block size 变化会提升 generation。
    """

    cache_key = (
        pod_state.radix_query_generation,
        id(start_node),
        token_chunk,
    )
    cached = pod_state.prefix_cursor_cache.get(cache_key)
    if cached is not None:
        pod_state.prefix_cursor_cache.move_to_end(cache_key)
        return cached, True

    current = start_node
    for token_id in token_chunk:
        current = current.children.get(token_id)
        if current is None:
            return None, False

    pod_state.prefix_cursor_cache[cache_key] = current
    pod_state.prefix_cursor_cache.move_to_end(cache_key)
    while len(pod_state.prefix_cursor_cache) > _PREFIX_CURSOR_CACHE_MAX_ENTRIES:
        pod_state.prefix_cursor_cache.popitem(last=False)
    return current, False


def _select_node_block_hash(
    pod_state: PodCacheState,
    node: RadixNode,
    locations: set[CacheLocation] | None = None,
) -> str | None:
    """从一个逻辑 block 边界上的多个 hash 别名中选择有效代表。

    参数:
        pod_state: 当前 Worker 的 Router KV cache 状态。
        node: 当前请求前缀对应的 RadixTree 节点。
        locations: 可选位置过滤集合；为空时允许任意缓存层级。

    返回:
        最适合后续预取或跨 Worker 传输的 block hash；没有有效元数据时返回 None。

    说明:
        跨 Worker 发布后，vLLM 可能在同一 token 边界再次上报进程本地 hash。
        这些 hash 是同一个逻辑 block 的别名，匹配数量必须按节点而不是按 hash
        计算。优先选择 NPU、具有物理 ID 且最近更新的别名。
    """

    location_priority = {
        CacheLocation.NPU: 3,
        CacheLocation.CPU: 2,
        CacheLocation.REMOTE: 1,
    }
    candidates: list[tuple[tuple[int, int, float, int], str]] = []
    for insertion_index, block_hash in enumerate(node.block_ids):
        block_info = pod_state.cached_blocks.get(block_hash)
        if block_info is None:
            continue
        if locations is not None and block_info.location not in locations:
            continue
        priority = (
            location_priority.get(block_info.location, 0),
            int(block_info.physical_block_id is not None),
            block_info.last_accessed,
            insertion_index,
        )
        candidates.append((priority, block_hash))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


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

    def insert(self, token_ids: list[int], start_node: RadixNode | None = None) -> tuple[RadixNode, int]:
        """插入 token 序列，返回最终节点和新插入 token 数量（写操作）

        Args:
            token_ids: token ID 列表
            start_node: 起始节点（默认从 root 开始）

        Returns:
            (最终节点, 新插入的 token 数量)

        """
        self._rw_lock.acquire_write()
        try:
            current = start_node if start_node is not None else self.root
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
                        token_id=token_id,
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
            _rebuild_active_radix_node_refs(initial_state)
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

    def _handle_store_event(self, pod_state: PodCacheState, event: CacheEvent):
        """处理 store 事件"""
        block_size = int(event.engine_specific.get("block_size", self.block_size))
        _set_radix_block_size(pod_state, block_size)
        if event.token_ids and pod_state.radix_tree_root:
            radix_tree = RadixTree()
            radix_tree.root = pod_state.radix_tree_root
            radix_tree.insert(event.token_ids)
            pod_state.radix_tree_root = radix_tree.root

            current = radix_tree.root
            for i, token_id in enumerate(event.token_ids):
                current = current.children.get(token_id, current)
                block_idx = i // block_size
                if (i + 1) % block_size == 0 and block_idx < len(event.block_hashes):
                    block_hash = event.block_hashes[block_idx]
                    _bind_block_to_radix_node(
                        pod_state,
                        block_hash,
                        current,
                        event.location,
                    )

            remaining = len(event.token_ids) % block_size
            if remaining != 0:
                last_block_idx = len(event.token_ids) // block_size
                if last_block_idx < len(event.block_hashes):
                    block_hash = event.block_hashes[last_block_idx]
                    _bind_block_to_radix_node(
                        pod_state,
                        block_hash,
                        current,
                        event.location,
                    )

        for idx, block_hash in enumerate(event.block_hashes):
            token_count = block_size
            if idx == len(event.block_hashes) - 1 and event.token_ids:
                remainder = len(event.token_ids) % block_size
                token_count = remainder or block_size
            pod_state.cached_blocks[block_hash] = BlockInfo(
                block_hash=block_hash,
                token_count=token_count,
                location=event.location,
            )

    @staticmethod
    def _handle_evict_event(pod_state: PodCacheState, event: CacheEvent):
        """处理 evict 事件"""
        for block_hash in event.block_hashes:
            pod_state.cached_blocks.pop(block_hash, None)
            _unbind_block_from_radix_node(pod_state, block_hash)

    @staticmethod
    def _handle_hit_event(pod_state: PodCacheState, event: CacheEvent):
        """处理 hit 事件"""
        now = time.time()
        for block_hash in event.block_hashes:
            block_info = pod_state.cached_blocks.get(block_hash)
            if block_info:
                block_info.last_accessed = now
                block_info.ref_count += 1

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
            "removed": self._handle_evict_event,
            "evict": self._handle_evict_event,
            "hit": self._handle_hit_event,
        }

        handler = event_handlers.get(event.event_type)
        if handler:
            handler(pod_state, event)

        for block_hash in event.block_hashes:
            if event.event_type == "store":
                self.prefix_index.setdefault(block_hash, set()).add(pod_id)
            elif event.event_type in {"removed", "evict"} and block_hash in self.prefix_index:
                self.prefix_index[block_hash].discard(pod_id)
                if not self.prefix_index[block_hash]:
                    del self.prefix_index[block_hash]

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

    def score_vllm(self, pod_state: PodCacheState, token_ids: list[int]) -> float:
        """VLLM 评分：基于块级前缀匹配（已弃用，保留兼容）

        Args:
            pod_state: Pod 缓存状态
            token_ids: token ID 列表

        Returns:
            评分（0-1）

        """
        return self.score(pod_state, token_ids)

    def score_sglang(self, pod_state: PodCacheState, token_ids: list[int]) -> float:
        """SGLang 评分：基于 Radix Tree 最长前缀匹配（已弃用，保留兼容）

        Args:
            pod_state: Pod 缓存状态
            token_ids: token ID 列表

        Returns:
            评分（0-1）

        """
        return self.score(pod_state, token_ids)

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


@dataclass
class CacheStateUpdate:
    """缓存状态更新参数"""

    worker_id: str
    event_type: str
    block_hashes: list[str] = field(default_factory=list)
    token_count: int = 0
    location: str = "gpu"
    token_ids: list[int] = field(default_factory=list)
    engine_specific: dict[str, Any] = field(default_factory=dict)


def _worker_cache_synchronized(method):
    """只串行化同一 worker 的缓存状态读写，不阻塞其他 worker。"""

    @wraps(method)
    def synchronized(self, *args, **kwargs):
        subject = args[0] if args else kwargs.get("worker_id")
        if isinstance(subject, str):
            worker_id = subject
        else:
            worker_id = getattr(subject, "worker_id", "")
            if not worker_id and subject is not None:
                worker_id = getattr(subject, "engine_specific", {}).get("worker_id", "")

        pod_state = self.index.pod_states.get(worker_id)
        if pod_state is None:
            return method(self, *args, **kwargs)

        with pod_state.state_lock:
            return method(self, *args, **kwargs)

    return synchronized


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
        self.kvbm_adapter = None

        # 初始化事件管理器

        self.event_manager = event_manager or KVEventManager()

        # 注册事件处理器
        self._register_event_handlers()
        self._register_kvbm_adapter(settings)

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

    def _register_kvbm_adapter(self, settings):
        """注册独立 KVBM 管理面适配器。

        Args:
            settings: router 全局配置对象

        Returns:
            无返回值

        """
        if not getattr(settings, "kvbm_enabled", True):
            logger.info("KVBM adapter disabled by settings")
            return

        try:
            from openjiuwentools.infer_router.kvbm import RouterKVBMAdapter

            self.kvbm_adapter = RouterKVBMAdapter.from_settings(settings)
            self.kvbm_adapter.register_handlers(self.event_manager)
            logger.info("KVBM adapter registered")
        except Exception as exc:
            self.kvbm_adapter = None
            logger.warning(f"KVBM adapter not available: {exc}")

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
        self.event_manager.register_handler("remove", self._handle_removed_event)
        self.event_manager.register_handler("cleared", self._handle_cleared_event)
        self.event_manager.register_handler("clear", self._handle_cleared_event)
        self.event_manager.register_handler("evict", self._handle_evict_event)
        self.event_manager.register_handler("hit", self._handle_hit_event)
        self.event_manager.register_handler("move", self._handle_move_event)
        self.event_manager.register_handler("transfer", self._handle_move_event)
        logger.debug("Registered event handlers")

    @staticmethod
    def _worker_id_from_event(event: CacheEvent) -> str:
        return event.worker_id or event.engine_specific.get("worker_id", "")

    def _event_block_hashes(self, event: CacheEvent, block_size: int) -> list[str]:
        """Return event block hashes, deriving stable hashes when workers omit them."""
        block_size = int(block_size or self.block_size)
        if event.block_hashes:
            return event.block_hashes
        if not event.token_ids:
            return []

        block_hashes = []
        for start in range(0, len(event.token_ids), block_size):
            block_tokens = event.token_ids[start:start + block_size]
            block_hashes.append(self.scorer.compute_block_hash(block_tokens))
        return block_hashes

    @staticmethod
    def _event_physical_block_ids(event: CacheEvent) -> list[int]:
        """读取事件携带的物理 block id。

        参数:
            event: Worker 上报的缓存事件。

        返回:
            按 block_hashes 相同顺序排列的物理 block id；缺失时返回空列表。
        """

        raw_ids = event.engine_specific.get("physical_block_ids")
        if raw_ids is None:
            raw_id = event.engine_specific.get("physical_block_id")
            raw_ids = [] if raw_id is None else [raw_id]
        if not isinstance(raw_ids, list | tuple):
            raw_ids = [raw_ids]
        result: list[int] = []
        for value in raw_ids:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                continue
        return result

    @_worker_cache_synchronized
    def _handle_store_event(self, event):
        """处理 store 事件，利用 parent_block_hash 还原 block 链关系插入 RadixTree"""
        worker_id = self._worker_id_from_event(event)
        if not worker_id:
            return
        pod_state = self.index.pod_states.get(worker_id)
        if not pod_state:
            return

        pod_state.last_updated = time.time()
        block_size = int(event.engine_specific.get("block_size", self.block_size))
        _set_radix_block_size(pod_state, block_size)
        block_hashes = self._event_block_hashes(event, block_size)
        physical_block_ids = self._event_physical_block_ids(event)

        if pod_state.radix_tree_root and event.token_ids:
            radix_tree = RadixTree()
            radix_tree.root = pod_state.radix_tree_root

            parent_hash = event.engine_specific.get("parent_block_hash")
            start_node = pod_state.block_node_map.get(parent_hash) if parent_hash else None
            if start_node is None:
                start_node = radix_tree.root

            _, new_count = radix_tree.insert(event.token_ids, start_node=start_node)
            pod_state.radix_tree_root = radix_tree.root

            if block_hashes:
                current = start_node
                for i, tid in enumerate(event.token_ids):
                    current = current.children.get(tid, current)
                    block_idx = i // block_size
                    if (i + 1) % block_size == 0 and block_idx < len(block_hashes):
                        block_hash = block_hashes[block_idx]
                        _bind_block_to_radix_node(
                            pod_state,
                            block_hash,
                            current,
                            event.location,
                        )
                remaining = len(event.token_ids) % block_size
                if remaining != 0:
                    last_block_idx = len(event.token_ids) // block_size
                    if last_block_idx < len(block_hashes):
                        block_hash = block_hashes[last_block_idx]
                        _bind_block_to_radix_node(
                            pod_state,
                            block_hash,
                            current,
                            event.location,
                        )

            if new_count > 0:
                logger.debug(
                    f"[KV_CACHE] RadixTree updated for {worker_id}: "
                    f"+{new_count} tokens, {len(block_hashes)} blocks, "
                    f"parent={parent_hash is not None}"
                )

        for idx, block_hash in enumerate(block_hashes):
            token_count = block_size
            if idx == len(block_hashes) - 1 and event.token_ids:
                remainder = len(event.token_ids) % block_size
                token_count = remainder or block_size
            physical_block_id = physical_block_ids[idx] if idx < len(physical_block_ids) else None
            self._add_block(
                worker_id,
                block_hash,
                token_count,
                event.location,
                physical_block_id=physical_block_id,
            )

    @_worker_cache_synchronized
    def _handle_removed_event(self, event):
        """处理 removed 事件

        Args:
            event: 缓存事件

        """
        worker_id = self._worker_id_from_event(event)
        if not worker_id:
            return
        block_size = int(event.engine_specific.get("block_size", self.block_size))
        for block_hash in self._event_block_hashes(event, block_size):
            self._remove_block(worker_id, block_hash)

    @_worker_cache_synchronized
    def _handle_cleared_event(self, event):
        """处理 Cleared 事件

        Args:
            event: 缓存事件

        """
        worker_id = self._worker_id_from_event(event)
        if worker_id:
            # 清空工作器的缓存
            self._clear_worker_cache(worker_id)
            logger.info(f"[KV_CACHE] Cleared cache for worker {worker_id}")

    @_worker_cache_synchronized
    def _handle_evict_event(self, event):
        """处理 evict 事件

        Args:
            event: 缓存事件

        """
        self._handle_removed_event(event)

    @_worker_cache_synchronized
    def _handle_hit_event(self, event):
        """处理 hit 事件

        Args:
            event: 缓存事件

        """
        worker_id = self._worker_id_from_event(event)
        if not worker_id:
            return
        block_size = int(event.engine_specific.get("block_size", self.block_size))
        pod_state = self.index.pod_states.get(worker_id)
        if not pod_state:
            return

        now = time.time()
        pod_state.last_updated = now
        for block_hash in self._event_block_hashes(event, block_size):
            block_info = pod_state.cached_blocks.get(block_hash)
            if block_info:
                block_info.last_accessed = now
                block_info.ref_count += 1
            node = pod_state.block_node_map.get(block_hash)
            if node:
                node.last_accessed = now
                node.ref_count += 1

    @_worker_cache_synchronized
    def _handle_move_event(self, event):
        """处理缓存块在 NPU/CPU/REMOTE 等 tier 之间移动的事件。"""
        worker_id = self._worker_id_from_event(event)
        if not worker_id:
            return
        pod_state = self.index.pod_states.get(worker_id)
        if not pod_state:
            return

        pod_state.last_updated = time.time()
        block_size = int(event.engine_specific.get("block_size", self.block_size))
        physical_block_ids = self._event_physical_block_ids(event)
        for idx, block_hash in enumerate(self._event_block_hashes(event, block_size)):
            block_info = pod_state.cached_blocks.get(block_hash)
            if block_info:
                block_info.location = event.location
                block_info.last_accessed = pod_state.last_updated
                if idx < len(physical_block_ids):
                    block_info.physical_block_id = physical_block_ids[idx]
            node = pod_state.block_node_map.get(block_hash)
            if node:
                node.location = event.location

    @_worker_cache_synchronized
    def update_cache_state(self, update: CacheStateUpdate):
        """更新缓存状态

        Args:
            update: 缓存状态更新参数

        """
        event = CacheEvent(
            event_type=update.event_type,
            block_hashes=update.block_hashes,
            location=cache_location_from_medium(update.location),
            token_ids=update.token_ids,
            token_count=update.token_count,
            worker_id=update.worker_id,
            engine_specific={"worker_id": update.worker_id, **update.engine_specific},
        )

        self.event_manager.process_event(event)

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
        include_workers: set[str] | None = None,
        timing: dict[str, float] | None = None,
        snapshot_output: dict[str, WorkerPrefixSnapshot] | None = None,
    ) -> dict[str, int]:
        """查找所有工作器的 KV 缓存最长前缀匹配 token 数。

        结果存储为字典，键为工作器 ID，值为 matched token length。
        注意：这里返回的是 token 数，不是 block 数；调用方如需比例应除以输入 token 数。

        Args:
            token_ids: token ID 列表
            model: 模型名称
            exclude_workers: 排除的工作器 ID 集合
            include_workers: 仅参与匹配的工作器 ID 集合；为 None 时不限制
            timing: 可选耗时统计字典，函数会写入毫秒级统计
            snapshot_output: 可选输出字典；提供时写入本次匹配生成的 Worker
                前缀快照，供 scorer、PD direct 和 prefetch 复用。

        Returns:
            worker_id -> matched token length 的字典

        """
        exclude_workers = exclude_workers or set()

        candidate_workers = self.index.get_pods_for_model(model)
        candidate_workers = [w for w in candidate_workers if w not in exclude_workers]
        if include_workers is not None:
            candidate_workers = [w for w in candidate_workers if w in include_workers]

        # 如果 Radix Tree 功能未启用，返回所有 worker 的匹配长度为 0
        if not self.enable_radix_tree:
            return dict.fromkeys(candidate_workers, 0)

        overlap_scores: dict[str, int] = {}
        started = time.perf_counter()
        lock_wait_ms = 0.0
        cursor_cache_hits = 0.0
        cursor_cache_misses = 0.0

        for worker_id in candidate_workers:
            snapshot_timing: dict[str, float] = {}
            snapshot = self.get_worker_prefix_snapshot(
                worker_id,
                token_ids,
                timing=snapshot_timing,
            )
            overlap_scores[worker_id] = snapshot.matched_tokens
            lock_wait_ms += snapshot_timing.get("lock_wait_ms", 0.0)
            cursor_cache_hits += snapshot_timing.get("cursor_cache_hits", 0.0)
            cursor_cache_misses += snapshot_timing.get("cursor_cache_misses", 0.0)
            if snapshot_output is not None:
                snapshot_output[worker_id] = snapshot

        total_s = time.perf_counter() - started
        if timing is not None:
            timing.update(
                total_ms=total_s * 1000,
                lock_wait_ms=lock_wait_ms,
                compute_ms=max(total_s * 1000.0 - lock_wait_ms, 0.0),
                workers=float(len(candidate_workers)),
                cursor_cache_hits=cursor_cache_hits,
                cursor_cache_misses=cursor_cache_misses,
            )

        return overlap_scores

    def get_worker_prefix_snapshot(
        self,
        worker_id: str,
        token_ids: list[int],
        *,
        timing: dict[str, float] | None = None,
    ) -> WorkerPrefixSnapshot:
        """按 block 游标构建 Worker 对当前请求的连续前缀快照。

        参数:
            worker_id: 待查询的 Worker ID。
            token_ids: 当前请求的完整 prompt token ID 列表。
            timing: 可选耗时统计字典，写入锁等待、游标缓存命中和总耗时。

        返回:
            包含逻辑 block、各位置 hash 别名及物理 ID 的只读快照；Worker
            不存在、RadixTree 未启用或无匹配时返回 blocks 为空的快照。
        """

        started = time.perf_counter()
        pod_state = self.index.pod_states.get(worker_id)
        if not self.enable_radix_tree or pod_state is None:
            return WorkerPrefixSnapshot(worker_id, "", 0, ())
        wait_started = time.perf_counter()
        with pod_state.state_lock:
            lock_acquired = time.perf_counter()
            current = pod_state.radix_tree_root
            if current is None:
                return WorkerPrefixSnapshot(worker_id, pod_state.model_id, 0, ())
            location_priority = {
                CacheLocation.NPU: 3,
                CacheLocation.CPU: 2,
                CacheLocation.REMOTE: 1,
            }
            blocks: list[PrefixBlockSnapshot] = []
            matched_tokens = 0
            cursor_cache_hits = 0
            cursor_cache_misses = 0
            block_size = max(pod_state.radix_block_size or self.block_size, 1)
            token_end = 0
            for chunk_start in range(0, len(token_ids), block_size):
                token_chunk = tuple(token_ids[chunk_start:chunk_start + block_size])
                current, cache_hit = _advance_radix_cursor(
                    pod_state,
                    current,
                    token_chunk,
                )
                if cache_hit:
                    cursor_cache_hits += 1
                else:
                    cursor_cache_misses += 1
                if current is None:
                    break
                token_end += len(token_chunk)
                if pod_state.active_node_ref_counts.get(id(current), 0) <= 0:
                    continue
                aliases: list[PrefixBlockAliasSnapshot] = []
                for insertion_index, block_hash in enumerate(current.block_ids):
                    block_info = pod_state.cached_blocks.get(block_hash)
                    if block_info is None:
                        continue
                    priority = (
                        location_priority.get(block_info.location, 0),
                        int(block_info.physical_block_id is not None),
                        block_info.last_accessed,
                        insertion_index,
                    )
                    aliases.append(
                        PrefixBlockAliasSnapshot(
                            block_hash=block_hash,
                            location=block_info.location,
                            physical_block_id=block_info.physical_block_id,
                            token_count=block_info.token_count,
                            priority=priority,
                        )
                    )
                if aliases:
                    aliases.sort(key=lambda alias: alias.priority, reverse=True)
                    blocks.append(
                        PrefixBlockSnapshot(
                            token_end=token_end,
                            aliases=tuple(aliases),
                        )
                    )
                    matched_tokens = token_end
            snapshot = WorkerPrefixSnapshot(
                worker_id=worker_id,
                model_id=pod_state.model_id,
                matched_tokens=matched_tokens,
                blocks=tuple(blocks),
            )
        if timing is not None:
            timing.update(
                lock_wait_ms=max((lock_acquired - wait_started) * 1000.0, 0.0),
                total_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
                cursor_cache_hits=float(cursor_cache_hits),
                cursor_cache_misses=float(cursor_cache_misses),
            )
        return snapshot

    def get_worker_block_size(self, worker_id: str) -> int:
        """读取指定 Worker 自动上报的 KV block token 大小。

        参数:
            worker_id: 需要查询的 Worker ID。

        返回:
            Worker 已上报时返回真实 block size；尚无事件时返回 Router 兼容兜底值。
        """
        pod_state = self.index.pod_states.get(worker_id)
        if pod_state is None:
            return max(int(self.block_size), 1)
        with pod_state.state_lock:
            return max(int(pod_state.radix_block_size or self.block_size), 1)

    def get_matched_block_hashes(
        self,
        worker_id: str,
        token_ids: list[int],
        limit: int | None = None,
        locations: set[CacheLocation] | None = None,
        snapshot: WorkerPrefixSnapshot | None = None,
    ) -> list[str]:
        """返回指定 worker 对本次请求命中的连续前缀 block hash。

        参数:
            worker_id: 目标 worker ID。
            token_ids: 本次请求的 token ID 列表。
            limit: 最多返回的 block hash 数量；为 None 时不限制。
            locations: 可选目标位置集合；为 None 时返回所有位置的命中 block。
            snapshot: 可选请求级前缀快照；提供时不再遍历 RadixTree。

        返回:
            按前缀顺序排列的 block hash 列表；无命中或未启用 RadixTree 时返回空列表。
        """

        resolved = snapshot or self.get_worker_prefix_snapshot(worker_id, token_ids)
        return resolved.block_hashes(limit=limit, locations=locations)

    def get_matched_block_location_counts(
        self,
        worker_id: str,
        token_ids: list[int],
        snapshot: WorkerPrefixSnapshot | None = None,
    ) -> dict[CacheLocation, int]:
        """统计指定 worker 对本次请求命中的连续前缀 block 所在层级。

        参数:
            worker_id: 目标 worker ID。
            token_ids: 本次请求的 token ID 列表。
            snapshot: 可选请求级前缀快照；提供时不再遍历 RadixTree。

        返回:
            以 CacheLocation 为 key 的命中 block 数量字典；无命中或未启用 RadixTree 时返回空字典。
        """

        resolved = snapshot or self.get_worker_prefix_snapshot(worker_id, token_ids)
        return resolved.location_counts()

    def get_matched_block_location_token_counts(
        self,
        worker_id: str,
        token_ids: list[int],
        snapshot: WorkerPrefixSnapshot | None = None,
    ) -> dict[CacheLocation, int]:
        """统计指定 Worker 连续前缀在各缓存层级命中的 token 数。

        参数:
            worker_id: 目标 Worker ID。
            token_ids: 当前请求的完整 prompt token ID 列表。
            snapshot: 可选请求级前缀快照；提供时不再遍历 RadixTree。

        返回:
            CacheLocation 到命中 token 数的映射；同一逻辑 block 的多个 hash
            别名只统计一次，总命中数不会超过请求 token 数。
        """

        resolved = snapshot or self.get_worker_prefix_snapshot(worker_id, token_ids)
        token_counts: dict[CacheLocation, int] = {}
        counted_tokens = 0
        for block in resolved.blocks:
            alias = block.select()
            if alias is None:
                continue
            remaining = max(0, len(token_ids) - counted_tokens)
            effective_tokens = min(max(int(alias.token_count), 0), remaining)
            if effective_tokens <= 0:
                continue
            token_counts[alias.location] = token_counts.get(alias.location, 0) + effective_tokens
            counted_tokens += effective_tokens
        return token_counts

    def plan_cross_worker_transfer(
        self,
        target_worker_id: str,
        token_ids: list[int],
        *,
        candidate_worker_ids: list[str] | None = None,
        limit: int = 8,
    ) -> CrossWorkerTransferPlan:
        """一次扫描生成跨 Worker 传输候选、诊断和目标前缀快照。

        参数:
            target_worker_id: 即将接收推理请求的目标 Worker ID。
            token_ids: 当前请求的 prompt token ID 列表。
            candidate_worker_ids: 可作为源端的健康 Worker ID；为空时检查所有同模型 Worker。
            limit: 单次最多传输的 block 数量。

        返回:
            共享同一批 WorkerPrefixSnapshot 的传输规划结果。候选为空时，
            diagnostics 已包含跳过原因，调用方不应再次扫描 RadixTree。

        说明:
            每个 Worker 的 token 路径只遍历一次，同时解析 hash、location 和
            physical block ID。第一版仍只从单个源 Worker 的 NPU 拉取。
        """

        target_snapshot = self.get_worker_prefix_snapshot(target_worker_id, token_ids)
        target_state = self.index.pod_states.get(target_worker_id)
        eligible_ids = set(candidate_worker_ids or self.index.pod_states)
        diagnostics: dict[str, Any] = {
            "radix_tree_enabled": self.enable_radix_tree,
            "limit": limit,
            "target_registered": target_state is not None,
            "target_model": target_snapshot.model_id,
            "target_matched_blocks": len(target_snapshot.blocks),
            "target_npu_blocks": sum(block.select({CacheLocation.NPU}) is not None for block in target_snapshot.blocks),
            "sources": [],
        }
        if not self.enable_radix_tree:
            diagnostics["blocker"] = "radix_tree_disabled"
            return CrossWorkerTransferPlan(None, diagnostics, target_snapshot)
        if limit <= 0:
            diagnostics["blocker"] = "transfer_limit_zero"
            return CrossWorkerTransferPlan(None, diagnostics, target_snapshot)
        if target_state is None:
            diagnostics["blocker"] = "target_not_registered"
            return CrossWorkerTransferPlan(None, diagnostics, target_snapshot)

        best: CrossWorkerTransferCandidate | None = None
        for source_worker_id, source_state in self.index.pod_states.items():
            if source_worker_id == target_worker_id:
                continue
            eligible = source_worker_id in eligible_ids
            same_model = source_state.model_id == target_state.model_id
            source_snapshot = (
                self.get_worker_prefix_snapshot(source_worker_id, token_ids)
                if eligible and same_model
                else WorkerPrefixSnapshot(
                    source_worker_id,
                    source_state.model_id,
                    0,
                    (),
                )
            )
            transferable_hashes: list[str] = []
            source_block_ids: list[int] = []
            source_npu_aliases = [block.select({CacheLocation.NPU}) for block in source_snapshot.blocks]
            source_npu_blocks = sum(alias is not None for alias in source_npu_aliases)
            physical_mapped_npu_blocks = sum(
                alias is not None and alias.physical_block_id is not None for alias in source_npu_aliases
            )
            first_unselected_reason = ""
            for block_index, source_block in enumerate(source_snapshot.blocks):
                source_npu_alias = source_npu_aliases[block_index]

                if block_index < len(target_snapshot.blocks):
                    # 两端按请求前缀位置对齐，不依赖跨进程 hash 相等。
                    if target_snapshot.blocks[block_index].select({CacheLocation.NPU}) is None:
                        first_unselected_reason = "target_local_non_npu"
                        break
                    continue
                if source_npu_alias is None:
                    preferred_alias = source_block.select()
                    first_unselected_reason = (
                        "source_metadata_missing" if preferred_alias is None else "source_block_not_npu"
                    )
                    break
                if source_npu_alias.physical_block_id is None:
                    first_unselected_reason = "missing_physical_block_id"
                    break
                transferable_hashes.append(source_npu_alias.block_hash)
                source_block_ids.append(source_npu_alias.physical_block_id)
                if len(transferable_hashes) >= limit:
                    first_unselected_reason = "transfer_limit_reached"
                    break
            if not first_unselected_reason:
                first_unselected_reason = (
                    "source_prefix_exhausted" if transferable_hashes else "target_already_has_prefix"
                )
            diagnostic_blocker = first_unselected_reason
            if not eligible:
                diagnostic_blocker = "worker_not_healthy"
            elif not same_model:
                diagnostic_blocker = "model_mismatch"
            elif not source_snapshot.blocks:
                diagnostic_blocker = "source_prefix_miss"
            diagnostics["sources"].append(
                {
                    "worker_id": source_worker_id,
                    "eligible": eligible,
                    "same_model": same_model,
                    "matched_blocks": len(source_snapshot.blocks),
                    "npu_blocks": source_npu_blocks,
                    "physical_mapped_npu_blocks": physical_mapped_npu_blocks,
                    "transferable_blocks": len(transferable_hashes),
                    "first_blocker": diagnostic_blocker,
                }
            )
            candidate = CrossWorkerTransferCandidate(
                source_worker_id=source_worker_id,
                block_hashes=tuple(transferable_hashes),
                source_block_ids=tuple(source_block_ids),
                source_matched_blocks=len(source_snapshot.blocks),
                source_npu_blocks=source_npu_blocks,
                physical_mapped_npu_blocks=physical_mapped_npu_blocks,
                target_matched_blocks=len(target_snapshot.blocks),
                target_npu_blocks=diagnostics["target_npu_blocks"],
                first_unselected_reason=first_unselected_reason,
            )
            if eligible and same_model and candidate.block_hashes:
                if best is None or len(candidate.block_hashes) > len(best.block_hashes):
                    best = candidate
        if not diagnostics["sources"]:
            diagnostics["blocker"] = "no_other_worker_state"
        return CrossWorkerTransferPlan(best, diagnostics, target_snapshot)

    def get_cross_worker_transfer_candidate(
        self,
        target_worker_id: str,
        token_ids: list[int],
        *,
        candidate_worker_ids: list[str] | None = None,
        limit: int = 8,
    ) -> CrossWorkerTransferCandidate | None:
        """选择能够补齐目标 Worker 缺失前缀的最佳 NPU 源 Worker。

        参数:
            target_worker_id: 即将接收推理请求的目标 Worker ID。
            token_ids: 当前请求的 prompt token ID 列表。
            candidate_worker_ids: 可作为源端的健康 Worker ID。
            limit: 单次最多传输的 block 数量。

        返回:
            最佳传输候选；无可传输 block 时返回 None。
        """

        return self.plan_cross_worker_transfer(
            target_worker_id,
            token_ids,
            candidate_worker_ids=candidate_worker_ids,
            limit=limit,
        ).candidate

    def record_cross_worker_transfer(
        self,
        source_worker_id: str,
        target_worker_id: str,
        block_hashes: list[str] | tuple[str, ...],
        destination_block_ids: list[int] | tuple[int, ...],
        *,
        timing: dict[str, float] | None = None,
    ) -> int:
        """把已发布的跨 Worker block 立即提交到 Router 前缀索引。

        参数:
            source_worker_id: 提供 block 与完整 token 路径的源 Worker ID。
            target_worker_id: 已成功发布目标 block 的 Worker ID。
            block_hashes: 按传输顺序排列的稳定 block hash。
            destination_block_ids: 与 block_hashes 对齐的目标物理 block ID。
            timing: 可选耗时输出字典，用于记录 Router 索引锁等待和更新耗时。

        返回:
            成功写入目标 Worker RadixTree 的 block 数量；状态缺失或参数不对齐时
            返回 0，并等待 Worker location event 最终收敛。
        """

        phase_ms = timing if timing is not None else {}
        phase_ms.setdefault("router_index_lock_wait_ms", 0.0)
        phase_ms.setdefault("router_index_update_ms", 0.0)
        if not self.enable_radix_tree or source_worker_id == target_worker_id or not block_hashes:
            return 0
        if len(block_hashes) != len(destination_block_ids):
            return 0
        source_state = self.index.pod_states.get(source_worker_id)
        target_state = self.index.pod_states.get(target_worker_id)
        if source_state is None or target_state is None or source_state.model_id != target_state.model_id:
            return 0

        first_state, second_state = sorted(
            (source_state, target_state),
            key=lambda state: state.pod_id,
        )
        lock_started = time.perf_counter()
        with first_state.state_lock:
            with second_state.state_lock:
                update_started = time.perf_counter()
                phase_ms["router_index_lock_wait_ms"] = max(
                    (update_started - lock_started) * 1000.0,
                    0.0,
                )
                radix_tree = RadixTree()
                radix_tree.root = target_state.radix_tree_root
                updated = 0
                now = time.time()
                transferable = []
                for block_hash, destination_block_id in zip(
                    block_hashes,
                    destination_block_ids,
                    strict=True,
                ):
                    source_node = source_state.block_node_map.get(block_hash)
                    source_info = source_state.cached_blocks.get(block_hash)
                    if source_node is None or source_info is None:
                        continue
                    transferable.append((block_hash, int(destination_block_id), source_node, source_info))

                # 候选 block 按请求前缀顺序排列，最后一个节点应是此前节点的
                # 后代。只重放一次最长 token 路径，再按深度绑定每个 block，
                # 避免 40 个 block 对 6000 token 前缀执行 40 次根节点遍历。
                target_nodes_by_source_id: dict[int, RadixNode] = {}
                if transferable:
                    deepest_source_node = transferable[-1][2]
                    source_path_nodes = []
                    current_source = deepest_source_node
                    while current_source.parent is not None:
                        source_path_nodes.append(current_source)
                        current_source = current_source.parent
                    source_path_nodes.reverse()
                    longest_token_path = [node.token_id for node in source_path_nodes]
                    radix_tree.insert(longest_token_path)

                    current_target = radix_tree.root
                    for source_path_node in source_path_nodes:
                        current_target = current_target.children[source_path_node.token_id]
                        target_nodes_by_source_id[id(source_path_node)] = current_target

                for block_hash, destination_block_id, source_node, source_info in transferable:
                    target_node = target_nodes_by_source_id.get(id(source_node))
                    if target_node is None:
                        # 非连续或异常别名不应影响正常前缀批次，保留兼容路径。
                        target_node, _new_count = radix_tree.insert(source_node.get_token_ids())
                    _bind_block_to_radix_node(
                        target_state,
                        block_hash,
                        target_node,
                        CacheLocation.NPU,
                    )
                    target_info = target_state.cached_blocks.get(block_hash)
                    if target_info is None:
                        target_info = BlockInfo(
                            block_hash=block_hash,
                            token_count=source_info.token_count,
                            location=CacheLocation.NPU,
                            physical_block_id=destination_block_id,
                        )
                        target_state.cached_blocks[block_hash] = target_info
                    else:
                        target_info.token_count = source_info.token_count
                        target_info.location = CacheLocation.NPU
                        target_info.physical_block_id = destination_block_id
                        target_info.last_accessed = now
                    self.index.prefix_index.setdefault(block_hash, set()).add(target_worker_id)
                    updated += 1
                target_state.last_updated = now
                phase_ms["router_index_update_ms"] = max(
                    (time.perf_counter() - update_started) * 1000.0,
                    0.0,
                )
                return updated

    @_worker_cache_synchronized
    def record_pd_import(
        self,
        target_worker_id: str,
        block_hashes: list[str] | tuple[str, ...],
        destination_block_ids: list[int] | tuple[int, ...],
        token_ids: list[int] | tuple[int, ...],
        *,
        block_size: int | None = None,
    ) -> int:
        """把 D Worker 已 attach 的完整前缀 block 立即发布到 Router 索引。

        参数:
            target_worker_id: 完成 PD import 的目标 D Worker ID。
            block_hashes: 按 prompt 顺序排列的完整、可复用 block hash；不应包含
                request-scoped partial block。
            destination_block_ids: 与 ``block_hashes`` 一一对应的 D Worker 物理
                block ID。
            token_ids: 当前请求的完整 prompt token ID，用于恢复 RadixTree 路径。
            block_size: D Worker 每个 KV block 对应的 token 数；未指定时使用
                Worker 已上报值或 Router 默认值。

        返回:
            成功写入或幂等确认的完整 block 数量；参数不完整、Worker 未注册或
            RadixTree 未启用时返回 0。
        """

        if not self.enable_radix_tree or not block_hashes or len(block_hashes) != len(destination_block_ids):
            return 0
        pod_state = self.index.pod_states.get(target_worker_id)
        if pod_state is None:
            return 0

        resolved_block_size = int(block_size or pod_state.radix_block_size or self.block_size)
        required_tokens = len(block_hashes) * resolved_block_size
        if resolved_block_size <= 0 or len(token_ids) < required_tokens:
            return 0

        _set_radix_block_size(pod_state, resolved_block_size)
        radix_tree = RadixTree()
        radix_tree.root = pod_state.radix_tree_root
        prefix_tokens = list(token_ids[:required_tokens])
        radix_tree.insert(prefix_tokens)

        current = radix_tree.root
        now = time.time()
        for token_position, token_id in enumerate(prefix_tokens, start=1):
            current = current.children[token_id]
            if token_position % resolved_block_size != 0:
                continue
            block_index = token_position // resolved_block_size - 1
            block_hash = str(block_hashes[block_index])
            physical_block_id = int(destination_block_ids[block_index])
            _bind_block_to_radix_node(
                pod_state,
                block_hash,
                current,
                CacheLocation.NPU,
            )
            block_info = pod_state.cached_blocks.get(block_hash)
            if block_info is None:
                block_info = BlockInfo(
                    block_hash=block_hash,
                    token_count=resolved_block_size,
                    location=CacheLocation.NPU,
                    physical_block_id=physical_block_id,
                )
                pod_state.cached_blocks[block_hash] = block_info
            else:
                block_info.token_count = resolved_block_size
                block_info.location = CacheLocation.NPU
                block_info.physical_block_id = physical_block_id
                block_info.last_accessed = now
            self.index.prefix_index.setdefault(block_hash, set()).add(target_worker_id)

        pod_state.radix_tree_root = radix_tree.root
        pod_state.last_updated = now
        return len(block_hashes)

    def diagnose_cross_worker_transfer(
        self,
        target_worker_id: str,
        token_ids: list[int],
        *,
        candidate_worker_ids: list[str] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        """解释跨 Worker 传输候选为空的原因。

        参数:
            target_worker_id: Router 已选定的目标 Worker ID。
            token_ids: 当前请求的 prompt token ID 列表。
            candidate_worker_ids: 健康检查后允许作为源端的 Worker ID。
            limit: 单次传输 block 上限。

        返回:
            包含目标状态和逐源 Worker 匹配、位置、物理映射及首个阻塞原因的字典。
        """

        return self.plan_cross_worker_transfer(
            target_worker_id,
            token_ids,
            candidate_worker_ids=candidate_worker_ids,
            limit=limit,
        ).diagnostics

    @staticmethod
    def _active_matched_len(pod_state: PodCacheState, token_ids: list[int], matched_len: int) -> int:
        """Clamp a radix match to the deepest prefix backed by live block metadata."""
        if not pod_state.active_node_ref_counts:
            return 0

        current = pod_state.radix_tree_root
        active_len = 0
        for idx, token_id in enumerate(token_ids[:matched_len], start=1):
            if current is None or token_id not in current.children:
                break
            current = current.children[token_id]
            if pod_state.active_node_ref_counts.get(id(current), 0) > 0:
                active_len = idx
        return active_len

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
            block_tokens = token_ids[:self.block_size]

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

        if self.kvbm_adapter:
            stats["kvbm_stats"] = self.kvbm_adapter.get_stats()

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
            _unbind_block_from_radix_node(pod_state, block_hash)
            if block_hash in self.index.prefix_index:
                self.index.prefix_index[block_hash].discard(worker_id)
                if not self.index.prefix_index[block_hash]:
                    del self.index.prefix_index[block_hash]

        if expired_blocks:
            logger.info(f"[KV_CACHE] Cleaned {len(expired_blocks)} expired blocks from worker {worker_id}")

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
                _unbind_block_from_radix_node(pod_state, block_hash)
                if block_hash in self.index.prefix_index:
                    self.index.prefix_index[block_hash].discard(worker_id)
                    if not self.index.prefix_index[block_hash]:
                        del self.index.prefix_index[block_hash]
                trimmed += 1

        if trimmed > 0:
            logger.info(
                f"[KV_CACHE] Trimmed {trimmed} blocks from worker {worker_id}, new size: {len(pod_state.cached_blocks)}"
            )

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

    @_worker_cache_synchronized
    def _add_block(
        self,
        worker_id: str,
        block_hash: str,
        token_count: int,
        location: CacheLocation = CacheLocation.NPU,
        *,
        physical_block_id: int | None = None,
    ) -> None:
        """添加缓存块

        Args:
            worker_id: 工作器 ID
            block_hash: 块哈希
            token_count: token 数量
            location: 缓存所在层级。
            physical_block_id: Worker 内部 vLLM 物理 block 编号；未知时为 None。

        Returns:
            无返回值。

        """
        pod_state = self.index.pod_states.get(worker_id)
        if pod_state:
            block_info = pod_state.cached_blocks.get(block_hash)
            if block_info:
                block_info.token_count = token_count
                block_info.location = location
                if physical_block_id is not None:
                    block_info.physical_block_id = physical_block_id
                block_info.last_accessed = time.time()
                block_info.ref_count += 1
            else:
                block_info = BlockInfo(
                    block_hash=block_hash,
                    token_count=token_count,
                    location=location,
                    physical_block_id=physical_block_id,
                )
                pod_state.cached_blocks[block_hash] = block_info
            # 更新前缀索引
            if block_hash not in self.index.prefix_index:
                self.index.prefix_index[block_hash] = set()
            self.index.prefix_index[block_hash].add(worker_id)

    @_worker_cache_synchronized
    def _remove_block(self, worker_id: str, block_hash: str) -> None:
        """移除缓存块

        Args:
            worker_id: 工作器 ID
            block_hash: 块哈希

        """
        pod_state = self.index.pod_states.get(worker_id)
        if pod_state:
            pod_state.cached_blocks.pop(block_hash, None)
            _unbind_block_from_radix_node(pod_state, block_hash)
            # 更新前缀索引
            if block_hash in self.index.prefix_index:
                self.index.prefix_index[block_hash].discard(worker_id)
                if not self.index.prefix_index[block_hash]:
                    del self.index.prefix_index[block_hash]

    @_worker_cache_synchronized
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
            pod_state.block_node_map.clear()
            pod_state.active_node_ref_counts.clear()
            pod_state.radix_tree_root = RadixTree().root
            _invalidate_prefix_cursor_cache(pod_state)

    def _update_block_access_time(self, worker_id: str, block_hash: str) -> None:
        """更新块的访问时间

        Args:
            worker_id: 工作器 ID
            block_hash: 块哈希

        """
        pod_state = self.index.pod_states.get(worker_id)
        if pod_state and block_hash in pod_state.cached_blocks:
            pod_state.cached_blocks[block_hash].last_accessed = time.time()
