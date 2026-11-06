import hashlib
import json
import os
import time
from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any

from loguru import logger

from openjiuwentools.infer_router.config.config import settings

_tokenizers: dict[str, Any] = {}
_tokenizer_lock = Lock()


@dataclass(frozen=True)
class _ChatPrefixCacheEntry:
    """一段经过chat template处理的安全消息前缀。"""

    rendered_text: str
    token_ids: tuple[int, ...]
    last_role: str
    terminal_special_token: bool


@dataclass(frozen=True)
class _CompletionPrefixCacheEntry:
    """一条可用于Completion增量分词的完整文本记录。"""

    prompt: str
    token_ids: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]
    token_end_offsets: tuple[int, ...]
    bucket_key: tuple[str, str, str]


class TokenizerManager:
    """Tokenizer管理器，与vllm/sglang兼容

    使用HuggingFace transformers的AutoTokenizer来处理tokenize，
    确保与vllm/sglang工作器生成的token id序列一致。
    """

    def __init__(self, load_from_file: bool = False, local_dir: str | None = None):
        self._tokenizers: dict[str, Any] = {}
        self._lock = Lock()
        self._load_from_file = load_from_file
        self._local_dir = local_dir or settings.tokenizer_local_dir
        self._prefix_cache_enabled = bool(getattr(settings, "tokenizer_prefix_cache_enabled", True))
        self._prefix_cache_max_entries = max(
            1,
            int(getattr(settings, "tokenizer_prefix_cache_max_entries", 2048)),
        )
        self._prefix_cache_max_tokens = max(
            1,
            int(getattr(settings, "tokenizer_prefix_cache_max_tokens", 2_000_000)),
        )
        self._chat_prefix_cache: OrderedDict[
            tuple[str, str, tuple[str, ...]],
            _ChatPrefixCacheEntry,
        ] = OrderedDict()
        self._chat_full_cache: OrderedDict[
            tuple[str, str, tuple[str, ...]],
            tuple[int, ...],
        ] = OrderedDict()
        self._completion_prefix_cache: OrderedDict[
            tuple[str, str, str],
            _CompletionPrefixCacheEntry,
        ] = OrderedDict()
        self._completion_prefix_buckets: dict[
            tuple[str, str, str],
            OrderedDict[tuple[str, str, str], None],
        ] = {}
        self._boundary_validation: dict[tuple[str, str, str, str], bool] = {}
        self._completion_boundary_validation: dict[tuple[str, str, str], bool] = {}
        self._tokenizer_namespace_cache: dict[int, tuple[Any, str]] = {}
        self._prefix_cache_tokens = 0

        # Completion前缀按文本开头分桶，每个桶只比较少量最近候选，
        # 避免缓存规模增长后退化为线性扫描。
        self._completion_bucket_chars = 256
        self._completion_bucket_candidates = 4
        self._completion_prefix_min_tokens = 128
        self._completion_boundary_overlap_tokens = 16

    def preload_models(self, models: list[str] | tuple[str, ...] | set[str]) -> dict[str, bool]:
        """在 Router 开始监听前加载并预热指定模型的 tokenizer。

        Args:
            models: 需要预加载的模型名称或路径集合；重复和空值会被忽略。

        Returns:
            模型名称到预加载结果的映射。加载和短文本编码均成功时值为 True；
            未启用文件 tokenizer 或加载失败时值为 False。
        """

        results: dict[str, bool] = {}
        for model in sorted({str(item) for item in models if str(item)}):
            total_started = time.perf_counter()
            tokenizer_get_started = time.perf_counter()
            tokenizer = self._get_tokenizer(model)
            tokenizer_get_ms = (time.perf_counter() - tokenizer_get_started) * 1000.0
            warmup_started = time.perf_counter()
            succeeded = tokenizer is not None
            if tokenizer is not None:
                try:
                    tokenizer.encode("warmup", add_special_tokens=False)
                    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                        self._tokenize_with_chat_template(
                            tokenizer,
                            [{"role": "user", "content": "warmup"}],
                        )
                except Exception as exc:
                    succeeded = False
                    logger.warning(
                        "Tokenizer warm-up failed for model {}: {}",
                        model,
                        exc,
                    )
            finished = time.perf_counter()
            results[model] = succeeded
            logger.info(
                "[TokenizerPreload] model={} tokenizer_get={:.2f}ms warmup={:.2f}ms total={:.2f}ms ready={}",
                model,
                tokenizer_get_ms,
                max((finished - warmup_started) * 1000.0, 0.0),
                max((finished - total_started) * 1000.0, 0.0),
                succeeded,
            )
        return results

    def get_cached_tokenizer(self, model: str) -> Any:
        """Return an already loaded tokenizer without triggering a model load."""
        return self._tokenizers.get(model)

    @staticmethod
    def format_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Expose the same message formatting used by tokenization diagnostics."""
        return TokenizerManager._format_chat_messages(messages)

    def _get_tokenizer(self, model: str) -> Any:
        """获取或加载tokenizer

        Args:
            model: 模型名称或路径

        Returns:
            tokenizer实例

        """
        # 默认不加载tokenizer，使用fallback
        if not self._load_from_file:
            return None

        with self._lock:
            if model in self._tokenizers:
                return self._tokenizers[model]

            try:
                from transformers import AutoTokenizer

                tokenizer_path = self._resolve_model_path(model)

                logger.info(f"Loading tokenizer for model: {tokenizer_path}")

                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True,
                    use_fast=True,
                )

                self._tokenizers[model] = tokenizer
                logger.info(f"Tokenizer loaded successfully for model: {model}")
                return tokenizer

            except Exception as e:
                logger.warning(f"Failed to load tokenizer for model {model}: {e}")
                return None

    def _resolve_model_path(self, model: str) -> str:
        """解析模型路径

        支持以下优先级：
        1. 如果是绝对路径且存在，直接返回
        2. 如果配置了本地目录，尝试在本地目录中查找
        3. 检查相对路径是否存在
        4. 返回原始模型名称（用于HuggingFace下载）

        Args:
            model: 模型名称或路径

        Returns:
            解析后的模型路径

        """
        # 如果是绝对路径且存在，直接返回
        if os.path.isabs(model) and os.path.exists(model):
            return model

        # 如果配置了本地目录，尝试在本地目录中查找
        if self._local_dir:
            local_path = os.path.join(self._local_dir, model)
            if os.path.exists(local_path):
                logger.info(f"Found tokenizer in local directory: {local_path}")
                return local_path

        # 检查相对路径是否存在
        if os.path.exists(model):
            return model

        # 检查相对路径拼接本地目录
        if self._local_dir:
            # 尝试多种命名格式
            for suffix in ["", "-tokenizer", "_tokenizer", "/tokenizer"]:
                local_path = os.path.join(self._local_dir, model + suffix)
                if os.path.exists(local_path):
                    logger.info(f"Found tokenizer in local directory: {local_path}")
                    return local_path

        # 检查环境变量中的HF token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            logger.debug(f"Using HF token for model: {model}")

        # 返回原始模型名称，用于HuggingFace下载
        return model

    @staticmethod
    def _format_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """保留 Worker 使用的完整 Chat 消息结构，并隔离模板可能的修改。

        Args:
            messages: 原始OpenAI消息列表。

        Returns:
            完整消息副本，包括 tool_calls、tool_call_id、name 和 content:null。
        """

        return deepcopy(messages)

    @staticmethod
    def _supports_text_prefix_cache(messages: list[dict[str, Any]]) -> bool:
        """只为已验证的普通文本消息启用增量模板边界缓存。"""
        return all(
            set(message) <= {"role", "content"}
            and message.get("role") in {"system", "user", "assistant"}
            and isinstance(message.get("content"), str)
            for message in messages
        )

    @staticmethod
    def _chat_cache_namespace(tokenizer: Any) -> str:
        """构造可感知tokenizer和chat template变化的缓存命名空间。

        Args:
            tokenizer: 当前模型的HuggingFace tokenizer。

        Returns:
            tokenizer路径、模板和特殊token配置的稳定摘要。
        """

        payload = json.dumps(
            {
                "name_or_path": getattr(tokenizer, "name_or_path", ""),
                "chat_template": getattr(tokenizer, "chat_template", ""),
                "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
            },
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def _get_tokenizer_namespace(self, tokenizer: Any) -> str:
        """读取或计算tokenizer与chat template配置命名空间。

        Args:
            tokenizer: 当前模型使用的HuggingFace tokenizer对象。

        Returns:
            可感知tokenizer配置变化的稳定摘要。
        """

        cache_key = id(tokenizer)
        with self._lock:
            cached = self._tokenizer_namespace_cache.get(cache_key)
            if cached is not None and cached[0] is tokenizer:
                return cached[1]
        namespace = self._chat_cache_namespace(tokenizer)
        with self._lock:
            self._tokenizer_namespace_cache[cache_key] = (tokenizer, namespace)
        return namespace

    @staticmethod
    def _chat_message_fingerprints(
        messages: list[dict[str, Any]],
    ) -> tuple[str, ...]:
        """按消息生成稳定指纹，支持最长消息前缀查找。

        Args:
            messages: 已规范化的role/content消息列表。

        Returns:
            与消息顺序一一对应的SHA-256指纹元组。
        """

        return tuple(
            hashlib.sha256(
                json.dumps(
                    message,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for message in messages
        )

    def _evict_chat_cache_locked(self) -> None:
        """在持锁状态下按LRU限制缓存条目数和token总量。

        Returns:
            无返回值；直接回收实例缓存中的最旧条目。
        """

        while (
            len(self._chat_prefix_cache) + len(self._chat_full_cache) + len(self._completion_prefix_cache)
            > self._prefix_cache_max_entries
            or self._prefix_cache_tokens > self._prefix_cache_max_tokens
        ):
            if self._chat_full_cache:
                _key, token_ids = self._chat_full_cache.popitem(last=False)
                self._prefix_cache_tokens -= len(token_ids)
            elif self._chat_prefix_cache:
                _key, entry = self._chat_prefix_cache.popitem(last=False)
                self._prefix_cache_tokens -= len(entry.token_ids)
            elif self._completion_prefix_cache:
                key = next(iter(self._completion_prefix_cache))
                self._remove_completion_prefix_locked(key)
            else:
                break

    def _remove_completion_prefix_locked(
        self,
        key: tuple[str, str, str],
    ) -> None:
        """在持锁状态下删除Completion前缀记录及其分桶索引。

        Args:
            key: 模型、tokenizer命名空间和完整文本摘要组成的缓存键。

        Returns:
            无返回值；记录不存在时不执行操作。
        """

        entry = self._completion_prefix_cache.pop(key, None)
        if entry is None:
            return
        self._prefix_cache_tokens -= len(entry.token_ids)
        bucket = self._completion_prefix_buckets.get(entry.bucket_key)
        if bucket is None:
            return
        bucket.pop(key, None)
        if not bucket:
            self._completion_prefix_buckets.pop(entry.bucket_key, None)

    def _get_full_chat_cache(
        self,
        key: tuple[str, str, tuple[str, ...]],
    ) -> tuple[int, ...] | None:
        """读取完整请求缓存并刷新LRU顺序。

        Args:
            key: 模型、tokenizer命名空间和消息指纹组成的缓存键。

        Returns:
            命中时返回不可变token元组，未命中时返回None。
        """

        with self._lock:
            token_ids = self._chat_full_cache.get(key)
            if token_ids is not None:
                self._chat_full_cache.move_to_end(key)
            return token_ids

    def _put_full_chat_cache(
        self,
        key: tuple[str, str, tuple[str, ...]],
        token_ids: list[int] | tuple[int, ...],
    ) -> tuple[int, ...]:
        """写入完整请求token缓存并执行容量回收。

        Args:
            key: 完整请求缓存键。
            token_ids: 完整请求token ID列表。

        Returns:
            写入缓存的不可变token ID元组，
            可供其他索引共享而避免重复复制。
        """

        value = token_ids if isinstance(token_ids, tuple) else tuple(int(item) for item in token_ids)
        with self._lock:
            previous = self._chat_full_cache.pop(key, None)
            if previous is not None:
                self._prefix_cache_tokens -= len(previous)
            self._chat_full_cache[key] = value
            self._prefix_cache_tokens += len(value)
            self._evict_chat_cache_locked()
        return value

    def _get_longest_chat_prefix(
        self,
        *,
        model: str,
        namespace: str,
        fingerprints: tuple[str, ...],
    ) -> tuple[_ChatPrefixCacheEntry, int] | None:
        """读取当前消息序列的最长已缓存真前缀。

        Args:
            model: 当前模型名称。
            namespace: tokenizer与chat template命名空间。
            fingerprints: 当前完整消息指纹序列。

        Returns:
            最长前缀条目及其消息数；不存在真前缀时返回None。
        """

        with self._lock:
            for length in range(len(fingerprints) - 1, 0, -1):
                key = (model, namespace, fingerprints[:length])
                entry = self._chat_prefix_cache.get(key)
                if entry is not None:
                    self._chat_prefix_cache.move_to_end(key)
                    return entry, length
        return None

    def _put_chat_prefix(
        self,
        *,
        key: tuple[str, str, tuple[str, ...]],
        entry: _ChatPrefixCacheEntry,
    ) -> None:
        """写入一段可在特殊token边界安全复用的消息前缀。

        Args:
            key: 消息前缀缓存键。
            entry: 已渲染、分词并记录边界属性的前缀条目。

        Returns:
            无返回值。
        """

        with self._lock:
            previous = self._chat_prefix_cache.pop(key, None)
            if previous is not None:
                self._prefix_cache_tokens -= len(previous.token_ids)
            self._chat_prefix_cache[key] = entry
            self._prefix_cache_tokens += len(entry.token_ids)
            self._evict_chat_cache_locked()

    @staticmethod
    def _common_text_prefix_length(left: str, right: str) -> int:
        """使用二分比较计算两个字符串的最长公共前缀长度。

        Args:
            left: 第一段文本。
            right: 第二段文本。

        Returns:
            两段文本从起始位置连续相同的字符数量。
        """

        low = 0
        high = min(len(left), len(right))
        while low < high:
            middle = (low + high + 1) // 2
            if left[:middle] == right[:middle]:
                low = middle
            else:
                high = middle - 1
        return low

    def _completion_bucket_key(
        self,
        *,
        model: str,
        namespace: str,
        prompt: str,
    ) -> tuple[str, str, str] | None:
        """计算Completion文本前缀分桶键。

        Args:
            model: 当前模型名称。
            namespace: tokenizer配置命名空间。
            prompt: 当前完整文本。

        Returns:
            文本足够长时返回分桶键，否则返回None并跳过前缀查找。
        """

        if len(prompt) < self._completion_bucket_chars:
            return None
        bucket_prefix = prompt[:self._completion_bucket_chars]
        digest = hashlib.sha256(bucket_prefix.encode("utf-8")).hexdigest()
        return model, namespace, digest

    def _get_longest_completion_prefix(
        self,
        *,
        model: str,
        namespace: str,
        prompt: str,
    ) -> tuple[_CompletionPrefixCacheEntry, int] | None:
        """在有界候选集合中查找最长Completion文本前缀。

        Args:
            model: 当前模型名称。
            namespace: tokenizer配置命名空间。
            prompt: 当前完整文本。

        Returns:
            命中时返回缓存记录和公共字符数，未命中时返回None。
        """

        bucket_key = self._completion_bucket_key(
            model=model,
            namespace=namespace,
            prompt=prompt,
        )
        if bucket_key is None:
            return None
        with self._lock:
            bucket = self._completion_prefix_buckets.get(bucket_key)
            if not bucket:
                return None
            candidate_keys = list(reversed(bucket))[:self._completion_bucket_candidates]
            candidates = [(key, self._completion_prefix_cache.get(key)) for key in candidate_keys]

        best_key: tuple[str, str, str] | None = None
        best_entry: _CompletionPrefixCacheEntry | None = None
        best_length = 0
        for key, entry in candidates:
            if entry is None or entry.prompt == prompt:
                continue
            common_length = self._common_text_prefix_length(entry.prompt, prompt)
            if common_length > best_length:
                best_key = key
                best_entry = entry
                best_length = common_length
        if best_key is None or best_entry is None:
            return None
        with self._lock:
            if best_key in self._completion_prefix_cache:
                self._completion_prefix_cache.move_to_end(best_key)
                bucket = self._completion_prefix_buckets.get(best_entry.bucket_key)
                if bucket is not None and best_key in bucket:
                    bucket.move_to_end(best_key)
        return best_entry, best_length

    def _put_completion_prefix(
        self,
        *,
        model: str,
        namespace: str,
        prompt: str,
        prompt_digest: str,
        token_ids: list[int] | tuple[int, ...],
        token_offsets: list[tuple[int, int]] | None,
    ) -> None:
        """写入支持字符到token边界映射的Completion缓存记录。

        Args:
            model: 当前模型名称。
            namespace: tokenizer配置命名空间。
            prompt: 已完成分词的完整文本。
            prompt_digest: 完整文本已经计算过的SHA-256摘要。
            token_ids: 完整文本对应的token ID列表。
            token_offsets: 每个token在原始文本中的字符起止位置。

        Returns:
            无返回值；缺少可靠offset mapping时不会写入前缀缓存。
        """

        bucket_key = self._completion_bucket_key(
            model=model,
            namespace=namespace,
            prompt=prompt,
        )
        if bucket_key is None or token_offsets is None or len(token_offsets) != len(token_ids):
            return
        key = (model, namespace, prompt_digest)
        normalized_offsets = tuple((int(start), int(end)) for start, end in token_offsets)
        entry = _CompletionPrefixCacheEntry(
            prompt=prompt,
            token_ids=(token_ids if isinstance(token_ids, tuple) else tuple(int(item) for item in token_ids)),
            token_offsets=normalized_offsets,
            token_end_offsets=tuple(end for _start, end in normalized_offsets),
            bucket_key=bucket_key,
        )
        with self._lock:
            self._remove_completion_prefix_locked(key)
            self._completion_prefix_cache[key] = entry
            self._prefix_cache_tokens += len(entry.token_ids)
            bucket = self._completion_prefix_buckets.setdefault(
                bucket_key,
                OrderedDict(),
            )
            bucket[key] = None
            while len(bucket) > self._completion_bucket_candidates:
                oldest_key, _value = bucket.popitem(last=False)
                self._remove_completion_prefix_locked(oldest_key)
            self._evict_chat_cache_locked()

    @staticmethod
    def _encode_completion_with_offsets(
        tokenizer: Any,
        text: str,
    ) -> tuple[list[int], list[tuple[int, int]] | None]:
        """编码Completion文本并尽量取得字符offset mapping。

        Args:
            tokenizer: 当前HuggingFace tokenizer。
            text: 需要编码的文本。

        Returns:
            token ID列表，以及可用时与token一一对应的字符区间列表。
        """

        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            input_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            if offsets and isinstance(offsets[0], list) and offsets[0]:
                if isinstance(offsets[0][0], (list, tuple)):
                    offsets = offsets[0]
            token_ids = [int(item) for item in input_ids]
            normalized_offsets = [(int(item[0]), int(item[1])) for item in offsets]
            if len(token_ids) == len(normalized_offsets):
                return token_ids, normalized_offsets
        except (KeyError, TypeError, ValueError, NotImplementedError):
            pass
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return [int(item) for item in token_ids], None

    def _tokenize_chat_with_prefix_cache(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        model: str,
    ) -> tuple[list[int], dict[str, Any]]:
        """使用经过边界校验的消息前缀缓存执行增量分词。

        Args:
            tokenizer: 当前模型的HuggingFace tokenizer。
            messages: 原始OpenAI消息列表。
            model: 当前请求模型名称。

        Returns:
            token ID列表和模板、编码、缓存命中诊断字段。
        """

        metrics: dict[str, Any] = {
            "template_render_ms": 0.0,
            "tokenizer_encode_ms": 0.0,
            "token_prefix_cache_hit": False,
            "token_full_cache_hit": False,
            "cached_prefix_tokens": 0,
        }
        formatted = self._format_chat_messages(messages)
        namespace = self._get_tokenizer_namespace(tokenizer)
        fingerprints = self._chat_message_fingerprints(formatted)
        full_key = (model, namespace, fingerprints)
        cached_full = self._get_full_chat_cache(full_key)
        if cached_full is not None:
            metrics["token_full_cache_hit"] = True
            metrics["cached_prefix_tokens"] = len(cached_full)
            return list(cached_full), metrics

        render_started = time.perf_counter()
        rendered_full = tokenizer.apply_chat_template(
            formatted,
            tokenize=False,
            add_generation_prompt=True,
        )
        metrics["template_render_ms"] = (time.perf_counter() - render_started) * 1000.0

        prefix_match = self._get_longest_chat_prefix(
            model=model,
            namespace=namespace,
            fingerprints=fingerprints,
        )
        token_ids: list[int] | None = None
        used_incremental = False
        validation_key: tuple[str, str, str, str] | None = None
        if prefix_match is not None:
            entry, prefix_length = prefix_match
            next_role = formatted[prefix_length].get("role", "user")
            validation_key = (model, namespace, entry.last_role, next_role)
            with self._lock:
                validation = self._boundary_validation.get(validation_key)
            if validation is not False and str(rendered_full).startswith(entry.rendered_text):
                suffix_text = str(rendered_full)[len(entry.rendered_text):]
                encode_started = time.perf_counter()
                suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
                metrics["tokenizer_encode_ms"] += (time.perf_counter() - encode_started) * 1000.0
                special_ids = {int(item) for item in getattr(tokenizer, "all_special_ids", [])}
                safe_boundary = entry.terminal_special_token or (bool(suffix_ids) and int(suffix_ids[0]) in special_ids)
                if not safe_boundary:
                    with self._lock:
                        self._boundary_validation[validation_key] = False
                    metrics["cached_prefix_tokens"] = 0
                    suffix_ids = []
                if safe_boundary:
                    candidate = [
                        *entry.token_ids,
                        *(int(item) for item in suffix_ids),
                    ]
                    metrics["cached_prefix_tokens"] = len(entry.token_ids)
                    if validation is True:
                        token_ids = candidate
                        used_incremental = True
                    else:
                        encode_started = time.perf_counter()
                        full_ids = self._tokenize_with_chat_template(
                            tokenizer,
                            formatted,
                        )
                        metrics["tokenizer_encode_ms"] += (time.perf_counter() - encode_started) * 1000.0
                        valid = candidate == full_ids
                        with self._lock:
                            self._boundary_validation[validation_key] = valid
                        token_ids = full_ids
                        used_incremental = False
                        if not valid:
                            metrics["cached_prefix_tokens"] = 0

        performed_full_tokenize = token_ids is None or not used_incremental
        if token_ids is None:
            encode_started = time.perf_counter()
            token_ids = self._tokenize_with_chat_template(tokenizer, formatted)
            metrics["tokenizer_encode_ms"] += (time.perf_counter() - encode_started) * 1000.0
        metrics["token_prefix_cache_hit"] = used_incremental
        self._put_full_chat_cache(full_key, token_ids)

        if performed_full_tokenize and len(formatted) > 1:
            prefix_messages = formatted[:-1]
            prefix_key = (model, namespace, fingerprints[:-1])
            with self._lock:
                prefix_exists = prefix_key in self._chat_prefix_cache
            if not prefix_exists:
                prefix_render_started = time.perf_counter()
                rendered_prefix = tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                metrics["template_render_ms"] += (time.perf_counter() - prefix_render_started) * 1000.0
                prefix_encode_started = time.perf_counter()
                prefix_ids = self._tokenize_with_chat_template(
                    tokenizer,
                    prefix_messages,
                    add_generation_prompt=False,
                )
                metrics["tokenizer_encode_ms"] += (time.perf_counter() - prefix_encode_started) * 1000.0
                special_ids = {int(item) for item in getattr(tokenizer, "all_special_ids", [])}
                self._put_chat_prefix(
                    key=prefix_key,
                    entry=_ChatPrefixCacheEntry(
                        rendered_text=str(rendered_prefix),
                        token_ids=tuple(prefix_ids),
                        last_role=prefix_messages[-1].get("role", "user"),
                        terminal_special_token=bool(prefix_ids) and int(prefix_ids[-1]) in special_ids,
                    ),
                )
        return token_ids, metrics

    def tokenize_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """将消息列表转换为token ID列表

        使用与vllm/sglang相同的方式处理chat messages：
        1. 使用tokenizer的apply_chat_template方法
        2. 如果不支持chat_template，则手动拼接

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称或路径
            tools: OpenAI 工具定义；存在时传给模型 chat template

        Returns:
            token ID列表

        """
        total_started = time.perf_counter()
        tokenizer_get_started = time.perf_counter()
        tokenizer = self._get_tokenizer(model)
        tokenizer_get_ms = (time.perf_counter() - tokenizer_get_started) * 1000.0
        tokenize_started = time.perf_counter()

        if tokenizer is None:
            logger.warning(f"No tokenizer available for model {model}, using fallback")
            token_ids = self._fallback_tokenize(messages)
            self._log_tokenizer_timing(
                model=model,
                mode="chat",
                tokenizer_get_ms=tokenizer_get_ms,
                tokenize_started=tokenize_started,
                total_started=total_started,
                tokenizer_available=False,
            )
            return token_ids

        try:
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                if tools or not self._supports_text_prefix_cache(messages):
                    # 工具及结构化历史走完整模板，不能复用普通文本的边界验证。
                    token_ids = self._tokenize_with_chat_template(
                        tokenizer,
                        messages,
                        tools=tools,
                    )
                    cache_metrics = {}
                elif self._prefix_cache_enabled:
                    token_ids, cache_metrics = self._tokenize_chat_with_prefix_cache(
                        tokenizer,
                        messages,
                        model,
                    )
                else:
                    token_ids = self._tokenize_with_chat_template(tokenizer, messages)
                    cache_metrics = {}
            else:
                token_ids = self._tokenize_manually(tokenizer, messages)
                cache_metrics = {}

            logger.debug(f"Tokenized {len(messages)} messages to {len(token_ids)} tokens")
            self._log_tokenizer_timing(
                model=model,
                mode="chat",
                tokenizer_get_ms=tokenizer_get_ms,
                tokenize_started=tokenize_started,
                total_started=total_started,
                tokenizer_available=True,
                **cache_metrics,
            )
            return token_ids

        except Exception as e:
            logger.error(f"Error tokenizing messages for model {model}: {e}")
            token_ids = self._fallback_tokenize(messages)
            self._log_tokenizer_timing(
                model=model,
                mode="chat_fallback",
                tokenizer_get_ms=tokenizer_get_ms,
                tokenize_started=tokenize_started,
                total_started=total_started,
                tokenizer_available=True,
            )
            return token_ids

    def tokenize_prompt(
        self,
        prompt: str | list[int],
        model: str,
    ) -> list[int]:
        """将原始prompt转换为token ID列表

        Args:
            prompt: 提示文本（str）或已有的token ID列表
            model: 模型名称

        Returns:
            token ID列表

        """
        if isinstance(prompt, list):
            return prompt

        total_started = time.perf_counter()
        tokenizer_get_started = time.perf_counter()
        tokenizer = self._get_tokenizer(model)
        tokenizer_get_ms = (time.perf_counter() - tokenizer_get_started) * 1000.0
        tokenize_started = time.perf_counter()
        if tokenizer is None:
            logger.warning(f"No tokenizer available for model {model}, using fallback for prompt")
            token_ids = [ord(c) for c in prompt]
            self._log_tokenizer_timing(
                model=model,
                mode="completion",
                tokenizer_get_ms=tokenizer_get_ms,
                tokenize_started=tokenize_started,
                total_started=total_started,
                tokenizer_available=False,
            )
            return token_ids

        try:
            namespace = self._get_tokenizer_namespace(tokenizer)
            full_lookup_started = time.perf_counter()
            prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            prompt_cache_key = (
                model,
                namespace,
                ("completion:" + prompt_digest,),
            )
            if self._prefix_cache_enabled:
                cached = self._get_full_chat_cache(prompt_cache_key)
                if cached is not None:
                    token_ids = list(cached)
                    full_lookup_ms = (time.perf_counter() - full_lookup_started) * 1000.0
                    self._log_tokenizer_timing(
                        model=model,
                        mode="completion",
                        tokenizer_get_ms=tokenizer_get_ms,
                        tokenize_started=tokenize_started,
                        total_started=total_started,
                        tokenizer_available=True,
                        token_full_cache_hit=True,
                        cached_prefix_tokens=len(token_ids),
                        completion_full_lookup_ms=full_lookup_ms,
                    )
                    return token_ids
                full_lookup_ms = (time.perf_counter() - full_lookup_started) * 1000.0
            else:
                full_lookup_ms = (time.perf_counter() - full_lookup_started) * 1000.0

            tokenizer_encode_ms = 0.0
            prefix_lookup_ms = 0.0
            suffix_encode_ms = 0.0
            token_merge_ms = 0.0
            full_cache_update_ms = 0.0
            prefix_index_update_ms = 0.0
            cached_prefix_tokens = 0
            used_incremental = False
            boundary_validated = False
            should_store_prefix_entry = True
            token_ids: list[int] | None = None
            token_offsets: list[tuple[int, int]] | None = None

            if self._prefix_cache_enabled:
                lookup_started = time.perf_counter()
                prefix_match = self._get_longest_completion_prefix(
                    model=model,
                    namespace=namespace,
                    prompt=prompt,
                )
                prefix_lookup_ms = (time.perf_counter() - lookup_started) * 1000.0
                if prefix_match is not None:
                    entry, common_chars = prefix_match
                    matched_tokens = bisect_right(
                        entry.token_end_offsets,
                        common_chars,
                    )
                    reusable_tokens = max(
                        matched_tokens - self._completion_boundary_overlap_tokens,
                        0,
                    )
                    should_store_prefix_entry = reusable_tokens < self._completion_prefix_min_tokens
                    if reusable_tokens >= self._completion_prefix_min_tokens:
                        suffix_start = entry.token_offsets[reusable_tokens - 1][1]
                        boundary_context = entry.prompt[
                            max(suffix_start - 64, 0):min(
                                suffix_start + 64,
                                common_chars,
                            )
                        ]
                        boundary_digest = hashlib.sha256(boundary_context.encode("utf-8")).hexdigest()[:20]
                        validation_key = (model, namespace, boundary_digest)
                        with self._lock:
                            validation = self._completion_boundary_validation.get(validation_key)
                        if validation is False:
                            reusable_tokens = 0
                    if reusable_tokens >= self._completion_prefix_min_tokens:
                        suffix_start = entry.token_offsets[reusable_tokens - 1][1]
                        suffix_started = time.perf_counter()
                        suffix_ids = tokenizer.encode(
                            prompt[suffix_start:],
                            add_special_tokens=False,
                        )
                        suffix_ids = [int(item) for item in suffix_ids]
                        suffix_encode_ms = (time.perf_counter() - suffix_started) * 1000.0
                        tokenizer_encode_ms += suffix_encode_ms
                        merge_started = time.perf_counter()
                        candidate_ids = [
                            *entry.token_ids[:reusable_tokens],
                            *suffix_ids,
                        ]
                        token_merge_ms += (time.perf_counter() - merge_started) * 1000.0
                        if validation is True:
                            token_ids = candidate_ids
                            cached_prefix_tokens = reusable_tokens
                            used_incremental = True
                        else:
                            full_started = time.perf_counter()
                            full_ids, full_offsets = self._encode_completion_with_offsets(
                                tokenizer,
                                prompt,
                            )
                            tokenizer_encode_ms += (time.perf_counter() - full_started) * 1000.0
                            boundary_validated = candidate_ids == full_ids
                            with self._lock:
                                self._completion_boundary_validation[validation_key] = boundary_validated
                            token_ids = full_ids
                            token_offsets = full_offsets

            if token_ids is None:
                encode_started = time.perf_counter()
                token_ids, token_offsets = self._encode_completion_with_offsets(
                    tokenizer,
                    prompt,
                )
                tokenizer_encode_ms += (time.perf_counter() - encode_started) * 1000.0
            if self._prefix_cache_enabled:
                full_cache_update_started = time.perf_counter()
                cached_token_ids = self._put_full_chat_cache(
                    prompt_cache_key,
                    token_ids,
                )
                full_cache_update_ms = (time.perf_counter() - full_cache_update_started) * 1000.0
                if should_store_prefix_entry and token_offsets is not None:
                    prefix_update_started = time.perf_counter()
                    self._put_completion_prefix(
                        model=model,
                        namespace=namespace,
                        prompt=prompt,
                        prompt_digest=prompt_digest,
                        token_ids=cached_token_ids,
                        token_offsets=token_offsets,
                    )
                    prefix_index_update_ms = (time.perf_counter() - prefix_update_started) * 1000.0
            logger.debug(f"Tokenized prompt to {len(token_ids)} tokens")
            self._log_tokenizer_timing(
                model=model,
                mode="completion",
                tokenizer_get_ms=tokenizer_get_ms,
                tokenize_started=tokenize_started,
                total_started=total_started,
                tokenizer_available=True,
                tokenizer_encode_ms=tokenizer_encode_ms,
                token_prefix_cache_hit=used_incremental,
                cached_prefix_tokens=cached_prefix_tokens,
                completion_prefix_lookup_ms=prefix_lookup_ms,
                completion_suffix_encode_ms=suffix_encode_ms,
                completion_boundary_validated=boundary_validated,
                completion_full_lookup_ms=full_lookup_ms,
                completion_token_merge_ms=token_merge_ms,
                completion_full_cache_update_ms=full_cache_update_ms,
                completion_prefix_index_update_ms=prefix_index_update_ms,
            )
            return list(token_ids)
        except Exception as e:
            logger.error(f"Error tokenizing prompt for model {model}: {e}")
            token_ids = [ord(c) for c in prompt]
            self._log_tokenizer_timing(
                model=model,
                mode="completion_fallback",
                tokenizer_get_ms=tokenizer_get_ms,
                tokenize_started=tokenize_started,
                total_started=total_started,
                tokenizer_available=True,
            )
            return token_ids

    @staticmethod
    def _log_tokenizer_timing(
        *,
        model: str,
        mode: str,
        tokenizer_get_ms: float,
        tokenize_started: float,
        total_started: float,
        tokenizer_available: bool,
        template_render_ms: float = 0.0,
        tokenizer_encode_ms: float = 0.0,
        token_prefix_cache_hit: bool = False,
        token_full_cache_hit: bool = False,
        cached_prefix_tokens: int = 0,
        completion_prefix_lookup_ms: float = 0.0,
        completion_suffix_encode_ms: float = 0.0,
        completion_boundary_validated: bool = False,
        completion_full_lookup_ms: float = 0.0,
        completion_token_merge_ms: float = 0.0,
        completion_full_cache_update_ms: float = 0.0,
        completion_prefix_index_update_ms: float = 0.0,
    ) -> None:
        """输出 Router 分词阶段的固定字段耗时。

        Args:
            model: 当前请求使用的模型名称或路径。
            mode: 分词路径，区分 chat、completion 及其 fallback。
            tokenizer_get_ms: 获取或首次加载 tokenizer 的耗时，单位毫秒。
            tokenize_started: 开始执行模板处理或编码时的单调时钟时间。
            total_started: 整个分词调用开始时的单调时钟时间。
            tokenizer_available: 是否成功取得真实 tokenizer。
            template_render_ms: chat template渲染耗时，单位毫秒。
            tokenizer_encode_ms: tokenizer实际编码耗时，单位毫秒。
            token_prefix_cache_hit: 是否使用了已验证的增量消息前缀。
            token_full_cache_hit: 是否完整命中相同请求token缓存。
            cached_prefix_tokens: 本次避免重复编码的前缀token数量。
            completion_prefix_lookup_ms: Completion公共文本前缀查找耗时，单位毫秒。
            completion_suffix_encode_ms: Completion增量后缀编码耗时，单位毫秒。
            completion_boundary_validated: 本次是否完成增量结果与完整编码校验。
            completion_full_lookup_ms: Completion完整请求缓存查找耗时，单位毫秒。
            completion_token_merge_ms: Completion前缀与后缀token合并耗时，单位毫秒。
            completion_full_cache_update_ms: Completion完整请求缓存更新时间，
                单位毫秒。
            completion_prefix_index_update_ms: Completion前缀索引更新时间，单位毫秒。

        Returns:
            无返回值；只输出性能诊断日志。
        """

        finished = time.perf_counter()
        logger.info(
            "[TokenizerTiming] model={} mode={} tokenizer_get={:.2f}ms "
            "tokenize={:.2f}ms total={:.2f}ms tokenizer_available={} "
            "template_render={:.2f}ms tokenizer_encode={:.2f}ms "
            "token_prefix_cache_hit={} token_full_cache_hit={} "
            "cached_prefix_tokens={} completion_prefix_lookup={:.2f}ms "
            "completion_suffix_encode={:.2f}ms "
            "completion_boundary_validated={} completion_full_lookup={:.2f}ms "
            "completion_token_merge={:.2f}ms "
            "completion_full_cache_update={:.2f}ms "
            "completion_prefix_index_update={:.2f}ms",
            model,
            mode,
            tokenizer_get_ms,
            max((finished - tokenize_started) * 1000.0, 0.0),
            max((finished - total_started) * 1000.0, 0.0),
            tokenizer_available,
            template_render_ms,
            tokenizer_encode_ms,
            token_prefix_cache_hit,
            token_full_cache_hit,
            cached_prefix_tokens,
            completion_prefix_lookup_ms,
            completion_suffix_encode_ms,
            completion_boundary_validated,
            completion_full_lookup_ms,
            completion_token_merge_ms,
            completion_full_cache_update_ms,
            completion_prefix_index_update_ms,
        )

    @staticmethod
    def _tokenize_with_chat_template(
        tokenizer: Any,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """使用chat_template进行tokenize

        这是vllm推荐的方式，确保与工作器的tokenize结果一致。

        Args:
            tokenizer: tokenizer实例
            messages: 消息列表

        Returns:
            token ID列表
        """
        formatted_messages = TokenizerManager._format_chat_messages(messages)

        template_kwargs: dict[str, Any] = {}
        if tools:
            template_kwargs["tools"] = tools
        token_ids = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

        # 处理返回值可能是字典或 BatchEncoding对象
        if isinstance(token_ids, dict):
            # 如果是字典，优先用 input_ids
            if "input_ids" in token_ids:
                token_ids = token_ids["input_ids"]
            elif "input_ids" in token_ids:
                token_ids = token_ids["input_ids"]
        elif hasattr(token_ids, "input_ids"):
            # BatchEncoding对象
            token_ids = token_ids.input_ids

        # 转换为 Python 列表
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            # 如果是二维列表，取第一个
            token_ids = token_ids[0]

        return list(token_ids)

    @staticmethod
    def _tokenize_manually(tokenizer: Any, messages: list[dict]) -> list[int]:
        """手动拼接消息并tokenize

        用于不支持chat_template的tokenizer。

        Args:
            tokenizer: tokenizer实例
            messages: 消息列表

        Returns:
            token ID列表

        """
        text_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                text_parts.append(f"<|{role}|>\n{content}")
            elif isinstance(content, list):
                content_text = ""
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_text += part.get("text", "")
                    elif isinstance(part, dict) and "text" in part:
                        content_text += part["text"]
                text_parts.append(f"<|{role}|>\n{content_text}")

        full_text = "\n".join(text_parts)

        token_ids = tokenizer.encode(full_text, add_special_tokens=True)
        return list(token_ids)

    @staticmethod
    def _fallback_tokenize(messages: list[dict]) -> list[int]:
        """后备tokenize方法

        当无法加载tokenizer时使用简单的字符级tokenize。

        Args:
            messages: 消息列表

        Returns:
            token ID列表

        """
        token_ids = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            role_tokens = [ord(c) for c in f"<|{role}|>"]
            token_ids.extend(role_tokens)

            if isinstance(content, str):
                content_tokens = [ord(c) for c in content]
                token_ids.extend(content_tokens)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        content_tokens = [ord(c) for c in part["text"]]
                        token_ids.extend(content_tokens)

        return token_ids

    def decode_tokens(self, token_ids: list[int], model: str) -> str:
        """将token ID列表解码为文本

        Args:
            token_ids: token ID列表
            model: 模型名称或路径

        Returns:
            解码后的文本

        """
        tokenizer = self._get_tokenizer(model)

        if tokenizer is None:
            return "".join(chr(t) if t < 128 else f"<{t}>" for t in token_ids)

        try:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            return "".join(chr(t) if t < 128 else f"<{t}>" for t in token_ids)

    def get_vocab_size(self, model: str) -> int:
        """获取模型的词表大小

        Args:
            model: 模型名称或路径

        Returns:
            词表大小

        """
        tokenizer = self._get_tokenizer(model)

        if tokenizer is None:
            return 0

        return len(tokenizer)
