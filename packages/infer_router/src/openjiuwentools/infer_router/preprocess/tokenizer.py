import os
from threading import Lock
from typing import Any

from loguru import logger

from openjiuwentools.infer_router.config.config import settings

_tokenizers: dict[str, Any] = {}
_tokenizer_lock = Lock()


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

    def tokenize_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> list[int]:
        """将消息列表转换为token ID列表

        使用与vllm/sglang相同的方式处理chat messages：
        1. 使用tokenizer的apply_chat_template方法
        2. 如果不支持chat_template，则手动拼接

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称或路径

        Returns:
            token ID列表

        """
        tokenizer = self._get_tokenizer(model)

        if tokenizer is None:
            logger.warning(f"No tokenizer available for model {model}, using fallback")
            return self._fallback_tokenize(messages)

        try:
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                token_ids = self._tokenize_with_chat_template(tokenizer, messages)
            else:
                token_ids = self._tokenize_manually(tokenizer, messages)

            logger.debug(f"Tokenized {len(messages)} messages to {len(token_ids)} tokens")
            return token_ids

        except Exception as e:
            logger.error(f"Error tokenizing messages for model {model}: {e}")
            return self._fallback_tokenize(messages)

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

        tokenizer = self._get_tokenizer(model)
        if tokenizer is None:
            logger.warning(f"No tokenizer available for model {model}, using fallback for prompt")
            return [ord(c) for c in prompt]

        try:
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            logger.debug(f"Tokenized prompt to {len(token_ids)} tokens")
            return list(token_ids)
        except Exception as e:
            logger.error(f"Error tokenizing prompt for model {model}: {e}")
            return [ord(c) for c in prompt]

    @staticmethod
    def _tokenize_with_chat_template(tokenizer: Any, messages: list[dict]) -> list[int]:
        """使用chat_template进行tokenize

        这是vllm推荐的方式，确保与工作器的tokenize结果一致。

        Args:
            tokenizer: tokenizer实例
            messages: 消息列表

        Returns:
            token ID列表
        """
        formatted_messages = []
        for msg in messages:
            formatted_msg = {"role": msg.get("role", "user")}
            content = msg.get("content", "")

            if isinstance(content, str):
                formatted_msg["content"] = content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                formatted_msg["content"] = "\n".join(text_parts)
            else:
                formatted_msg["content"] = str(content)

            formatted_messages.append(formatted_msg)

        token_ids = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=True,
            add_generation_prompt=True,
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
