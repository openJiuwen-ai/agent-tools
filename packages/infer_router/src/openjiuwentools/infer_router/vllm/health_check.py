import asyncio
import logging
from typing import Any

logger = logging.getLogger("jiuwen.health_check")

_ENV_VAR = "JIUWEN_HEALTH_CHECK_PAYLOAD"


def _load_payload_from_env() -> dict | None:
    import json
    import os

    env_value = os.environ.get(_ENV_VAR)
    if not env_value:
        return None
    try:
        if env_value.startswith("@"):
            with open(env_value[1:]) as f:
                parsed = json.load(f)
        else:
            parsed = json.loads(env_value)
        if not isinstance(parsed, dict):
            logger.warning("%s must be a JSON object, got %s", _ENV_VAR, type(parsed).__name__)
            return None
        return parsed
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to parse %s: %s", _ENV_VAR, e)
        return None


def _get_bos_token_id(engine) -> int:
    try:
        tokenizer_group = getattr(engine, "tokenizer", None)
        if tokenizer_group:
            tokenizer = getattr(tokenizer_group, "tokenizer", None)
            if tokenizer:
                bos_token_id = getattr(tokenizer, "bos_token_id", None)
                if bos_token_id is not None:
                    logger.info("Using model BOS token ID for health check: %d", bos_token_id)
                    return int(bos_token_id)
    except Exception as e:
        logger.debug("Failed to get BOS token from engine: %s", e)
    return 1


class HealthChecker:
    """通过向引擎发送最小推理请求来验证 vLLM 引擎是否正常工作。

    payload 优先级: 环境变量 JIUWEN_HEALTH_CHECK_PAYLOAD > 默认 payload（BOS token + max_tokens=1）
    """

    def __init__(self, engine, model_name: str) -> None:
        self.engine = engine
        self.model_name = model_name
        self._payload: dict[str, Any] | None = None

    @staticmethod
    def _default_payload() -> dict:
        return {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }

    def payload(self) -> dict:
        if self._payload is None:
            self._payload = _load_payload_from_env() or self._default_payload()
        return self._payload

    async def check(self) -> tuple[bool, str]:
        """执行一次健康检查，返回 (healthy, detail)。"""
        from openjiuwentools.infer_router.vllm.transport import do_chat_completion

        try:
            result = await asyncio.wait_for(
                do_chat_completion(self.engine, self.payload(), self.model_name),
                timeout=30.0,
            )
            if "error" in result:
                return False, result["error"]
            return True, "ok"
        except asyncio.TimeoutError:
            return False, "health check timed out (30s)"
        except Exception as e:
            return False, str(e)
