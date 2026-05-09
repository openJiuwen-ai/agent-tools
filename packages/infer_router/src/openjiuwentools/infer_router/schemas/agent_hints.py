from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CacheControl(BaseModel):
    """缓存控制模型"""

    type: str | None = Field(default="ephemeral", description="缓存固定策略类型")
    ttl: str | None = Field(default=None, description="缓存生存时间")


class AgentHints(BaseModel):
    """Agent Hints模型定义"""

    priority: int | None = Field(default=0, description="请求优先级，值越高越重要")
    estimated_output_tokens: int | None = Field(default=128, description="预期输出token数量")
    next_turn_prefill: bool | None = Field(default=False, description="是否启用下一轮预填充")
    prefix_id: str | None = Field(default=None, description="前缀ID，用于标识同一对话或工作流的请求")
    total_requests: int | None = Field(default=10, description="预期请求总数")
    iat: int | None = Field(default=250, description="预期请求间隔时间（毫秒）")


class JWExt(BaseModel):
    """Jiuwen扩展字段模型"""

    agent_hints: AgentHints | None = Field(default=None, description="Agent Hints信息")
    cache_control: CacheControl | None = Field(default=None, description="缓存控制信息")


class ChatCompletionRequest(BaseModel):
    """聊天完成请求模型"""

    model: str = Field(description="模型名称")
    messages: list[dict[str, Any]] = Field(description="消息列表")
    jiuwenext: JWExt | None = Field(default=None, description="Jiuwen扩展字段")
    # 其他标准字段
    max_tokens: int | None = Field(default=128)
    temperature: float | None = Field(default=1.0)
    top_p: float | None = Field(default=1.0)
    n: int | None = Field(default=1)
    stream: bool | None = Field(default=False)


class RouteHint(BaseModel):
    """内部路由提示模型"""

    priority: int
    estimated_output_tokens: int
    next_turn_prefill: bool
    request_id: str
    model: str
    prefix_id: str | None = None
    total_requests: int | None = None
    iat: int | None = None
    token_ids: list[int] | None = None


class WorkerType(str, Enum):
    """工作器类型枚举"""

    PREFILL = "prefill"
    DECODE = "decode"
    COMBINED = "combined"


class WorkerInfo(BaseModel):
    """推理工作器信息模型"""

    worker_id: str
    model: str
    total_tokens: int = Field(default=1000000, description="工作器可以承载的token个数")
    current_load: float = 0.0
    cached_prefixes: list[str] = []
    engine_type: str = "vllm"
    url: str = ""
    api_key: str | None = None
    worker_type: WorkerType = WorkerType.COMBINED
    group: str = "default"
    kv_addr: str = Field(default="", description="KV存储地址，用于P2P disagg connector")

    @property
    def available_memory(self) -> int:
        """兼容旧代码的属性，返回 total_tokens"""
        return self.total_tokens


class KVCacheBlock(BaseModel):
    """KV缓存块模型"""

    block_id: str
    prefix_hash: str
    last_accessed: float
    is_aging: bool
