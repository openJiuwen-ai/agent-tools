from loguru import logger

from openjiuwentools.infer_router.config.config import settings
from openjiuwentools.infer_router.preprocess.tokenizer import TokenizerManager
from openjiuwentools.infer_router.schemas.agent_hints import (
    AgentHints,
    ChatCompletionRequest,
    CompletionRequest,
    RouteHint,
)


class Preprocessor:
    """预处理层，将AgentHints转换为内部路由提示"""

    def __init__(self):
        self.default_priority = 0
        self.default_estimated_output_tokens = 128
        self.default_next_turn_prefill = False
        self.default_total_requests = 10
        self.default_iat = 250
        self.tokenizer_manager = TokenizerManager(load_from_file=settings.tokenizer_load_from_file)

    def process(
        self, request: ChatCompletionRequest, agent_hints: AgentHints, request_id: str
    ) -> RouteHint:
        """处理请求并转换为路由提示"""
        logger.debug(f"Preprocessing request {request_id}")

        priority = self.default_priority
        estimated_output_tokens = self.default_estimated_output_tokens
        next_turn_prefill = self.default_next_turn_prefill
        prefix_id = None
        total_requests = self.default_total_requests
        iat = self.default_iat

        if agent_hints:
            priority = agent_hints.priority or self.default_priority
            estimated_output_tokens = (
                agent_hints.estimated_output_tokens or self.default_estimated_output_tokens
            )
            next_turn_prefill = agent_hints.next_turn_prefill or self.default_next_turn_prefill
            prefix_id = agent_hints.prefix_id
            total_requests = agent_hints.total_requests or self.default_total_requests
            iat = agent_hints.iat or self.default_iat

        priority = max(0, min(10, priority))
        estimated_output_tokens = max(1, estimated_output_tokens)
        total_requests = max(1, min(50, total_requests))
        iat = max(1, iat)

        token_ids = self.tokenizer_manager.tokenize_messages(
            messages=request.messages,
            model=request.model,
        )

        route_hint = RouteHint(
            priority=priority,
            estimated_output_tokens=estimated_output_tokens,
            next_turn_prefill=next_turn_prefill,
            request_id=request_id,
            model=request.model,
            prefix_id=prefix_id,
            total_requests=total_requests,
            iat=iat,
            token_ids=token_ids,
        )

        logger.debug(f"Preprocessing completed for {request_id}: {route_hint.model_dump()}")
        return route_hint

    def process_completion(
        self, request: CompletionRequest, agent_hints: AgentHints, request_id: str
    ) -> RouteHint:
        """处理 /v1/completions 请求并转换为路由提示"""
        logger.debug(f"Preprocessing completion request {request_id}")

        priority = self.default_priority
        estimated_output_tokens = self.default_estimated_output_tokens
        next_turn_prefill = self.default_next_turn_prefill
        prefix_id = None
        total_requests = self.default_total_requests
        iat = self.default_iat

        if agent_hints:
            priority = agent_hints.priority or self.default_priority
            estimated_output_tokens = (
                agent_hints.estimated_output_tokens or self.default_estimated_output_tokens
            )
            next_turn_prefill = agent_hints.next_turn_prefill or self.default_next_turn_prefill
            prefix_id = agent_hints.prefix_id
            total_requests = agent_hints.total_requests or self.default_total_requests
            iat = agent_hints.iat or self.default_iat

        priority = max(0, min(10, priority))
        estimated_output_tokens = max(1, estimated_output_tokens)
        total_requests = max(1, min(50, total_requests))
        iat = max(1, iat)

        token_ids = self.tokenizer_manager.tokenize_prompt(
            prompt=request.prompt,
            model=request.model,
        )

        route_hint = RouteHint(
            priority=priority,
            estimated_output_tokens=estimated_output_tokens,
            next_turn_prefill=next_turn_prefill,
            request_id=request_id,
            model=request.model,
            prefix_id=prefix_id,
            total_requests=total_requests,
            iat=iat,
            token_ids=token_ids,
        )

        logger.debug(f"Preprocessing completed for {request_id}: {route_hint.model_dump()}")
        return route_hint
