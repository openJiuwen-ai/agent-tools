import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Generic, Optional, TypeVar

from loguru import logger

from openjiuwentools.infer_router.config.config import settings

T = TypeVar("T")


class RetryResult(Generic[T]):
    """重试结果"""

    def __init__(
        self,
        success: bool,
        result: T | None = None,
        error: Exception | None = None,
        attempts: int = 0,
    ):
        self.success = success
        self.result = result
        self.error = error
        self.attempts = attempts


async def retry_async(
    func: Callable[..., Any],
    *args,
    max_attempts: int = settings.retry_attempts,
    delay: float = settings.retry_delay,
    retry_exceptions: tuple = (Exception,),
    jitter: bool = True,
    jitter_factor: float = 0.1,
    circuit_breaker: Optional["CircuitBreaker"] = None,
    **kwargs,
) -> Any:
    """异步重试装饰器

    Args:
        func: 要执行的函数
        args: 函数位置参数
        max_attempts: 最大重试次数
        delay: 基础延迟时间
        retry_exceptions: 需要重试的异常类型
        jitter: 是否添加随机抖动
        jitter_factor: 抖动因子（0-1之间）
        circuit_breaker: 断路器实例，用于避免对不健康的服务重试
        kwargs: 函数关键字参数

    Returns:
        函数执行结果

    """
    attempts = 0

    while attempts < max_attempts:
        try:
            # 检查断路器状态
            if circuit_breaker:
                try:
                    circuit_breaker.check_state()
                except Exception as cb_e:
                    logger.warning(f"Circuit breaker check failed: {cb_e}")
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    continue

            result = await func(*args, **kwargs)

            # 通知断路器成功
            if circuit_breaker:
                circuit_breaker.handle_success()

            if attempts > 0:
                logger.info(f"Operation succeeded after {attempts + 1} attempts")
            return result
        except retry_exceptions as e:
            attempts += 1
            logger.warning(f"Operation failed on attempt {attempts}/{max_attempts}: {e}")

            # 通知断路器失败
            if circuit_breaker:
                circuit_breaker.handle_failure()

            if attempts < max_attempts:
                # 指数退避
                wait_time = delay * (2 ** (attempts - 1))

                # 添加随机抖动
                if jitter:
                    jitter_value = wait_time * jitter_factor * random.random()  # noqa: S311
                    wait_time += jitter_value

                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Operation failed after {max_attempts} attempts")
                raise


def retry_sync(
    func: Callable[..., Any],
    *args,
    max_attempts: int = settings.retry_attempts,
    delay: float = settings.retry_delay,
    retry_exceptions: tuple = (Exception,),
    jitter: bool = True,
    jitter_factor: float = 0.1,
    circuit_breaker: Optional["CircuitBreaker"] = None,
    **kwargs,
) -> Any:
    """同步重试装饰器

    Args:
        func: 要执行的函数
        args: 函数位置参数
        max_attempts: 最大重试次数
        delay: 基础延迟时间
        retry_exceptions: 需要重试的异常类型
        jitter: 是否添加随机抖动
        jitter_factor: 抖动因子（0-1之间）
        circuit_breaker: 断路器实例，用于避免对不健康的服务重试
        kwargs: 函数关键字参数

    Returns:
        函数执行结果

    """
    attempts = 0

    while attempts < max_attempts:
        try:
            # 检查断路器状态
            if circuit_breaker:
                try:
                    circuit_breaker.check_state()
                except Exception as cb_e:
                    logger.warning(f"Circuit breaker check failed: {cb_e}")
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    continue

            result = func(*args, **kwargs)

            # 通知断路器成功
            if circuit_breaker:
                circuit_breaker.handle_success()

            if attempts > 0:
                logger.info(f"Operation succeeded after {attempts + 1} attempts")
            return result
        except retry_exceptions as e:
            attempts += 1
            logger.warning(f"Operation failed on attempt {attempts}/{max_attempts}: {e}")

            # 通知断路器失败
            if circuit_breaker:
                circuit_breaker.handle_failure()

            if attempts < max_attempts:
                # 指数退避
                wait_time = delay * (2 ** (attempts - 1))

                # 添加随机抖动
                if jitter:
                    jitter_value = wait_time * jitter_factor * random.random()  # noqa: S311
                    wait_time += jitter_value

                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Operation failed after {max_attempts} attempts")
                raise


@dataclass
class RetryConfig:
    """重试配置参数"""

    max_attempts: int = settings.retry_attempts
    delay: float = settings.retry_delay
    retry_exceptions: tuple = field(default=(Exception,))
    jitter: bool = True
    jitter_factor: float = 0.1


def retry(
    config: RetryConfig | None = None,
    circuit_breaker: Optional["CircuitBreaker"] = None,
):
    """通用重试装饰器

    Args:
        config: 重试配置参数，为None时使用默认配置
        circuit_breaker: 断路器实例，用于避免对不健康的服务重试

    """
    cfg = config or RetryConfig()

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await retry_async(
                func,
                *args,
                max_attempts=cfg.max_attempts,
                delay=cfg.delay,
                retry_exceptions=cfg.retry_exceptions,
                jitter=cfg.jitter,
                jitter_factor=cfg.jitter_factor,
                circuit_breaker=circuit_breaker,
                **kwargs,
            )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return retry_sync(
                func,
                *args,
                max_attempts=cfg.max_attempts,
                delay=cfg.delay,
                retry_exceptions=cfg.retry_exceptions,
                jitter=cfg.jitter,
                jitter_factor=cfg.jitter_factor,
                circuit_breaker=circuit_breaker,
                **kwargs,
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class CircuitBreaker:
    """断路器模式实现"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exceptions: tuple = (Exception,),
        half_open_max_calls: int = 3,
    ):
        """初始化断路器

        Args:
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时时间（秒）
            expected_exceptions: 预期的异常类型
            half_open_max_calls: 半开状态下的最大调用次数

        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0

        # 监控指标
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_transitions = 0

    def check_state(self) -> bool:
        """检查断路器状态

        Returns:
            是否允许调用

        Raises:
            Exception: 断路器拒绝请求

        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit breaker transitioning from OPEN to HALF_OPEN")
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
                self.state_transitions += 1
            else:
                logger.warning("Circuit breaker is OPEN, rejecting request")
                raise Exception("Circuit breaker is OPEN")

        if self.state == "HALF_OPEN" and self.half_open_calls >= self.half_open_max_calls:
            logger.warning("Circuit breaker is HALF_OPEN with max calls, rejecting request")
            raise Exception("Circuit breaker is HALF_OPEN with max calls")

        return True

    def handle_success(self):
        """处理成功调用"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                logger.info("Circuit breaker transitioning from HALF_OPEN to CLOSED")
                self.state = "CLOSED"
                self.failure_count = 0
                self.success_count = 0
                self.last_failure_time = None
                self.state_transitions += 1

        self.total_successes += 1

    def handle_failure(self):
        """处理失败调用"""
        self.total_failures += 1

        if self.state == "CLOSED":
            self.failure_count += 1
            logger.warning(
                f"Circuit breaker failure count: {self.failure_count}/{self.failure_threshold}"
            )

            if self.failure_count >= self.failure_threshold:
                logger.info("Circuit breaker transitioning from CLOSED to OPEN")
                self.state = "OPEN"
                self.last_failure_time = time.time()
                self.state_transitions += 1

        elif self.state == "HALF_OPEN":
            logger.info("Circuit breaker transitioning from HALF_OPEN to OPEN")
            self.state = "OPEN"
            self.last_failure_time = time.time()
            self.success_count = 0
            self.state_transitions += 1

    async def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """调用受保护的函数

        Args:
            func: 要执行的函数
            args: 函数参数
            kwargs: 函数关键字参数

        Returns:
            函数执行结果

        Raises:
            Exception: 断路器处于OPEN状态或函数执行失败

        """
        self.total_calls += 1
        self.check_state()

        try:
            if self.state == "HALF_OPEN":
                self.half_open_calls += 1

            result = await func(*args, **kwargs)
            self.handle_success()
            return result

        except self.expected_exceptions:
            self.handle_failure()
            raise

    def get_stats(self) -> dict:
        """获取断路器统计信息

        Returns:
            包含统计信息的字典

        """
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "state_transitions": self.state_transitions,
        }
