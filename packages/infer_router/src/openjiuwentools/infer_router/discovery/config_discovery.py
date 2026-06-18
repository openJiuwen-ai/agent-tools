"""从配置文件发现工作器"""

import json
from pathlib import Path

from loguru import logger

from openjiuwentools.infer_router.discovery.base import WorkerDiscovery
from openjiuwentools.infer_router.schemas.agent_hints import WorkerInfo, WorkerType


class ConfigDiscovery(WorkerDiscovery):
    """从配置文件发现工作器"""

    def __init__(self, config_path: str | None = None):
        """初始化配置文件发现器

        Args:
            config_path: 配置文件路径，支持JSON和YAML格式

        """
        self.config_path = config_path or "workers.json"
        self._workers_cache: list[WorkerInfo] | None = None

    def _load_config(self, config_file: Path) -> dict:
        """加载配置文件

        Args:
            config_file: 配置文件路径

        Returns:
            dict: 配置数据

        Raises:
            ValueError: 不支持的文件格式

        """
        suffix = config_file.suffix.lower()

        if suffix == ".json":
            with open(config_file, encoding="utf-8") as f:
                return json.load(f)
        elif suffix in [".yaml", ".yml"]:
            try:
                import yaml

                with open(config_file, encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except ImportError:
                logger.error("PyYAML package not installed. Install it with: pip install pyyaml")
                raise
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. Supported formats: .json, .yaml, .yml"
            )

    async def discover(self) -> list[WorkerInfo]:
        """从配置文件读取工作器信息

        Returns:
            List[WorkerInfo]: 工作器列表

        """
        try:
            config_file = Path(self.config_path)

            if not config_file.exists():
                logger.warning(f"Worker config file not found: {self.config_path}")
                return []

            config_data = self._load_config(config_file)

            workers = []
            for worker_data in config_data.get("workers", []):
                try:
                    worker_type_str = worker_data.get("worker_type", "combined").lower()
                    worker_type = (
                        WorkerType(worker_type_str)
                        if worker_type_str in [t.value for t in WorkerType]
                        else WorkerType.COMBINED
                    )

                    total_tokens = worker_data.get("total_tokens", 0)
                    if total_tokens <= 0:
                        total_tokens = worker_data.get("available_memory", 0)

                    worker_kwargs = {
                        "worker_id": worker_data["worker_id"],
                        "model": worker_data["model"],
                        "url": worker_data["url"],
                        "current_load": worker_data.get("current_load", 0),
                        "cached_prefixes": worker_data.get("cached_prefixes", []),
                        "engine_type": worker_data.get("engine_type", "vllm"),
                        "api_key": worker_data.get("api_key"),
                        "worker_type": worker_type,
                        "group": worker_data.get("group", "default"),
                        "kv_addr": worker_data.get("kv_addr", ""),
                        "publisher_endpoint": worker_data.get("publisher_endpoint", ""),
                    }
                    if total_tokens > 0:
                        worker_kwargs["total_tokens"] = total_tokens

                    worker = WorkerInfo(**worker_kwargs)
                    workers.append(worker)
                except KeyError as e:
                    logger.error(f"Invalid worker config, missing field {e}: {worker_data}")
                except Exception as e:
                    logger.error(f"Failed to parse worker config: {e}")

            logger.info(f"Discovered {len(workers)} workers from config file")
            self._workers_cache = workers
            return workers

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON config file: {e}")
            return []
        except ValueError as e:
            logger.error(f"Config file format error: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to read worker config file: {e}")
            return []

    async def start(self):
        """启动发现服务（配置文件方式不需要启动）"""
        logger.info(f"Config discovery initialized with file: {self.config_path}")

    async def stop(self):
        """停止发现服务（配置文件方式不需要停止）"""
        logger.info("Config discovery stopped")
