"""测试配置"""

import asyncio
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_worker_config_json(tmp_path: Path) -> Path:
    """创建示例JSON配置文件"""
    config_file = tmp_path / "workers.json"
    config_file.write_text(
        """{
  "workers": [
    {
      "worker_id": "test-worker-1",
      "model": "test-model",
      "url": "http://localhost:8001/v1",
      "available_memory": 1000000,
      "current_load": 10,
      "cached_prefixes": ["prefix-1", "prefix-2"]
    },
    {
      "worker_id": "test-worker-2",
      "model": "test-model",
      "url": "http://localhost:8002/v1",
      "available_memory": 800000,
      "current_load": 5,
      "cached_prefixes": ["prefix-2", "prefix-3"]
    }
  ]
}"""
    )
    return config_file


@pytest.fixture
def sample_worker_config_yaml(tmp_path: Path) -> Path:
    """创建示例YAML配置文件"""
    config_file = tmp_path / "workers.yaml"
    config_file.write_text(
        """workers:
  - worker_id: test-worker-1
    model: test-model
    url: http://localhost:8001/v1
    available_memory: 1000000
    current_load: 10
    cached_prefixes:
      - prefix-1
      - prefix-2

  - worker_id: test-worker-2
    model: test-model
    url: http://localhost:8002/v1
    available_memory: 800000
    current_load: 5
    cached_prefixes:
      - prefix-2
      - prefix-3
"""
    )
    return config_file


@pytest.fixture
def invalid_worker_config(tmp_path: Path) -> Path:
    """创建无效的配置文件"""
    config_file = tmp_path / "invalid_workers.json"
    config_file.write_text(
        """{
  "workers": [
    {
      "worker_id": "test-worker-1",
      "model": "test-model"
    }
  ]
}"""
    )
    return config_file


@pytest.fixture
def sample_worker_config_with_types(tmp_path: Path) -> Path:
    """创建包含worker_type和group的配置文件"""
    config_file = tmp_path / "workers_with_types.json"
    config_file.write_text(
        """{
  "workers": [
    {
      "worker_id": "prefill-worker-1",
      "model": "test-model",
      "url": "http://localhost:8001/v1",
      "available_memory": 1000000,
      "current_load": 10,
      "cached_prefixes": ["prefix-1"],
      "worker_type": "prefill",
      "group": "group-a"
    },
    {
      "worker_id": "decode-worker-1",
      "model": "test-model",
      "url": "http://localhost:8002/v1",
      "available_memory": 800000,
      "current_load": 5,
      "cached_prefixes": ["prefix-2"],
      "worker_type": "decode",
      "group": "group-a"
    },
    {
      "worker_id": "combined-worker-1",
      "model": "test-model",
      "url": "http://localhost:8003/v1",
      "available_memory": 1200000,
      "current_load": 3,
      "cached_prefixes": ["prefix-3"],
      "worker_type": "combined",
      "group": "group-b"
    },
    {
      "worker_id": "combined-worker-2",
      "model": "test-model",
      "url": "http://localhost:8004/v1",
      "available_memory": 1500000,
      "current_load": 8,
      "cached_prefixes": ["prefix-4"],
      "worker_type": "combined",
      "group": "group-b"
    },
    {
      "worker_id": "default-worker",
      "model": "test-model",
      "url": "http://localhost:8005/v1",
      "available_memory": 900000,
      "current_load": 15,
      "cached_prefixes": []
    }
  ]
}"""
    )
    return config_file
