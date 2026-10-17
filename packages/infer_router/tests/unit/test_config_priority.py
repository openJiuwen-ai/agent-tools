"""配置加载优先级：命令行（main 处理）> 环境变量 > YAML > 默认值。"""

from openjiuwentools.infer_router.config.config import Settings, create_settings


def _write_config(tmp_path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


_YAML = """
port: 8000
host: yaml-host
kv_cache:
  max_blocks: 7
"""


def test_environment_overrides_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("HOST", "env-host")
    monkeypatch.setenv("KV_CACHE_MAX_BLOCKS", "123")

    cfg = create_settings(_write_config(tmp_path, _YAML))

    assert cfg.port == 9999
    assert cfg.host == "env-host"
    assert cfg.kv_cache_max_blocks == 123


def test_yaml_overrides_defaults_when_environment_is_absent(monkeypatch, tmp_path):
    for name in ("PORT", "HOST", "KV_CACHE_MAX_BLOCKS"):
        monkeypatch.delenv(name, raising=False)

    cfg = create_settings(_write_config(tmp_path, _YAML))

    assert cfg.port == 8000
    assert cfg.host == "yaml-host"
    assert cfg.kv_cache_max_blocks == 7


def test_from_yaml_uses_the_same_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("PORT", "9999")

    cfg = Settings.from_yaml(_write_config(tmp_path, _YAML))

    assert cfg.port == 9999
    assert cfg.host == "yaml-host"
