import pytest

from openjiuwentools.infer_router.vllm.cli import _merge_config_and_cli


@pytest.mark.parametrize(
    "content",
    [
        "port: 8010\n",  # model missing
        "model: [unclosed\n",  # malformed YAML
        "- model: /tmp/model\n",  # YAML root is not a mapping
    ],
)
def test_invalid_worker_config_is_reported_as_argparse_error(tmp_path, capsys, content):
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        _merge_config_and_cli(["--config", str(config_path)])

    assert exc_info.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_missing_worker_config_file_is_reported_as_argparse_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        _merge_config_and_cli(["--config", str(tmp_path / "missing.yaml")])

    assert exc_info.value.code == 2
    assert "Worker config not found" in capsys.readouterr().err
