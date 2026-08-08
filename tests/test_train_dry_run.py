"""Configuration-driven training dry-run tests."""

from pathlib import Path

import pytest
import yaml

from scripts.train import DEFAULT_CONFIG, load_dry_run_config, run_dry_run


def _write_config(path: Path, values: dict[object, object]) -> Path:
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_checked_in_dry_run_config_is_the_default() -> None:
    config = load_dry_run_config(DEFAULT_CONFIG)

    assert config.device == "cpu"
    assert config.state_dim == 9
    assert config.action_dim == 7
    assert config.chunk_size == 8


def test_dry_run_uses_configured_shapes(tmp_path, capsys) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["batch_size"] = 1
    raw["model"]["dummy_input_dim"] = 5
    raw["model"]["backbone_hidden_size"] = 7
    raw["model"]["state_dim"] = 6
    raw["model"]["state_hidden_dim"] = 8
    raw["model"]["action_dim"] = 3
    raw["model"]["chunk_size"] = 4
    path = _write_config(tmp_path / "custom-smoke.yaml", raw)

    run_dry_run(path)

    assert "Prediction shape: (1, 4, 3)" in capsys.readouterr().out


def test_dry_run_rejects_non_cpu_configuration(tmp_path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["device"] = "cuda"
    path = _write_config(tmp_path / "gpu-smoke.yaml", raw)

    with pytest.raises(ValueError, match="CPU device"):
        load_dry_run_config(path)
